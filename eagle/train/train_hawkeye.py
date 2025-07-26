import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import datasets
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
import csv
from datetime import datetime
import json
import argparse
import itertools
from tqdm import tqdm

# --- 全局设置 ---
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 模型定义 ---
class LSTMPolicyNet(nn.Module):
    def __init__(self, state_dim=10, lstm_hidden=128, mlp_hidden=128, num_layers=1, dropout=0.1):
        super().__init__()
        self.state_dim = state_dim
        self.lstm_hidden = lstm_hidden
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden + state_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 2)
        )

    def forward(self, state_seq, hidden=None):
        lstm_out, hidden = self.lstm(state_seq, hidden)
        logits = self.mlp(torch.cat((lstm_out, state_seq), dim=-1))
        return logits, hidden

    def act(self, state_seq, hidden=None, deterministic=False):
        logits, hidden = self.forward(state_seq, hidden)
        probs = torch.softmax(logits, dim=-1)
        if deterministic:
            actions = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()
        return actions, probs, hidden

    def reset_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers, batch_size, self.lstm_hidden).to(next(self.parameters()).device)
        c0 = torch.zeros(self.num_layers, batch_size, self.lstm_hidden).to(next(self.parameters()).device)
        return (h0, c0)

# --- 数据集处理 ---
def avg_acc_len_fn(stop_probs):
    return np.sum(np.arange(len(stop_probs)) * np.array(stop_probs, dtype=np.float32))

def avg_acc_len_fn_tensor(stop_probs: torch.Tensor):
    return torch.sum(torch.arange(stop_probs.size(-1), device=stop_probs.device) * stop_probs, dim=-1)

def padding_stops(stops, max_len=9):
    padded_stops = stops + [0.0] * (max_len - len(stops))
    return padded_stops[:max_len]

def normalize_probs(arr: np.array):
    arr = np.clip(arr, 0, None)
    row_sums = arr.sum(axis=-1, keepdims=True)
    if np.any(row_sums <= 1e-8):
        arr = arr / (row_sums + 1e-8)
    else:
        arr = arr / row_sums
    return arr.tolist()

class SharedStatesDataset(Dataset):
    def __init__(self, dataset, max_len=7):
        self.samples = []
        for sample in tqdm(dataset, desc="Loading dataset"):
            state_seq = [sample[f'eagle_{i}_forward'] for i in range(1, max_len + 1)]
            stops = [padding_stops(sample[f'action_{i}']['stop'], max_len=max_len + 2) for i in range(max_len + 1)]
            stops = [normalize_probs(np.array(stop)) for stop in stops]
            avg_acc_lens_op = avg_acc_len_fn(stops[-1])
            self.samples.append({
                "states": state_seq,
                "stop_probs": stops,
                "avg_acc_lens_op": avg_acc_lens_op
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    states = torch.tensor([x["states"] for x in batch], dtype=torch.float32)
    stop_probs = torch.tensor([x["stop_probs"] for x in batch], dtype=torch.float32)
    avg_acc_lens_op = torch.tensor([x["avg_acc_lens_op"] for x in batch], dtype=torch.float32)
    return states, stop_probs, avg_acc_lens_op

# --- 辅助函数 ---
def eagen_time(eagen_minus_time, eaforward_time, max_len):
    return eagen_minus_time + eaforward_time * max_len

def make_speed_matrix_hawkeye(eagen_minus_time, eaforward_time, eye_time, max_len):
    acc_len = torch.arange(0, max_len + 2)
    action_idx = torch.arange(max_len + 1)
    action_idx_ = torch.arange(max_len + 1)
    action_idx_[:-1] += 1
    t = eagen_minus_time + eaforward_time * action_idx + eye_time * action_idx_
    speed_matrix = acc_len.unsqueeze(1) / t.unsqueeze(0)
    return speed_matrix

def make_reward_matrix(maxlen, beta, alpha, speed_matrix):
    action_ext = torch.tensor([[1] * i + [0] * (maxlen - i + 1) for i in range(maxlen + 1)])
    reward_ext = -alpha * action_ext.unsqueeze(0).repeat_interleave(maxlen + 2, dim=0)
    for acclen in range(maxlen + 2):
        for action_idx in range(maxlen + 1):
            reward_ext[acclen, action_idx, action_idx] = beta * speed_matrix[acclen, action_idx]
    return reward_ext

def make_g_matrix(reward_matrix, maxlen, gamma):
    G = reward_matrix.clone()
    for i in reversed(range(1, maxlen + 1)):
        G[:, :, i - 1].add_(gamma * G[:, :, i])
    return G[:, :, :-1]

def find_first_zero_pos(tensor):
    mask = tensor.eq(0)
    extended_mask = torch.cat([mask, torch.ones(mask.size(0), 1, dtype=torch.long, device=tensor.device)], dim=1)
    first_zero_pos = extended_mask.argmax(dim=1, keepdim=True)
    return first_zero_pos

# --- 评估和统计 ---
def cal_avg_len(model, data_loader, eatime, speed_matrix, num_samples=10):
    model.eval()
    device = next(model.parameters()).device
    len_dict = defaultdict(list)
    with torch.no_grad():
        for states, stop_probs, avg_acc_lens_op in tqdm(data_loader, desc="Calculating Stats", leave=False):
            states, stop_probs, avg_acc_lens_op = states.to(device), stop_probs.to(device), avg_acc_lens_op.to(device)
            speed_matrix = speed_matrix.to(device)
            batch_size = states.size(0)

            batch_action_lengths = []
            batch_eye_speeds = []
            batch_avg_acc_lens = []

            for _ in range(num_samples):
                # 1. 每次都从模型采样新的动作
                hidden = None
                actions, _, _ = model.act(states, hidden) # [B, T]
                action_length = find_first_zero_pos(actions) # [B, 1]

                # 2. 根据新采样的动作，获取对应的接受概率分布
                expanded_index = action_length.unsqueeze(-1).expand(-1, -1, stop_probs.size(-1))
                action_stop_probs = stop_probs.gather(dim=1, index=expanded_index).squeeze(1) # [B, max_len+2]

                # 3. 从接受概率中采样一次（因为动作已经每次都重新采样了）
                acc_lens_sampled = torch.multinomial(action_stop_probs, num_samples=1) # [B, 1]

                # 4. 计算本次采样的Hawkeye速度
                speed_hawkeye = speed_matrix[acc_lens_sampled.squeeze(-1), action_length.squeeze(-1)] # [B]
                
                # 存储本次采样的结果
                batch_action_lengths.append(action_length)
                batch_eye_speeds.append(speed_hawkeye)
                
                # avg_acc_lens也应该是基于本次采样的动作
                batch_avg_acc_lens.append(avg_acc_len_fn_tensor(action_stop_probs))

            # 5. 对多次采样的结果取平均
            # 将列表中的Tensors堆叠起来，然后在num_samples维度上取平均
            avg_action_length = torch.stack(batch_action_lengths).float().mean(dim=0)
            avg_eye_speed = torch.stack(batch_eye_speeds).mean(dim=0)
            avg_acc_lens = torch.stack(batch_avg_acc_lens).mean(dim=0)

            # 6. 将平均后的结果存入len_dict
            len_dict["action_length"].extend(avg_action_length.squeeze(1).cpu().numpy())
            len_dict["eye_speed"].extend(avg_eye_speed.cpu().numpy())
            len_dict["avg_acc_lens"].extend(avg_acc_lens.cpu().numpy())

            # Eagle speed 和 optimal diff 是固定的，不需要在循环内计算
            speed_ea = avg_acc_lens_op / eatime
            len_dict["eagle_speed"].extend(speed_ea.cpu().numpy())
            
            # optimal_avg_acc_lens_diff 现在应该用平均后的 avg_acc_lens 计算
            optimal_avg_acc_lens_diff = avg_acc_lens - avg_acc_lens_op
            len_dict["optimal_avg_acc_lens_diff"].extend(optimal_avg_acc_lens_diff.cpu().numpy())
            
    return len_dict

def evaluate_loss(model, data_loader, G):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for states, stop_probs, _ in tqdm(data_loader, desc="Evaluating Loss", leave=False):
            states, stop_probs = states.to(device), stop_probs.to(device)
            logits, _ = model(states)
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
            
            action_dist = torch.distributions.Categorical(probs=probs)
            sampled_actions = action_dist.sample()
            action_indices = find_first_zero_pos(sampled_actions)
            
            accept_length_dist_probs = stop_probs.gather(dim=1, index=action_indices.unsqueeze(-1).expand(-1, -1, stop_probs.size(-1))).squeeze(1)
            accept_length_dist = torch.distributions.Categorical(probs=accept_length_dist_probs)
            accept_lengths = accept_length_dist.sample()
            
            g_matrix = G[accept_lengths, action_indices.squeeze(-1), :]
            action_log_probs = log_probs.gather(dim=-1, index=sampled_actions.unsqueeze(-1)).squeeze(-1)
            loss = -(action_log_probs * g_matrix).mean()
            total_loss += loss.item()
    model.train()
    return total_loss / len(data_loader) if len(data_loader) > 0 else float('inf')

def save_to_csv(params, len_dict, filename):
    data_row = params.copy()
    data_row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for key, values in len_dict.items():
        if values:
            data_row[f"{key}_mean"] = np.mean(values)
            data_row[f"{key}_std"] = np.std(values)
            data_row[f"{key}_25%"] = np.percentile(values, 25)
            data_row[f"{key}_75%"] = np.percentile(values, 75)
    
    if len_dict.get("eye_speed") and len_dict.get("eagle_speed"):
        mean_eye_speed = np.mean(len_dict["eye_speed"])
        mean_eagle_speed = np.mean(len_dict["eagle_speed"])
        data_row["ratio"] = mean_eye_speed / mean_eagle_speed if mean_eagle_speed > 0 else 0

    file_exists = os.path.isfile(filename)
    fieldnames = list(data_row.keys())
    
    if file_exists and os.path.getsize(filename) > 0:
        with open(filename, 'r', newline='') as f:
            reader = csv.reader(f)
            try:
                existing_headers = next(reader)
                for key in fieldnames:
                    if key not in existing_headers:
                        existing_headers.append(key)
                fieldnames = existing_headers
            except StopIteration:
                pass

    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists or f.tell() == 0:
            writer.writeheader()
        writer.writerow(data_row)


# --- 核心训练评估函数 ---
def train_and_evaluate(params, train_loader, test_loader, project_root):
    # --- 参数设置 ---
    max_len = params['max_len']
    bench_name = params['bench_name']
    base_model_name = params['base_model_name']
    td = params['td']
    
    # 动态生成文件名
    name_str = f"b-{params['beta']:.5f}-a-{params['alpha']:.5f}-g-{params['gamma']:.2f}-lr-{params['lr']:.0e}-wd-{params['weight_decay']:.0e}-dr-{params['dropout']:.2f}"

    # 创建输出目录
    output_dir = os.path.join(project_root, f"output/{params['bench_name']}/{params['base_model_name']}/{params['td']}")
    pt_dir = os.path.join(output_dir, "pt")
    pic_dir = os.path.join(output_dir, "pic")
    os.makedirs(pt_dir, exist_ok=True)
    os.makedirs(pic_dir, exist_ok=True)
    
    model_path = os.path.join(pt_dir, f"{name_str}.pt")
    loss_curve_path = os.path.join(pic_dir, f"{name_str}-loss_curve.png")
    dist_plot_path = os.path.join(pic_dir, f"{name_str}-distributions.png")
    csv_path = os.path.join(output_dir, "results.csv")

    # --- 模型、优化器和调度器 ---
    model = LSTMPolicyNet(
        state_dim=params['state_dim'],
        lstm_hidden=params['lstm_hidden'],
        mlp_hidden=params['mlp_hidden'],
        num_layers=params['num_layers'],
        dropout=params['dropout']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=params['num_epochs'], eta_min=1e-6)

    # --- 奖励和速度矩阵 ---
    speed_matrix = make_speed_matrix_hawkeye(params['eagen_minus_time'], params['eaforward_time'], params['eye_time'], max_len).to(device)
    reward_matrix = make_reward_matrix(max_len, params['beta'], params['alpha'], speed_matrix).to(device)
    G = make_g_matrix(reward_matrix, max_len, params['gamma']).to(device)

    # --- 训练循环 ---
    train_losses, test_losses = [], []
    best_test_loss = float('inf')

    epoch_pbar = tqdm(range(params['num_epochs']), desc="Epochs")
    for epoch in epoch_pbar:
        model.train()
        total_loss = 0
        
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{params['num_epochs']}", leave=False)
        for states, stop_probs, _ in batch_pbar:
            states, stop_probs = states.to(device), stop_probs.to(device)
            
            logits, _ = model(states)
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
            
            action_dist = torch.distributions.Categorical(probs=probs)
            sampled_actions = action_dist.sample()
            action_indices = find_first_zero_pos(sampled_actions)
            
            accept_length_dist_probs = stop_probs.gather(dim=1, index=action_indices.unsqueeze(-1).expand(-1, -1, stop_probs.size(-1))).squeeze(1)
            accept_length_dist = torch.distributions.Categorical(probs=accept_length_dist_probs)
            accept_lengths = accept_length_dist.sample()

            g_matrix = G[accept_lengths, action_indices.squeeze(-1), :]
            action_log_probs = log_probs.gather(dim=-1, index=sampled_actions.unsqueeze(-1)).squeeze(-1)
            
            loss = -(action_log_probs * g_matrix).mean()
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            batch_pbar.set_postfix(loss=loss.item())

        scheduler.step()
        
        avg_train_loss = total_loss / len(train_loader)
        avg_test_loss = evaluate_loss(model, test_loader, G)
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        
        epoch_pbar.set_postfix(train_loss=f"{avg_train_loss:.4f}", test_loss=f"{avg_test_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), model_path)

    print(f"\n  -> Final best test loss: {best_test_loss:.4f}, model saved to {model_path}")
    
    # --- 结果可视化和保存 ---
    plt.figure()
    plt.plot(range(1, params['num_epochs'] + 1), train_losses, label="Train Loss")
    plt.plot(range(1, params['num_epochs'] + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Loss Curve\n{name_str}")
    plt.tight_layout()
    plt.savefig(loss_curve_path)
    plt.close()

    model.load_state_dict(torch.load(model_path))
    eatime = eagen_time(params['eagen_minus_time'], params['eaforward_time'], max_len)
    len_dict = cal_avg_len(model, test_loader, eatime, speed_matrix)

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle(f"Performance Distributions\n{name_str}", fontsize=16)
    sns.histplot(len_dict["action_length"], kde=True, ax=axes[0, 0], bins=max_len+1).set_title("Action Length")
    sns.histplot(len_dict["avg_acc_lens"], kde=True, ax=axes[0, 1], bins=30).set_title("Average Accept Length")
    sns.histplot(len_dict["optimal_avg_acc_lens_diff"], kde=True, ax=axes[0, 2], bins=30).set_title("Deviation from Optimal")
    sns.histplot(len_dict["eagle_speed"], kde=True, ax=axes[1, 0], bins=30).set_title("Eagle Speed")
    sns.histplot(len_dict["eye_speed"], kde=True, ax=axes[1, 1], bins=30).set_title("Eye Speed")
    axes[1, 2].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(dist_plot_path)
    plt.close()

    save_to_csv(params, len_dict, csv_path)
    print(f"Results for this run saved to {csv_path}")


# --- 主程序 ---
if __name__ == '__main__':
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
    print(f"Project Root correctly detected: {PROJECT_ROOT}")

    # 2. --- 路径解析函数 ---
    def resolve_path(path_from_config):
        """
        将配置文件中的路径（无论是相对还是绝对）转换为绝对路径。
        如果路径是相对的，则假定它相对于项目根目录。
        """
        if os.path.isabs(path_from_config):
            return path_from_config
        # 使用 os.path.normpath 来正确处理 ".." 等情况
        return os.path.normpath(os.path.join(PROJECT_ROOT, path_from_config))

    parser = argparse.ArgumentParser(description="Train a policy model with hyperparameter grid search from a config file.")
    parser.add_argument('--config', type=str, default='eagle/train/train_debug.json', help='Path to the configuration JSON file.')
    args = parser.parse_args()

    # 3. --- 加载配置 ---
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        # 如果是相对路径，尝试从项目根目录解析
        try:
            config_path_from_root = resolve_path(args.config)
            with open(config_path_from_root, 'r') as f:
                config = json.load(f)
            print(f"Loaded configuration from: {config_path_from_root}")
        except FileNotFoundError:
             print(f"Error: Configuration file not found at {args.config} or {config_path_from_root}")
             exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the configuration file: {args.config}")
        exit(1)

    # --- 生成参数组合 ---
    search_params = {key: value for key, value in config.items() if isinstance(value, list)}
    fixed_params = {key: value for key, value in config.items() if not isinstance(value, list)}

    keys, values = zip(*search_params.items()) if search_params else ([], [])
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)] if search_params else [{}]
    
    print(f"Starting hyperparameter search with {len(param_combinations)} combinations.")

    # --- 数据加载 ---
    max_len = fixed_params.get('max_len', 7)
    
    dataset_path_from_config = fixed_params.get('dataset_path')
    if not dataset_path_from_config:
        print("Error: 'dataset_path' must be defined in the config file.")
        exit(1)

    dataset_path = resolve_path(dataset_path_from_config)
    batch_size = fixed_params.get('batch_size', 64)
        
    print(f"Attempting to load dataset from resolved path: {dataset_path}")
    try:
        ReplayDataset = datasets.load_from_disk(dataset_path)
    except FileNotFoundError:
        print(f"\nFATAL ERROR: The directory '{dataset_path}' does not exist.")
        print("Please check the 'dataset_path' in your config file and ensure it's correct relative to the project root.")
        exit(1)
    
    train_dataset = SharedStatesDataset(ReplayDataset["train"], max_len=max_len)
    test_dataset = SharedStatesDataset(ReplayDataset["test"], max_len=max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


    # --- 遍历所有组合进行训练 ---
    main_pbar = tqdm(param_combinations, desc="Hyperparameter Search")
    for i, combo in enumerate(main_pbar):
        params = {**fixed_params, **combo}
        main_pbar.set_description(f"Combination {i+1}/{len(param_combinations)}")
        
        print("\n" + "="*80)
        print(f"Running combination {i+1}/{len(param_combinations)}")
        print(json.dumps(params, indent=4))
        print("="*80)

        try:
            train_and_evaluate(params, train_loader, test_loader, PROJECT_ROOT)
        except Exception as e:
            print(f"\nAn unexpected error occurred during training for combination {i+1}:")
            print(f"Params: {json.dumps(params, indent=4)}")
            print(f"Error: {e}")
            # 记录错误日志
            with open("error_log.txt", "a") as f:
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Config File: {args.config}\n")
                f.write(f"Params: {json.dumps(params, indent=4)}\n")
                f.write(f"Error: {e}\n\n")
            continue