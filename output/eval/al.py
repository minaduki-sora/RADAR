import json
import os
import numpy as np
import pandas as pd

# ==============================================================================
# 请在此处配置您的测试参数
# ==============================================================================

# 1. 定义模型和路径配置
MODELS_CONFIG = [
    {
        "model_name": "llama3.1",
        "pt_list": [
            # "b-0.03084-a-0.02000-g-0.99-lr-1e-04-wd-1e-04-dr-0.20.pt",
            "b-0.03084-a-0.02000-g-0.99-lr-1e-04-wd-1e-04-dr-0.10.pt"
        ],
    },
    {
        "model_name": "vicuna13",
        "pt_list": [
            # "b-0.02877-a-0.04000-g-0.99-lr-1e-04-wd-1e-04-dr-0.20.pt",
            "b-0.02877-a-0.06000-g-0.99-lr-1e-04-wd-1e-04-dr-0.10.pt"
        ],
    },
    {
        "model_name": "deepseek",
        "pt_list": [
            # "b-0.03237-a-0.02000-g-0.99-lr-1e-04-wd-1e-04-dr-0.20.pt",
            "b-0.03237-a-0.01000-g-0.99-lr-1e-04-wd-1e-04-dr-0.10.pt"
        ],
    },
]

# 2. 定义要分析的基准测试
BENCH_LIST = [
    "mt_bench",
    "humaneval",
    "gsm8k",
    "alpaca",
    "mbpp"
]

# 3. 定义 action 文件中的 num_choices (根据示例文件名 ...-action-3.jsonl)
ACTION_NUM_CHOICES = 3

# 4. 定义结果文件的根目录
OUTPUT_DIR_PREFIX = "../"

# 5. 定义处理的数据点数量 (使用 None 来处理完整数据集)
NUM_DATAPOINTS = None # 设置为 None 来处理所有数据

# ==============================================================================
# 脚本核心逻辑
# ==============================================================================

def calculate_avg_generation_count(jsonl_file, num_choices, num_datapoints=None):
    """
    从指定的 action jsonl 文件中计算平均生成次数。
    计算方法: sum(action_lengths) / sum(idx + 1 for idx in idxs)
    """
    if not os.path.exists(jsonl_file):
        return None

    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))

    if not data:
        print(f"Warning: File is empty, skipping: {jsonl_file}")
        return None

    generation_counts_per_datapoint = []
    
    datapoints_to_process = data
    if num_datapoints is not None:
        datapoints_to_process = data[:int(min(num_datapoints, len(data)))]
    
    for datapoint in datapoints_to_process:
        for j in range(num_choices):
            if j >= len(datapoint["choices"]):
                continue
            
            choice = datapoint["choices"][j]
            
            action_lengths = choice.get('action_lengths')
            idxs = choice.get('idxs')

            # 确保所需字段都存在且有效
            if action_lengths and isinstance(action_lengths, list) and idxs and isinstance(idxs, list):
                total_action_length = sum(action_lengths)
                # 总验证次数 = 每个idxs值+1后再求和
                total_verification_steps = sum(idx + 1 for idx in idxs)
                
                if total_verification_steps > 0 and total_action_length > 0:
                    generation_counts_per_datapoint.append(total_action_length / total_verification_steps)

    if not generation_counts_per_datapoint:
        return None
        
    # 返回所有数据点和 choices 的平均值
    return np.mean(generation_counts_per_datapoint)


def main():
    """主函数，遍历所有配置，为 Hawkeye action 文件计算平均生成次数。"""
    summary_results = []

    for model_config in MODELS_CONFIG:
        model_name = model_config["model_name"]
        print(f"\n{'='*30}\nProcessing Model: {model_name}\n{'='*30}")
        
        for bench_name in BENCH_LIST:
            base_path = f"{OUTPUT_DIR_PREFIX}/{bench_name}/{model_name}/t1d7/time"

            # 遍历 Hawkeye 的 pt 文件
            for pt_file in model_config["pt_list"]:
                pt_name = pt_file.replace('.pt', '')
                
                # 构建 action 文件的路径
                action_file = f"{base_path}/hawkeye-{pt_name}-action-{ACTION_NUM_CHOICES}.jsonl"
                
                # 计算平均生成次数
                avg_gen_count = calculate_avg_generation_count(action_file, ACTION_NUM_CHOICES, NUM_DATAPOINTS)
                
                hawkeye_type_name = f"Hawkeye ({pt_name})"
                
                if avg_gen_count is not None:
                    print(f"  {bench_name} - {hawkeye_type_name}: {avg_gen_count:.2f}")
                    summary_results.append({
                        "Model": model_name,
                        "Benchmark": bench_name,
                        "Type": hawkeye_type_name,
                        "Avg Generation Count": f"{avg_gen_count:.2f}"
                    })
                else:
                    print(f"  {bench_name} - {hawkeye_type_name}: Not found or failed to calculate.")


    # --- 处理并保存摘要结果 ---
    if summary_results:
        df_summary = pd.DataFrame(summary_results)
        
        print("\n" + "="*90)
        print(" " * 30 + "Hawkeye Average Generation Count Summary")
        print("="*90)
        
        # 使用 pivot_table 来获得更清晰的视图
        df_pivot = df_summary.pivot_table(
            index=["Model", "Type"], 
            columns="Benchmark", 
            values="Avg Generation Count",
            aggfunc='first' # 使用 'first' 因为每个组合只有一个值
        )
        print(df_pivot)
        print("="*90)

        output_summary_csv = "hawkeye_avg_generation_count.csv"
        df_pivot.to_csv(output_summary_csv)
        print(f"\n[SUCCESS] Hawkeye average generation count results saved to: {output_summary_csv}")

    else:
        print("\nNo action file results to display. Please check your configuration and file paths.")


if __name__ == "__main__":
    main()