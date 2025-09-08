import json
import os
from transformers import AutoTokenizer
import numpy as np
import pandas as pd

# ==============================================================================
# 请在此处配置您的测试参数
# ==============================================================================

# 1. 定义模型和路径配置
MODELS_CONFIG = [
    {
        "model_name": "llama3.1",
        "tokenizer_path": "/home/majunjie/code/weights/hf/Meta-Llama-3.1-8B-Instruct/",
        "pt_list": [
            "b-0.03084-a-0.02000-g-0.99-lr-1e-04-wd-1e-04-dr-0.20.pt",
            "b-0.03084-a-0.02000-g-0.99-lr-1e-04-wd-1e-04-dr-0.10.pt"
        ],
    },
    {
        "model_name": "vicuna13",
        "tokenizer_path": "/home/majunjie/code/weights/hf/vicuna-13b-v1.3/",
        "pt_list": [
            "b-0.02877-a-0.04000-g-0.99-lr-1e-04-wd-1e-04-dr-0.20.pt",
            "b-0.02877-a-0.06000-g-0.99-lr-1e-04-wd-1e-04-dr-0.10.pt"
        ],
    },
    {
        "model_name": "deepseek",
        "tokenizer_path": "/home/majunjie/code/weights/hf/DeepSeek-R1-Distill-Llama-8B/",
        "pt_list": [
            "b-0.03237-a-0.02000-g-0.99-lr-1e-04-wd-1e-04-dr-0.20.pt",
            "b-0.03237-a-0.01000-g-0.99-lr-1e-04-wd-1e-04-dr-0.10.pt"
        ],
    },
]

# 2. 定义要分析的基准测试和 num_choices
BENCH_LIST = [
    "mt_bench",
    "humaneval",
    "gsm8k",
    "alpaca",
    "mbpp"
]
NUM_CHOICES_LIST = [10]

# 3. 定义结果文件的根目录
OUTPUT_DIR_PREFIX = "../"

# 4. 定义处理的数据点数量 (使用 None 来处理完整数据集)
NUM_DATAPOINTS = None # 设置为 None 来处理所有数据

# ==============================================================================
# 脚本核心逻辑
# ==============================================================================

def calculate_metrics(jsonl_file, num_choices, num_datapoints=None):
    """
    从指定的jsonl文件中计算平均解码速度、平均接受长度以及每个choice的详细指标。
    使用 "new_tokens" 字段计算总token数。
    根据最新的逻辑计算平均接受长度。
    """
    if not os.path.exists(jsonl_file):
        return (None,) * 6

    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))

    if not data:
        print(f"Warning: File is empty, skipping: {jsonl_file}")
        return (None,) * 6

    speeds_per_choice = []
    acceptance_lengths_per_choice = []
    
    datapoints_to_process = data
    if num_datapoints is not None:
        datapoints_to_process = data[:int(min(num_datapoints, len(data)))]
    
    for j in range(num_choices):
        speeds_for_this_choice = []
        acceptance_lengths_for_this_choice = []
        for datapoint in datapoints_to_process:
            if j >= len(datapoint["choices"]):
                continue
            
            choice = datapoint["choices"][j]
            wall_time = sum(choice.get('wall_time', [0]))
            total_tokens = sum(choice.get('new_tokens', [0]))

            # 计算速度
            if wall_time > 0 and total_tokens > 0:
                speeds_for_this_choice.append(total_tokens / wall_time)

            # 计算平均接受长度 (最终修正逻辑)
            idxs = choice.get('idxs')
            # 检查idxs是否存在且为非空列表
            if idxs and isinstance(idxs, list) and len(idxs) > 0:
                # 总验证次数 = 每个idxs值+1后再求和
                total_verification_steps = sum(idx + 1 for idx in idxs)
                if total_verification_steps > 0 and total_tokens > 0:
                    acceptance_lengths_for_this_choice.append(total_tokens / total_verification_steps)
                else:
                    acceptance_lengths_for_this_choice.append(0)
            else:
                acceptance_lengths_for_this_choice.append(0)

        if speeds_for_this_choice:
            speeds_per_choice.append(np.mean(speeds_for_this_choice))
        if acceptance_lengths_for_this_choice:
            valid_lengths = [l for l in acceptance_lengths_for_this_choice if l > 0]
            if valid_lengths:
                acceptance_lengths_per_choice.append(np.mean(valid_lengths))
            else:
                acceptance_lengths_per_choice.append(0)

    if not speeds_per_choice:
        return (None,) * 6

    # 计算最终统计
    final_avg_speed = np.mean(speeds_per_choice) if speeds_per_choice else 0
    final_std_dev_speed = np.std(speeds_per_choice) if speeds_per_choice else 0
    
    valid_final_lengths = [l for l in acceptance_lengths_per_choice if l > 0]
    if valid_final_lengths:
        final_avg_acceptance_length = np.mean(valid_final_lengths)
    else:
        final_avg_acceptance_length = 0

    return (final_avg_speed, final_std_dev_speed, speeds_per_choice, 
            final_avg_acceptance_length, None, acceptance_lengths_per_choice)


def main():
    """主函数，遍历所有配置、计算指标并保存摘要和详细结果。"""
    summary_results = []
    detailed_results = []

    for model_config in MODELS_CONFIG:
        model_name = model_config["model_name"]
        print(f"\n{'='*30}\nProcessing Model: {model_name}\n{'='*30}")
        
        try:
            AutoTokenizer.from_pretrained(model_config["tokenizer_path"])
        except Exception as e:
            print(f"Warning: Could not load tokenizer for {model_name}. This is not required for current calculations but might indicate a config issue. Error: {e}")

        for bench_name in BENCH_LIST:
            for num_choices in NUM_CHOICES_LIST:
                base_path = f"{OUTPUT_DIR_PREFIX}/{bench_name}/{model_name}/t1d7/time"

                # 1. 计算 Baseline 的指标
                baseline_file = f"{base_path}/baseline-speedtest-3.jsonl"
                baseline_speed, _, bl_speeds_pc, _, _, bl_acc_lens_pc = calculate_metrics(baseline_file, 1, NUM_DATAPOINTS)
                
                if bl_speeds_pc:
                    for i, (speed, acc_len) in enumerate(zip(bl_speeds_pc, bl_acc_lens_pc)):
                        detailed_results.append({
                            "Model": model_name, "Benchmark": bench_name,
                            "Type": "Baseline", "Choice Index": i, "Speed": speed,
                            "Accept Length": acc_len if acc_len > 0 else "N/A"
                        })

                # --- 辅助函数：记录结果 ---
                def record_results(type_name, choices, speed, std_speed, speeds_pc, acc_len, std_acc_len, acc_lens_pc):
                    if speed is not None and speed > 0:
                        speedup_ratio = (speed / baseline_speed) if baseline_speed and baseline_speed > 0 else 0
                        summary_results.append({
                            "Model": model_name, "Benchmark": bench_name, "Choices": choices,
                            "Type": type_name, "Speed": f"{speed:.2f}",
                            "Baseline Speed": f"{baseline_speed:.2f}" if baseline_speed is not None else "N/A",
                            "Speedup": f"{speedup_ratio:.2f}x", "Std Dev (Speed)": f"{std_speed:.2f}",
                            "Avg Accept Length": f"{acc_len:.2f}" if acc_len > 0 else "N/A"
                        })
                        if speeds_pc:
                            for i, (s, al) in enumerate(zip(speeds_pc, acc_lens_pc)):
                                detailed_results.append({
                                    "Model": model_name, "Benchmark": bench_name,
                                    "Type": type_name, "Choice Index": i, "Speed": s,
                                    "Accept Length": al if al > 0 else "N/A"
                                })

                # 2. 计算 Eagle2 的指标
                if model_name in ["llama3.1", "vicuna13"]:
                    eagle2_choices = 3
                    eagle2_file = f"{base_path}/eagle2-speedtest-{eagle2_choices}.jsonl"
                    metrics = calculate_metrics(eagle2_file, eagle2_choices, NUM_DATAPOINTS)
                    record_results("Eagle2 (EA)", eagle2_choices, *metrics)

                # 3. 计算 Eagle3 的指标
                eagle_file = f"{base_path}/eagle3-speedtest-{num_choices}.jsonl"
                metrics = calculate_metrics(eagle_file, num_choices, NUM_DATAPOINTS)
                record_results("Eagle3 (EA)", num_choices, *metrics)

                # 4. 计算 Hawkeye 的指标
                for pt_file in model_config["pt_list"]:
                    pt_name = pt_file.replace('.pt', '')
                    hawkeye_file = f"{base_path}/hawkeye-{pt_name}-speedtest-{num_choices}.jsonl"
                    hawkeye_type_name = f"Hawkeye ({pt_name})"
                    metrics = calculate_metrics(hawkeye_file, num_choices, NUM_DATAPOINTS)
                    record_results(hawkeye_type_name, num_choices, *metrics)

    # --- 处理并保存摘要结果 ---
    if summary_results:
        df_summary = pd.DataFrame(summary_results)
        df_summary.set_index(["Model", "Benchmark", "Choices", "Type"], inplace=True)
        
        print("\n" + "="*90)
        print(" " * 35 + "Metrics Comparison Summary")
        print("="*90)
        print(df_summary)
        print("="*90)

        output_summary_csv = "metrics_summary.csv"
        df_summary.to_csv(output_summary_csv)
        print(f"\n[SUCCESS] Summary results saved to: {output_summary_csv}")

    # --- 处理并保存详细结果 ---
    if detailed_results:
        df_detailed = pd.DataFrame(detailed_results)
        
        print("\n" + "="*90)
        print(" " * 30 + "Detailed Metrics Per Choice (Sample)")
        print("="*90)
        print(df_detailed.head())
        print("...")
        print(f"Total detailed records: {len(df_detailed)}")
        print("="*90)

        output_detailed_csv = "metrics_per_choice_details.csv"
        df_detailed.to_csv(output_detailed_csv, index=False)
        print(f"\n[SUCCESS] Detailed per-choice metrics saved to: {output_detailed_csv}")

    if not summary_results and not detailed_results:
        print("\nNo results to display. Please check your configuration and file paths.")


if __name__ == "__main__":
    main()