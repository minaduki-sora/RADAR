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

def calculate_average_speed(jsonl_file, tokenizer, num_choices, num_datapoints=None):
    """
    从指定的jsonl文件中计算平均解码速度和每个choice的速度。
    """
    if not os.path.exists(jsonl_file):
        return None, None, None

    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))

    if not data:
        print(f"Warning: File is empty, skipping: {jsonl_file}")
        return None, None, None

    avg_speeds_per_choice = []
    
    # 如果 num_datapoints 为 None，则处理所有数据
    datapoints_to_process = data
    if num_datapoints is not None:
        datapoints_to_process = data[:int(min(num_datapoints, len(data)))]
    
    for j in range(num_choices):
        speeds_for_this_choice = []
        for datapoint in datapoints_to_process:
            if j >= len(datapoint["choices"]):
                continue
            
            choice = datapoint["choices"][j]
            answer_turns = choice['turns']
            wall_time = sum(choice['wall_time'])
            
            if wall_time == 0:
                continue

            total_tokens = sum(len(tokenizer(turn).input_ids) - 1 for turn in answer_turns)
            speeds_for_this_choice.append(total_tokens / wall_time)
        
        if speeds_for_this_choice:
            avg_speeds_per_choice.append(np.mean(speeds_for_this_choice))

    if not avg_speeds_per_choice:
        return None, None, None
        
    final_avg_speed = np.mean(avg_speeds_per_choice)
    final_std_dev = np.std(avg_speeds_per_choice)
    
    return final_avg_speed, final_std_dev, avg_speeds_per_choice


def main():
    """主函数，遍历所有配置、计算速度并保存摘要和详细结果。"""
    summary_results = []
    detailed_results = []

    for model_config in MODELS_CONFIG:
        model_name = model_config["model_name"]
        print(f"\n{'='*30}\nProcessing Model: {model_name}\n{'='*30}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_config["tokenizer_path"])
        except Exception as e:
            print(f"Error loading tokenizer for {model_name}. Skipping. Error: {e}")
            continue

        for bench_name in BENCH_LIST:
            for num_choices in NUM_CHOICES_LIST:
                base_path = f"{OUTPUT_DIR_PREFIX}/{bench_name}/{model_name}/t1d7/time"

                # 1. 计算 Baseline 的速度
                baseline_file = f"{base_path}/baseline-speedtest-3.jsonl"
                baseline_speed, _, baseline_speeds_per_choice = calculate_average_speed(baseline_file, tokenizer, 1, NUM_DATAPOINTS)
                
                # 记录 baseline 的详细速度
                if baseline_speeds_per_choice:
                    for i, speed in enumerate(baseline_speeds_per_choice):
                        detailed_results.append({
                            "Model": model_name, "Benchmark": bench_name,
                            "Type": "Baseline", "Choice Index": i, "Speed": speed
                        })

                # 2. 计算 Eagle2 的速度并对比 (仅针对 llama3 和 vicuna13)
                if model_name in ["llama3.1", "vicuna13"]:
                    eagle2_choices = 3
                    eagle2_file = f"{base_path}/eagle2-speedtest-{eagle2_choices}.jsonl"
                    eagle2_speed, eagle2_std, eagle2_speeds_per_choice = calculate_average_speed(eagle2_file, tokenizer, eagle2_choices, NUM_DATAPOINTS)
                    if eagle2_speed is not None:
                        ratio = (eagle2_speed / baseline_speed) if baseline_speed and baseline_speed > 0 else 0
                        summary_results.append({
                            "Model": model_name, "Benchmark": bench_name, "Choices": eagle2_choices,
                            "Type": "Eagle2 (EA)", "Speed": f"{eagle2_speed:.2f}",
                            "Baseline Speed": f"{baseline_speed:.2f}" if baseline_speed is not None else "N/A",
                            "Speedup": f"{ratio:.2f}x", "Std Dev": f"{eagle2_std:.2f}"
                        })
                        # 记录 Eagle2 的详细速度
                        if eagle2_speeds_per_choice:
                            for i, speed in enumerate(eagle2_speeds_per_choice):
                                detailed_results.append({
                                    "Model": model_name, "Benchmark": bench_name,
                                    "Type": "Eagle2 (EA)", "Choice Index": i, "Speed": speed
                                })

                # 3. 计算 Eagle3 (EA) 的速度并对比
                eagle_file = f"{base_path}/eagle3-speedtest-{num_choices}.jsonl"
                eagle_speed, eagle_std, eagle_speeds_per_choice = calculate_average_speed(eagle_file, tokenizer, num_choices, NUM_DATAPOINTS)
                if eagle_speed is not None:
                    ratio = (eagle_speed / baseline_speed) if baseline_speed and baseline_speed > 0 else 0
                    summary_results.append({
                        "Model": model_name, "Benchmark": bench_name, "Choices": num_choices,
                        "Type": "Eagle3 (EA)", "Speed": f"{eagle_speed:.2f}",
                        "Baseline Speed": f"{baseline_speed:.2f}" if baseline_speed is not None else "N/A",
                        "Speedup": f"{ratio:.2f}x", "Std Dev": f"{eagle_std:.2f}"
                    })
                    # 记录 Eagle 的详细速度
                    if eagle_speeds_per_choice:
                        for i, speed in enumerate(eagle_speeds_per_choice):
                            detailed_results.append({
                                "Model": model_name, "Benchmark": bench_name,
                                "Type": "Eagle3 (EA)", "Choice Index": i, "Speed": speed
                            })

                # 4. 计算 Hawkeye (EA-WE) 的速度并对比
                for pt_file in model_config["pt_list"]:
                    pt_name = pt_file.replace('.pt', '')
                    hawkeye_file = f"{base_path}/hawkeye-{pt_name}-speedtest-{num_choices}.jsonl"
                    hawkeye_speed, hawkeye_std, hawkeye_speeds_per_choice = calculate_average_speed(hawkeye_file, tokenizer, num_choices, NUM_DATAPOINTS)
                    if hawkeye_speed is not None:
                        ratio = (hawkeye_speed / baseline_speed) if baseline_speed and baseline_speed > 0 else 0
                        hawkeye_type_name = f"Hawkeye ({pt_name})"
                        summary_results.append({
                            "Model": model_name, "Benchmark": bench_name, "Choices": num_choices,
                            "Type": hawkeye_type_name, "Speed": f"{hawkeye_speed:.2f}",
                            "Baseline Speed": f"{baseline_speed:.2f}" if baseline_speed is not None else "N/A",
                            "Speedup": f"{ratio:.2f}x", "Std Dev": f"{hawkeye_std:.2f}"
                        })
                        # 记录 Hawkeye 的详细速度
                        if hawkeye_speeds_per_choice:
                            for i, speed in enumerate(hawkeye_speeds_per_choice):
                                detailed_results.append({
                                    "Model": model_name, "Benchmark": bench_name,
                                    "Type": hawkeye_type_name, "Choice Index": i, "Speed": speed
                                })

    # --- 处理并保存摘要结果 ---
    if summary_results:
        df_summary = pd.DataFrame(summary_results)
        df_summary.set_index(["Model", "Benchmark", "Choices", "Type"], inplace=True)
        
        print("\n" + "="*80)
        print(" " * 30 + "Speed Comparison Summary")
        print("="*80)
        print(df_summary)
        print("="*80)

        output_summary_csv = "speed_summary-1.csv"
        df_summary.to_csv(output_summary_csv)
        print(f"\n[SUCCESS] Summary results saved to: {output_summary_csv}")

    # --- 处理并保存详细结果 ---
    if detailed_results:
        df_detailed = pd.DataFrame(detailed_results)
        
        print("\n" + "="*80)
        print(" " * 25 + "Detailed Speeds Per Choice (Sample)")
        print("="*80)
        print(df_detailed.head()) # 打印前5行作为示例
        print("...")
        print(f"Total detailed records: {len(df_detailed)}")
        print("="*80)

        output_detailed_csv = "speed_per_choice_details-1.csv"
        df_detailed.to_csv(output_detailed_csv, index=False)
        print(f"\n[SUCCESS] Detailed per-choice speed results saved to: {output_detailed_csv}")

    if not summary_results and not detailed_results:
        print("\nNo results to display. Please check your configuration and file paths.")


if __name__ == "__main__":
    main()