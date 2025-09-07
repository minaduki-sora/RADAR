import json

# ==============================================================================
# 请在此处配置您的测试参数
# ==============================================================================

# 1. 为每个模型提供 .pt 文件名列表
#    gen_ea_we_answer 测试将为列表中的每个 .pt 文件生成一个命令。
llama3_pt = [
    # "b-0.03084-a-0.02000-g-0.99-lr-1e-04-wd-1e-04-dr-0.20.pt",
    "b-0.03084-a-0.02000-g-0.99-lr-1e-04-wd-1e-04-dr-0.10.pt"
]

vicuna13_pt = [
    # "b-0.02877-a-0.04000-g-0.99-lr-1e-04-wd-1e-04-dr-0.20.pt",
    "b-0.02877-a-0.06000-g-0.99-lr-1e-04-wd-1e-04-dr-0.10.pt",
]

ds_pt = [
    # "b-0.03237-a-0.02000-g-0.99-lr-1e-04-wd-1e-04-dr-0.20.pt",
    "b-0.03237-a-0.01000-g-0.99-lr-1e-04-wd-1e-04-dr-0.10.pt",
]

# 2. 提供需要测试的 bench 名称列表
bench_list = [
    "mt_bench",
    "humaneval",
    "gsm8k",
    "alpaca",
    "mbpp"
]

# 3. 配置输出文件的绝对路径前缀
#    脚本将使用此路径为 "output" 字段生成完整的绝对路径。
#    请确保以 '/' 结尾。
output_path_prefix = "/home/majunjie/code/EAGLE/"

# ==============================================================================
# 脚本核心逻辑
# ==============================================================================

# 模型配置
MODELS_CONFIG = [
    {
        "model_name": "llama3.1",
        "script_model_name": "llama3chat",
        "ea_model_path": "../weights/eagle/EAGLE3-LLaMA3.1-Instruct-8B",# "../weights/eagle/EAGLE-LLaMA3.1-Instruct-8B",
        "base_model_path": "../weights/hf/Meta-Llama-3.1-8B-Instruct",
        "eye_model_path_prefix": "output/shareGPT/llama3.1/t1d7/pt/",
        "pt_list": llama3_pt,
    },
    {
        "model_name": "vicuna13",
        "script_model_name": "vicuna",
        "ea_model_path": "../weights/eagle/EAGLE3-Vicuna1.3-13B",#"../weights/eagle/EAGLE-Vicuna-13B-v1.3",
        "base_model_path": "../weights/hf/vicuna-13b-v1.3",
        "eye_model_path_prefix": "output/shareGPT/vicuna13/t1d7/pt/",
        "pt_list": vicuna13_pt,
    },
    {
        "model_name": "deepseek",
        "script_model_name": "ds",
        "ea_model_path": "../weights/eagle/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B",
        "base_model_path": "../weights/hf/DeepSeek-R1-Distill-Llama-8B",
        "eye_model_path_prefix": "output/shareGPT/ds/t1d7/pt/",
        "pt_list": ds_pt,
    },
]

# 静态参数
STATIC_PARAMS = "--num-gpus-total 1 --depth 7 --top-k 10 --temperature 1.0"
NUM_CHOICES_LIST = [3]

def generate_commands_in_order():
    """
    按最终要求的顺序生成所有命令，并为每个命令添加一个'idx'。
    顺序: num_choices=5 (llama, vicuna), num_choices=10 (llama, vicuna)
    """
    final_commands = []
    idx_counter = 34
    
    # 外层循环控制 num_choices，确保所有 5 的测试先于 10
    for num_choices in NUM_CHOICES_LIST:
        # 第二层循环控制模型，确保模型按配置顺序执行
        for model_config in MODELS_CONFIG:
            # 内层循环处理每个 bench
            for bench_name in bench_list:
                # 1. Baseline command
                # answer_file_baseline = f"output/{bench_name}/{model_config['model_name']}/t1d7/time/baseline-speedtest-3.jsonl"
                # cmd_baseline = (
                #     f"python -m eagle.evaluation.gen_baseline_answer_{model_config['script_model_name']} "
                #     f"--ea-model-path {model_config['ea_model_path']} "
                #     f"--base-model-path {model_config['base_model_path']} "
                #     f"--bench-name {bench_name} "
                #     f"{STATIC_PARAMS} "
                #     f"--num-choices 1 "
                #     f"--answer-file {answer_file_baseline}"
                # )
                # final_commands.append({
                #     "idx": idx_counter,
                #     "cmd": cmd_baseline,
                #     "output": f"{output_path_prefix}{answer_file_baseline}"
                # })
                # idx_counter += 1

                # 2. EA Answer command
                # answer_file_ea = f"output/{bench_name}/{model_config['model_name']}/t1d7/time/eagle3-speedtest-{num_choices}.jsonl"
                # cmd_ea = (
                #     f"python -m eagle.evaluation.gen_ea_answer_{model_config['script_model_name']} "
                #     f"--ea-model-path {model_config['ea_model_path']} "
                #     f"--base-model-path {model_config['base_model_path']} "
                #     f"--bench-name {bench_name} "
                #     f"{STATIC_PARAMS} "
                #     f"--num-choices {num_choices} "
                #     f"--answer-file {answer_file_ea}"
                # )
                # final_commands.append({
                #     "idx": idx_counter,
                #     "cmd": cmd_ea,
                #     "output": f"{output_path_prefix}{answer_file_ea}"
                # })
                # idx_counter += 1

                # answer_file_ea = f"output/{bench_name}/{model_config['model_name']}/t1d7/time/eagle2-speedtest-3.jsonl"
                # cmd_ea = (
                #     f"python -m eagle.evaluation.gen_ea2_answer_{model_config['script_model_name']} "
                #     f"--ea-model-path {model_config['ea_model_path']} "
                #     f"--base-model-path {model_config['base_model_path']} "
                #     f"--bench-name {bench_name} "
                #     f"{STATIC_PARAMS} "
                #     f"--num-choices 3 "
                #     f"--answer-file {answer_file_ea}"
                # )
                # final_commands.append({
                #     "idx": idx_counter,
                #     "cmd": cmd_ea,
                #     "output": f"{output_path_prefix}{answer_file_ea}"
                # })cd 
                # idx_counter += 1

                # 3. EA WE Answer commands (one for each pt file)
                if len(model_config["pt_list"]) == 0:
                    continue
                for pt_file in model_config["pt_list"]:
                    pt_name = pt_file.replace('.pt', '')
                    answer_file_ea_we = f"output/{bench_name}/{model_config['model_name']}/t1d7/time/hawkeye-{pt_name}-action-{num_choices}.jsonl"
                    # answer_file_ea_we = f"output/{bench_name}/{model_config['model_name']}/t1d7/time/hawkeye-{pt_name}-speedtest-{num_choices}.jsonl"
                    eye_model_path = f"{model_config['eye_model_path_prefix']}{pt_file}"
                    cmd_ea_we = (
                        f"python -m eagle.evaluation.gen_ea_we_answer_{model_config['script_model_name']}_log "
                        # f"python -m eagle.evaluation.gen_ea_we_answer_{model_config['script_model_name']} "
                        f"--ea-model-path {model_config['ea_model_path']} "
                        f"--base-model-path {model_config['base_model_path']} "
                        f"--eye-model-path {eye_model_path} "
                        f"--bench-name {bench_name} "
                        f"{STATIC_PARAMS} "
                        f"--num-choices {num_choices} "
                        f"--answer-file {answer_file_ea_we}"
                    )
                    final_commands.append({
                        "idx": idx_counter,
                        "cmd": cmd_ea_we,
                        "output": f"{output_path_prefix}{answer_file_ea_we}"
                    })
                    idx_counter += 1
    return final_commands

if __name__ == "__main__":
    final_commands_list = generate_commands_in_order()

    with open("commands.json", "w") as f:
        json.dump(final_commands_list, f, indent=4)

    print(f"Successfully generated commands.json with {len(final_commands_list)} commands.")