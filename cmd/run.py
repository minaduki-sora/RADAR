import json
import logging
import os
import shlex
import subprocess
import sys
import time
from collections import deque
from datetime import datetime

import pynvml

# --- 配置 ---
COMMANDS_FILE = 'commands.json'
COMPLETED_LOG_FILE = 'completed_tasks.txt'
MAIN_LOG_FILE = 'test_runner.txt'

# 检查GPU是否空闲的间隔时间（秒）
IDLE_CHECK_INTERVAL = 600  # 10 分钟

# 任务运行时检查GPU状态的间隔时间（秒）
RUNNING_CHECK_INTERVAL = 300  # 5 分钟

# Conda 环境名
CONDA_ENV_NAME = 'eagle'

# --- 日志设置 ---
def setup_logging():
    """配置日志记录器"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(MAIN_LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )

# --- GPU 监控功能 ---
def initialize_nvml():
    """初始化 NVML 库"""
    try:
        pynvml.nvmlInit()
        logging.info("成功初始化 NVML。")
        return True
    except pynvml.NVMLError as e:
        logging.error(f"无法初始化 NVML: {e}。请确保 NVIDIA 驱动已正确安装。")
        return False

def get_gpu_processes():
    """获取所有 GPU 上的正在运行的进程信息"""
    processes = {}
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                if procs:
                    processes[i] = [{'pid': p.pid, 'used_memory_mb': p.usedGpuMemory / (1024**2)} for p in procs]
            except pynvml.NVMLError:
                # 在某些系统上，没有进程时此调用可能失败
                continue
    except pynvml.NVMLError as e:
        logging.error(f"获取 GPU 信息时出错: {e}")
    return processes

def are_gpus_idle():
    """检查所有 GPU 是否都处于空闲状态"""
    processes = get_gpu_processes()
    if not processes:
        logging.info("GPU 状态：所有 GPU 均空闲。")
        return True
    else:
        logging.warning(f"GPU 状态：检测到以下进程：{processes}")
        return False

def check_gpus_during_run(test_pid):
    """在测试运行时检查 GPU，忽略当前的测试进程"""
    processes = get_gpu_processes()
    for gpu_id, procs in processes.items():
        for proc in procs:
            if proc['pid'] != test_pid:
                logging.error(f"中断！在 GPU {gpu_id} 上检测到外部进程 (PID: {proc['pid']})。")
                return False
    return True

# --- 主逻辑 ---
def run_command_in_conda(command_str, conda_env):
    """在指定的 Conda 环境中运行命令"""
    # 找到 conda 的路径
    conda_path = os.environ.get("CONDA_EXE")
    if not conda_path:
        # 如果环境变量中没有，尝试 common-path
        conda_path = "conda"

    # 构建完整的 shell 命令
    # 'source activate' 在子进程中可能不起作用，使用 'conda run' 更可靠
    full_cmd = f'{conda_path} run -n {conda_env} --no-capture-output {command_str}'
    
    logging.info(f"准备执行命令: {full_cmd}")
    
    # 使用 shlex.split 处理命令字符串，以避免 shell 注入问题
    # 但由于 'conda run' 的复杂性，这里我们使用 shell=True，并确保 command_str 是可信的
    # 注意：在生产环境中，要非常小心 shell=True
    return subprocess.Popen(full_cmd, shell=True, preexec_fn=os.setsid)


def main():
    """主执行函数"""
    setup_logging()
    
    if not initialize_nvml():
        sys.exit(1)

    try:
        with open(COMMANDS_FILE, 'r') as f:
            tasks = json.load(f)
    except FileNotFoundError:
        logging.error(f"错误: 命令文件 '{COMMANDS_FILE}' 不存在。")
        sys.exit(1)
    except json.JSONDecodeError:
        logging.error(f"错误: 命令文件 '{COMMANDS_FILE}' 格式不正确。")
        sys.exit(1)

    task_queue = deque(tasks)
    
    try:
        while task_queue:
            logging.info("---------------------------------------------------------")
            logging.info(f"等待所有 GPU 空闲。下次检查在 {IDLE_CHECK_INTERVAL / 60:.1f} 分钟后。")
            
            while not are_gpus_idle():
                time.sleep(IDLE_CHECK_INTERVAL)

            # GPU 空闲，可以开始新任务
            current_task = task_queue.popleft()
            cmd = current_task['cmd']
            output_file = current_task['output']
            
            logging.info(f"即将开始新任务: {cmd}")

            # 启动任务子进程
            process = run_command_in_conda(cmd, CONDA_ENV_NAME)
            test_pid = process.pid
            logging.info(f"任务已启动，进程 PID: {test_pid}")

            interrupted = False
            task_start_time = time.time()

            # 监控任务执行
            while process.poll() is None: # 当进程仍在运行时
                time.sleep(RUNNING_CHECK_INTERVAL)
                
                # 检查是否有外部 GPU 进程
                if not check_gpus_during_run(test_pid):
                    interrupted = True
                    logging.warning(f"检测到外部 GPU 活动，正在中断任务 PID: {test_pid}")
                    
                    # 使用 os.killpg 发送信号到整个进程组，确保所有子进程都被终止
                    os.killpg(os.getpgid(process.pid), 9) # SIGKILL
                    
                    logging.info(f"已终止进程组 for PID: {test_pid}")
                    break
                else:
                    elapsed_time = (time.time() - task_start_time) / 60
                    logging.info(f"任务 (PID: {test_pid}) 已运行 {elapsed_time:.2f} 分钟。GPU 状态正常。")

            # 任务结束后的处理
            if interrupted:
                logging.error(f"任务被中断: {cmd}")
                
                # 删除不完整的输出文件
                if os.path.exists(output_file):
                    try:
                        os.remove(output_file)
                        logging.info(f"已删除不完整的输出文件: {output_file}")
                    except OSError as e:
                        logging.error(f"删除文件 {output_file} 失败: {e}")
                
                # 将任务重新放回队列头部，等待下次执行
                task_queue.appendleft(current_task)
                logging.info("任务已重新加入队列。")
            
            elif process.returncode == 0:
                logging.info(f"任务成功完成: {cmd}")
                with open(COMPLETED_LOG_FILE, 'a') as f:
                    f.write(f"{datetime.now().isoformat()} - SUCCESS - {cmd}\n")
            
            else:
                logging.error(f"任务失败，返回码: {process.returncode}。命令: {cmd}")
                # 根据您的需求，决定失败的任务是否也需要重新排队
                # task_queue.appendleft(current_task) 
                # logging.info("失败的任务已重新加入队列。")

        logging.info("所有任务均已成功完成！")

    except KeyboardInterrupt:
        logging.info("检测到手动中断 (Ctrl+C)。正在退出...")
    finally:
        pynvml.nvmlShutdown()
        logging.info("NVML 已关闭。程序退出。")


if __name__ == '__main__':
    main()