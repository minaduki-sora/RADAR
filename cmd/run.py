import json
import logging
import os
import shlex
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from queue import Queue, Empty
from threading import Thread

# 尝试导入 pynvml，如果失败则给出提示
try:
    import pynvml
except ImportError:
    print("错误: pynvml 库未安装。请在您的 conda 环境中运行 'pip install pynvml' 进行安装。")
    sys.exit(1)

# --- 配置 ---
COMMANDS_FILE = 'commands.json'
COMPLETED_LOG_FILE = 'completed_tasks.log'
MAIN_LOG_FILE = 'test_runner.log'
IDLE_CHECK_INTERVAL = 600
RUNNING_CHECK_INTERVAL = 300
CONDA_ENV_NAME = 'eagle'

# --- 日志设置 ---
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(MAIN_LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )

# --- I/O 读取线程 (新增) ---
def enqueue_output(stream, queue):
    """将流(stream)的输出逐行放入队列(queue)中"""
    try:
        for line in iter(stream.readline, ''):
            queue.put(line)
    except ValueError:
        # 当流被关闭时，readline 可能会抛出 ValueError
        pass
    finally:
        stream.close()

# --- GPU 监控功能 (与上一版相同) ---
def initialize_nvml():
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
                    processes[i] = [{'pid': p.pid, 'used_memory_mb': p.usedGpuMemory / (1024**2)} for p in procs if p.usedGpuMemory]
            except pynvml.NVMLError:
                # 在某些系统上，没有进程时此调用可能失败
                continue
    except pynvml.NVMLError as e:
        logging.error(f"获取 GPU 信息时出错: {e}")
    return processes

def are_gpus_idle():
    processes = get_gpu_processes()
    if not processes:
        logging.info("GPU 状态：所有 GPU 均空闲。")
        return True
    else:
        log_msg = "GPU 状态：检测到以下进程：\n"
        for gpu_id, procs in processes.items():
            log_msg += f"  - GPU {gpu_id}: {procs}\n"
        logging.warning(log_msg.strip())
        return False

def check_gpus_during_run(test_pid):
    processes = get_gpu_processes()
    external_process_found = False
    for gpu_id, procs in processes.items():
        for proc in procs:
            if proc['pid'] != test_pid:
                logging.error(f"中断！在 GPU {gpu_id} 上检测到外部进程 (PID: {proc['pid']})。")
                external_process_found = True
    return not external_process_found

# --- 主逻辑 (已修改) ---
def run_command_in_conda(command_str, conda_env):
    conda_path = os.environ.get("CONDA_EXE", "conda")
    full_cmd = f'{conda_path} run -n {conda_env} --no-capture-output {command_str}'
    logging.info(f"准备执行命令: {full_cmd}")
    # 【修改点】:
    # 1. 添加 stdout=subprocess.PIPE 和 stderr=subprocess.PIPE 来捕获输出
    # 2. 使用 text=True (或 universal_newlines=True) 使输出为文本字符串
    # 3. 添加 bufsize=1 设置为行缓冲模式
    return subprocess.Popen(
        full_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid
    )

def main():
    setup_logging()
    if not initialize_nvml(): sys.exit(1)

    try:
        with open(COMMANDS_FILE, 'r') as f:
            tasks = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"无法加载命令文件 '{COMMANDS_FILE}': {e}")
        sys.exit(1)

    task_queue = deque(tasks)
    
    try:
        while task_queue:
            logging.info("---------------------------------------------------------")
            logging.info(f"等待所有 GPU 空闲。下次检查在 {IDLE_CHECK_INTERVAL / 60:.1f} 分钟后。")
            
            while not are_gpus_idle():
                time.sleep(IDLE_CHECK_INTERVAL)

            current_task = task_queue.popleft()
            cmd, output_file = current_task['cmd'], current_task['output']
            
            logging.info(f"即将开始新任务: {cmd}")
            process = run_command_in_conda(cmd, CONDA_ENV_NAME)
            test_pid = process.pid
            logging.info(f"任务已启动，进程 PID: {test_pid}")

            # 【修改点】: 创建队列和线程来处理子进程的输出
            q_stdout = Queue()
            q_stderr = Queue()
            t_stdout = Thread(target=enqueue_output, args=(process.stdout, q_stdout))
            t_stderr = Thread(target=enqueue_output, args=(process.stderr, q_stderr))
            t_stdout.daemon = True
            t_stderr.daemon = True
            t_stdout.start()
            t_stderr.start()

            interrupted = False
            task_start_time = time.time()

            while process.poll() is None:
                # 【修改点】: 从队列中读取并记录子进程的输出
                try:
                    while True:
                        line = q_stdout.get_nowait().strip()
                        if line: logging.info(f"[子进程 PID:{test_pid} STDOUT] {line}")
                except Empty:
                    pass
                try:
                    while True:
                        line = q_stderr.get_nowait().strip()
                        if line: logging.error(f"[子进程 PID:{test_pid} STDERR] {line}")
                except Empty:
                    pass

                # 检查外部GPU进程
                if not check_gpus_during_run(test_pid):
                    interrupted = True
                    logging.warning(f"检测到外部 GPU 活动，正在中断任务 PID: {test_pid}")
                    try:
                        os.killpg(os.getpgid(process.pid), 9)
                        logging.info(f"已终止进程组 for PID: {test_pid}")
                    except ProcessLookupError:
                        logging.warning(f"尝试终止进程组时未找到 PID {test_pid}，可能已经结束。")
                    break
                
                elapsed_time = (time.time() - task_start_time)
                if elapsed_time % RUNNING_CHECK_INTERVAL < 1: # 近似检查，避免频繁记录
                    logging.info(f"任务 (PID: {test_pid}) 已运行 {elapsed_time/60:.2f} 分钟。GPU 状态正常。")

                time.sleep(1) # 短暂休眠，避免CPU空转

            process.wait() # 等待进程完全终止
            t_stdout.join() # 等待I/O线程结束
            t_stderr.join()

            # 任务结束后的处理... (与之前版本相同)
            if interrupted:
                logging.error(f"任务因外部GPU活动被中断: {cmd}")
                if os.path.exists(output_file):
                    try:
                        os.remove(output_file)
                        logging.info(f"已删除不完整的输出文件: {output_file}")
                    except OSError as e:
                        logging.error(f"删除文件 {output_file} 失败: {e}")
                task_queue.appendleft(current_task)
                logging.info("任务已重新加入队列。")
            elif process.returncode == 0:
                logging.info(f"任务成功完成: {cmd}")
                with open(COMPLETED_LOG_FILE, 'a') as f:
                    f.write(f"{datetime.now().isoformat()} - SUCCESS - {cmd}\n")
            else:
                logging.error(f"任务失败，返回码: {process.returncode}。命令: {cmd}")

        logging.info("所有任务均已成功完成！")
    except KeyboardInterrupt:
        logging.info("检测到手动中断 (Ctrl+C)。正在退出...")
    finally:
        pynvml.nvmlShutdown()
        logging.info("NVML 已关闭。程序退出。")

if __name__ == '__main__':
    main()