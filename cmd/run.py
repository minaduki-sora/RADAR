import argparse
import json
import logging
import os
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

# --- 默认配置 ---
DEFAULT_COMMANDS_FILE = 'commands.json'
DEFAULT_COMPLETED_LOG = 'completed_tasks.log'
DEFAULT_MAIN_LOG = 'test_runner.log'
IDLE_CHECK_INTERVAL = 600  # 任务空闲时检查GPU的间隔（秒）
RUNNING_CHECK_INTERVAL = 300  # 任务运行时检查GPU的间隔（秒）
CONDA_ENV_NAME = 'eagle'

# --- 日志设置 ---
def setup_logging(log_file_path):
    """配置日志记录器"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout)
        ]
    )

# --- I/O 读取线程 ---
def enqueue_output(stream, queue):
    """将流(stream)的输出逐行放入队列(queue)中"""
    try:
        for line in iter(stream.readline, ''):
            queue.put(line)
    except (ValueError, IOError):
        # 当流被关闭时，readline 可能会抛出异常
        pass
    finally:
        if not stream.closed:
            stream.close()

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
    """获取所有 GPU 上的正在运行的计算和图形进程信息"""
    all_processes = {}
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            seen_pids = set()
            procs_info = []
            process_types = {
                "Compute": pynvml.nvmlDeviceGetComputeRunningProcesses,
                "Graphics": pynvml.nvmlDeviceGetGraphicsRunningProcesses
            }
            for p_type, getter in process_types.items():
                try:
                    procs = getter(handle)
                    if not procs: continue
                    for p in procs:
                        if p.pid not in seen_pids:
                            used_memory = p.usedGpuMemory
                            used_memory_mb = 'N/A' if used_memory is None else f"{used_memory / (1024**2):.2f}"
                            procs_info.append({'pid': p.pid, 'used_memory_mb': used_memory_mb})
                            seen_pids.add(p.pid)
                except pynvml.NVMLError as e:
                    logging.debug(f"无法获取 GPU {i} 的 {p_type} 进程: {e}")
            if procs_info:
                all_processes[i] = procs_info
    except pynvml.NVMLError as e:
        logging.error(f"获取 GPU 信息时出错: {e}")
    return all_processes

def are_gpus_idle():
    """检查所有 GPU 是否都处于空闲状态"""
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

def check_gpus_during_run(task_pgid):
    """
    在测试运行时检查 GPU，忽略属于当前任务进程组的所有进程。
    :param task_pgid: 我们启动的任务的进程组ID。
    """
    processes_on_gpu = get_gpu_processes()
    external_process_found = False
    for gpu_id, procs in processes_on_gpu.items():
        for proc in procs:
            gpu_proc_pid = proc['pid']
            try:
                # 获取该GPU上进程的进程组ID
                gpu_proc_pgid = os.getpgid(gpu_proc_pid)
                
                # 如果这个进程的PGID不等于我们任务的PGID，它就是外部进程
                if gpu_proc_pgid != task_pgid:
                    logging.error(f"中断！在 GPU {gpu_id} 上检测到外部进程 (PID: {gpu_proc_pid}, PGID: {gpu_proc_pgid})，"
                                  f"不属于我们的任务进程组 (PGID: {task_pgid})。")
                    external_process_found = True
            except ProcessLookupError:
                # 如果在我们检查之前进程就结束了，忽略它
                logging.debug(f"检查 PGID 时未找到进程 {gpu_proc_pid}，可能已结束。")
                continue
    
    return not external_process_found

# --- 主逻辑 ---
def run_command_in_conda(command_str, conda_env, workplace):
    """在指定的 Conda 环境中运行命令"""
    conda_path = os.environ.get("CONDA_EXE", "conda")
    full_cmd = f'{conda_path} run -n {conda_env} --no-capture-output {command_str}'
    logging.info(f"准备在目录 '{workplace}' 中执行命令: {full_cmd}")
    
    return subprocess.Popen(
        full_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, preexec_fn=os.setsid, cwd=workplace
    )

def main():
    """主执行函数"""
    parser = argparse.ArgumentParser(description="一个健壮的GPU测试任务执行器。")
    parser.add_argument('--workplace', type=str, required=True, help="项目根目录（工作区）。")
    parser.add_argument('--commands-file', type=str, default=DEFAULT_COMMANDS_FILE, help=f"命令JSON文件的路径，相对于工作区。")
    args = parser.parse_args()
    
    workplace = os.path.abspath(args.workplace)
    commands_file_path = os.path.join(workplace, args.commands_file)
    completed_log_path = os.path.join(workplace, DEFAULT_COMPLETED_LOG)
    main_log_path = os.path.join(workplace, DEFAULT_MAIN_LOG)
    
    if not os.path.isdir(workplace):
        print(f"错误：指定的工作目录不存在: {workplace}")
        sys.exit(1)

    setup_logging(main_log_path)
    logging.info(f"脚本启动。工作目录: {workplace}")
    logging.info(f"读取命令自: {commands_file_path}")
    
    if not initialize_nvml(): sys.exit(1)

    try:
        with open(commands_file_path, 'r') as f:
            tasks = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"无法加载命令文件 '{commands_file_path}': {e}")
        sys.exit(1)

    task_queue = deque(tasks[16:21]) # 调整任务起止
    print(f"读取到 {len(task_queue)} 个任务。")
    
    try:
        while task_queue:
            logging.info("---------------------------------------------------------")
            logging.info(f"等待所有 GPU 空闲。下次检查在 {IDLE_CHECK_INTERVAL / 60:.1f} 分钟后。")
            
            while not are_gpus_idle():
                time.sleep(IDLE_CHECK_INTERVAL)

            current_task = task_queue.popleft()
            cmd = current_task['cmd']
            output_file_path = os.path.join(workplace, current_task['output']) if current_task.get('output') else None
            
            logging.info(f"即将开始新任务: {cmd}")
            process = run_command_in_conda(cmd, CONDA_ENV_NAME, workplace)
            
            # 由于 preexec_fn=os.setsid, process.pid 也是新进程组的ID (PGID)
            task_pgid = process.pid
            logging.info(f"任务已启动，进程组 ID (PGID): {task_pgid}")

            q_stdout, q_stderr = Queue(), Queue()
            t_stdout = Thread(target=enqueue_output, args=(process.stdout, q_stdout))
            t_stderr = Thread(target=enqueue_output, args=(process.stderr, q_stderr))
            t_stdout.daemon = t_stderr.daemon = True
            t_stdout.start(); t_stderr.start()

            interrupted = False
            task_start_time = time.time()
            
            while process.poll() is None:
                try:
                    while True:
                        line = q_stdout.get_nowait().strip()
                        if line: logging.info(f"[任务 PGID:{task_pgid} STDOUT] {line}")
                except Empty: pass
                try:
                    while True:
                        line = q_stderr.get_nowait().strip()
                        if line: logging.error(f"[任务 PGID:{task_pgid} STDERR] {line}")
                except Empty: pass

                # 使用 PGID 进行检查
                if not check_gpus_during_run(task_pgid):
                    interrupted = True
                    logging.warning(f"检测到外部 GPU 活动，正在中断任务组 PGID: {task_pgid}")
                    try:
                        os.killpg(task_pgid, 9) # SIGKILL
                        logging.info(f"已终止进程组 PGID: {task_pgid}")
                    except ProcessLookupError:
                        logging.warning(f"尝试终止进程组时未找到 PGID {task_pgid}，可能已经结束。")
                    break
                
                time.sleep(1)

            process.wait(); t_stdout.join(); t_stderr.join()

            if interrupted:
                logging.error(f"任务因外部GPU活动被中断: {cmd}")
                if output_file_path and os.path.exists(output_file_path):
                    try:
                        os.remove(output_file_path)
                        logging.info(f"已删除不完整的输出文件: {output_file_path}")
                    except OSError as e:
                        logging.error(f"删除文件 {output_file_path} 失败: {e}")
                task_queue.appendleft(current_task)
                logging.info("任务已重新加入队列。")
            elif process.returncode == 0:
                logging.info(f"任务成功完成: {cmd}")
                with open(completed_log_path, 'a') as f:
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