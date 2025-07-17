#!/bin/bash

# 配置参数
LOG_DIR="./gpu_monitor_logs"  # 日志存放目录
INTERVAL=300                  # 5分钟（单位：秒）
MAX_DAYS=30                   # 日志保留天数
DURATION=0                 # 运行时长（单位：秒），0表示无限运行

# 显示帮助信息
usage() {
    echo "Usage: $0 [-d duration_seconds] [-i interval_seconds] [-l log_directory] [-m max_days]"
    echo "Options:"
    echo "  -d 运行时长（秒），0表示无限运行（默认：0）"
    echo "  -i 监控间隔时间（秒）（默认：300）"
    echo "  -l 日志存放目录（默认：./gpu_monitor_logs）"
    echo "  -m 日志保留天数（默认：30）"
    exit 1
}

# 解析命令行参数
while getopts ":d:i:l:m:h" opt; do
    case $opt in
        d) DURATION="$OPTARG" ;;
        i) INTERVAL="$OPTARG" ;;
        l) LOG_DIR="$OPTARG" ;;
        m) MAX_DAYS="$OPTARG" ;;
        h) usage ;;
        \?) echo "无效选项: -$OPTARG" >&2; usage ;;
        :) echo "选项 -$OPTARG 需要参数." >&2; usage ;;
    esac
done

# 创建日志目录
mkdir -p "$LOG_DIR"

# 计算结束时间（如果设置了DURATION）
if [ "$DURATION" -gt 0 ]; then
    RUN_END_TIME=$(( $(date +%s) + DURATION ))
    echo "监控将在 $(date -d @$RUN_END_TIME +'%Y-%m-%d %H:%M:%S') 自动结束"
fi

# 主监控循环
while true; do
    # 检查是否达到运行时长限制
    if [ "$DURATION" -gt 0 ] && [ $(date +%s) -ge $RUN_END_TIME ]; then
        echo "达到设定的运行时长，监控结束"
        exit 0
    fi
    
    # 生成带时间戳的文件名
    LOG_FILE="${LOG_DIR}/gpu_log_$(date +'%Y%m%d_%H%M%S').log"
    
    # 获取精确开始时间
    START_TIME=$(date +'%Y-%m-%d %H:%M:%S.%3N')
    
    # 记录开始标记和完整命令输出
    echo "===== GPU MONITOR RECORD [START] $(date +'%Y-%m-%d %H:%M:%S %Z') =====" > "$LOG_FILE"
    echo "Monitoring started at: $START_TIME" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    
    # 执行并完整记录nvidia-smi输出
    nvidia-smi >> "$LOG_FILE" 2>&1
    
    # 添加结束标记和时间统计
    END_TIME=$(date +'%Y-%m-%d %H:%M:%S.%3N')
    echo "" >> "$LOG_FILE"
    echo "Monitoring ended at: $END_TIME" >> "$LOG_FILE"
    echo "===== GPU MONITOR RECORD [END] =====" >> "$LOG_FILE"
    
    # 清理旧日志（保持最近MAX_DAYS天）
    find "$LOG_DIR" -name "gpu_log_*.log" -mtime +$MAX_DAYS -exec rm {} \;
    
    # 等待下一个采集周期
    sleep $INTERVAL
done