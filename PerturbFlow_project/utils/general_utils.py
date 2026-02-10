"""
CausalKAN-Flow 通用工具模块
提供基本的工具函数和辅助功能
"""

import sys
import torch
from typing import List, Tuple, Any
from time import perf_counter_ns
from datetime import datetime, timedelta
from contextlib import redirect_stdout, redirect_stderr
import torch
import random
import numpy as np
import subprocess  # 用于检查系统时间同步状态
import os

def setup_random_seed(seed: int) -> None:
    """设置全局随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)  # 设置Python哈希种子
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.enabled = True  # 启用cuDNN
    torch.use_deterministic_algorithms(True)  # 使PyTorch使用确定性算法

def setup_separated_randomness(data_seed: int, model_seed: int = 1) -> dict:
    """
    设置分离的随机种子控制系统
    - 数据划分：使用不同种子，评估数据划分的影响
    - 模型训练：固定种子，确保模型一致性

    参数:
        data_seed: 数据划分的随机种子（可变，评估数据划分影响）
        model_seed: 模型训练和初始化的随机种子（默认1，确保一致性）

    返回:
        dict: 包含种子配置信息的字典
    """

    # 设置模型训练种子（固定）
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(model_seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.enabled = True
    torch.use_deterministic_algorithms(True)

    # 保存种子配置
    seed_config = {
        'data_seed': data_seed,
        'model_seed': model_seed,
        'setup_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    return seed_config

def get_default_model_seed() -> int:
    """获取默认的模型训练种子（固定）"""
    return 1

def print_seed_configuration(seed_config: dict) -> None:
    """
    打印种子配置信息

    参数:
        seed_config: 种子配置字典
    """
    print_system("🎯 种子配置信息:")
    print_system(f"  数据种子: {seed_config['data_seed']}")
    print_system(f"  模型种子: {seed_config['model_seed']}")
    print_system(f"  设置时间: {seed_config['setup_time']}")

def check_ntp_sync():
    """检查Linux系统时间同步状态"""
    try:
        # 检查chronyd状态
        import subprocess
        chrony = subprocess.run(['chronyc', 'tracking'], 
                              capture_output=True, text=True)
        if chrony.returncode == 0:
            print(chrony.stdout)
            return
        
        # 检查ntpd状态  
        ntp = subprocess.run(['ntpq', '-p'], 
                           capture_output=True, text=True)
        if ntp.returncode == 0:
            print(ntp.stdout)
            return
        
        # 检查systemd-timesyncd状态
        timedatectl = subprocess.run(['timedatectl', 'status'],
                                   capture_output=True, text=True)
        print(timedatectl.stdout)
        
    except FileNotFoundError:
        print("未找到时间同步检查工具，跳过系统时间验证")

def format_ns(ns):
    """智能格式化时间（自动选择单位）"""
    if ns < 1000:
        return f"{ns} ns"
    elif ns < 1_000_000:
        return f"{ns/1000:.3f} μs" 
    elif ns < 1_000_000_000:
        return f"{ns/1_000_000:.3f} ms"
    else:
        return f"{ns/1_000_000_000:.3f} s"

class PrecisionTimer:
    """高精度计时上下文管理器"""
    def __enter__(self):
        self.start = perf_counter_ns()
        return self
    
    def __exit__(self, *args):
        self.end = perf_counter_ns()
        self.duration = self.end - self.start
    
    def report(self, name):
        print(f"{name}耗时: {format_ns(self.duration)}")

def parse_single_perturbation(perturbation: str) -> str:
    """解析单一扰动"""
    parts = perturbation.split('+')
    if parts[0] == 'ctrl':
        return parts[1]
    else:
        return parts[0]

def parse_combo_perturbation(perturbation: str) -> Tuple[str, str]:
    """解析组合扰动"""
    parts = perturbation.split('+')
    return parts[0], parts[1]

def parse_any_perturbation(perturbation: str) -> List[str]:
    """解析任意类型的扰动"""
    if ('ctrl' in perturbation) and (perturbation != 'ctrl'):
        return [parse_single_perturbation(perturbation)]
    elif 'ctrl' not in perturbation:
        out = parse_combo_perturbation(perturbation)
        return [out[0], out[1]]
    return []

def condition_sort(condition: str) -> str:
    """对条件名称进行排序，确保"A+B"和"B+A"被视为相同条件"""
    return '+'.join(sorted(condition.split('+')))



# 全局日志文件路径
LOG_FILE = None
LOG_DIR = "./logs"

def setup_logger(dataset_name: str, seed: int) -> None:
    """设置日志记录器"""
    global LOG_FILE
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    LOG_FILE = os.path.join(LOG_DIR, f"log_{dataset_name}_seed_{seed}_{timestamp}.log")

def print_system(message: str) -> None:
    """系统打印函数（包含文件记录）"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted_message = f"[{timestamp}] {message}"
    
    # 打印到控制台
    print(formatted_message, flush=True, file=sys.stderr)
    
    # 写入日志文件
    if LOG_FILE is not None:
        try:
            with open(LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(formatted_message + '\n')
        except Exception as e:
            print(f"日志写入错误: {e}", file=sys.stderr)