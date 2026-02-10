"""
PerturbFlow 交互模型模块
处理双基因扰动之间的交互效应
"""

import torch
import torch.nn as nn
from utils.general_utils import print_system

class DualPerturbationInteraction(nn.Module):
    """双基因扰动交互模型 - 处理两个基因扰动之间的协同效应(官方实现)"""
    
    def __init__(self, hidden_dim):
        """
        初始化双基因扰动交互模型
        
        参数:
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.hidden_dim = hidden_dim
    
    def forward(self, emb1, emb2):
        """前向传播 - 计算两个扰动的交互效应"""
        # 确保所有张量在相同设备上 - 使用PyTorch自动设备追踪
        device = emb1.device
        emb2 = emb2.to(device)
        # 相加交互
        interaction = emb1 + emb2
        return interaction
