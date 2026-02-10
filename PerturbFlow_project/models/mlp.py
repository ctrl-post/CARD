"""
CausalKAN-Flow MLP模块
实现多层感知机组件
"""

import torch
import torch.nn as nn

class MLP(nn.Module):
    """多层感知机 - 用于特征转换和融合"""
    
    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        """
        初始化多层感知机
        """
        super().__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                nn.Linear(sizes[s], sizes[s + 1]),
                nn.BatchNorm1d(sizes[s + 1]) if batch_norm and s < len(sizes) - 1 else None,
                nn.ReLU()
            ]
        
        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        self.network = nn.Sequential(*layers)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """前向传播"""
        return self.network(x)
