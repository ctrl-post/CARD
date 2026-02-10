"""
CausalKAN-Flow 数据工具模块
提供预测过程中创建细胞图数据的数据处理相关的工具函数
"""

import torch
import numpy as np
from torch_geometric.data import Data
from typing import List, Optional
from typing import Any
from .general_utils import print_system

def create_cell_graph_for_prediction(
    X: np.ndarray, 
    perturbation_idx: Optional[List[int]], 
    perturbation_genes: List[str]
) -> Data:
    """为预测创建细胞图"""
    
    if perturbation_idx is None:
        perturbation_idx = [-1]
    
    # 创建图数据对象
    return Data(
        x=torch.Tensor(X).T,
        perturbation_idx=perturbation_idx,
        perturbation=perturbation_genes
    )

def create_cell_graph_dataset_for_prediction(
    perturbation_genes: List[str], 
    control_adata: Any, 
    gene_names: List[str],
    device: torch.device, 
    num_samples: int = 300
) -> List[Data]:
    """为预测创建细胞图数据集"""
    
    # 获取扰动的索引
    perturbation_idx = [
        np.where(p == np.array(gene_names))[0][0] for p in perturbation_genes
    ]
    
    # 随机采样对照组细胞
    Xs = control_adata[
        np.random.randint(0, len(control_adata), num_samples), :
    ].X.toarray()
    
    # 创建细胞图
    cell_graphs = [
        create_cell_graph_for_prediction(X, perturbation_idx, perturbation_genes).to(device) 
        for X in Xs
    ]
    
    return cell_graphs