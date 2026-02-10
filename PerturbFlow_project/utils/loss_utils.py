"""
CausalKAN-Flow 损失函数模块
提供模型训练中使用的各种损失函数
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from .general_utils import print_system

def uncertainty_loss_function(
    pred: torch.Tensor, 
    logvar: torch.Tensor, 
    y: torch.Tensor, 
    perturbations: List[str], 
    reg: float = 0.1, 
    control: Optional[torch.Tensor] = None,
    direction_lambda: float = 1e-3, 
    filter_dict: Optional[Dict] = None
) -> torch.Tensor:
    """不确定性损失函数"""
    
    gamma = 2
    perturbations = np.array(perturbations)
    losses = torch.tensor(0.0, requires_grad=True).to(pred.device)
    
    for p in set(perturbations):
        if p != 'ctrl':
            retain_idx = filter_dict[p]
            pred_p = pred[np.where(perturbations == p)[0]][:, retain_idx]
            y_p = y[np.where(perturbations == p)[0]][:, retain_idx]
            logvar_p = logvar[np.where(perturbations == p)[0]][:, retain_idx]
        else:
            pred_p = pred[np.where(perturbations == p)[0]]
            y_p = y[np.where(perturbations == p)[0]]
            logvar_p = logvar[np.where(perturbations == p)[0]]
        
        # 不确定性损失
        losses += torch.sum(
            (pred_p - y_p)**(2 + gamma) + reg * torch.exp(-logvar_p) * (pred_p - y_p)**(2 + gamma)
        ) / pred_p.shape[0] / pred_p.shape[1]
        
        # 方向损失
        if p != 'ctrl':
            losses += torch.sum(
                direction_lambda * (torch.sign(y_p - control[retain_idx]) - torch.sign(pred_p - control[retain_idx]))**2
            ) / pred_p.shape[0] / pred_p.shape[1]
        else:
            losses += torch.sum(
                direction_lambda * (torch.sign(y_p - control) - torch.sign(pred_p - control))**2
            ) / pred_p.shape[0] / pred_p.shape[1]
    
    return losses / (len(set(perturbations)))

def loss_function(
    pred: torch.Tensor, 
    y: torch.Tensor, 
    perturbations: List[str], 
    control: Optional[torch.Tensor] = None, 
    direction_lambda: float = 1e-3, 
    filter_dict: Optional[Dict] = None
) -> torch.Tensor:
    """主MSE损失函数，包含方向损失"""
    gamma = 2
    perturbations = np.array(perturbations)
    losses = torch.tensor(0.0, requires_grad=True).to(pred.device)
    
    for p in set(perturbations):
        perturbation_idx = np.where(perturbations == p)[0]
        
        # during training, we remove the all zero genes into calculation of loss.
        # this gives a cleaner direction loss. empirically, the performance stays the same.
        if p != 'ctrl':
            retain_idx = filter_dict[p]
            pred_p = pred[perturbation_idx][:, retain_idx]
            y_p = y[perturbation_idx][:, retain_idx]
        else:
            pred_p = pred[perturbation_idx]
            y_p = y[perturbation_idx]
        
        losses = losses + torch.sum((pred_p - y_p)**(2 + gamma)) / pred_p.shape[0] / pred_p.shape[1]
        
        ## direction loss
        if p != 'ctrl':
            losses = losses + torch.sum(
                direction_lambda * (torch.sign(y_p - control[retain_idx]) - torch.sign(pred_p - control[retain_idx]))**2
            ) / pred_p.shape[0] / pred_p.shape[1]
        else:
            losses = losses + torch.sum(
                direction_lambda * (torch.sign(y_p - control) - torch.sign(pred_p - control))**2
            ) / pred_p.shape[0] / pred_p.shape[1]
    
    return losses / (len(set(perturbations)))