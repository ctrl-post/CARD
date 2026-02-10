"""
CausalKAN-Flow 训练器模块
负责模型的训练、验证和测试流程
"""

import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import numpy as np
from typing import Dict, Any, Tuple
from copy import deepcopy

from utils.loss_utils import loss_function, uncertainty_loss_function
from .evaluator import evaluate_model, compute_metrics, deeper_analysis, non_dropout_analysis
from .fast_evaluator import enhanced_compute_metrics_fast
from utils.general_utils import print_system


def update_ema(target_params, source_params, rate=0.99):
    """
    使用指数移动平均更新目标参数

    参数:
        target_params: 目标参数序列
        source_params: 源参数序列
        rate: EMA 率 (越接近1更新越慢)
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


class PerturbFlowTrainer:
    """PerturbFlow 训练器 - 处理模型训练和评估"""
    
    def __init__(self, model, config, model_config=None, device='cuda'):
        """
        初始化训练器

        参数:
            model: 要训练的模型
            config: 训练配置 (TrainingConfig)
            model_config: 模型配置 (ModelConfig)，可选，为保持向后兼容性
            device: 训练设备
        """
        self.model = model
        self.config = config
        self.model_config = model_config
        # 确保device是torch.device对象
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.best_model = None
        self.train_history = []
        self.adata = None  # 添加adata属性用于存储数据
        self.split = None  # 添加split属性用于存储分割类型
        
        # 初始化权重和偏置跟踪（如果配置）
        if self.config.use_wandb:
            import wandb
            wandb.init(project=self.config.wandb_project, name=self.config.wandb_experiment)
            self.wandb = wandb
            
            # 根据是否提供model_config来决定使用哪个配置对象的属性
            if self.model_config is not None:
                wandb_config = {
                    'hidden_size': self.model_config.hidden_size,
                    'num_gene_gnn_layers': self.model_config.num_gene_gnn_layers,
                    'decoder_hidden_size': self.model_config.decoder_hidden_size,
                    'num_similar_genes_co_express_graph': self.model_config.num_similar_genes_co_express_graph,
                    'coexpress_threshold': self.model_config.coexpression_threshold,
                    'uncertainty': self.model_config.uncertainty,
                    'uncertainty_reg': self.model_config.uncertainty_reg,
                    'direction_lambda': self.model_config.direction_lambda,
                    'device': self.device,
                    'no_perturbation': self.model_config.no_perturbation,
                }
            else:
                # 向后兼容：尝试从config中访问属性，可能导致错误
                wandb_config = {
                    'hidden_size': self.model_config.hidden_size,
                    'num_gene_gnn_layers': self.model_config.num_gene_gnn_layers,
                    'decoder_hidden_size': self.model_config.decoder_hidden_size,
                    'num_similar_genes_co_express_graph': self.model_config.num_similar_genes_co_express_graph,
                    'coexpress_threshold': self.model_config.coexpression_threshold,
                    'uncertainty': self.model_config.uncertainty,
                    'uncertainty_reg': self.model_config.uncertainty_reg,
                    'direction_lambda': self.model_config.direction_lambda,
                    'device': self.device,
                    'no_perturbation': self.model_config.no_perturbation,
                }
            
            self.wandb.config.update(wandb_config, allow_val_change=True)
        else:
            self.wandb = None

        # 初始化 EMA (指数移动平均) 参数
        self.use_ema = getattr(self.config, 'use_ema', False)
        self.ema_params = []
        self.best_model_state = None  # 用于保存最佳模型状态

        if self.use_ema:
            self.ema_rate = getattr(self.config, 'ema_rate', 0.9999)
            # 复制模型参数作为 EMA 参数，并确保在同一设备上
            self.ema_params = [p.clone().detach().to(p.device) for p in self.model.parameters()]
        else:
            print_system("[EMA] 未启用指数移动平均")

    
    def setup_optimizer(self) -> Tuple[optim.Optimizer, Any]:
        """设置优化器和学习率调度器"""
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        scheduler = StepLR(optimizer, step_size=self.config.step_size, gamma=self.config.scheduler_gamma)
        return optimizer, scheduler
    
    def train_epoch(self, train_loader, optimizer, epoch) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_loader):
            batch = batch.to(self.device)
            optimizer.zero_grad()
            
            # 获取目标值
            y = batch.y
            
            # 前向传播和损失计算
            if self.model_config is not None and self.model_config.uncertainty:
                model_output = self.model(batch)
                # 处理ModelOutput格式
                if hasattr(model_output, 'pred') and hasattr(model_output, 'logvar'):
                    pred = model_output.pred
                    logvar = model_output.logvar
                elif isinstance(model_output, tuple) and len(model_output) == 2:
                    pred, logvar = model_output
                else:
                    # 如果没有logvar，创建一个默认值
                    pred = model_output.pred if hasattr(model_output, 'pred') else model_output
                    logvar = torch.zeros_like(pred)

                loss = uncertainty_loss_function(
                    pred, logvar, y, batch.pert,
                    reg=self.model_config.uncertainty_reg,
                    control=self.ctrl_expression,
                    filter_dict=self.dict_filter,
                    direction_lambda=self.model_config.direction_lambda
                )
            elif self.model_config is not None and self.model_config.no_perturbation:
                # no perturb模式：模型返回(loss, pred)元组以保持接口一致
                loss_result = self.model(batch)
                if isinstance(loss_result, tuple) and len(loss_result) == 2:
                    loss, pred = loss_result
                else:
                    # 如果只返回预测值（安全backup）
                    pred = loss_result
                    loss = nn.functional.mse_loss(pred, y)
            else:
                # 普通模式：模型返回ModelOutput对象或元组
                loss_result = self.model(batch)

                # 处理ModelOutput格式（新格式）
                if hasattr(loss_result, 'pred') and hasattr(loss_result, 'loss'):
                    pred = loss_result.pred
                    loss = loss_result.loss
                    loss_components = getattr(loss_result, 'loss_components', None)
                # 处理元组格式（旧格式，向后兼容）
                elif isinstance(loss_result, tuple):
                    if len(loss_result) == 3:
                        loss, pred, loss_components = loss_result
                    elif len(loss_result) == 2:
                        loss, pred = loss_result
                        loss_components = None
                    else:
                        # 异常情况处理
                        pred = loss_result[0]
                        loss = nn.functional.mse_loss(pred, y)
                        loss_components = None
                else:
                    # 如果只返回预测值（异常情况处理）
                    pred = loss_result
                    loss = nn.functional.mse_loss(pred, y)
                    loss_components = None

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            optimizer.step()

            # 更新 EMA 参数
            if self.use_ema and len(self.ema_params) > 0:
                update_ema(self.ema_params, list(self.model.parameters()), rate=self.ema_rate)

            total_loss += loss.item()
            
            # 记录训练损失和分量
            if self.wandb:
                log_dict = {'train_total_loss': loss.item()}
                if loss_components is not None:
                    log_dict.update({
                        'train_nll_loss': loss_components['nll_loss'].item(),
                        'train_adjusted_nll': loss_components['adjusted_nll'].item(),
                        'train_full_mse': loss_components['full_mse'].item(),
                        'train_top20_mse': loss_components['top20_mse'].item(),
                        'train_correlation': loss_components['correlation'].item(),
                        'train_recon_loss': loss_components['recon_loss'].item(),
                        'nll_category': loss_components['nll_category']
                    })
                self.wandb.log(log_dict)

        return total_loss / len(train_loader)
    
    def validate(self, val_loader, compute_loss_components=False) -> Dict[str, float]:
        """验证模型性能

        参数:
            val_loader: 验证数据加载器
            compute_loss_components: 是否计算损失分量（用于验证损失曲线）

        返回:
            val_metrics: 验证指标字典
        """
        self.model.eval()

        if compute_loss_components:
            # 计算验证损失分量
            return self._validate_with_loss_components(val_loader)
        else:
            
            uncertainty_enabled = self.model_config is not None and self.model_config.uncertainty
            val_res = evaluate_model(val_loader, self.model, uncertainty_enabled, self.device)

            # 使用优化版增强评估器计算更多生物相关指标
            val_metrics, _ = enhanced_compute_metrics_fast(val_res)

            return val_metrics

    def _validate_with_loss_components(self, val_loader) -> Dict[str, float]:
        """验证模型性能并计算损失分量"""

        # 初始化损失累积器
        loss_accumulators = {
            'total_loss': [],
            'nll_loss': [],
            'adjusted_nll': [],
            'full_mse': [],
            'top20_mse': [],
            'correlation': []
        }

        # 用于计算验证指标的预测和目标
        all_preds = []
        all_targets = []
        all_preds_de = []
        all_targets_de = []
        all_perturbations = []

        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                data = data.to(self.device)

                # 使用新的验证损失计算功能
                model_output = self.model(data, return_loss_components=True)

                # 处理ModelOutput格式
                if hasattr(model_output, 'pred') and hasattr(model_output, 'loss_components'):
                    pred = model_output.pred
                    loss_components = model_output.loss_components
                # 处理元组格式（向后兼容）
                elif isinstance(model_output, tuple) and len(model_output) == 2:
                    pred, loss_components = model_output
                else:
                    # 异常情况处理
                    pred = model_output if hasattr(model_output, 'pred') else model_output
                    loss_components = None
                target = data.y.reshape(pred.shape[0], -1)

                # 累积损失分量
                if loss_components is not None:
                    for key in loss_accumulators.keys():
                        if key in loss_components:
                            if torch.is_tensor(loss_components[key]):
                                loss_accumulators[key].append(loss_components[key].item())
                            else:
                                loss_accumulators[key].append(loss_components[key])

                # 收集预测和目标用于计算验证指标
                all_preds.append(pred.cpu())
                all_targets.append(target.cpu())

                # 收集差异表达基因的预测和目标
                if hasattr(data, 'de_idx'):
                    for itr, de_idx in enumerate(data.de_idx):
                        all_preds_de.append(pred[itr, de_idx].cpu())
                        all_targets_de.append(target[itr, de_idx].cpu())
                else:
                    # 如果没有de_idx，使用所有基因（作为fallback）
                    for itr in range(pred.shape[0]):
                        all_preds_de.append(pred[itr].cpu())
                        all_targets_de.append(target[itr].cpu())

                # 收集扰动类别
                if hasattr(data, 'pert'):
                    all_perturbations.extend(data.pert)
                else:
                    # 如果没有扰动信息，标记为unknown
                    all_perturbations.extend(['unknown'] * pred.shape[0])

        # 合并所有预测
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_preds_de = torch.stack(all_preds_de)
        all_targets_de = torch.stack(all_targets_de)

        # 计算平均损失分量
        val_loss_components = {}
        for key, values in loss_accumulators.items():
            if values:
                val_loss_components[f'val_{key}'] = np.mean(values)
            else:
                val_loss_components[f'val_{key}'] = 0.0

        # 计算验证指标
        # 构建完整的结果字典，匹配评估器期望的格式
        val_res = {
            'predictions': all_preds.numpy(),
            'truths': all_targets.numpy(),
            'predictions_de': all_preds_de.numpy(),
            'truths_de': all_targets_de.numpy(),
            'perturbation_categories': np.array(all_perturbations)
        }
        val_metrics, _ = enhanced_compute_metrics_fast(val_res)

        # 合并损失分量和验证指标
        val_results = {**val_loss_components, **val_metrics}

        return val_results
    
    def train(self, train_loader, val_loader, test_loader=None, 
              ctrl_expression=None, dict_filter=None, adata=None, split=None, subgroup=None) -> None:
        """
        完整训练流程
        
        参数:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器（可选）
            ctrl_expression: 对照组表达均值
            dict_filter: 非零基因索引字典
            adata: AnnData对象，用于深入分析
            split: 分割类型，用于子组分析
            subgroup: 子组信息，用于深入分析中的子组分析
        """
        # 保存对照组表达和非零基因索引
        self.ctrl_expression = ctrl_expression
        self.dict_filter = dict_filter
        self.adata = adata
        self.split = split
        self.subgroup = subgroup
                
        # 设置优化器和学习率调度器
        optimizer, scheduler = self.setup_optimizer()
        min_val = float('inf')
        
        # 将模型移动到设备
        self.model = self.model.to(self.device)
        best_model = deepcopy(self.model)

        # 将 EMA 参数移动到与模型相同的设备
        if self.use_ema and len(self.ema_params) > 0:
            self.ema_params = [p.to(self.device) for p in self.ema_params]

        # 训练循环
        for epoch in range(self.config.epochs):
            # 训练阶段
            train_loss = self.train_epoch(train_loader, optimizer, epoch)
            
            # 验证阶段 - ✅ 添加损失分量计算
            compute_val_losses = (epoch % 5 == 0) or (epoch < 10)  # 早期或每5个epoch计算详细损失
            val_metrics = self.validate(val_loader, compute_loss_components=compute_val_losses)
            
            # 学习率调整
            scheduler.step()
            
            # 保存最佳模型
            if val_metrics['mse_de'] < min_val:
                min_val = val_metrics['mse_de']
                best_model = deepcopy(self.model)
            
            # 记录训练历史
            self.train_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_metrics': val_metrics
            })
            
            # 打印评估结果 - 修正：train_loss是总训练损失，不是MSE
            log = "Epoch {}: Train Total Loss: {:.4f} Validation Overall MSE: {:.4f}"
            
            log = "Validation Top 20 DE MSE: {:.4f}"
            
            # 记录到WandB - 包含验证损失曲线
            if self.wandb:
                # 基础日志记录
                log_dict = {
                    'train_total_loss': train_loss,
                    'val_mse': val_metrics['mse'],
                    'val_de_mse': val_metrics['mse_de'],
                    'val_pearson': val_metrics['pearson'],
                    'val_de_pearson': val_metrics['pearson_de']
                }

                # 添加验证损失分量（如果存在）
                val_loss_keys = [k for k in val_metrics.keys() if k.startswith('val_') and k.endswith('_loss')]
                for key in val_loss_keys:
                    log_dict[key] = val_metrics[key]

                self.wandb.log(log_dict)
        
        # 保存最佳模型的状态字典而不是整个对象，避免内存泄漏
        self.best_model_state = best_model.state_dict()
        self.best_model_config = self.model_config
        self.best_val_score = min_val

        # 清理内存
        del best_model
        import gc
        gc.collect()
        
        # 测试阶段（如果有测试集）
        if test_loader is not None:
            self.test(test_loader, adata=self.adata, split=self.split, subgroup=self.subgroup)
        

    def load_best_model(self):
        """加载最佳模型（从状态字典恢复）"""
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        else:
            print_system("警告：未找到最佳模型状态")

    def test(self, test_loader, adata=None, split=None, subgroup=None) -> Dict[str, float]:
        """测试模型性能"""

        # 使用新的最佳模型加载方法
        self.load_best_model()
        
        # 评估测试集
        uncertainty_enabled = self.model_config is not None and self.model_config.uncertainty
        test_res = evaluate_model(test_loader, self.model, uncertainty_enabled, self.device)
        test_metrics, test_perturbation_res = compute_metrics(test_res)
        
        # 打印测试结果
        log = "最佳模型: Test Top 20 DE MSE: {:.4f}"
        print_system(log.format(test_metrics['mse_de']))
        
        # 记录到WandB
        if self.wandb:
            metrics = ['mse', 'pearson']
            for m in metrics:
                self.wandb.log({
                    'test_' + m: test_metrics[m],
                    'test_de_' + m: test_metrics[m + '_de']
                })
        
        # 深入分析
        # 使用提供的adata参数或者内部保存的adata对象
        actual_adata = adata if adata is not None else (self.adata if self.adata is not None else {})
        actual_split = split if split is not None else (self.split if self.split is not None else None)
        actual_subgroup = subgroup if subgroup is not None else (getattr(self, 'subgroup', None))
        
        out = deeper_analysis(actual_adata, test_res)
        out_non_dropout = non_dropout_analysis(actual_adata, test_res)
        
        # 记录深入分析结果到WandB
        if self.wandb:
            metrics = ['pearson_delta']
            metrics_non_dropout = [
                'frac_opposite_direction_top20_non_dropout',
                'frac_sigma_below_1_non_dropout',
                'mse_top20_de_non_dropout'
            ]
            
            for m in metrics:
                self.wandb.log({
                    'test_' + m: np.mean([j[m] for i, j in out.items() if m in j])
                })
            
            for m in metrics_non_dropout:
                self.wandb.log({
                    'test_' + m: np.mean([j[m] for i, j in out_non_dropout.items() if m in j])
                })
        
        # 子组分析（如果是仿真分割）
        if actual_split == 'simulation' and actual_subgroup is not None:
            subgroup = actual_subgroup
            subgroup_analysis = {}
            
            # 初始化结果字典
            for name in subgroup['test_subgroup'].keys():
                subgroup_analysis[name] = {}
                for m in list(list(test_perturbation_res.values())[0].keys()):
                    subgroup_analysis[name][m] = []
            
            # 收集子组结果
            for name, perturbation_list in subgroup['test_subgroup'].items():
                for perturbation in perturbation_list:
                    for m, res in test_perturbation_res[perturbation].items():
                        subgroup_analysis[name][m].append(res)
            
            # 计算并打印子组平均值
            for name, result in subgroup_analysis.items():
                for m in result.keys():
                    subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
                    if self.wandb:
                        self.wandb.log({
                            'test_' + name + '_' + m: subgroup_analysis[name][m]
                        })

            # 更深层次的分析
            subgroup_analysis = {}
            metrics = ['pearson_delta']
            metrics_non_dropout = [
                'frac_opposite_direction_top20_non_dropout',
                'frac_sigma_below_1_non_dropout',
                'mse_top20_de_non_dropout'
            ]
            
            # 初始化结果字典
            for name in subgroup['test_subgroup'].keys():
                subgroup_analysis[name] = {}
                for m in metrics:
                    subgroup_analysis[name][m] = []
                for m in metrics_non_dropout:
                    subgroup_analysis[name][m] = []
            
            # 收集子组结果
            for name, perturbation_list in subgroup['test_subgroup'].items():
                for perturbation in perturbation_list:
                    for m in metrics:
                        subgroup_analysis[name][m].append(out[perturbation][m])
                    for m in metrics_non_dropout:
                        subgroup_analysis[name][m].append(out_non_dropout[perturbation][m])
            
            # 计算并打印子组平均值
            for name, result in subgroup_analysis.items():
                for m in result.keys():
                    subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
                    if self.wandb:
                        self.wandb.log({
                            'test_' + name + '_' + m: subgroup_analysis[name][m]
                        })

        return test_metrics
    
    def save_model(self, path, data_name=None):
        """
        保存模型
        
        参数:
            path: 保存路径
            data_name: 数据集名称
        """
        import os
        import pickle
        
        # 创建保存目录
        save_dir = os.path.join(path, f"model_{data_name}") if data_name else path
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存配置
        with open(os.path.join(save_dir, 'config.pkl'), 'wb') as f:
            pickle.dump(self.config, f)
        
        # 保存模型状态
        # 使用 best_model_state 而不是 best_model.state_dict()
        if hasattr(self, 'best_model_state') and self.best_model_state is not None:
            torch.save(self.best_model_state, os.path.join(save_dir, 'model.pt'))
        else:
            # 如果没有 best_model_state，使用当前模型
            torch.save(self.model.state_dict(), os.path.join(save_dir, 'model.pt'))
        
    
    def load_model(self, path, data_name=None):
        """
        加载模型
        
        参数:
            path: 加载路径
            data_name: 数据集名称
        """
        import os
        import pickle
        
        # 构建模型路径
        model_dir = os.path.join(path, f"model_{data_name}") if data_name else path
        
        # 加载配置
        with open(os.path.join(model_dir, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        
        # 加载模型状态
        state_dict = torch.load(os.path.join(model_dir, 'model.pt'), map_location=torch.device('cpu'))
        
        # 处理数据并行的情况
        if next(iter(state_dict))[:7] == 'module.':
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        # 加载模型状态
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.best_model = self.model

        # 将 EMA 参数移动到与模型相同的设备（如果使用 EMA）
        if self.use_ema and len(self.ema_params) > 0:
            self.ema_params = [p.to(self.device) for p in self.ema_params]

