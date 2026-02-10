"""
CausalKAN-Flow 推理模块
提供模型预测和结果可视化功能
"""

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader
from typing import Dict, List, Optional, Tuple

from utils.data_utils import  create_cell_graph_dataset_for_prediction
from utils.general_utils import print_system


class PerturbFlowInference:
    """PerturbFlow 推理器 - 处理模型预测和结果可视化"""
    
    def __init__(self, model, config, data_loader):
        """
        初始化推理器
        
        参数:
            model: 训练好的模型
            config: 配置字典
            data_loader: 数据加载器实例
        """
        self.model = model
        self.config = config
        self.data_loader = data_loader
        self.saved_predictions = {}
        self.saved_logvar_sum = {}
        
        # 设置设备
        self.device = config.get('device', 'cuda')
        self.model = self.model.to(self.device)
        self.model.eval()
        
    
    def predict(self, perturbation_list: List[List[str]]) -> Dict[str, np.ndarray]:
        """
        预测给定基因/基因组合扰动后的转录组
        
        参数:
            perturbation_list: 要扰动的基因/基因组合列表，如 [['BRCA1'], ['TP53'], ['BRCA1','TP53']]
                
        返回:
            results_pred: 预测的转录组字典，键为扰动名称，值为预测表达值
            results_logvar_sum: 不确定性分数字典(当uncertainty模式开启时返回)
        """
        
        # 验证扰动基因是否在已知列表中
        for perturbation in perturbation_list:
            for gene in perturbation:
                if gene not in self.data_loader.pert_names:
                    error_msg = f"基因 '{gene}' 不在扰动图中，请从 {self.data_loader.pert_names[:5]}... 中选择"
                    print_system(error_msg)
                    raise ValueError(error_msg)
                
        # 准备结果字典
        if self.config['uncertainty']:
            results_logvar = {}
            
        results_pred = {}
        results_logvar_sum = {}
        
        # 遍历预测扰动列表里的每种扰动组合
        for perturbation in perturbation_list:
            perturbation_name = '_'.join(perturbation)
            
            # 检查是否已有缓存结果
            try:
                results_pred[perturbation_name] = self.saved_predictions[perturbation_name]
                if self.config['uncertainty']:
                    results_logvar_sum[perturbation_name] = self.saved_logvar_sum[perturbation_name]
                continue
            except:
                pass
            
            # 为每一组扰动类别创建细胞图数据集
            cell_graphs = create_cell_graph_dataset_for_prediction(
                perturbation,
                self.data_loader.ctrl_adata,
                self.data_loader.pert_names,
                self.device
            )

            # 将 perturbation_idx 重命名为 pert_idx（模型期望的属性名）
            for graph in cell_graphs:
                if hasattr(graph, 'perturbation_idx'):
                    graph.pert_idx = graph.perturbation_idx
                    delattr(graph, 'perturbation_idx')

            # 从数据集中获取该扰动条件的真实表达值
            query_condition = '+'.join(perturbation) if len(perturbation) == 2 else perturbation[0] + '+ctrl'
            adata = self.data_loader.adata
            if query_condition in adata.obs['condition'].unique():
                # 随机采样真实表达值作为 y
                real_cells = adata[adata.obs['condition'] == query_condition].X.toarray()
                sampled_real = real_cells[np.random.choice(len(real_cells), len(cell_graphs), replace=True)]
                # 将真实值添加到每个细胞图的 y 属性
                for i, graph in enumerate(cell_graphs):
                    graph.y = torch.tensor(sampled_real[i], dtype=torch.float32)
            
            # 创建数据加载器
            loader = DataLoader(cell_graphs, 300, shuffle=False)
            batch = next(iter(loader))
            batch.to(self.device)
                        
            # 进行预测
            with torch.no_grad():
                if self.config['uncertainty']:
                    output = self.model(batch)
                    pred = output.pred  # 从 ModelOutput 中提取 pred
                    unc = output.uncertainty  # 从 ModelOutput 中提取 uncertainty
                    results_logvar[perturbation_name] = np.mean(unc.detach().cpu().numpy(), axis=0)
                    results_logvar_sum[perturbation_name] = np.exp(-np.mean(results_logvar[perturbation_name]))
                else:
                    output = self.model(batch)
                    pred = output.pred  # 从 ModelOutput 中提取 pred

            # 保存预测结果
            results_pred[perturbation_name] = np.mean(pred.detach().cpu().numpy(), axis=0)
        
        # 更新缓存
        self.saved_predictions.update(results_pred)
        
        if self.config['uncertainty']:
            self.saved_logvar_sum.update(results_logvar_sum)
            return results_pred, results_logvar_sum
        else:
            return results_pred
    
    def plot_perturbation(self, query: str, save_file: Optional[str] = None,
                        save_data_file: Optional[str] = None) -> None:
        """
        可视化扰动效果

        参数:
            query: 查询条件，如 "BRCA1+TP53"
            save_file: 保存图像的文件路径(可选)
            save_data_file: 保存画图数据的文件路径(可选，.pkl格式)
        """
        # 设置绘图样式
        sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=1.5)

        # 准备数据
        adata = self.data_loader.adata
        gene_to_idx = self.data_loader.node_map
        condition_to_name = dict(adata.obs[['condition', 'condition_name']].values)
        gene_raw_to_id = dict(zip(adata.var.index.values, adata.var.gene_name.values))

        # 获取差异表达基因
        de_idx = [gene_to_idx[gene_raw_to_id[i]] for i in
                  adata.uns['top_non_dropout_de_20'][condition_to_name[query]]]
        genes = [gene_raw_to_id[i] for i in
                 adata.uns['top_non_dropout_de_20'][condition_to_name[query]]]

        # 获取真实值和预测值
        truth = adata[adata.obs.condition == query].X.toarray()[:, de_idx]
        query_ = [q for q in query.split('+') if q != 'ctrl']
        pred = self.predict([query_])['_'.join(query_)][de_idx]
        ctrl_means = adata[adata.obs['condition'] == 'ctrl'].to_df().mean()[
            de_idx].values

        # 计算相对于对照组的变化
        pred_change = pred - ctrl_means
        truth_change = truth - ctrl_means

        # 保存画图数据（如果需要）
        if save_data_file:
            import pickle
            plot_data = {
                'query': query,
                'genes': genes,
                'de_idx': de_idx,
                'pred_original': pred,           # 原始预测值
                'truth_original': truth,         # 原始真实值
                'ctrl_means': ctrl_means,         # 对照组均值
                'pred_change': pred_change,       # 预测变化量
                'truth_change': truth_change,     # 真实变化量（所有细胞）
            }
            with open(save_data_file, 'wb') as f:
                pickle.dump(plot_data, f)

        # 创建图形
        plt.figure(figsize=[16.5, 4.5])
        plt.title(query)

        # 绘制箱线图(真实值)
        plt.boxplot(truth_change, showfliers=False, medianprops=dict(linewidth=0))

        # 绘制预测点
        for i in range(pred_change.shape[0]):
            plt.scatter(i+1, pred_change[i], color='red')

        # 添加参考线
        plt.axhline(0, linestyle="dashed", color='green')

        # 设置坐标轴
        ax = plt.gca()
        ax.xaxis.set_ticklabels(genes, rotation=90)

        plt.ylabel("Change in Gene Expression over Control", labelpad=10)
        plt.tick_params(axis='x', which='major', pad=5)
        plt.tick_params(axis='y', which='major', pad=5)
        sns.despine()

        # 保存或显示图形
        if save_file:
            plt.savefig(save_file, bbox_inches='tight')

        plt.show()
    
    def predict_perturbation_effect(self, perturbation_genes: List[str], 
                                   num_samples: int = 300) -> Dict[str, np.ndarray]:
        """
        预测扰动效果
        
        参数:
            perturbation_genes: 扰动基因列表
            num_samples: 采样数量
            
        返回:
            预测的基因表达变化
        """
        
        # 获取对照组表达
        ctrl_expression = np.mean(
            self.data_loader.adata[self.data_loader.adata.obs.condition == 'ctrl'].X,
            axis=0
        )
        
        # 预测扰动后表达
        pred_results = self.predict([perturbation_genes])
        pred_expression = pred_results['_'.join(perturbation_genes)]
        
        # 计算变化
        perturbation_effect = pred_expression - ctrl_expression
                
        return {
            'perturbation_effect': perturbation_effect,
            'predicted_expression': pred_expression,
            'control_expression': ctrl_expression
        }
    
    def compare_perturbations(self, perturbation_list: List[List[str]]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        比较多个扰动的效果
        
        参数:
            perturbation_list: 扰动列表
            
        返回:
            包含所有扰动比较结果的字典
        """        
        results = {}
        ctrl_expression = np.mean(
            self.data_loader.adata[self.data_loader.adata.obs.condition == 'ctrl'].X,
            axis=0
        )
        
        # 预测所有扰动
        pred_results = self.predict(perturbation_list)
        
        # 计算每个扰动的效果
        for perturbation_name, pred_expression in pred_results.items():
            perturbation_effect = pred_expression - ctrl_expression
            
            # 获取差异表达基因
            de_genes = self._get_differential_genes(perturbation_effect, top_n=20)
            
            results[perturbation_name] = {
                'perturbation_effect': perturbation_effect,
                'predicted_expression': pred_expression,
                'differential_genes': de_genes,
                'max_effect': np.max(np.abs(perturbation_effect)),
                'mean_effect': np.mean(np.abs(perturbation_effect))
            }
            
        return results
    
    def _get_differential_genes(self, perturbation_effect: np.ndarray, 
                               top_n: int = 20) -> List[Tuple[str, float]]:
        """
        获取差异表达基因
        
        参数:
            perturbation_effect: 扰动效应数组
            top_n: 返回的顶部基因数量
            
        返回:
            差异表达基因列表（基因名称，效应值）
        """
        # 获取效应绝对值最大的基因
        effect_abs = np.abs(perturbation_effect)
        top_indices = np.argsort(effect_abs)[-top_n:][::-1]
        
        # 获取基因名称
        gene_names = self.data_loader.adata.var.gene_name.values
        
        # 构建结果列表
        differential_genes = []
        for idx in top_indices:
            gene_name = gene_names[idx]
            effect_value = perturbation_effect[idx]
            differential_genes.append((gene_name, effect_value))
        
        return differential_genes
    
    def save_predictions(self, file_path: str) -> None:
        """
        保存预测结果到文件
        
        参数:
            file_path: 文件路径
        """
        import pickle
        
        with open(file_path, 'wb') as f:
            pickle.dump({
                'predictions': self.saved_predictions,
                'logvar_sum': self.saved_logvar_sum if self.config['uncertainty'] else None
            }, f)
        
    
    def load_predictions(self, file_path: str) -> None:
        """
        从文件加载预测结果
        
        参数:
            file_path: 文件路径
        """
        import pickle
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        self.saved_predictions = data['predictions']
        if self.config['uncertainty']:
            self.saved_logvar_sum = data['logvar_sum']
        
