"""
CausalKAN-Flow 网络工具模块
提供基因相似性网络相关的功能
新增：支持因果约束的网络构建
"""

import pandas as pd
import networkx as nx
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from .general_utils import print_system
from .causal_network_utils import get_causal_coexpression_network

def calculate_pearson_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """计算Pearson相关系数矩阵"""
    
    xv = x - x.mean(axis=0)
    yv = y - y.mean(axis=0)
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)
    
    result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))
    result = np.maximum(np.minimum(result, 1.0), -1.0)
    
    return result

class GeneSimilarityNetwork:
    """基因相似性网络类"""
    
    def __init__(self, edge_list: pd.DataFrame, gene_list: List[str], node_map: Dict[str, int]):
        """初始化基因相似性网络"""
        self.edge_list = edge_list
        
        self.graph = nx.from_pandas_edgelist(
            self.edge_list, 
            source='source',
            target='target', 
            edge_attr=['importance'],
            create_using=nx.DiGraph()
        )
        
        self.gene_list = gene_list
        
        for n in self.gene_list:
            if n not in self.graph.nodes():
                self.graph.add_node(n)
        
        edge_index_ = [
            (node_map[e[0]], node_map[e[1]]) for e in self.graph.edges
        ]
        self.edge_index = torch.tensor(edge_index_, dtype=torch.long).T
        
        edge_attr = nx.get_edge_attributes(self.graph, 'importance')
        importance = np.array([edge_attr[e] for e in self.graph.edges])
        self.edge_weight = torch.Tensor(importance)
        
def get_similarity_network(
    network_type: str,
    adata: Any,
    threshold: float,
    k: int,
    data_path: str,
    data_name: str,
    split: str,
    seed: int,
    train_gene_set_size: float,
    set2conditions: Dict[str, List[str]],
    default_perturbation_graph: bool = True,
    perturbation_list: Optional[List[str]] = None,
    use_causal_constraint: bool = False,
    causal_alpha: float = 0.05,
    causal_n_jobs: int = 4
) -> pd.DataFrame:
    """获取相似性网络"""
    
    if network_type == 'co-express':
        df_out = get_coexpression_network_from_train(
            adata, threshold, k, data_path, data_name, split,
            seed, train_gene_set_size, set2conditions,
            use_causal_constraint=use_causal_constraint,
            causal_alpha=causal_alpha,
            causal_n_jobs=causal_n_jobs
        )
    
    return df_out

def get_coexpression_network_from_train(
    adata: Any,
    threshold: float,
    k: int,
    data_path: str,
    data_name: str,
    split: str,
    seed: int,
    train_gene_set_size: float,
    set2conditions: Dict[str, List[str]],
    use_causal_constraint: bool = False,  # 新增参数
    causal_alpha: float = 0.05,           # 新增参数
    causal_n_jobs: int = 4                # 新增参数
) -> pd.DataFrame:
    """
    从训练数据推断共表达网络
    新增：支持因果约束精炼 + 全连通因果约束模式
    """
    import os
    
    fname = os.path.join(
        os.path.join(data_path, data_name), 
        split + '_' + str(seed) + '_' + str(train_gene_set_size) + '_' +
        str(threshold) + '_' + str(k) + '_co_expression_network.csv'
    )
    
    if os.path.exists(fname):
        return pd.read_csv(fname)
    else:        
        gene_list = [f for f in adata.var.gene_name.values]
        idx2gene = dict(zip(range(len(gene_list)), gene_list))
        
        X = adata.X
        train_perturbations = set2conditions['train']
        
        # 构建基因表达数据 - 修复：包含对照和组合扰动数据（方案1）
        ctrl_and_combo = ['ctrl'] + [j for j in train_perturbations if 'ctrl+' in j or '+ctrl' in j]
        X_tr = X[np.isin(adata.obs.condition, ctrl_and_combo)]

        gene_list = adata.var['gene_name'].values
        
        # 计算皮尔逊相关系数
        X_tr = X_tr.toarray()
        out = calculate_pearson_correlation(X_tr, X_tr)
        out[np.isnan(out)] = 0
        out = np.abs(out)
        out_sort_idx = np.argsort(out)[:, -(k + 1):]
        out_sort_val = np.sort(out)[:, -(k + 1):]

        df_g = []
        for i in range(out_sort_idx.shape[0]):
            target = idx2gene[i]
            for j in range(out_sort_idx.shape[1]):
                df_g.append((idx2gene[out_sort_idx[i, j]], target, out_sort_val[i, j]))

        df_g = [i for i in df_g if i[2] > threshold]

        # 如果启用因果约束，应用因果精炼
        if use_causal_constraint:
            # 普通因果约束模式（非全连通，会排除自环）
            try:
                # 检查样本量和特征数
                n_samples, n_genes = X_tr.shape

                # 应用因果精炼
                causal_network = get_causal_coexpression_network(
                    data_mat=X_tr,
                    gene_list=gene_list,
                    initial_threshold=threshold,
                    k_neighbors=k,
                    alpha=causal_alpha,
                    n_jobs=causal_n_jobs
                )

                # 如果因果精炼成功，使用精炼后的结果
                if len(causal_network) > 0:
                    df_co_expression = causal_network
                else:
                    df_co_expression = pd.DataFrame(df_g).rename(columns={
                        0: 'source', 1: 'target', 2: 'importance'
                    })

            except Exception as e:
                # 如果因果精炼失败，使用原始方法
                df_co_expression = pd.DataFrame(df_g).rename(columns={
                    0: 'source', 1: 'target', 2: 'importance'
                })
        else:
            # 原始方法（无因果约束）
            df_co_expression = pd.DataFrame(df_g).rename(columns={
                0: 'source', 1: 'target', 2: 'importance'
            })

        df_co_expression.to_csv(fname, index=False)
        return df_co_expression