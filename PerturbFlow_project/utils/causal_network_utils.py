"""
因果约束的基因网络构建模块
基于PC算法的条件独立性检验改进共表达网络
"""

import numpy as np
import pandas as pd
import networkx as nx
import torch
from typing import Dict, List, Any, Optional, Tuple
from itertools import combinations
from scipy.stats import pearsonr
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")

try:
    from causallearn.utils.cit import CIT
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CAUSAL_LEARN_AVAILABLE = False
    warnings.warn("causallearn 库未安装，将使用简化的条件独立性检验")

from .general_utils import print_system

def edge_key(u: str, v: str) -> tuple:
    """规范化边表示，确保 (u,v) 和 (v,u) 被视为相同边"""
    return (min(u, v), max(u, v))

# 定义本地的皮尔逊相关系数计算函数，避免循环导入
def _calculate_pearson_correlation_local(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """本地版本的皮尔逊相关系数计算"""

    xv = x - x.mean(axis=0)
    yv = y - y.mean(axis=0)
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)

    # Handle potential division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))
        result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)

    return np.clip(result, -1.0, 1.0)

# 使用本地函数替代，避免循环导入
def calculate_pearson_correlation_local(data_mat):
    """本地的皮尔逊相关系数计算函数"""
    return _calculate_pearson_correlation_local(data_mat, data_mat)


class CausalConstraintNetworkBuilder:
    """因果约束网络构建器"""

    def __init__(self, alpha: float = 0.05, max_cond_size: int = 2,
                 n_jobs: int = 4, use_fdr: bool = True):
        """
        初始化因果约束网络构建器

        Args:
            alpha: 显著性水平
            max_cond_size: 最大条件集大小
            n_jobs: 并行进程数
            use_fdr: 是否使用FDR校正
        """
        self.alpha = alpha
        self.max_cond_size = max_cond_size
        self.n_jobs = n_jobs
        self.use_fdr = use_fdr
        self.cit_method = 'fisherz' if CAUSAL_LEARN_AVAILABLE else 'pearson_partial'

    def build_causal_coexpression_network(
        self,
        data_mat: np.ndarray,
        gene_list: List[str],
        initial_threshold: float = 0.3,
        k_neighbors: int = 20,
        min_correlation: float = 0.1
    ) -> pd.DataFrame:
        """
        构建因果约束的共表达网络

        Args:
            data_mat: 基因表达矩阵 (n_samples, n_genes)
            gene_list: 基因名称列表
            initial_threshold: 初选相关性阈值
            k_neighbors: 每个基因考虑的最大邻居数
            min_correlation: 最小相关性阈值

        Returns:
            因果精炼后的边列表DataFrame
        """
        # 步骤1: 构建初步共表达网络
        initial_edges = self._build_initial_coexpression_network(
            data_mat, gene_list, initial_threshold, k_neighbors
        )

        # 步骤2: 因果精炼 - 应用PC算法思想
        causal_edges = self._apply_causal_constraints(
            data_mat, gene_list, initial_edges
        )

        # 步骤3: 构建最终网络
        final_network = self._build_final_network(causal_edges, gene_list)
        return final_network

    def _build_initial_coexpression_network(
        self,
        data_mat: np.ndarray,
        gene_list: List[str],
        threshold: float,
        k_neighbors: int
    ) -> List[Tuple[str, str, float]]:
        """构建初步共表达网络"""
        n_genes = data_mat.shape[1]

        # 计算相关性矩阵
        corr_matrix = calculate_pearson_correlation_local(data_mat)
        corr_matrix = np.abs(corr_matrix)  # 取绝对值

        # 处理可能的NaN值
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # 为每个基因选择top-k相关性最高的邻居
        initial_edges = []
        neighbors_dict = {}

        for i in range(n_genes):
            gene_i = gene_list[i]

            # 获取该基因与其他基因的相关性
            correlations = corr_matrix[i, :]

            # 排除自身，并设置负无穷以避免自环
            correlations[i] = -np.inf

            # 选择top-k相关性
            top_k_indices = np.argpartition(correlations, -k_neighbors)[-k_neighbors:]
            top_k_indices = top_k_indices[correlations[top_k_indices] > threshold]

            # 添加到邻居集合
            neighbors_dict[gene_i] = set()
            for j in top_k_indices:
                if i != j and correlations[j] > threshold:  # 确保不创建自环
                    gene_j = gene_list[j]
                    weight = float(correlations[j])
                    edge = (gene_i, gene_j, weight)

                    if edge_key(gene_i, gene_j) not in [(edge_key(e[0], e[1])) for e in initial_edges]:
                        initial_edges.append(edge)
                        neighbors_dict[gene_i].add(gene_j)

        self.neighbors_dict = neighbors_dict  # 保存用于后续条件独立性检验
        return initial_edges

    def _apply_causal_constraints(
        self,
        data_mat: np.ndarray,
        gene_list: List[str],
        initial_edges: List[Tuple[str, str, float]]
    ) -> List[Tuple[str, str, float]]:
        """应用因果约束精炼网络"""

        # 构建基因到索引的映射
        gene_to_idx = {gene: i for i, gene in enumerate(gene_list)}
        edge_weights = {}
        for u, v, w in initial_edges:
            edge_weights[edge_key(u, v)] = w


        # 分批处理以避免内存问题
        batch_size = min(100, len(initial_edges) // self.n_jobs + 1)
        causal_edges = []

        for i in range(0, len(initial_edges), batch_size):
            batch_edges = initial_edges[i:i+batch_size]

            with Pool(processes=self.n_jobs) as pool:
                test_results = pool.starmap(
                    self._test_edge_with_conditions,
                    [(edge, data_mat, gene_to_idx, self.alpha, self.neighbors_dict, edge_weights)
                     for edge in batch_edges]
                )

            # 收集保留的边
            for result in test_results:
                if result is not None and result[3]:  # 保留的边
                    u, v, w, _ = result
                    causal_edges.append((u, v, w))

        return causal_edges

    def _test_edge_with_conditions(
        self,
        edge: Tuple[str, str, float],
        data_mat: np.ndarray,
        gene_to_idx: Dict[str, int],
        alpha: float,
        neighbors_dict: Dict[str, set],
        edge_weights: Dict[Tuple[str, str], float]
    ) -> Optional[Tuple[str, str, float, bool]]:
        """对单条边进行条件独立性检验"""
        u, v, w = edge
        u_idx = gene_to_idx[u]
        v_idx = gene_to_idx[v]

        # 获取共同邻居作为条件集
        common_neighbors = neighbors_dict[u] & neighbors_dict[v]

        # 如果没有共同邻居，保留边（无法检验独立性）
        if not common_neighbors:
            return (u, v, w, True)

        # 排序共同邻居，优先选择权重高的
        condition_candidates = []
        for neighbor in common_neighbors:
            # 计算与u和v的平均相关性作为重要性指标
            edge_u_n = edge_weights.get(edge_key(u, neighbor), 0)
            edge_v_n = edge_weights.get(edge_key(v, neighbor), 0)
            importance = (edge_u_n + edge_v_n) / 2
            condition_candidates.append((neighbor, importance))

        # 按重要性排序，选择前max_cond_size个
        condition_candidates.sort(key=lambda x: x[1], reverse=True)
        selected_conditions = [c[0] for c in condition_candidates[:self.max_cond_size]]

        # 条件独立性检验
        is_independent = False

        # 使用简化的偏相关系数检验
        if self.cit_method == 'pearson_partial' or not CAUSAL_LEARN_AVAILABLE:
            is_independent = self._partial_correlation_test(
                data_mat, u_idx, v_idx, [gene_to_idx[c] for c in selected_conditions], alpha
            )
        else:
            # 使用FisherZ检验（如果causallearn可用）
            try:
                cit = CIT(data_mat, method='fisherz')
                # 对每种条件大小的组合进行检验
                p_values = []
                for cond_size in range(1, min(self.max_cond_size + 1, len(selected_conditions) + 1)):
                    for cond_subset in combinations(selected_conditions, cond_size):
                        cond_idx = [gene_to_idx[c] for c in cond_subset]
                        try:
                            p = cit(u_idx, v_idx, cond_idx)
                            p_values.append(p)
                        except:
                            continue

                # 如果任何一个检验显示独立，则删除边
                if p_values and any(p > alpha for p in p_values):
                    is_independent = True

            except Exception as e:
                # 如果FisherZ检验失败，回退到偏相关
                is_independent = self._partial_correlation_test(
                    data_mat, u_idx, v_idx, [gene_to_idx[c] for c in selected_conditions], alpha
                )

        # 如果条件独立，删除边；否则保留
        if is_independent:
            return None  # 删除边
        else:
            return (u, v, w, True)  # 保留边

    def _partial_correlation_test(
        self,
        data_mat: np.ndarray,
        u_idx: int,
        v_idx: int,
        cond_idx: List[int],
        alpha: float
    ) -> bool:
        """
        使用偏相关系数进行条件独立性检验
        """
        try:
            if not cond_idx:
                # 无条件时退化为简单的相关性检验
                correlation, _ = pearsonr(data_mat[:, u_idx], data_mat[:, v_idx])
                # 粗略的显著性判断（简化版）
                n = data_mat.shape[0]
                t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2 + 1e-10))
                p_value = 2 * (1 - np.abs(t_stat) / np.sqrt(n - 2 + t_stat**2))
                return p_value > alpha

            # 计算偏相关系数
            if len(cond_idx) == 1:
                # 单条件变量时的偏相关
                r_uv = np.corrcoef(data_mat[:, u_idx], data_mat[:, v_idx])[0, 1]
                r_uc = np.corrcoef(data_mat[:, u_idx], data_mat[:, cond_idx[0]])[0, 1]
                r_vc = np.corrcoef(data_mat[:, v_idx], data_mat[:, cond_idx[0]])[0, 1]

                # 偏相关系数公式
                r_uv_c = (r_uv - r_uc * r_vc) / (np.sqrt(1 - r_uc**2) * np.sqrt(1 - r_vc**2) + 1e-10)
            else:
                # 多条件变量，使用矩阵方法
                variables = [u_idx, v_idx] + cond_idx
                cov_matrix = np.cov(data_mat[:, variables].T)

                # 计算偏相关系数矩阵
                p = len(variables)
                inv_cov = np.linalg.inv(cov_matrix + np.eye(p) * 1e-10)
                r_uv_c = -inv_cov[0, 1] / np.sqrt(inv_cov[0, 0] * inv_cov[1, 1] + 1e-10)

            # 粗略的显著性检验（简化版）
            n = data_mat.shape[0]
            df = n - len(cond_idx) - 2  # 自由度
            if df <= 0:
                return False

            t_stat = r_uv_c * np.sqrt(df / (1 - r_uv_c**2 + 1e-10))
            p_value = 2 * (1 - np.abs(t_stat) / np.sqrt(df + t_stat**2))

            return p_value > alpha

        except Exception:
            # 如果计算失败，保守地保留边
            return False

    def _build_final_network(self, causal_edges: List[Tuple[str, str, float]],
                           gene_list: List[str]) -> pd.DataFrame:
        """构建最终网络"""
        if not causal_edges:
            return pd.DataFrame(columns=['source', 'target', 'importance'])

        # 构建DataFrame
        df = pd.DataFrame(causal_edges, columns=['source', 'target', 'importance'])

        # 规范化边顺序
        df['source'] = df['source'].astype(str)
        df['target'] = df['target'].astype(str)

        # 确保source < target
        mask = df['source'] > df['target']
        df.loc[mask, ['source', 'target']] = df.loc[mask, ['target', 'source']].values

        # 去重并排序
        df = df.drop_duplicates(subset=['source', 'target']).sort_values(['source', 'target'])

        return df


def get_causal_coexpression_network(
    data_mat: np.ndarray,
    gene_list: List[str],
    initial_threshold: float = 0.3,
    k_neighbors: int = 20,
    alpha: float = 0.05,
    max_cond_size: int = 2,
    n_jobs: int = 4,
    preserve_self_loops: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    快速构建因果约束共表达网络的函数接口 - 智能自环处理版本

    特点：从Pearson相关网络开始，只对非自环边进行PC算法精炼，最后加回自环边

    Args:
        data_mat: 基因表达矩阵
        gene_list: 基因名列表
        initial_threshold: 初选相关性阈值
        k_neighbors: 每个基因的邻居数
        alpha: 显著性水平
        max_cond_size: 最大条件集大小
        n_jobs: 并行进程数
        preserve_self_loops: 是否保留自环边

    Returns:
        包含精炼非自环边 + 保留自环边的完整边列表
    """
    # 第一步：基于所有基因生成初步的Pearson相关网络（包括自环）
    # 计算完整的皮尔逊相关性矩阵（包括自环）
    corr_matrix = calculate_pearson_correlation_local(data_mat)
    corr_matrix = np.abs(corr_matrix)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    n_genes = data_mat.shape[1]
    all_edges = []
    initial_interaction_edges = []  # 非自环边

    for i in range(n_genes):
        gene_i = gene_list[i]
        correlations = corr_matrix[i, :]

        # 选择top-k相关性
        top_k_indices = np.argpartition(correlations, -(k_neighbors + 1))[-(k_neighbors + 1):]
        top_k_indices = top_k_indices[correlations[top_k_indices] > initial_threshold]

        for j in top_k_indices:
            gene_j = gene_list[j]
            weight = float(correlations[j])

            if i == j and preserve_self_loops:  # 自环边处理
                # 为自环边分配一个合理的权重（高权重）
                weight = max(weight, 0.95)  # 确保自环边有充足权重
                edge = (gene_i, gene_j, weight)
                all_edges.append(edge)
            elif i != j:  # 非自环边
                edge = (gene_i, gene_j, weight)
                if edge_key(gene_i, gene_j) not in [(edge_key(e[0], e[1])) for e in initial_interaction_edges]:
                    initial_interaction_edges.append(edge)

    # 第二步：只对非自环边应用因果约束精炼
    refined_interaction_edges = []
    if initial_interaction_edges:
        # 构建邻居字典（仅用于非自环边的精炼）
        neighbors_dict = {}
        for u, v, w in initial_interaction_edges:
            if u not in neighbors_dict:
                neighbors_dict[u] = set()
            neighbors_dict[u].add(v)
            if v not in neighbors_dict:
                neighbors_dict[v] = set()
            neighbors_dict[v].add(u)

        # 基因到索引映射
        gene_to_idx = {gene: i for i, gene in enumerate(gene_list)}
        edge_weights = {(edge_key(u, v)): w for u, v, w in initial_interaction_edges}

        # 使用我们之前定义的精炼逻辑
        builder = CausalConstraintNetworkBuilder(alpha=alpha, max_cond_size=max_cond_size, n_jobs=n_jobs)
        builder.neighbors_dict = neighbors_dict  # 手动设置邻居字典

        refined_interaction_edges = builder._apply_causal_constraints(
            data_mat, gene_list, initial_interaction_edges
        )

    else:
        refined_interaction_edges = initial_interaction_edges

    # 第三步：合并精炼后的非自环边与自环边
    final_edges = refined_interaction_edges + all_edges

    # 第四步：构建最终的DataFrame
    return builder._build_final_network(final_edges, gene_list)