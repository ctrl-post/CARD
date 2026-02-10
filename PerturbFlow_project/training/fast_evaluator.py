"""
优化版 CausalKAN-Flow 评估器模块
专门解决扰动特异性生物指标计算的性能问题
"""

import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from typing import Dict, Any, List, Tuple
from utils.general_utils import print_system
from .evaluator import evaluate_model, compute_metrics


def compute_perturbation_specific_metrics_fast(results) -> Dict[str, Dict[str, float]]:
    """
    优化版扰动特异性生物指标计算
    使用向量化操作替代双重嵌套循环

    参数:
        results: 包含预测结果的字典

    返回:
        按扰动分类的生物指标
    """
    perturbation_metrics = {}

    # 获取每个扰动的预测和真实值
    for perturbation in np.unique(results['perturbation_categories']):
        perturbation_idx = np.where(results['perturbation_categories'] == perturbation)[0]

        if len(perturbation_idx) == 0:
            continue

        # 获取该扰动的数据
        pred_pert = results['predictions'][perturbation_idx]
        true_pert = results['truths'][perturbation_idx]

        # 计算该扰动的特异性指标
        pert_metrics = {}

        # 1. 扰动效应大小 (Perturbation Effect Size)
        if len(pred_pert) > 0 and len(true_pert) > 0:
            pert_metrics['effect_size_pred'] = np.mean(pred_pert, axis=0)
            pert_metrics['effect_size_true'] = np.mean(true_pert, axis=0)

            # 使用向量化相关性计算
            if len(pert_metrics['effect_size_true']) > 1:
                pert_metrics['effect_size_correlation'] = np.corrcoef(
                    pert_metrics['effect_size_true'],
                    pert_metrics['effect_size_pred']
                )[0, 1]
            else:
                pert_metrics['effect_size_correlation'] = 0.0
        else:
            pert_metrics['effect_size_correlation'] = 0.0

        # 2. 扰动特异性基因预测准确性
        if perturbation != 'ctrl' and 'predictions_de' in results and 'truths_de' in results:
            pred_de_pert = results['predictions_de'][perturbation_idx]
            true_de_pert = results['truths_de'][perturbation_idx]

            if len(pred_de_pert) > 0:
                # 向量化Spearman相关性计算
                if pred_de_pert.size > 1:
                    pert_metrics['de_spearman'] = spearmanr(
                        true_de_pert.flatten(),
                        pred_de_pert.flatten()
                    )[0]
                else:
                    pert_metrics['de_spearman'] = 0.0
            else:
                pert_metrics['de_spearman'] = 0.0
        else:
            pert_metrics['de_spearman'] = 0.0

        # 3. 扰动一致性 (Perturbation Consistency) - 优化版
        if len(pred_pert) > 1:
            # 使用向量化操作替代双重嵌套循环
            try:
                # 计算所有样本对之间的相关性矩阵
                pred_pert_array = np.array(pred_pert)

                # 使用numpy的corrcoef进行批量相关性计算
                if pred_pert_array.shape[0] > 1 and pred_pert_array.shape[1] > 1:
                    correlation_matrix = np.corrcoef(pred_pert_array)

                    # 提取上三角矩阵的相关性（排除对角线）
                    upper_triangle_indices = np.triu_indices(len(correlation_matrix), k=1)
                    sample_correlations = correlation_matrix[upper_triangle_indices]

                    # 过滤掉NaN值
                    sample_correlations = sample_correlations[~np.isnan(sample_correlations)]

                    if len(sample_correlations) > 0:
                        pert_metrics['consistency'] = np.mean(sample_correlations)
                    else:
                        pert_metrics['consistency'] = 0.0
                else:
                    pert_metrics['consistency'] = 1.0

            except Exception as e:
                # 如果向量化计算失败，使用采样方法
                sample_correlations = []
                max_samples = min(100, len(pred_pert))  # 限制最大采样数量

                if len(pred_pert) > max_samples:
                    # 随机采样
                    indices = np.random.choice(len(pred_pert), max_samples, replace=False)
                    sampled_pred = pred_pert[indices]
                else:
                    sampled_pred = pred_pert

                for i in range(len(sampled_pred)):
                    for j in range(i+1, len(sampled_pred)):
                        try:
                            corr = pearsonr(sampled_pred[i], sampled_pred[j])[0]
                            if not np.isnan(corr):
                                sample_correlations.append(corr)
                        except:
                            continue

                pert_metrics['consistency'] = np.mean(sample_correlations) if sample_correlations else 0.0
        else:
            pert_metrics['consistency'] = 1.0  # 单个样本视为完全一致

        perturbation_metrics[perturbation] = pert_metrics

    return perturbation_metrics


def compute_biological_metrics_fast(results) -> Dict[str, float]:
    """
    优化版生物相关指标计算

    参数:
        results: 包含预测结果的字典

    返回:
        包含生物指标的字典
    """

    bio_metrics = {}

    # 获取预测和真实值
    predictions = np.array(results['predictions'])
    truths = np.array(results['truths'])

    # 1. 基因表达水平恢复度 - 向量化计算
    if len(predictions) > 0:
        # 批量计算R²
        r2_scores = []
        for i in range(len(predictions)):
            if len(np.unique(predictions[i])) > 1:  # 避免除零错误
                r2 = r2_score(truths[i], predictions[i])
                if not np.isnan(r2):
                    r2_scores.append(r2)

        bio_metrics['expression_recovery'] = np.mean(r2_scores) if r2_scores else 0.0

        # 2. 表达谱相关性 - 向量化计算
        spearman_correlations = []
        for i in range(len(predictions)):
            if len(np.unique(predictions[i])) > 1:
                try:
                    corr = spearmanr(truths[i], predictions[i])[0]
                    if not np.isnan(corr):
                        spearman_correlations.append(corr)
                except:
                    continue

        bio_metrics['expression_spearman'] = np.mean(spearman_correlations) if spearman_correlations else 0.0

        # 3. 高表达基因预测准确性
        high_expr_threshold = np.percentile(truths, 90, axis=1)
        high_expr_mask = truths > high_expr_threshold[:, np.newaxis]

        if np.any(high_expr_mask):
            high_expr_r2 = []
            for i in range(len(predictions)):
                mask = high_expr_mask[i]
                if np.sum(mask) > 0:
                    r2 = r2_score(truths[i][mask], predictions[i][mask])
                    if not np.isnan(r2):
                        high_expr_r2.append(r2)
            bio_metrics['high_expression_r2'] = np.mean(high_expr_r2) if high_expr_r2 else 0.0
        else:
            bio_metrics['high_expression_r2'] = 0.0

        # 4. 低表达基因预测准确性
        low_expr_threshold = np.percentile(truths, 10, axis=1)
        low_expr_mask = truths < low_expr_threshold[:, np.newaxis]

        if np.any(low_expr_mask):
            low_expr_r2 = []
            for i in range(len(predictions)):
                mask = low_expr_mask[i]
                if np.sum(mask) > 0:
                    r2 = r2_score(truths[i][mask], predictions[i][mask])
                    if not np.isnan(r2):
                        low_expr_r2.append(r2)
            bio_metrics['low_expression_r2'] = np.mean(low_expr_r2) if low_expr_r2 else 0.0
        else:
            bio_metrics['low_expression_r2'] = 0.0

    else:
        # 如果没有数据，设置默认值
        bio_metrics['expression_recovery'] = 0.0
        bio_metrics['expression_spearman'] = 0.0
        bio_metrics['high_expression_r2'] = 0.0
        bio_metrics['low_expression_r2'] = 0.0

    # 5. 差异表达基因准确性
    if 'predictions_de' in results and 'truths_de' in results:
        predictions_de = results['predictions_de']
        truths_de = results['truths_de']

        if len(predictions_de) > 0:
            de_r2_scores = []
            for i in range(len(predictions_de)):
                if len(np.unique(predictions_de[i])) > 1:
                    r2 = r2_score(truths_de[i], predictions_de[i])
                    if not np.isnan(r2):
                        de_r2_scores.append(r2)

            bio_metrics['de_accuracy'] = np.mean(de_r2_scores) if de_r2_scores else 0.0
        else:
            bio_metrics['de_accuracy'] = 0.0
    else:
        bio_metrics['de_accuracy'] = 0.0


    return bio_metrics


def enhanced_compute_metrics_fast(results) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    优化版增强指标计算，大幅提升性能

    参数:
        results: 包含预测结果的字典

    返回:
        包含所有指标的元组 (基础指标+生物指标, 扰动特异性指标)
    """
    # 首先计算基础指标
    base_metrics, metrics_perturbation = compute_metrics(results)

    # 计算生物相关指标（优化版）
    bio_metrics = compute_biological_metrics_fast(results)

    # 计算扰动特异性生物指标（优化版）
    pert_bio_metrics = compute_perturbation_specific_metrics_fast(results)

    # 合并基础指标和生物指标
    enhanced_metrics = {**base_metrics, **bio_metrics}

    # 合并扰动特异性指标
    for pert in metrics_perturbation:
        if pert in pert_bio_metrics:
            metrics_perturbation[pert].update(pert_bio_metrics[pert])

    return enhanced_metrics, metrics_perturbation