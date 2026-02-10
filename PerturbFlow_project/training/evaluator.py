"""
CausalKAN-Flow 评估器模块
负责模型性能评估和指标计算
"""

import torch
import numpy as np
import scanpy as sc
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from typing import Dict, Any, List, Tuple  # 添加 Tuple 到导入列表中
from utils.general_utils import print_system

def evaluate_model(loader, model, uncertainty, device) -> Dict[str, Any]:
    """
    评估模型性能
    
    参数:
        loader: 数据加载器
        model: 待评估模型
        uncertainty: 是否计算不确定性
        device: 计算设备
        
    返回:
        包含评估结果的字典
    """
    
    model.eval()
    model.to(device)
    
    # 初始化结果存储
    perturbation_categories = []
    predictions = []
    truths = []
    predictions_de = []
    truths_de = []
    results = {}
    logvars = []
    
    for itr, batch in enumerate(loader):
        batch.to(device)
        perturbation_categories.extend(batch.pert)
                
        with torch.no_grad():
            if uncertainty:
                model_output = model(batch)
                # 处理ModelOutput格式
                if hasattr(model_output, 'pred'):
                    pred = model_output.pred
                    unc = model_output.unc if hasattr(model_output, 'unc') else None
                    if unc is not None:
                        logvars.extend(unc.cpu())
                elif isinstance(model_output, tuple) and len(model_output) == 2:
                    pred, unc = model_output
                    logvars.extend(unc.cpu())
                else:
                    pred = model_output
            else:
                model_output = model(batch)
                # 处理ModelOutput格式
                if hasattr(model_output, 'pred'):
                    pred = model_output.pred
                else:
                    pred = model_output

            truth = batch.y
            predictions.extend(pred.cpu())
            truths.extend(truth.cpu())
            
            # 差异表达基因
            for itr, de_idx in enumerate(batch.de_idx):
                predictions_de.append(pred[itr, de_idx])
                truths_de.append(truth[itr, de_idx])
    
    # 整理所有基因的结果
    results['perturbation_categories'] = np.array(perturbation_categories)
    predictions = torch.stack(predictions)
    truths = torch.stack(truths)
    results['predictions'] = predictions.detach().cpu().numpy()
    results['truths'] = truths.detach().cpu().numpy()
    
    # 整理差异表达基因的结果
    predictions_de = torch.stack(predictions_de)
    truths_de = torch.stack(truths_de)
    results['predictions_de'] = predictions_de.detach().cpu().numpy()
    results['truths_de'] = truths_de.detach().cpu().numpy()
    
    if uncertainty:
        results['logvars'] = torch.stack(logvars).detach().cpu().numpy()
    
    return results

def compute_metrics(results) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    计算评估指标
    
    参数:
        results: 包含预测结果的字典
        
    返回:
        包含平均指标和每个扰动指标的元组
    """
    
    # 初始化指标存储字典
    metrics = {}
    metrics_perturbation = {}
    
    # 定义要计算的指标及其对应的计算函数
    metric_to_function = {
        'mse': mean_squared_error,
        'pearson': lambda x, y: pearsonr(x, y)[0]
    }
    
    # 初始化指标列表
    for m in metric_to_function.keys():
        metrics[m] = []
        metrics[m + '_de'] = []
    
    # 对每个扰动类别单独计算指标
    for perturbation in np.unique(results['perturbation_categories']):
        metrics_perturbation[perturbation] = {}
        perturbation_idx = np.where(results['perturbation_categories'] == perturbation)[0]
        
        # 计算所有基因的指标
        for m, fct in metric_to_function.items():
            if m == 'pearson':
                val = fct(
                    results['predictions'][perturbation_idx].mean(0),
                    results['truths'][perturbation_idx].mean(0)
                )
                if np.isnan(val):
                    val = 0
            else:
                val = fct(
                    results['predictions'][perturbation_idx].mean(0),
                    results['truths'][perturbation_idx].mean(0)
                )
            
            metrics_perturbation[perturbation][m] = val
            metrics[m].append(metrics_perturbation[perturbation][m])
        
        # 对非对照扰动，计算差异表达基因的指标
        if perturbation != 'ctrl':
            for m, fct in metric_to_function.items():
                if m == 'pearson':
                    val = fct(
                        results['predictions_de'][perturbation_idx].mean(0),
                        results['truths_de'][perturbation_idx].mean(0)
                    )
                    if np.isnan(val):
                        val = 0
                else:
                    val = fct(
                        results['predictions_de'][perturbation_idx].mean(0),
                        results['truths_de'][perturbation_idx].mean(0)
                    )
                
                metrics_perturbation[perturbation][m + '_de'] = val
                metrics[m + '_de'].append(metrics_perturbation[perturbation][m + '_de'])
        else:
            # 对照组没有差异表达基因，指标设为0
            for m in metric_to_function.keys():
                metrics_perturbation[perturbation][m + '_de'] = 0
    
    # 计算所有扰动的平均指标
    for m in metric_to_function.keys():
        metrics[m] = np.mean(metrics[m])
        metrics[m + '_de'] = np.mean(metrics[m + '_de'])
    
    return metrics, metrics_perturbation

def deeper_analysis(adata, test_res, de_column_prefix='rank_genes_groups_cov', 
                   most_variable_genes=None) -> Dict[str, Dict[str, float]]:
    """
    深入分析模型性能
    
    参数:
        adata: AnnData对象
        test_res: 测试结果
        de_column_prefix: 差异表达基因列前缀
        most_variable_genes: 高变异基因列表
        
    返回:
        包含深入分析结果的字典
    """

    metric2fct = {
           'pearson': pearsonr,
           'mse': mean_squared_error
    }

    pert_metric = {}

    # 创建各种映射字典
    ## in silico modeling and upperbounding
    pert2pert_full_id = dict(adata.obs[['condition', 'condition_name']].values)
    geneid2name = dict(zip(adata.var.index.values, adata.var['gene_name']))
    geneid2idx = dict(zip(adata.var.index.values, range(len(adata.var.index.values))))
    
    
    # calculate mean expression for each condition
    unique_conditions = adata.obs.condition.unique()
    conditions2index = {}
    for i in unique_conditions:
        conditions2index[i] = np.where(adata.obs.condition == i)[0]

    condition2mean_expression = {}
    for i, j in conditions2index.items():
        condition2mean_expression[i] = np.mean(adata.X[j], axis = 0)
    
    pert_list = np.array(list(condition2mean_expression.keys()))
    mean_expression = np.array(list(condition2mean_expression.values())).reshape(len(adata.obs.condition.unique()), adata.X.toarray().shape[1])
    ctrl = mean_expression[np.where(pert_list == 'ctrl')[0]]

    # 如果没有提供高变异基因，则根据表达变异选择前200个
    if most_variable_genes is None:
        most_variable_genes = np.argsort(np.std(mean_expression, axis = 0))[-200:]

    gene_list = adata.var['gene_name'].values

    # 对每个扰动类别进行分析
    for pert in np.unique(test_res['perturbation_categories']):
        pert_metric[pert] = {}

        # 获取不同数量的差异表达基因索引
        de_idx = [geneid2idx[i] for i in adata.uns['rank_genes_groups_cov_all'][pert2pert_full_id[pert]][:20]]
        de_idx_200 = [geneid2idx[i] for i in adata.uns['rank_genes_groups_cov_all'][pert2pert_full_id[pert]][:200]]
        de_idx_100 = [geneid2idx[i] for i in adata.uns['rank_genes_groups_cov_all'][pert2pert_full_id[pert]][:100]]
        de_idx_50 = [geneid2idx[i] for i in adata.uns['rank_genes_groups_cov_all'][pert2pert_full_id[pert]][:50]]

        pert_idx = np.where(test_res['perturbation_categories'] == pert)[0]    
        pred_mean = np.mean(test_res['predictions_de'][pert_idx], axis = 0).reshape(-1,)
        true_mean = np.mean(test_res['truths_de'][pert_idx], axis = 0).reshape(-1,)
        
        # 计算所有基因的方向一致性
        direc_change = np.abs(np.sign(test_res['predictions'][pert_idx].mean(0) - ctrl[0]) - np.sign(test_res['truths'][pert_idx].mean(0) - ctrl[0]))            
        frac_correct_direction = len(np.where(direc_change == 0)[0])/len(geneid2name)
        pert_metric[pert]['frac_correct_direction_all'] = frac_correct_direction

        # 计算不同数量差异表达基因的方向一致性
        de_idx_map = {20: de_idx,
                      50: de_idx_50,
                      100: de_idx_100,
                      200: de_idx_200
                     }
        
        for val in [20, 50, 100, 200]:
            
            direc_change = np.abs(np.sign(test_res['predictions'][pert_idx].mean(0)[de_idx_map[val]] - ctrl[0][de_idx_map[val]]) - np.sign(test_res['truths'][pert_idx].mean(0)[de_idx_map[val]] - ctrl[0][de_idx_map[val]]))            
            frac_correct_direction = len(np.where(direc_change == 0)[0])/val
            pert_metric[pert]['frac_correct_direction_' + str(val)] = frac_correct_direction

        # 计算统计量    
        mean = np.mean(test_res['truths_de'][pert_idx], axis = 0)
        std = np.std(test_res['truths_de'][pert_idx], axis = 0)
        min_ = np.min(test_res['truths_de'][pert_idx], axis = 0)
        max_ = np.max(test_res['truths_de'][pert_idx], axis = 0)
        q25 = np.quantile(test_res['truths_de'][pert_idx], 0.25, axis = 0)
        q75 = np.quantile(test_res['truths_de'][pert_idx], 0.75, axis = 0)
        q55 = np.quantile(test_res['truths_de'][pert_idx], 0.55, axis = 0)
        q45 = np.quantile(test_res['truths_de'][pert_idx], 0.45, axis = 0)
        q40 = np.quantile(test_res['truths_de'][pert_idx], 0.4, axis = 0)
        q60 = np.quantile(test_res['truths_de'][pert_idx], 0.6, axis = 0)


        # 识别零表达和非零表达基因
        zero_des = np.intersect1d(np.where(min_ == 0)[0], np.where(max_ == 0)[0])
        nonzero_des = np.setdiff1d(list(range(20)), zero_des)

        
        if len(nonzero_des) == 0:
            print("警告: 所有差异表达基因都是零表达")
            pass
            # pert that all de genes are 0...
        else:            
            # 计算非零基因的方向一致性
            direc_change = np.abs(np.sign(pred_mean[nonzero_des] - ctrl[0][de_idx][nonzero_des]) - np.sign(true_mean[nonzero_des] - ctrl[0][de_idx][nonzero_des]))            
            frac_correct_direction = len(np.where(direc_change == 0)[0])/len(nonzero_des)
            pert_metric[pert]['frac_correct_direction_20_nonzero'] = frac_correct_direction

            # 计算预测值在不同范围内的比例
            in_range = (pred_mean[nonzero_des] >= min_[nonzero_des]) & (pred_mean[nonzero_des] <= max_[nonzero_des])
            frac_in_range = sum(in_range)/len(nonzero_des)
            pert_metric[pert]['frac_in_range'] = frac_in_range

            in_range_5 = (pred_mean[nonzero_des] >= q45[nonzero_des]) & (pred_mean[nonzero_des] <= q55[nonzero_des])
            frac_in_range_45_55 = sum(in_range_5)/len(nonzero_des)
            pert_metric[pert]['frac_in_range_45_55'] = frac_in_range_45_55

            in_range_10 = (pred_mean[nonzero_des] >= q40[nonzero_des]) & (pred_mean[nonzero_des] <= q60[nonzero_des])
            frac_in_range_40_60 = sum(in_range_10)/len(nonzero_des)
            pert_metric[pert]['frac_in_range_40_60'] = frac_in_range_40_60

            in_range_25 = (pred_mean[nonzero_des] >= q25[nonzero_des]) & (pred_mean[nonzero_des] <= q75[nonzero_des])
            frac_in_range_25_75 = sum(in_range_25)/len(nonzero_des)
            pert_metric[pert]['frac_in_range_25_75'] = frac_in_range_25_75

            # 计算sigma统计量
            zero_idx = np.where(std > 0)[0]
            sigma = (np.abs(pred_mean[zero_idx] - mean[zero_idx]))/(std[zero_idx])
            pert_metric[pert]['mean_sigma'] = np.mean(sigma)
            pert_metric[pert]['std_sigma'] = np.std(sigma)
            pert_metric[pert]['frac_sigma_below_1'] = 1 - len(np.where(sigma > 1)[0])/len(zero_idx)
            pert_metric[pert]['frac_sigma_below_2'] = 1 - len(np.where(sigma > 2)[0])/len(zero_idx)

        # 计算相对于对照的差异指标    
        ## correlation on delta
        p_idx = np.where(test_res['perturbation_categories'] == pert)[0]

        for m, fct in metric2fct.items():
            if m != 'mse':
                # 所有基因的差异指标
                val = fct(test_res['predictions'][p_idx].mean(0)- ctrl[0], test_res['truths'][p_idx].mean(0)-ctrl[0])[0]
                if np.isnan(val):
                    val = 0

                pert_metric[pert][m + '_delta'] = val
                
                # 差异表达基因的差异指标
                val = fct(test_res['predictions'][p_idx].mean(0)[de_idx] - ctrl[0][de_idx], test_res['truths'][p_idx].mean(0)[de_idx]-ctrl[0][de_idx])[0]
                if np.isnan(val):
                    val = 0

                pert_metric[pert][m + '_delta_de'] = val

        # 计算fold change差异        
        ## up fold changes > 10?
        pert_mean = np.mean(test_res['truths'][p_idx], axis = 0).reshape(-1,)

        fold_change = pert_mean/ctrl
        fold_change[np.isnan(fold_change)] = 0
        fold_change[np.isinf(fold_change)] = 0
        # 去除表达量过低的无意义fold change
        ## this is to remove the ones that are super low and the fold change becomes unmeaningful
        fold_change[0][np.where(pert_mean < 0.5)[0]] = 0


        # 计算不同fold change范围的差异
        o =  np.where(fold_change[0] > 0)[0]

        pred_fc = test_res['predictions'][p_idx].mean(0)[o]
        true_fc = test_res['truths'][p_idx].mean(0)[o]
        ctrl_fc = ctrl[0][o]

        if len(o) > 0:
            pert_metric[pert]['fold_change_gap_all'] = np.mean(np.abs(pred_fc/ctrl_fc - true_fc/ctrl_fc))


        o = np.intersect1d(np.where(fold_change[0] <0.333)[0], np.where(fold_change[0] > 0)[0])

        pred_fc = test_res['predictions'][p_idx].mean(0)[o]
        true_fc = test_res['truths'][p_idx].mean(0)[o]
        ctrl_fc = ctrl[0][o]

        if len(o) > 0:
            pert_metric[pert]['fold_change_gap_downreg_0.33'] = np.mean(np.abs(pred_fc/ctrl_fc - true_fc/ctrl_fc))


        o = np.intersect1d(np.where(fold_change[0] <0.1)[0], np.where(fold_change[0] > 0)[0])

        pred_fc = test_res['predictions'][p_idx].mean(0)[o]
        true_fc = test_res['truths'][p_idx].mean(0)[o]
        ctrl_fc = ctrl[0][o]

        if len(o) > 0:
            pert_metric[pert]['fold_change_gap_downreg_0.1'] = np.mean(np.abs(pred_fc/ctrl_fc - true_fc/ctrl_fc))

        o = np.where(fold_change[0] > 3)[0]

        pred_fc = test_res['predictions'][p_idx].mean(0)[o]
        true_fc = test_res['truths'][p_idx].mean(0)[o]
        ctrl_fc = ctrl[0][o]

        if len(o) > 0:
            pert_metric[pert]['fold_change_gap_upreg_3'] = np.mean(np.abs(pred_fc/ctrl_fc - true_fc/ctrl_fc))

        o = np.where(fold_change[0] > 10)[0]

        pred_fc = test_res['predictions'][p_idx].mean(0)[o]
        true_fc = test_res['truths'][p_idx].mean(0)[o]
        ctrl_fc = ctrl[0][o]

        if len(o) > 0:
            pert_metric[pert]['fold_change_gap_upreg_10'] = np.mean(np.abs(pred_fc/ctrl_fc - true_fc/ctrl_fc))

        # 计算高变异基因的指标
        ## most variable genes
        for m, fct in metric2fct.items():
            if m != 'mse':
                val = fct(test_res['predictions'][p_idx].mean(0)[most_variable_genes] - ctrl[0][most_variable_genes], test_res['truths'][p_idx].mean(0)[most_variable_genes]-ctrl[0][most_variable_genes])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_delta_top200_hvg'] = val


                val = fct(test_res['predictions'][p_idx].mean(0)[most_variable_genes], test_res['truths'][p_idx].mean(0)[most_variable_genes])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_top200_hvg'] = val
            else:
                val = fct(test_res['predictions'][p_idx].mean(0)[most_variable_genes], test_res['truths'][p_idx].mean(0)[most_variable_genes])
                pert_metric[pert][m + '_top200_hvg'] = val


        ## top 20/50/100/200 DEs
        ## 计算不同数量差异表达基因的指标
        for m, fct in metric2fct.items():
            if m != 'mse':
                val = fct(test_res['predictions'][p_idx].mean(0)[de_idx] - ctrl[0][de_idx], test_res['truths'][p_idx].mean(0)[de_idx]-ctrl[0][de_idx])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_delta_top20_de'] = val


                val = fct(test_res['predictions'][p_idx].mean(0)[de_idx], test_res['truths'][p_idx].mean(0)[de_idx])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_top20_de'] = val
            else:
                val = fct(test_res['predictions'][p_idx].mean(0)[de_idx] - ctrl[0][de_idx], test_res['truths'][p_idx].mean(0)[de_idx]-ctrl[0][de_idx])
                pert_metric[pert][m + '_top20_de'] = val

        
        for m, fct in metric2fct.items():
            if m != 'mse':
                val = fct(test_res['predictions'][p_idx].mean(0)[de_idx_200] - ctrl[0][de_idx_200], test_res['truths'][p_idx].mean(0)[de_idx_200]-ctrl[0][de_idx_200])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_delta_top200_de'] = val


                val = fct(test_res['predictions'][p_idx].mean(0)[de_idx_200], test_res['truths'][p_idx].mean(0)[de_idx_200])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_top200_de'] = val
            else:
                val = fct(test_res['predictions'][p_idx].mean(0)[de_idx_200] - ctrl[0][de_idx_200], test_res['truths'][p_idx].mean(0)[de_idx_200]-ctrl[0][de_idx_200])
                pert_metric[pert][m + '_top200_de'] = val

        for m, fct in metric2fct.items():
            if m != 'mse':

                val = fct(test_res['predictions'][p_idx].mean(0)[de_idx_100] - ctrl[0][de_idx_100], test_res['truths'][p_idx].mean(0)[de_idx_100]-ctrl[0][de_idx_100])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_delta_top100_de'] = val


                val = fct(test_res['predictions'][p_idx].mean(0)[de_idx_100], test_res['truths'][p_idx].mean(0)[de_idx_100])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_top100_de'] = val
            else:
                val = fct(test_res['predictions'][p_idx].mean(0)[de_idx_100] - ctrl[0][de_idx_100], test_res['truths'][p_idx].mean(0)[de_idx_100]-ctrl[0][de_idx_100])
                pert_metric[pert][m + '_top100_de'] = val

        for m, fct in metric2fct.items():
            if m != 'mse':

                val = fct(test_res['predictions'][p_idx].mean(0)[de_idx_50] - ctrl[0][de_idx_50], test_res['truths'][p_idx].mean(0)[de_idx_50]-ctrl[0][de_idx_50])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_delta_top50_de'] = val


                val = fct(test_res['predictions'][p_idx].mean(0)[de_idx_50], test_res['truths'][p_idx].mean(0)[de_idx_50])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_top50_de'] = val
            else:
                val = fct(test_res['predictions'][p_idx].mean(0)[de_idx_50] - ctrl[0][de_idx_50], test_res['truths'][p_idx].mean(0)[de_idx_50]-ctrl[0][de_idx_50])
                pert_metric[pert][m + '_top50_de'] = val

    return pert_metric


def non_dropout_analysis(adata, test_res) -> Dict[str, Dict[str, float]]:
    """
    分析非dropout基因的预测性能
    
    参数:
        adata: 包含基因表达数据的AnnData对象
        test_res: 测试结果字典
    """

    metric2fct = {
           'pearson': pearsonr,
           'mse': mean_squared_error
    }

    pert_metric = {}

    ## in silico modeling and upperbounding
    # pert2pert_full_id：{key:condition,value：condition_name}
    pert2pert_full_id = dict(adata.obs[['condition', 'condition_name']].values)
    # geneid2name：{key:adata.var.index(ENSGXXX),value:gene_name}
    geneid2name = dict(zip(adata.var.index.values, adata.var['gene_name']))
    # geneid2idx:{key:adata.var.index(ENSGXXX),value:0
    geneid2idx = dict(zip(adata.var.index.values, range(len(adata.var.index.values))))
    
    # calculate mean expression for each condition
    unique_conditions = adata.obs.condition.unique()
    conditions2index = {}
    for i in unique_conditions:
        conditions2index[i] = np.where(adata.obs.condition == i)[0]

    condition2mean_expression = {}
    for i, j in conditions2index.items():
        condition2mean_expression[i] = np.mean(adata.X[j], axis = 0)
    
    pert_list = np.array(list(condition2mean_expression.keys()))
    mean_expression = np.array(list(condition2mean_expression.values())).reshape(len(adata.obs.condition.unique()), adata.X.toarray().shape[1])
    ctrl = mean_expression[np.where(pert_list == 'ctrl')[0]]

    gene_list = adata.var['gene_name'].values

    # 初始化用于存储所有pert指标值的字典
    all_pearson_delta = []
    all_pearson = []
    all_mse = []
    all_mse_scouter = []  # 新增：用于存储所有pert的mse_scouter值
    
    for pert in np.unique(test_res['perturbation_categories']):
        pert_metric[pert] = {}
        # pert_idx：在测试集里每个扰动类别对应的索引
        pert_idx = np.where(test_res['perturbation_categories'] == pert)[0]    
        
        # 获取差异表达基因索引
        de_idx = [geneid2idx[i] for i in adata.uns['top_non_dropout_de_20'][pert2pert_full_id[pert]]]
        non_zero_idx = adata.uns['non_zeros_gene_idx'][pert2pert_full_id[pert]]
        non_dropout_gene_idx = adata.uns['non_dropout_gene_idx'][pert2pert_full_id[pert]]

        # 计算方向一致性(多种基因集)
        direc_change = np.abs(np.sign(test_res['predictions'][pert_idx].mean(0)[de_idx] - ctrl[0][de_idx]) - np.sign(test_res['truths'][pert_idx].mean(0)[de_idx] - ctrl[0][de_idx]))            
        frac_correct_direction = len(np.where(direc_change == 0)[0])/len(de_idx)
        pert_metric[pert]['frac_correct_direction_top20_non_dropout'] = frac_correct_direction
        
        frac_direction_opposite = len(np.where(direc_change == 2)[0])/len(de_idx)
        pert_metric[pert]['frac_opposite_direction_top20_non_dropout'] = frac_direction_opposite
        
        frac_direction_opposite = len(np.where(direc_change == 1)[0])/len(de_idx)
        pert_metric[pert]['frac_0/1_direction_top20_non_dropout'] = frac_direction_opposite
        
        direc_change = np.abs(np.sign(test_res['predictions'][pert_idx].mean(0)[non_zero_idx] - ctrl[0][non_zero_idx]) - np.sign(test_res['truths'][pert_idx].mean(0)[non_zero_idx] - ctrl[0][non_zero_idx]))            
        frac_correct_direction = len(np.where(direc_change == 0)[0])/len(non_zero_idx)
        pert_metric[pert]['frac_correct_direction_non_zero'] = frac_correct_direction

        frac_direction_opposite = len(np.where(direc_change == 2)[0])/len(non_zero_idx)
        pert_metric[pert]['frac_opposite_direction_non_zero'] = frac_direction_opposite
        
        frac_direction_opposite = len(np.where(direc_change == 1)[0])/len(non_zero_idx)
        pert_metric[pert]['frac_0/1_direction_non_zero'] = frac_direction_opposite
        
        direc_change = np.abs(np.sign(test_res['predictions'][pert_idx].mean(0)[non_dropout_gene_idx] - ctrl[0][non_dropout_gene_idx]) - np.sign(test_res['truths'][pert_idx].mean(0)[non_dropout_gene_idx] - ctrl[0][non_dropout_gene_idx]))            
        frac_correct_direction = len(np.where(direc_change == 0)[0])/len(non_dropout_gene_idx)
        pert_metric[pert]['frac_correct_direction_non_dropout'] = frac_correct_direction
        
        frac_direction_opposite = len(np.where(direc_change == 2)[0])/len(non_dropout_gene_idx)
        pert_metric[pert]['frac_opposite_direction_non_dropout'] = frac_direction_opposite
        
        frac_direction_opposite = len(np.where(direc_change == 1)[0])/len(non_dropout_gene_idx)
        pert_metric[pert]['frac_0/1_direction_non_dropout'] = frac_direction_opposite


        # 计算统计量
        mean = np.mean(test_res['truths'][pert_idx][:, de_idx], axis = 0)
        std = np.std(test_res['truths'][pert_idx][:, de_idx], axis = 0)
        min_ = np.min(test_res['truths'][pert_idx][:, de_idx], axis = 0)
        max_ = np.max(test_res['truths'][pert_idx][:, de_idx], axis = 0)
        q25 = np.quantile(test_res['truths'][pert_idx][:, de_idx], 0.25, axis = 0)
        q75 = np.quantile(test_res['truths'][pert_idx][:, de_idx], 0.75, axis = 0)
        q55 = np.quantile(test_res['truths'][pert_idx][:, de_idx], 0.55, axis = 0)
        q45 = np.quantile(test_res['truths'][pert_idx][:, de_idx], 0.45, axis = 0)
        q40 = np.quantile(test_res['truths'][pert_idx][:, de_idx], 0.4, axis = 0)
        q60 = np.quantile(test_res['truths'][pert_idx][:, de_idx], 0.6, axis = 0)
        
        # 识别零表达和非零表达基因
        zero_des = np.intersect1d(np.where(min_ == 0)[0], np.where(max_ == 0)[0])
        nonzero_des = np.setdiff1d(list(range(20)), zero_des)
        
        if len(nonzero_des) == 0:
            print("警告: 所有差异表达基因都是零表达")
            pass
            # pert that all de genes are 0...
        else:
            # 预测值在当前扰动类别条件下的差异表达基因沿着列的均值
            # 这20个差异表达基因在扰动条件下的均值
            pred_mean = np.mean(test_res['predictions'][pert_idx][:, de_idx], axis = 0).reshape(-1,)
            true_mean = np.mean(test_res['truths'][pert_idx][:, de_idx], axis = 0).reshape(-1,)
            
            # 计算预测值在不同范围内的比例
            in_range = (pred_mean[nonzero_des] >= min_[nonzero_des]) & (pred_mean[nonzero_des] <= max_[nonzero_des])
            frac_in_range = sum(in_range)/len(nonzero_des)
            pert_metric[pert]['frac_in_range_non_dropout'] = frac_in_range

            in_range_5 = (pred_mean[nonzero_des] >= q45[nonzero_des]) & (pred_mean[nonzero_des] <= q55[nonzero_des])
            frac_in_range_45_55 = sum(in_range_5)/len(nonzero_des)
            pert_metric[pert]['frac_in_range_45_55_non_dropout'] = frac_in_range_45_55

            in_range_10 = (pred_mean[nonzero_des] >= q40[nonzero_des]) & (pred_mean[nonzero_des] <= q60[nonzero_des])
            frac_in_range_40_60 = sum(in_range_10)/len(nonzero_des)
            pert_metric[pert]['frac_in_range_40_60_non_dropout'] = frac_in_range_40_60

            in_range_25 = (pred_mean[nonzero_des] >= q25[nonzero_des]) & (pred_mean[nonzero_des] <= q75[nonzero_des])
            frac_in_range_25_75 = sum(in_range_25)/len(nonzero_des)
            pert_metric[pert]['frac_in_range_25_75_non_dropout'] = frac_in_range_25_75

            # 计算sigma统计量
            zero_idx = np.where(std > 0)[0]
            sigma = (np.abs(pred_mean[zero_idx] - mean[zero_idx]))/(std[zero_idx])
            pert_metric[pert]['mean_sigma_non_dropout'] = np.mean(sigma)
            pert_metric[pert]['std_sigma_non_dropout'] = np.std(sigma)
            pert_metric[pert]['frac_sigma_below_1_non_dropout'] = 1 - len(np.where(sigma > 1)[0])/len(zero_idx)
            pert_metric[pert]['frac_sigma_below_2_non_dropout'] = 1 - len(np.where(sigma > 2)[0])/len(zero_idx)

        # 计算差异表达基因的指标
        # delta 指的是相对对照的差异
        p_idx = np.where(test_res['perturbation_categories'] == pert)[0]
        for m, fct in metric2fct.items():
            if m != 'mse':
                # 计算相对于对照的差异
                # metric['Pearson'][condition] = pearsonr(true-ctrl, pred-ctrl)[0]
                val = fct(test_res['predictions'][p_idx].mean(0)[de_idx] - ctrl[0][de_idx], test_res['truths'][p_idx].mean(0)[de_idx]-ctrl[0][de_idx])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_delta_top20_de_non_dropout'] = val
                # 将所有的相对对照组的 pearson 指数添加到数据中
                all_pearson_delta.append(val)

                # 计算绝对值的指标
                val = fct(test_res['predictions'][p_idx].mean(0)[de_idx], test_res['truths'][p_idx].mean(0)[de_idx])[0]
                if np.isnan(val):
                    val = 0
                pert_metric[pert][m + '_top20_de_non_dropout'] = val
                # 把所有的绝对值 pearson 指数添加到数据中
                all_pearson.append(val)

            else:
                # 计算MSE（原始版本）
                val = fct(test_res['predictions'][p_idx].mean(0)[de_idx] - ctrl[0][de_idx], test_res['truths'][p_idx].mean(0)[de_idx]-ctrl[0][de_idx])
                pert_metric[pert][m + '_top20_de_non_dropout'] = val
                # 把所有的 mse 值添加到数据中
                all_mse.append(val)
                
                mse_true_pred = mean_squared_error(test_res['predictions'][p_idx].mean(0)[de_idx], 
                       test_res['truths'][p_idx].mean(0)[de_idx])
                mse_true_ctrl = mean_squared_error(ctrl[0][de_idx], 
                                test_res['truths'][p_idx].mean(0)[de_idx])
                
                # 避免除以0的情况
                if mse_true_ctrl == 0:
                    mse_scouter_val = float('inf')  # 如果分母为0，设为无穷大
                else:
                    mse_scouter_val = mse_true_pred / mse_true_ctrl
                
                pert_metric[pert]['mse_scouter_top20_de_non_dropout'] = mse_scouter_val

                all_mse_scouter.append(mse_scouter_val)
                
   
    return pert_metric