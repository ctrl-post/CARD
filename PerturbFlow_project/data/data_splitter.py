"""
CausalKAN-Flow 数据分割器模块
负责将数据划分为训练集、验证集和测试集
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.general_utils import parse_any_perturbation, print_system

class DataSplitter:
    """
    数据分割类，用于将数据划分为训练集、验证集和测试集
    
    参数:
        adata: AnnData对象
        split_type: 分割类型('single', 'combo', 'simulation'等)
        seen: 在测试集中见过的基因数量(用于combo分割)
    """
    
    def __init__(self, adata, split_type='single', seen=0):
        self.adata = adata
        self.split_type = split_type
        self.seen = seen
    
    def split_data(self, test_size=0.1, test_perturbation_genes=None,
                   test_perturbations=None, split_name='split', seed=None, val_size=0.1,
                   train_gene_set_size=0.75, combo_seen2_train_frac=0.75, only_test_set_perturbations=False):
        """
        分割数据集并将分割结果作为新列添加到数据中

        参数:
            test_size: 测试集比例
            test_perturbation_genes: 测试集中包含的基因列表
            test_perturbations: 直接指定的测试集扰动列表
            split_name: 分割结果列名
            seed: 数据划分随机种子
            val_size: 验证集比例
            train_gene_set_size: 训练基因集大小比例
            combo_seen2_train_frac: combo分割中seen2的训练比例
            only_test_set_perturbations: 是否仅使用指定的测试集扰动
        """
        if seed is not None:
            np.random.seed(seed=seed)

        unique_perturbations = [p for p in self.adata.obs['condition'].unique() if p != 'ctrl']

        if self.split_type == 'simulation':
            train, test, test_subgroup = self._get_simulation_split(unique_perturbations,
                                                                  train_gene_set_size,
                                                                  combo_seen2_train_frac, 
                                                                  seed, test_perturbations, only_test_set_perturbations)
    
            train, val, val_subgroup = self._get_simulation_split(train,
                                                                  0.9,
                                                                  0.9,
                                                                  seed)
            train.append('ctrl')

        elif self.split_type == 'simulation_single':
            train, test, test_subgroup = self._get_simulation_split_single(unique_perturbations,
                                                                  train_gene_set_size,
                                                                  seed, test_perturbations, only_test_set_perturbations)

            train, val, val_subgroup = self._get_simulation_split_single(train,
                                                                  0.9,
                                                                  seed)

        elif self.split_type == 'no_test':
            train, val = self._get_split_list(unique_perturbations,
                                          test_size=val_size) 

        else:
            train, test = self._get_split_list(unique_perturbations,
                                          test_perturbation_genes=test_perturbation_genes,
                                          test_perturbations=test_perturbations,
                                          test_size=test_size)
            
            train, val = self._get_split_list(train, test_size=val_size)

        map_dict = {x: 'train' for x in train}
        map_dict.update({x: 'val' for x in val})
        if self.split_type != 'no_test':
            map_dict.update({x: 'test' for x in test})
        map_dict.update({'ctrl': 'train'})
        
        self.adata.obs[split_name] = self.adata.obs['condition'].map(map_dict)

        if self.split_type == 'simulation':
            return self.adata, {'test_subgroup': test_subgroup, 
                                'val_subgroup': val_subgroup
                               }
        else:
            return self.adata
    
    def _get_simulation_split_single(self, perturbation_list, train_gene_set_size=0.85, 
                                    seed=1, test_set_perturbations=None, only_test_set_perturbations=False):
        """
        单扰动仿真分割
        
        参数:
            perturbation_list: 扰动列表
            train_gene_set_size: 训练基因集大小比例
            seed: 随机种子
            test_set_perturbations: 指定的测试集扰动
            only_test_set_perturbations: 是否仅使用指定的测试集扰动
        """
        unique_perturbation_genes = self._get_genes_from_perturbations(perturbation_list)

        perturbation_train = []
        perturbation_test = []
        np.random.seed(seed=seed)
        
        if only_test_set_perturbations and (test_set_perturbations is not None):
            ood_genes = np.array(test_set_perturbations)
            train_gene_candidates = np.setdiff1d(unique_perturbation_genes, ood_genes)
        else:
            train_gene_candidates = np.random.choice(unique_perturbation_genes,
                                                    int(len(unique_perturbation_genes) * train_gene_set_size), replace=False)

            if test_set_perturbations is not None:
                num_overlap = len(np.intersect1d(train_gene_candidates, test_set_perturbations))

                train_gene_candidates = train_gene_candidates[~np.isin(train_gene_candidates, test_set_perturbations)]
                ood_genes_exclude_test_set = np.setdiff1d(unique_perturbation_genes, np.union1d(train_gene_candidates, test_set_perturbations))
                train_set_addition = np.random.choice(ood_genes_exclude_test_set, num_overlap, replace=False)
                train_gene_candidates = np.concatenate((train_gene_candidates, train_set_addition))

            ood_genes = np.setdiff1d(unique_perturbation_genes, train_gene_candidates)  

        perturbation_single_train = self._get_perturbations_from_genes(train_gene_candidates, perturbation_list, 'single')
        unseen_single = self._get_perturbations_from_genes(ood_genes, perturbation_list, 'single')

        assert len(unseen_single) + len(perturbation_single_train) == len(perturbation_list)

        return perturbation_single_train, unseen_single, {'unseen_single': unseen_single}
    
    def _get_simulation_split(self, perturbation_list, train_gene_set_size=0.85, 
                             combo_seen2_train_frac=0.85, seed=1, test_set_perturbations=None, 
                             only_test_set_perturbations=False):
        """
        组合扰动仿真分割
        
        参数:
            perturbation_list: 扰动列表
            train_gene_set_size: 训练基因集大小比例
            combo_seen2_train_frac: combo分割中seen2的训练比例
            seed: 随机种子
            test_set_perturbations: 指定的测试集扰动
            only_test_set_perturbations: 是否仅使用指定的测试集扰动
        """
        unique_perturbation_genes = self._get_genes_from_perturbations(perturbation_list)

        perturbation_train = []
        perturbation_test = []
        np.random.seed(seed=seed)
        
        if only_test_set_perturbations and (test_set_perturbations is not None):
            ood_genes = np.array(test_set_perturbations)
            train_gene_candidates = np.setdiff1d(unique_perturbation_genes, ood_genes)
        else:
            train_gene_candidates = np.random.choice(unique_perturbation_genes,
                                                    int(len(unique_perturbation_genes) * train_gene_set_size), replace=False)

            if test_set_perturbations is not None:
                num_overlap = len(np.intersect1d(train_gene_candidates, test_set_perturbations))

                train_gene_candidates = train_gene_candidates[~np.isin(train_gene_candidates, test_set_perturbations)]
                ood_genes_exclude_test_set = np.setdiff1d(unique_perturbation_genes, np.union1d(train_gene_candidates, test_set_perturbations))
                train_set_addition = np.random.choice(ood_genes_exclude_test_set, num_overlap, replace=False)
                train_gene_candidates = np.concatenate((train_gene_candidates, train_set_addition))

            ood_genes = np.setdiff1d(unique_perturbation_genes, train_gene_candidates)                

        perturbation_single_train = self._get_perturbations_from_genes(train_gene_candidates, perturbation_list, 'single')
        perturbation_combo = self._get_perturbations_from_genes(train_gene_candidates, perturbation_list, 'combo')

        perturbation_train.extend(perturbation_single_train)

        combo_seen1 = [x for x in perturbation_combo if len([t for t in x.split('+') if
                                     t in train_gene_candidates]) == 1]
        perturbation_test.extend(combo_seen1)

        perturbation_combo = np.setdiff1d(perturbation_combo, combo_seen1)
        
        perturbation_combo_train = np.random.choice(perturbation_combo, int(len(perturbation_combo) * combo_seen2_train_frac), replace=False)

        combo_seen2 = np.setdiff1d(perturbation_combo, perturbation_combo_train).tolist()
        perturbation_test.extend(combo_seen2)
        perturbation_train.extend(perturbation_combo_train)

        unseen_single = self._get_perturbations_from_genes(ood_genes, perturbation_list, 'single')
        combo_ood = self._get_perturbations_from_genes(ood_genes, perturbation_list, 'combo')
        perturbation_test.extend(unseen_single)

        combo_seen0 = [x for x in combo_ood if len([t for t in x.split('+') if
                                     t in train_gene_candidates]) == 0]
        perturbation_test.extend(combo_seen0)

        assert len(combo_seen1) + len(combo_seen0) + len(unseen_single) + len(perturbation_train) + len(combo_seen2) == len(perturbation_list)

        return perturbation_train, perturbation_test, {'combo_seen0': combo_seen0,
                                       'combo_seen1': combo_seen1,
                                       'combo_seen2': combo_seen2,
                                       'unseen_single': unseen_single}
    
    def _get_split_list(self, perturbation_list, test_size=0.1,
                       test_perturbation_genes=None, test_perturbations=None,
                       hold_outs=True):
        """
        通用分割方法，根据split_type分割扰动列表
        
        参数:
            perturbation_list: 扰动列表
            test_size: 测试集比例
            test_perturbation_genes: 测试基因列表
            test_perturbations: 直接指定的测试扰动
            hold_outs: 是否保留某些扰动
        """

        single_perturbations = [p for p in perturbation_list if 'ctrl' in p and p != 'ctrl']
        combo_perturbations = [p for p in perturbation_list if 'ctrl' not in p]
        unique_perturbation_genes = self._get_genes_from_perturbations(perturbation_list)
        hold_out = []

        if test_perturbation_genes is None:
            test_perturbation_genes = np.random.choice(unique_perturbation_genes,
                                        int(len(single_perturbations) * test_size))

        if self.split_type == 'single' or self.split_type == 'single_only':
            test_perturbations = self._get_perturbations_from_genes(test_perturbation_genes, perturbation_list, 'single')
            
            if self.split_type == 'single_only':
                hold_out = combo_perturbations
            else:
                hold_out = self._get_perturbations_from_genes(test_perturbation_genes, perturbation_list, 'combo')

        elif self.split_type == 'no_test':
            if test_perturbations is None:
                test_perturbations = np.random.choice(perturbation_list,
                                    int(len(perturbation_list) * test_size))

        elif self.split_type == 'combo':
            if self.seen == 0:
                single_perturbations = self._get_perturbations_from_genes(test_perturbation_genes,
                                                         perturbation_list, 'single')
                combo_perturbations = self._get_perturbations_from_genes(test_perturbation_genes,
                                                        perturbation_list, 'combo')

                if hold_outs:
                    hold_out = [t for t in combo_perturbations if
                                len([t for t in t.split('+') if
                                     t not in test_perturbation_genes]) > 0]
                
                combo_perturbations = [c for c in combo_perturbations if c not in hold_out]
                test_perturbations = single_perturbations + combo_perturbations

            elif self.seen == 1:
                single_perturbations = self._get_perturbations_from_genes(test_perturbation_genes,
                                                         perturbation_list, 'single')
                combo_perturbations = self._get_perturbations_from_genes(test_perturbation_genes,
                                                        perturbation_list, 'combo')

                if hold_outs:
                    hold_out = [t for t in combo_perturbations if
                                len([t for t in t.split('+') if
                                     t not in test_perturbation_genes]) > 1]

                combo_perturbations = [c for c in combo_perturbations if c not in hold_out]
                test_perturbations = single_perturbations + combo_perturbations

            elif self.seen == 2:
                if test_perturbations is None:
                    test_perturbations = np.random.choice(combo_perturbations,
                                         int(len(combo_perturbations) * test_size))       
                
       
                else:
                    test_perturbations = np.array(test_perturbations)
        
        else:
            if test_perturbations is None:
                test_perturbations = np.random.choice(combo_perturbations,
                                    int(len(combo_perturbations) * test_size))
        
        train_perturbations = [p for p in perturbation_list if (p not in test_perturbations)
                                        and (p not in hold_out)]

        return train_perturbations, test_perturbations

    def _get_perturbations_from_genes(self, genes, perturbation_list, type_='both'):
        """
        获取包含指定基因的扰动
        
        参数:
            genes: 基因列表
            perturbation_list: 扰动列表
            type_: 扰动类型('single', 'combo', 'both')
        """

        single_perturbations = [p for p in perturbation_list if ('ctrl' in p) and (p != 'ctrl')]
        combo_perturbations = [p for p in perturbation_list if 'ctrl' not in p]
        
        perturbations = []
        
        if type_ == 'single':
            perturbation_candidate_list = single_perturbations

        elif type_ == 'combo':
            perturbation_candidate_list = combo_perturbations

        elif type_ == 'both':
            perturbation_candidate_list = perturbation_list
            
        for p in perturbation_candidate_list:
            for g in genes:
                if g in parse_any_perturbation(p):
                    perturbations.append(p)
                    break
        
        return perturbations

    def _get_genes_from_perturbations(self, perturbations):
        """
        从扰动中提取基因列表
        
        参数:
            perturbations: 扰动列表
        """
        if type(perturbations) is str:
            perturbations = [perturbations]

        gene_list = [p.split('+') for p in np.unique(perturbations)]
        gene_list = [item for sublist in gene_list for item in sublist]
        gene_list = [g for g in gene_list if g != 'ctrl']
        return np.unique(gene_list)