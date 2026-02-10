"""
CausalKAN-Flow 数据加载器模块
负责加载和处理基因扰动数据
"""

import os
import pickle
import numpy as np
import torch
import scanpy as sc
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

from .data_splitter import DataSplitter
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.general_utils import print_system, condition_sort

class PerturbationDataLoader:
    """
    用于加载和处理扰动数据的类
    
    属性:
        data_path: str - 保存/加载数据的路径
        default_perturbation_graph: bool - 是否使用默认的扰动图
        gene_set_path: str - 用于扰动图的基因集的路径
        dataset_name: str - 数据集的名称
        dataset_path: str - 数据集的路径
        adata: AnnData - 包含数据集的 AnnData 对象
        dataset_processed: bool - 数据集是否已处理
        ctrl_adata: AnnData - 包含对照样本的 AnnData 对象
        gene_names: list - 基因名称列表
        node_map: dict - 将基因名称映射到索引的字典
        split: str - 划分类型
        seed: int - 划分时使用的随机种子
        subgroup: str - 划分时使用的子组
        train_gene_set_size: int - 用于训练的基因数量
    """
    
    def __init__(self, data_path, gene_set_path=None, default_perturbation_graph=True):
        """
        初始化 PerturbationDataLoader
        
        参数:
            data_path: str - 数据保存/加载路径
            gene_set_path: str - 用于扰动图的基因集路径
            default_perturbation_graph: bool - 是否使用默认扰动图
        """
        # Dataset/Dataloader attributes
        self.data_path = data_path
        self.default_perturbation_graph = default_perturbation_graph
        self.gene_set_path = gene_set_path
        self.dataset_name = None
        self.dataset_path = None
        self.adata = None
        self.dataset_processed = None
        self.ctrl_adata = None
        self.gene_names = []
        self.node_map = {}
        self.select_GPT_embedding = None
        
        # Split attributes
        self.split = None
        self.seed = None
        self.subgroup = None
        self.train_gene_set_size = None

        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        print_system("数据路径: " + self.data_path)
    
    def set_perturbation_genes(self, gpt_embedding_path: str = None):
        """
        设置可以被扰动并包含在扰动图中的基因列表

        参数:
            gpt_embedding_path: GPT嵌入文件路径，如果为None则使用默认路径

        该方法有三种工作模式：
        1. 如果指定了gene_set_path，则使用预定义的基因集
        2. 如果default_perturbation_graph为False，则使用较小的扰动图
        3. 否则使用本地预存的完整基因集文件
        """

        # 使用传入的GPT嵌入路径或默认路径
        if gpt_embedding_path is None:
            # 构建默认路径
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            embd_path = os.path.join(base_path, 'data', 'GPT_embedding', 'GPT', 'GenePT_gene_embedding_ada_text.pickle')
        else:
            embd_path = gpt_embedding_path

        if not os.path.exists(embd_path):
            raise FileNotFoundError(f"GPT嵌入文件不存在: {embd_path}")

        with open(embd_path, 'rb') as f:
            embd = pd.DataFrame(pickle.load(f)).T
        
        gene_name_embd = embd.index.tolist()
        # uniq_conds：唯一的扰动条件 condition 的列名的统计
        uniq_conds = self.adata.obs['condition'].unique().tolist()
        
        
        # 获取所有扰动基因名
        gene_name_pert = np.unique(sum([p.split('+') for p in uniq_conds], []))

        # 找出嵌入矩阵和扰动条件中匹配的基因
        matched_genes = sorted(list(np.intersect1d(gene_name_embd, gene_name_pert)))
        
        # 精简嵌入矩阵只包含匹配基因
        embd = embd.loc[matched_genes]
        self.select_GPT_embedding = embd
        
        ## 修改 matched_genes 为 NumPy 数组
        matched_genes_array = np.array(matched_genes)
        
        # self.pert_names 的数据类型是 NumPy 数组
        self.pert_names = matched_genes_array
        
        unmatched_genes = gene_name_pert[[p not in matched_genes for p in gene_name_pert]]
        
        # 使用NumPy的逻辑运算过滤数组（不修改原始数组）
        filtered_genes = unmatched_genes[unmatched_genes != 'ctrl']

        # 如果需要更新原始数组，可以重新赋值
        self.real_unmatched_genes = filtered_genes
        
        self.node_map_pert = {x: it for it, x in enumerate(self.pert_names)}
    
    def load(self, data_name=None, data_path=None):
        """
        加载现有数据加载器
        
        参数:
            data_name: str - 数据集名称，支持 'norman', 'adamson', 'dixit', 'replogle_k562_essential', 'replogle_rpe1_essential'
            data_path: str - 自定义数据集的路径
            
        返回:
            None
        """
        # 检查参数有效性
        if data_name is None and data_path is None:
            raise ValueError("必须提供 data_name 或 data_path 参数")
        
        # 处理预定义数据集
        if data_name in ['norman', 'adamson', 'dixit', 
                        'replogle_k562_essential', 
                        'replogle_rpe1_essential']:
            
            # 设置本地数据路径
            data_path = os.path.join(self.data_path, data_name)
            
            # 检查本地数据目录是否存在
            if not os.path.exists(data_path):
                error_msg = (
                    f"本地数据集目录未找到: {data_path}\n"
                    "请手动执行以下操作：\n"
                    f"1. 下载 {data_name} 数据集\n"
                    f"2. 解压到目录: {data_path}\n"
                    f"3. 确保包含 'perturb_processed.h5ad' 文件"
                )
                raise FileNotFoundError(error_msg)
            
            # 检查h5ad文件是否存在
            adata_path = os.path.join(data_path, 'perturb_processed.h5ad')
            if not os.path.exists(adata_path):
                error_msg = (
                    f"h5ad文件未找到: {adata_path}\n"
                    "请确保数据集已正确解压，包含 'perturb_processed.h5ad' 文件"
                )
                raise FileNotFoundError(error_msg)
            
            # 加载数据
            self.dataset_name = data_path.split('/')[-1]
            self.dataset_path = data_path

            self.adata = sc.read_h5ad(adata_path)
            
            # 对条件名称进行排序处理
            self.adata.obs['condition'] = self.adata.obs['condition'].astype(str).apply(
                lambda x: condition_sort(x)).astype('category')
            
            
            # 获取 'condition' 列的唯一值
            unique_conditions = self.adata.obs['condition'].unique()
            

        # 处理自定义路径数据集
        elif os.path.exists(data_path):
            adata_path = os.path.join(data_path, 'perturb_processed.h5ad')
            
            if not os.path.exists(adata_path):
                error_msg = f"h5ad文件未找到: {adata_path}"
                raise FileNotFoundError(error_msg)
                
            self.adata = sc.read_h5ad(adata_path)
            self.dataset_name = data_path.split('/')[-1]
            self.dataset_path = data_path
        
        else:
            error_msg = (
                "数据属性必须是以下之一: norman, adamson, dixit, "
                "replogle_k562 或 replogle_rpe1 "
                "或者是h5ad文件的路径"
            )
            raise ValueError(error_msg)
        
        # 设置扰动基因
        self.set_perturbation_genes()
        

        delete = self.adata.obs['condition'].str.split('+').apply(
            lambda x: any(i in self.real_unmatched_genes for i in x))

        self.adata = self.adata[~delete].copy()

        # 处理PyG图数据
        pyg_path = os.path.join(data_path, 'data_pyg')
        
        if not os.path.exists(pyg_path):
            os.mkdir(pyg_path)
        
        dataset_fname = os.path.join(pyg_path, 'cell_graphs.pkl')
        
        if os.path.isfile(dataset_fname):
            self.dataset_processed = pickle.load(open(dataset_fname, "rb"))
        else:
            self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
            self.gene_names = self.adata.var.gene_name
            
            self.create_dataset_file()
            
            pickle.dump(self.dataset_processed, open(dataset_fname, "wb"))
        
    
    def prepare_split(self, split='simulation', seed=1, train_gene_set_size=0.75,
                     combo_seen2_train_frac=0.75, combo_single_split_test_set_fraction=0.1,
                     test_perturbations=None, only_test_set_perturbations=False,
                     test_perturbation_genes=None, split_dict_path=None):
        """
        准备训练集和测试集的划分

        参数:
            split: str - 要使用的划分类型。目前支持"simulation"（模拟）、
                "simulation_single"（单模拟）、"combo_seen0"（组合已见 0）、
                "combo_seen1"（组合已见 1）、"combo_seen2"（组合已见 2）、
                "single"（单）、"no_test"（无测试集）、"no_split"（不划分）、"custom"（自定义）
            seed: int - 随机种子
            train_gene_set_size: float - 用于训练的基因比例
            combo_seen2_train_frac: float - 组合已见 2 的扰动中用于训练的比例
            combo_single_split_test_set_fraction: float - 组合单扰动中用于测试的比例
            test_perturbations: list - 用于测试的扰动列表
            only_test_set_perturbations: bool - 如果为 True，则仅使用测试集的扰动进行测试
            test_perturbation_genes: list - 用于测试的基因列表
            split_dict_path: str - 用于自定义划分的字典路径

        返回值:
            无
        """

        # 1. 验证分割类型有效性
        available_splits = ['simulation', 'simulation_single', 'combo_seen0',
                            'combo_seen1', 'combo_seen2', 'single', 'no_test',
                            'no_split', 'custom']
        if split not in available_splits:
            error_msg = f"不支持的分割类型: {split}，当前支持: {', '.join(available_splits)}"
            raise ValueError('currently, we only support ' + ','.join(available_splits))
        
        self.split = split
        self.seed = seed
        self.subgroup = None
        

        # 2. 处理自定义分割
        if split == 'custom':
            if not split_dict_path:
                error_msg = "自定义分割需要提供 split_dict_path 参数"
                raise ValueError(error_msg)
            
            try:
                with open(split_dict_path, 'rb') as f:
                    self.set2conditions = pickle.load(f)
                return
            
            except Exception as e:
                error_msg = f"加载自定义分割失败: {str(e)}"
                raise ValueError(error_msg)

        # 3. 设置分割文件路径

        self.train_gene_set_size = train_gene_set_size
        split_folder = os.path.join(self.dataset_path, f'splits_tuning_no_perturbation_seed_{seed}')
        
        if not os.path.exists(split_folder):
            os.mkdir(split_folder)

        split_file = self.dataset_name + '_' + split + '_' + str(seed) + '_' \
                                       +  str(train_gene_set_size) + '.pkl'

        split_path = os.path.join(split_folder, split_file)

        
        if test_perturbations:
            split_path = split_path[:-4] + '_' + test_perturbations + '.pkl'
        


        # 4. 检查是否已有分割文件
        if os.path.exists(split_path):
            set2conditions = pickle.load(open(split_path, "rb"))

            if split == 'simulation':
                subgroup_path = split_path[:-4] + '_subgroup.pkl'
                if os.path.exists(subgroup_path):
                    subgroup = pickle.load(open(subgroup_path, "rb"))
                    self.subgroup = subgroup

        else:
            # 5. 创建新分割
            if test_perturbations:
                test_perturbations = test_perturbations.split('_')
                    
            if split in ['simulation', 'simulation_single']:
                
                DS = DataSplitter(self.adata, split_type=split)
                
                adata, subgroup = DS.split_data(train_gene_set_size=train_gene_set_size, 
                                                combo_seen2_train_frac=combo_seen2_train_frac,
                                                seed=seed,
                                                test_perturbations=test_perturbations,
                                                only_test_set_perturbations=only_test_set_perturbations
                                               )
                subgroup_path = split_path[:-4] + '_subgroup.pkl'
                pickle.dump(subgroup, open(subgroup_path, "wb"))
                self.subgroup = subgroup

                
            elif split[:5] == 'combo':
                split_type = 'combo'
                seen = int(split[-1])

                if test_perturbation_genes:
                    test_perturbation_genes = test_perturbation_genes.split('_')

                DS = DataSplitter(self.adata, split_type=split_type, seen=int(seen))
                adata = DS.split_data(test_size=combo_single_split_test_set_fraction,
                                      test_perturbations=test_perturbations,
                                      test_perturbation_genes=test_perturbation_genes,
                                      seed=seed)

            elif split == 'single':
                DS = DataSplitter(self.adata, split_type=split)
                adata = DS.split_data(test_size=combo_single_split_test_set_fraction,
                                      seed=seed)

            elif split == 'no_test':
                DS = DataSplitter(self.adata, split_type=split)
                adata = DS.split_data(seed=seed)
            
            elif split == 'no_split':
                adata = self.adata
                adata.obs['split'] = 'test'
            
             # 6. 保存分割结果
            set2conditions = dict(adata.obs.groupby('split').agg({'condition':
                                                        lambda x: x}).condition)
            set2conditions = {i: j.unique().tolist() for i,j in set2conditions.items()} 
            pickle.dump(set2conditions, open(split_path, "wb"))

        # 7. 设置分割结果
        self.set2conditions = set2conditions
        
    def get_dataloader(self, batch_size, test_batch_size=None):
        """
        获取训练和测试的数据加载器
        
        参数:
            batch_size: int - 训练批大小
            test_batch_size: int - 测试批大小(默认为None，即使用batch_size)
                
        返回:
            dict - 包含数据加载器的字典，键为'train_loader'、'val_loader'、'test_loader'
        """

        if test_batch_size is None:
            test_batch_size = batch_size

        self.node_map = {x: it for it, x in enumerate(self.adata.var.gene_name)}
        self.gene_names = self.adata.var.gene_name

        # Create cell graphs
        cell_graphs = {}

        if self.split == 'no_split':
            print_system("\n[no_split模式处理]")
            print_system("所有数据将作为测试集")
            i = 'test'
            cell_graphs[i] = []

            print_system(f"测试集条件数量: {len(self.set2conditions[i])}")
            print_system(f"示例条件(前3个): {self.set2conditions[i][:3]}")

            for p in self.set2conditions[i]:
                if p != 'ctrl':
                    cell_graphs[i].extend(self.dataset_processed[p])
            
            print_system(f"收集到的测试细胞图数量: {len(cell_graphs[i])}")

            print_system("Creating dataloaders....")
            print_system("\n[创建数据加载器]")
            test_loader = DataLoader(cell_graphs['test'],
                                batch_size=batch_size, shuffle=False)

            print_system(f"测试加载器创建完成，批大小: {test_batch_size}")

            print_system("\n=== 数据加载器创建完成 ===")
            print_system("Dataloaders created...")
            return {'test_loader': test_loader}
        else:
            if self.split =='no_test':
                splits = ['train','val']
            else:
                splits = ['train','val','test']
            
            for i in splits:
                cell_graphs[i] = []

                for p in self.set2conditions[i]:
                    cell_graphs[i].extend(self.dataset_processed[p])
                

            train_loader = DataLoader(cell_graphs['train'],
                                batch_size=batch_size, shuffle=True, drop_last=True)

            val_loader = DataLoader(cell_graphs['val'],
                                batch_size=batch_size, shuffle=True)

            if self.split !='no_test':
                test_loader = DataLoader(cell_graphs['test'],
                                batch_size=batch_size, shuffle=False)

                self.dataloader =  {'train_loader': train_loader,
                                    'val_loader': val_loader,
                                    'test_loader': test_loader}

            else: 
                self.dataloader =  {'train_loader': train_loader,
                                    'val_loader': val_loader}

            # 返回数据加载器字典
            return self.dataloader
    
    def get_perturbation_idx(self, perturbation_category):
        """
        获取扰动类别对应的基因索引
        
        参数:
            perturbation_category: str - 扰动类别，格式如"BRCA1+TP53"
            
        返回:
            list - 扰动基因的索引列表，如[0, 1]对应BRCA1和TP53
        """

        try:
            # 预处理扰动类别
            perturbation_parts = [p.strip() for p in perturbation_category.split('+') if p.strip() and p.strip() != 'ctrl']

            if not perturbation_parts:
                # 只有ctrl或空扰动
                return []

            # 验证每个扰动名称
            valid_perturbations = []
            for p in perturbation_parts:
                if p not in self.pert_names:
                    print_system(f"未找到扰动 '{p}'，可用扰动: {list(self.pert_names)[:10]}...")
                    return []  # 返回空列表而不是None
                valid_perturbations.append(p)

            # 获取扰动索引
            perturbation_idx = [np.where(p == self.pert_names)[0][0] for p in valid_perturbations]

            # 验证索引有效性
            if len(perturbation_idx) != len(valid_perturbations):
                print_system(f"索引匹配失败: {len(perturbation_idx)} vs {len(valid_perturbations)}")
                return []

            return perturbation_idx

        except IndexError as e:
            print_system(f"扰动索引越界错误: {perturbation_category} - {str(e)}")
            return []
        except ValueError as e:
            print_system(f"扰动值错误: {perturbation_category} - {str(e)}")
            return []
        except Exception as e:
            print_system(f"处理扰动类别时发生未预期错误:")
            print_system(f"扰动类别: {perturbation_category}")
            print_system(f"错误类型: {type(e).__name__}")
            print_system(f"错误信息: {str(e)}")
            return []  # 始终返回空列表，避免None值
    
    def create_cell_graph(self, X, y, de_idx, pert, pert_idx=None):
        """
        创建单个细胞图
        
        参数:
            X: np.ndarray - 基因表达矩阵 (对照组)
            y: np.ndarray - 目标表达矩阵 (扰动组)
            de_idx: np.ndarray - 差异表达基因索引
            pert: str - 扰动类别
            pert_idx: list - 扰动基因索引
                
        返回:
            torch_geometric.data.Data - 细胞图数据对象
        """
        feature_mat = torch.Tensor(X).T

        if pert_idx is None:
            pert_idx = [-1]

        cell_graph = Data(
            x=feature_mat,
            pert_idx=pert_idx,
            y=torch.Tensor(y),
            de_idx=de_idx,
            pert=pert
        )
        
        return cell_graph
    
    def create_cell_graph_dataset(self, split_adata, perturbation_category, num_samples=1):
        """
        为特定扰动类别创建细胞图数据集
        
        参数:
            split_adata: anndata.AnnData - 分割后的单细胞数据
            perturbation_category: str - 扰动类别
            num_samples: int - 每个扰动细胞对应的对照细胞采样数
                
        返回:
            list - 细胞图列表
        """

        num_de_genes = 20

        adata_ = split_adata[split_adata.obs['condition'] == perturbation_category]

        if 'rank_genes_groups_cov_all' in adata_.uns:
            de_genes = adata_.uns['rank_genes_groups_cov_all']
            de = True
        else:
            de = False
            num_de_genes = 1

        Xs = []
        ys = []

        if perturbation_category != 'ctrl':
            pert_idx = self.get_perturbation_idx(perturbation_category)

            perturbation_de_category = adata_.obs['condition_name'][0]

            if de:
                top_de_genes = np.array(de_genes[perturbation_de_category][:num_de_genes])
                de_idx = np.where(adata_.var_names.isin(top_de_genes))[0]
                
            else:
                de_idx = [-1] * num_de_genes

            for cell_z in adata_.X:
                ctrl_samples = self.ctrl_adata[np.random.randint(0,
                                        len(self.ctrl_adata), num_samples), :]
                for c in ctrl_samples.X:
                    Xs.append(c)
                    ys.append(cell_z)

        else:
            pert_idx = None
            de_idx = [-1] * num_de_genes
            for cell_z in adata_.X:
                Xs.append(cell_z)
                ys.append(cell_z)

        cell_graphs = []
        for X, y in zip(Xs, ys):
            cell_graphs.append(self.create_cell_graph(X.toarray(),
                                y.toarray(), de_idx, perturbation_category, pert_idx))

        return cell_graphs
    
    def create_dataset_file(self):
        """
        为所有扰动条件创建数据集文件
        """

        self.dataset_processed = {}
        for p in tqdm(self.adata.obs['condition'].unique()):
            self.dataset_processed[p] = self.create_cell_graph_dataset(self.adata, p)
            
