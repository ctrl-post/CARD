## 项目简介

精准医疗中，预测基因或治疗干预对患者的具体效应是核心挑战之一。提出 CausalKAN-Flow 框架，面向高维稀疏生物医学数据进行回归建模预测，通过**局部因果子图约束**与 **LLM 语义融合**增强表征，以 **KAN 参数化的 Rectified Flow** 构建反事实轨迹，解决精准医疗中基因或治疗干预效应难以预测的问题。


## 项目结构

```
CausalKAN-Flow/
├── README.md
├── data/                              # 数据集存放目录
└── PerturbFlow_project/               # 核心代码
    ├── main.py                        # 主入口（训练/预测/评估）
    ├── inference.py                   # 推理模块
    ├── config/
    │   └── config.py                  # 配置管理（DataConfig, ModelConfig, TrainingConfig）
    ├── data/
    │   ├── data_loader.py             # 数据加载与 PyG 图构建
    │   └── data_splitter.py           # 数据集划分策略
    ├── models/
    │   ├── perturbation_flow_model.py # 主模型（PerturbFlowModel）
    │   ├── kan_layer.py               # KAN 基础层（B样条激活）
    │   ├── kan_reflow.py              # KAN-ReFlow 动力学映射
    │   ├── bio_adaptive_gating.py     # 生物学自适应门控
    │   ├── interaction_model.py       # 双基因扰动交互
    │   └── mlp.py                     # MLP 组件
    ├── training/
    │   ├── trainer.py                 # 训练器
    │   ├── evaluator.py               # 评估器（深度分析）
    │   └── fast_evaluator.py          # 快速评估器
    └── utils/
        ├── general_utils.py           # 通用工具
        ├── loss_utils.py              # 损失函数
        ├── network_utils.py           # 共表达网络构建
        ├── causal_network_utils.py    # 因果网络工具
        └── data_utils.py              # 数据处理工具
```

## 环境配置

### 依赖安装

```bash
conda create -n causalkan python=3.10
conda activate causalkan

# PyTorch (根据 CUDA 版本选择)
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118

# PyTorch Geometric
pip install torch-geometric

# 其他依赖
pip install scanpy anndata numpy scipy scikit-learn pandas tqdm optuna lightgbm
```

### 核心依赖

| 依赖 | 用途 |
|------|------|
| PyTorch >= 2.3.0 | 深度学习框架 |
| PyTorch Geometric | 图神经网络 |
| scanpy / anndata | 单细胞数据处理 |
| scipy | 统计计算（Pearson 相关等） |
| scikit-learn | 评估指标 |

## 快速开始

### 数据准备

将癌症扰动数据集放置在 `data/` 目录下，支持以下数据集：

- `norman` — Norman et al. 单基因/双基因扰动
- `adamson` — Adamson et al. 扰动数据
- `dixit` — Dixit et al. 扰动数据
- `replogle_k562_essential` — Replogle K562 必需基因扰动
- `replogle_rpe1_essential` — Replogle RPE1 必需基因扰动

数据集目录需包含 `perturb_processed.h5ad` 文件。

### 训练

```bash
cd PerturbFlow_project

# 默认配置训练（norman 数据集，simulation 划分）
python main.py --dataset norman --seed 1

# 自定义参数训练
python main.py --dataset norman \
    --seed 1 \
    --hidden_size 64 \
    --kan_hidden_dim 256 \
    --kan_num_layers 3 \
    --kan_grid_size 5 \
    --reflow_ode_steps 10 \
    --epochs 20 \
    --learning_rate 1e-3 \
    --batch_size 32 \
    --device cuda:0
```

### 预测

```bash
python main.py --dataset norman --mode predict \
    --model_path /path/to/saved/model \
    --perturbations BRCA1 TP53
```

### 评估

```bash
python main.py --dataset norman --mode evaluate \
    --model_path /path/to/saved/model
```

## 主要命令行参数

### 数据参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | (必填) | 数据集名称 |
| `--data_path` | `./data` | 数据存储路径 |
| `--split_type` | `simulation` | 划分类型 |
| `--seed` | `1` | 数据划分随机种子 |
| `--train_gene_set_size` | `0.75` | 训练基因集比例 |
...

