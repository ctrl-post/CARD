# CausalKAN-Flow

面向高维稀疏生物医学数据的基因扰动响应预测框架，通过**局部因果子图约束**与 **LLM 语义融合**增强表征，以 **KAN 参数化的 Rectified Flow** 构建反事实轨迹，解决精准医疗中基因或治疗干预效应难以预测的问题。

## 项目简介

精准医疗中，预测基因或治疗干预对患者细胞状态的效应是核心挑战之一。单细胞扰动数据具有高维、稀疏的特点，传统回归建模方法难以有效捕获基因间的复杂调控关系。

CausalKAN-Flow 提出了一种融合因果推断与生成模型的新框架，核心思路包括：

- **局部因果子图约束**：基于共表达网络与因果发现（PC 算法）构建扰动感知的局部子图，提供结构先验
- **LLM 语义融合**：引入 GPT 基因语义嵌入作为功能表征，弥补纯数据驱动方法的生物学知识缺失
- **KAN-ReFlow 动力学映射**：以 KAN (Kolmogorov-Arnold Network) 参数化 Rectified Flow 的速度场，通过 ODE 求解构建从对照态到扰动态的反事实轨迹


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
        ├── causal_network_utils.py    # 因果网络工具（PC算法）
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

将数据集放置在 `data/` 目录下，支持以下数据集：

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

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--hidden_size` | `64` | GNN 隐藏层维度 |
| `--kan_hidden_dim` | `256` | KAN 网络隐藏维度 |
| `--kan_num_layers` | `3` | KAN 网络层数 |
| `--kan_grid_size` | `5` | B样条网格大小（3-7 推荐） |
| `--reflow_ode_steps` | `10` | 推理时 ODE 求解步数 |
| `--reflow_ode_solver` | `euler` | ODE 求解器（`euler` / `rk4`） |
| `--use_bio_gating` | `True` | 启用 Bio-Adaptive Gating |
| `--verbose` | `False` | 打印详细调试日志 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | `20` | 训练轮数 |
| `--learning_rate` | `1e-3` | 学习率 |
| `--batch_size` | `32` | 训练批大小 |
| `--weight_decay` | `1e-5` | 权重衰减 |
| `--use_ema` | `False` | 启用指数移动平均 |
| `--use_wandb` | `False` | 启用 W&B 日志 |

## 模型架构

```
输入: 对照组细胞表达 (x) + 扰动基因索引 (pert_idx)
  │
  ├─ Gene Embedding + Co-expression GNN ──→ 结构嵌入 (X1)
  ├─ GPT Gene Embedding Projection ───────→ 功能嵌入 (X2)
  └─ Control Expression Encoder ──────────→ 状态嵌入 (X3)
  │
  ▼
  Bio-Adaptive Gating: c = gate([X1; X2; X3])
  │
  ▼
  KAN-ReFlow: z_0 → ODE求解 → z_1 (预测扰动后表达)
  │
  ▼
输出: 预测的扰动后基因表达
```

## 评估指标

| 指标 | 说明 |
|------|------|
| MSE | 全基因表达均方误差 |
| MSE (DE) | Top 20 差异表达基因的 MSE |
| Pearson | 全基因 Pearson 相关系数 |
| Pearson (DE) | Top 20 差异表达基因的 Pearson 相关 |
| Pearson Delta | 相对对照组变化量的 Pearson 相关 |
| Direction Accuracy | 基因表达变化方向预测准确率 |
