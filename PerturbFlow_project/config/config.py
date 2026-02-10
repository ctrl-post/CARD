"""
CausalKAN-Flow 项目配置管理
定义数据、模型和训练的参数配置类
"""

from dataclasses import dataclass
import torch
from typing import Optional, List, Dict, Any


@dataclass
class DataConfig:
    """数据加载和处理配置"""
    data_path: str = "./data"
    gene_set_path: Optional[str] = None
    gpt_embedding_path: Optional[str] = None  # 使用None让data_loader自动构建绝对路径
    default_perturbation_graph: bool = True
    split_type: str = "simulation"
    seed: int = 1
    train_gene_set_size: float = 0.75
    combo_seen2_train_fraction: float = 0.75
    batch_size: int = 32
    test_batch_size: int = 128
    combo_single_split_test_set_fraction: float = 0.10
    test_perturbations: Optional[List[str]] = None
    only_test_set_perturbations: bool = False
    test_perturbation_genes: Optional[List[str]] = None
    split_dict_path: Optional[str] = None


@dataclass
class ModelConfig:
    """PerturbFlow 模型配置"""
    hidden_size: int = 64
    num_gene_gnn_layers: int = 1
    decoder_hidden_size: int = 16
    num_similar_genes_co_express_graph: int = 20
    coexpression_threshold: float = 0.40
    uncertainty: bool = False
    uncertainty_reg: float = 1.00
    direction_lambda: float = 0.10
    device: str = "cuda"
    no_perturbation: bool = False
    perturbation_embedding_lambda: float = 0.20
    gpt_embedding_path: Optional[str] = None  # 使用None让data_loader自动构建绝对路径
    G_coexpress: Optional[torch.Tensor] = None
    G_coexpress_weight: Optional[torch.Tensor] = None

    # === Bio-Adaptive Gating 参数 (生物学自适应门控) ===
    use_bio_gating: bool = True  # 是否启用Bio-Adaptive Gating模块
    gating_hidden_dim: Optional[int] = None  # 门控MLP隐藏层维度（None则使用cond_dim）
    gating_dropout: float = 0.1  # 门控模块Dropout率
    gating_use_layer_norm: bool = True  # 是否使用LayerNorm

    # === KAN-ReFlow 参数 (Kolmogorov-Arnold Network + Rectified Flow) ===
    use_kan_reflow: bool = True  # 是否使用KAN-ReFlow替换RealNVP
    kan_hidden_dim: int = 256  # KAN隐藏层维度
    kan_num_layers: int = 3  # KAN网络层数（深度）
    kan_grid_size: int = 5  # KAN样条网格大小（控制样条分辨率，3-7推荐）
    kan_spline_order: int = 3  # B样条阶数（通常为3，即三次样条）
    kan_reg_weight: float = 0.001  # KAN正则化损失权重

    # === Rectified Flow 参数 ===
    reflow_ode_steps: int = 10  # 推理时ODE求解步数（5-20推荐）
    reflow_ode_solver: str = 'euler'  # ODE求解器类型 ('euler' 或 'rk4')
    reflow_train_timesteps: int = 1  # 训练时采样的时间步数（多样性）
    reflow_loss_weight: float = 1.0  # Rectified Flow速度损失权重

    # 其他模型参数
    dropout_rate: float = 0.0  # Dropout率

    # === 性能优化参数 ===
    verbose: bool = False  # 是否打印详细调试日志（训练时建议False以加速）


@dataclass
class TrainingConfig:
    """PerturbFlow 训练配置"""
    epochs: int = 20
    learning_rate: float = 0.001
    weight_decay: float = 0.00001
    use_wandb: bool = False
    wandb_project: str = "PerturbFlow"
    wandb_experiment: str = "PerturbFlow_Experiment"
    gamma: int = 2
    step_size: int = 1
    scheduler_gamma: float = 0.50

    use_ema: bool = False  # 是否使用指数移动平均 (EMA)
    ema_rate: float = 0.9999  # EMA 衰减率 (越接近1更新越慢)


@dataclass
class EvaluationConfig:
    """PerturbFlow 评估配置"""
    num_de_genes: int = 20
    num_samples: int = 300
    test_composition_analysis: bool = True
    non_zero_analysis: bool = True
    non_dropout_analysis: bool = True
    deeper_analysis: bool = True
    most_variable_genes: Optional[List[int]] = None
    de_column_prefix: str = 'rank_genes_groups_cov'


@dataclass
class ProjectConfig:
    """PerturbFlow 项目整体配置"""
    project_name: str = "PerturbFlow"
    version: str = "1.0.0"
    description: str = "Gene Perturbation Prediction using Graph Neural Networks and Flow Models"
    author: str = "PerturbFlow Team"
    license: str = "MIT"
    repo_url: str = "https://github.com/example/PerturbFlow"


def create_default_config() -> Dict[str, Any]:
    """创建默认配置字典
    
    返回:
        包含所有默认配置的字典
    """
    return {
        "data": DataConfig(),
        "model": ModelConfig(),
        "training": TrainingConfig(),
        "evaluation": EvaluationConfig(),
        "project": ProjectConfig()
    }


def validate_config(config: Dict[str, Any]) -> bool:
    """验证配置参数的有效性
    
    参数:
        config: 配置字典
        
    返回:
        配置是否有效的布尔值
    """
    # 验证数据配置
    data_config = config.get("data", DataConfig())
    if not 0 <= data_config.train_gene_set_size <= 1:
        raise ValueError("train_gene_set_size must be between 0 and 1")
    
    if not 0 <= data_config.combo_seen2_train_fraction <= 1:
        raise ValueError("combo_seen2_train_fraction must be between 0 and 1")
    
    # 验证模型配置
    model_config = config.get("model", ModelConfig())
    if model_config.hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    
    if model_config.num_gene_gnn_layers <= 0:
        raise ValueError("num_gene_gnn_layers must be positive")
    
    # 验证训练配置
    training_config = config.get("training", TrainingConfig())

    # no_perturbation 模式下允许 epochs=0（仅评估模式，恒等映射不需要训练）
    if model_config.no_perturbation:
        if training_config.epochs < 0:
            raise ValueError("epochs must be non-negative (can be 0 in no_perturbation mode)")
    else:
        if training_config.epochs <= 0:
            raise ValueError("epochs must be positive")

    if training_config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    
    return True


def load_config_from_dict(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """从字典加载配置
    
    参数:
        config_dict: 包含配置参数的字典
        
    返回:
        包含配置对象的字典
    """
    # 创建默认配置
    config = create_default_config()
    
    # 更新数据配置
    if "data" in config_dict:
        data_params = {k: v for k, v in config_dict["data"].items() 
                      if hasattr(DataConfig, k)}
        config["data"] = DataConfig(**data_params)
    
    # 更新模型配置
    if "model" in config_dict:
        model_params = {k: v for k, v in config_dict["model"].items() 
                       if hasattr(ModelConfig, k)}
        config["model"] = ModelConfig(**model_params)
    
    # 更新训练配置
    if "training" in config_dict:
        training_params = {k: v for k, v in config_dict["training"].items() 
                          if hasattr(TrainingConfig, k)}
        config["training"] = TrainingConfig(**training_params)
    
    # 更新评估配置
    if "evaluation" in config_dict:
        evaluation_params = {k: v for k, v in config_dict["evaluation"].items() 
                            if hasattr(EvaluationConfig, k)}
        config["evaluation"] = EvaluationConfig(**evaluation_params)
    
    # 更新项目配置
    if "project" in config_dict:
        project_params = {k: v for k, v in config_dict["project"].items() 
                         if hasattr(ProjectConfig, k)}
        config["project"] = ProjectConfig(**project_params)
    
    # 验证配置
    validate_config(config)
    
    return config


def save_config_to_file(config: Dict[str, Any], file_path: str) -> None:
    """将配置保存到文件
    
    参数:
        config: 配置字典
        file_path: 文件路径
    """
    import json
    from dataclasses import asdict
    
    # 将数据类转换为字典
    config_dict = {}
    for key, value in config.items():
        config_dict[key] = asdict(value)
    
    # 保存到文件
    with open(file_path, 'w') as f:
        json.dump(config_dict, f, indent=4)


def load_config_from_file(file_path: str) -> Dict[str, Any]:
    """从文件加载配置
    
    参数:
        file_path: 文件路径
        
    返回:
        包含配置对象的字典
    """
    import json
    
    with open(file_path, 'r') as f:
        config_dict = json.load(f)
    
    return load_config_from_dict(config_dict)