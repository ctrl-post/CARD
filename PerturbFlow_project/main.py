"""
CausalKAN-Flow 项目主入口点
提供命令行接口和主要工作流程
"""

import os
import sys
import torch
import random
import numpy as np
import argparse
from time import perf_counter_ns
from datetime import datetime, timedelta
from contextlib import redirect_stdout, redirect_stderr
import subprocess  # 用于检查系统时间同步状态
from utils.network_utils import get_similarity_network, GeneSimilarityNetwork # 构建共表达图
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import DataConfig, ModelConfig, TrainingConfig, create_default_config, validate_config
from data.data_loader import PerturbationDataLoader
from models.perturbation_flow_model import PerturbFlowModel
from training.trainer import PerturbFlowTrainer
from inference import PerturbFlowInference
from utils.general_utils import print_system, check_ntp_sync, PrecisionTimer, format_ns, setup_logger, \
    setup_separated_randomness, get_default_model_seed, print_seed_configuration

# 全局时间统计变量
process_start = perf_counter_ns()
start_time = datetime.now()

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="PerturbFlow - Gene Perturbation Prediction System")
    
    # 数据参数
    parser.add_argument("--data_path", type=str, default="./data", help="数据存储路径")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=['norman', 'adamson', 'dixit', 'replogle_k562_essential', 'replogle_rpe1_essential'],
                       help="要使用的数据集名称")
    parser.add_argument("--split_type", type=str, default="simulation",
                       choices=['simulation', 'simulation_single', 'combo_seen0', 'combo_seen1',
                               'combo_seen2', 'single', 'no_test', 'no_split', 'custom'],
                       help="数据分割类型")
    parser.add_argument("--seed", type=int, default=1, help="数据划分随机种子（用于训练/验证/测试集划分）")
    parser.add_argument("--model_seed", type=int, default=42, help="模型训练随机种子（固定，用于权重初始化等）")
    parser.add_argument("--gpt_embedding_path", type=str, help="GPT嵌入文件路径")
    parser.add_argument("--train_gene_set_size", type=float, default=0.75, help="训练基因集比例")
    
    # 模型参数
    parser.add_argument("--hidden_size", type=int, default=64, help="隐藏层大小")
    parser.add_argument("--num_gene_gnn_layers", type=int, default=1, help="基因GNN层数")
    parser.add_argument("--decoder_hidden_size", type=int, default=64, help="解码器隐藏层大小")
    parser.add_argument("--num_similar_genes_co_express_graph", type=int, default=20, help="共表达图相似基因数量")
    parser.add_argument("--coexpression_threshold", type=float, default=0.4, help="共表达阈值")
    parser.add_argument("--uncertainty", action="store_true", help="启用不确定性估计")
    parser.add_argument("--uncertainty_reg", type=float, default=1.0, help="不确定性正则化系数")
    parser.add_argument("--direction_lambda", type=float, default=0.1, help="方向损失权重")
    parser.add_argument("--perturbation_embedding_lambda", type=float, default=0.2, help="扰动嵌入权重")

    # === Bio-Adaptive Gating 参数 ===
    parser.add_argument("--use_bio_gating", action="store_true", default=True, help="启用Bio-Adaptive Gating (默认启用)")
    parser.add_argument("--no_bio_gating", dest="use_bio_gating", action="store_false", help="禁用Bio-Adaptive Gating")
    parser.add_argument("--gating_hidden_dim", type=int, default=None, help="Gating MLP隐藏维度 (默认None，自动设为cond_dim)")
    parser.add_argument("--gating_dropout", type=float, default=0.1, help="Gating dropout率")
    parser.add_argument("--gating_use_layer_norm", action="store_true", default=True, help="Gating使用LayerNorm (默认启用)")

    # === KAN-ReFlow 动力学模块参数 ===
    parser.add_argument("--use_kan_reflow", action="store_true", default=True, help="启用KAN-ReFlow (默认启用)")
    parser.add_argument("--kan_hidden_dim", type=int, default=256, help="KAN网络隐藏层维度")
    parser.add_argument("--kan_num_layers", type=int, default=3, help="KAN网络层数")
    parser.add_argument("--kan_grid_size", type=int, default=5, help="KAN B样条网格大小")
    parser.add_argument("--kan_spline_order", type=int, default=3, help="KAN B样条阶数 (3=三次样条)")
    parser.add_argument("--kan_reg_weight", type=float, default=0.001, help="KAN正则化权重")

    # === Rectified Flow 参数 ===
    parser.add_argument("--reflow_ode_steps", type=int, default=10, help="推理时ODE求解步数")
    parser.add_argument("--reflow_ode_solver", type=str, default="euler", choices=["euler", "rk4"], help="ODE求解器 (euler或rk4)")
    parser.add_argument("--reflow_train_timesteps", type=int, default=1, help="训练时每批次采样的时间步数")
    parser.add_argument("--reflow_loss_weight", type=float, default=1.0, help="Rectified Flow损失权重")

    # 其他模型参数
    parser.add_argument("--no_perturbation", action="store_true", help="禁用扰动模式")

    # EMA (Exponential Moving Average) 参数
    parser.add_argument("--use_ema", action="store_true", help="启用EMA (指数移动平均)")
    parser.add_argument("--ema_rate", type=float, default=0.9999, help="EMA衰减率")
    

    
    # === 性能优化参数 ===
    parser.add_argument("--verbose", action="store_true", help="打印详细调试日志（会显著降低训练速度）")

    # 因果网络构建参数 (PC算法)
    parser.add_argument("--use_causal_constraint", action="store_true", help="启用因果约束(PC算法)精炼网络")
    parser.add_argument("--causal_alpha", type=float, default=0.05, help="因果检验显著性水平")
    parser.add_argument("--causal_n_jobs", type=int, default=4, help="因果检验并行进程数")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="学习率")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--test_batch_size", type=int, default=128, help="测试批次大小")
    parser.add_argument("--weight_decay", type=float, default=0.00001, help="权重衰减")
    parser.add_argument("--dropout_rate", type=float, default=0.0, help="Dropout率")

    # 其他参数
    parser.add_argument("--mode", type=str, default="train", choices=['train', 'predict', 'evaluate'],
                       help="运行模式")
    parser.add_argument("--model_path", type=str, help="模型路径（用于预测或评估模式）")
    parser.add_argument("--perturbations", type=str, nargs='+', help="要预测的扰动列表")
    parser.add_argument("--device", type=str, default="auto",
                   help="设备选择，如 'cpu', 'cuda:0', 'cuda:1'")
    parser.add_argument("--use_wandb", action="store_true", help="使用Weights & Biases记录")
    parser.add_argument("--wandb_project", type=str, default="PerturbFlow", help="WandB项目名称")
    parser.add_argument("--wandb_name", type=str, help="WandB运行名称")
    parser.add_argument("--wandb_tags", type=str, nargs='+', help="WandB标签")
    
    return parser.parse_args()



def train_mode(config, dataset_name, args):
    """训练模式"""

    # 加载数据
    data_loader = PerturbationDataLoader(config['data'].data_path)
    data_loader.load(data_name=dataset_name)

    # 设置扰动基因，传入GPT嵌入路径
    if hasattr(config['data'], 'gpt_embedding_path'):
        data_loader.set_perturbation_genes(config['data'].gpt_embedding_path)

    data_loader.prepare_split(
        split=config['data'].split_type,
        seed=config['data'].seed
    )

    # 数据划分完成后，恢复 numpy 随机种子为 model_seed
    # 这样可以确保模型训练过程中的所有随机性都使用 model_seed
    np.random.seed(args.model_seed)
    data_loader.get_dataloader(
        batch_size=config['data'].batch_size,
        test_batch_size=config['data'].test_batch_size
    )
    
    edge_list = get_similarity_network(
        network_type='co-express',
        adata=data_loader.adata,
        threshold=config['model'].coexpression_threshold,
        k=config['model'].num_similar_genes_co_express_graph,
        data_path=config['data'].data_path,
        data_name=data_loader.dataset_name,
        split=config['data'].split_type,
        seed=config['data'].seed,
        train_gene_set_size=config['data'].train_gene_set_size,
        set2conditions=data_loader.set2conditions,
        use_causal_constraint=getattr(args, 'use_causal_constraint', False),
        causal_alpha=getattr(args, 'causal_alpha', 0.05),
        causal_n_jobs=getattr(args, 'causal_n_jobs', 4)
    )
    
    gene_list = data_loader.gene_names.tolist()
    node_map = {name: idx for idx, name in enumerate(gene_list)}
    sim_network = GeneSimilarityNetwork(edge_list, gene_list, node_map)
    
    # 更新模型配置，添加共表达图
    config['model'].G_coexpress = sim_network.edge_index
    config['model'].G_coexpress_weight = sim_network.edge_weight
    
    # 初始化模型
    model = PerturbFlowModel(
        config['model'],
        data_loader.gene_names,
        data_loader.pert_names,
        data_loader.select_GPT_embedding
    )
    
    # 初始化训练器
    trainer = PerturbFlowTrainer(
        model,
        config['training'],
        model_config=config['model'],
        device=config['model'].device
    )
    
    # 计算对照组表达均值
    ctrl_expression = torch.tensor(
        np.mean(data_loader.adata[data_loader.adata.obs.condition == 'ctrl'].X, axis=0)
    ).reshape(-1, ).to(config['model'].device)
    
    # 准备非零基因索引字典
    pert_full_id_to_pert = dict(data_loader.adata.obs[['condition_name', 'condition']].values)
    dict_filter = {pert_full_id_to_pert[i]: j for i, j in
                  data_loader.adata.uns['non_zeros_gene_idx'].items() if i in pert_full_id_to_pert}
    
    # 开始训练并计时
    with PrecisionTimer() as timer:
        trainer.train(
            data_loader.dataloader['train_loader'],
            data_loader.dataloader['val_loader'],
            data_loader.dataloader.get('test_loader', None),
            ctrl_expression=ctrl_expression,
            dict_filter=dict_filter,
            adata=data_loader.adata,
            split=config['data'].split_type,
            subgroup=getattr(data_loader, 'subgroup', None)
        )
    timer.report("训练")
    
    # 保存模型
    save_dir = f"saved_models/saved_models_{dataset_name}_seed_{config['data'].seed}"
    trainer.save_model(save_dir, data_name=dataset_name)
    


def predict_mode(config, dataset_name, model_path, perturbations, args):
    """预测模式"""    
    if not perturbations:
        print_system("错误: 预测模式需要指定扰动列表")
        return
    
    # 加载数据
    data_loader = PerturbationDataLoader(config['data'].data_path)
    data_loader.load(data_name=dataset_name)
    
    # 构建共表达图
    from utils.network_utils import get_similarity_network, GeneSimilarityNetwork
    
    edge_list = get_similarity_network(
        network_type='co-express',
        adata=data_loader.adata,
        threshold=config['model'].coexpression_threshold,
        k=config['model'].num_similar_genes_co_express_graph,
        data_path=config['data'].data_path,
        data_name=data_loader.dataset_name,
        split=config['data'].split_type,
        seed=config['data'].seed,
        train_gene_set_size=config['data'].train_gene_set_size,
        set2conditions=data_loader.set2conditions,
        use_causal_constraint=getattr(args, 'use_causal_constraint', False),
        causal_alpha=getattr(args, 'causal_alpha', 0.05),
        causal_n_jobs=getattr(args, 'causal_n_jobs', 4)
    )
    
    gene_list = data_loader.gene_names.tolist()
    node_map = {name: idx for idx, name in enumerate(gene_list)}
    sim_network = GeneSimilarityNetwork(edge_list, gene_list, node_map)
    
    # 更新模型配置，添加共表达图
    config['model'].G_coexpress = sim_network.edge_index
    config['model'].G_coexpress_weight = sim_network.edge_weight
        
    # 加载模型
    model = PerturbFlowModel(
        config['model'],
        data_loader.gene_names,
        data_loader.pert_names,
        data_loader.select_GPT_embedding
    )
    
    # 加载模型权重
    if model_path:
        model.load_state_dict(torch.load(os.path.join(model_path, 'model.pt')))
    else:
        # 尝试从默认路径加载
        default_path = f"saved_models/saved_models_{dataset_name}_seed_{config['data'].seed}"
        model.load_state_dict(torch.load(os.path.join(default_path, 'model.pt')))
    
    model = model.to(config['model'].device)
    model.eval()
    
    # 初始化推理器
    inference = PerturbFlowInference(model, config['model'], data_loader)
    
    # 准备扰动列表
    perturbation_list = []
    for pert in perturbations:
        if '+' in pert:
            perturbation_list.append(pert.split('+'))
        else:
            perturbation_list.append([pert])
    
    # 进行预测并计时
    with PrecisionTimer() as timer:
        results = inference.predict(perturbation_list)
    timer.report("预测")
    
    
    # 保存结果
    output_file = f"predictions_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    inference.save_predictions(output_file)
    
def evaluate_mode(config, dataset_name, model_path, args):
    """评估模式"""
    # 加载数据
    data_loader = PerturbationDataLoader(config['data'].data_path)
    data_loader.load(data_name=dataset_name)
    data_loader.prepare_split(
        split=config['data'].split_type,
        seed=config['data'].seed
    )

    # 数据划分完成后，恢复 numpy 随机种子为 model_seed
    # 这样可以确保评估过程中的所有随机性都使用 model_seed
    np.random.seed(args.model_seed)
    data_loader.get_dataloader(
        batch_size=config['data'].batch_size,
        test_batch_size=config['data'].test_batch_size
    )
    
    # 构建共表达图
    from utils.network_utils import get_similarity_network, GeneSimilarityNetwork
    
    edge_list = get_similarity_network(
        network_type='co-express',
        adata=data_loader.adata,
        threshold=config['model'].coexpression_threshold,
        k=config['model'].num_similar_genes_co_express_graph,
        data_path=config['data'].data_path,
        data_name=data_loader.dataset_name,
        split=config['data'].split_type,
        seed=config['data'].seed,
        train_gene_set_size=config['data'].train_gene_set_size,
        set2conditions=data_loader.set2conditions,
        use_causal_constraint=getattr(args, 'use_causal_constraint', False),
        causal_alpha=getattr(args, 'causal_alpha', 0.05),
        causal_n_jobs=getattr(args, 'causal_n_jobs', 4)
    )
    
    gene_list = data_loader.gene_names.tolist()
    node_map = {name: idx for idx, name in enumerate(gene_list)}
    sim_network = GeneSimilarityNetwork(edge_list, gene_list, node_map)
    
    # 更新模型配置，添加共表达图
    config['model'].G_coexpress = sim_network.edge_index
    config['model'].G_coexpress_weight = sim_network.edge_weight
    
    
    # 加载模型
    model = PerturbFlowModel(
        config['model'],
        data_loader.gene_names,
        data_loader.pert_names,
        data_loader.select_GPT_embedding
    )
    
    # 加载模型权重
    if model_path:
        model.load_state_dict(torch.load(os.path.join(model_path, 'model.pt')))
    else:
        # 尝试从默认路径加载
        default_path = f"saved_models/saved_models_{dataset_name}_seed_{config['data'].seed}"
        model.load_state_dict(torch.load(os.path.join(default_path, 'model.pt')))
    
    model = model.to(config['model'].device)
    model.eval()
    
    # 初始化训练器（用于评估）
    trainer = PerturbFlowTrainer(
        model,
        config['training'],
        model_config=config['model'],
        device=config['model'].device
    )
    
    # 进行测试并计时
    with PrecisionTimer() as timer:
        test_metrics = trainer.test(data_loader.dataloader['test_loader'])
    timer.report("测试")
    
    

def main():
    args = parse_arguments()
    
    # 设置日志记录
    setup_logger(args.dataset, args.seed)

    try:
        # 系统时间检查
        check_ntp_sync()
        
        # 运行主程序
        run_main_logic(args)
    finally:
        try:
            from utils.general_utils import LOG_FILE
            if LOG_FILE and os.path.exists(LOG_FILE):
                print_system(f"📝 所有日志已保存到: {LOG_FILE}")
        except:
            pass

def run_main_logic(args):
    """主要运行逻辑"""
    try:
        # 系统时间检查
        check_ntp_sync()
        
        # 保存开始时间用于统计
        from datetime import datetime
        start_time = datetime.now()

        # 创建默认配置
        config = create_default_config()
        
        # 更新配置
        config['data'].data_path = args.data_path
        config['data'].split_type = args.split_type
        config['data'].seed = args.seed
        config['data'].batch_size = args.batch_size
        config['data'].test_batch_size = args.test_batch_size
        config['data'].train_gene_set_size = args.train_gene_set_size
        config['data'].gpt_embedding_path = args.gpt_embedding_path if hasattr(args, 'gpt_embedding_path') and args.gpt_embedding_path else config['data'].gpt_embedding_path

        config['model'].hidden_size = args.hidden_size
        config['model'].num_gene_gnn_layers = args.num_gene_gnn_layers
        config['model'].decoder_hidden_size = args.decoder_hidden_size
        config['model'].uncertainty = args.uncertainty
        config['model'].num_similar_genes_co_express_graph = args.num_similar_genes_co_express_graph
        config['model'].coexpression_threshold = args.coexpression_threshold
        config['model'].uncertainty_reg = args.uncertainty_reg if hasattr(args, 'uncertainty_reg') else config['model'].uncertainty_reg
        config['model'].direction_lambda = args.direction_lambda if hasattr(args, 'direction_lambda') else config['model'].direction_lambda
        config['model'].perturbation_embedding_lambda = args.perturbation_embedding_lambda if hasattr(args, 'perturbation_embedding_lambda') else config['model'].perturbation_embedding_lambda

        # === Bio-Adaptive Gating 参数 ===
        config['model'].use_bio_gating = args.use_bio_gating if hasattr(args, 'use_bio_gating') else config['model'].use_bio_gating
        config['model'].gating_hidden_dim = args.gating_hidden_dim if hasattr(args, 'gating_hidden_dim') else config['model'].gating_hidden_dim
        config['model'].gating_dropout = args.gating_dropout if hasattr(args, 'gating_dropout') else config['model'].gating_dropout
        config['model'].gating_use_layer_norm = args.gating_use_layer_norm if hasattr(args, 'gating_use_layer_norm') else config['model'].gating_use_layer_norm

        # === KAN-ReFlow 动力学模块参数 ===
        config['model'].use_kan_reflow = args.use_kan_reflow if hasattr(args, 'use_kan_reflow') else config['model'].use_kan_reflow
        config['model'].kan_hidden_dim = args.kan_hidden_dim if hasattr(args, 'kan_hidden_dim') else config['model'].kan_hidden_dim
        config['model'].kan_num_layers = args.kan_num_layers if hasattr(args, 'kan_num_layers') else config['model'].kan_num_layers
        config['model'].kan_grid_size = args.kan_grid_size if hasattr(args, 'kan_grid_size') else config['model'].kan_grid_size
        config['model'].kan_spline_order = args.kan_spline_order if hasattr(args, 'kan_spline_order') else config['model'].kan_spline_order
        config['model'].kan_reg_weight = args.kan_reg_weight if hasattr(args, 'kan_reg_weight') else config['model'].kan_reg_weight

        # === Rectified Flow 参数 ===
        config['model'].reflow_ode_steps = args.reflow_ode_steps if hasattr(args, 'reflow_ode_steps') else config['model'].reflow_ode_steps
        config['model'].reflow_ode_solver = args.reflow_ode_solver if hasattr(args, 'reflow_ode_solver') else config['model'].reflow_ode_solver
        config['model'].reflow_train_timesteps = args.reflow_train_timesteps if hasattr(args, 'reflow_train_timesteps') else config['model'].reflow_train_timesteps
        config['model'].reflow_loss_weight = args.reflow_loss_weight if hasattr(args, 'reflow_loss_weight') else config['model'].reflow_loss_weight

        # 其他模型参数
        config['model'].no_perturbation = args.no_perturbation if hasattr(args, 'no_perturbation') else config['model'].no_perturbation
        
        # EMA 参数
        config['training'].use_ema = args.use_ema if hasattr(args, 'use_ema') else False
        config['training'].ema_rate = args.ema_rate if hasattr(args, 'ema_rate') else 0.9999
    
    
    
        # === 性能优化参数 ===
        config['model'].verbose = args.verbose if hasattr(args, 'verbose') else config['model'].verbose

        if args.device == "auto":
            config['model'].device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            config['model'].device = args.device

        config['training'].epochs = args.epochs
        config['training'].learning_rate = args.learning_rate
        config['training'].weight_decay = args.weight_decay if hasattr(args, 'weight_decay') else config['training'].weight_decay
        config['training'].use_wandb = args.use_wandb

        # 模型参数
        config['model'].dropout_rate = args.dropout_rate

        # WandB相关配置
        if args.use_wandb:
            config['training'].wandb_project = args.wandb_project
            config['training'].wandb_name = args.wandb_name if args.wandb_name else f"{args.dataset}_training"
            config['training'].wandb_tags = args.wandb_tags if args.wandb_tags else [args.dataset, "perturbflow"]
        
        # 验证配置
        validate_config(config)
        
        # 设置分离的随机种子系统
        seed_config_obj = setup_separated_randomness(
            data_seed=args.seed,
            model_seed=args.model_seed
        )
        print_seed_configuration(seed_config_obj)
        
        # 根据模式执行相应操作
        if args.mode == "train":
            train_mode(config, args.dataset, args)
        elif args.mode == "predict":
            predict_mode(config, args.dataset, args.model_path, args.perturbations, args)
        elif args.mode == "evaluate":
            evaluate_mode(config, args.dataset, args.model_path, args)
        else:
            print_system(f"未知模式: {args.mode}")
            
        # 完成后的时间统计
        from datetime import datetime
        end_time = datetime.now()
        total_ns = perf_counter_ns() - process_start
        
    except KeyboardInterrupt:
        print_system("\n🛑 用户中断运行")
    except Exception as e:
        print_system(f"\n❌ 程序执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()