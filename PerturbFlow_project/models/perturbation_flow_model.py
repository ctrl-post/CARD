"""
CausalKAN-Flow 主模型模块
定义基因扰动预测的主神经网络模型
"""

import torch
import torch.nn as nn
from torch_geometric.nn import SGConv, global_mean_pool
from typing import Dict, Any, NamedTuple, Union, List

from .interaction_model import DualPerturbationInteraction
from .mlp import MLP
from .bio_adaptive_gating import BioAdaptiveGating
from .kan_reflow import KANReFlow
from utils.general_utils import print_system
from config.config import ModelConfig  # 正确的路径

class ModelOutput(NamedTuple):
    """模型输出的统一接口"""
    pred: torch.Tensor
    loss: Union[torch.Tensor, None] = None
    loss_components: Union[Dict[str, Any], None] = None

class PerturbFlowModel(nn.Module):
    """PerturbFlow 主模型 - 基因扰动预测神经网络"""
    
    def __init__(self, config: ModelConfig, gene_names: List[str], perturbation_names: List[str], select_GPT_embedding):  # 新增select_GPT_embedding参数
        """
        初始化 PerturbFlow 模型
        
        参数:
            config: ModelConfig 配置对象
            gene_names: 基因名称列表
            perturbation_names: 扰动名称列表
            select_GPT_embedding: GPT嵌入矩阵
        """
        super().__init__()
        
        # 保存配置参数
        self.config = config
        
        # 设置模型参数 - 使用点号访问属性
        self.num_genes = len(gene_names)
        self.num_perturbations = len(perturbation_names)
        self.hidden_size = config.hidden_size  # 修改为实例属性
        self.uncertainty = config.uncertainty  # 修改这里
        self.indv_out_hidden_size = config.decoder_hidden_size  # 修改这里
        self.num_layers_gene_pos = config.num_gene_gnn_layers  # 修改这里
        self.no_perturbation = config.no_perturbation  # 修改这里
        self.perturbation_embedding_lambda = 0.2
        
        # 创建设备管理方法确保张量在正确设备上
        self.select_GPT_embedding_tensor = torch.tensor(
            select_GPT_embedding.values, dtype=torch.float32)
        self._ensure_device_consistency(config.device)
        
        # 初始化模型组件
        self._init_embedding_layers(self.hidden_size)
        self._init_gnn_layers(self.hidden_size)
        self._init_flow_model()
        self._init_interaction_layers()

        # 批量归一化层
        self.bn_emb = nn.BatchNorm1d(self.hidden_size)

        # 计算条件向量的归一化维度
        cond_bn_dim = 3 * self.hidden_size

        self.bn_cond = nn.BatchNorm1d(cond_bn_dim)

        # 添加一个标志位表示是否返回预测值
        self.return_pred = False
    
    def _init_embedding_layers(self, hidden_size: int) -> None:
        """初始化嵌入层"""

        self.gene_emb = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.perturbation_emb = nn.Embedding(self.num_perturbations, hidden_size, max_norm=True)
        self.emb_pos = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        
        # 转换层
        self.emb_trans = nn.ReLU()
        self.perturbation_base_trans = nn.ReLU()
        self.transform = nn.ReLU()
        self.emb_trans_v2 = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        self.perturbation_fuse = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
    
    def _init_gnn_layers(self, hidden_size: int) -> None:
        """初始化基因共表达GNN层"""
        
        # 使用配置的共表达图
        self.G_coexpress = self.config.G_coexpress.to(self.config.device)
        self.G_coexpress_weight = self.config.G_coexpress_weight.to(self.config.device)
        
        self.layers_emb_pos = nn.ModuleList()
        for i in range(1, self.num_layers_gene_pos + 1):
            self.layers_emb_pos.append(SGConv(hidden_size, hidden_size, 1))
    
    def _init_flow_model(self) -> None:
        """初始化流模型（KAN-ReFlow架构）"""

        # GPT嵌入投影层
        self.gpt_proj = nn.Linear(1536, self.hidden_size)

        # 对照表达编码器（细胞状态嵌入）
        self.control_encoder = MLP([self.num_genes, 128, self.hidden_size], last_layer_act='linear')

        # 计算条件维度
        # 基础条件维度: GNN全局嵌入(hidden_size) + GPT扰动嵌入(hidden_size) + 对照表达(hidden_size) = 3 * hidden_size
        total_cond_dim = 3 * self.hidden_size

        # Bio-Adaptive Gating模块
        use_bio_gating = getattr(self.config, 'use_bio_gating', True)
        if use_bio_gating:
            gating_hidden_dim = getattr(self.config, 'gating_hidden_dim', total_cond_dim)
            self.bio_gating = BioAdaptiveGating(
                cond_dim=total_cond_dim,
                hidden_dim=gating_hidden_dim,
                dropout=getattr(self.config, 'dropout_rate', 0.1),
                use_layer_norm=True,
                verbose=getattr(self.config, 'verbose', False)  # 添加verbose参数
            )
        else:
            self.bio_gating = None

        # 初始化 KAN-ReFlow 流模型
        kan_hidden_dim = getattr(self.config, 'kan_hidden_dim', 256)
        kan_num_layers = getattr(self.config, 'kan_num_layers', 3)
        kan_grid_size = getattr(self.config, 'kan_grid_size', 5)
        reflow_ode_steps = getattr(self.config, 'reflow_ode_steps', 10)
        reflow_ode_solver = getattr(self.config, 'reflow_ode_solver', 'euler')

        self.flow_model = KANReFlow(
            input_dim=self.num_genes,
            cond_dim=total_cond_dim,
            hidden_dim=kan_hidden_dim,
            num_layers=kan_num_layers,
            kan_grid_size=kan_grid_size,
            num_ode_steps=reflow_ode_steps,
            ode_solver=reflow_ode_solver,
            dropout=getattr(self.config, 'dropout_rate', 0.1),
            verbose=getattr(self.config, 'verbose', False)
        )

    def _init_interaction_layers(self) -> None:
        """初始化基因交互层"""

        # ✅ 使用 DualPerturbationInteraction 处理双基因扰动交互
        # 在初始化时创建，而不是在前向传播中重复创建
        self.dual_perturbation_interaction = DualPerturbationInteraction(hidden_dim=self.hidden_size)


    def _log(self, message: str, force: bool = False) -> None:
        """
        条件日志输出辅助方法

        参数:
            message: 要输出的消息
            force: 是否强制输出（忽略verbose设置）
        """
        if force or getattr(self.config, 'verbose', False):
            print_system(message)

    def _ensure_device_consistency(self, device: str) -> None:
        """确保所有张量在正确的设备上"""

        # 移动GPT嵌入张量
        if hasattr(self, 'select_GPT_embedding_tensor'):
            self.select_GPT_embedding_tensor = self.select_GPT_embedding_tensor.to(device)

        # 移动图数据
        if hasattr(self.config, 'G_coexpress') and self.config.G_coexpress is not None:
            self.config.G_coexpress = self.config.G_coexpress.to(device)
            if hasattr(self.config, 'G_coexpress_weight') and self.config.G_coexpress_weight is not None:
                self.config.G_coexpress_weight = self.config.G_coexpress_weight.to(device)

    def to(self, device: str):
        """重写to方法以确保设备一致性"""
        super().to(device)
        self._ensure_device_consistency(device)
        return self

    def forward_legacy(self, data, return_loss_components=False):
        """兼容性方法：保持原有的返回格式

        为了向后兼容，提供原有的返回格式：
        - 训练模式: (loss, pred, loss_components) 或 (loss, pred)
        - 评估模式: pred 或 (pred, loss_components)
        """
        output = self.forward(data, return_loss_components)

        if self.training:
            # 训练模式：保持原有的元组格式
            if output.loss_components is not None:
                return output.loss, output.pred, output.loss_components
            else:
                return output.loss, output.pred
        else:
            # 评估模式：保持原有的返回格式
            if return_loss_components and output.loss_components is not None:
                return output.pred, output.loss_components
            else:
                return output.pred

    def forward(self, data, return_loss_components=False):
        """前向传播 - ✅ 使用统一输出接口

        参数:
            data: 输入数据
            return_loss_components: 是否在评估模式下也返回损失分量

        返回:
            ModelOutput: 统一的输出接口
                - pred: 预测值
                - loss: 损失值（训练模式下存在）
                - loss_components: 损失分量（可选）
        """
        # 只包含扰动基因的索引，不包含 ctrl
        x, perturbation_idx = data.x, data.pert_idx

        # 获取verbose设置
        verbose = getattr(self.config, 'verbose', False)

        # 无扰动模式处理
        if self.no_perturbation:
            # no perturb模式：直接返回输入作为预测，避免复杂计算
            out = x.reshape(-1, 1)
            out = torch.split(torch.flatten(out), self.num_genes)           
            pred = torch.stack(out)
            
            if self.training:
                # 训练模式：计算损失并返回统一接口
                target = data.y.reshape(-1, self.num_genes)
                loss = nn.functional.mse_loss(pred, target)
                return ModelOutput(pred=pred, loss=loss, loss_components=None)
            else:
                # 评估模式：只返回预测结果
                return ModelOutput(pred=pred, loss=None, loss_components=None) 
        else:
            num_graphs = len(data.batch.unique())
            self._log(f"批次大小: {num_graphs}, 基因数量: {self.num_genes}")
            
            # 1. 基因基础嵌入
            gene_indices = torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.config.device)  # 修改这里
            emb = self.gene_emb(gene_indices)        
            emb = self.bn_emb(emb)
            base_emb = self.emb_trans(emb)   
            self._log(f"基础基因嵌入形状: {base_emb.shape}")
            
            # 2. 共表达GNN嵌入
            pos_emb = self.emb_pos(gene_indices)
            for idx, layer in enumerate(self.layers_emb_pos):
                pos_emb = layer(pos_emb, self.G_coexpress, self.G_coexpress_weight)
                if idx < len(self.layers_emb_pos) - 1:
                    pos_emb = nn.functional.relu(pos_emb)
            
            # 3. 融合基础嵌入和位置嵌入
            base_emb = base_emb + 0.2 * pos_emb
            base_emb = self.emb_trans_v2(base_emb)
            self._log(f"增强后基因嵌入形状: {base_emb.shape}")
            
            # 智能扰动感知池化
            self._log("\n=== 开始智能扰动感知图池化 ===")
            self._log("[智能池化] 需要先构建扰动索引...")

            # 处理扰动索引
            perturbation_index = []
            for idx, i in enumerate(perturbation_idx):
                for j in i:
                    if j != -1:  # j != -1，也就是 ctrl 的不加进去
                        perturbation_index.append([idx, j])

            if perturbation_index:
                perturbation_index = torch.tensor(perturbation_index).T
            else:
                perturbation_index = torch.empty((2, 0), dtype=torch.long).to(self.config.device)

            self._log(f"perturbation_index 构建完成. 形状: {perturbation_index.shape}")

            base_emb_reshaped = base_emb.reshape(num_graphs, self.num_genes, -1)
            self._log(f"池化输入形状: {base_emb_reshaped.shape}")

            graph_embed = self._get_smart_graph_embedding(base_emb_reshaped, perturbation_index, self.G_coexpress)
            self._log(f"智能池化后形状: {graph_embed.shape}")
            self._log(f"perturbation_index 的形状: {perturbation_index.shape}")
            
            # 扰动嵌入处理
            perturbation_track = {}
            self._log("\n=== 开始扰动嵌入处理 ===")
            
            if perturbation_index.shape[1] != 0:  # 检查是否有扰动样本
                self._log(f"\n发现{perturbation_index.shape[1]}个扰动样本")
                self._log(f"原始GPT嵌入形状: {self.select_GPT_embedding_tensor.shape}")
                
                gpt_compressed = self.gpt_proj(self.select_GPT_embedding_tensor)
                self._log(f"压缩后基因嵌入形状: {gpt_compressed.shape}")
                
                for i, j in enumerate(perturbation_index[0]):
                    sample_idx = j.item()
                    gene_idx = perturbation_index[1][i].item()
                    self._log(f"\n处理第{i}个扰动: 样本{sample_idx}的基因{gene_idx}")
                    
                    if sample_idx in perturbation_track:
                        self._log(f"样本{sample_idx}已有扰动 -> 触发双扰动处理")

                        prev_gene = perturbation_track[sample_idx]
                        self._log(f"前一个基因嵌入shape: {prev_gene.shape}")
                        self._log(f"当前基因嵌入shape: {gpt_compressed[gene_idx].shape}")

                        # 使用预先初始化的 DualPerturbationInteraction 模型
                        # 直接传入两个基因嵌入，无需拼接
                        interaction_emb = self.dual_perturbation_interaction(prev_gene, gpt_compressed[gene_idx])
                        perturbation_track[sample_idx] = interaction_emb
                        self._log(f"双基因交互后嵌入shape: {interaction_emb.shape}")
                    else:
                        self._log(f"样本{sample_idx}首次扰动 -> 存储初始嵌入")
                        perturbation_track[sample_idx] = gpt_compressed[gene_idx]
                        self._log(f"存储的嵌入shape: {perturbation_track[sample_idx].shape}")
                        
                self._log("\n=== 处理完成 ===")
                self._log(f"最终perturbation_track包含样本: {list(perturbation_track.keys())}")
                
                # 为未出现的样本添加零向量
                embedding_dim = gpt_compressed.shape[1]
                for sample_idx in range(num_graphs):
                    if sample_idx not in perturbation_track:
                        perturbation_track[sample_idx] = torch.zeros(embedding_dim, device=self.config.device)  # 修改这里
                
                # 将字典转换为列表，确保顺序正确
                perturbation_cond = torch.stack([perturbation_track[i] for i in range(num_graphs)])
            else:
                # 若没有扰动样本，直接创建全零的perturbation_cond
                embedding_dim = self.gpt_proj.out_features
                perturbation_cond = torch.zeros(num_graphs, embedding_dim, device=self.config.device)  # 修改这里
            
            self._log(f"perturbation_cond 的形状: {perturbation_cond.shape}")
            
            # 6. 处理对照表达
            control_expr = data.x.reshape(num_graphs, self.num_genes)
            self._log(f"[对照表达] 输入形状: {control_expr.shape}")
            
            control_cond = self.control_encoder(control_expr)
            self._log(f"[对照编码] 输出形状: {control_cond.shape}")

            # 7. 组合条件向量（构建c_raw）
            # c_raw = [X1_结构嵌入(graph_embed), X2_功能嵌入(perturbation_cond), X3_状态嵌入(control_cond)]
            self._log(f"[条件拼接] X1结构({graph_embed.shape}) + X2功能({perturbation_cond.shape}) + X3状态({control_cond.shape})")
            cond_raw = torch.cat([graph_embed, perturbation_cond, control_cond], dim=1)

            # 7.5. Bio-Adaptive Gating（生物学自适应门控）
            if self.bio_gating is not None:
                self._log("\n[Bio-Adaptive Gating] 应用门控调节")
                cond = self.bio_gating(cond_raw)
                self._log(f"  门控后条件向量形状: {cond.shape}")
            else:
                cond = cond_raw

            # 归一化
            cond = self.bn_cond(cond)

            # 8. KAN-ReFlow 流模型
            target = data.y.reshape(num_graphs, self.num_genes)
            z_0 = control_expr  # 初始状态（未扰动的对照表达）

            # KAN-ReFlow动力学映射
            self._log("\n[KAN-ReFlow动力学映射]")

            if self.training:
                # 训练模式：计算速度场损失
                pred, rf_loss = self.flow_model(z_0, cond, y_true=target, mode='train')
                self._log(f"  Rectified Flow损失: {rf_loss.item():.6f}")
                self._log(f"  Flow预测形状: {pred.shape}")
            else:
                # 推理模式：ODE求解
                pred, _ = self.flow_model(z_0, cond, mode='inference')
                rf_loss = None
                self._log(f"  Flow预测形状: {pred.shape}")

            if self.training:
                # 训练模式：计算损失
                self._log("\n[KAN-ReFlow损失计算]")

                # 1. 速度场损失（主损失）
                if rf_loss is None:
                    # 防御性编程：如果rf_loss未计算，重新计算
                    _, rf_loss = self.flow_model(z_0, cond, y_true=target, mode='train')

                # 2. 辅助损失：MSE重建损失
                recon_loss = nn.functional.mse_loss(pred, target)

                # 3. 计算多任务损失
                total_loss, loss_components = self.compute_improved_loss(
                    pred, target,
                    nll_loss=rf_loss,  # 用rf_loss替代nll_loss作为主损失
                    recon_loss=recon_loss
                )

                # 4. 添加KAN正则化损失
                kan_reg_weight = getattr(self.config, 'kan_reg_weight', 0.001)
                if hasattr(self.flow_model, 'get_regularization_loss'):
                    kan_reg_loss = self.flow_model.get_regularization_loss()
                    total_loss = total_loss + kan_reg_weight * kan_reg_loss
                    loss_components['kan_reg_loss'] = kan_reg_loss
                    loss_components['kan_reg_weight'] = kan_reg_weight
                    self._log(f"  KAN正则化损失: {kan_reg_loss.item():.6f} (权重: {kan_reg_weight})")

                # 使用统一输出接口
                return ModelOutput(pred=pred, loss=total_loss, loss_components=loss_components)
            else:
                # 评估模式
                if return_loss_components:
                    # 评估模式下计算损失分量（用于验证损失曲线）
                    with torch.no_grad():
                        # KAN-ReFlow评估损失
                        _, rf_loss = self.flow_model(z_0, cond, y_true=target, mode='train')
                        nll_loss = rf_loss  # 使用rf_loss作为nll_loss

                        # 计算重建损失
                        recon_loss = nn.functional.mse_loss(pred, target)

                        # 计算损失分量
                        total_loss, loss_components = self.compute_improved_loss(pred, target, nll_loss, recon_loss)

                    return ModelOutput(pred=pred, loss=total_loss, loss_components=loss_components)
                else:
                    # 普通评估模式：只返回预测结果
                    return ModelOutput(pred=pred, loss=None, loss_components=None)

    def _get_smart_graph_embedding(self, gene_embeddings: torch.Tensor,
                                  perturbation_index: torch.Tensor,
                                  edge_index: torch.Tensor) -> torch.Tensor:
        """
        智能扰动感知图嵌入池化函数

        参数:
            gene_embeddings: 基因嵌入矩阵 [num_graphs, num_genes, hidden_dim]
            perturbation_index: 扰动索引张量 [2, num_perturbations] - 样本索引和基因索引
            edge_index: 图边索引 [num_edges, 2] - 用于查找邻居

        返回:
            smart_graph_embed: 智能池化后的图嵌入 [num_graphs, hidden_dim]
        """
        num_graphs, num_genes, hidden_dim = gene_embeddings.shape
        device = gene_embeddings.device

        self._log("[智能池化] 开始扰动感知图嵌入处理")
        self._log(f"[智能池化] 输入形状: {gene_embeddings.shape}")
        self._log(f"[智能池化] 扰动索引形状: {perturbation_index.shape}")
        self._log(f"[智能池化] 图边数量: {edge_index.size(1)}")

        if perturbation_index.shape[1] == 0:
            # 情况1: 无扰动基因 - 使用全局池化
            self._log("[智能池化] 检测到无扰动样本 -> 使用全局池化")

            batch_index = torch.arange(num_graphs, device=device).repeat_interleave(num_genes)
            base_emb_flat = gene_embeddings.view(-1, hidden_dim)
            smart_graph_embed = global_mean_pool(base_emb_flat, batch_index)
            smart_graph_embed = smart_graph_embed.view(num_graphs, -1)

        else:
            # 情况2: 有扰动基因 - 使用Local图池化
            self._log("[智能池化] 检测到扰动样本 -> 使用Local邻居池化")

            smart_graph_embed = torch.zeros(num_graphs, hidden_dim, device=device)

            # 为每个样本构建扰动基因索引列表
            sample_perturb_genes = {i: [] for i in range(num_graphs)}
            for i in range(perturbation_index.shape[1]):
                sample_idx = perturbation_index[0, i].item()
                gene_idx = perturbation_index[1, i].item()

                # 确保sample_idx在有效范围内
                if 0 <= sample_idx < num_graphs:
                    sample_perturb_genes[sample_idx].append(gene_idx)

            self._log(f"[智能池化] 每个样本的扰动基因分布: {[len(v) for v in sample_perturb_genes.values()]}")

            # 为每个有扰动的样本构建局部子图
            for sample_idx in range(num_graphs):
                if sample_idx in sample_perturb_genes and sample_perturb_genes[sample_idx]:
                    # 这个样本有扰动 - 构建局部图嵌入
                    perturb_genes = sample_perturb_genes[sample_idx]

                    self._log(f"[智能池化] 处理样本{sample_idx}: 扰动基因{perturb_genes}")

                    # 根据扰动基因数量选择处理策略
                    if len(perturb_genes) == 2:
                        # 双基因扰动：分别计算两个基因的邻居局部嵌入，然后融合
                        gene1_idx, gene2_idx = perturb_genes[0], perturb_genes[1]
                        self._log(f"[智能池化] 双基因扰动 -> 分别计算基因{gene1_idx}和基因{gene2_idx}的局部嵌入后融合")

                        # 计算基因1的局部嵌入
                        local_nodes_1 = {gene1_idx}
                        edge_src, edge_dst = edge_index[0], edge_index[1]
                        gene_neighbors = torch.where(edge_src == gene1_idx)[0]
                        for neighbor_edge_idx in gene_neighbors:
                            local_nodes_1.add(edge_dst[neighbor_edge_idx].item())
                        gene_neighbors = torch.where(edge_dst == gene1_idx)[0]
                        for neighbor_edge_idx in gene_neighbors:
                            local_nodes_1.add(edge_src[neighbor_edge_idx].item())
                        local_nodes_1 = sorted(list(local_nodes_1))
                        local_embeddings_1 = gene_embeddings[sample_idx, local_nodes_1, :]
                        attention_weights_1 = torch.ones(len(local_nodes_1), device=device)
                        attention_weights_1[local_nodes_1.index(gene1_idx)] = 2.0
                        attention_weights_1 = attention_weights_1 / attention_weights_1.sum()
                        weighted_embed_1 = (local_embeddings_1 * attention_weights_1.unsqueeze(-1)).sum(dim=0)

                        # 计算基因2的局部嵌入
                        local_nodes_2 = {gene2_idx}
                        gene_neighbors = torch.where(edge_src == gene2_idx)[0]
                        for neighbor_edge_idx in gene_neighbors:
                            local_nodes_2.add(edge_dst[neighbor_edge_idx].item())
                        gene_neighbors = torch.where(edge_dst == gene2_idx)[0]
                        for neighbor_edge_idx in gene_neighbors:
                            local_nodes_2.add(edge_src[neighbor_edge_idx].item())
                        local_nodes_2 = sorted(list(local_nodes_2))
                        local_embeddings_2 = gene_embeddings[sample_idx, local_nodes_2, :]
                        attention_weights_2 = torch.ones(len(local_nodes_2), device=device)
                        attention_weights_2[local_nodes_2.index(gene2_idx)] = 2.0
                        attention_weights_2 = attention_weights_2 / attention_weights_2.sum()
                        weighted_embed_2 = (local_embeddings_2 * attention_weights_2.unsqueeze(-1)).sum(dim=0)

                        # 融合两个嵌入
                        smart_graph_embed[sample_idx] = (weighted_embed_1 + weighted_embed_2) / 2
                    else:
                        # 单基因或多基因：使用原有逻辑
                        # 获取扰动基因的一阶邻居
                        local_nodes = set(perturb_genes)  # 先包含扰动基因本身

                        # 从edge_index中找到这些基因的邻居
                        edge_src, edge_dst = edge_index[0], edge_index[1]
                        for gene_idx in perturb_genes:
                            # 找到与该基因相连的邻居
                            gene_neighbors = torch.where(edge_src == gene_idx)[0]
                            for neighbor_edge_idx in gene_neighbors:
                                neighbor_idx = edge_dst[neighbor_edge_idx].item()
                                local_nodes.add(neighbor_idx)

                            # 反向边（因为是无向图）
                            gene_neighbors = torch.where(edge_dst == gene_idx)[0]
                            for neighbor_edge_idx in gene_neighbors:
                                neighbor_idx = edge_src[neighbor_edge_idx].item()
                                local_nodes.add(neighbor_idx)

                        local_nodes = sorted(list(local_nodes))
                        self._log(f"[智能池化] 样本{sample_idx}局部子图包含节点: {local_nodes} (数量: {len(local_nodes)})")

                        # 提取local gene embeddings
                        local_embeddings = gene_embeddings[sample_idx, local_nodes, :]

                        # 对局部子图进行注意力加权池化 - 扰动基因权重更高
                        attention_weights = torch.ones(len(local_nodes), device=device)

                        # 增强扰动基因的注意力权重
                        for i, node_idx in enumerate(local_nodes):
                            if node_idx in perturb_genes:
                                attention_weights[i] = 2.0  # 扰动基因权重设为2倍

                        attention_weights = attention_weights / attention_weights.sum()

                        # 加权池化
                        weighted_embed = (local_embeddings * attention_weights.unsqueeze(-1)).sum(dim=0)
                        smart_graph_embed[sample_idx] = weighted_embed

                else:
                    # 这个样本无扰动 - 保持全局表达特征
                    self._log(f"[智能池化] 样本{sample_idx}无扰动 -> 使用基因级平均池化")
                    smart_graph_embed[sample_idx] = gene_embeddings[sample_idx].mean(dim=0)

        self._log(f"[智能池化] 输出形状验证: {smart_graph_embed.shape}")
        self._log(f"[智能池化] 完成！包含{num_graphs}个样本的智能池化")

        return smart_graph_embed

    def compute_improved_loss(self, pred, target, nll_loss, recon_loss):
        """
        改进的损失函数，更好地对齐验证指标（mse_de和pearson）

        参数:
            pred: 预测值 [batch_size, num_genes]
            target: 真实值 [batch_size, num_genes]
            nll_loss: 实际为速度场损失
            recon_loss: 重建损失

        返回:
            total_loss: 总损失
            loss_components: 损失分量字典
        """

        # 1. 直接使用标准NLL损失，避免梯度不连续
        adjusted_nll = nll_loss

        # 2. 基于生物学意义选择Top20 DE基因
        # 使用基因表达的变异性作为生物学重要性的指标
        # 这比基于预测误差更合理，符合生物学研究标准
        gene_variability = target.std(dim=0, unbiased=False)  # 基因在批次中的表达变异性

        # 选择变异性最大的20个基因作为重要的生物学基因
        top_k = min(20, gene_variability.size(0))
        top20_indices = torch.topk(gene_variability, top_k).indices
        top20_mse = nn.functional.mse_loss(pred[:, top20_indices], target[:, top20_indices])

        # 3. 增强数值稳定性的Pearson相关系数计算
        pred_flat = pred.flatten()
        target_flat = target.flatten()

        # 更严格的数值稳定性检查
        pred_std = pred_flat.std(unbiased=False)
        target_std = target_flat.std(unbiased=False)

        # 检查数值边界条件
        numerical_stability_threshold = 1e-6
        has_valid_variance = (
            pred_std > numerical_stability_threshold and
            target_std > numerical_stability_threshold and
            not torch.isnan(pred_std) and not torch.isnan(target_std) and
            not torch.isinf(pred_std) and not torch.isinf(target_std)
        )

        if has_valid_variance:
            try:
                # 使用更稳定的数值计算方法
                pred_centered = pred_flat - pred_flat.mean()
                target_centered = target_flat - target_flat.mean()

                # 计算协方差
                covariance = (pred_centered * target_centered).mean()

                # 计算相关系数，添加小常数避免除零
                correlation = covariance / (pred_std * target_std + 1e-8)

                # 确保相关系数在有效范围内 [-1, 1]
                correlation = torch.clamp(correlation, -1.0, 1.0)

                # 再次检查结果的有效性
                if torch.isnan(correlation) or torch.isinf(correlation):
                    correlation = torch.tensor(0.0, device=pred.device)

            except Exception as e:
                correlation = torch.tensor(0.0, device=pred.device)
        else:
            correlation = torch.tensor(0.0, device=pred.device)

        # 4. 设置多任务损失权重（可调参数）
        lambda_nll = 0.1        # NLL损失权重（较小，因为我们主要关心重建）
        lambda_mse = 0.5        # 全MSE权重
        lambda_de_mse = 2.0      # Top20 DE MSE权重（最重要）
        lambda_pearson = -0.1    # Pearson相关系数权重（负号，因为我们要最大化相关系数）

        # 5. 计算多任务总损失
        total_loss = (
            lambda_nll * adjusted_nll +
            lambda_mse * recon_loss +
            lambda_de_mse * top20_mse +
            lambda_pearson * correlation
        )

        # 确保损失是标量
        self._log(f"[损失调试] total_loss形状: {total_loss.shape}, 类型: {type(total_loss)}")
        if total_loss.numel() > 1:
            self._log(f"[损失调试] 损失不是标量，取平均值")
            total_loss = total_loss.mean()

        # 6. 创建详细的损失监控
        loss_components = {
            'total_loss': total_loss,
            'nll_loss': nll_loss,
            'adjusted_nll': adjusted_nll,
            'full_mse': recon_loss,
            'top20_mse': top20_mse,
            'correlation': correlation,
            'recon_loss': recon_loss,  # 保持兼容性
            'gene_selection_method': 'variability_based',  # 更新选择方法标识
            'top20_genes_count': top_k,  # 添加实际选择的基因数量
            'nll_category': 'normal',  # 添加缺失的nll_category键
            'loss_weights': {
                'lambda_nll': lambda_nll,
                'lambda_mse': lambda_mse,
                'lambda_de_mse': lambda_de_mse,
                'lambda_pearson': lambda_pearson
            }
        }

        return total_loss, loss_components

