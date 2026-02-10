"""
KAN-ReFlow 动力学映射模块

基于Rectified Flow (Liu et al., 2022)，使用KAN网络增强非线性建模能力。
实现从未扰动状态到扰动状态的连续轨迹学习。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from utils.general_utils import print_system
from .kan_layer import KANNetwork


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    创建正弦时间步嵌入 (来自Transformer位置编码)

    将时间步t映射为高维向量，帮助模型感知时间信息

    参数:
        timesteps: [N] 时间步，范围[0, 1]
        dim: 嵌入维度
        max_period: 控制嵌入的最小频率

    返回:
        embedding: [N, dim] 时间步嵌入
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class KANReFlowVelocityNet(nn.Module):
    """
    KAN-ReFlow 速度场网络

    核心组件: 预测速度 v_θ(z_t, t, c)
    - 输入: 当前状态z_t、时间步t、条件向量c
    - 输出: 速度预测v

    设计:
    1. 时间步嵌入: 正弦位置编码
    2. 条件嵌入: 线性投影
    3. KAN主干网络: 多层样条激活
    4. 特征融合: 时间和条件在每层注入
    """

    def __init__(
        self,
        input_dim: int,          # 输入维度（基因数）
        cond_dim: int,           # 条件维度（门控后的条件向量）
        hidden_dim: int = 256,   # KAN隐藏层维度
        num_layers: int = 3,     # KAN层数
        time_embed_dim: int = 128,  # 时间嵌入维度
        kan_grid_size: int = 5,  # KAN样条网格大小
        dropout: float = 0.1,
        verbose: bool = False    # 是否打印详细日志
    ):
        """
        初始化速度场网络

        参数:
            input_dim: 输入维度（基因数量）
            cond_dim: 条件向量维度
            hidden_dim: 隐藏层维度
            num_layers: KAN层数
            time_embed_dim: 时间嵌入维度
            kan_grid_size: KAN样条网格大小
            dropout: Dropout比率
            verbose: 是否打印详细调试日志
        """
        super().__init__()

        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.time_embed_dim = time_embed_dim  # 保存time_embed_dim
        self.verbose = verbose

        # 时间步嵌入网络
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 条件投影
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # KAN主干网络
        # 构建层维度：[hidden_dim] * (num_layers + 1)
        layer_dims = [hidden_dim] * (num_layers + 1)
        self.kan_network = KANNetwork(
            layer_dims=layer_dims,
            grid_size=kan_grid_size,
            use_layer_norm=True,
            dropout=dropout
        )

        # 时间/条件注入层（在每个KAN层后注入）
        self.time_modulations = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.cond_modulations = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # 输出投影（预测速度）
        self.output_proj = nn.Linear(hidden_dim, input_dim)

        # 零初始化输出层（稳定训练初期）
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        if self.verbose:
            print_system(f"[KANReFlowVelocityNet] 初始化完成")
            print_system(f"  输入维度: {input_dim}, 条件维度: {cond_dim}")
            print_system(f"  隐藏维度: {hidden_dim}, KAN层数: {num_layers}, 网格大小: {kan_grid_size}")

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 预测速度

        参数:
            z_t: [batch_size, input_dim] 当前状态
            t: [batch_size] 时间步，范围[0, 1]
            cond: [batch_size, cond_dim] 条件向量（门控后）

        返回:
            v_pred: [batch_size, input_dim] 预测的速度向量
        """
        batch_size = z_t.size(0)

        if self.verbose:
            print_system(f"\n[速度场网络输入]")
            print_system(f"  z_t形状: {z_t.shape}, 范围: [{z_t.min().item():.4f}, {z_t.max().item():.4f}]")
            print_system(f"  t范围: [{t.min().item():.4f}, {t.max().item():.4f}]")
            print_system(f"  cond形状: {cond.shape}")

        # 1. 时间步嵌入
        t_emb_raw = timestep_embedding(t, self.time_embed_dim)
        t_emb = self.time_embed(t_emb_raw)  # [batch_size, hidden_dim]

        # 2. 条件嵌入
        cond_emb = self.cond_proj(cond)  # [batch_size, hidden_dim]

        # 3. 输入投影
        h = self.input_proj(z_t)  # [batch_size, hidden_dim]

        # 4. KAN网络 + 时间/条件调制
        for i, kan_layer in enumerate(self.kan_network.layers):
            # KAN变换
            h = kan_layer(h)

            # 注入时间信息（加性调制）
            h = h + self.time_modulations[i](t_emb)

            # 注入条件信息（加性调制）
            h = h + self.cond_modulations[i](cond_emb)

            if self.verbose:
                print_system(f"  KAN层{i+1}后: 形状={h.shape}, "
                            f"范围=[{h.min().item():.4f}, {h.max().item():.4f}]")

        # 5. 输出速度预测
        v_pred = self.output_proj(h)

        if self.verbose:
            print_system(f"[速度场输出] v_pred形状: {v_pred.shape}, "
                        f"范围: [{v_pred.min().item():.4f}, {v_pred.max().item():.4f}]")

        return v_pred


class KANReFlow(nn.Module):
    """
    KAN-ReFlow 完整模块

    功能:
    1. 训练: 学习速度场 v_θ
    2. 推理: ODE求解生成预测

    训练流程:
        t ~ U[0,1]
        z_t = (1-t)*z_0 + t*y_true
        v_target = y_true - z_0
        loss = MSE(v_θ(z_t, t, c), v_target)

    推理流程:
        z_0 = 初始状态（对照表达）
        for i in steps:
            v = v_θ(z_i, t_i, c)
            z_{i+1} = z_i + dt * v
        return z_1
    """

    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        kan_grid_size: int = 5,
        num_ode_steps: int = 10,  # 推理时ODE求解步数
        ode_solver: str = 'euler',  # 'euler' 或 'rk4'
        dropout: float = 0.1,
        verbose: bool = False  # 是否打印详细日志
    ):
        """
        初始化KAN-ReFlow模块

        参数:
            input_dim: 输入维度（基因数）
            cond_dim: 条件维度
            hidden_dim: 隐藏层维度
            num_layers: KAN层数
            kan_grid_size: KAN样条网格大小
            num_ode_steps: 推理时ODE求解步数
            ode_solver: ODE求解器类型
            dropout: Dropout比率
            verbose: 是否打印详细调试日志
        """
        super().__init__()

        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.num_ode_steps = num_ode_steps
        self.ode_solver = ode_solver
        self.verbose = verbose

        # 速度场网络
        self.velocity_net = KANReFlowVelocityNet(
            input_dim=input_dim,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            time_embed_dim=128,
            kan_grid_size=kan_grid_size,
            dropout=dropout,
            verbose=verbose  # 传递verbose参数
        )

        if self.verbose:
            print_system(f"\n[KANReFlow] 初始化完成")
            print_system(f"  ODE求解器: {ode_solver}, 推理步数: {num_ode_steps}")

    def compute_velocity_loss(
        self,
        z_0: torch.Tensor,
        y_true: torch.Tensor,
        cond: torch.Tensor,
        num_train_steps: int = 1
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算Rectified Flow速度损失（训练时调用）

        参数:
            z_0: [batch_size, input_dim] 初始状态（未扰动）
            y_true: [batch_size, input_dim] 目标状态（扰动后）
            cond: [batch_size, cond_dim] 条件向量
            num_train_steps: 训练时采样的时间步数（多样性）

        返回:
            loss: 速度预测损失
            loss_dict: 损失分量字典
        """
        batch_size = z_0.size(0)
        device = z_0.device

        if self.verbose:
            print_system(f"\n[计算速度损失]")
            print_system(f"  z_0形状: {z_0.shape}, y_true形状: {y_true.shape}")

        # 1. 随机采样时间步 t ~ U[0,1]
        # 多步采样（增加训练多样性）
        t_samples = torch.rand(batch_size, num_train_steps, device=device)

        total_loss = 0
        loss_components = []

        for step_idx in range(num_train_steps):
            t = t_samples[:, step_idx]  # [batch_size]

            # 2. 线性插值路径: z_t = (1-t)*z_0 + t*y_true
            t_expanded = t.view(-1, 1)  # [batch_size, 1]
            z_t = (1 - t_expanded) * z_0 + t_expanded * y_true

            # 3. 目标速度: v_target = y_true - z_0
            v_target = y_true - z_0

            # 4. 预测速度: v_pred = v_θ(z_t, t, c)
            v_pred = self.velocity_net(z_t, t, cond)

            # 5. 计算MSE损失
            loss_step = F.mse_loss(v_pred, v_target)
            total_loss += loss_step

            loss_components.append({
                't_mean': t.mean().item(),
                'loss': loss_step.item(),
                'v_pred_norm': v_pred.norm(dim=1).mean().item(),
                'v_target_norm': v_target.norm(dim=1).mean().item()
            })

            if self.verbose:
                print_system(f"  步骤{step_idx+1}: t均值={t.mean().item():.3f}, "
                            f"损失={loss_step.item():.6f}")

        # 平均损失
        avg_loss = total_loss / num_train_steps

        # 汇总损失字典
        loss_dict = {
            'rf_loss': avg_loss.item(),
            'v_pred_norm_mean': sum(lc['v_pred_norm'] for lc in loss_components) / num_train_steps,
            'v_target_norm_mean': sum(lc['v_target_norm'] for lc in loss_components) / num_train_steps
        }

        if self.verbose:
            print_system(f"[平均速度损失] {avg_loss.item():.6f}")

        return avg_loss, loss_dict

    def sample(
        self,
        z_0: torch.Tensor,
        cond: torch.Tensor,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        ODE求解生成预测（推理时调用）

        从z_0出发，沿着速度场积分到t=1，得到预测y_pred

        参数:
            z_0: [batch_size, input_dim] 初始状态
            cond: [batch_size, cond_dim] 条件向量
            num_steps: ODE求解步数（默认使用初始化时的值）

        返回:
            y_pred: [batch_size, input_dim] 预测的扰动后状态
        """
        num_steps = num_steps or self.num_ode_steps
        device = z_0.device
        batch_size = z_0.size(0)

        if self.verbose:
            print_system(f"\n[ODE求解采样] 步数: {num_steps}, 求解器: {self.ode_solver}")

        # 时间步划分: t=0 → t=1
        dt = 1.0 / num_steps
        timesteps = torch.linspace(0, 1, num_steps + 1, device=device)

        z = z_0.clone()

        for i in range(num_steps):
            t_current = timesteps[i]
            t_batch = torch.full((batch_size,), t_current, device=device)

            if self.ode_solver == 'euler':
                # Euler方法: z_{i+1} = z_i + dt * v_θ(z_i, t_i, c)
                v = self.velocity_net(z, t_batch, cond)
                z = z + dt * v

            elif self.ode_solver == 'rk4':
                # Runge-Kutta 4阶方法（更精确）
                k1 = self.velocity_net(z, t_batch, cond)

                t_mid = t_batch + dt / 2
                k2 = self.velocity_net(z + dt * k1 / 2, t_mid, cond)
                k3 = self.velocity_net(z + dt * k2 / 2, t_mid, cond)

                t_next = t_batch + dt
                k4 = self.velocity_net(z + dt * k3, t_next, cond)

                z = z + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

            else:
                raise ValueError(f"不支持的ODE求解器: {self.ode_solver}")

            if self.verbose and i % 2 == 0:  # 每2步打印一次
                print_system(f"  步骤{i+1}/{num_steps}: t={t_current:.3f}, "
                            f"z范围=[{z.min().item():.4f}, {z.max().item():.4f}]")

        if self.verbose:
            print_system(f"[ODE求解完成] 最终预测范围: [{z.min().item():.4f}, {z.max().item():.4f}]")

        return z

    def forward(
        self,
        z_0: torch.Tensor,
        cond: torch.Tensor,
        y_true: Optional[torch.Tensor] = None,
        mode: str = 'train'
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播（统一接口）

        参数:
            z_0: [batch_size, input_dim] 初始状态
            cond: [batch_size, cond_dim] 条件向量
            y_true: [batch_size, input_dim] 真实目标（训练时需要）
            mode: 'train' 或 'inference'

        返回:
            训练模式: (y_pred, loss)
            推理模式: (y_pred, None)
        """
        if mode == 'train':
            if y_true is None:
                raise ValueError("训练模式需要提供y_true")

            # 计算速度损失
            loss, loss_dict = self.compute_velocity_loss(z_0, y_true, cond)

            # 同时生成预测（用于辅助损失计算）
            with torch.no_grad():
                y_pred = self.sample(z_0, cond)

            return y_pred, loss

        else:  # inference
            # ODE求解生成预测
            y_pred = self.sample(z_0, cond)
            return y_pred, None

    def get_regularization_loss(self) -> torch.Tensor:
        """获取KAN正则化损失"""
        return self.velocity_net.kan_network.regularization_loss()
