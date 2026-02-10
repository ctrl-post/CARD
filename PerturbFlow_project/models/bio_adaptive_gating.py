"""
Bio-Adaptive Gating Module
生物学自适应门控模块

根据扰动类型自动调节结构嵌入（X1）、功能嵌入（X2）、状态嵌入（X3）的权重分布，
使模型能够学习扰动特异性的生物学调控偏好。

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.general_utils import print_system


class BioAdaptiveGating(nn.Module):
    """
    生物学自适应门控模块
    """

    def __init__(self, cond_dim: int, hidden_dim: int = None,
                 dropout: float = 0.1, use_layer_norm: bool = True,
                 verbose: bool = False):
        """
        初始化Bio-Adaptive Gating模块

        参数:
            cond_dim: 条件向量维度（通常为 3 * hidden_size = 768）
            hidden_dim: 门控MLP的隐藏层维度（默认为cond_dim）
            dropout: Dropout比率
            use_layer_norm: 是否使用LayerNorm稳定训练
            verbose: 是否打印详细调试日志
        """
        super().__init__()

        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim or cond_dim
        self.use_layer_norm = use_layer_norm
        self.verbose = verbose

        # 门控MLP网络
        # 设计：2层MLP，中间层ReLU激活，输出层Sigmoid
        self.gate_mlp = nn.Sequential(
            nn.Linear(cond_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, cond_dim),
            nn.Sigmoid()  # 输出[0,1]范围的门控权重
        )

        # 初始化 α = 0.5
        # 允许模型学习是否使用门控（α * gated + (1-α) * raw）
        self.residual_weight = nn.Parameter(torch.tensor(0.5))

        if self.verbose:
            print_system(f"[BioAdaptiveGating] 初始化完成")
            print_system(f"  条件维度: {cond_dim}, 隐藏维度: {self.hidden_dim}")
            print_system(f"  LayerNorm: {use_layer_norm}, Dropout: {dropout}")

    def forward(self, c_raw: torch.Tensor, return_gate_weights: bool = False):
        """
        前向传播

        参数:
            c_raw: [batch_size, cond_dim] 原始条件向量
            return_gate_weights: 是否返回门控权重（用于可解释性分析）

        返回:
            c_gated: [batch_size, cond_dim] 门控后的条件向量
            gate_weights: [batch_size, cond_dim] 门控权重
        """
        batch_size = c_raw.size(0)

        if self.verbose:
            print_system(f"[Gating输入] c_raw形状: {c_raw.shape}, "
                        f"范围: [{c_raw.min().item():.4f}, {c_raw.max().item():.4f}]")

        # 计算门控权重 g
        gate_weights = self.gate_mlp(c_raw)  # [batch_size, cond_dim]

        if self.verbose:
            print_system(f"[门控权重] 范围: [{gate_weights.min().item():.4f}, "
                        f"{gate_weights.max().item():.4f}], "
                        f"均值: {gate_weights.mean().item():.4f}")

        # 应用门控（元素级乘法）
        c_gated_pure = c_raw * gate_weights

        # 添加残差连接，避免过度抑制
        # c_final = α * c_gated + (1-α) * c_raw
        alpha = torch.sigmoid(self.residual_weight)  # 将权重限制在[0,1]
        c_gated = alpha * c_gated_pure + (1 - alpha) * c_raw

        if self.verbose:
            print_system(f"[门控输出] c_gated形状: {c_gated.shape}, "
                        f"范围: [{c_gated.min().item():.4f}, {c_gated.max().item():.4f}]")
            print_system(f"[残差权重] α={alpha.item():.4f}")

        if return_gate_weights:
            return c_gated, gate_weights
        return c_gated

    def analyze_gating_pattern(self, c_raw: torch.Tensor,
                               segment_names: list = None):
        """
        分析门控模式（用于生物学可解释性）

        假设c_raw由三部分拼接：[X1_structure, X2_function, X3_state]
        分析每部分的平均门控权重，揭示扰动特异性偏好

        参数:
            c_raw: [batch_size, cond_dim] 原始条件向量
            segment_names: 各部分的名称，例如['structure', 'function', 'state']

        返回:
            analysis: dict - 各部分的统计信息
        """
        with torch.no_grad():
            gate_weights = self.gate_mlp(c_raw)  # [batch_size, cond_dim]

            # 假设均匀划分为3部分
            segment_size = self.cond_dim // 3
            segments = {
                'structure (X1)': gate_weights[:, :segment_size],
                'function (X2)': gate_weights[:, segment_size:2*segment_size],
                'state (X3)': gate_weights[:, 2*segment_size:]
            }

            if segment_names and len(segment_names) == 3:
                segments = {
                    segment_names[0]: gate_weights[:, :segment_size],
                    segment_names[1]: gate_weights[:, segment_size:2*segment_size],
                    segment_names[2]: gate_weights[:, 2*segment_size:]
                }

            analysis = {}
            for name, weights in segments.items():
                analysis[name] = {
                    'mean': weights.mean().item(),
                    'std': weights.std().item(),
                    'min': weights.min().item(),
                    'max': weights.max().item()
                }

            return analysis
