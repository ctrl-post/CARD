"""
KAN (Kolmogorov-Arnold Network) 基础层实现

基于Kolmogorov-Arnold表示定理，使用可学习样条函数作为激活，
相比传统MLP能以更少参数捕获高阶非线性关系。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.general_utils import print_system


class KANLinear(nn.Module):
    """
    KAN线性层（基于B样条的可学习激活函数）
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation: nn.Module = nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: tuple = (-1, 1),
    ):
        """
        初始化KAN线性层

        参数:
            in_features: 输入维度
            out_features: 输出维度
            grid_size: 样条网格数量（控制样条分辨率）
            spline_order: 样条阶数（通常为3，即三次B样条）
            scale_noise: 初始化噪声缩放
            scale_base: 基线激活权重缩放
            scale_spline: 样条激活权重缩放
            enable_standalone_scale_spline: 是否为每个样条独立缩放
            base_activation: 基线激活函数（默认SiLU）
            grid_eps: 网格扩展系数
            grid_range: 输入值范围（用于初始化网格）
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # 计算样条基函数数量
        # B样条基函数数 = grid_size + spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        self.num_spline_bases = grid_size + spline_order

        # 初始化网格点（均匀分布）
        # 扩展grid以支持边界外的值
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * h
            + grid_range[0]
        )
        # 注册为buffer（不参与梯度更新，但会被保存）
        self.register_buffer("grid", grid)

        # 基线权重（类似传统MLP的权重）
        self.base_weight = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )

        # 样条系数（可学习参数）
        # 形状: [out_features, in_features, num_spline_bases]
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, self.num_spline_bases)
        )


        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.Tensor(out_features) 
            )
        else:
            self.register_parameter("spline_scaler", None)

        # 缩放因子
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        """初始化参数"""
        # 基线权重：Xavier初始化
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5) * self.scale_base)

        # 样条权重：添加小噪声（初始时贡献较小）
        with torch.no_grad():
            noise = (
                (torch.rand(self.spline_weight.size()) - 0.5)
                * self.scale_noise
                / self.num_spline_bases
            )
            self.spline_weight.data.copy_(
                self.scale_spline * noise
            )

        # 样条缩放器：初始化为1
        if self.spline_scaler is not None:
            nn.init.constant_(self.spline_scaler, 1.0)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算B样条基函数值

        使用Cox-de Boor递推公式高效计算B样条

        参数:
            x: [batch_size, in_features] 输入值

        返回:
            bases: [batch_size, in_features, num_spline_bases] B样条基函数值
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        # 将输入裁剪到扩展网格范围内
        grid = self.grid  # [num_grid_points]
        x = x.unsqueeze(-1)  # [batch_size, in_features, 1]

        # 初始化：0阶B样条（指示函数）
        bases = ((x >= grid[:-1]) & (x < grid[1:])).to(x.dtype)

        # Cox-de Boor递推计算高阶B样条
        for k in range(1, self.spline_order + 1):
            # 左侧递推项
            left_num = x - grid[:-k-1]
            left_den = grid[k:-1] - grid[:-k-1]
            left_den = torch.where(left_den == 0, torch.ones_like(left_den), left_den)
            left = (left_num / left_den) * bases[..., :-1]

            # 右侧递推项
            right_num = grid[k+1:] - x
            right_den = grid[k+1:] - grid[1:-k]
            right_den = torch.where(right_den == 0, torch.ones_like(right_den), right_den)
            right = (right_num / right_den) * bases[..., 1:]

            # 合并
            bases = left + right

        return bases.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: [batch_size, in_features]

        返回:
            y: [batch_size, out_features]
        """
        # 1. 基线激活（传统MLP路径）
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # 2. 样条激活（KAN核心）
        # 计算B样条基函数
        spline_bases = self.b_splines(x)  # [batch_size, in_features, num_bases]

        # 应用样条权重
        spline_output = torch.einsum(
            'bik,oik->bo',
            spline_bases,
            self.spline_weight
        )

        # 应用独立缩放（如果启用）
        if self.spline_scaler is not None:
            spline_output = spline_output * self.spline_scaler.unsqueeze(0)

        # 3. 合并基线和样条输出
        y = base_output + spline_output

        return y

    def regularization_loss(self, regularize_activation: float = 1.0,
                           regularize_entropy: float = 1.0) -> torch.Tensor:
        """
        KAN正则化损失

        1. 激活正则：鼓励样条系数平滑
        2. 熵正则：鼓励稀疏激活

        参数:
            regularize_activation: 激活正则系数
            regularize_entropy: 熵正则系数

        返回:
            reg_loss: 正则化损失
        """
        # L1正则（稀疏性）
        l1_loss = self.spline_weight.abs().mean()

        # L2正则（平滑性）
        l2_loss = self.spline_weight.pow(2).mean()

        reg_loss = regularize_activation * l1_loss + regularize_entropy * l2_loss

        return reg_loss


class KANLayer(nn.Module):
    """
    KAN层（包含归一化和残差连接）

    结构: LayerNorm → KANLinear → Residual

    设计目的: 稳定训练，提升深层网络性能
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        use_layer_norm: bool = True,
        use_residual: bool = False,
        dropout: float = 0.0
    ):
        """
        初始化KAN层

        参数:
            in_features: 输入维度
            out_features: 输出维度
            grid_size: 样条网格大小
            use_layer_norm: 是否使用LayerNorm
            use_residual: 是否使用残差连接（需要in=out）
            dropout: Dropout比率
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.use_residual = use_residual and (in_features == out_features)

        # LayerNorm
        self.norm = nn.LayerNorm(in_features) if use_layer_norm else nn.Identity()

        # KAN线性层
        self.kan_linear = KANLinear(
            in_features=in_features,
            out_features=out_features,
            grid_size=grid_size
        )

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: [batch_size, in_features]

        返回:
            y: [batch_size, out_features]
        """
        # 归一化
        normed_x = self.norm(x)

        # KAN变换
        y = self.kan_linear(normed_x)

        # Dropout
        y = self.dropout(y)

        # 残差连接
        if self.use_residual:
            y = y + x

        return y


class KANNetwork(nn.Module):
    """
    多层KAN网络

    堆叠多个KAN层构建深层网络
    """

    def __init__(
        self,
        layer_dims: list,
        grid_size: int = 5,
        use_layer_norm: bool = True,
        dropout: float = 0.1
    ):
        """
        初始化KAN网络

        参数:
            layer_dims: 各层维度，例如[256, 128, 64]表示两层KAN
            grid_size: 样条网格大小
            use_layer_norm: 是否使用LayerNorm
            dropout: Dropout比率
        """
        super().__init__()

        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims) - 1

        # 构建KAN层
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = KANLayer(
                in_features=layer_dims[i],
                out_features=layer_dims[i + 1],
                grid_size=grid_size,
                use_layer_norm=use_layer_norm,
                use_residual=(i > 0 and layer_dims[i] == layer_dims[i+1]),
                dropout=dropout
            )
            self.layers.append(layer)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)

        return x

    def regularization_loss(self) -> torch.Tensor:
        """计算所有层的正则化损失"""
        reg_loss = 0
        for layer in self.layers:
            if hasattr(layer.kan_linear, 'regularization_loss'):
                reg_loss += layer.kan_linear.regularization_loss()

        return reg_loss / self.num_layers if self.num_layers > 0 else reg_loss
