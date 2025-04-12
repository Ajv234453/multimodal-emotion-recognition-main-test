# -*- coding: utf-8 -*-
"""
Modified Transformer with Dynamic Tanh Normalization
Original codebase: timm library https://github.com/rwightman/pytorch-image-models
Modifications:
1. 所有LayerNorm替换为DyT
2. 保持BN层（但原始代码未使用BN）
3. 移除norm_layer参数依赖
"""

import torch
from torch import nn


# ------------------------- 新增DyT模块 -------------------------
class DyT(nn.Module):
    """Dynamic Tanh Normalization Layer with Channel-wise Scaling
    Args:
        dim (int): 输入特征通道数（必须提供）
        init_alpha (float): 初始缩放因子，默认0.5
        init_gamma (float): 初始通道缩放因子，默认1.0
    """

   #这里可以修改α与γ的初始值
    def __init__(self, dim, init_alpha=0.5, init_gamma=1.0):
        super().__init__()
        if dim is None:
            raise ValueError("必须指定dim参数（输入特征通道数）")

        # 可学习的全局缩放因子
        self.alpha = nn.Parameter(torch.tensor(init_alpha))

        # 可学习的逐通道缩放参数 [C,]
        self.gamma = nn.Parameter(torch.full((dim,), init_gamma))  # 形状为(dim,)

        # 初始化参数
        self._reset_parameters(init_alpha, init_gamma)

    def _reset_parameters(self, alpha, gamma):
        """参数初始化策略"""
        nn.init.constant_(self.alpha, alpha)
        nn.init.constant_(self.gamma, gamma)

    def forward(self, x):
        """
        输入x形状: (B, C, ...) 或 (B, Seq, C)
        输出形状与输入保持一致
        """
        # 动态非线性变换
        x_tanh = torch.tanh(self.alpha * x)  # [B, C, ...]

        # 获取维度信息
        if x.dim() == 3:  # 序列数据 (B, Seq, C)
            gamma = self.gamma[None, None, :]  # [1, 1, C]
        elif x.dim() == 4:  # 图像特征 (B, C, H, W)
            gamma = self.gamma[None, :, None, None]  # [1, C, 1, 1]
        else:
            raise NotImplementedError(f"不支持的输入维度: {x.dim()}")

        # 逐通道缩放
        return gamma * x_tanh  # 广播乘法 [B, C, ...]


# ------------------------- MLP模块（未修改） -------------------------
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks """

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., use_conv1=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.use_conv1 = use_conv1

        if use_conv1:
            self.fc1 = nn.Conv1d(in_features, hidden_features, 3, 1, padding='same')
            self.fc2 = nn.Conv1d(hidden_features, out_features, 3, 1, padding='same')
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)

        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        if self.use_conv1:
            x = x.transpose(1, 2)
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            x = x.transpose(1, 2)
        else:
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
        return x


# ------------------------- DropPath模块（未修改）-------------------------
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# ------------------------- Attention模块（未修改）-------------------------
class Attention(nn.Module):
    def __init__(self, in_dim_k, in_dim_q, out_dim, num_heads=8,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = out_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(in_dim_q, out_dim, bias=qkv_bias)
        self.kv = nn.Linear(in_dim_k, out_dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.qkmatrix = None

    def forward(self, x, x_q):
        B, Nk, Ck = x.shape
        B, Nq, Cq = x_q.shape

        # Query处理
        q = self.q(x_q).reshape(B, Nq, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        # Key-Value处理
        kv = self.kv(x).reshape(B, Nk, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        self.qkmatrix = attn
        attn = self.attn_drop(attn)

        # 输出投影
        x = (attn @ v).transpose(1, 2).reshape(B, Nq, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, self.qkmatrix


# ------------------------- 修改后的AttentionBlock模块 -------------------------
class AttentionBlock(nn.Module):
    def __init__(self, in_dim_k, in_dim_q, out_dim, num_heads, mlp_ratio=2.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, use_conv1=False):  # 移除norm_layer参数
        super().__init__()

        # 替换LayerNorm为DyT
        self.norm1_q = DyT(in_dim_q)
        self.norm1_k = DyT(in_dim_k)

        self.attn = Attention(
            in_dim_k=in_dim_k,
            in_dim_q=in_dim_q,
            out_dim=out_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = DyT(out_dim)  # 第二层归一化

        # MLP配置
        mlp_hidden_dim = int(out_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=out_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            use_conv1=use_conv1
        )

    def forward(self, xk, xq):
        # 第一层归一化 + 注意力
        x, attn_matrix = self.attn(
            self.norm1_k(xk),  # 对Key的归一化
            self.norm1_q(xq)  # 对Query的归一化
        )
        x = self.drop_path(x)

        # 残差连接 + MLP
        x = x + self.drop_path(
            self.mlp(
                self.norm2(x)  # 第二层归一化
            )
        )
        return x