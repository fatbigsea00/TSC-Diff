"""
RBE-增强的ControlNet条件嵌入模块
=====================================
核心思想：将区域边界增强模块(RBE)集成到 ControlNet 的 conditioning_embedding 中，
通过 Sobel 边缘提取与特征融合，增强分割掩码的边界信息。

数据流：
    conditioning_image (3ch, 512×512)
        ↓ conv_in
        [16ch, 512×512]
        ↓ blocks
        [256ch, 64×64]   ← 深层特征
        ↓ RBE (Sobel边缘增强)
        [256ch, 64×64]   ← 边缘增强后的特征
        ↓ conv_out (zero_module)
        [320ch, 64×64]   ← 最终条件嵌入

改进优势（相对标准ControlNet）：
    1. Sobel边缘提取：显式建模分割边界
    2. 边缘区域增强：强化掩码轮廓对生成的控制力
    3. 即插即用：可加载预训练主干权重，RBE随机初始化后微调
"""

import numpy as np
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def zero_module(module: nn.Module) -> nn.Module:
    """将模块所有参数置零（用于zero convolution初始化）"""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


# ─────────────────────────────────────────────────────────────────────────────
# Sobel 边缘提取 (参考 ISGLNet / Refinement_Module_s)
# ─────────────────────────────────────────────────────────────────────────────

def get_sobel_kernels(in_chan: int, out_chan: int):
    """
    构建 Sobel 3×3 卷积核 (x 和 y 方向)
    标准 Sobel 算子:
        filter_x: 水平梯度   filter_y: 垂直梯度
    """
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ], dtype=np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ], dtype=np.float32)

    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    return torch.from_numpy(filter_x), torch.from_numpy(filter_y)


class RegionBoundaryEnhancement(nn.Module):
    """
    区域边界增强模块 (RBE — Region Boundary Enhancement)

    数据流:
        x [B, C, H, W]
          ├── Sobel (固定权重) → edge_mag [B, 1, H, W]  (梯度幅值)
          │       ↓
          │   edge_gate (可学习 1×1 bottleneck) → attn [B, C, H, W]  (逐通道空间注意力)
          │
          ├── edge_conv (depthwise 3×3 + BN) → edge_feat [B, C, H, W]  (轻量边缘特征)
          │
          └── out = x + attn * edge_feat   (残差融合：仅在边缘区域叠加增强)

    设计要点:
        1. Sobel 提取梯度幅值，范围 [0, +∞)
        2. edge_gate 用 bottleneck(1→C/4→C) + Sigmoid，
           让网络学习 "哪些通道在边缘处需要增强"，非边缘处 attn→0
        3. edge_conv 用 depthwise conv 保持轻量，不引入跨通道耦合
        4. 残差连接：attn≈0 时 out≈x，模块对非边缘几乎无影响
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        # ── Sobel 固定卷积核 (不参与梯度) ──
        self.sobel_x = nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1, bias=False)
        kx, ky = get_sobel_kernels(channels, 1)
        self.sobel_x.weight = nn.Parameter(kx, requires_grad=False)
        self.sobel_y.weight = nn.Parameter(ky, requires_grad=False)

        # ── 可学习边缘门控: edge_mag (1ch) → 逐通道注意力 (Cch) ──
        mid = max(channels // 4, 16)
        self.edge_gate = nn.Sequential(
            nn.Conv2d(1, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # ── 轻量边缘特征提取: depthwise 3×3 ──
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Sobel 梯度幅值 [B, 1, H, W]
        g_x = self.sobel_x(x)
        g_y = self.sobel_y(x)
        edge_mag = torch.sqrt(g_x.pow(2) + g_y.pow(2) + 1e-8)

        # 2. 逐通道空间注意力 [B, C, H, W]，非边缘处→0，边缘处→大
        attn = self.edge_gate(edge_mag)

        # 3. 轻量边缘特征 [B, C, H, W]
        edge_feat = self.edge_conv(x)

        # 4. 残差融合：仅在边缘区域叠加增强特征
        return x + attn * edge_feat


# Backward-compatible aliases
ResidualBoundaryEnhancement = RegionBoundaryEnhancement
EdgeRefinementModule = RegionBoundaryEnhancement


# ─────────────────────────────────────────────────────────────────────────────
# RBE 增强的 Conditioning Embedding
# ─────────────────────────────────────────────────────────────────────────────

class ControlNetConditioningEmbeddingWithRBE(nn.Module):
    """
    RBE 增强的 ControlNet 条件嵌入模块
    在深层特征处插入 RegionBoundaryEnhancement，增强分割边界的条件表达。
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(
            conditioning_channels,
            block_out_channels[0],
            kernel_size=3,
            padding=1,
        )

        self.blocks = nn.ModuleList([])
        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(
                block_out_channels[-1],
                conditioning_embedding_channels,
                kernel_size=3,
                padding=1,
            )
        )

        deep_dim = block_out_channels[-1]
        self.rbe = RegionBoundaryEnhancement(deep_dim)

        self.block_out_channels = block_out_channels
        self.deep_dim = deep_dim

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = F.silu(block(embedding))

        embedding = self.rbe(embedding)

        embedding = self.conv_out(embedding)
        return embedding

    def load_backbone_from_standard(self, standard_embedding: nn.Module) -> None:
        """从标准 ControlNetConditioningEmbedding 加载主干权重（不包含 RBE）"""
        with torch.no_grad():
            self.conv_in.weight.copy_(standard_embedding.conv_in.weight)
            self.conv_in.bias.copy_(standard_embedding.conv_in.bias)

            for i, block in enumerate(standard_embedding.blocks):
                self.blocks[i].weight.copy_(block.weight)
                self.blocks[i].bias.copy_(block.bias)

            self.conv_out.weight.copy_(standard_embedding.conv_out.weight)
            self.conv_out.bias.copy_(standard_embedding.conv_out.bias)

        print("[OK] 主干权重已从标准 ControlNetConditioningEmbedding 加载")
        print("  RBE 模块保持随机初始化，等待微调训练")


# Backward-compatible aliases
ControlNetConditioningEmbeddingWithERM = ControlNetConditioningEmbeddingWithRBE


def _infer_block_out_channels(embedding: nn.Module) -> Tuple[int, ...]:
    channels = [embedding.conv_in.out_channels]
    for i in range(0, len(embedding.blocks), 2):
        channels.append(embedding.blocks[i + 1].out_channels)
    return tuple(channels)


def apply_rbe_to_controlnet(controlnet, input_resolution: int = 64):
    """
    就地给已有 ControlNet 替换为 RBE 增强版 conditioning_embedding。
    """
    orig_embed = controlnet.controlnet_cond_embedding

    block_out_channels = _infer_block_out_channels(orig_embed)
    conditioning_embedding_channels = orig_embed.conv_out.out_channels
    conditioning_channels = orig_embed.conv_in.in_channels

    rbe_embedding = ControlNetConditioningEmbeddingWithRBE(
        conditioning_embedding_channels=conditioning_embedding_channels,
        conditioning_channels=conditioning_channels,
        block_out_channels=block_out_channels,
    )

    rbe_embedding.load_backbone_from_standard(orig_embed)

    controlnet.controlnet_cond_embedding = rbe_embedding

    rbe_params = sum(p.numel() for p in rbe_embedding.rbe.parameters())
    print(f"[OK] RBE 已注入 ControlNet conditioning_embedding  (RBE 新增参数: {rbe_params:,})")

    return controlnet


# Backward-compatible alias
apply_erm_to_controlnet = apply_rbe_to_controlnet


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print("\n" + "=" * 50)
    print("测试: ControlNetConditioningEmbeddingWithRBE 前向传播")
    print("=" * 50)

    batch_size = 2
    H, W = 512, 512

    rbe_embed = ControlNetConditioningEmbeddingWithRBE(
        conditioning_embedding_channels=320,
        conditioning_channels=3,
        block_out_channels=(16, 32, 96, 256),
    ).to(device)

    conditioning = torch.randn(batch_size, 3, H, W).to(device)
    with torch.no_grad():
        out = rbe_embed(conditioning)

    print(f"  输入: {conditioning.shape}")
    print(f"  输出: {out.shape}  期望: [2, 320, 64, 64]")
    assert out.shape == (batch_size, 320, H // 8, W // 8), f"输出形状错误: {out.shape}"
    print("  [OK] 前向传播正确")

    total = sum(p.numel() for p in rbe_embed.parameters())
    rbe_only = sum(p.numel() for p in rbe_embed.rbe.parameters())
    print(f"\n  总参数量: {total:,}")
    print(f"  RBE参数量: {rbe_only:,}")
    print(f"  RBE占比: {rbe_only/total*100:.1f}%")
