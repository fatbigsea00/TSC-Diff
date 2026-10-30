"""
RBE-enhanced ControlNet conditioning embedding module
=====================================
Core idea: integrate the Region Boundary Enhancement (RBE) module into ControlNet's
conditioning_embedding, enhancing the boundary information of the segmentation mask through
Sobel edge extraction and feature fusion.

Data flow:
    conditioning_image (3ch, 512×512)
        ↓ conv_in
        [16ch, 512×512]
        ↓ blocks
        [256ch, 64×64]   ← deep features
        ↓ RBE (Sobel edge enhancement)
        [256ch, 64×64]   ← edge-enhanced features
        ↓ conv_out (zero_module)
        [320ch, 64×64]   ← final conditioning embedding

Advantages over standard ControlNet:
    1. Sobel edge extraction: explicitly models segmentation boundaries
    2. Edge region enhancement: strengthens the control of the mask contour over generation
    3. Plug-and-play: can load pretrained backbone weights, with RBE randomly initialized and then fine-tuned
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
    """Sets all parameters of the module to zero (used for zero convolution initialization)."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


# ─────────────────────────────────────────────────────────────────────────────
# Sobel edge extraction (refer to ISGLNet / Refinement_Module_s)
# ─────────────────────────────────────────────────────────────────────────────

def get_sobel_kernels(in_chan: int, out_chan: int):
    """
    Builds Sobel 3×3 convolution kernels (x and y directions).
    Standard Sobel operator:
        filter_x: horizontal gradient   filter_y: vertical gradient
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
    Region Boundary Enhancement module (RBE)

    Data flow:
        x [B, C, H, W]
          ├── Sobel (fixed weights) → edge_mag [B, 1, H, W]  (gradient magnitude)
          │       ↓
          │   edge_gate (learnable 1×1 bottleneck) → attn [B, C, H, W]  (per-channel spatial attention)
          │
          ├── edge_conv (depthwise 3×3 + BN) → edge_feat [B, C, H, W]  (lightweight edge features)
          │
          └── out = x + attn * edge_feat   (residual fusion: enhancement is added only in edge regions)

    Design points:
        1. Sobel extracts the gradient magnitude, with range [0, +∞).
        2. edge_gate uses bottleneck(1→C/4→C) + Sigmoid, letting the network learn
           "which channels need enhancement at edges"; attn→0 in non-edge regions.
        3. edge_conv uses depthwise conv to stay lightweight, introducing no cross-channel coupling.
        4. Residual connection: when attn≈0, out≈x, so the module has almost no effect on non-edge regions.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        # ── Sobel fixed convolution kernels (excluded from gradients) ──
        self.sobel_x = nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1, bias=False)
        kx, ky = get_sobel_kernels(channels, 1)
        self.sobel_x.weight = nn.Parameter(kx, requires_grad=False)
        self.sobel_y.weight = nn.Parameter(ky, requires_grad=False)

        # ── Learnable edge gating: edge_mag (1ch) → per-channel attention (Cch) ──
        mid = max(channels // 4, 16)
        self.edge_gate = nn.Sequential(
            nn.Conv2d(1, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # ── Lightweight edge feature extraction: depthwise 3×3 ──
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Sobel gradient magnitude [B, 1, H, W]
        g_x = self.sobel_x(x)
        g_y = self.sobel_y(x)
        edge_mag = torch.sqrt(g_x.pow(2) + g_y.pow(2) + 1e-8)

        # 2. Per-channel spatial attention [B, C, H, W]: →0 in non-edge regions, →large at edges
        attn = self.edge_gate(edge_mag)

        # 3. Lightweight edge features [B, C, H, W]
        edge_feat = self.edge_conv(x)

        # 4. Residual fusion: add enhancement features only in edge regions
        return x + attn * edge_feat


# Backward-compatible aliases
ResidualBoundaryEnhancement = RegionBoundaryEnhancement
EdgeRefinementModule = RegionBoundaryEnhancement


# ─────────────────────────────────────────────────────────────────────────────
# RBE-enhanced Conditioning Embedding
# ─────────────────────────────────────────────────────────────────────────────

class ControlNetConditioningEmbeddingWithRBE(nn.Module):
    """
    RBE-enhanced ControlNet conditioning embedding module.
    Inserts RegionBoundaryEnhancement at the deep features to enhance the conditioning
    representation of segmentation boundaries.
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
        """Loads backbone weights from a standard ControlNetConditioningEmbedding (excluding RBE)."""
        with torch.no_grad():
            self.conv_in.weight.copy_(standard_embedding.conv_in.weight)
            self.conv_in.bias.copy_(standard_embedding.conv_in.bias)

            for i, block in enumerate(standard_embedding.blocks):
                self.blocks[i].weight.copy_(block.weight)
                self.blocks[i].bias.copy_(block.bias)

            self.conv_out.weight.copy_(standard_embedding.conv_out.weight)
            self.conv_out.bias.copy_(standard_embedding.conv_out.bias)

        print("[OK] Backbone weights loaded from standard ControlNetConditioningEmbedding")
        print("  RBE module remains randomly initialized, awaiting fine-tuning")


def _infer_block_out_channels(embedding: nn.Module) -> Tuple[int, ...]:
    channels = [embedding.conv_in.out_channels]
    for i in range(0, len(embedding.blocks), 2):
        channels.append(embedding.blocks[i + 1].out_channels)
    return tuple(channels)


def apply_rbe_to_controlnet(controlnet, input_resolution: int = 64):
    """
    Replaces an existing ControlNet's conditioning_embedding in place with the RBE-enhanced version.
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
    print(f"[OK] RBE injected into ControlNet conditioning_embedding  (RBE new parameters: {rbe_params:,})")

    return controlnet


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n" + "=" * 50)
    print("Test: ControlNetConditioningEmbeddingWithRBE forward pass")
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

    print(f"  Input: {conditioning.shape}")
    print(f"  Output: {out.shape}  Expected: [2, 320, 64, 64]")
    assert out.shape == (batch_size, 320, H // 8, W // 8), f"Incorrect output shape: {out.shape}"
    print("  [OK] Forward pass correct")

    total = sum(p.numel() for p in rbe_embed.parameters())
    rbe_only = sum(p.numel() for p in rbe_embed.rbe.parameters())
    print(f"\n  Total parameters: {total:,}")
    print(f"  RBE parameters: {rbe_only:,}")
    print(f"  RBE proportion: {rbe_only/total*100:.1f}%")
