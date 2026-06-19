"""Transformer building blocks for JWST_DINO's Vision Transformer.

Plain, readable ViT components in the style of LowResPT/model/modules.py.
We deliberately use the same primitives LowResPT trains with on this cluster:
explicit-softmax attention (no SDPA/flash) and an unfold+Linear patch embedding
(no Conv2d). All crops in a forward pass share one token length,
so no attention padding mask is ever needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """LayerNorm with optional bias (matches LowResPT)."""

    def __init__(self, dim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias)


class MLP(nn.Module):
    """Two-layer GELU MLP."""

    def __init__(self, dim: int, hidden_dim: int, bias: bool = True, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention via F.scaled_dot_product_attention. No mask (crops
    are equal-length). On torch>=2.5 this dispatches to flash / mem-efficient kernels
    on both A100 (sm80) and H100 (sm90), which are O(T) in memory (no (B,H,T,T) matrix
    materialized for backward) — unlike the old explicit-softmax path."""

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, proj_bias: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_drop = dropout

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (B, num_heads, T, head_dim)

        out = F.scaled_dot_product_attention(  # scales by 1/sqrt(head_dim) internally
            q, k, v, dropout_p=self.attn_drop if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class LayerScale(nn.Module):
    """Per-channel learnable scale on a residual branch (DINOv2 default)."""

    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class DropPath(nn.Module):
    """Stochastic depth: drop the whole residual branch per-sample at train time."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # broadcast over all but batch
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class PatchEmbed(nn.Module):
    """Image -> patch tokens via unfold + Linear (equivalent to a strided conv,
    but cuDNN-free). Supports patch_size != stride (overlapping patches)."""

    def __init__(self, in_chans: int, embed_dim: int, patch_size: int, patch_stride: int):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.proj = nn.Linear(in_chans * patch_size * patch_size, embed_dim)

    @staticmethod
    def num_tokens_per_side(image_size: int, patch_size: int, patch_stride: int) -> int:
        return (image_size - patch_size) // patch_stride + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, C*p*p, N) -> (B, N, C*p*p) -> (B, N, D)
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_stride)
        return self.proj(patches.transpose(1, 2))


class TransformerBlock(nn.Module):
    """Pre-norm block with LayerScale + DropPath on both residual branches."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 proj_bias: bool = True, ffn_bias: bool = True, drop_path: float = 0.0,
                 layerscale_init: float = 1e-5, dropout: float = 0.0):
        super().__init__()
        self.norm1 = LayerNorm(dim, bias=True)
        self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, dropout=dropout)
        self.ls1 = LayerScale(dim, layerscale_init)
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = LayerNorm(dim, bias=True)
        self.mlp = MLP(dim, int(dim * mlp_ratio), bias=ffn_bias, dropout=dropout)
        self.ls2 = LayerScale(dim, layerscale_init)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
