"""Transformer modules for LowResPT.

Includes: LayerNorm, TransformerBlock, SelfAttention, MLP.
Copied verbatim from ma_specformer/model/modules.py.
"""

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""

    def __init__(self, dim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.layer_norm(x, (x.shape[-1],), self.weight, self.bias)
        return out


class MLP(nn.Module):
    """Two-layer MLP with activation."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        activation: Callable = F.gelu,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or 4 * in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class SelfAttention(nn.Module):
    """Self-attention head with optional key_padding_mask support."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        dropout: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: shape (B, T, D)
            key_padding_mask: shape (B, T), True where to mask (skip) positions

        Returns:
            out: shape (B, T, D)
        """
        B, T, D = x.shape

        qkv = self.qkv(x)  # (B, T, 3*D)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = q @ k.transpose(-2, -1) * self.scale  # (B, num_heads, T, T)

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  # (B, 1, 1, T)
                float("-inf"),
            )

        if self.causal:
            causal_mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=x.device),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = attn.masked_fill(torch.isnan(attn), 0.0)
        attn = self.dropout(attn)

        out = attn @ v  # (B, num_heads, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, D)  # (B, T, D)
        out = self.proj(out)
        out = self.dropout(out)

        return out


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN(x) -> Attention -> LN(x) -> MLP."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        bias: bool = True,
        dropout: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()
        self.ln1 = LayerNorm(embed_dim, bias=True)
        self.attn = SelfAttention(
            embed_dim,
            num_heads=num_heads,
            bias=bias,
            dropout=dropout,
            causal=causal,
        )
        self.ln2 = LayerNorm(embed_dim, bias=True)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            out_features=embed_dim,
            activation=F.gelu,
            bias=bias,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x


def _init_by_depth(m: nn.Module, depth_frac: float) -> None:
    """GPT-2 style depth-aware initialization for residual projections."""
    if not isinstance(m, nn.Linear):
        return
    std = math.sqrt(depth_frac / 2)
    nn.init.normal_(m.weight, mean=0, std=std)
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)
