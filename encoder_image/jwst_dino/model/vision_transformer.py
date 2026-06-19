"""Vision Transformer backbone for JWST_DINO.

Self-contained ViT whose size is set entirely from __init__ arguments — there are
no vit_small/base/large presets. Supports iBOT-style masking (masked patches are
replaced by a learnable mask token before the encoder) and DINOv2-style register
tokens. forward returns the CLS embedding and the per-patch token embeddings.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .modules import LayerNorm, PatchEmbed, TransformerBlock


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 72,          # reference (global) crop size that pos_embed is built for
        patch_size: int = 6,
        patch_stride: int = 3,
        in_chans: int = 1,
        embed_dim: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 7,
        drop_path_rate: float = 0.0,
        layerscale_init: float = 1e-5,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.num_register_tokens = num_register_tokens

        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size, patch_stride)
        side = PatchEmbed.num_tokens_per_side(img_size, patch_size, patch_stride)
        self.ref_side = side
        num_patches = side * side

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens > 0 else None
        )
        # Positional embedding covers [CLS, patch tokens]; register tokens get none.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Linearly increasing stochastic-depth rate across blocks (DINOv2 default).
        dpr = [drop_path_rate * i / max(1, depth - 1) for i in range(depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                proj_bias=proj_bias, ffn_bias=ffn_bias, drop_path=dpr[i],
                layerscale_init=layerscale_init,
            )
            for i in range(depth)
        ])
        self.norm = LayerNorm(embed_dim, bias=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.register_tokens is not None:
            nn.init.trunc_normal_(self.register_tokens, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _interpolate_pos_embed(self, side: int) -> Tensor:
        """Return pos_embed resized to a `side`x`side` patch grid (CLS kept as-is)."""
        if side == self.ref_side:
            return self.pos_embed
        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]                                  # (1, ref^2, D)
        d = patch_pos.shape[-1]
        patch_pos = patch_pos.reshape(1, self.ref_side, self.ref_side, d).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(side, side), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, side * side, d)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, x: Tensor, masks: Optional[Tensor] = None) -> dict:
        """Args:
            x:     (B, C, H, W) square image crop.
            masks: (B, N) bool, True = patch is masked (replaced by mask_token).
        Returns dict: {"cls": (B, D), "patch": (B, N, D)}.
        """
        B = x.shape[0]
        side = PatchEmbed.num_tokens_per_side(x.shape[-1], self.patch_size, self.patch_stride)

        tokens = self.patch_embed(x)  # (B, N, D)
        if masks is not None:
            tokens = torch.where(masks.unsqueeze(-1), self.mask_token.to(tokens.dtype), tokens)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1)          # (B, 1+N, D)
        x = x + self._interpolate_pos_embed(side)

        if self.register_tokens is not None:
            reg = self.register_tokens.expand(B, -1, -1)
            x = torch.cat([x[:, :1], reg, x[:, 1:]], dim=1)  # [CLS, REG..., patch...]

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        patch_start = 1 + self.num_register_tokens
        return {"cls": x[:, 0], "patch": x[:, patch_start:]}
