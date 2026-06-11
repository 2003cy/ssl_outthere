"""
LowResPT: Low-Resolution Patch Transformer for spectral MAE pre-training.
"""

import math
from typing import List, Optional, Tuple

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .masking import mask_patches
from .modules import LayerNorm, TransformerBlock #, _init_by_depth


class LowResPT(L.LightningModule):
    """Patch Transformer with wavelength position encoding for low-res spectra."""

    def __init__(
        self,
        embed_dim: int = 64,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.01,
        bias: bool = True,
        # Patch tokenization
        patch_size: int = 4,
        stride: int = 2,
        patch_invalid_threshold: float = 0.25,
        # Masking
        mask_ratio: float = 0.40,
        min_unmasked: int = 4,
        max_line_blocks: int = 3,
        line_prominence: float = 1.3,
        # Candidate pool of strongest detected lines; max_line_blocks of them are
        # masked at random each draw (clamped to >= max_line_blocks). Equal to
        # max_line_blocks → always mask the strongest lines.
        num_line_detect: int = 1,
        # Length C of each masked block. Each selected anchor (line or random)
        # expands into a contiguous run of C valid tokens, anchor at a random
        # position inside it; blocks never overlap. 1 = single-token masking.
        continuous_patch_length: int = 1,
        min_std: float = 0.1,
        # Wavelength positional encoding — observed-frame reference range (µm).
        # Linked from data.wl_ref_min/max via LightningCLI in trainer.py.
        wl_ref_min: float = 1.0,
        wl_ref_max: float = 2.0,
        # Decoder MLP head — hidden layer widths between embed_dim and patch_size.
        # Empty list = linear projection. Default [embed_dim*2] reproduces the
        # original 2-layer GELU head.
        decoder_hidden_dims: Optional[List[int]] = None,
        # Multiplier on the MSE of line-masked tokens. 1.0 = original uniform MSE;
        # >1 amplifies line gradients while leaving continuum gradients unchanged
        # (loss is normalised by pixel count, not weight sum).
        line_loss_weight: float = 1.0,
        # Of the tokens picked by mask_patches (line + random), the fraction
        # that is ACTUALLY zeroed in the encoder input. The remaining
        # (1 - selected_mask_prob) are left visible but STILL contribute to the
        # loss — they teach the head the all-visible regime that val_loss /
        # recon.ipynb evaluates in, and (unlike supervising every unselected
        # token) the subdivision inherits the line/continuum mix of the
        # selection, so line tokens are supervised in both regimes too.
        # 1.0 = original MAE behaviour.
        selected_mask_prob: float = 1.0,
        # If True, concatenate per-patch (mean, std) — computed in globally-
        # normalised flux space — to each patch vector before patch_embed.
        # Mitigates the "regress toward global mean → over/under-predict at
        # blue/red ends" bias by giving each token explicit local stats.
        # (OmniSpectra-style; arXiv:2601.15351.)
        use_patch_stats: bool = False,
        # Optimizer / LR schedule
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.95),
        warmup_steps: int = 400,
        min_lr: float = 5e-6,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Patch embedding: flattened patch vector -> embed_dim.
        # If use_patch_stats, input is [μ_patch, σ_patch] concatenated with the
        # patch flux values → size patch_size + 2.
        patch_in_dim = patch_size + 2 if use_patch_stats else patch_size
        self.patch_embed = nn.Linear(patch_in_dim, embed_dim)

        # CLS token: learnable, injected with per-sample flux stats
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.stats_embed = nn.Linear(2, embed_dim)

        # Transformer encoder
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    bias=bias,
                    dropout=dropout,
                    causal=False,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_ln = LayerNorm(embed_dim, bias=True)

        # MLP decoder head — hidden widths come from `decoder_hidden_dims`.
        # None defaults to [embed_dim * 2] (preserves the original 2-layer head).
        hidden_dims = decoder_hidden_dims if decoder_hidden_dims is not None else [embed_dim * 2]
        head_layers: List[nn.Module] = []
        in_dim = embed_dim
        for h in hidden_dims:
            head_layers.append(nn.Linear(in_dim, h))
            head_layers.append(nn.GELU())
            in_dim = h
        head_layers.append(nn.Linear(in_dim, patch_size))
        self.head = nn.Sequential(*head_layers)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                std = 1.0 / math.sqrt(self.embed_dim)
                nn.init.trunc_normal_(m.weight, std=std, a=-3 * std, b=3 * std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # GPT-2 style depth scaling on residual output projections
        depth_std = 1.0 / math.sqrt(2 * self.num_layers * self.embed_dim)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, std=depth_std)
            nn.init.normal_(block.mlp.fc2.weight, std=depth_std)
            if block.attn.proj.bias is not None:
                nn.init.constant_(block.attn.proj.bias, 0)
            if block.mlp.fc2.bias is not None:
                nn.init.constant_(block.mlp.fc2.bias, 0)

    # ──────────────────────────────────────────────────────────────────────────
    # Static helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _wavelength_positional_encoding(
        self,
        wave_token: Tensor,
        embed_dim: int,
        token_valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Fourier positional encoding from per-token mean wavelength.

        Uses a GLOBAL [wl_ref_min, wl_ref_max] (hparams) normalisation so the
        same physical wavelength maps to the same encoding across all samples.

        Args:
            wave_token:        (B, N) mean wavelength per token (observed-frame, µm).
            embed_dim:         Output feature dimension.
            token_valid_mask:  (B, N) bool; unused (kept for API compatibility).

        Returns:
            pos_emb: (B, N, embed_dim)
        """
        device = wave_token.device
        wl_min = self.hparams.wl_ref_min
        wl_max = self.hparams.wl_ref_max
        wl_norm = (wave_token - wl_min) / (wl_max - wl_min)  # ~[0,1] inside ref range

        num_freqs = embed_dim // 2
        # Frequencies 1–100: matches the ~28-token sequence length; the old
        # logspace(0, 4) reached 10^4, which only produced aliasing noise.
        freqs = torch.logspace(0, 2, num_freqs, device=device)  # (num_freqs,)

        # (B, N, 1) * (1, 1, num_freqs) → (B, N, num_freqs)
        phase = 2 * np.pi * wl_norm.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)
        pos_emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)  # (B, N, embed_dim)

        if pos_emb.shape[-1] > embed_dim:
            pos_emb = pos_emb[..., :embed_dim]
        elif pos_emb.shape[-1] < embed_dim:
            pos_emb = F.pad(pos_emb, (0, embed_dim - pos_emb.shape[-1]))

        return pos_emb

    @staticmethod
    def _normalize_flux(
        flux: Tensor,
        valid_mask: Tensor,
        min_std: float = 0.1,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Per-sample normalization using valid pixels only.

        Returns:
            flux_norm: (B, L) normalized to ~N(0,1) at valid positions, 0 elsewhere.
            mean:      (B, 1)
            std:       (B, 1) clipped to min_std
        """
        valid_float = valid_mask.float()
        valid_count = valid_float.sum(dim=1, keepdim=True).clamp(min=1)

        mean = (flux * valid_float).sum(dim=1, keepdim=True) / valid_count
        var  = (((flux - mean) ** 2) * valid_float).sum(dim=1, keepdim=True) / valid_count
        std  = var.sqrt().clamp(min=min_std)

        flux_norm = (flux - mean) / std * valid_float
        return flux_norm, mean, std

    def _patchify(
        self,
        flux_norm: Tensor,
        wavelength: Tensor,
        valid_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, int]:
        """Extract overlapping patches, dropping any trailing incomplete patch.

        Returns:
            patches:          (B, N, P)
            wave_token:       (B, N)   mean rest-frame wavelength per token
            valid_patches:    (B, N, P) bool
            token_valid_mask: (B, N)   bool  (invalid_ratio <= threshold)
            L_used:           int       number of pixels actually used
        """
        P = self.hparams.patch_size
        S = self.hparams.stride
        L = flux_norm.shape[1]

        N = (L - P) // S + 1
        L_used = (N - 1) * S + P  # drop trailing pixels to make N exact

        patches       = flux_norm[:, :L_used].unfold(1, P, S)        # (B, N, P)
        wave_patches  = wavelength[:, :L_used].unfold(1, P, S)       # (B, N, P)
        valid_patches = valid_mask[:, :L_used].unfold(1, P, S)       # (B, N, P)

        # Token-level valid mask: token invalid if > threshold pixels are bad
        invalid_ratio    = (~valid_patches).float().mean(-1)          # (B, N)
        token_valid_mask = invalid_ratio <= self.hparams.patch_invalid_threshold  # (B, N)

        # Mean wavelength per token (weighted by valid pixels)
        w_sum    = (wave_patches * valid_patches.float()).sum(-1)
        w_count  = valid_patches.float().sum(-1).clamp(min=1)
        wave_token = w_sum / w_count                                  # (B, N)

        return patches, wave_token, valid_patches, token_valid_mask, L_used

    def _compute_patch_stats(
        self,
        patches: Tensor,
        valid_patches: Tensor,
    ) -> Tensor:
        """Per-patch (mean, std) over valid pixels of each patch.

        Input `patches` is already in globally-normalised flux space, so these
        stats describe each patch's local deviation from the spectrum mean (=0).

        Returns:
            patch_stats: (B, N, 2) where last dim is [μ_patch, σ_patch].
        """
        vp_f   = valid_patches.float()
        cnt    = vp_f.sum(-1).clamp(min=1)
        mu_p   = (patches * vp_f).sum(-1) / cnt
        var_p  = ((patches - mu_p.unsqueeze(-1)) ** 2 * vp_f).sum(-1) / cnt
        sig_p  = var_p.sqrt().clamp(min=self.hparams.min_std)
        return torch.stack([mu_p, sig_p], dim=-1)

    # ──────────────────────────────────────────────────────────────────────────
    # Forward / encode
    # ──────────────────────────────────────────────────────────────────────────

    def encode(
        self,
        patches: Tensor,
        wave_token: Tensor,
        token_valid_mask: Tensor,
        stats: Tensor,
        patch_stats: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode a batch of (possibly masked) patches into token embeddings.

        Args:
            patches:          (B, N, P) normalised patch vectors (masked tokens zeroed)
            wave_token:       (B, N)    mean wavelength per token
            token_valid_mask: (B, N)    bool, True = valid token
            stats:            (B, 2)    [mean, std] per-sample normalization stats
            patch_stats:      (B, N, 2) optional per-patch (mean, std). Required
                              when self.hparams.use_patch_stats is True. Caller
                              is responsible for zeroing stats of MASKED tokens
                              (label-leakage prevention).

        Returns:
            x: (B, N+1, D) — index 0 is CLS, 1..N are patch tokens
        """
        B, N, P = patches.shape

        # Optionally concatenate per-patch stats to each patch vector.
        # If caller didn't provide stats, derive them from the patch values
        # directly (invalid pixels are already 0 in normalised flux space).
        # Training paths pass an explicit, validity-aware, mask-zeroed version
        # via `patch_stats=...`; inference paths can rely on this auto-compute.
        if self.hparams.use_patch_stats:
            if patch_stats is None:
                mu_p  = patches.mean(dim=-1)                                   # (B, N)
                sig_p = patches.std(dim=-1, unbiased=False).clamp(
                    min=self.hparams.min_std
                )                                                              # (B, N)
                patch_stats = torch.stack([mu_p, sig_p], dim=-1)               # (B, N, 2)
            patches = torch.cat([patches, patch_stats], dim=-1)  # (B, N, P+2)

        # Patch embedding + positional encoding
        patch_tokens = self.patch_embed(patches)  # (B, N, D)
        pos_emb      = self._wavelength_positional_encoding(
            wave_token, self.embed_dim, token_valid_mask
        )  # (B, N, D)
        tokens = patch_tokens + pos_emb           # (B, N, D)

        # CLS token: learnable + stats injection
        cls_tokens = self.cls_token.expand(B, -1, -1) + self.stats_embed(stats).unsqueeze(1)

        # Sequence: [CLS, patch tokens]
        x = torch.cat([cls_tokens, tokens], dim=1)  # (B, N+1, D)

        # Attention mask: CLS always valid; invalid tokens excluded
        cls_mask         = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        key_padding_mask = torch.cat([cls_mask, ~token_valid_mask], dim=1)  # (B, N+1)

        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)

        return self.final_ln(x)  # (B, N+1, D)

    def forward(
        self,
        patches: Tensor,
        wave_token: Tensor,
        token_valid_mask: Tensor,
        stats: Tensor,
        patch_stats: Optional[Tensor] = None,
    ) -> dict:
        """Full encode + decode pass.

        Returns dict with:
            recon_patches    (B, N, P)   per-patch flux reconstruction
            cls_embedding    (B, D)      global CLS token
            token_embeddings (B, N, D)   per-token encoder outputs
        """
        x = self.encode(patches, wave_token, token_valid_mask, stats, patch_stats)

        cls_embedding    = x[:, 0, :]   # (B, D)
        token_embeddings = x[:, 1:, :]  # (B, N, D)
        recon_patches    = self.head(token_embeddings)  # (B, N, P)

        return {
            "recon_patches":    recon_patches,
            "cls_embedding":    cls_embedding,
            "token_embeddings": token_embeddings,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Training / validation
    # ──────────────────────────────────────────────────────────────────────────

    def training_step(self, batch: dict, batch_idx: int = 0) -> Tensor:
        flux       = batch["flux"]        # (B, L)
        wavelength = batch["wavelength"]  # (B, L) — already rest-frame from dataset
        valid_mask = batch["valid_mask"]  # (B, L)

        # Per-sample flux normalization
        flux_norm, mean, std = self._normalize_flux(
            flux, valid_mask, self.hparams.min_std
        )
        stats = torch.cat([mean, std], dim=-1)  # (B, 2)

        # Patchify
        patches, wave_token, valid_patches, token_valid_mask, L_used = self._patchify(
            flux_norm, wavelength, valid_mask
        )
        target = patches.clone()  # (B, N, P) — reconstruction target
        patch_stats = self._compute_patch_stats(patches, valid_patches)  # (B,N,2)

        # Line-aware masking at token level — selects which tokens enter the
        # loss (line + random). Discard mask_patches' own zeroed patches and
        # rebuild below after subdividing the selection into hidden vs visible.
        _, sel_token, line_mask_token = mask_patches(
            patches,
            flux_norm[:, :L_used],
            valid_patches,
            token_valid_mask,
            mask_ratio      = self.hparams.mask_ratio,
            min_unmasked    = self.hparams.min_unmasked,
            max_line_blocks = self.hparams.max_line_blocks,
            line_prominence = self.hparams.line_prominence,
            patch_size      = self.hparams.patch_size,
            stride          = self.hparams.stride,
            continuous_patch_length = self.hparams.continuous_patch_length,
            num_line_detect = self.hparams.num_line_detect,
        )

        # Subdivide the selected tokens into "hidden" (encoder input zeroed)
        # and "visible" (kept in input). Loss covers BOTH subsets, so the head
        # is trained in the masked AND the all-visible regimes on the same
        # line/continuum mix the selection produced — line tokens get visible-
        # regime supervision too (unlike "supervise all unselected" schemes).
        p_hide = self.hparams.selected_mask_prob
        if p_hide >= 1.0:
            hide_token = sel_token
        else:
            coin       = torch.rand_like(sel_token, dtype=torch.float)
            hide_token = sel_token & (coin < p_hide)
        vis_token = sel_token & ~hide_token

        # Build encoder input: zero only the hidden subset; patch_stats of
        # hidden tokens also zeroed to prevent leakage of patch mean/std.
        masked_patches     = torch.where(hide_token.unsqueeze(-1),
                                         torch.zeros_like(patches), patches)
        masked_patch_stats = patch_stats * (~hide_token).unsqueeze(-1).float()

        out = self.forward(masked_patches, wave_token, token_valid_mask, stats,
                           patch_stats=masked_patch_stats)
        recon_patches = out["recon_patches"]  # (B, N, P)

        # Loss: MSE on ALL selected & valid tokens (hidden + visible), with
        # extra weight on line tokens. Normalise by the unweighted pixel count
        # so the continuum gradient magnitude is unchanged.
        loss_mask     = (sel_token & token_valid_mask)                   # (B, N)
        loss_mask_px  = loss_mask.unsqueeze(-1).expand_as(recon_patches) # (B, N, P)
        line_weight   = self.hparams.line_loss_weight
        token_w       = torch.where(line_mask_token, line_weight, 1.0)   # (B, N)
        weight_px     = (loss_mask.float() * token_w).unsqueeze(-1)      # (B, N, 1)

        n_loss = loss_mask_px.sum().clamp(min=1)
        loss = ((recon_patches - target) ** 2 * weight_px).sum() / n_loss

        # Full-sequence MSE (no_grad): same metric as val_loss, for direct comparison
        with torch.no_grad():
            valid_mask_px = token_valid_mask.unsqueeze(-1).expand_as(recon_patches)
            n_valid_px    = valid_mask_px.sum().clamp(min=1)
            train_full_mse = F.mse_loss(
                recon_patches * valid_mask_px,
                target * valid_mask_px,
                reduction="sum",
            ) / n_valid_px

        # Split metrics — diagnostics on the selected-token loss pool.
        # line vs continuum: same as before, on the full selection.
        # hidden vs visible: lets you see masked-regime vs all-visible-regime
        # MSE on identically-distributed tokens (the core point of the split).
        with torch.no_grad():
            cont_mask    = loss_mask & ~line_mask_token                    # (B, N)
            sq_per_token = ((recon_patches - target) ** 2).mean(-1)        # (B, N)
            n_line       = line_mask_token.sum().clamp(min=1).float()
            n_cont       = cont_mask.sum().clamp(min=1).float()
            train_line_mse = (sq_per_token * line_mask_token.float()).sum() / n_line
            train_cont_mse = (sq_per_token * cont_mask.float()).sum() / n_cont

            n_hid = hide_token.sum().clamp(min=1).float()
            n_vis = vis_token.sum().clamp(min=1).float()
            train_hid_mse = (sq_per_token * hide_token.float()).sum() / n_hid
            train_vis_mse = (sq_per_token * vis_token.float()).sum() / n_vis

        # Metrics — all on_step=True, on_epoch=False for clean per-step CSV rows
        n_sel    = sel_token.sum().float()
        n_hidden = hide_token.sum().float()
        n_valid  = token_valid_mask.sum().float()
        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log("train_loss",        loss,              prog_bar=True, on_step=True, on_epoch=False)
        self.log("train_full_mse",    train_full_mse,    prog_bar=False, on_step=True, on_epoch=False)
        self.log("train_line_mse",    train_line_mse,    prog_bar=False, on_step=True, on_epoch=False)
        self.log("train_cont_mse",    train_cont_mse,    prog_bar=False, on_step=True, on_epoch=False)
        self.log("train_hid_mse",     train_hid_mse,     prog_bar=False, on_step=True, on_epoch=False)
        self.log("train_vis_mse",     train_vis_mse,     prog_bar=False, on_step=True, on_epoch=False)
        self.log("lr",                current_lr,        on_step=True,  on_epoch=False)
        self.log("selected_tokens",   n_sel,             on_step=True,  on_epoch=False)
        self.log("hidden_tokens",     n_hidden,          on_step=True,  on_epoch=False)
        self.log("valid_tokens",      n_valid,           on_step=True,  on_epoch=False)
        self.log("hide_ratio_actual", n_hidden / n_sel.clamp(min=1), on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch: dict, batch_idx: int = 0) -> Tensor:
        flux       = batch["flux"]
        wavelength = batch["wavelength"]
        valid_mask = batch["valid_mask"]

        flux_norm, mean, std = self._normalize_flux(
            flux, valid_mask, self.hparams.min_std
        )
        stats = torch.cat([mean, std], dim=-1)

        patches, wave_token, valid_patches, token_valid_mask, L_used = self._patchify(
            flux_norm, wavelength, valid_mask
        )
        target = patches.clone()
        patch_stats = self._compute_patch_stats(patches, valid_patches)

        # Mirror training: line + random selection, then hide/vis split. Mask
        # decisions are deterministic per (epoch, batch_idx) so val metrics are
        # reproducible across runs / checkpoints (no run-to-run noise from RNG).
        _, sel_token, line_mask_token = mask_patches(
            patches, flux_norm[:, :L_used], valid_patches, token_valid_mask,
            mask_ratio      = self.hparams.mask_ratio,
            min_unmasked    = self.hparams.min_unmasked,
            max_line_blocks = self.hparams.max_line_blocks,
            line_prominence = self.hparams.line_prominence,
            patch_size      = self.hparams.patch_size,
            stride          = self.hparams.stride,
            continuous_patch_length = self.hparams.continuous_patch_length,
            num_line_detect = self.hparams.num_line_detect,
        )

        p_hide = self.hparams.selected_mask_prob
        if p_hide >= 1.0:
            hide_token = sel_token
        else:
            g = torch.Generator(device=sel_token.device).manual_seed(
                int(self.current_epoch) * 10_000 + int(batch_idx)
            )
            coin = torch.rand(sel_token.shape, generator=g, device=sel_token.device)
            hide_token = sel_token & (coin < p_hide)
        vis_token = sel_token & ~hide_token

        masked_patches     = torch.where(hide_token.unsqueeze(-1),
                                         torch.zeros_like(patches), patches)
        masked_patch_stats = patch_stats * (~hide_token).unsqueeze(-1).float()

        out = self.forward(masked_patches, wave_token, token_valid_mask, stats,
                           patch_stats=masked_patch_stats)
        recon_patches = out["recon_patches"]

        # Unweighted MSE — val is an unbiased physical metric, no line_loss_weight.
        sq_per_token = ((recon_patches - target) ** 2).mean(-1)             # (B, N)
        sel          = sel_token & token_valid_mask
        n_sel = sel.sum().clamp(min=1).float()
        n_hid = hide_token.sum().clamp(min=1).float()
        n_vis = vis_token.sum().clamp(min=1).float()

        val_loss     = (sq_per_token * sel.float()).sum() / n_sel
        val_hid_loss = (sq_per_token * hide_token.float()).sum() / n_hid
        val_vis_loss = (sq_per_token * vis_token.float()).sum() / n_vis

        self.log("val_loss",     val_loss,     prog_bar=True,  on_step=False, on_epoch=True)
        self.log("val_hid_loss", val_hid_loss, prog_bar=True,  on_step=False, on_epoch=True)
        self.log("val_vis_loss", val_vis_loss, prog_bar=False, on_step=False, on_epoch=True)
        return val_hid_loss

    def test_step(self, batch: dict, batch_idx: int = 0) -> Tensor:
        return self.validation_step(batch, batch_idx)

    # ──────────────────────────────────────────────────────────────────────────
    # Optimizer with warmup-cosine LR schedule
    # ──────────────────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        opt = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=tuple(self.hparams.betas),
        )

        total_steps   = self.trainer.estimated_stepping_batches
        warmup_steps  = self.hparams.warmup_steps
        min_lr_ratio  = self.hparams.min_lr / self.hparams.lr

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = LambdaLR(opt, lr_lambda)
        return {
            "optimizer":    opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Downstream embedding API
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def embed(
        self,
        flux: Tensor,
        wavelength: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        """Extract CLS embedding for downstream tasks.

        Args:
            flux, wavelength, valid_mask: (B, L) — wavelength already rest-frame.

        Returns:
            cls_embedding: (B, embed_dim)
        """
        flux_norm, mean, std = self._normalize_flux(
            flux, valid_mask, self.hparams.min_std
        )
        stats   = torch.cat([mean, std], dim=-1)
        patches, wave_token, valid_patches, token_valid_mask, _ = self._patchify(
            flux_norm, wavelength, valid_mask
        )
        patch_stats = self._compute_patch_stats(patches, valid_patches)
        x = self.encode(patches, wave_token, token_valid_mask, stats,
                        patch_stats=patch_stats)
        return x[:, 0, :]  # CLS token: (B, D)

    @torch.no_grad()
    def reconstruct(
        self,
        flux: Tensor,
        wavelength: Tensor,
        valid_mask: Tensor,
        masked: bool = False,
        block_k: int = 1,
    ) -> dict:
        """Reconstruct spectra from raw input under one of two protocols.

        Shared prefix: normalize → patchify → encode → decode, returning the
        intermediates a caller needs to map patches back to pixel space
        (overlap averaging).

        masked=False (full-visible):
            Every token is visible; one forward pass reconstructs all patches.
            With use_patch_stats=True, `forward` auto-computes per-patch stats
            from the all-visible patches — matching the all-visible regime of
            training/validation.

        masked=True (MAE-consistent, leave-block-out):
            For each token t, a block of `block_k` consecutive tokens centred on
            t is hidden from the encoder (patch values AND per-patch stats zeroed,
            to prevent stat leakage), and only that token's prediction is kept.
            Leak-free iff the block fully hides t's pixels (e.g. block_k=1 when
            stride == patch_size; use 2–3 when stride < patch_size).

        Args:
            flux, wavelength, valid_mask: (B, L) — observed-frame wavelength (µm).
            masked:  select the masked (True) or full-visible (False) protocol.
            block_k: number of consecutive tokens hidden per target (masked only).

        Returns:
            recon_patches:    (B, N, P) reconstructed patches (normalised flux space)
            target:           (B, N, P) input patches = the reconstruction target
            flux_norm:        (B, L)    per-pixel normalised input flux
            wave_token:       (B, N)    mean wavelength per token
            valid_patches:    (B, N, P) bool, per-pixel patch validity
            token_valid_mask: (B, N)    bool, per-token validity
            stats:            (B, 2)    per-sample [mean, std]
            L_used:           int       pixels used (trailing partial patch dropped)
        """
        flux_norm, mean, std = self._normalize_flux(
            flux, valid_mask, self.hparams.min_std
        )
        stats = torch.cat([mean, std], dim=-1)
        patches, wave_token, valid_patches, token_valid_mask, L_used = self._patchify(
            flux_norm, wavelength, valid_mask
        )

        if not masked:
            recon_patches = self.forward(
                patches, wave_token, token_valid_mask, stats
            )["recon_patches"]
        else:
            B, N, P = patches.shape
            patch_stats = self._compute_patch_stats(patches, valid_patches)
            half_lo = (block_k - 1) // 2
            half_hi = block_k // 2
            recon_patches = torch.zeros(B, N, P, device=patches.device,
                                        dtype=patches.dtype)
            for t in range(N):
                tlo = max(0, t - half_lo)
                thi = min(N - 1, t + half_hi)
                mp = patches.clone();     mp[:, tlo:thi + 1, :] = 0.0
                ps = patch_stats.clone(); ps[:, tlo:thi + 1, :] = 0.0
                out = self.forward(mp, wave_token, token_valid_mask, stats,
                                   patch_stats=ps)
                recon_patches[:, t, :] = out["recon_patches"][:, t, :]

        return {
            "recon_patches":    recon_patches,
            "target":           patches,
            "flux_norm":        flux_norm,
            "wave_token":       wave_token,
            "valid_patches":    valid_patches,
            "token_valid_mask": token_valid_mask,
            "stats":            stats,
            "L_used":           L_used,
        }
