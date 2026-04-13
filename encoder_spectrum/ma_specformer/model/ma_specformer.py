"""Multi-band SpecFormer: point-wise spectrum encoding with wavelength-based position embeddings.

Key features:
- Each flux value is a single token (no patching)
- Wavelength-based continuous position encoding (Fourier features)
- Explicit train_mask + valid_mask + loss_mask for robust masked reconstruction training
"""

import math
from typing import Optional, Tuple

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .masking import mask_input
from .modules import LayerNorm, TransformerBlock, _init_by_depth


class MASpecFormer(L.LightningModule):
    """Point-wise spectrum Transformer with wavelength position encoding."""

    def __init__(
        self,
        embed_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.15,       # Total fraction of valid pixels to mask
        min_unmasked: int = 1,          # Minimum unmasked tokens (to preserve some context)
        dropout: float = 0.1,
        bias: bool = True,
        min_std: float = 0.1,           # Minimum std for normalization
        max_line_blocks: int = 3,       # Mask up to this many emission-line centered blocks (0 = pure random)
        line_block_size: int = 3,       # Pixels per line block, centered on peak (1=peak only, 3=peak±1, …)
        line_prominence: float = 1.0,   # scipy find_peaks prominence threshold on normalised flux
        cls_aux_weight: float = 0.1,    # Weight for CLS auxiliary loss (0 = disabled)
        redshift_corr: bool = False,    # If True, convert wavelengths to rest-frame before encoding
    ):
        """
        Args:
            embed_dim: Embedding dimension
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: Hidden layer ratio in MLP
            mask_ratio: Total fraction of valid pixels to mask per spectrum
            min_unmasked: Minimum number of unmasked tokens per sample
            dropout: Dropout rate
            bias: Whether to use bias in linear layers
            min_std: Minimum std for normalization to avoid division by small numbers
            max_line_blocks: Max emission-line blocks to place (by prominence); 0 = pure random
            line_block_size: Width of each centered block around a detected line peak
            line_prominence: Prominence threshold for scipy.signal.find_peaks on normalised flux
        """
        super().__init__()
        self.save_hyperparameters()

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        

        # Token embedding: normalized flux only -> embed_dim
        self.token_embed = nn.Linear(1, embed_dim)
        
        # CLS token: learnable embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Stats embedding: project mean/std to embed_dim for CLS token
        self.stats_embed = nn.Linear(2, embed_dim)

        # Position encoding: Fourier features from wavelength
        # We'll create continuous encodings at forward time (see _wavelength_positional_encoding)

        # Transformer blocks
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

        # Reconstruction head: embed_dim -> 1 (reconstruct flux)
        self.head = nn.Linear(embed_dim, 1, bias=True)

        # CLS reconstruction head: decodes the full spectrum from CLS + wavelength position.
        #
        # At each position p, the decode input is:
        #   cat([cls_embedding (B, D), pos_emb[p] (D)]) → (B, 2D)
        #
        # Concatenation (not addition) keeps global context and positional encoding
        # in separate channels, so the first linear layer can independently weight
        # each source before computing cross-term interactions via GELU.  Addition
        # collapses both into one D-dim vector and forces the MLP to implicitly
        # disentangle them, which limits expressiveness for position-dependent
        # features like emission lines.
        #
        # Loss is computed on ALL valid positions (not just masked ones), which
        # forces CLS to encode the complete spectral shape rather than just a
        # summary statistic.  This acts as a true information bottleneck:
        # the only path from the full spectrum to a per-position prediction is
        # through the CLS token.
        self.cls_recon_head = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim, bias=True),
            nn.GELU(),
            nn.Linear(embed_dim, 1, bias=True),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                std = 1 / math.sqrt(self.embed_dim)
                nn.init.trunc_normal_(m.weight, std=std, a=-3 * std, b=3 * std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize CLS token
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # GPT-2 style depth-aware scaling: apply ONLY to the two residual output
        # projections per block (attn.proj and mlp.fc2).  Formula:
        #   std = base_std / sqrt(2 * num_layers)
        #       = (1/sqrt(embed_dim)) / sqrt(2*N)
        #       = 1 / sqrt(2 * N * embed_dim)
        # This gives a value strictly smaller than base_std for any N >= 1,
        # dampening the residual stream at init and improving stability.
        depth_std = 1.0 / math.sqrt(2 * self.num_layers * self.embed_dim)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, std=depth_std)
            nn.init.normal_(block.mlp.fc2.weight, std=depth_std)
            if block.attn.proj.bias is not None:
                nn.init.constant_(block.attn.proj.bias, 0)
            if block.mlp.fc2.bias is not None:
                nn.init.constant_(block.mlp.fc2.bias, 0)

    @staticmethod
    def _wavelength_positional_encoding(
        wavelengths: Tensor,
        embed_dim: int,
        device: torch.device,
        valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Fourier position encoding from wavelength values.

        Args:
            wavelengths: shape (B, T) or (T,), wavelength values
            embed_dim: Output dimension
            device: Device to place result on
            valid_mask: shape (B, T) bool, True = valid pixel.  When provided,
                        min/max are computed only over valid positions so that
                        batch-padding zeros do not corrupt the normalisation.

        Returns:
            pos_emb: shape (..., embed_dim), positional encoding
        """
        # Normalize wavelengths to [0, 1] per-sample for stable Fourier encoding.
        # Per-sample (not batch-global) so that each spectrum's position encoding
        # is independent of other spectra in the batch (critical for variable-length data).
        # Use valid_mask to exclude padding zeros from min/max when available.
        if wavelengths.dim() == 2:  # (B, T)
            if valid_mask is not None:
                # Replace invalid positions with +inf / -inf before taking min/max
                # so that padding zeros don't drag wl_min down to 0.
                _large = wavelengths.new_full(wavelengths.shape, float("inf"))
                _small = wavelengths.new_full(wavelengths.shape, float("-inf"))
                wl_min = torch.where(valid_mask, wavelengths, _large).min(dim=1, keepdim=True).values
                wl_max = torch.where(valid_mask, wavelengths, _small).max(dim=1, keepdim=True).values
            else:
                wl_min = wavelengths.min(dim=1, keepdim=True).values  # (B, 1)
                wl_max = wavelengths.max(dim=1, keepdim=True).values  # (B, 1)
        else:  # (T,)
            wl_min = wavelengths.min()
            wl_max = wavelengths.max()
        wl_norm = (wavelengths - wl_min) / (wl_max - wl_min + 1e-8)  # [0, 1]

        # Fourier features: sin/cos of different frequencies.
        # Upper limit 1e4: beyond that, float32 sin/cos precision degrades and
        # the high-frequency dimensions collapse to numerical noise.
        num_freqs = embed_dim // 2
        freqs = torch.logspace(0, 4, num_freqs, device=device)  # [1, ..., 1e4]

        # Expand to match shape
        if wavelengths.dim() == 2:  # (B, T)
            B, T = wavelengths.shape
            wl_norm_expanded = wl_norm.unsqueeze(-1)  # (B, T, 1)
            freqs_expanded = freqs.unsqueeze(0).unsqueeze(0)  # (1, 1, num_freqs)
        else:  # (T,)
            T = wavelengths.shape[0]
            wl_norm_expanded = wl_norm.unsqueeze(-1)  # (T, 1)
            freqs_expanded = freqs.unsqueeze(0)  # (1, num_freqs)

        # Compute sin/cos
        phase = 2 * np.pi * wl_norm_expanded * freqs_expanded
        sin_feats = torch.sin(phase)
        cos_feats = torch.cos(phase)

        # Concatenate sin and cos
        pos_emb = torch.cat([sin_feats, cos_feats], dim=-1)  # (..., 2*num_freqs)

        # Trim/pad to exactly embed_dim
        if pos_emb.shape[-1] > embed_dim:
            pos_emb = pos_emb[..., :embed_dim]
        elif pos_emb.shape[-1] < embed_dim:
            pad_size = embed_dim - pos_emb.shape[-1]
            pos_emb = F.pad(pos_emb, (0, pad_size), mode="constant", value=0)

        return pos_emb

    def encode(
        self,
        flux: Tensor,
        wavelengths: Tensor,
        valid_mask: Optional[Tensor] = None,
        stats: Optional[Tensor] = None,  # (B, 2) with [mean, std] if provided
        return_cls_only: bool = False,
        _pos_emb: Optional[Tensor] = None,  # Pre-computed pos encoding; computed internally if None
    ) -> Tensor:
        """Encode flux and wavelengths into embeddings.
        
        This method can be used directly to extract embeddings for downstream tasks.
        Inputs must already be normalized per sample; the model no longer
        performs normalization internally.
        
        Args:
            flux: shape (B, T), normalized flux values
            wavelengths: shape (B, T), wavelength values
            valid_mask: shape (B, T), bool (True = valid, False = invalid/padding)
                        Invalid positions are masked in attention (cannot be attended to)
            stats: shape (B, 2), per-sample [mean, std] stats to inject into CLS.
                   Defaults to zeros when not provided.
            return_cls_only: If True, return only CLS token embedding (B, embed_dim)
        
        Returns:
            If return_cls_only=False:
                embeddings: shape (B, T+1, embed_dim), transformer output embeddings
                            Index 0 is CLS token, indices 1: are flux tokens
            If return_cls_only=True:
                cls_embedding: shape (B, embed_dim), global spectrum embedding
        """
        B, T = flux.shape
        flux_norm = flux

        # Embed flux tokens: normalized_flux -> (B, T, 1) -> (B, T, embed_dim)
        flux_embedded = self.token_embed(flux_norm.unsqueeze(-1))

        # Wavelength-based position encoding: (B, T, embed_dim)
        # Use pre-computed pos_emb if provided (avoids redundant computation in forward())
        if _pos_emb is None:
            _pos_emb = self._wavelength_positional_encoding(
                wavelengths, self.embed_dim, flux.device, valid_mask=valid_mask
            )
        pos_emb = _pos_emb

        # Combine flux embeddings with position encoding
        flux_tokens = flux_embedded + pos_emb  # (B, T, embed_dim)

        # Create CLS token with stats information
        # CLS token = learnable embedding + stats embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        if stats is None:
            stats = torch.zeros(B, 2, device=flux.device)
        stats_emb = self.stats_embed(stats).unsqueeze(1)  # (B, 1, embed_dim)
        cls_tokens = cls_tokens + stats_emb  # (B, 1, embed_dim)

        # Prepend CLS token to sequence
        x = torch.cat([cls_tokens, flux_tokens], dim=1)  # (B, T+1, embed_dim)

        # Create key_padding_mask for attention: True = mask out (invalid positions)
        # CLS token is always valid (False = not masked)
        if valid_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=flux.device)
            key_padding_mask = torch.cat([cls_mask, ~valid_mask], dim=1)  # (B, T+1)
        else:
            key_padding_mask = None

        # Apply transformer blocks with attention mask
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)

        # Final layer norm
        x = self.final_ln(x)  # (B, T+1, embed_dim)

        if return_cls_only:
            return x[:, 0, :]  # (B, embed_dim) - CLS token only
        return x

    def forward(
        self,
        flux: Tensor,
        wavelengths: Tensor,
        valid_mask: Optional[Tensor] = None,
        stats: Optional[Tensor] = None,
    ) -> dict:
        """Full encode + decode pass.

        Args:
            flux:        (B, T) normalized flux (already mean/std normalised by caller)
            wavelengths: (B, T) wavelength values
            valid_mask:  (B, T) bool, True = valid pixel
            stats:       (B, 2) per-sample [mean, std]; zeros if not provided

        Returns dict with:
            reconstructions  (B, T)       token-level flux predictions
            cls_recons       (B, T)       CLS global reconstruction at each wavelength
            cls_embedding    (B, D)       global spectrum embedding (CLS token)
            token_embeddings (B, T, D)    per-token transformer outputs
        """
        # Compute pos_emb once here so both encode() and cls_recon_head can use it
        pos_emb = self._wavelength_positional_encoding(wavelengths, self.embed_dim, flux.device, valid_mask=valid_mask)

        x = self.encode(flux, wavelengths, valid_mask, stats=stats, _pos_emb=pos_emb)  # (B, T+1, D)

        cls_embedding    = x[:, 0, :]   # (B, D)
        token_embeddings = x[:, 1:, :]  # (B, T, D)

        # Token-level reconstruction (local context)
        reconstructions = self.head(token_embeddings).squeeze(-1)  # (B, T)

        # CLS global reconstruction: cat(cls, pos_emb) → flux at each wavelength
        cls_expanded = cls_embedding.unsqueeze(1).expand(-1, pos_emb.shape[1], -1)  # (B, T, D)
        cls_decode   = torch.cat([cls_expanded, pos_emb], dim=-1)                   # (B, T, 2D)
        cls_recons   = self.cls_recon_head(cls_decode).squeeze(-1)                  # (B, T)

        return {
            "reconstructions":  reconstructions,
            "cls_recons":       cls_recons,
            "cls_embedding":    cls_embedding,
            "token_embeddings": token_embeddings,
        }

    def training_step(self, batch: dict) -> Tensor:
        """Training step with masked reconstruction (BERT-style).
        
        Mask strategy:
        - valid_mask: data quality mask (True = valid) → used in attention (invalid positions not attended)
        - train_mask: random mask for reconstruction (True = masked) → only affects input values
        - loss_mask = train_mask & valid_mask → both token and CLS losses computed here only
        - Target is NORMALIZED flux (stable training)
        
        Batch dict contains:
            - flux: shape (B, T)
            - wavelength: shape (B, T)
            - valid_mask: shape (B, T), bool (True = valid)
        """
        flux = batch["flux"]  # (B, T)
        wavelength = batch["wavelength"]  # (B, T)
        valid_mask = batch["valid_mask"]  # (B, T) bool

        # Redshift correction: use rest-frame wavelengths for positional encoding
        if self.hparams.redshift_corr:
            wavelength = self._apply_redshift_corr(wavelength, batch)

        # Normalize flux per-sample
        flux_norm, mean, std = self._normalize_flux(flux, valid_mask)

        # Target is NORMALIZED flux (for stable training)
        target = flux_norm.clone() # (B, T)
        stats = torch.cat([mean, std], dim=-1)  # (B, 2)

        # Apply random mask to input (zero out masked positions among valid ones)
        input_flux, train_mask = self._mask_input(flux_norm, valid_mask)

        out = self.forward(input_flux, wavelength, valid_mask, stats=stats)
        reconstructions = out["reconstructions"]  # (B, T)
        cls_recons      = out["cls_recons"]       # (B, T)

        # loss_mask: positions that are both train-masked AND valid
        loss_mask = (train_mask & valid_mask).float()  # (B, T)

        # Token reconstruction loss — masked valid positions only
        if loss_mask.sum() > 0:
            recon_loss = F.mse_loss(
                reconstructions * loss_mask,
                target * loss_mask,
                reduction="sum",
            ) / loss_mask.sum().clamp(min=1)
        else:
            recon_loss = torch.tensor(0.0, device=flux.device)

        # CLS global reconstruction loss — ALL valid positions (not just masked).
        # cls_recon_head uses only [CLS, pos_emb] as input (no token embeddings),
        # so CLS is the sole information channel.  Computing the loss on all valid
        # positions forces CLS to encode the complete spectral template, which is
        # the representation needed for downstream redshift prediction.
        # This gives CLS a distinct objective from the token head (masked-only).
        valid_mask_float = valid_mask.float()
        cls_loss = torch.tensor(0.0, device=flux.device)
        if self.hparams.cls_aux_weight > 0 and valid_mask_float.sum() > 0:
            cls_loss = F.mse_loss(
                cls_recons * valid_mask_float,
                target * valid_mask_float,
                reduction="sum",
            ) / valid_mask_float.sum().clamp(min=1)

        loss = recon_loss + self.hparams.cls_aux_weight * cls_loss

        # === Additional metrics for monitoring ===
        B, T = flux.shape
        
        # Mask statistics
        mask_count = train_mask.sum().float()
        valid_count = valid_mask.sum().float()
        mask_ratio_actual = mask_count / valid_count.clamp(min=1)
        
        # Per-sample reconstruction error (for variance analysis)
        with torch.no_grad():
            per_pos_error = ((reconstructions - target) ** 2) * loss_mask
            if loss_mask.sum() > 0:
                recon_error_std = per_pos_error[loss_mask > 0].std()
            else:
                recon_error_std = torch.tensor(0.0, device=flux.device)
        
        # Flux normalization statistics (batch-level)
        flux_mean_avg = mean.mean()
        flux_std_avg = std.mean()
        
        # Log all metrics  (lr is logged by WarmupCosineLR callback)
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        self.log("recon_loss", recon_loss, on_step=True)
        self.log("cls_loss", cls_loss, on_step=True)
        self.log("mask_count", mask_count, on_step=True)
        self.log("valid_count", valid_count, on_step=True)
        self.log("mask_ratio_actual", mask_ratio_actual, on_step=True)
        self.log("recon_error_std", recon_error_std, on_step=True)
        self.log("flux_mean_avg", flux_mean_avg, on_step=True)
        self.log("flux_std_avg", flux_std_avg, on_step=True)
        
        return loss

    def validation_step(self, batch: dict) -> Tensor:
        """Validation step (no masking - evaluate full reconstruction)."""
        flux = batch["flux"]
        wavelength = batch["wavelength"]
        valid_mask = batch["valid_mask"]

        # Redshift correction: use rest-frame wavelengths for positional encoding
        if self.hparams.redshift_corr:
            wavelength = self._apply_redshift_corr(wavelength, batch)

        # Normalize flux per-sample
        flux_norm, mean, std = self._normalize_flux(flux, valid_mask)
        target = flux_norm.clone()  # Target is normalized flux

        # NO masking in validation - evaluate on full unmasked spectra
        input_flux = flux_norm

        stats = torch.cat([mean, std], dim=-1)  # (B, 2)

        out = self.forward(input_flux, wavelength, valid_mask, stats=stats)
        reconstructions = out["reconstructions"]  # (B, T)
        cls_recons      = out["cls_recons"]        # (B, T)

        valid_float = valid_mask.float()
        count = valid_float.sum().clamp(min=1)

        # Token reconstruction loss (all valid positions, no masking in validation)
        if valid_float.sum() > 0:
            recon_loss = F.mse_loss(
                reconstructions * valid_float,
                target * valid_float,
                reduction="sum",
            ) / count
        else:
            recon_loss = torch.tensor(0.0, device=flux.device)

        # CLS reconstruction loss
        cls_loss = torch.tensor(0.0, device=flux.device)
        if self.hparams.cls_aux_weight > 0 and valid_float.sum() > 0:
            cls_loss = F.mse_loss(
                cls_recons * valid_float,
                target * valid_float,
                reduction="sum",
            ) / count

        loss = recon_loss + self.hparams.cls_aux_weight * cls_loss

        self.log("val_loss",       loss,       prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_recon_loss", recon_loss, on_epoch=True, sync_dist=True)
        self.log("val_cls_loss",   cls_loss,   on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch: dict) -> Tensor:
        """Test step with masking; not the same as validation."""
        flux = batch["flux"]
        wavelength = batch["wavelength"]
        valid_mask = batch["valid_mask"]

        # Redshift correction: use rest-frame wavelengths for positional encoding
        if self.hparams.redshift_corr:
            wavelength = self._apply_redshift_corr(wavelength, batch)

        # Normalize flux per-sample
        flux_norm, mean, std = self._normalize_flux(flux, valid_mask)
        target = flux_norm.clone()  # Target is normalized flux

        input_flux, train_mask = self._mask_input(flux_norm, valid_mask)

        stats = torch.cat([mean, std], dim=-1)  # (B, 2)
        out = self(input_flux, wavelength, valid_mask, stats=stats)
        reconstructions = out["reconstructions"]  # (B, T)

        loss_mask = (train_mask & valid_mask).float()

        # Compute loss only on masked valid positions
        if loss_mask.sum() > 0:
            loss = F.mse_loss(
                reconstructions * loss_mask,
                target * loss_mask,
                reduction="sum",
            ) / loss_mask.sum().clamp(min=1)
        else:
            loss = torch.tensor(0.0, device=flux.device)

        self.log("test_loss", loss, prog_bar=True)
        return loss

    def _apply_redshift_corr(self, wavelength: Tensor, batch: dict) -> Tensor:
        """Convert observed wavelengths to rest-frame: λ_rest = λ_obs / (1 + z).

        Args:
            wavelength: (B, T) observed wavelengths
            batch: batch dict containing 'redshift' key with shape (B,)

        Returns:
            wavelength_rest: (B, T) rest-frame wavelengths
        """
        z = batch["redshift"].unsqueeze(-1)  # (B, 1)
        return wavelength / (1.0 + z)

    def _normalize_flux(self, flux: Tensor, valid_mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Normalize flux per-sample using valid positions only (specformer style).
        
        Args:
            flux: shape (B, T), flux values
            valid_mask: shape (B, T), bool (True = valid)
        
        Returns:
            normalized_flux: shape (B, T), normalized flux
            mean: shape (B, 1), per-sample mean
            std: shape (B, 1), per-sample std (clipped to min_std)
        """
        B, T = flux.shape
        min_std = self.hparams.min_std
        
        # Compute mean and std using only valid positions
        valid_flux = flux.clone()
        valid_flux[~valid_mask] = 0.0  # Zero out invalid positions for sum
        
        # Count valid positions per sample
        valid_count = valid_mask.sum(dim=1, keepdim=True).float().clamp(min=1)  # (B, 1)
        
        # Mean over valid positions
        mean = valid_flux.sum(dim=1, keepdim=True) / valid_count  # (B, 1)
        
        # Std over valid positions
        sq_diff = ((flux - mean) ** 2) * valid_mask.float()
        var = sq_diff.sum(dim=1, keepdim=True) / valid_count
        std = var.sqrt().clamp(min=min_std)  # (B, 1)
        
        # Normalize: (flux - mean) / std
        normalized_flux = (flux - mean) / std
        
        # Set invalid positions to 0 in normalized flux
        normalized_flux = normalized_flux * valid_mask.float()
        
        return normalized_flux, mean, std

    def _mask_input(self, flux: Tensor, valid_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Line-aware masked autoencoding — see model.masking for the full algorithm."""
        return mask_input(
            flux, valid_mask,
            mask_ratio      = self.hparams.mask_ratio,
            min_unmasked    = self.hparams.min_unmasked,
            max_line_blocks = self.hparams.max_line_blocks,
            line_block_size = self.hparams.line_block_size,
            line_prominence = self.hparams.line_prominence,
        )
