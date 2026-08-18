"""Multimodal fusion model: N modalities → shared latent space via per-modality projectors.

Design goals:
- Flexible: each modality can have any input dimension.
- Extensible: adding a new modality requires only register_modality().
- Missing-modality aware: pairwise contrastive loss is computed only on samples
  where both modalities in a pair are available.
"""

from itertools import combinations
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .losses import info_nce_loss


# All pooling heads share the signature forward(x, key_padding_mask) where
# x is (B, T, dim) and key_padding_mask is (B, T) bool with True = IGNORE the
# token (PyTorch MultiheadAttention convention). None = every token is valid.
# `POOL_OUT_MULT` records how each strategy scales the feature dim, so the
# downstream projector can be sized correctly.
POOL_OUT_MULT = {"mean": 1, "max": 1, "meanmax": 2, "attention": 1}


class MaskedMeanPool(nn.Module):
    """Mean over valid tokens. Output (B, dim). No parameters."""

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        if key_padding_mask is None:
            return x.mean(dim=1)
        valid = (~key_padding_mask).unsqueeze(-1).to(x.dtype)   # (B, T, 1)
        summed = (x * valid).sum(dim=1)
        denom = valid.sum(dim=1).clamp(min=1.0)                 # all-invalid → /1 → 0 vec
        return summed / denom


class MaskedMaxPool(nn.Module):
    """Max over valid tokens. Output (B, dim). No parameters."""

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        if key_padding_mask is None:
            return x.max(dim=1).values
        valid = (~key_padding_mask).unsqueeze(-1)               # (B, T, 1) bool
        out = x.masked_fill(~valid, float("-inf")).max(dim=1).values
        return torch.nan_to_num(out, neginf=0.0)                # all-invalid rows → 0


class MaskedMeanMaxPool(nn.Module):
    """Concatenation of masked mean and masked max. Output (B, 2*dim)."""

    def __init__(self):
        super().__init__()
        self.mean = MaskedMeanPool()
        self.max = MaskedMaxPool()

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        return torch.cat(
            [self.mean(x, key_padding_mask), self.max(x, key_padding_mask)], dim=-1
        )


class AttentionPool(nn.Module):
    """Learnable-query cross-attention pooling (a.k.a. attentive pooling / MAP head).

    Pools a variable-length token sequence into one fixed vector, the way AstroCLIP
    aggregates encoder tokens: instead of taking [CLS] or mean-pooling, a single
    learnable query attends over all tokens and returns their attention-weighted sum::

        z* = Σ_i α_i · (W_v x_i),   α = softmax(q·(W_k x_i) / √d)

    Query is a learned parameter independent of the input (it does not come from the
    tokens — that is what makes this *cross*-attention, not self-attention).

    Args:
        dim:       Token embedding dimension (also the output dimension).
        num_heads: Multi-head attention heads.
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x:                (B, T, dim) token sequence.
            key_padding_mask: (B, T) bool, True = IGNORE this token (PyTorch
                              convention). Pass ``~valid_mask``. None = attend all.
        Returns:
            (B, dim) pooled vector.
        """
        B = x.shape[0]
        if key_padding_mask is not None:
            # A row with every token masked makes softmax see all -inf → NaN.
            # Unmask such rows (their pooled output is meaningless but those
            # samples are unavailable downstream and excluded from the loss).
            all_masked = key_padding_mask.all(dim=1)
            if all_masked.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked] = False
        q = self.query.expand(B, -1, -1)                       # (B, 1, dim)
        pooled, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask)
        return pooled.squeeze(1)                                # (B, dim)


class StatsEncoder(nn.Module):
    """Encode per-sample absolute-scale stats into a pooled-space vector to ADD
    after pooling.

    Both encoders normalise away the absolute flux scale before tokenising, so
    their patch tokens carry only the *shape* of the signal. Each modality's raw
    per-sample (mean, std) — measured on the raw flux, before any per-sample
    normalisation — is re-injected here so the contrastive head can align on
    brightness too.

    Each stat column is divided by a fixed per-column reference ``stats_scale``
    and passed through asinh, which is linear near zero and logarithmic beyond
    it: with the reference set near the column median the transform compresses
    the multi-decade flux range about its centre. asinh also tolerates zero and
    negative values (sky-dominated cutouts, noisy spectra), unlike log. The
    transform is elementwise and sample-independent, so a bad row cannot affect
    any other row in the batch.
    """

    def __init__(
        self,
        stats_dim: int,
        out_dim: int,
        stats_scale: Optional[Union[float, list]] = None,
    ):
        super().__init__()
        if stats_scale is None:
            scale = torch.ones(stats_dim)
        elif isinstance(stats_scale, (int, float)):
            scale = torch.full((stats_dim,), float(stats_scale))
        else:
            scale = torch.tensor([float(v) for v in stats_scale])
            if scale.numel() != stats_dim:
                raise ValueError(
                    f"stats_scale has {scale.numel()} entries but stats_dim={stats_dim}."
                )
        if not bool((scale > 0).all()):
            raise ValueError(f"stats_scale entries must be > 0, got {scale.tolist()}.")
        self.register_buffer("stats_scale", scale)
        self.proj = nn.Linear(stats_dim, out_dim)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, stats: Tensor) -> Tensor:
        """stats: (B, stats_dim) → (B, out_dim)."""
        # A modality can be available for a sample while an individual stat is
        # NaN, so those entries are not caught by the caller's ~avail zeroing.
        # Impute to 0, which asinh maps to 0 (no injection for that column).
        s = torch.nan_to_num(stats)
        s = torch.asinh(s / self.stats_scale)
        return self.proj(s)


class ModalityProjector(nn.Module):
    """Projects a modality CLS embedding to the shared latent space.

    Architecture (hidden_dim as list [h1, h2, ...]):
        Linear(input_dim → h1) → GELU → Linear(h1 → h2) → GELU → … → Linear(hN → latent_dim) → LayerNorm
        hidden_dim=None → Linear(input_dim → latent_dim) → LayerNorm  (no hidden layers)
        hidden_dim=int  → treated as [int]  (single hidden layer, backward compatible)

    The output is L2-normalized so that cosine similarity = dot product.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: Optional[Union[int, list]] = None,
    ):
        super().__init__()
        if hidden_dim is None:
            dims = [input_dim, latent_dim]
        elif isinstance(hidden_dim, int):
            dims = [input_dim, hidden_dim, latent_dim]
        else:
            dims = [input_dim] + list(hidden_dim) + [latent_dim]

        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        layers.append(nn.LayerNorm(latent_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, input_dim)
        Returns:
            (B, latent_dim) L2-normalized
        """
        return F.normalize(self.net(x), dim=-1)


class MultimodalFusion(nn.Module):
    """Aligns N modalities in a shared latent space via per-modality projectors.

    Usage example::

        fusion = MultimodalFusion(latent_dim=256, temperature=0.07)
        fusion.register_modality("image",    input_dim=768, hidden_dim=512)
        fusion.register_modality("spectrum", input_dim=128, hidden_dim=256)

        # During training:
        embeddings = fusion(
            inputs={"image": img_cls, "spectrum": spec_cls},
            availability={"image": img_avail, "spectrum": spec_avail},
        )
        loss, loss_dict = fusion.compute_pairwise_loss(embeddings, availability)

    To add a 3rd modality later (no other code changes needed)::

        fusion.register_modality("photometry", input_dim=64, hidden_dim=128)

    Args:
        latent_dim:   Shared embedding dimension D.
        temperature:  InfoNCE softmax temperature (default 0.07).
    """

    def __init__(self, latent_dim: int = 256, temperature: float = 0.07):
        super().__init__()
        self.latent_dim = latent_dim
        self.temperature = temperature
        # ModuleDict so projectors are registered as parameters
        self.projectors = nn.ModuleDict()
        # Optional per-modality token pooling head (token-sequence modalities only).
        self.pools = nn.ModuleDict()
        # Optional per-modality side-feature encoders for per-sample absolute-scale
        # stats (e.g. spectrum [mean, std]). Injected (added) after pooling.
        self.stats_encoders = nn.ModuleDict()
        # Modalities whose pooled result is the CLS vector directly (pool="cls").
        # Tracked separately because "cls" has no pooling module.
        self._cls_modalities: set[str] = set()

    def register_modality(
        self,
        name: str,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        pool: Optional[str] = None,
        num_heads: int = 4,
        stats_dim: Optional[int] = None,
        stats_scale: Optional[Union[float, list]] = None,
    ) -> None:
        """Add (or overwrite) the projector for a modality.

        Args:
            name:       Modality identifier string (e.g. "image", "spectrum").
            input_dim:  Feature dimension of the encoder output for this modality
                        (the per-token dim for token-sequence modalities).
            hidden_dim: Hidden layer width.  None → single Linear layer.
            stats_dim:  If set, register a StatsEncoder mapping a per-sample
                        side-feature of this width (e.g. 2 for [mean, std]) into
                        the pooled space; forward() ADDs it after pooling. None
                        → no stats injection for this modality.
            stats_scale: Per-column reference value each stat is divided by
                        before the asinh stretch — a scalar broadcast to all
                        columns, or a list of length stats_dim. Set it near each
                        column's median so the stretch compresses about the bulk
                        of the distribution. None → 1.0 (no rescaling).
            pool:       Token-pooling strategy applied to a (B, T, input_dim) token
                        sequence before the projector. All strategies respect the
                        per-modality valid-token mask passed to forward():
                          - "mean":      masked mean         → (B, input_dim)
                          - "max":       masked max          → (B, input_dim)
                          - "meanmax":   concat(mean, max)   → (B, 2*input_dim)
                          - "attention": learnable-query cross-attention (B, input_dim)
                          - None:        modality is already a (B, input_dim) vector;
                                         no pooling.
            num_heads:  Heads for the AttentionPool (ignored unless pool="attention").
        """
        pooled_dim = input_dim
        if pool in (None, "none"):
            pooler = None
        elif pool == "cls":
            # The input IS the CLS representation already — point modality_keys at the
            # CLS dataset. No pooling module; forward uses it directly (token 0 if the
            # input still carries a token axis).
            pooler = None
            self._cls_modalities.add(name)
        elif pool == "attention":
            pooler = AttentionPool(input_dim, num_heads=num_heads)
        elif pool == "mean":
            pooler = MaskedMeanPool()
        elif pool == "max":
            pooler = MaskedMaxPool()
        elif pool == "meanmax":
            pooler = MaskedMeanMaxPool()
        else:
            raise ValueError(
                f"Unknown pool='{pool}' for modality '{name}' "
                f"(use one of: mean, max, meanmax, attention, cls, None)."
            )
        if pooler is not None:
            self.pools[name] = pooler
            pooled_dim = input_dim * POOL_OUT_MULT[pool]
        self.projectors[name] = ModalityProjector(pooled_dim, self.latent_dim, hidden_dim)
        if stats_dim is not None:
            self.stats_encoders[name] = StatsEncoder(stats_dim, pooled_dim, stats_scale)

    def forward(
        self,
        inputs: dict[str, Tensor],
        availability: dict[str, Tensor],
        masks: Optional[dict[str, Tensor]] = None,
        stats: Optional[dict[str, Tensor]] = None,
    ) -> dict[str, Tensor]:
        """Project all available modality embeddings to the shared latent space.

        Args:
            inputs:       {name: (B, d_in)} pre-pooled vectors, or
                          {name: (B, T, d_in)} token sequences for modalities
                          registered with pool="attention".
                          Only modalities present in the batch need to be included.
            availability: {name: (B,) bool} — True where the modality exists for a sample.
            masks:        {name: (B, T) bool} valid-token masks for token-sequence
                          modalities (True = valid token). Optional; None for a
                          modality means all tokens are valid.
            stats:        {name: (B, S) float} per-sample side-features (e.g.
                          [mean, std]); ADDed in pooled space via the modality's
                          StatsEncoder. Only used for modalities registered with
                          stats_dim. Optional.

        Returns:
            {name: (B, D)} — L2-normalized embeddings in the shared latent space.

        Raises:
            ValueError: if a modality name in `inputs` was not registered.
        """
        masks = masks or {}
        stats = stats or {}
        embeddings: dict[str, Tensor] = {}
        for name, x in inputs.items():
            if name not in self.projectors:
                raise ValueError(
                    f"Modality '{name}' not registered. "
                    f"Call register_modality('{name}', input_dim=...) first."
                )
            avail = availability[name]
            x = x.clone()
            if name in self._cls_modalities:
                # pool="cls": input is the CLS itself; take token 0 if it still
                # carries a token axis, otherwise use the vector as-is.
                if x.dim() == 3:
                    x = x[:, 0, :]
            elif name in self.pools:
                # Token-sequence modality → learnable-query attention pooling.
                # Neutralize missing/NaN rows before attention so they cannot
                # produce NaNs that poison the pool's gradients.
                x[~avail] = 0.0
                m = masks.get(name)
                kpm = ~m if m is not None else None              # PyTorch: True = ignore
                x = self.pools[name](x, key_padding_mask=kpm)    # (B, d_in)
            # Inject per-sample side-feature (e.g. [mean, std] of the raw flux) in
            # pooled space. Missing/masked rows carry arbitrary values, so zero
            # them; those rows are zeroed wholesale below anyway.
            if name in self.stats_encoders and name in stats:
                s = stats[name].clone()
                s[~avail] = 0.0
                x = x + self.stats_encoders[name](s)
            # Zero out missing-modality rows before projecting to prevent NaN
            # gradients: F.normalize backward on NaN inputs (even with zero
            # incoming grad) produces NaN, which poisons projector weight grads.
            x[~avail] = 0.0
            embeddings[name] = self.projectors[name](x)
        return embeddings

    def compute_pairwise_loss(
        self,
        embeddings: dict[str, Tensor],
        availability: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute symmetric InfoNCE losses for all valid modality pairs.

        For each pair (A, B):
          1. Identify samples where both modalities are available.
          2. Skip the pair if fewer than 2 such samples exist (can't form negatives).
          3. Compute symmetric InfoNCE on the filtered subset.

        The returned total loss is the mean over all computed pairs, so adding more
        modalities does not automatically increase the loss scale.

        Args:
            embeddings:   {name: (B, D)} L2-normalized (output of forward()).
            availability: {name: (B,) bool} which samples have each modality.

        Returns:
            total_loss: Scalar Tensor — mean InfoNCE over valid pairs.
                        Returns 0.0 (with grad) if no valid pair exists.
            loss_dict:  {f"{A}_{B}": scalar Tensor} — per-pair losses for logging.
        """
        device = next(iter(embeddings.values())).device
        total_loss = torch.zeros(1, device=device).squeeze()
        loss_dict: dict[str, Tensor] = {}
        num_valid_pairs = 0

        for name_a, name_b in combinations(embeddings.keys(), 2):
            avail_a = availability[name_a]   # (B,) bool
            avail_b = availability[name_b]   # (B,) bool
            valid = avail_a & avail_b        # samples where both are present

            if valid.sum() < 2:
                continue

            embed_a = embeddings[name_a][valid]   # (N, D)
            embed_b = embeddings[name_b][valid]   # (N, D)

            pair_loss = info_nce_loss(embed_a, embed_b, self.temperature)
            key = f"{name_a}_{name_b}"
            loss_dict[key] = pair_loss
            total_loss = total_loss + pair_loss
            num_valid_pairs += 1

        if num_valid_pairs > 0:
            total_loss = total_loss / num_valid_pairs

        return total_loss, loss_dict
