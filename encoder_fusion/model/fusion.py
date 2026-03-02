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

    def register_modality(
        self,
        name: str,
        input_dim: int,
        hidden_dim: Optional[int] = None,
    ) -> None:
        """Add (or overwrite) the projector for a modality.

        Args:
            name:       Modality identifier string (e.g. "image", "spectrum").
            input_dim:  Dimension of the encoder's CLS output for this modality.
            hidden_dim: Hidden layer width.  None → single Linear layer.
        """
        self.projectors[name] = ModalityProjector(input_dim, self.latent_dim, hidden_dim)

    def forward(
        self,
        inputs: dict[str, Tensor],
        availability: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Project all available modality embeddings to the shared latent space.

        Args:
            inputs:       {name: (B, d_in)} — raw CLS embeddings from each encoder.
                          Only modalities present in the batch need to be included.
            availability: {name: (B,) bool} — True where the modality exists for a sample.

        Returns:
            {name: (B, D)} — L2-normalized embeddings in the shared latent space.

        Raises:
            ValueError: if a modality name in `inputs` was not registered.
        """
        embeddings: dict[str, Tensor] = {}
        for name, x in inputs.items():
            if name not in self.projectors:
                raise ValueError(
                    f"Modality '{name}' not registered. "
                    f"Call register_modality('{name}', input_dim=...) first."
                )
            # Zero out missing-modality rows before projecting to prevent NaN
            # gradients: F.normalize backward on NaN inputs (even with zero
            # incoming grad) produces NaN, which poisons projector weight grads.
            x = x.clone()
            x[~availability[name]] = 0.0
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
