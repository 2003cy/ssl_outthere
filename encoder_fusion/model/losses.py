"""Contrastive losses for multimodal fusion."""

import torch
import torch.nn.functional as F
from torch import Tensor


def info_nce_loss(embed_a: Tensor, embed_b: Tensor, temperature: float = 0.07) -> Tensor:
    """Symmetric InfoNCE (NT-Xent) loss between two sets of L2-normalized embeddings.

    For each pair (embed_a[i], embed_b[i]) the positive pair, all other
    (embed_a[i], embed_b[j]) with j != i are negatives.

    Args:
        embed_a: (N, D) L2-normalized embeddings from modality A.
        embed_b: (N, D) L2-normalized embeddings from modality B.
        temperature: Softmax temperature. Lower values → sharper distribution.

    Returns:
        Scalar loss (mean of A→B and B→A cross-entropy losses).
    """
    N = embed_a.shape[0]
    # (N, N) similarity matrix
    logits = embed_a @ embed_b.T / temperature
    labels = torch.arange(N, device=embed_a.device)
    loss_ab = F.cross_entropy(logits, labels)
    loss_ba = F.cross_entropy(logits.T, labels)
    return (loss_ab + loss_ba) / 2
