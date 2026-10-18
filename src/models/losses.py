"""
Loss functions for fingerprint metric learning.

Two strategies are provided, both through a unified interface:

  loss_type    class                   how it works
  ──────────   ─────────────────────   ───────────────────────────────────────
  'arcface'    ArcFaceLoss             Angular-margin softmax. Trains like a
                                       classifier but forces the embedding space
                                       to be discriminative via an additive
                                       angular margin. The loss module holds a
                                       learnable weight matrix — include it in
                                       the optimizer (see get_optimizer_params).

  'triplet'    TripletLoss             Triplet margin loss + semi-hard mining.
                                       Classic metric-learning approach. Mining
                                       is mandatory — random triplets carry
                                       near-zero gradient and teach nothing.
                                       No learnable parameters in the loss itself.

Recommendation
──────────────
Start with ArcFace: simpler training loop (standard cross-entropy machinery,
no pair/triplet sampling), SOTA in face/fingerprint literature, trains stably
with a normal DataLoader. Add Triplet as an ablation once ArcFace converges —
"we compared angular-margin vs. direct metric learning" is a strong report section.

Typical usage
─────────────
    from src.models.losses import get_loss

    # ArcFace
    criterion = get_loss('arcface', num_classes=4200, embedding_dim=128)
    optimizer = torch.optim.AdamW([
        *model.get_param_groups(base_lr=1e-3),
        *criterion.get_optimizer_params(lr=1e-3),   # ← include loss params!
    ])
    loss = criterion(embeddings, labels)

    # Triplet
    criterion = get_loss('triplet')
    loss = criterion(embeddings, labels)
"""

import torch
import torch.nn as nn
from pytorch_metric_learning import losses as pml_losses
from pytorch_metric_learning import miners


# ── ArcFace ───────────────────────────────────────────────────────────────────

class ArcFaceLoss(nn.Module):
    """
    ArcFace (Additive Angular Margin) loss.

    Adds an angular margin between embedding vectors and their class centres in
    the hyperspherical embedding space.  After training, discard this loss
    module and keep only the embedding network — the class weight matrix is not
    needed at inference time.

    Parameters
    ----------
    num_classes : int
        Number of unique finger identities in the *training* split (4200 for
        the default SOCOFing split).  Must match the remapped labels from
        FingerprintDataset.num_classes.
    embedding_dim : int
        Must match FingerprintEmbeddingNet.embedding_dim (default 128).
    margin : float
        Angular margin in degrees (default 28.6° ≈ 0.5 rad — from the
        original ArcFace paper; standard for face/fingerprint tasks).
    scale : float
        Softmax temperature scaling factor (default 64 — standard value that
        works across a wide range of embedding dimensions and class counts).

    Important
    ---------
    This module contains a learnable weight matrix of shape
    (num_classes, embedding_dim).  It MUST be included in the optimizer —
    call get_optimizer_params() to get its parameter dict.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 128,
        margin: float = 28.6,
        scale: float = 64.0,
    ) -> None:
        super().__init__()
        self.loss_fn = pml_losses.ArcFaceLoss(
            num_classes=num_classes,
            embedding_size=embedding_dim,
            margin=margin,
            scale=scale,
        )
        self.num_classes  = num_classes
        self.embedding_dim = embedding_dim

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        embeddings : (B, embedding_dim) L2-normalised unit vectors.
        labels     : (B,) long tensor of class labels (0-indexed, contiguous).

        Returns
        -------
        loss : scalar tensor.
        """
        return self.loss_fn(embeddings, labels)

    def get_optimizer_params(self, lr: float) -> list:
        """
        Return a parameter group dict for the loss's weight matrix.

        Pass this to the optimizer alongside the model's param groups so the
        class-centre weight matrix is updated during training:

            optimizer = torch.optim.AdamW([
                *model.get_param_groups(base_lr=lr),
                *criterion.get_optimizer_params(lr=lr),
            ])
        """
        return [{"params": self.loss_fn.parameters(), "lr": lr,
                 "name": "arcface_weight_matrix"}]


# ── Triplet ───────────────────────────────────────────────────────────────────

class TripletLoss(nn.Module):
    """
    Triplet margin loss with semi-hard mining.

    Selects informative (anchor, positive, negative) triplets from each batch
    rather than sampling them randomly — random triplets mostly satisfy the
    margin constraint already and produce near-zero gradients.

    'semihard' mining: negatives that are farther than the positive but still
    within the margin.  This is the regime where the loss is non-zero but
    the triplet is not yet trivially violated — the network gets a useful
    gradient signal without chasing extremely hard negatives (which can
    destabilise early training).

    Parameters
    ----------
    margin : float
        Minimum desired distance between positive and negative pairs (default
        0.2 — a common starting point; tune if loss collapses or doesn't converge).
    type_of_triplets : str
        Passed to TripletMarginMiner. Options: 'semihard' (default),
        'hard', 'all'.  'semihard' is recommended for most cases.

    Note
    ----
    This loss has NO learnable parameters — no extra optimizer setup needed.
    However, it requires sufficiently large batches to find valid triplets.
    Increase batch size or use 'hard' mining if the miner consistently finds
    0 triplets.
    """

    def __init__(
        self,
        margin: float = 0.2,
        type_of_triplets: str = "semihard",
    ) -> None:
        super().__init__()
        self.miner   = miners.TripletMarginMiner(
            margin=margin,
            type_of_triplets=type_of_triplets,
        )
        self.loss_fn = pml_losses.TripletMarginLoss(margin=margin)
        self.margin           = margin
        self.type_of_triplets = type_of_triplets

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        embeddings : (B, embedding_dim) L2-normalised unit vectors.
        labels     : (B,) long tensor of class labels.

        Returns
        -------
        loss : scalar tensor.
              Returns zero if the miner finds no valid triplets in the batch
              (can happen with very small batches or near-converged models).
        """
        triplets = self.miner(embeddings, labels)
        if triplets[0].numel() == 0:
            # No valid triplets found — return a zero loss that still has a
            # grad_fn so the training loop doesn't break.
            return embeddings.sum() * 0.0
        return self.loss_fn(embeddings, labels, triplets)

    def num_triplets_in_last_batch(self) -> int:
        """Convenience: how many triplets the miner selected last forward pass."""
        return self.miner.num_triplets if hasattr(self.miner, "num_triplets") else -1


# ── Factory ───────────────────────────────────────────────────────────────────

def get_loss(
    loss_type: str,
    num_classes: int = 0,
    embedding_dim: int = 128,
    **kwargs,
) -> nn.Module:
    """
    Build and return a loss module.

    Parameters
    ----------
    loss_type : {'arcface', 'triplet'}
    num_classes : int
        Required for 'arcface' — number of training classes (e.g. 4200).
        Ignored for 'triplet'.
    embedding_dim : int
        Required for 'arcface' — must match model embedding_dim.
        Ignored for 'triplet'.
    **kwargs
        Forwarded to the loss constructor (e.g. margin, scale).

    Returns
    -------
    nn.Module with a forward(embeddings, labels) signature.

    Examples
    --------
        arcface  = get_loss('arcface', num_classes=4200, embedding_dim=128)
        triplet  = get_loss('triplet', margin=0.2)
        triplet2 = get_loss('triplet', margin=0.3, type_of_triplets='hard')
    """
    loss_type = loss_type.lower().strip()

    if loss_type == "arcface":
        if num_classes <= 0:
            raise ValueError("num_classes must be > 0 for ArcFace loss.")
        return ArcFaceLoss(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    if loss_type == "triplet":
        return TripletLoss(**kwargs)

    raise ValueError(
        f"Unknown loss_type '{loss_type}'. Choose from: 'arcface', 'triplet'."
    )
