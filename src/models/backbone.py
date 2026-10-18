"""
Modular embedding backbone for the fingerprint identification project.

Three architectures share a single public interface (FingerprintEmbeddingNet):

  backbone_name     feature_dim    notes
  ──────────────    ───────────    ──────────────────────────────────────────
  'resnet18'              512      Lightest/fastest; recommended baseline
  'densenet121'          1024      Dense skip-connections; richer texture repr
  'vgg16'                4096      Uses pretrained FC layers; heaviest model

All backbones:
  - Accept  (B, 3, 128, 128) ImageNet-normalised input (from dataset.py)
  - Output  (B, embedding_dim) L2-normalised unit vectors

Typical usage
─────────────
    from src.models.backbone import FingerprintEmbeddingNet

    model = FingerprintEmbeddingNet(backbone_name='resnet18', embedding_dim=128)
    model.freeze_backbone()          # train head only for first few epochs
    # ... later:
    model.unfreeze_backbone()
    optimizer = torch.optim.AdamW(
        model.get_param_groups(base_lr=1e-3, backbone_lr_scale=0.1)
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision.models import (
    ResNet18_Weights,
    DenseNet121_Weights,
    VGG16_Weights,
)


# ── Per-backbone feature extractor modules ────────────────────────────────────

class _ResNet18Features(nn.Module):
    """
    ResNet18 trunk up to and including the adaptive average-pool layer.
    Output: (B, 512)
    """
    feature_dim: int = 512

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = tv_models.resnet18(weights=weights)
        # children() order:
        #   conv1 bn1 relu maxpool layer1 layer2 layer3 layer4 avgpool fc
        # Drop the 1000-class fc; keep avgpool so spatial dims collapse to (1,1).
        self.body = nn.Sequential(*list(base.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x).flatten(1)          # (B, 512, 1, 1) → (B, 512)


class _DenseNet121Features(nn.Module):
    """
    DenseNet121 dense-block trunk with global average pooling.
    Output: (B, 1024)

    DenseNet has no separate avgpool child — the features block ends with a
    BatchNorm, so we apply relu + adaptive_avg_pool manually.
    """
    feature_dim: int = 1024

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        base = tv_models.densenet121(weights=weights)
        self.features = base.features           # Sequential of dense blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)                 # (B, 1024, H', W')
        feat = F.relu(feat, inplace=True)
        feat = F.adaptive_avg_pool2d(feat, (1, 1))
        return feat.flatten(1)                  # (B, 1024)


class _VGG16Features(nn.Module):
    """
    VGG16 convolutional trunk + pretrained FC layers (classifier[:-1]).

    VGG's conv blocks are shallower than ResNet/DenseNet, so the FC layers
    carry substantial discriminative capacity — we keep their pretrained
    weights rather than discarding them.

    Pipeline:
        features      (B, 512, H', W')
        avgpool       (B, 512, 7,  7)   AdaptiveAvgPool2d(7, 7)
        flatten       (B, 25088)
        classifier[0] Linear(25088, 4096) + ReLU + Dropout
        classifier[3] Linear(4096,  4096) + ReLU + Dropout
        output        (B, 4096)

    classifier[-1] (Linear 4096→1000) is dropped.
    """
    feature_dim: int = 4096

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        base = tv_models.vgg16(weights=weights)
        self.features = base.features           # conv blocks
        self.avgpool  = base.avgpool            # AdaptiveAvgPool2d(7, 7)
        # classifier[-1] is the 4096→1000 head; drop it, keep indices 0-5.
        self.fc_body  = base.classifier[:-1]    # output: (B, 4096)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)                # 512 × 7 × 7 = 25 088
        return self.fc_body(x)                 # (B, 4096)


# Internal registry — maps backbone name → extractor class
_REGISTRY: dict = {
    "resnet18":    _ResNet18Features,
    "densenet121": _DenseNet121Features,
    "vgg16":       _VGG16Features,
}


# ── Public model class ────────────────────────────────────────────────────────

class FingerprintEmbeddingNet(nn.Module):
    """
    Fixed-length fingerprint embedding network.

    Composes a backbone feature extractor with a linear projection head and
    L2-normalisation to produce unit-norm embedding vectors.

    Parameters
    ----------
    backbone_name : {'resnet18', 'densenet121', 'vgg16'}
        Which pretrained backbone to use.
    embedding_dim : int
        Output embedding dimensionality (default 128).
    pretrained : bool
        Initialise backbone with ImageNet weights (default True).
    """

    SUPPORTED_BACKBONES = tuple(_REGISTRY.keys())

    def __init__(
        self,
        backbone_name: str = "resnet18",
        embedding_dim: int = 128,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if backbone_name not in _REGISTRY:
            raise ValueError(
                f"backbone_name must be one of {self.SUPPORTED_BACKBONES}, "
                f"got '{backbone_name}'"
            )

        self.backbone_name  = backbone_name
        self.embedding_dim  = embedding_dim

        self.backbone       = _REGISTRY[backbone_name](pretrained=pretrained)
        feature_dim         = self.backbone.feature_dim

        # Projection: backbone features → fixed-length embedding
        self.embedding_head = nn.Linear(feature_dim, embedding_dim)
        nn.init.xavier_uniform_(self.embedding_head.weight)
        nn.init.zeros_(self.embedding_head.bias)

    # ── Forward ───────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (B, 3, H, W), ImageNet-normalised.

        Returns
        -------
        emb : Tensor of shape (B, embedding_dim) — L2-normalised unit vectors.
        """
        feat = self.backbone(x)                         # (B, feature_dim)
        emb  = self.embedding_head(feat)                # (B, embedding_dim)
        return F.normalize(emb, p=2, dim=1)             # unit vectors

    # ── Freeze / unfreeze — for progressive fine-tuning ───────────────

    def freeze_backbone(self) -> None:
        """
        Freeze all backbone parameters so only the embedding head is trained.

        Use at the start of training:
          - No gradients stored for frozen layers → lower GPU memory usage.
          - Lets the randomly-initialised head stabilise before the pretrained
            backbone weights are disturbed.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze all backbone parameters for end-to-end fine-tuning.

        Call after the head has converged (a few epochs), then use
        get_param_groups() so the backbone gets a lower LR than the head.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_param_groups(
        self, base_lr: float, backbone_lr_scale: float = 0.1
    ) -> list:
        """
        Return two optimizer parameter groups with different learning rates:
          - backbone params : base_lr × backbone_lr_scale  (fine-tune gently)
          - embedding head  : base_lr                       (full LR)

        Example
        -------
            optimizer = torch.optim.AdamW(
                model.get_param_groups(base_lr=1e-3, backbone_lr_scale=0.1)
            )
        """
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params     = list(self.embedding_head.parameters())
        return [
            {"params": backbone_params, "lr": base_lr * backbone_lr_scale,
             "name": f"backbone_{self.backbone_name}"},
            {"params": head_params,     "lr": base_lr,
             "name": "embedding_head"},
        ]

    # ── Diagnostics ───────────────────────────────────────────────────

    def param_summary(self) -> dict:
        """Return a dict of total / trainable / frozen parameter counts."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def __repr__(self) -> str:
        ps = self.param_summary()
        return (
            f"FingerprintEmbeddingNet("
            f"backbone={self.backbone_name}, "
            f"embedding_dim={self.embedding_dim}, "
            f"params={ps['total']:,} total / {ps['trainable']:,} trainable)"
        )
