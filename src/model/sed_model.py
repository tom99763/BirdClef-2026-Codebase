"""Sound Event Detection (SED) model for BirdClef 2026.

Architecture inspired by BirdCLEF 2025 top solutions:
  - 1st place  : SED models for frame-level temporal annotation
  - 5th place  : 13 SED models (EfficientNetV2-S, EfficientNet-B3-NS, B0-NS, B3)
  - 2nd place  : EfficientNetV2-S (in21k) + ECA-NFNet-L0

Pipeline:
  Mel spectrogram (1, n_mels, T)
      → CNN backbone (EfficientNetV2-S / EfficientNet-B3-NS via timm)
      → Frequency pooling → time sequence (B, T', C)
      → Attention pooling → clip-level prediction (B, num_classes)

The attention pooling block simultaneously produces:
  - clip_pred   : (B, num_classes) — used for training loss
  - frame_pred  : (B, T', num_classes) — per-frame probability (for analysis / SED output)

Requirements:
    pip install torch timm torchaudio
"""

import numpy as np
import torch
import torch.nn as nn


# ── Attention Pooling ─────────────────────────────────────────────────────────

class AttentionPooling(nn.Module):
    """
    Attention pooling over the time dimension.

    Produces a clip-level prediction by computing a weighted sum of
    frame-level predictions, where the weights (attention) are learned.

    Reference: DCASE 2019 / BirdCLEF standard SED head.

    Args:
        in_features  : Backbone output channels (C).
        num_classes  : Number of target species.
    """

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.att = nn.Linear(in_features, num_classes)   # attention logits
        self.cls = nn.Linear(in_features, num_classes)   # class logits

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, C) feature sequence.

        Returns:
            clip_pred  : (B, num_classes) — sigmoid clip-level probabilities.
            frame_pred : (B, T, num_classes) — sigmoid frame-level probabilities.
        """
        att_w = torch.softmax(self.att(x), dim=1)         # (B, T, num_classes)
        frame_pred = torch.sigmoid(self.cls(x))            # (B, T, num_classes)
        clip_pred = (att_w * frame_pred).sum(dim=1)        # (B, num_classes)
        return clip_pred, frame_pred


# ── SED Model ─────────────────────────────────────────────────────────────────

class SEDModel(nn.Module):
    """
    Sound Event Detection model — CNN backbone + attention pooling.

    Args:
        backbone     : timm model name. Recommended options:
                         "tf_efficientnetv2_s_in21k"   (2nd/5th place BirdCLEF25)
                         "tf_efficientnet_b3_ns"        (5th place, noisy-student)
                         "tf_efficientnet_b0_ns"        (5th place, lightweight)
                         "eca_nfnet_l0"                 (2nd place)
        num_classes  : Number of target species (234 for BirdClef 2026).
        in_chans     : 1 for mono mel spectrogram (timm adapts the first conv).
        pretrained   : Load ImageNet / ImageNet-21k pretrained weights.
        drop_rate    : Dropout on backbone.
    """

    def __init__(
        self,
        backbone: str = "tf_efficientnetv2_s_in21k",
        num_classes: int = 234,
        in_chans: int = 1,
        pretrained: bool = True,
        drop_rate: float = 0.3,
    ):
        super().__init__()

        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm is required for SEDModel.\n"
                "Install with: pip install timm"
            )

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,       # remove classifier
            global_pool="",      # remove global pooling (keep spatial dims)
            drop_rate=drop_rate,
        )

        # Probe feature channels with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, in_chans, 128, 501)
            feat = self.backbone(dummy)          # (1, C, H', W')
            self.feature_dim = feat.shape[1]

        self.attention = AttentionPooling(self.feature_dim, num_classes)
        self.num_classes = num_classes
        self.backbone_name = backbone

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[SEDModel] backbone={backbone}  feature_dim={self.feature_dim}  "
              f"params={n_params:,}")

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 1, n_mels, T) mel spectrogram tensor.

        Returns:
            clip_pred  : (B, num_classes) clip-level sigmoid probabilities.
            frame_pred : (B, T', num_classes) frame-level sigmoid probabilities.
        """
        feat = self.backbone(x)          # (B, C, H', W')
        feat = feat.mean(dim=2)          # pool frequency axis  → (B, C, W')
        feat = feat.permute(0, 2, 1)     # → (B, T', C)
        clip_pred, frame_pred = self.attention(feat)
        return clip_pred, frame_pred

    # ── Checkpointing ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save state dict to <path>.pt"""
        torch.save(self.state_dict(), path + ".pt")
        print(f"  SED checkpoint saved → {path}.pt")

    def load(self, path: str) -> None:
        """Load state dict from <path>.pt"""
        state = torch.load(path + ".pt", map_location="cpu")
        self.load_state_dict(state)
        print(f"  SED checkpoint loaded ← {path}.pt")


# ── Focal Loss (PyTorch) ──────────────────────────────────────────────────────

class FocalBCELossTorch(nn.Module):
    """
    Focal Binary Cross-Entropy loss (PyTorch version).

    BirdCLEF 2025 2nd & 5th place technique for class imbalance.

    Args:
        gamma          : Focusing parameter (0 = standard BCE, 2 = default).
        alpha          : Class balance weight (0.25 typical; -1 = disabled).
        label_smoothing: Applied before focal weighting.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, num_classes) raw logits.
            targets : (B, num_classes) float labels in [0, 1].
        """
        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        focal_w = (1.0 - p_t) ** self.gamma

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * focal_w * bce
        else:
            loss = focal_w * bce

        return loss.mean()
