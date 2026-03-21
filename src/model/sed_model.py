"""Sound Event Detection (SED) model for BirdClef 2026.

Architecture inspired by BirdCLEF 2025 top solutions:
  - 1st place  : SED models for frame-level temporal annotation
  - 5th place  : 13 SED models (EfficientNetV2-S, EfficientNet-B3-NS, B0-NS, B3)
  - 2nd place  : EfficientNetV2-S (in21k) + ECA-NFNet-L0

Pipeline:
  Mel spectrogram (C, n_mels, T)
      → CNN backbone (EfficientNetV2-S / EfficientNet-B3-NS via timm)
      → GEMFreqPool → time sequence (B, C, T')
      → AttentionSEDHead → clip-level prediction (B, num_classes)

Key improvement over mean pooling: GEMFreqPool with learnable p-norm.
LB=0.862 baseline uses B0 + GEMFreqPool.

Requirements:
    pip install torch timm torchaudio
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── GEM Frequency Pooling ─────────────────────────────────────────────────────

class GEMFreqPool(nn.Module):
    """
    Generalized Mean (GEM) pooling over the frequency axis.

    Learnable exponent p allows the model to interpolate between
    avg pooling (p=1) and max pooling (p→∞). Default p=3 works well.

    Input : (B, C, H, W)  where H = freq bins, W = time frames
    Output: (B, C, W)     frequency axis pooled
    """

    def __init__(self, p_init: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p.clamp(min=1.0)
        return x.clamp(min=self.eps).pow(p).mean(dim=2).pow(1.0 / p)


# ── Attention SED Head (Conv1d, matches BirdCLEF 2025 baseline) ───────────────

class AttentionSEDHead(nn.Module):
    """
    Conv1d attention head over the time axis.

    Compared to the simpler Linear attention head, this uses:
    - tanh attention (bounded) + softmax over time
    - A projection FC before attention for better feature mixing

    Input : (B, C, T)  where C = backbone feature dim
    Output: clip_pred (B, num_classes), frame_logit (B, T, num_classes)
    """

    def __init__(self, feat_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.att_conv = nn.Conv1d(feat_dim, num_classes, kernel_size=1)
        self.cls_conv = nn.Conv1d(feat_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # x: (B, C, T)
        x = x.permute(0, 2, 1)         # → (B, T, C)
        x = self.fc(x)
        x = x.permute(0, 2, 1)         # → (B, C, T)
        att = F.softmax(torch.tanh(self.att_conv(x)), dim=-1)   # (B, num_classes, T)
        cls = self.cls_conv(x)                                   # (B, num_classes, T)
        clipwise_logit = (att * cls).sum(dim=-1)                 # (B, num_classes)
        return torch.sigmoid(clipwise_logit), cls.permute(0, 2, 1)  # (B, C), (B, T, C)


# ── Legacy Linear Attention (kept for backward compat) ────────────────────────

class AttentionPooling(nn.Module):
    """Linear attention pooling (original, kept for backward compat)."""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.att = nn.Linear(in_features, num_classes)
        self.cls = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor):
        att_w = torch.softmax(self.att(x), dim=1)
        frame_pred = torch.sigmoid(self.cls(x))
        clip_pred = (att_w * frame_pred).sum(dim=1)
        return clip_pred, frame_pred


# ── SED Model ─────────────────────────────────────────────────────────────────

class SEDModel(nn.Module):
    """
    Sound Event Detection model — CNN backbone + GEMFreqPool + AttentionSEDHead.

    Args:
        backbone    : timm model name. Options:
                        "tf_efficientnetv2_s_in21k"    (BirdCLEF25 2nd/5th place, strongest)
                        "tf_efficientnet_b0.ns_jft_in1k" (notebook baseline, LB=0.862)
                        "tf_efficientnet_b3_ns"         (5th place, noisy-student)
        num_classes : 234 for BirdClef 2026.
        in_chans    : 1 (default, mono mel) or 3 (replicated for EfficientNet pretrain).
        pretrained  : Load ImageNet / ImageNet-21k pretrained weights.
        drop_rate   : Dropout on backbone.
        use_gem     : Use GEMFreqPool (True) or simple mean pooling (False, legacy).
        gem_p_init  : Initial p for GEMFreqPool.
        n_mels      : Mel bins (used for dummy probe only).
        n_frames    : Time frames (used for dummy probe only).
    """

    def __init__(
        self,
        backbone: str = "tf_efficientnetv2_s_in21k",
        num_classes: int = 234,
        in_chans: int = 1,
        pretrained: bool = True,
        drop_rate: float = 0.3,
        use_gem: bool = True,
        gem_p_init: float = 3.0,
        n_mels: int = 128,
        n_frames: int = 501,
    ):
        super().__init__()

        try:
            import timm
        except ImportError:
            raise ImportError("timm is required: pip install timm")

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
            global_pool="",
            drop_rate=drop_rate,
        )

        # Probe feature channels
        with torch.no_grad():
            dummy = torch.zeros(1, in_chans, n_mels, n_frames)
            feat = self.backbone(dummy)      # (1, C, H', W')
            self.feature_dim = feat.shape[1]

        self.use_gem = use_gem
        self._in_chans = in_chans
        if use_gem:
            self.freq_pool = GEMFreqPool(p_init=gem_p_init)
            self.head = AttentionSEDHead(self.feature_dim, num_classes, dropout=drop_rate)
        else:
            self.head = AttentionPooling(self.feature_dim, num_classes)

        self.num_classes = num_classes
        self.backbone_name = backbone

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[SEDModel] backbone={backbone}  feat_dim={self.feature_dim}  "
              f"gem={use_gem}  in_chans={in_chans}  params={n_params:,}")

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 1, n_mels, T) mel spectrogram tensor.
               If in_chans=3 but x has 1 channel, channels are replicated.

        Returns:
            clip_pred  : (B, num_classes) clip-level sigmoid probabilities.
            frame_pred : (B, T', num_classes) frame-level probabilities.
        """
        if hasattr(self, '_in_chans') and self._in_chans == 3 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)     # replicate mono mel to 3-channel
        # Skip autograd tracking for frozen backbone (saves memory + speeds up forward)
        backbone_frozen = not any(p.requires_grad for p in self.backbone.parameters())
        if backbone_frozen:
            with torch.no_grad():
                feat = self.backbone(x)       # (B, C, H', W')
                if self.use_gem:
                    feat = self.freq_pool(feat)
                else:
                    feat = feat.mean(dim=2)
                    feat = feat.permute(0, 2, 1)
        else:
            feat = self.backbone(x)           # (B, C, H', W')
            if self.use_gem:
                feat = self.freq_pool(feat)   # GEM pool freq → (B, C, T')
            else:
                feat = feat.mean(dim=2)       # mean pool freq → (B, C, T')
                feat = feat.permute(0, 2, 1)  # → (B, T', C)
        return self.head(feat)

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save(self, path: str, epoch: int = 0, metrics: dict = None) -> None:
        """Save checkpoint as dict (compatible with submission notebook)."""
        torch.save({
            "model_state_dict": self.state_dict(),
            "epoch": epoch,
            "metrics": metrics or {},
        }, path + ".pt")
        print(f"  SED checkpoint saved → {path}.pt  (epoch={epoch})")

    def load(self, path: str):
        """Load checkpoint. Handles both dict format and raw state_dict."""
        ckpt = torch.load(path + ".pt", map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            self.load_state_dict(ckpt["model_state_dict"])
            epoch = ckpt.get("epoch", 0)
            metrics = ckpt.get("metrics", {})
            print(f"  SED checkpoint loaded ← {path}.pt  "
                  f"(epoch={epoch}, metrics={metrics})")
            return epoch, metrics
        else:
            # Legacy: raw state dict
            self.load_state_dict(ckpt)
            print(f"  SED checkpoint loaded ← {path}.pt  (legacy format)")
            return 0, {}


# ── Focal Loss (PyTorch) ──────────────────────────────────────────────────────

class FocalBCELossTorch(nn.Module):
    """
    Focal Binary Cross-Entropy loss.
    BirdCLEF 2025 2nd & 5th place technique for class imbalance.
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
        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        focal_w = (1.0 - p_t) ** self.gamma
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * focal_w * bce
        else:
            loss = focal_w * bce
        return loss.mean()
