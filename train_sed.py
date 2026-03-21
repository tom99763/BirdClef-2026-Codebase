"""BirdClef 2026 — SED Model Training Script

Trains a Sound Event Detection model (EfficientNetV2 + GEMFreqPool + attention)
using PyTorch, inspired by BirdCLEF 2025 top solutions (1st and 5th place).

Usage:
    # Standard run
    python train_sed.py --config configs/sed_default.yaml

    # Quick debug
    python train_sed.py --config configs/sed_debug.yaml

    # Override config values
    python train_sed.py --config configs/sed_default.yaml \\
        training.learning_rate=5e-4 \\
        model.backbone=tf_efficientnet_b3_ns \\
        experiment.name=sed-b3-ns
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader

import torchaudio.transforms as AT

from src.utils.config import load_config, save_config, DotDict
from src.utils.metrics import competition_roc_auc, padded_cmap
from src.data.dataset import build_species_mapping, compute_class_weights
from src.data.mel_dataset import MelClipDataset, MelSoundscapeDataset, SoftPseudoSoundscapeDataset
from src.model.sed_model import SEDModel, FocalBCELossTorch


# ── Iterable wrapper for PyTorch DataLoader ───────────────────────────────────

class _IterableWrapper(IterableDataset):
    """Wraps MelClipDataset.generate_samples() for PyTorch DataLoader."""

    def __init__(self, mel_dataset: MelClipDataset):
        self.ds = mel_dataset

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Each worker re-seeds with a unique seed so random crops differ
            np.random.seed(worker_info.seed % (2**32))
        for x, label in self.ds.generate_samples():
            yield torch.from_numpy(x.copy()), torch.from_numpy(label)


# ── GPU Mel Transform ──────────────────────────────────────────────────────────

def build_gpu_mel_transform(config, device: torch.device) -> nn.Module:
    """Build a torchaudio MelSpectrogram + AmplitudeToDB on the given device."""
    transform = nn.Sequential(
        AT.MelSpectrogram(
            sample_rate=config.audio.sample_rate,
            n_fft=config.mel.n_fft,
            hop_length=config.mel.hop_length,
            n_mels=config.mel.n_mels,
            f_min=config.mel.fmin,
            f_max=config.mel.fmax,
            power=2.0,
            norm="slaney",
            mel_scale="htk",
        ),
        AT.AmplitudeToDB(stype="power", top_db=80.0),
    ).to(device)
    return transform


@torch.no_grad()
def apply_gpu_mel(waveforms: torch.Tensor, mel_tf: nn.Module) -> torch.Tensor:
    """
    waveforms: (B, clip_samples) float32 on GPU
    Returns:   (B, 1, n_mels, T) min-max normalized mel on GPU

    Peak-normalizes each waveform before mel conversion (matches reference LB=0.862).
    """
    # Per-sample peak normalization: audio / max(|audio|)
    peak = waveforms.abs().amax(dim=1, keepdim=True).clamp(min=1e-7)
    waveforms = waveforms / peak

    mel = mel_tf(waveforms)                # (B, n_mels, T)
    flat = mel.reshape(mel.shape[0], -1)
    mel_min = flat.min(1, keepdim=True)[0].unsqueeze(-1)
    mel_max = flat.max(1, keepdim=True)[0].unsqueeze(-1)
    mel = (mel - mel_min) / (mel_max - mel_min + 1e-7)
    return mel.unsqueeze(1)                # (B, 1, n_mels, T)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="BirdClef 2026 SED Training")
    parser.add_argument("--config", default="configs/sed_default.yaml")
    parser.add_argument("overrides", nargs="*",
                        help="Config overrides in key=value format")
    parser.add_argument("--gpu", default=None,
                        help="CUDA_VISIBLE_DEVICES (e.g. 0, 1, 0,1)")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint (.pt) to resume training from")
    parser.add_argument("--extra_epochs", type=int, default=20,
                        help="Additional epochs to train when resuming (default: 20)")
    parser.add_argument("--pretrained_backbone", default=None,
                        help="Path to backbone state dict from embed distillation "
                             "(checkpoints/embed-distill-b0-v1/best_backbone.pt). "
                             "Initialises backbone+gem before training.")
    parser.add_argument("--no_freeze_backbone", action="store_true",
                        help="Load pretrained_backbone weights but do NOT freeze them. "
                             "Use backbone_lr_multiplier in config to set differential LR.")
    return parser.parse_args()


def _parse_overrides(override_list: list) -> dict:
    result = {}
    for item in override_list:
        key, value = item.split("=", 1)
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except ValueError:
                pass
        if isinstance(value, str):
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.lower() in ("null", "none"):
                value = None
        result[key] = value
    return result


# ── Per-class positive weights for BCE loss ───────────────────────────────────

def compute_pos_weights(
    train_csv: str,
    species_to_idx: dict,
    num_classes: int,
    max_weight: float = 20.0,
) -> np.ndarray:
    """
    Compute per-class positive weight = sqrt(n_neg / n_pos), clipped to [1, max_weight].

    Used with BCEPosWeightLoss to balance rare species.
    Replaces the problematic FocalBCE alpha=0.25 which causes trivial-minimum collapse.
    """
    df = pd.read_csv(train_csv)
    n_pos = np.ones(num_classes, dtype=np.float32)   # Laplace smoothing avoids /0
    for _, row in df.iterrows():
        sp = str(row["primary_label"]).strip()
        if sp in species_to_idx:
            n_pos[species_to_idx[sp]] += 1.0
    n_total = float(len(df)) + num_classes
    n_neg = n_total - n_pos
    pos_w = np.sqrt(n_neg / n_pos).clip(1.0, max_weight)
    return pos_w.astype(np.float32)


class BCEPosWeightLoss(nn.Module):
    """
    Binary cross-entropy with per-class positive weights.

    Accepts sigmoid *probabilities* (not raw logits) because SEDModel's
    AttentionSEDHead already applies torch.sigmoid.

    This replaces FocalBCELossTorch(alpha=0.25) whose trivial minimum at
    all-zero predictions causes train_loss to freeze at ~0.127 from epoch 2.
    """

    def __init__(self, pos_weight: torch.Tensor):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.clamp(1e-7, 1.0 - 1e-7)
        bce = -(
            self.pos_weight * targets * preds.log()
            + (1.0 - targets) * (1.0 - preds).log()
        )
        return bce.mean()


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification (BirdCLEF 2025 multiple top teams).

    Key idea: use different focusing levels for positive and negative examples.
      - gamma_pos: focusing parameter for positives (usually 0 — no focusing, keep all)
      - gamma_neg: focusing parameter for negatives (usually 4 — hard negative mining)
      - clip: probability margin shift; clips easy negatives below this threshold,
              which reduces the contribution of very-easy negatives.

    Reference: "Asymmetric Loss For Multi-Label Classification" (Ridnik et al., 2021).
    Commonly used in BirdCLEF 2025 1st/2nd/3rd place solutions.

    Accepts sigmoid *probabilities* (not raw logits).
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0, clip: float = 0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        preds = preds.clamp(eps, 1.0 - eps)

        # Shift negatives: p_m = max(p - clip, 0)
        preds_neg = (preds - self.clip).clamp(min=0)

        # Positive component: -(1-p)^gamma_pos * log(p)
        loss_pos = targets * (1.0 - preds).pow(self.gamma_pos) * preds.log()

        # Negative component: -(p_m)^gamma_neg * log(1 - p_m)
        loss_neg = (1.0 - targets) * preds_neg.pow(self.gamma_neg) * (1.0 - preds_neg).log()

        return -(loss_pos + loss_neg).mean()


def apply_cutmix(
    mel_batch: torch.Tensor,
    label_batch: torch.Tensor,
    alpha: float = 1.0,
) -> tuple:
    """
    CutMix augmentation on mel spectrograms (BirdCLEF 2025 multiple teams).

    Cuts a random rectangular patch from one sample's mel and pastes it
    into another sample, mixing labels proportionally to patch area.

    Args:
        mel_batch:   (B, C, H, W) mel spectrograms on GPU.
        label_batch: (B, num_classes) soft labels.
        alpha:       Beta distribution parameter for mixing ratio.

    Returns:
        (mixed_mel, mixed_labels) with the same shapes.
    """
    B, C, H, W = mel_batch.shape
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(B, device=mel_batch.device)

    # Random bounding box whose area fraction ≈ 1 - lam
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    cy = np.random.randint(H)
    cx = np.random.randint(W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)

    mel_batch = mel_batch.clone()
    mel_batch[:, :, y1:y2, x1:x2] = mel_batch[perm, :, y1:y2, x1:x2]

    # Adjust lambda based on actual patch area
    lam_actual = 1.0 - (y2 - y1) * (x2 - x1) / float(H * W)
    mixed_labels = lam_actual * label_batch + (1.0 - lam_actual) * label_batch[perm]
    return mel_batch, mixed_labels


def apply_freq_mixstyle(
    mel_batch: torch.Tensor,
    alpha: float = 0.6,
    prob: float = 0.5,
) -> torch.Tensor:
    """
    Freq-MixStyle: mix frequency-axis mean/std statistics between batch samples.
    (DG-SED, DCASE 2024, arXiv 2407.03654)

    Normalizes each mel along the time axis (per-frequency instance norm), then
    re-scales with a convex combination of two samples' statistics.  This
    synthesizes virtual recording conditions bridging directional mics (train)
    and omnidirectional ARUs (test), with no inference cost.
    """
    B = mel_batch.size(0)
    if B < 2:
        return mel_batch

    mask = torch.rand(B, device=mel_batch.device) < prob
    if not mask.any():
        return mel_batch

    # Per-frequency mean/std over the time dimension  →  (B, C, H, 1)
    mean = mel_batch.mean(dim=3, keepdim=True)
    std  = mel_batch.var(dim=3, keepdim=True).add_(1e-6).sqrt_()

    # Instance-normalize
    mel_norm = (mel_batch - mean) / std

    # Mixing coefficient and permutation
    lam  = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(B, device=mel_batch.device)

    mixed_mean = lam * mean + (1.0 - lam) * mean[perm]
    mixed_std  = lam * std  + (1.0 - lam) * std[perm]

    mel_mixed = mel_norm * mixed_std + mixed_mean

    mel_out = mel_batch.clone()
    mel_out[mask] = mel_mixed[mask]
    return mel_out


def apply_multilabel_mixup(
    mel_batch: torch.Tensor,
    label_batch: torch.Tensor,
    n_clips: int = 3,
) -> tuple:
    """
    Multi-clip mixup with Dirichlet-sampled gains (BirdSet 2024, arXiv 2403.10380).

    Extends pairwise mixup to n_clips simultaneous clips, simulating the
    multi-species chorus density of Pantanal ARU recordings.  Labels become
    soft multi-hot vectors weighted by each clip's Dirichlet gain.
    """
    B = mel_batch.size(0)
    n_clips = max(2, n_clips)

    # Dirichlet(1,...,1) weights: (B, n_clips)
    weights_np = np.random.dirichlet([1.0] * n_clips, size=B).astype(np.float32)
    weights = torch.from_numpy(weights_np).to(mel_batch.device)  # (B, n_clips)

    # First clip is the sample itself
    mel_mixed   = weights[:, 0].view(B, 1, 1, 1) * mel_batch
    label_mixed = weights[:, 0:1] * label_batch

    for k in range(1, n_clips):
        perm        = torch.randperm(B, device=mel_batch.device)
        w_mel       = weights[:, k].view(B, 1, 1, 1)
        mel_mixed   = mel_mixed   + w_mel            * mel_batch[perm]
        label_mixed = label_mixed + weights[:, k:k+1] * label_batch[perm]

    return mel_mixed, label_mixed


# ── LR schedule ──────────────────────────────────────────────────────────────

def cosine_lr_with_warmup(
    epoch: int,
    total_epochs: int,
    base_lr: float,
    warmup_epochs: int,
) -> float:
    if epoch <= warmup_epochs:
        return base_lr * epoch / max(warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))


def cosine_warm_restarts_lr(
    epoch: int,
    base_lr: float,
    T_0: int = 5,
    T_mult: int = 1,
    eta_min: float = 1e-6,
) -> float:
    """CosineAnnealingWarmRestarts: LR resets every T_0 epochs.
    Matches torch.optim.lr_scheduler.CosineAnnealingWarmRestarts semantics.
    epoch is 1-indexed (first epoch = 1).
    """
    ep = epoch - 1  # convert to 0-indexed
    t = T_0
    while ep >= t:
        ep -= t
        t = t * T_mult
    progress = ep / max(t - 1, 1)
    return eta_min + 0.5 * (base_lr - eta_min) * (1.0 + np.cos(np.pi * progress))


# ── Training step ─────────────────────────────────────────────────────────────

def _compute_loss(loss_fn, clip_pred, frame_logit, labels, clip_w=1.0, frame_w=0.0,
                  loss_mode="bce", label_smoothing=0.0):
    """Compute loss supporting plain BCE (loss_fn=None), custom loss, or CrossEntropy."""
    import torch.nn.functional as F
    if loss_mode == "ce":
        # CrossEntropy: treat each clip as single-label (argmax of soft labels).
        # Recovers logits from sigmoid probs via torch.logit (numerically safe).
        # BirdCLEF 2024 1st place: +0.044 over BCE on EfficientNet-B0 SED.
        logits = torch.logit(clip_pred.clamp(1e-6, 1.0 - 1e-6))
        hard_labels = labels.argmax(dim=1)          # (B,) integer class index
        return clip_w * F.cross_entropy(logits, hard_labels, label_smoothing=label_smoothing)
    elif loss_mode == "soft_ce":
        # Soft CrossEntropy: compatible with mixup soft labels.
        # Instead of argmax→hard label, computes CE against the full soft distribution.
        # loss = -sum(labels * log_softmax(logits)) — preserves mixup's mixed targets.
        logits = torch.logit(clip_pred.clamp(1e-6, 1.0 - 1e-6))
        log_probs = F.log_softmax(logits, dim=1)    # (B, C)
        # Normalize soft labels to sum=1 (mixup may produce rows summing to 1 already)
        soft = labels / (labels.sum(dim=1, keepdim=True).clamp(min=1e-6))
        return clip_w * -(soft * log_probs).sum(dim=1).mean()
    elif loss_fn is None:
        # Plain BCE: clip uses sigmoid probs, frame uses raw logits
        clip_loss = F.binary_cross_entropy(clip_pred, labels)
        if frame_w > 0 and frame_logit is not None:
            frame_labels = labels.unsqueeze(1).expand_as(frame_logit)
            frame_loss = F.binary_cross_entropy_with_logits(frame_logit, frame_labels)
            return clip_w * clip_loss + frame_w * frame_loss
        return clip_loss
    else:
        return loss_fn(clip_pred, labels)


def train_epoch(
    model: SEDModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    mixup_alpha: float,
    gpu_mel_tf: nn.Module = None,
    ss_data: np.ndarray = None,
    ss_labels: np.ndarray = None,
    ss_batch_size: int = 64,
    ss_oversample: int = 3,
    clip_loss_w: float = 1.0,
    frame_loss_w: float = 0.0,
    loss_mode: str = "bce",
    label_smoothing: float = 0.0,
    freq_mask_param: int = 0,
    time_mask_param: int = 0,
    cutmix_alpha: float = 0.0,
    freq_mixstyle_prob: float = 0.0,
    freq_mixstyle_alpha: float = 0.6,
    clip_mix_n_clips: int = 2,
) -> float:
    """
    Train one epoch on clip data, then on soundscape data (if provided).

    ss_data   : (N, clip_len) raw waveforms or (N, 1, n_mels, T) mel specs
    ss_labels : (N, num_classes) binary labels
    ss_oversample: repeat soundscape data this many times per epoch
    """
    model.train()
    total_loss, n_batches = 0.0, 0

    pbar = tqdm(loader, desc="  train", ncols=100, leave=False, file=sys.stdout, mininterval=30)
    for raw_batch, label_batch in pbar:
        raw_batch   = raw_batch.to(device)
        label_batch = label_batch.to(device)

        # Convert raw waveform → mel on GPU (60× faster than CPU librosa)
        if gpu_mel_tf is not None:
            mel_batch = apply_gpu_mel(raw_batch, gpu_mel_tf)
        else:
            mel_batch = raw_batch   # already mel if yield_raw_audio=False

        # Frequency masking on mel spectrogram (BirdCLEF 2024 1st place, +0.013 LB)
        if freq_mask_param > 0:
            import torchaudio.transforms as TAT
            mel_batch = TAT.FrequencyMasking(freq_mask_param=freq_mask_param)(mel_batch)

        # Time masking (SpecAugment — competitor uses time_mask_param=30)
        if time_mask_param > 0:
            import torchaudio.transforms as TAT
            mel_batch = TAT.TimeMasking(time_mask_param=time_mask_param)(mel_batch)

        # Freq-MixStyle: mix frequency-axis statistics (DG-SED, DCASE 2024)
        if freq_mixstyle_prob > 0:
            mel_batch = apply_freq_mixstyle(mel_batch, alpha=freq_mixstyle_alpha,
                                            prob=freq_mixstyle_prob)

        # Mixup augmentation (BirdCLEF 2025 1st place)
        # If clip_mix_n_clips > 2, use multi-clip Dirichlet mixup (BirdSet 2024)
        if clip_mix_n_clips > 2 and mel_batch.size(0) >= 2:
            mel_batch, label_batch = apply_multilabel_mixup(mel_batch, label_batch,
                                                             n_clips=clip_mix_n_clips)
        elif mixup_alpha > 0:
            batch_size = mel_batch.size(0)
            lam = torch.tensor(
                np.random.beta(mixup_alpha, mixup_alpha, size=(batch_size, 1)),
                dtype=torch.float32, device=device,
            )
            perm = torch.randperm(batch_size, device=device)
            lam_audio = lam.view(batch_size, 1, 1, 1)
            mel_batch = lam_audio * mel_batch + (1.0 - lam_audio) * mel_batch[perm]
            label_batch = lam * label_batch + (1.0 - lam) * label_batch[perm]

        # CutMix augmentation (BirdCLEF 2025 multiple teams, +0.005-0.01 LB)
        if cutmix_alpha > 0 and mel_batch.size(0) >= 2:
            mel_batch, label_batch = apply_cutmix(mel_batch, label_batch, alpha=cutmix_alpha)

        optimizer.zero_grad()
        clip_pred, frame_logit = model(mel_batch)
        loss = _compute_loss(loss_fn, clip_pred, frame_logit, label_batch,
                             clip_loss_w, frame_loss_w, loss_mode, label_smoothing)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}", refresh=False)

    # ── Soundscape domain-adaptation pass ────────────────────────────────────
    # Train on soundscape segments (oversampled) to bridge train_audio → test domain.
    if ss_data is not None and ss_labels is not None and len(ss_data) > 0:
        for _ in range(ss_oversample):
            idx = np.random.permutation(len(ss_data))
            for start in range(0, len(ss_data), ss_batch_size):
                batch_idx = idx[start: start + ss_batch_size]
                if len(batch_idx) == 0:
                    continue
                batch = torch.tensor(
                    ss_data[batch_idx], dtype=torch.float32, device=device
                )
                label = torch.tensor(
                    ss_labels[batch_idx], dtype=torch.float32, device=device
                )
                if gpu_mel_tf is not None:
                    batch = apply_gpu_mel(batch, gpu_mel_tf)

                optimizer.zero_grad()
                clip_pred, frame_logit = model(batch)
                loss = _compute_loss(loss_fn, clip_pred, frame_logit, label, clip_loss_w, frame_loss_w)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

    return total_loss / max(n_batches, 1)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    model: SEDModel,
    val_data: np.ndarray,
    val_labels: np.ndarray,
    batch_size: int,
    device: torch.device,
    gpu_mel_tf: nn.Module = None,
) -> tuple:
    """Returns (roc_auc, padded_cmap) both on soundscape validation set."""
    model.eval()
    preds = []
    with torch.no_grad():
        for start in tqdm(range(0, len(val_data), batch_size),
                          desc="  val", ncols=100, leave=False, file=sys.stdout, mininterval=30):
            batch = torch.tensor(
                val_data[start: start + batch_size], dtype=torch.float32, device=device
            )
            if gpu_mel_tf is not None:
                batch = apply_gpu_mel(batch, gpu_mel_tf)
            clip_pred, _ = model(batch)
            preds.append(clip_pred.cpu().numpy())
    all_preds = np.concatenate(preds, axis=0)
    roc = competition_roc_auc(val_labels, all_preds)
    cmap = padded_cmap(val_labels, all_preds)
    return roc, cmap


# ── Result persistence ────────────────────────────────────────────────────────

def _save_results(
    out_dir: str,
    run_name: str,
    config: DotDict,
    epoch_history: list,
    best_roc: float,
    best_epoch: int,
    total_time_s: float = None,
    finished: bool = False,
) -> None:
    result = {
        "run_name": run_name,
        "finished": finished,
        "best_val_roc_auc": round(best_roc, 6),
        "best_epoch": best_epoch,
        "total_epochs_run": len(epoch_history),
        "total_time_s": round(total_time_s, 1) if total_time_s else None,
        "model_type": "SED",
        "hparams": {
            "backbone": config.model.backbone,
            "use_gem": config.model.get("use_gem", True),
            "in_chans": config.model.get("in_chans", 1),
            "n_mels": config.mel.n_mels,
            "hop_length": config.mel.hop_length,
            "learning_rate": config.training.learning_rate,
            "mixup_alpha": config.training.mixup_alpha,
            "label_smoothing": config.training.label_smoothing,
            "batch_size": config.training.batch_size,
            "epochs": config.training.epochs,
            "focal_gamma": config.training.get("focal_gamma", 2.0),
            "class_weight_mode": config.training.get("class_weight_mode", "none"),
        },
        "epoch_history": epoch_history,
    }
    path = os.path.join(out_dir, "result.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = load_config(args.config, _parse_overrides(args.overrides))

    import random
    random.seed(config.experiment.seed)
    np.random.seed(config.experiment.seed)
    torch.manual_seed(config.experiment.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Directories ──────────────────────────────────────────────────────────
    run_name = config.experiment.name
    out_dir = os.path.join(config.output.dir, run_name)
    ckpt_dir = os.path.join(config.output.checkpoint_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    save_config(config, os.path.join(out_dir, "config.yaml"))

    # ── WandB ─────────────────────────────────────────────────────────────────
    wandb_run = None
    if config.wandb.enabled:
        import wandb
        wandb_run = wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=run_name,
            config=dict(config),
            tags=list(config.wandb.tags),
        )
        print(f"WandB run: {wandb_run.url}")

    # ── Species mapping ───────────────────────────────────────────────────────
    target_species, species_to_idx = build_species_mapping(
        config.data.sample_submission_csv
    )
    num_classes = len(target_species)
    print(f"\nTarget species: {num_classes}")

    # ── Mel params ────────────────────────────────────────────────────────────
    mel_kw = dict(
        n_fft=config.mel.n_fft,
        hop_length=config.mel.hop_length,
        n_mels=config.mel.n_mels,
        fmin=config.mel.fmin,
        fmax=config.mel.fmax,
    )

    # ── Soundscape file-level train/val split (prevents data leak) ───────────
    use_gpu_mel = config.model.get("use_gpu_mel", True)
    ss_val_frac = config.training.get("soundscape_val_frac", 1.0)
    # Load full soundscape labels and split by file
    _ss_df_full = pd.read_csv(config.data.soundscapes_labels_csv)
    _ss_files   = sorted(_ss_df_full["filename"].unique())

    # Support explicit val file list for k-fold CV (soundscape_val_files_txt)
    _val_files_txt = config.data.get("soundscape_val_files_txt", None)
    if _val_files_txt and os.path.isfile(_val_files_txt):
        with open(_val_files_txt) as _f:
            _val_files = set(l.strip() for l in _f if l.strip())
        _trn_files = set(_ss_files) - _val_files
        print(f"[soundscape fold] Using explicit val file list: {_val_files_txt}")
    else:
        _n_val      = max(1, int(len(_ss_files) * ss_val_frac))
        _val_files  = set(_ss_files[-_n_val:])
        _trn_files  = set(_ss_files[:-_n_val]) if ss_val_frac < 1.0 else set()
    _ss_val_df  = _ss_df_full[_ss_df_full["filename"].isin(_val_files)]
    _ss_trn_df  = _ss_df_full[_ss_df_full["filename"].isin(_trn_files)]

    import tempfile, os as _os
    _tmp_val_csv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    _ss_val_df.to_csv(_tmp_val_csv.name, index=False); _tmp_val_csv.close()

    print(f"\nSoundscape split: val={len(_val_files)} files ({len(_ss_val_df)} clips)  "
          f"train={len(_trn_files)} files ({len(_ss_trn_df)} clips)  val_frac={ss_val_frac}")
    print(f"Loading validation soundscapes {'as raw audio' if use_gpu_mel else 'as mel spectrograms'} …")
    val_ds = MelSoundscapeDataset(
        soundscapes_dir=config.data.train_soundscapes_dir,
        labels_csv=_tmp_val_csv.name,
        species_to_idx=species_to_idx,
        num_classes=num_classes,
        sample_rate=config.audio.sample_rate,
        clip_duration=config.audio.clip_duration,
        yield_raw_audio=use_gpu_mel,
        **mel_kw,
    )
    val_data, val_labels = val_ds.get_all_samples()
    _os.unlink(_tmp_val_csv.name)
    print(f"Validation clips: {len(val_data)}  shape: {val_data[0].shape}")

    # ── Extra pseudo-labeled soundscape data ──────────────────────────────────
    extra_ss_csv = config.data.get("extra_soundscape_csv", None)
    if extra_ss_csv and os.path.isfile(extra_ss_csv):
        _extra_df = pd.read_csv(extra_ss_csv)
        # Exclude val files to prevent leakage
        _extra_df = _extra_df[~_extra_df["filename"].isin(_val_files)].reset_index(drop=True)
        print(f"\nExtra pseudo soundscapes: {len(_extra_df)} clips "
              f"from {_extra_df['filename'].nunique()} files (excluded {len(_val_files)} val files)")
        _ss_trn_df = pd.concat([_ss_trn_df, _extra_df], ignore_index=True)
    elif extra_ss_csv:
        print(f"\nWARN: extra_soundscape_csv not found: {extra_ss_csv}")

    # ── Soundscape training data (domain adaptation) ──────────────────────────
    ss_train_data, ss_train_labels = None, None
    if config.training.get("use_soundscapes_in_train", False):
        print("\nLoading soundscape training data (domain adaptation) …")
        if len(_trn_files) > 0 or (extra_ss_csv and os.path.isfile(extra_ss_csv or "")):
            _tmp_trn_csv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
            _ss_trn_df.to_csv(_tmp_trn_csv.name, index=False); _tmp_trn_csv.close()
            _ss_trn_ds = MelSoundscapeDataset(
                soundscapes_dir=config.data.train_soundscapes_dir,
                labels_csv=_tmp_trn_csv.name,
                species_to_idx=species_to_idx,
                num_classes=num_classes,
                sample_rate=config.audio.sample_rate,
                clip_duration=config.audio.clip_duration,
                yield_raw_audio=use_gpu_mel,
                **mel_kw,
            )
            ss_train_data, ss_train_labels = _ss_trn_ds.get_all_samples()
            _os.unlink(_tmp_trn_csv.name)
        else:
            # soundscape_val_frac=1.0 means ALL soundscapes for val only — no train domain adaptation
            ss_train_data, ss_train_labels = None, None
            print("  soundscape_val_frac=1.0: all soundscapes used for validation, none for training.")
        ss_oversample = config.training.get("soundscape_oversample", 3)
        if ss_train_data is not None:
            print(f"  Soundscape train clips: {len(ss_train_data)}  oversample={ss_oversample}×")

    # ── Soft pseudo soundscape data (KD from Perch soft probabilities) ─────────
    soft_pseudo_csv = config.data.get("soft_pseudo_csv", None)
    if soft_pseudo_csv and os.path.isfile(soft_pseudo_csv):
        _val_stems = [re.sub(r"\.ogg$", "", f, flags=re.IGNORECASE) for f in _val_files]
        _sp_ds = SoftPseudoSoundscapeDataset(
            soundscapes_dir=config.data.train_soundscapes_dir,
            soft_pseudo_csv=soft_pseudo_csv,
            species_list=target_species,
            num_classes=num_classes,
            sample_rate=config.audio.sample_rate,
            clip_duration=config.audio.clip_duration,
            yield_raw_audio=use_gpu_mel,
            val_stems=_val_stems,
            **mel_kw,
        )
        _sp_data, _sp_labels = _sp_ds.get_all_samples()
        soft_oversample = config.training.get("soft_pseudo_oversample", ss_oversample)
        if ss_train_data is not None:
            ss_train_data   = np.concatenate([ss_train_data,   _sp_data],   axis=0)
            ss_train_labels = np.concatenate([ss_train_labels, _sp_labels], axis=0)
        else:
            ss_train_data, ss_train_labels = _sp_data, _sp_labels
            ss_oversample = soft_oversample
        print(f"  Soft pseudo clips added: {len(_sp_data)}  "
              f"total soundscape+soft: {len(ss_train_data)}")
    elif soft_pseudo_csv:
        print(f"\nWARN: soft_pseudo_csv not found: {soft_pseudo_csv}")

    # ── Class weights ─────────────────────────────────────────────────────────
    class_weight_mode = config.training.get("class_weight_mode", "none")
    class_weights = None
    if class_weight_mode != "none":
        print(f"Computing class weights (mode={class_weight_mode}) …")
        class_weights = compute_class_weights(
            config.data.train_csv, species_to_idx, num_classes, mode=class_weight_mode
        )

    # ── Training data ─────────────────────────────────────────────────────────
    print("\nBuilding training dataset …")
    use_gpu_mel = config.model.get("use_gpu_mel", True)  # default: GPU mel for speed

    # soundscape_only: skip focal recordings entirely (train only on soundscapes)
    _soundscape_only = config.training.get("soundscape_only", False)
    _max_files = 0 if _soundscape_only else config.data.get("max_files", None)

    train_ds_obj = MelClipDataset(
        train_csv=config.data.train_csv,
        audio_dir=config.data.train_audio_dir,
        species_to_idx=species_to_idx,
        num_classes=num_classes,
        sample_rate=config.audio.sample_rate,
        clip_duration=config.audio.clip_duration,
        n_clips_per_file=config.audio.n_clips_per_file,
        is_train=True,
        use_secondary_labels=config.data.use_secondary_labels,
        secondary_label_weight=config.data.get("secondary_label_weight", 1.0),
        min_rating=config.data.min_rating,
        max_files=_max_files,
        augment_config=dict(config.augmentation),
        class_weights=class_weights,
        noise_dir=config.data.get("noise_dir", None),
        yield_raw_audio=use_gpu_mel,
        **mel_kw,
    )

    n_workers = 4 if use_gpu_mel else 0
    train_loader = DataLoader(
        _IterableWrapper(train_ds_obj),
        batch_size=config.training.batch_size,
        num_workers=n_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(n_workers > 0),
    )
    print(f"DataLoader: num_workers={n_workers}, gpu_mel={use_gpu_mel}")

    # GPU mel transform (only when use_gpu_mel=True)
    gpu_mel_tf = build_gpu_mel_transform(config, device) if use_gpu_mel else None

    # ── Model ─────────────────────────────────────────────────────────────────
    in_chans = config.model.get("in_chans", 1)
    use_gem  = config.model.get("use_gem", True)
    gem_p    = config.model.get("gem_p_init", 3.0)
    # n_frames = clip_samples / hop_length + 1
    clip_samples = config.audio.clip_duration * config.audio.sample_rate
    n_frames = clip_samples // config.mel.hop_length + 1

    print(f"\nBuilding SED model (backbone={config.model.backbone}, "
          f"in_chans={in_chans}, use_gem={use_gem}, n_frames={n_frames}) …")
    model = SEDModel(
        backbone=config.model.backbone,
        num_classes=num_classes,
        in_chans=in_chans,
        pretrained=config.model.pretrained,
        drop_rate=config.model.dropout,
        use_gem=use_gem,
        gem_p_init=gem_p,
        n_mels=config.mel.n_mels,
        n_frames=n_frames,
    ).to(device)

    # ── Load embed-distilled backbone weights (optional) ──────────────────────
    if args.pretrained_backbone:
        ckpt = torch.load(args.pretrained_backbone, map_location=device)
        bb_state = ckpt['backbone']
        # Filter out keys with shape mismatch (e.g. conv_stem.weight 1ch vs 3ch)
        model_bb_state = model.backbone.state_dict()
        bb_state = {k: v for k, v in bb_state.items()
                    if k in model_bb_state and v.shape == model_bb_state[k].shape}
        missing_bb, unexpected_bb = model.backbone.load_state_dict(
            bb_state, strict=False)
        if model.use_gem:
            missing_gem, _ = model.freq_pool.load_state_dict(
                ckpt['freq_pool'], strict=False)
        else:
            missing_gem = []
        print(f"  [embed-distill] Loaded backbone from {args.pretrained_backbone}")
        if missing_bb:
            print(f"    missing backbone keys: {missing_bb[:5]}")

        if args.no_freeze_backbone:
            # Full fine-tune: load weights but keep all params trainable
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  → backbone NOT frozen (full fine-tune, {n_trainable:,} trainable params)")
        else:
            # Freeze backbone + GEM — only train the SED head (like Perch embedding_head mode)
            for param in model.backbone.parameters():
                param.requires_grad = False
            if model.use_gem:
                for param in model.freq_pool.parameters():
                    param.requires_grad = False
            n_frozen  = sum(p.numel() for p in model.parameters() if not p.requires_grad)
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  → backbone + GEM frozen  ({n_frozen:,} params frozen, {n_trainable:,} trainable)")

    # ── Optimizer & loss ──────────────────────────────────────────────────────
    # Support differential LR: backbone gets lr * backbone_lr_multiplier, head gets lr
    backbone_lr_mult = config.training.get("backbone_lr_multiplier", 1.0)
    if (args.pretrained_backbone and not args.no_freeze_backbone) or backbone_lr_mult == 1.0:
        # Frozen backbone or no differential LR — single param group
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        param_groups = trainable_params
    else:
        # Differential LR: backbone/GEM at lower LR, head at full LR
        backbone_params = list(model.backbone.parameters())
        gem_params = list(model.freq_pool.parameters()) if model.use_gem else []
        head_params = [p for p in model.head.parameters()]
        param_groups = [
            {"params": backbone_params + gem_params,
             "lr": config.training.learning_rate * backbone_lr_mult,
             "name": "backbone"},
            {"params": head_params,
             "lr": config.training.learning_rate,
             "name": "head"},
        ]
        print(f"  → differential LR: backbone={config.training.learning_rate * backbone_lr_mult:.2e}  "
              f"head={config.training.learning_rate:.2e}  (backbone_lr_mult={backbone_lr_mult})")

    if config.training.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            param_groups, lr=config.training.learning_rate
        )

    loss_mode        = config.training.get("loss", "focal_bce")
    clip_loss_w      = config.training.get("clip_loss_weight",  1.0)
    frame_loss_w     = config.training.get("frame_loss_weight", 0.0)
    label_smoothing  = config.training.get("label_smoothing", 0.0)
    freq_mask_param  = config.augmentation.get("freq_mask_param", 0) \
                       if config.augmentation.get("freq_masking", False) else 0
    time_mask_param  = config.augmentation.get("time_mask_param", 0) \
                       if config.augmentation.get("time_masking", False) else 0

    if loss_mode in ("ce", "soft_ce"):
        loss_fn = None   # CE/SoftCE computed inline in _compute_loss
        print(f"Loss: {'SoftCE (mixup-compatible)' if loss_mode=='soft_ce' else 'CrossEntropy'}  "
              f"label_smoothing={label_smoothing}  clip_w={clip_loss_w}")
    elif loss_mode == "bce":
        loss_fn = None   # plain BCE computed inline
        print(f"Loss: BCE  clip_w={clip_loss_w}  frame_w={frame_loss_w}")
    elif loss_mode == "bce_pos_weight":
        print("Computing per-class positive weights …")
        pos_w = compute_pos_weights(config.data.train_csv, species_to_idx, num_classes)
        loss_fn = BCEPosWeightLoss(
            pos_weight=torch.tensor(pos_w, dtype=torch.float32, device=device)
        )
        print(f"Loss: BCEPosWeight  avg={pos_w.mean():.2f}  max={pos_w.max():.2f}  "
              f"min={pos_w.min():.2f}")
    elif loss_mode == "asl":
        loss_fn = AsymmetricLoss(
            gamma_neg=config.training.get("asl_gamma_neg", 4.0),
            gamma_pos=config.training.get("asl_gamma_pos", 0.0),
            clip=config.training.get("asl_clip", 0.05),
        )
        print(f"Loss: ASL (gamma_neg={config.training.get('asl_gamma_neg',4.0)}, "
              f"gamma_pos={config.training.get('asl_gamma_pos',0.0)}, "
              f"clip={config.training.get('asl_clip',0.05)})")
    else:
        loss_fn = FocalBCELossTorch(
            gamma=config.training.get("focal_gamma", 2.0),
            alpha=config.training.get("focal_alpha", 0.25),
            label_smoothing=config.training.label_smoothing,
        )
        print(f"Loss: FocalBCE (gamma={config.training.get('focal_gamma',2.0)}, "
              f"alpha={config.training.get('focal_alpha',0.25)})")

    # ── Resume from checkpoint ─────────────────────────────────────────────────
    start_epoch = 1
    best_roc = 0.0
    best_epoch = 0
    epoch_history = []
    total_epochs = config.training.epochs

    if args.resume:
        ckpt_path = args.resume
        print(f"\nResuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        # Remap competitor checkpoint keys if needed
        if any("gem_pool" in k for k in state):
            state = {k.replace("gem_pool", "freq_pool"): v for k, v in state.items()}
        model.load_state_dict(state)
        print(f"  Loaded model weights (epoch={ckpt.get('epoch','?')})")

        # Load previous epoch history from result.json
        result_path = os.path.join(out_dir, "result.json")
        if os.path.isfile(result_path):
            with open(result_path) as f:
                prev = json.load(f)
            epoch_history = prev.get("epoch_history", [])
            best_roc = prev.get("best_val_roc_auc", 0.0)
            best_epoch = prev.get("best_epoch", 0)
            start_epoch = len(epoch_history) + 1
            print(f"  Restored {len(epoch_history)} epochs | best={best_roc:.4f}@ep{best_epoch}")

        total_epochs = start_epoch - 1 + args.extra_epochs
        print(f"  Continuing ep{start_epoch} → ep{total_epochs} (+{args.extra_epochs} extra epochs)\n")

    # ── Top-k checkpoint tracking for Model Soup ──────────────────────────────
    import heapq
    soup_topk = config.training.get("save_topk_checkpoints", 3)
    topk_heap = []   # min-heap of (val_roc, epoch, path)

    # ── Early stopping ─────────────────────────────────────────────────────────
    early_stop_patience = config.training.get("early_stopping_patience", 0)
    no_improve_count = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    val_batch_size = config.training.batch_size * 2
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"  Experiment : {run_name}")
    print(f"  Backbone   : {config.model.backbone}")
    print(f"  GEMFreqPool: {use_gem}")
    print(f"  Epochs     : {total_epochs} (start={start_epoch})")
    print(f"  Batch size : {config.training.batch_size}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, total_epochs + 1):
        t0 = time.time()

        if config.training.scheduler == "cosine":
            lr = cosine_lr_with_warmup(
                epoch, total_epochs,
                config.training.learning_rate,
                config.training.warmup_epochs,
            )
        elif config.training.scheduler == "warm_restarts":
            lr = cosine_warm_restarts_lr(
                epoch,
                config.training.learning_rate,
                T_0=config.training.get("scheduler_T0", 5),
                T_mult=config.training.get("scheduler_T_mult", 1),
                eta_min=config.training.get("scheduler_eta_min", 1e-6),
            )
        else:
            lr = config.training.learning_rate
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, device,
            mixup_alpha=config.training.mixup_alpha,
            gpu_mel_tf=gpu_mel_tf,
            ss_data=ss_train_data,
            ss_labels=ss_train_labels,
            ss_batch_size=config.training.batch_size,
            ss_oversample=config.training.get("soundscape_oversample", 3),
            clip_loss_w=clip_loss_w,
            frame_loss_w=frame_loss_w,
            loss_mode=loss_mode,
            label_smoothing=label_smoothing,
            freq_mask_param=freq_mask_param,
            time_mask_param=time_mask_param,
            cutmix_alpha=config.training.get("cutmix_alpha", 0.0),
            freq_mixstyle_prob=config.augmentation.get("freq_mixstyle_prob", 0.0)
                               if config.augmentation.get("freq_mixstyle", False) else 0.0,
            freq_mixstyle_alpha=config.augmentation.get("freq_mixstyle_alpha", 0.6),
            clip_mix_n_clips=config.augmentation.get("clip_mix_n_clips", 2),
        )

        val_roc, val_cmap = evaluate(model, val_data, val_labels, val_batch_size, device, gpu_mel_tf)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{total_epochs} | "
            f"loss={train_loss:.4f} | "
            f"val_roc_auc={val_roc:.4f} | val_cmap={val_cmap:.4f} | "
            f"lr={lr:.2e} | {elapsed:.1f}s"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_roc_auc": round(val_roc, 6),
            "val_cmap": round(val_cmap, 6),
            "lr": round(float(lr), 8),
            "epoch_time_s": round(elapsed, 1),
        }
        epoch_history.append(epoch_record)
        _save_results(out_dir, run_name, config, epoch_history, best_roc, best_epoch)

        if wandb_run:
            wandb_run.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/roc_auc": val_roc,
                "val/padded_cmap": val_cmap,
                "lr": lr,
            })

        if val_roc > best_roc:
            best_roc = val_roc
            best_epoch = epoch
            no_improve_count = 0
            ckpt_path = os.path.join(ckpt_dir, "best_sed")
            model.save(ckpt_path, epoch=epoch, metrics={"macro_auc": val_roc})
            print(f"  ↑ New best ROC-AUC={best_roc:.4f} (epoch {epoch})")
            if wandb_run:
                wandb_run.log({"val/best_roc_auc": best_roc, "epoch": epoch})
        else:
            no_improve_count += 1

        # ── Model Soup: save top-k checkpoints by val AUC ─────────────────────
        if soup_topk > 0:
            topk_path = os.path.join(ckpt_dir, f"soup_ep{epoch:03d}_sed")
            model.save(topk_path, epoch=epoch, metrics={"macro_auc": val_roc})
            heapq.heappush(topk_heap, (val_roc, epoch, topk_path + ".pt"))
            if len(topk_heap) > soup_topk:
                _, _, old_path = heapq.heappop(topk_heap)
                if os.path.isfile(old_path):
                    os.remove(old_path)

        # ── Early stopping check ───────────────────────────────────────────────
        if early_stop_patience > 0 and no_improve_count >= early_stop_patience:
            print(f"  ✗ Early stopping at epoch {epoch}: "
                  f"no improvement for {early_stop_patience} epochs "
                  f"(best={best_roc:.4f} @ ep{best_epoch})")
            break

    total_time = time.time() - t_start
    _save_results(
        out_dir, run_name, config, epoch_history, best_roc, best_epoch,
        total_time_s=total_time, finished=True,
    )

    print(f"\nSED training complete.  Best val ROC-AUC: {best_roc:.4f}  (epoch {best_epoch})")
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
