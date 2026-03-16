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
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader

import torchaudio.transforms as AT

from src.utils.config import load_config, save_config, DotDict
from src.utils.metrics import competition_roc_auc, padded_cmap
from src.data.dataset import build_species_mapping, compute_class_weights
from src.data.mel_dataset import MelClipDataset, MelSoundscapeDataset
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


# ── Training step ─────────────────────────────────────────────────────────────

def _compute_loss(loss_fn, clip_pred, frame_logit, labels, clip_w=1.0, frame_w=0.0):
    """Compute loss supporting plain BCE (loss_fn=None) or custom loss."""
    import torch.nn.functional as F
    if loss_fn is None:
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
) -> float:
    """
    Train one epoch on clip data, then on soundscape data (if provided).

    ss_data   : (N, clip_len) raw waveforms or (N, 1, n_mels, T) mel specs
    ss_labels : (N, num_classes) binary labels
    ss_oversample: repeat soundscape data this many times per epoch
    """
    model.train()
    total_loss, n_batches = 0.0, 0

    for raw_batch, label_batch in loader:
        raw_batch   = raw_batch.to(device)
        label_batch = label_batch.to(device)

        # Convert raw waveform → mel on GPU (60× faster than CPU librosa)
        if gpu_mel_tf is not None:
            mel_batch = apply_gpu_mel(raw_batch, gpu_mel_tf)
        else:
            mel_batch = raw_batch   # already mel if yield_raw_audio=False

        # Mixup augmentation (BirdCLEF 2025 1st place)
        if mixup_alpha > 0:
            batch_size = mel_batch.size(0)
            lam = torch.tensor(
                np.random.beta(mixup_alpha, mixup_alpha, size=(batch_size, 1)),
                dtype=torch.float32, device=device,
            )
            perm = torch.randperm(batch_size, device=device)
            lam_audio = lam.view(batch_size, 1, 1, 1)
            mel_batch = lam_audio * mel_batch + (1.0 - lam_audio) * mel_batch[perm]
            label_batch = lam * label_batch + (1.0 - lam) * label_batch[perm]

        optimizer.zero_grad()
        clip_pred, frame_logit = model(mel_batch)
        loss = _compute_loss(loss_fn, clip_pred, frame_logit, label_batch, clip_loss_w, frame_loss_w)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

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
        for start in range(0, len(val_data), batch_size):
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

    # ── Soundscape training data (domain adaptation) ──────────────────────────
    ss_train_data, ss_train_labels = None, None
    if config.training.get("use_soundscapes_in_train", False):
        print("\nLoading soundscape training data (domain adaptation) …")
        if len(_trn_files) > 0:
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
        min_rating=config.data.min_rating,
        max_files=config.data.get("max_files", None),
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

    # ── Optimizer & loss ──────────────────────────────────────────────────────
    if config.training.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.training.learning_rate
        )

    loss_mode = config.training.get("loss", "focal_bce")
    clip_loss_w  = config.training.get("clip_loss_weight",  1.0)
    frame_loss_w = config.training.get("frame_loss_weight", 0.0)
    if loss_mode == "bce":
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
    else:
        loss_fn = FocalBCELossTorch(
            gamma=config.training.get("focal_gamma", 2.0),
            alpha=config.training.get("focal_alpha", 0.25),
            label_smoothing=config.training.label_smoothing,
        )
        print(f"Loss: FocalBCE (gamma={config.training.get('focal_gamma',2.0)}, "
              f"alpha={config.training.get('focal_alpha',0.25)})")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_roc = 0.0
    best_epoch = 0
    val_batch_size = config.training.batch_size * 2
    epoch_history = []
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"  Experiment : {run_name}")
    print(f"  Backbone   : {config.model.backbone}")
    print(f"  GEMFreqPool: {use_gem}")
    print(f"  Epochs     : {config.training.epochs}")
    print(f"  Batch size : {config.training.batch_size}")
    print(f"{'='*60}\n")

    for epoch in range(1, config.training.epochs + 1):
        t0 = time.time()

        lr = (
            cosine_lr_with_warmup(
                epoch,
                config.training.epochs,
                config.training.learning_rate,
                config.training.warmup_epochs,
            )
            if config.training.scheduler == "cosine"
            else config.training.learning_rate
        )
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
        )

        val_roc, val_cmap = evaluate(model, val_data, val_labels, val_batch_size, device, gpu_mel_tf)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{config.training.epochs} | "
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
            ckpt_path = os.path.join(ckpt_dir, "best_sed")
            model.save(ckpt_path, epoch=epoch, metrics={"macro_auc": val_roc})
            print(f"  ↑ New best ROC-AUC={best_roc:.4f} (epoch {epoch})")
            if wandb_run:
                wandb_run.log({"val/best_roc_auc": best_roc, "epoch": epoch})

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
