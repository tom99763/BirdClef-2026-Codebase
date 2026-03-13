"""BirdClef 2026 — SED Model Training Script

Trains a Sound Event Detection model (EfficientNetV2 + attention pooling)
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
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader

from src.utils.config import load_config, save_config, DotDict
from src.utils.metrics import padded_cmap
from src.data.dataset import build_species_mapping, compute_class_weights
from src.data.mel_dataset import MelClipDataset, MelSoundscapeDataset
from src.model.sed_model import SEDModel, FocalBCELossTorch


# ── Iterable wrapper for PyTorch DataLoader ───────────────────────────────────

class _IterableWrapper(IterableDataset):
    """Wraps MelClipDataset.generate_samples() for PyTorch DataLoader."""

    def __init__(self, mel_dataset: MelClipDataset):
        self.ds = mel_dataset

    def __iter__(self):
        for mel, label in self.ds.generate_samples():
            yield torch.from_numpy(mel), torch.from_numpy(label)


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

def train_epoch(
    model: SEDModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    mixup_alpha: float,
) -> float:
    model.train()
    total_loss, n_batches = 0.0, 0

    for mel_batch, label_batch in loader:
        mel_batch = mel_batch.to(device)
        label_batch = label_batch.to(device)

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
        clip_pred, _ = model(mel_batch)
        loss = loss_fn(clip_pred, label_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    model: SEDModel,
    val_mels: np.ndarray,
    val_labels: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(val_mels), batch_size):
            batch = torch.tensor(
                val_mels[start: start + batch_size], dtype=torch.float32, device=device
            )
            clip_pred, _ = model(batch)
            preds.append(clip_pred.cpu().numpy())
    return padded_cmap(val_labels, np.concatenate(preds, axis=0))


# ── Result persistence ────────────────────────────────────────────────────────

def _save_results(
    out_dir: str,
    run_name: str,
    config: DotDict,
    epoch_history: list,
    best_cmap: float,
    best_epoch: int,
    total_time_s: float = None,
    finished: bool = False,
) -> None:
    result = {
        "run_name": run_name,
        "finished": finished,
        "best_val_cmap": round(best_cmap, 6),
        "best_epoch": best_epoch,
        "total_epochs_run": len(epoch_history),
        "total_time_s": round(total_time_s, 1) if total_time_s else None,
        "model_type": "SED",
        "hparams": {
            "backbone": config.model.backbone,
            "learning_rate": config.training.learning_rate,
            "mixup_alpha": config.training.mixup_alpha,
            "label_smoothing": config.training.label_smoothing,
            "batch_size": config.training.batch_size,
            "epochs": config.training.epochs,
            "focal_gamma": config.training.get("focal_gamma", 2.0),
            "class_weight_mode": config.training.get("class_weight_mode", "none"),
            "n_mels": config.mel.n_mels,
            "hop_length": config.mel.hop_length,
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

    # ── Validation data ───────────────────────────────────────────────────────
    print("\nLoading validation soundscapes as mel spectrograms …")
    val_ds = MelSoundscapeDataset(
        soundscapes_dir=config.data.train_soundscapes_dir,
        labels_csv=config.data.soundscapes_labels_csv,
        species_to_idx=species_to_idx,
        num_classes=num_classes,
        sample_rate=config.audio.sample_rate,
        clip_duration=config.audio.clip_duration,
        n_fft=config.mel.n_fft,
        hop_length=config.mel.hop_length,
        n_mels=config.mel.n_mels,
        fmin=config.mel.fmin,
        fmax=config.mel.fmax,
    )
    val_mels, val_labels = val_ds.get_all_samples()
    print(f"Validation clips: {len(val_mels)}  mel shape: {val_mels[0].shape}")

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
    aug_config = dict(config.augmentation)
    noise_dir = config.data.get("noise_dir", None)

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
        augment_config=aug_config,
        class_weights=class_weights,
        noise_dir=noise_dir,
        n_fft=config.mel.n_fft,
        hop_length=config.mel.hop_length,
        n_mels=config.mel.n_mels,
        fmin=config.mel.fmin,
        fmax=config.mel.fmax,
    )

    train_loader = DataLoader(
        _IterableWrapper(train_ds_obj),
        batch_size=config.training.batch_size,
        num_workers=0,        # IterableDataset: set >0 only with worker_init_fn
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nBuilding SED model …")
    model = SEDModel(
        backbone=config.model.backbone,
        num_classes=num_classes,
        in_chans=1,
        pretrained=config.model.pretrained,
        drop_rate=config.model.dropout,
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

    loss_fn = FocalBCELossTorch(
        gamma=config.training.get("focal_gamma", 2.0),
        alpha=config.training.get("focal_alpha", 0.25),
        label_smoothing=config.training.label_smoothing,
    )
    print(f"Loss: FocalBCE (gamma={config.training.get('focal_gamma',2.0)}, "
          f"alpha={config.training.get('focal_alpha',0.25)})")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_cmap = 0.0
    best_epoch = 0
    val_batch_size = config.training.batch_size * 2
    epoch_history = []
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"  Experiment : {run_name}")
    print(f"  Backbone   : {config.model.backbone}")
    print(f"  Epochs     : {config.training.epochs}")
    print(f"  Batch size : {config.training.batch_size}")
    print(f"{'='*60}\n")

    for epoch in range(1, config.training.epochs + 1):
        t0 = time.time()

        # LR update
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

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, device,
            mixup_alpha=config.training.mixup_alpha,
        )

        # Validate
        val_cmap = evaluate(model, val_mels, val_labels, val_batch_size, device)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{config.training.epochs} | "
            f"loss={train_loss:.4f} | "
            f"val_cmap={val_cmap:.4f} | "
            f"lr={lr:.2e} | "
            f"{elapsed:.1f}s"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_cmap": round(val_cmap, 6),
            "lr": round(float(lr), 8),
            "epoch_time_s": round(elapsed, 1),
        }
        epoch_history.append(epoch_record)
        _save_results(out_dir, run_name, config, epoch_history, best_cmap, best_epoch)

        if wandb_run:
            wandb_run.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/padded_cmap": val_cmap,
                "lr": lr,
            })

        if val_cmap > best_cmap:
            best_cmap = val_cmap
            best_epoch = epoch
            ckpt_path = os.path.join(ckpt_dir, "best_sed")
            model.save(ckpt_path)
            print(f"  ↑ New best cMAP={best_cmap:.4f}")
            if wandb_run:
                wandb_run.log({"val/best_padded_cmap": best_cmap, "epoch": epoch})

    total_time = time.time() - t_start
    _save_results(
        out_dir, run_name, config, epoch_history, best_cmap, best_epoch,
        total_time_s=total_time, finished=True,
    )

    print(f"\nSED training complete.  Best val cMAP: {best_cmap:.4f}  (epoch {best_epoch})")
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
