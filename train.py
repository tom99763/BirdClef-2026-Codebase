"""BirdClef 2026 — Training Script

Usage:
    # Standard run
    python train.py --config configs/default.yaml

    # Quick debug run (limits data, disables WandB)
    python train.py --config configs/debug.yaml

    # Override individual config values
    python train.py --config configs/default.yaml training.learning_rate=5e-4 model.dropout=0.5
"""

import argparse
import json
import os
import time
import numpy as np
import tensorflow as tf

from src.utils.config import load_config, save_config, DotDict
from src.utils.metrics import padded_cmap
from src.data.dataset import build_species_mapping, ClipDataset, SoundscapeDataset, compute_class_weights
from src.data.augment import apply_mixup_batch
from src.model.classifier import PerchClassifier
from src.model.losses import FocalBCELoss


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="BirdClef 2026 Training")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to YAML config file")
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


# ── Optimizer & LR schedule ──────────────────────────────────────────────────

def build_optimizer(config: DotDict) -> tf.keras.optimizers.Optimizer:
    lr = config.training.learning_rate
    if config.training.optimizer == "adamw":
        return tf.keras.optimizers.AdamW(learning_rate=lr,
                                         weight_decay=config.training.weight_decay)
    return tf.keras.optimizers.Adam(learning_rate=lr)


def cosine_lr_with_warmup(
    epoch: int,
    total_epochs: int,
    base_lr: float,
    warmup_epochs: int,
) -> float:
    """Cosine annealing with linear warm-up; epoch is 1-indexed."""
    if epoch <= warmup_epochs:
        return base_lr * epoch / max(warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))


# ── Training step ────────────────────────────────────────────────────────────

def train_epoch(
    model: PerchClassifier,
    dataset: tf.data.Dataset,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_fn,
    mixup_alpha: float,
) -> float:
    total_loss, n_batches = 0.0, 0

    for audio_batch, label_batch in dataset:
        audio_np = audio_batch.numpy()
        label_np = label_batch.numpy()

        if mixup_alpha > 0:
            audio_np, label_np = apply_mixup_batch(audio_np, label_np, mixup_alpha)

        audio_tf = tf.constant(audio_np, dtype=tf.float32)
        label_tf = tf.constant(label_np, dtype=tf.float32)

        with tf.GradientTape() as tape:
            logits = model(audio_tf, training=True)
            loss = loss_fn(label_tf, logits)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        total_loss += float(loss)
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(
    model: PerchClassifier,
    val_clips: np.ndarray,
    val_labels: np.ndarray,
    batch_size: int,
) -> float:
    """Run inference on all validation clips and return padded cMAP."""
    preds = []
    for start in range(0, len(val_clips), batch_size):
        batch = tf.constant(val_clips[start : start + batch_size], dtype=tf.float32)
        logits = model(batch, training=False)
        preds.append(tf.sigmoid(logits).numpy())
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
    """
    Write a self-contained result JSON to <out_dir>/result.json.

    The file is overwritten every epoch so it always reflects the latest state.
    analyze_results.py reads these files to compare experiments.
    """
    result = {
        "run_name": run_name,
        "finished": finished,
        "best_val_cmap": round(best_cmap, 6),
        "best_epoch": best_epoch,
        "total_epochs_run": len(epoch_history),
        "total_time_s": round(total_time_s, 1) if total_time_s else None,
        # Key hyperparameters surfaced at top level for easy comparison
        "hparams": {
            "learning_rate": config.training.learning_rate,
            "mixup_alpha": config.training.mixup_alpha,
            "label_smoothing": config.training.label_smoothing,
            "batch_size": config.training.batch_size,
            "epochs": config.training.epochs,
            "scheduler": config.training.scheduler,
            "warmup_epochs": config.training.warmup_epochs,
            "model_mode": config.model.mode,
            "hidden_dim": config.model.hidden_dim,
            "dropout": config.model.dropout,
            "n_clips_per_file": config.audio.n_clips_per_file,
            "min_rating": config.data.min_rating,
            "use_secondary_labels": config.data.use_secondary_labels,
            "augmentation_enabled": config.augmentation.enabled,
            "noise_level": config.augmentation.noise_level,
        },
        "epoch_history": epoch_history,
    }
    path = os.path.join(out_dir, "result.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = load_config(args.config, _parse_overrides(args.overrides))

    # Reproducibility
    import random
    random.seed(config.experiment.seed)
    np.random.seed(config.experiment.seed)
    tf.random.set_seed(config.experiment.seed)

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

    # ── Validation data (soundscapes → matches test conditions) ───────────────
    print("\nLoading validation soundscapes into memory …")
    val_ds = SoundscapeDataset(
        soundscapes_dir=config.data.train_soundscapes_dir,
        labels_csv=config.data.soundscapes_labels_csv,
        species_to_idx=species_to_idx,
        num_classes=num_classes,
        sample_rate=config.audio.sample_rate,
        clip_duration=config.audio.clip_duration,
    )
    val_clips, val_labels = val_ds.get_all_samples()
    print(f"Validation clips: {len(val_clips)}")

    # ── Training data ─────────────────────────────────────────────────────────
    print("\nBuilding training dataset …")
    aug_config = dict(config.augmentation)

    # Class-frequency weighting (BirdCLEF 2025 2nd place sqrt balancing)
    class_weight_mode = config.training.get("class_weight_mode", "none")
    class_weights = None
    if class_weight_mode != "none":
        print(f"Computing class weights (mode={class_weight_mode}) …")
        class_weights = compute_class_weights(
            config.data.train_csv, species_to_idx, num_classes, mode=class_weight_mode
        )
        print(f"  Weight range: [{class_weights.min():.3f}, {class_weights.max():.3f}]")

    noise_dir = config.data.get("noise_dir", None)

    train_ds_obj = ClipDataset(
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
        noise_dir=noise_dir,
        class_weights=class_weights,
    )

    clip_length = config.audio.clip_duration * config.audio.sample_rate
    tf_train_ds = (
        tf.data.Dataset.from_generator(
            train_ds_obj.generate_samples,
            output_signature=(
                tf.TensorSpec(shape=(clip_length,), dtype=tf.float32),
                tf.TensorSpec(shape=(num_classes,), dtype=tf.float32),
            ),
        )
        .shuffle(buffer_size=2000)
        .batch(config.training.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nBuilding model …")
    model = PerchClassifier(
        perch_dir=config.model.perch_dir,
        num_classes=num_classes,
        mode=config.model.mode,
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
    )

    # ── Optimizer & loss ──────────────────────────────────────────────────────
    optimizer = build_optimizer(config)
    loss_name = config.training.get("loss", "bce")
    if loss_name == "focal":
        loss_fn = FocalBCELoss(
            gamma=config.training.get("focal_gamma", 2.0),
            alpha=config.training.get("focal_alpha", 0.25),
            from_logits=True,
            label_smoothing=config.training.label_smoothing,
        )
        print(f"Loss: FocalBCE (gamma={config.training.get('focal_gamma',2.0)}, "
              f"alpha={config.training.get('focal_alpha',0.25)})")
    else:
        loss_fn = tf.keras.losses.BinaryCrossentropy(
            from_logits=True,
            label_smoothing=config.training.label_smoothing,
        )
        print("Loss: BinaryCrossentropy")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_cmap = 0.0
    best_epoch = 0
    val_batch_size = config.training.batch_size * 2
    epoch_history = []   # list of dicts, one per epoch
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"  Experiment : {run_name}")
    print(f"  Mode       : {config.model.mode}")
    print(f"  Epochs     : {config.training.epochs}")
    print(f"  Batch size : {config.training.batch_size}")
    print(f"{'='*60}\n")

    for epoch in range(1, config.training.epochs + 1):
        t0 = time.time()

        # Update learning rate
        lr = cosine_lr_with_warmup(
            epoch,
            config.training.epochs,
            config.training.learning_rate,
            config.training.warmup_epochs,
        ) if config.training.scheduler == "cosine" else config.training.learning_rate
        optimizer.learning_rate.assign(lr)

        # Train
        train_loss = train_epoch(
            model, tf_train_ds, optimizer, loss_fn,
            mixup_alpha=config.training.mixup_alpha,
        )

        # Validate
        val_cmap = evaluate(model, val_clips, val_labels, val_batch_size)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{config.training.epochs} | "
            f"loss={train_loss:.4f} | "
            f"val_cmap={val_cmap:.4f} | "
            f"lr={lr:.2e} | "
            f"{elapsed:.1f}s"
        )

        # Per-epoch record (written to disk every epoch so crashes don't lose data)
        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_cmap": round(val_cmap, 6),
            "lr": round(float(lr), 8),
            "epoch_time_s": round(elapsed, 1),
        }
        epoch_history.append(epoch_record)
        _save_results(out_dir, run_name, config, epoch_history, best_cmap, best_epoch)

        # WandB logging
        if wandb_run:
            wandb_run.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/padded_cmap": val_cmap,
                "lr": lr,
            })

        # Checkpoint best model
        if val_cmap > best_cmap:
            best_cmap = val_cmap
            best_epoch = epoch
            ckpt_path = os.path.join(ckpt_dir, "best_head")
            model.save_head(ckpt_path)
            print(f"  ↑ New best cMAP={best_cmap:.4f}")
            if wandb_run:
                wandb_run.log({"val/best_padded_cmap": best_cmap, "epoch": epoch})

    total_time = time.time() - t_start
    _save_results(out_dir, run_name, config, epoch_history, best_cmap, best_epoch,
                  total_time_s=total_time, finished=True)

    print(f"\nTraining complete.  Best val cMAP: {best_cmap:.4f}  (epoch {best_epoch})")
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
