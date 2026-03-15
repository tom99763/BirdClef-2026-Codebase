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
from tqdm import tqdm

from src.utils.config import load_config, save_config, DotDict
from src.utils.metrics import competition_roc_auc
from src.data.dataset import build_species_mapping, ClipDataset, SoundscapeDataset, CachedEmbeddingDataset, PseudoSoundscapeDataset, compute_class_weights, build_taxon_label_fn, TAXON_CLASSES
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

    pbar = tqdm(dataset, desc="  train", leave=False, unit="batch")
    for audio_batch, label_batch in pbar:
        if mixup_alpha > 0:
            audio_np, label_np = apply_mixup_batch(
                audio_batch.numpy(), label_batch.numpy(), mixup_alpha
            )
            audio_tf = tf.constant(audio_np, dtype=tf.float32)
            label_tf = tf.constant(label_np, dtype=tf.float32)
        else:
            audio_tf = audio_batch
            label_tf = label_batch

        with tf.GradientTape() as tape:
            logits = model(audio_tf, training=True)
            loss = loss_fn(label_tf, logits)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        total_loss += float(loss)
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

    return total_loss / max(n_batches, 1)


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(
    model: PerchClassifier,
    val_clips: np.ndarray,
    val_labels: np.ndarray,
    batch_size: int,
) -> float:
    """Run inference on all validation clips and return competition ROC-AUC."""
    preds = []
    for start in range(0, len(val_clips), batch_size):
        batch = tf.constant(val_clips[start : start + batch_size], dtype=tf.float32)
        logits = model(batch, training=False)
        preds.append(tf.sigmoid(logits).numpy())
    return competition_roc_auc(val_labels, np.concatenate(preds, axis=0))


# ── Cached-embedding fast path (embedding_head + pre-extracted cache) ─────────

@tf.function
def _train_step_cached(emb_batch, label_batch, head, optimizer, loss_fn, mixup_alpha):
    """Single gradient-update step on pre-computed embeddings (graph mode)."""
    if mixup_alpha > 0.0:
        batch_size = tf.shape(emb_batch)[0]
        indices = tf.random.shuffle(tf.range(batch_size))
        lam = tf.random.uniform([batch_size, 1], dtype=tf.float32)
        emb_batch   = lam * emb_batch   + (1.0 - lam) * tf.gather(emb_batch,   indices)
        label_batch = lam * label_batch + (1.0 - lam) * tf.gather(label_batch, indices)
    with tf.GradientTape() as tape:
        logits = head(emb_batch, training=True)
        loss = loss_fn(label_batch, logits)
    grads = tape.gradient(loss, head.trainable_variables)
    optimizer.apply_gradients(zip(grads, head.trainable_variables))
    return loss


def train_epoch_cached(
    model: PerchClassifier,
    dataset: tf.data.Dataset,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_fn,
    mixup_alpha: float,
) -> float:
    total_loss, n_batches = 0.0, 0
    pbar = tqdm(dataset, desc="  train", leave=False, unit="batch")
    for emb_batch, label_batch in pbar:
        loss = _train_step_cached(
            emb_batch, label_batch, model.head, optimizer, loss_fn, mixup_alpha
        )
        total_loss += float(loss)
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")
    return total_loss / max(n_batches, 1)


@tf.function
def _train_step_cached_multitask(
    emb_batch, species_label_batch, taxon_label_batch,
    head, optimizer, loss_fn, mixup_alpha, taxon_aux_weight
):
    """Cached training step with taxonomy auxiliary loss."""
    if mixup_alpha > 0.0:
        batch_size = tf.shape(emb_batch)[0]
        indices = tf.random.shuffle(tf.range(batch_size))
        lam = tf.random.uniform([batch_size, 1], dtype=tf.float32)
        emb_batch         = lam * emb_batch         + (1.0 - lam) * tf.gather(emb_batch, indices)
        species_label_batch = lam * species_label_batch + (1.0 - lam) * tf.gather(species_label_batch, indices)
        # Don't mix taxon labels — use hard label of primary sample
    with tf.GradientTape() as tape:
        species_logits, taxon_logits = head(emb_batch, training=True)
        species_loss = loss_fn(species_label_batch, species_logits)
        taxon_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=taxon_label_batch, logits=taxon_logits
            )
        )
        total_loss = species_loss + taxon_aux_weight * taxon_loss
    grads = tape.gradient(total_loss, head.trainable_variables)
    optimizer.apply_gradients(zip(grads, head.trainable_variables))
    return total_loss, species_loss, taxon_loss


def train_epoch_cached_multitask(
    model: PerchClassifier,
    dataset: tf.data.Dataset,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_fn,
    mixup_alpha: float,
    taxon_aux_weight: float,
) -> float:
    total_loss, n_batches = 0.0, 0
    pbar = tqdm(dataset, desc="  train", leave=False, unit="batch")
    for emb_batch, species_batch, taxon_batch in pbar:
        loss, sp_loss, tx_loss = _train_step_cached_multitask(
            emb_batch, species_batch, taxon_batch,
            model.head, optimizer, loss_fn, mixup_alpha,
            tf.constant(taxon_aux_weight, dtype=tf.float32)
        )
        total_loss += float(loss)
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}",
                         sp=f"{float(sp_loss):.4f}",
                         tx=f"{float(tx_loss):.4f}")
    return total_loss / max(n_batches, 1)


def evaluate_cached(
    model: PerchClassifier,
    val_embs: np.ndarray,
    val_labels: np.ndarray,
    batch_size: int,
) -> float:
    """Head-only inference on pre-computed soundscape embeddings."""
    preds = []
    for start in range(0, len(val_embs), batch_size):
        batch = tf.constant(val_embs[start : start + batch_size], dtype=tf.float32)
        out = model.head(batch, training=False)
        # out may be (species_logits, taxon_logits) when taxon_head is active
        logits = out[0] if isinstance(out, tuple) else out
        preds.append(tf.sigmoid(logits).numpy())
    return competition_roc_auc(val_labels, np.concatenate(preds, axis=0))


# ── Result persistence ────────────────────────────────────────────────────────

def _save_results(
    out_dir: str,
    run_name: str,
    config: DotDict,
    epoch_history: list,
    best_roc_auc: float,
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
        "best_val_roc_auc": round(best_roc_auc, 6),
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
            "noise_level": config.augmentation.get("noise_level", None),
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

    # ── Taxon multitask flag ──────────────────────────────────────────────────
    taxon_aux_weight = float(config.training.get("taxon_aux_weight", 0.0))

    # ── Class weights (needed regardless of cache) ────────────────────────────
    class_weight_mode = config.training.get("class_weight_mode", "none")
    class_weights = None
    if class_weight_mode == "taxon_upweight":
        from src.data.dataset import compute_taxon_weights
        nonbird_boost = float(config.training.get("taxon_nonbird_boost", 3.0))
        print(f"Computing taxon-aware class weights (nonbird_boost={nonbird_boost}) …")
        class_weights = compute_taxon_weights(
            config.data.train_csv, config.data.taxonomy_csv,
            species_to_idx, num_classes, nonbird_boost=nonbird_boost
        )
        print(f"  Weight range: [{class_weights.min():.3f}, {class_weights.max():.3f}]")
    elif class_weight_mode != "none":
        print(f"Computing class weights (mode={class_weight_mode}) …")
        class_weights = compute_class_weights(
            config.data.train_csv, species_to_idx, num_classes, mode=class_weight_mode
        )
        print(f"  Weight range: [{class_weights.min():.3f}, {class_weights.max():.3f}]")

    # ── Detect pre-computed embedding cache ───────────────────────────────────
    manifest_path = os.path.join(config.cache.cache_dir, "manifest.csv")
    use_cache = (
        config.cache.enabled
        and config.model.mode in ("embedding_head", "label_head")
        and os.path.isfile(manifest_path)
    )

    use_soundscapes_in_train = config.training.get("use_soundscapes_in_train", False)
    use_taxon_multitask = (taxon_aux_weight > 0.0 and use_cache)

    # Fixed soundscape train/val split (prevents data leakage when soundscapes
    # are used in training).  Falls back to all-soundscapes if file not found.
    sc_split_csv = config.data.get("soundscapes_split_csv",
                                   "birdclef-2026/soundscapes_split.csv")
    if not os.path.isfile(sc_split_csv):
        sc_split_csv = None
    _sc_train_split = "train" if (use_soundscapes_in_train and sc_split_csv) else None
    _sc_val_split   = "val"   if sc_split_csv else None

    if use_cache:
        print(f"\nUsing cached embeddings: {manifest_path}")

        # Build taxon_label_fn if needed
        taxon_label_fn = None
        if use_taxon_multitask:
            taxon_label_fn = build_taxon_label_fn(config.data.taxonomy_csv, species_to_idx)

        train_cache_ds = CachedEmbeddingDataset(
            manifest_csv=manifest_path,
            species_to_idx=species_to_idx,
            num_classes=num_classes,
            split="train",
            class_weights=class_weights,
            taxon_label_fn=taxon_label_fn,
        )
        emb_dim = train_cache_ds.embedding_dim

        if use_soundscapes_in_train:
            sc_oversample = int(config.training.get("soundscape_oversample", 1))
            sc_train_ds = CachedEmbeddingDataset(
                manifest_csv=manifest_path,
                species_to_idx=species_to_idx,
                num_classes=num_classes,
                split="soundscape",
                taxon_label_fn=taxon_label_fn,
                soundscape_split_csv=sc_split_csv,
                soundscape_split=_sc_train_split,
            )
            def _combined_gen():
                yield from train_cache_ds.generate_samples()
                for _ in range(sc_oversample):
                    yield from sc_train_ds.generate_samples()
            train_gen = _combined_gen
            print(f"  + soundscape segments added to training set (oversample={sc_oversample}x)")
        else:
            train_gen = train_cache_ds.generate_samples

        # ── Pseudo-label augmentation (cached embeddings path) ─────────────────
        use_pseudo = config.training.get("use_pseudo_labels", False)
        pseudo_csv = config.data.get("pseudo_labels_csv", None)
        pseudo_manifest = config.data.get("pseudo_manifest_csv", None)
        if use_pseudo and pseudo_manifest and os.path.isfile(pseudo_manifest):
            # Fast path: pre-extracted embeddings for pseudo-labeled segments
            pseudo_cache_ds = CachedEmbeddingDataset(
                manifest_csv=pseudo_manifest,
                species_to_idx=species_to_idx,
                num_classes=num_classes,
                split="pseudo",
            )
            _prev_gen = train_gen
            def _with_pseudo_gen():
                yield from _prev_gen()
                yield from pseudo_cache_ds.generate_samples()
            train_gen = _with_pseudo_gen
            print(f"  + pseudo-labeled embeddings added ({pseudo_manifest})")

        if use_taxon_multitask:
            tf_train_ds = (
                tf.data.Dataset.from_generator(
                    train_gen,
                    output_signature=(
                        tf.TensorSpec(shape=(emb_dim,), dtype=tf.float32),
                        tf.TensorSpec(shape=(num_classes,), dtype=tf.float32),
                        tf.TensorSpec(shape=(), dtype=tf.int32),
                    ),
                )
                .shuffle(buffer_size=4000)
                .batch(config.training.batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )
        else:
            tf_train_ds = (
                tf.data.Dataset.from_generator(
                    train_gen,
                    output_signature=(
                        tf.TensorSpec(shape=(emb_dim,), dtype=tf.float32),
                        tf.TensorSpec(shape=(num_classes,), dtype=tf.float32),
                    ),
                )
                .shuffle(buffer_size=4000)
                .batch(config.training.batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )

        print("Loading cached soundscape validation embeddings …")
        val_cache_ds = CachedEmbeddingDataset(
            manifest_csv=manifest_path,
            species_to_idx=species_to_idx,
            num_classes=num_classes,
            split="soundscape",
            soundscape_split_csv=sc_split_csv,
            soundscape_split=_sc_val_split,
        )
        val_clips, val_labels = val_cache_ds.get_all_samples()
        print(f"Validation embeddings: {len(val_clips)}")

    else:
        if config.cache.enabled and config.model.mode == "embedding_head":
            print(
                f"\nTip: run 'python extract_embeddings.py' to pre-cache Perch embeddings "
                f"and speed up training significantly."
            )

        # ── Validation data (soundscapes → matches test conditions) ───────────
        print("\nLoading validation soundscapes into memory …")
        val_ds = SoundscapeDataset(
            soundscapes_dir=config.data.train_soundscapes_dir,
            labels_csv=config.data.soundscapes_labels_csv,
            species_to_idx=species_to_idx,
            num_classes=num_classes,
            split_csv=sc_split_csv,
            split=_sc_val_split,
            sample_rate=config.audio.sample_rate,
            clip_duration=config.audio.clip_duration,
        )
        val_clips, val_labels = val_ds.get_all_samples()
        print(f"Validation clips: {len(val_clips)}")

        # ── Training data ─────────────────────────────────────────────────────
        print("\nBuilding training dataset …")
        aug_config = dict(config.augmentation)
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

        if use_soundscapes_in_train:
            sc_oversample = int(config.training.get("soundscape_oversample", 1))
            sc_ds_obj = SoundscapeDataset(
                soundscapes_dir=config.data.train_soundscapes_dir,
                labels_csv=config.data.soundscapes_labels_csv,
                species_to_idx=species_to_idx,
                num_classes=num_classes,
                sample_rate=config.audio.sample_rate,
                clip_duration=config.audio.clip_duration,
                split_csv=sc_split_csv,
                split=_sc_train_split,
            )
            def _combined_audio_gen():
                yield from train_ds_obj.generate_samples()
                for _ in range(sc_oversample):
                    yield from sc_ds_obj.generate_samples()
            train_audio_gen = _combined_audio_gen
            print(f"  + soundscape segments added to training set (raw audio, oversample={sc_oversample}x)")
        else:
            train_audio_gen = train_ds_obj.generate_samples

        # ── Pseudo-label augmentation ──────────────────────────────────────────
        use_pseudo = config.training.get("use_pseudo_labels", False)
        pseudo_csv = config.data.get("pseudo_labels_csv", None)
        if use_pseudo and pseudo_csv and os.path.isfile(pseudo_csv):
            pseudo_ds_obj = PseudoSoundscapeDataset(
                pseudo_csv=pseudo_csv,
                soundscapes_dir=config.data.train_soundscapes_dir,
                species_to_idx=species_to_idx,
                target_species=target_species,
                num_classes=num_classes,
                sample_rate=config.audio.sample_rate,
                clip_duration=config.audio.clip_duration,
                use_soft_labels=True,
            )
            _prev_gen = train_audio_gen
            def _with_pseudo_gen():
                yield from _prev_gen()
                yield from pseudo_ds_obj.generate_samples()
            train_audio_gen = _with_pseudo_gen
            print(f"  + pseudo-labeled soundscape segments added to training set")
        elif use_pseudo and pseudo_csv:
            print(f"  [WARNING] pseudo_labels_csv not found: {pseudo_csv}")

        tf_train_ds = (
            tf.data.Dataset.from_generator(
                train_audio_gen,
                output_signature=(
                    tf.TensorSpec(shape=(clip_length,), dtype=tf.float32),
                    tf.TensorSpec(shape=(num_classes,), dtype=tf.float32),
                ),
            )
            .shuffle(buffer_size=2000)
            .batch(config.training.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        emb_dim = None  # will load Perch to determine

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nBuilding model …")
    model = PerchClassifier(
        perch_dir=config.model.perch_dir,
        num_classes=num_classes,
        mode=config.model.mode,
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
        embedding_dim=emb_dim,
        num_taxon_classes=len(TAXON_CLASSES) if use_taxon_multitask else 0,
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
    best_roc_auc = 0.0
    best_epoch = 0
    val_batch_size = config.training.batch_size * 2
    epoch_history = []   # list of dicts, one per epoch
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"  Experiment : {run_name}")
    print(f"  Mode       : {config.model.mode}  {'[CACHED]' if use_cache else '[raw audio]'}")
    print(f"  Epochs     : {config.training.epochs}")
    print(f"  Batch size : {config.training.batch_size}")
    print(f"{'='*60}\n")

    epoch_pbar = tqdm(range(1, config.training.epochs + 1), desc="Epochs", unit="epoch")
    for epoch in epoch_pbar:
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
        if use_taxon_multitask:
            train_loss = train_epoch_cached_multitask(
                model, tf_train_ds, optimizer, loss_fn,
                mixup_alpha=config.training.mixup_alpha,
                taxon_aux_weight=taxon_aux_weight,
            )
        elif use_cache:
            train_loss = train_epoch_cached(
                model, tf_train_ds, optimizer, loss_fn,
                mixup_alpha=config.training.mixup_alpha,
            )
        else:
            train_loss = train_epoch(
                model, tf_train_ds, optimizer, loss_fn,
                mixup_alpha=config.training.mixup_alpha,
            )

        # Validate
        if use_cache:
            val_roc_auc = evaluate_cached(model, val_clips, val_labels, val_batch_size)
        else:
            val_roc_auc = evaluate(model, val_clips, val_labels, val_batch_size)

        elapsed = time.time() - t0
        epoch_pbar.set_postfix(
            loss=f"{train_loss:.4f}",
            cmap=f"{val_roc_auc:.4f}",
            best=f"{best_roc_auc:.4f}",
            lr=f"{lr:.1e}",
            t=f"{elapsed:.0f}s",
        )
        tqdm.write(
            f"Epoch {epoch:3d}/{config.training.epochs} | "
            f"loss={train_loss:.4f} | "
            f"val_roc_auc={val_roc_auc:.4f} | "
            f"lr={lr:.2e} | "
            f"{elapsed:.1f}s"
        )

        # Per-epoch record (written to disk every epoch so crashes don't lose data)
        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_roc_auc": round(val_roc_auc, 6),
            "lr": round(float(lr), 8),
            "epoch_time_s": round(elapsed, 1),
        }
        epoch_history.append(epoch_record)
        _save_results(out_dir, run_name, config, epoch_history, best_roc_auc, best_epoch)

        # WandB logging
        if wandb_run:
            wandb_run.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/roc_auc": val_roc_auc,
                "lr": lr,
            })

        # Checkpoint best model
        if val_roc_auc > best_roc_auc:
            best_roc_auc = val_roc_auc
            best_epoch = epoch
            ckpt_path = os.path.join(ckpt_dir, "best_head")
            model.save_head(ckpt_path)
            tqdm.write(f"  ↑ New best cMAP={best_roc_auc:.4f}")
            if wandb_run:
                wandb_run.log({"val/best_roc_auc": best_roc_auc, "epoch": epoch})

    total_time = time.time() - t_start
    _save_results(out_dir, run_name, config, epoch_history, best_roc_auc, best_epoch,
                  total_time_s=total_time, finished=True)

    print(f"\nTraining complete.  Best val cMAP: {best_roc_auc:.4f}  (epoch {best_epoch})")
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
