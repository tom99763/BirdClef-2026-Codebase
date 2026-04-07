"""BirdClef 2026 — Perch-as-Teacher Knowledge Distillation

Uses Perch model predictions on soundscape clips as soft labels to train the
SED student model. This transfers Perch's calibrated knowledge of co-occurring
Pantanal species into the EfficientNet-B0 architecture.

Key idea (from BirdCLEF community):
  "throw in a bunch of wave files (unlabeled soundscape files from this and
   previous competitions), extract their embeddings and make a database,
   distill to your favourite pytorch models and enjoy!"

How it works:
  1. Teacher: trained Perch head (nohuman-label-soundscape-train, holdout 0.9550)
     produces 234-dim calibrated probabilities for each 5-second soundscape clip
  2. Student: EfficientNet-B0 SED (sedp-v2-fusion config: PCEN+ASL+dual+LLRD)
  3. Distillation: soundscape clips use Perch soft predictions as clip targets
     instead of binary hard labels — richer, calibrated teacher signal
  4. Frame loss: uses hard labels (fine-grained temporal supervision unchanged)

Data sources:
  - train_audio: 35,549 labeled recordings (hard labels, unchanged)
  - soft soundscapes: up to 256K unlabeled soundscape clips with Perch soft labels
    (distill_b0_v1: 1176 clips from existing round5_pseudo.csv)
    (distill_b0_v2_full: all 10,658 soundscape files via extract_perch_teacher_all_ss.py)

Loss:
  clip_loss_w * ASL(clip_pred, teacher_soft_label)  [soft targets from Perch]
  + frame_loss_w * BCE(frame_logit, hard_label)       [hard labels for frames]

Usage:
    python train_distill.py --config configs/distill_b0_v1.yaml
    python train_distill.py --config configs/distill_b0_v2_full.yaml --gpu 1
"""

import argparse
import heapq
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, IterableDataset

from src.utils.config import load_config, save_config, DotDict
from src.utils.metrics import competition_roc_auc, padded_cmap
from src.data.dataset import build_species_mapping, compute_class_weights
from src.data.mel_dataset import MelClipDataset, MelSoundscapeDataset
from src.model.sed_model import SEDModel, FocalBCELossTorch
from src.model.pcen import AudioToMelPCEN
from src.utils.audio import load_audio, parse_time_str

from train_sed import (
    _IterableWrapper,
    build_gpu_mel_transform,
    apply_gpu_mel,
    cosine_lr_with_warmup,
    _parse_overrides,
    _save_results,
    compute_pos_weights,
    AsymmetricLoss,
    BCEPosWeightLoss,
    apply_cutmix,
)
from train_sedp import (
    MaskedBCELoss,
    build_llrd_optimizer,
    apply_circular_shift,
    mix_background_snr,
)

CLIP_SAMPLES_DEFAULT = 32_000 * 5


# ── Soft Soundscape Dataset ───────────────────────────────────────────────────

class SoftSoundscapeDataset(IterableDataset):
    """Loads soundscape clips paired with Perch soft teacher labels.

    Reads from a teacher_csv (format: row_id, species_1, ..., species_234)
    where row_id = {filename_no_ext}_{end_second}.
    Loads the corresponding audio clip from soundscapes_dir.
    Returns (raw_audio, soft_label) pairs.
    """

    def __init__(
        self,
        teacher_csv: str,
        soundscapes_dir: str,
        species_list: list,
        num_classes: int,
        sample_rate: int = 32_000,
        clip_duration: int = 5,
        max_rows: int = None,
        shuffle: bool = True,
    ):
        self.soundscapes_dir = soundscapes_dir
        self.num_classes     = num_classes
        self.clip_samples    = sample_rate * clip_duration
        self.sample_rate     = sample_rate
        self.clip_duration   = clip_duration
        self.shuffle         = shuffle

        print(f"\nLoading teacher predictions from {teacher_csv} …")
        df = pd.read_csv(teacher_csv)
        if max_rows:
            df = df.head(max_rows)

        sp_str = [str(sp) for sp in species_list]

        self.samples = []
        for _, row in df.iterrows():
            row_id = str(row["row_id"])
            # Parse: {basename_no_ext}_{end_second}
            try:
                parts   = row_id.rsplit("_", 1)
                fname   = parts[0] + ".ogg"
                end_sec = int(parts[1])
            except Exception:
                continue
            start_sec   = max(0, end_sec - clip_duration)
            soft_label  = np.array(
                [float(row.get(sp, 0.0)) for sp in sp_str], dtype=np.float32
            )
            self.samples.append((fname, start_sec, soft_label))

        print(f"  SoftSoundscapeDataset: {len(self.samples)} clips from teacher CSV")
        if len(self.samples) == 0:
            print("  WARN: No valid samples found. Check teacher_csv format.")

        # Pre-load all clips into RAM to eliminate per-epoch disk I/O.
        # 1176 clips × 5s × 32000 = ~740 MB — acceptable.
        print(f"  Pre-loading {len(self.samples)} distill clips into RAM …", flush=True)
        self._cache = []  # list of (clip_float32, soft_label)
        skipped = 0
        for fname, start_sec, soft_label in self.samples:
            filepath = os.path.join(soundscapes_dir, fname)
            audio = load_audio(filepath, sample_rate)
            if audio is None:
                skipped += 1
                continue
            start_sample = int(start_sec * sample_rate)
            clip = audio[start_sample: start_sample + self.clip_samples]
            if len(clip) < self.clip_samples:
                clip = np.pad(clip, (0, self.clip_samples - len(clip)))
            self._cache.append((clip.astype(np.float32), soft_label))
        print(f"  Cache ready: {len(self._cache)} clips loaded ({skipped} skipped)", flush=True)

    def __iter__(self):
        idxs = list(range(len(self._cache)))
        if self.shuffle:
            random.shuffle(idxs)
        for i in idxs:
            yield self._cache[i]


# ── Training Epoch with Distillation ─────────────────────────────────────────

def train_epoch_distill(
    model, train_loader, distill_loader, optimizer, scaler, device,
    loss_fn, loss_mode, clip_loss_w, frame_loss_w,
    pcen_tf=None, mel_tf=None, use_amp=False,
    mixup_alpha=0.0, circ_shift_prob=0.0, bg_pool=None,
    distill_weight=0.5, ss_oversample=1,
):
    """Training epoch with hard-label train_audio + soft-label soundscape distillation."""
    model.train()

    distill_iter  = iter(distill_loader) if distill_loader is not None else None
    total_loss    = 0.0
    n_batches     = 0

    for audio_batch, label_batch in train_loader:
        audio_batch = audio_batch.to(device)
        label_batch = label_batch.to(device)

        # ── Optional circular shift aug ───────────────────────────────────────
        if circ_shift_prob > 0:
            audio_batch = apply_circular_shift(audio_batch, prob=circ_shift_prob)

        # ── SNR background mix ────────────────────────────────────────────────
        if bg_pool is not None:
            audio_batch = mix_background_snr(audio_batch, bg_pool, device)

        # ── GPU mel transform ─────────────────────────────────────────────────
        if pcen_tf is not None:
            mel_batch = pcen_tf(audio_batch)
        elif mel_tf is not None:
            mel_batch = apply_gpu_mel(mel_tf, audio_batch)
        else:
            mel_batch = audio_batch

        # ── Mixup on train_audio clips ────────────────────────────────────────
        if mixup_alpha > 0 and random.random() < 0.5:
            lam  = np.random.beta(mixup_alpha, mixup_alpha)
            idx  = torch.randperm(mel_batch.size(0), device=device)
            mel_batch   = lam * mel_batch   + (1 - lam) * mel_batch[idx]
            label_batch = lam * label_batch + (1 - lam) * label_batch[idx]

        frame_labels = label_batch.unsqueeze(1).expand(-1, model(mel_batch)[1].shape[1], -1) \
                       if hasattr(model, '_last_frame_shape') else None

        with autocast(enabled=use_amp):
            out = model(mel_batch)
            clip_pred   = out[0] if isinstance(out, tuple) else out
            frame_logit = out[1] if isinstance(out, tuple) else None

            # Compute hard-label loss for train_audio
            if loss_mode == "masked_bce":
                clip_loss = loss_fn(clip_pred, label_batch)
            elif loss_mode == "asl":
                clip_loss = loss_fn(clip_pred, label_batch)
            elif loss_mode in ("bce", "bce_pos_weight"):
                clip_loss = F.binary_cross_entropy(
                    clip_pred.clamp(1e-7, 1 - 1e-7), label_batch.clamp(0, 1)
                )
            else:
                clip_loss = loss_fn(clip_pred, label_batch)

            if frame_loss_w > 0 and frame_logit is not None:
                fl = label_batch.unsqueeze(1).expand(-1, frame_logit.shape[1], -1)
                frame_loss = F.binary_cross_entropy_with_logits(
                    frame_logit, fl.clamp(0, 1)
                )
                loss = clip_loss_w * clip_loss + frame_loss_w * frame_loss
            else:
                loss = clip_loss_w * clip_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item()
        n_batches  += 1

        # ── Distillation step: soft soundscape clips ──────────────────────────
        if distill_iter is not None:
            for _ in range(ss_oversample):
                try:
                    audio_ss, soft_label = next(distill_iter)
                except StopIteration:
                    distill_iter = iter(distill_loader)
                    try:
                        audio_ss, soft_label = next(distill_iter)
                    except StopIteration:
                        break

                audio_ss   = audio_ss.to(device)
                soft_label = soft_label.to(device)

                if circ_shift_prob > 0:
                    audio_ss = apply_circular_shift(audio_ss, prob=circ_shift_prob)

                if pcen_tf is not None:
                    mel_ss = pcen_tf(audio_ss)
                elif mel_tf is not None:
                    mel_ss = apply_gpu_mel(mel_tf, audio_ss)
                else:
                    mel_ss = audio_ss

                with autocast(enabled=use_amp):
                    out_ss    = model(mel_ss)
                    clip_ss   = out_ss[0] if isinstance(out_ss, tuple) else out_ss
                    # frame loss for soundscapes is skipped (no frame-level labels)

                    # Use soft Perch labels as targets (ASL or BCE on soft probs)
                    if loss_mode == "asl":
                        distill_loss = loss_fn(clip_ss, soft_label)
                    else:
                        distill_loss = F.binary_cross_entropy(
                            clip_ss.clamp(1e-7, 1 - 1e-7), soft_label.clamp(0, 1)
                        )
                    distill_loss = distill_weight * distill_loss

                scaler.scale(distill_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                total_loss += distill_loss.item()
                n_batches  += 1

    return total_loss / max(n_batches, 1)


# ── Reuse evaluate from train_sedp ───────────────────────────────────────────

def evaluate(model, val_data, val_labels, device, mel_tf=None, pcen_tf=None,
             batch_size=64, use_gpu_mel=True, use_amp=False):
    from train_sedp import evaluate as _eval_sedp
    return _eval_sedp(
        model, val_data, val_labels, device,
        mel_tf=mel_tf, pcen_tf=pcen_tf,
        batch_size=batch_size, use_gpu_mel=use_gpu_mel, use_amp=use_amp,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--gpu",    default=None)
    p.add_argument("overrides", nargs="*",
                   help="key=value overrides (e.g. training.epochs=40)")
    return p.parse_args()


def main():
    import tempfile
    args = parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = load_config(args.config)
    for k, v in _parse_overrides(args.overrides).items():
        parts = k.split(".")
        d = config
        for p in parts[:-1]:
            d = getattr(d, p)
        setattr(d, parts[-1], v)

    random.seed(config.experiment.seed)
    np.random.seed(config.experiment.seed)
    torch.manual_seed(config.experiment.seed)

    target_species, species_to_idx = build_species_mapping(config.data.sample_submission_csv)
    num_classes = len(species_to_idx)
    run_name    = config.experiment.name

    print(f"\n{'='*60}")
    print(f"  Experiment : {run_name}  [Perch Distillation]")
    print(f"  Backbone   : {config.model.backbone}")
    print(f"  PCEN       : {config.model.get('use_pcen', False)}")
    print(f"  Epochs     : {config.training.epochs}")
    print(f"  Distill W  : {config.training.get('distill_weight', 0.5)}")
    print(f"{'='*60}\n")

    # ── W&B ──────────────────────────────────────────────────────────────────
    use_wandb = config.wandb.get("enabled", False)
    if use_wandb:
        try:
            import wandb
            wandb.init(project=config.wandb.get("project"), name=run_name,
                       config=dict(config), tags=config.wandb.get("tags", []))
        except Exception as e:
            print(f"WandB init failed: {e}")
            use_wandb = False

    # ── Model ─────────────────────────────────────────────────────────────────
    use_pcen    = config.model.get("use_pcen", False)
    use_gpu_mel = config.model.get("use_gpu_mel", True)
    use_amp     = config.training.get("use_amp", False)
    scaler      = GradScaler(enabled=use_amp)

    model = SEDModel(
        backbone    = config.model.backbone,
        num_classes = num_classes,
        in_chans    = config.model.get("in_chans", 3),
        pretrained  = config.model.get("pretrained", True),
        drop_rate   = config.model.get("dropout", 0.1),
        use_gem     = config.model.get("use_gem", True),
        gem_p_init  = config.model.get("gem_p_init", 3.0),
        n_mels      = config.mel.n_mels,
    ).to(device)

    mel_kw = dict(
        n_fft=config.mel.n_fft, hop_length=config.mel.hop_length,
        n_mels=config.mel.n_mels, fmin=config.mel.fmin, fmax=config.mel.fmax,
    )

    pcen_tf, mel_tf = None, None
    if use_pcen:
        pcen_tf = AudioToMelPCEN(
            sample_rate=config.audio.sample_rate,
            n_fft=config.mel.n_fft, hop_length=config.mel.hop_length,
            n_mels=config.mel.n_mels, fmin=config.mel.fmin, fmax=config.mel.fmax,
            trainable_pcen=config.model.get("trainable_pcen", True),
        ).to(device)
    elif use_gpu_mel:
        mel_tf = build_gpu_mel_transform(
            sample_rate=config.audio.sample_rate, **mel_kw
        ).to(device)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Loss ─────────────────────────────────────────────────────────────────
    loss_mode = config.training.get("loss", "bce")
    if loss_mode == "masked_bce":
        loss_fn = MaskedBCELoss(secondary_threshold=0.4)
    elif loss_mode == "asl":
        loss_fn = AsymmetricLoss(
            gamma_neg=config.training.get("asl_gamma_neg", 4.0),
            gamma_pos=config.training.get("asl_gamma_pos", 0.0),
            clip=config.training.get("asl_clip", 0.05),
        )
    elif loss_mode == "focal":
        loss_fn = FocalBCELossTorch(gamma=config.training.get("focal_gamma", 2.0))
    else:
        loss_fn = nn.BCELoss()
    loss_fn = loss_fn.to(device)

    clip_loss_w    = config.training.get("clip_loss_weight", 1.0)
    frame_loss_w   = config.training.get("frame_loss_weight", 0.0)
    distill_weight = config.training.get("distill_weight", 0.5)
    ss_oversample  = config.training.get("distill_ss_oversample", 3)

    # ── Soundscape val split ──────────────────────────────────────────────────
    ss_val_frac = config.training.get("soundscape_val_frac", 0.2)
    _ss_df_full = pd.read_csv(config.data.soundscapes_labels_csv)
    _ss_files   = sorted(_ss_df_full["filename"].unique())
    _n_val      = max(1, int(len(_ss_files) * ss_val_frac))
    _val_files  = set(_ss_files[-_n_val:])
    _ss_val_df  = _ss_df_full[_ss_df_full["filename"].isin(_val_files)]

    _tmp_val = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    _ss_val_df.to_csv(_tmp_val.name, index=False); _tmp_val.close()

    print(f"\nSoundscape split: val={len(_val_files)} files")
    val_ds = MelSoundscapeDataset(
        soundscapes_dir=config.data.train_soundscapes_dir,
        labels_csv=_tmp_val.name,
        species_to_idx=species_to_idx,
        num_classes=num_classes,
        sample_rate=config.audio.sample_rate,
        clip_duration=config.audio.clip_duration,
        yield_raw_audio=use_gpu_mel,
        **mel_kw,
    )
    val_data, val_labels = val_ds.get_all_samples()
    os.unlink(_tmp_val.name)
    print(f"Validation clips: {len(val_data)}")

    # ── Distillation dataset (soft soundscape labels from Perch teacher) ──────
    teacher_csv  = config.data.get("teacher_csv", None)
    distill_loader = None
    if teacher_csv and os.path.isfile(teacher_csv):
        distill_ds = SoftSoundscapeDataset(
            teacher_csv     = teacher_csv,
            soundscapes_dir = config.data.train_soundscapes_dir,
            species_list    = target_species,
            num_classes     = num_classes,
            sample_rate     = config.audio.sample_rate,
            clip_duration   = config.audio.clip_duration,
            shuffle         = True,
        )
        distill_loader = DataLoader(
            distill_ds,
            batch_size = config.training.batch_size,
            num_workers = 2,
            pin_memory  = True,
        )
        print(f"Distillation loader ready ({len(distill_ds.samples)} clips)")
    else:
        print(f"WARN: teacher_csv not found: {teacher_csv}  — running without distillation")

    # ── Train audio dataset ───────────────────────────────────────────────────
    secondary_label_weight = config.data.get("secondary_label_weight", 0.5)
    print(f"\nBuilding training dataset (secondary_label_weight={secondary_label_weight}) …")
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
        secondary_label_weight=secondary_label_weight,
        min_rating=config.data.min_rating,
        max_files=config.data.get("max_files", None),
        augment_config=dict(config.augmentation),
        yield_raw_audio=use_gpu_mel,
        **mel_kw,
    )
    train_loader = DataLoader(
        _IterableWrapper(train_ds_obj),
        batch_size  = config.training.batch_size,
        num_workers = 4,
        pin_memory  = True,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    use_llrd = config.training.get("use_llrd", False)
    params   = model.parameters()
    if use_llrd:
        optimizer = build_llrd_optimizer(
            model=model,
            base_lr=config.training.learning_rate,
            backbone_lr_mult=config.training.get("backbone_lr_mult", 0.1),
            weight_decay=config.training.get("weight_decay", 1e-4),
        )
    else:
        optimizer = torch.optim.AdamW(
            params,
            lr=config.training.learning_rate,
            weight_decay=config.training.get("weight_decay", 1e-4),
        )
    if pcen_tf is not None:
        optimizer.add_param_group({"params": pcen_tf.parameters(), "lr": config.training.learning_rate})

    # ── Checkpointing setup ───────────────────────────────────────────────────
    ckpt_dir    = os.path.join(config.output.get("checkpoint_dir", "checkpoints"), run_name)
    out_dir     = os.path.join(config.output.get("dir", "outputs"), run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(out_dir,  exist_ok=True)
    save_config(config, os.path.join(out_dir, "config.yaml"))

    save_topk    = config.training.get("save_topk_checkpoints", 3)
    best_heap    = []   # min-heap of (auc, path)
    best_auc     = 0.0
    best_epoch   = 0
    history      = []
    epochs       = config.training.epochs
    warmup_ep    = config.training.get("warmup_epochs", 3)
    mixup_alpha  = config.training.get("mixup_alpha", 0.0)
    circ_prob    = config.augmentation.get("circ_shift_prob", 0.0)

    # ── Early stopping ─────────────────────────────────────────────────────────
    early_stop_patience = config.training.get("early_stopping_patience", 0)
    no_improve_count = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        lr = cosine_lr_with_warmup(epoch, epochs, config.training.learning_rate, warmup_ep)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        t0 = time.time()
        train_loss = train_epoch_distill(
            model         = model,
            train_loader  = train_loader,
            distill_loader= distill_loader,
            optimizer     = optimizer,
            scaler        = scaler,
            device        = device,
            loss_fn       = loss_fn,
            loss_mode     = loss_mode,
            clip_loss_w   = clip_loss_w,
            frame_loss_w  = frame_loss_w,
            pcen_tf       = pcen_tf,
            mel_tf        = mel_tf,
            use_amp       = use_amp,
            mixup_alpha   = mixup_alpha,
            circ_shift_prob = circ_prob,
            distill_weight  = distill_weight,
            ss_oversample   = ss_oversample,
        )
        ep_time = time.time() - t0

        val_auc, val_cmap = evaluate(
            model, val_data, val_labels, device,
            mel_tf=mel_tf, pcen_tf=pcen_tf,
            use_gpu_mel=use_gpu_mel, use_amp=use_amp,
        )

        print(f"  ep {epoch:3d}/{epochs}  "
              f"loss={train_loss:.4f}  val_auc={val_auc:.4f}  "
              f"lr={lr:.5f}  {ep_time:.0f}s")

        # Save top-K
        ckpt_path = os.path.join(ckpt_dir, f"soup_ep{epoch:03d}_sed.pt")
        ckpt_data = {
            "epoch":            epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state":  optimizer.state_dict(),
            "metrics":          {"macro_auc": val_auc, "cmap": val_cmap},
        }
        if pcen_tf is not None:
            ckpt_data["pcen_state_dict"] = pcen_tf.state_dict()

        if len(best_heap) < save_topk:
            heapq.heappush(best_heap, (val_auc, ckpt_path))
            torch.save(ckpt_data, ckpt_path)
            print(f"    ep {epoch:3d}  val_auc={val_auc:.4f}  {os.path.basename(ckpt_path)}")
        elif val_auc > best_heap[0][0]:
            old_auc, old_path = heapq.heapreplace(best_heap, (val_auc, ckpt_path))
            torch.save(ckpt_data, ckpt_path)
            if os.path.exists(old_path):
                os.remove(old_path)
            print(f"    ep {epoch:3d}  val_auc={val_auc:.4f}  {os.path.basename(ckpt_path)}")

        if val_auc > best_auc:
            best_auc   = val_auc
            best_epoch = epoch
            no_improve_count = 0
            torch.save(ckpt_data, os.path.join(ckpt_dir, "best_sed.pt"))
        else:
            no_improve_count += 1

        hist_row = {"epoch": epoch, "train_loss": round(train_loss, 6),
                    "val_roc_auc": round(val_auc, 6), "lr": round(lr, 8),
                    "epoch_time_s": round(ep_time, 1)}
        history.append(hist_row)

        if use_wandb:
            try:
                import wandb
                wandb.log({"train_loss": train_loss, "val_auc": val_auc,
                           "lr": lr, "epoch": epoch})
            except Exception:
                pass

        _save_results(run_name, best_auc, best_epoch, epoch, history,
                      config, finished=False, out_dir=out_dir)

        # ── Early stopping check ───────────────────────────────────────────────
        if early_stop_patience > 0 and no_improve_count >= early_stop_patience:
            print(f"  ✗ Early stopping at epoch {epoch}: "
                  f"no improvement for {early_stop_patience} epochs "
                  f"(best={best_auc:.4f} @ ep{best_epoch})")
            break

    _save_results(run_name, best_auc, best_epoch, epochs, history,
                  config, finished=True, out_dir=out_dir)

    if use_wandb:
        try:
            import wandb; wandb.finish()
        except Exception:
            pass

    print(f"\n{'='*60}")
    print(f"  {run_name} complete")
    print(f"  Best val_auc = {best_auc:.4f} @ epoch {best_epoch}")
    print(f"  Checkpoint   : {ckpt_dir}/best_sed.pt")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
