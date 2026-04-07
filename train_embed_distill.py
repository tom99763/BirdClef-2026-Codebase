"""BirdClef 2026 — Perch Embedding Distillation (Representation Learning)

The correct distillation approach:
  "throw in a bunch of wave files (unlabeled soundscape files from this and
   previous competitions), extract their embeddings and make a database,
   distill (NOT train classifier head) to your favourite pytorch models and enjoy!"

How it works:
  1. Pre-compute Perch embeddings for all available audio → embeddings_cache/
     (already done: 107k train clips + 739 soundscape clips = 1536-dim each)
  2. Train EfficientNet-B0 backbone + projection head to MATCH Perch's embedding
     space via cosine similarity loss (NOT classification, no labels needed)
  3. Export the pretrained backbone → use as initialization for SED fine-tuning
     (replace projection head with AttentionSEDHead for classification)

Why this is better than output distillation:
  - Can use ANY unlabeled audio (not limited to Perch-labeled soundscapes)
  - Learns Perch's acoustic representation space, not just its class outputs
  - 100x more data available (all train_audio clips, all soundscapes)
  - Backbone generalises better when fine-tuned for classification

Usage:
    python train_embed_distill.py --config configs/embed_distill_b0_v1.yaml
    python train_embed_distill.py --config configs/embed_distill_b0_v1.yaml --gpu 0
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as AT
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.utils.config import load_config, save_config
from src.utils.audio import load_audio
from src.model.sed_model import GEMFreqPool


# ── Dataset ──────────────────────────────────────────────────────────────────

class EmbeddingPairDataset(Dataset):
    """Pairs audio clips with their pre-computed Perch embeddings.

    Two modes:
    1. Mel cache mode (fast): reads pre-computed mel .npy files from mel_cache_dir.
       Each mel is (N_MELS, T) float16.  No librosa needed → no fork deadlock.
    2. Raw audio mode (slow first epoch): loads .ogg files via librosa.

    Mode is selected automatically: if mel_cache_dir is set and files exist, use it.
    """

    def __init__(
        self,
        manifest_path: str,
        train_audio_dir: str,
        soundscapes_dir: str,
        sample_rate: int = 32000,
        clip_seconds: int = 5,
        max_rows: int = None,
        use_splits: list = None,
        _preloaded_df=None,
        mel_cache_dir: str = None,    # e.g. "outputs/mel_cache"
    ):
        if _preloaded_df is not None:
            df = _preloaded_df
        else:
            df = pd.read_csv(manifest_path)
            if use_splits:
                df = df[df['split'].isin(use_splits)]
            if max_rows:
                df = df.sample(min(max_rows, len(df)), random_state=42)
        self.df = df.reset_index(drop=True)
        self.train_audio_dir = train_audio_dir
        self.soundscapes_dir = soundscapes_dir
        self.sample_rate = sample_rate
        self.clip_samples = sample_rate * clip_seconds
        self.mel_cache_dir = mel_cache_dir

        # Check if mel cache exists
        if mel_cache_dir:
            row0 = self.df.iloc[0]
            stem = os.path.splitext(os.path.basename(row0['npy_path']))[0]
            test_path = os.path.join(mel_cache_dir, row0['split'], f"{stem}_mel.npy")
            self.use_mel_cache = os.path.exists(test_path)
        else:
            self.use_mel_cache = False

        mode = "mel-cache" if self.use_mel_cache else "raw-audio"
        print(f"[EmbeddingPairDataset] {len(self.df)} pairs  "
              f"(splits: {dict(df['split'].value_counts())})  mode={mode}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── Load Perch embedding ────────────────────────────────────────────
        emb = np.load(row['npy_path']).astype(np.float32)    # (1536,)

        if self.use_mel_cache:
            # ── Fast path: load pre-computed mel ───────────────────────────
            stem = os.path.splitext(os.path.basename(row['npy_path']))[0]
            mel_path = os.path.join(self.mel_cache_dir, row['split'],
                                    f"{stem}_mel.npy")
            mel = np.load(mel_path).astype(np.float32)       # (N_MELS, T) float16→32
            return torch.from_numpy(mel), torch.from_numpy(emb)

        else:
            # ── Slow path: load raw audio, return waveform ─────────────────
            if row['split'] == 'soundscape':
                audio_path = os.path.join(self.soundscapes_dir, row['source_file'])
            else:
                audio_path = os.path.join(self.train_audio_dir, row['source_file'])

            audio = load_audio(audio_path, self.sample_rate)
            if audio is None:
                return torch.zeros(self.clip_samples), torch.from_numpy(emb)

            start = int(row['clip_idx']) * self.clip_samples
            clip = audio[start: start + self.clip_samples]
            if len(clip) < self.clip_samples:
                clip = np.pad(clip, (0, self.clip_samples - len(clip)))

            return torch.from_numpy(clip.copy()).float(), torch.from_numpy(emb)


# ── Model ─────────────────────────────────────────────────────────────────────

class EmbeddingDistillModel(nn.Module):
    """
    EfficientNet-B0 backbone → GEMFreqPool → temporal avg → linear projection.

    Learns to produce embeddings matching Perch's 1536-dim representation space.

    After distillation, call get_backbone_state_dict() to extract backbone+gem
    weights for initialising a SEDModel for classification fine-tuning.
    """

    def __init__(
        self,
        backbone: str = "tf_efficientnet_b0.ns_jft_in1k",
        teacher_dim: int = 1536,
        pretrained: bool = True,
        drop_rate: float = 0.1,
        n_mels: int = 128,
        n_frames: int = 313,
        in_chans: int = 1,
    ):
        super().__init__()
        import timm
        self.in_chans = in_chans
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
            global_pool="",
            drop_rate=drop_rate,
        )

        # Probe feature dim
        with torch.no_grad():
            dummy = torch.zeros(1, in_chans, n_mels, n_frames)
            feat = self.backbone(dummy)     # (1, C, H', W')
            self.feat_dim = feat.shape[1]   # 1280 for B0

        self.gem = GEMFreqPool(p_init=3.0)   # pool over freq axis → (B, C, W)
        self.time_pool = nn.AdaptiveAvgPool1d(1)  # pool over time → (B, C, 1)

        # Projection head: map student features to teacher embedding space
        self.proj = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(self.feat_dim, teacher_dim, bias=False),
            nn.LayerNorm(teacher_dim),
        )

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[EmbeddingDistillModel] backbone={backbone}  feat_dim={self.feat_dim}"
              f"  teacher_dim={teacher_dim}  params={n_params:,}")

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, 1, n_mels, T) log-mel spectrogram (always 1-ch from cache)
        Returns:
            emb: (B, teacher_dim) L2-normalised embedding
        """
        if self.in_chans == 3:
            mel = mel.expand(-1, 3, -1, -1)  # replicate to (B, 3, n_mels, T)
        feat = self.backbone(mel)          # (B, C, H', W')
        feat = self.gem(feat)              # (B, 1280, W')  — freq pooled
        feat = self.time_pool(feat).squeeze(-1)  # (B, 1280) — time pooled
        proj = self.proj(feat)             # (B, 1536)
        return F.normalize(proj, dim=-1)   # unit-sphere → cosine loss = dot product

    def get_backbone_state_dict(self):
        """Extract backbone + GEM weights for SEDModel initialisation."""
        return {
            'backbone': self.backbone.state_dict(),
            'freq_pool': self.gem.state_dict(),
        }


# ── Mel transform ─────────────────────────────────────────────────────────────

def build_mel_transform(config, device):
    """Standard log-mel for the student (same pipeline as train_sed.py)."""
    mel_tf = nn.Sequential(
        AT.MelSpectrogram(
            sample_rate=config.audio.sample_rate,
            n_fft=config.mel.n_fft,
            hop_length=config.mel.hop_length,
            n_mels=config.mel.n_mels,
            f_min=config.mel.fmin,
            f_max=config.mel.fmax,
            power=2.0,
        ),
        AT.AmplitudeToDB(top_db=80.0),
    ).to(device)
    return mel_tf


# ── Training loop ─────────────────────────────────────────────────────────────

def cosine_lr(optimizer, epoch, total_epochs, base_lr, warmup=2):
    """Cosine LR with linear warmup."""
    if epoch < warmup:
        lr = base_lr * (epoch + 1) / warmup
    else:
        progress = (epoch - warmup) / max(total_epochs - warmup, 1)
        lr = base_lr * 0.5 * (1 + np.cos(np.pi * progress))
    lr = max(lr, base_lr * 1e-2)
    for g in optimizer.param_groups:
        g['lr'] = lr
    return lr


def _to_mel(x, mel_tf, use_mel_cache, device):
    """Convert input to (B,1,n_mels,T) mel tensor.
    If use_mel_cache: x is already (B, n_mels, T) pre-computed mel.
    Otherwise: x is (B, clip_samples) waveform → compute mel on GPU.
    """
    x = x.to(device)
    if use_mel_cache:
        mel = x.unsqueeze(1)                            # (B, 1, n_mels, T)
    else:
        mel = mel_tf(x).unsqueeze(1)                    # (B, 1, n_mels, T)
    mn = mel.flatten(1).min(1)[0].view(-1, 1, 1, 1)
    mx = mel.flatten(1).max(1)[0].view(-1, 1, 1, 1)
    return (mel - mn) / (mx - mn + 1e-7)


def infonce_loss(student_emb, teacher_emb, temperature=0.1):
    """NT-Xent / InfoNCE: diagonal = positive pairs, off-diagonal = negatives."""
    B = student_emb.shape[0]
    sim = torch.matmul(student_emb, teacher_emb.T) / temperature  # (B, B)
    labels = torch.arange(B, device=student_emb.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2


def train_epoch(model, mel_tf, loader, optimizer, scaler, device, use_mel_cache, aug_cfg=None, loss_fn='cosine'):
    model.train()
    total_loss = 0.0
    n = 0

    # Build SpecAugment transforms
    freq_masks, time_masks = [], []
    noise_std = 0.0
    mixup_alpha = 0.0
    if aug_cfg is not None and getattr(aug_cfg, 'enabled', False):
        freq_p   = getattr(aug_cfg, 'freq_mask_param', 30)
        time_p   = getattr(aug_cfg, 'time_mask_param', 48)
        n_freq   = getattr(aug_cfg, 'n_freq_masks', 1)
        n_time   = getattr(aug_cfg, 'n_time_masks', 1)
        noise_std   = getattr(aug_cfg, 'noise_std', 0.0)
        mixup_alpha = getattr(aug_cfg, 'mixup_alpha', 0.0)
        freq_masks = [AT.FrequencyMasking(freq_mask_param=freq_p) for _ in range(n_freq)]
        time_masks = [AT.TimeMasking(time_mask_param=time_p) for _ in range(n_time)]

    pbar = tqdm(loader, desc="  train", ncols=100, leave=False, file=sys.stdout, mininterval=30)
    for x, teacher_emb in pbar:
        teacher_emb = teacher_emb.to(device)            # (B, 1536)

        with autocast('cuda'):
            mel = _to_mel(x, mel_tf, use_mel_cache, device)  # (B,1,F,T) in [0,1]

            # Mixup (blend two mels; teacher stays clean → denoising objective)
            if mixup_alpha > 0.0:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                idx = torch.randperm(mel.shape[0], device=device)
                mel = lam * mel + (1 - lam) * mel[idx]

            # SpecAugment
            m = mel.squeeze(1)
            for fm in freq_masks: m = fm(m)
            for tm in time_masks: m = tm(m)
            mel = m.unsqueeze(1)

            # Gaussian noise on mel
            if noise_std > 0.0:
                mel = (mel + torch.randn_like(mel) * noise_std).clamp(0.0, 1.0)

            student_emb = model(mel)                    # (B, D) unit vectors
            teacher_emb_n = F.normalize(teacher_emb, dim=-1)

            if loss_fn == 'infonce':
                loss = infonce_loss(student_emb, teacher_emb_n)
            else:  # cosine
                loss = (1.0 - (student_emb * teacher_emb_n).sum(dim=-1)).mean()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{total_loss/n:.4f}", refresh=False)

    return total_loss / max(n, 1)


@torch.no_grad()
def val_epoch(model, mel_tf, loader, device, use_mel_cache):
    model.eval()
    total_cos = 0.0
    n = 0
    for x, teacher_emb in tqdm(loader, desc="  val", ncols=100, leave=False, file=sys.stdout, mininterval=30):
        teacher_emb = teacher_emb.to(device)
        mel = _to_mel(x, mel_tf, use_mel_cache, device)
        student_emb = model(mel)
        teacher_emb_n = F.normalize(teacher_emb, dim=-1)
        cos = (student_emb * teacher_emb_n).sum(dim=-1).mean().item()
        total_cos += cos
        n += 1
    return total_cos / max(n, 1)   # mean cosine similarity (higher = better)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/embed_distill_b0_v1.yaml")
    p.add_argument("--gpu", default=None)
    p.add_argument("--extra_epochs", type=int, default=0,
                   help="Resume from best_embed.pt and run N more epochs")
    return p.parse_args()


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_name = config.run_name
    out_dir = f"outputs/{run_name}"
    ckpt_dir = f"checkpoints/{run_name}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    save_config(config, f"{out_dir}/config.yaml")

    # ── Dataset & DataLoader (MUST be before wandb.init to avoid fork+thread deadlock) ──
    train_audio_dir = config.data.train_audio_dir
    soundscapes_dir = config.data.soundscapes_dir
    manifest_path   = config.data.manifest_path

    full_df = pd.read_csv(manifest_path)

    use_splits = ['train', 'soundscape']
    train_df = full_df[full_df['split'].isin(use_splits)].copy()
    val_df   = train_df.sample(frac=0.05, random_state=42)
    trn_df   = train_df.drop(val_df.index)

    print(f"Train pairs: {len(trn_df)}  Val pairs: {len(val_df)}")

    mel_cache_dir = config.data.get("mel_cache_dir", None)

    trn_ds = EmbeddingPairDataset(
        manifest_path=manifest_path,
        train_audio_dir=train_audio_dir,
        soundscapes_dir=soundscapes_dir,
        sample_rate=config.audio.sample_rate,
        clip_seconds=config.audio.clip_seconds,
        use_splits=use_splits,
        _preloaded_df=trn_df.reset_index(drop=True),
        mel_cache_dir=mel_cache_dir,
    )
    val_ds = EmbeddingPairDataset(
        manifest_path=manifest_path,
        train_audio_dir=train_audio_dir,
        soundscapes_dir=soundscapes_dir,
        sample_rate=config.audio.sample_rate,
        clip_seconds=config.audio.clip_seconds,
        use_splits=use_splits,
        _preloaded_df=val_df.reset_index(drop=True),
        mel_cache_dir=mel_cache_dir,
    )
    use_mel_cache = trn_ds.use_mel_cache

    # Create DataLoaders with fork BEFORE wandb.init (wandb spawns threads that break fork)
    trn_loader = DataLoader(
        trn_ds, batch_size=config.training.batch_size,
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.training.batch_size * 2,
        shuffle=False, num_workers=2, pin_memory=True,
    )

    # ── WandB (after DataLoader creation to avoid fork+thread issues) ──────
    try:
        import wandb
        wandb.init(project="birdclef-2026", name=run_name, config=dict(config))
        use_wandb = True
    except Exception:
        use_wandb = False

    # ── Model ──────────────────────────────────────────────────────────────
    model = EmbeddingDistillModel(
        backbone=config.model.backbone,
        teacher_dim=config.model.teacher_dim,
        pretrained=config.model.pretrained,
        drop_rate=config.model.drop_rate,
        n_mels=config.mel.n_mels,
        n_frames=313,
        in_chans=config.model.get("in_chans", 1),
    ).to(device)

    mel_tf = build_mel_transform(config, device)

    # ── Optimizer ──────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scaler = GradScaler('cuda', init_scale=256)  # lower init scale prevents float16 overflow on step 1

    # ── Resume from checkpoint (--extra_epochs) ────────────────────────────
    start_epoch = 1
    history = []
    best_cos = -1.0
    best_epoch = 0
    if args.extra_epochs > 0:
        ckpt_path = f"{ckpt_dir}/best_embed.pt"
        result_path = f"{out_dir}/result.json"
        if os.path.exists(ckpt_path) and os.path.exists(result_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            with open(result_path) as f:
                prev = json.load(f)
            history = prev.get("epoch_history", [])
            best_cos = prev.get("best_val_cos", -1.0)
            best_epoch = prev.get("best_epoch", 0)
            start_epoch = prev.get("total_epochs_run", 0) + 1
            print(f"Resumed from ep{start_epoch-1}  best_cos={best_cos:.4f}")

    # ── Training ───────────────────────────────────────────────────────────
    epochs = (start_epoch - 1) + (args.extra_epochs if args.extra_epochs > 0
                                   else config.training.epochs)
    patience = config.training.get("early_stopping_patience", 0)
    no_improve = 0

    print(f"\n{'='*60}")
    print(f"  Experiment : {run_name}  [Embedding Distillation]")
    print(f"  Backbone   : {config.model.backbone}")
    print(f"  Teacher dim: {config.model.teacher_dim}")
    print(f"  Train pairs: {len(trn_ds)}")
    print(f"  Epochs     : {epochs}")
    print(f"  Batch size : {config.training.batch_size}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        lr = cosine_lr(optimizer, epoch - 1, epochs, config.training.learning_rate,
                       warmup=config.training.get("warmup_epochs", 2))

        aug_cfg  = config.get("augmentation", None)
        loss_fn  = config.training.get("loss_fn", "cosine")
        train_loss = train_epoch(model, mel_tf, trn_loader, optimizer, scaler, device, use_mel_cache, aug_cfg, loss_fn)
        val_cos    = val_epoch(model, mel_tf, val_loader, device, use_mel_cache)
        elapsed    = time.time() - t0

        is_best = val_cos > best_cos
        if is_best:
            best_cos   = val_cos
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), f"{ckpt_dir}/best_embed.pt")
            torch.save(model.get_backbone_state_dict(),
                       f"{ckpt_dir}/best_backbone.pt")
            flag = "  ✓ best"
        else:
            no_improve += 1
            flag = ""

        print(f"  ep {epoch:3d}/{epochs}  loss={train_loss:.4f}  "
              f"val_cos={val_cos:.4f}  lr={lr:.2e}  t={elapsed:.0f}s{flag}")

        row = {"epoch": epoch, "train_loss": round(train_loss, 6),
               "val_cos": round(val_cos, 6), "lr": round(lr, 8),
               "epoch_time_s": round(elapsed, 1)}
        history.append(row)

        if use_wandb:
            wandb.log({"train_loss": train_loss, "val_cos": val_cos,
                       "lr": lr, "epoch": epoch})

        # Save result.json after each epoch
        result = {
            "run_name": run_name,
            "finished": False,
            "best_val_cos": round(best_cos, 6),
            "best_epoch": best_epoch,
            "total_epochs_run": epoch,
            "epoch_history": history,
        }
        with open(f"{out_dir}/result.json", "w") as f:
            json.dump(result, f, indent=2)

        if patience > 0 and no_improve >= patience:
            print(f"  ✗ Early stopping at epoch {epoch} "
                  f"(best={best_cos:.4f} @ ep{best_epoch})")
            break

    # Final save
    torch.save(model.state_dict(), f"{ckpt_dir}/final_embed.pt")
    torch.save(model.get_backbone_state_dict(), f"{ckpt_dir}/final_backbone.pt")
    result["finished"] = True
    result["total_time_s"] = sum(r["epoch_time_s"] for r in history)
    with open(f"{out_dir}/result.json", "w") as f:
        json.dump(result, f, indent=2)

    if use_wandb:
        wandb.finish()

    print(f"\n✓ Done. best_val_cos={best_cos:.4f} @ ep{best_epoch}")
    print(f"  Backbone weights → {ckpt_dir}/best_backbone.pt")
    print(f"  Use with train_sed.py --pretrained_backbone {ckpt_dir}/best_backbone.pt")


if __name__ == "__main__":
    main()
