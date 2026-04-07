"""EfficientNet-B0 SED — Knowledge Distillation from Competitor SED.

Teacher : competitor_sed_fold0.pt (EfficientNet-B0, soundscape AUC=0.9478)
Student : EfficientNet-B0 trained from scratch with:
  - Hard label loss   : FocalBCE(student, train.csv hard labels)
  - KD loss           : BCE(student_prob, competitor_soft_label)
  - Augmentation      : SumixFreq + SpecAug + MixUp (1st-place recipe)
  - Validation        : GroupKFold soundscape AUC (same as train_sed_ns.py)

Requires:
  1. competitor pseudo labels generated first:
     CUDA_VISIBLE_DEVICES=1 python scripts/gen_competitor_pseudo.py

  2. Then train:
     CUDA_VISIBLE_DEVICES=1 python train_distill_competitor.py \
         --config configs/distill_competitor_b0_v1.yaml --fold 0

Usage (full 5-fold):
    CUDA_VISIBLE_DEVICES=1 python train_distill_competitor.py \
        --config configs/distill_competitor_b0_v1.yaml
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import timm
import soundfile as sf
import librosa
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils.config import load_config

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

torch.backends.cudnn.benchmark = True

SR          = 32_000
NUM_CLASSES = 234
CLIP_SAMPLES = SR * 5   # default; overridden by config


# ── Model ─────────────────────────────────────────────────────────────────────

class GEMFreqPool(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.tensor(p_init))
        self.eps = eps

    def forward(self, x):
        p = self.p.clamp(min=1.0)
        return x.clamp(min=self.eps).pow(p).mean(dim=2).pow(1.0 / p)


class AttentionSEDHead(nn.Module):
    def __init__(self, feat_dim, num_classes, dropout=0.1):
        super().__init__()
        self.fc       = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)
        )
        self.att_conv = nn.Conv1d(feat_dim, num_classes, 1)
        self.cls_conv = nn.Conv1d(feat_dim, num_classes, 1)

    def forward(self, x):
        x   = self.fc(x.permute(0, 2, 1)).permute(0, 2, 1)
        att = F.softmax(torch.tanh(self.att_conv(x)), dim=-1)
        cls = self.cls_conv(x)
        logit = (att * cls).sum(-1)
        return {'clipwise_logit': logit, 'clipwise_prob': torch.sigmoid(logit)}


class SEDModel(nn.Module):
    def __init__(self, backbone='tf_efficientnet_b0.ns_jft_in1k',
                 num_classes=234, dropout=0.1, drop_path_rate=0.0,
                 gem_p_init=3.0):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=True, in_chans=3,
            features_only=False, global_pool='', num_classes=0,
            drop_path_rate=drop_path_rate,
        )
        self.gem_pool = GEMFreqPool(p_init=gem_p_init)
        feat_dim      = self.backbone.num_features
        self.head     = AttentionSEDHead(feat_dim, num_classes, dropout)

    def forward(self, x):
        return self.head(self.gem_pool(self.backbone(x)))


# ── Mel transform + Augmentation ──────────────────────────────────────────────

class MelTransform(nn.Module):
    def __init__(self, sr=SR, n_mels=224, n_fft=2048, hop_length=512,
                 fmin=0, fmax=16_000, top_db=80.0, power=2.0,
                 norm='slaney', mel_scale='htk', peak_norm=False, **_):
        super().__init__()
        self.peak_norm = peak_norm
        self.mel = T.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=fmin, f_max=fmax,
            power=power, norm=norm, mel_scale=mel_scale,
        )
        self.db  = T.AmplitudeToDB(stype='power', top_db=top_db)

    @torch.no_grad()
    def forward(self, wav):
        wav = torch.nan_to_num(wav.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if self.peak_norm:
            peak = wav.abs().amax(dim=1, keepdim=True).clamp(min=1e-7)
            wav  = wav / peak
        mel  = torch.nan_to_num(self.db(self.mel(wav)), nan=-80.0)
        B    = mel.shape[0]
        flat = mel.reshape(B, -1)
        mn   = flat.min(1, keepdim=True)[0].unsqueeze(-1)
        mx   = flat.max(1, keepdim=True)[0].unsqueeze(-1)
        mel  = torch.nan_to_num((mel - mn) / (mx - mn + 1e-7), nan=0.0)
        return mel.unsqueeze(1).repeat(1, 3, 1, 1)  # (B, 3, n_mels, T)


class SpecAug(nn.Module):
    def __init__(self, freq_mask_param=24, time_mask_param=32, n_freq=2, n_time=2):
        super().__init__()
        self.freq_masks = nn.ModuleList([
            T.FrequencyMasking(freq_mask_param, iid_masks=True) for _ in range(n_freq)
        ])
        self.time_masks = nn.ModuleList([
            T.TimeMasking(time_mask_param, iid_masks=True) for _ in range(n_time)
        ])

    def forward(self, x):
        for m in self.freq_masks:
            x = m(x)
        for m in self.time_masks:
            x = m(x)
        return x


def absmax_normalize(audio: np.ndarray) -> np.ndarray:
    m = np.abs(audio).max()
    return audio / (m + 1e-8) if m > 1e-8 else audio


def sumix_freq(mel: torch.Tensor, labels_a: torch.Tensor,
               labels_b: torch.Tensor) -> tuple:
    """SumixFreq on mel; labels_b is competitor soft (or same as labels_a)."""
    B = mel.shape[0]
    if B < 2:
        return mel, labels_a, labels_b
    idx  = torch.randperm(B, device=mel.device)
    mask = (torch.rand(mel.shape[2], device=mel.device) > 0.5).view(1, 1, -1, 1)
    mixed = torch.where(mask, mel[idx], mel)
    return mixed, torch.max(labels_a, labels_a[idx]), torch.max(labels_b, labels_b[idx])


def audio_mixup(x: torch.Tensor, hard: torch.Tensor,
                soft: torch.Tensor) -> tuple:
    """Fixed lam=0.5 MixUp on audio; both label tensors take max (union)."""
    B   = x.shape[0]
    idx = torch.randperm(B, device=x.device)
    return (
        0.5 * x + 0.5 * x[idx],
        torch.max(hard, hard[idx]),
        torch.max(soft, soft[idx]),
    )


# ── Datasets ──────────────────────────────────────────────────────────────────

def load_audio_clip(path: str, n_samples: int) -> np.ndarray:
    try:
        audio, orig_sr = sf.read(path, dtype='float32', always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if orig_sr != SR:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=SR)
    except Exception:
        return np.zeros(n_samples, dtype=np.float32)
    if len(audio) < n_samples:
        audio = np.pad(audio, (0, n_samples - len(audio)))
    else:
        start = np.random.randint(0, len(audio) - n_samples + 1)
        audio = audio[start: start + n_samples]
    return absmax_normalize(audio.astype(np.float32))


def load_ss_clip(path: str, offset_sec: int, n_samples: int) -> np.ndarray:
    try:
        start_sample = max(0, offset_sec - n_samples // SR) * SR
        audio, orig_sr = sf.read(path, start=start_sample, frames=n_samples * 2,
                                  dtype='float32', always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if orig_sr != SR:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=SR)
    except Exception:
        return np.zeros(n_samples, dtype=np.float32)
    if len(audio) < n_samples:
        audio = np.pad(audio, (0, n_samples - len(audio)))
    else:
        audio = audio[:n_samples]
    return absmax_normalize(audio.astype(np.float32))


class TrainAudioKDDataset(Dataset):
    """train_audio dataset returning (audio, hard_label, competitor_soft_label).

    competitor_soft_label comes from the pre-computed file-level mean predictions
    (from gen_competitor_pseudo.py).  Files not found in the npz get soft_label=zeros.
    """

    def __init__(self, df: pd.DataFrame, audio_dir: str, species_cols: list,
                 competitor_probs: dict, n_samples: int):
        self.df              = df.reset_index(drop=True)
        self.audio_dir       = audio_dir
        self.species_cols    = species_cols
        self.competitor_probs = competitor_probs   # {rel_path: (234,) float32}
        self.n_samples       = n_samples
        self._zeros          = np.zeros(NUM_CLASSES, dtype=np.float32)

        # Pre-cache all audio clips into RAM (float16) to eliminate WSL disk I/O
        print(f"  Pre-caching {len(self.df):,} audio clips into RAM …", flush=True)
        self._cache = []
        for i, row in enumerate(self.df.itertuples(index=False)):
            rel  = str(row.filename)
            path = os.path.join(self.audio_dir, rel)
            clip = load_audio_clip(path, self.n_samples).astype(np.float16)
            self._cache.append(clip)
            if (i + 1) % 5000 == 0:
                print(f"    cached {i+1:,}/{len(self.df):,}", flush=True)
        print("  Pre-cache complete.", flush=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        rel  = str(row['filename'])                    # e.g. '1161364/XC12345.ogg'
        audio = self._cache[idx].astype(np.float32)

        # Hard label
        hard = np.zeros(NUM_CLASSES, dtype=np.float32)
        if str(row.get('primary_label', '')) in self.species_cols:
            hard[self.species_cols.index(str(row['primary_label']))] = 1.0
        sec = str(row.get('secondary_labels', ''))
        if sec and sec not in ('[]', 'nan', ''):
            import re
            for sp in re.split(r"[;,\[\]'\s]+", sec):
                sp = sp.strip()
                if sp in self.species_cols:
                    hard[self.species_cols.index(sp)] = 0.5

        # Competitor soft label (file-level mean)
        soft = self.competitor_probs.get(rel, self._zeros)

        return (
            torch.from_numpy(audio),
            torch.from_numpy(hard),
            torch.from_numpy(soft),
        )


class SoundscapeKDDataset(Dataset):
    """All train_soundscapes paired with competitor soft labels (from npz).

    Used as unlabeled KD source — no ground-truth labels needed.
    row_id format in npz: {soundscape_stem}_{end_sec}
    """

    def __init__(self, ss_dir: str, ss_npz: str, n_samples: int,
                 exclude_files: set = None):
        self.ss_dir    = ss_dir
        self.n_samples = n_samples
        npz = np.load(ss_npz, allow_pickle=True)
        rids  = npz['row_ids'].tolist()
        probs = npz['probs'].astype(np.float32)

        self.samples = []
        for rid, prob in zip(rids, probs):
            parts = str(rid).rsplit('_', 1)
            fname = parts[0] + '.ogg'
            if exclude_files and fname in exclude_files:
                continue
            end_sec = int(parts[1]) if len(parts) == 2 else 5
            self.samples.append((fname, end_sec, prob))
        print(f"  SoundscapeKDDataset: {len(self.samples):,} clips")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, end_sec, soft = self.samples[idx]
        audio = load_ss_clip(os.path.join(self.ss_dir, fname), end_sec, self.n_samples)
        return torch.from_numpy(audio), torch.from_numpy(soft)


class SoundscapeValDataset(Dataset):
    def __init__(self, df: pd.DataFrame, ss_dir: str, species_cols: list,
                 n_samples: int):
        self.df          = df.reset_index(drop=True)
        self.ss_dir      = ss_dir
        self.species_cols = species_cols
        self.n_samples   = n_samples

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        audio = load_ss_clip(
            os.path.join(self.ss_dir, str(row['filename'])),
            int(row.get('end', 5)),
            self.n_samples,
        )
        label = np.array([row[sc] for sc in self.species_cols], dtype=np.float32)
        return torch.from_numpy(audio), torch.from_numpy(label)


# ── Loss ──────────────────────────────────────────────────────────────────────

class FocalBCE(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt  = torch.exp(-bce)
        return ((1 - pt) ** self.gamma * bce).mean()


# ── Val helpers ───────────────────────────────────────────────────────────────

def macro_auc(y_true, y_score):
    keep = y_true.sum(axis=0) > 0
    if keep.sum() == 0:
        return 0.0
    return roc_auc_score(y_true[:, keep], y_score[:, keep], average='macro')


def hhmmss_to_sec(t) -> int:
    if isinstance(t, (int, float)):
        return int(t)
    try:
        parts = str(t).split(':')
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return int(float(t))
    except Exception:
        return 0


def build_ss_val_df(sc_labels: pd.DataFrame, species_cols: list) -> pd.DataFrame:
    sp2idx = {sp: i for i, sp in enumerate(species_cols)}
    rows   = []
    for filename, grp in sc_labels.groupby('filename'):
        for _, row in grp.iterrows():
            end   = hhmmss_to_sec(row.get('end', 5))
            label = np.zeros(NUM_CLASSES, dtype=np.float32)
            for sp in str(row.get('primary_label', '')).split(';'):
                sp = sp.strip()
                if sp in sp2idx:
                    label[sp2idx[sp]] = 1.0
            row_dict = {'filename': filename, 'end': end}
            for j, sc in enumerate(species_cols):
                row_dict[sc] = label[j]
            rows.append(row_dict)
    return pd.DataFrame(rows)


# ── Training ──────────────────────────────────────────────────────────────────

def train_fold(fold: int, cfg: dict, device: torch.device,
               competitor_probs: dict) -> dict:
    t_cfg    = cfg['training']
    d_cfg    = cfg['data']
    m_cfg    = cfg.get('model', {})
    clip_dur = m_cfg.get('clip_duration', 5)
    n_samples = SR * clip_dur
    out_dir  = Path(cfg['output']['dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    train_df     = pd.read_csv(d_cfg['train_csv'])
    sc_labels    = pd.read_csv(d_cfg['soundscape_labels_csv'])
    taxonomy     = pd.read_csv(d_cfg['taxonomy_csv'])
    species_cols = taxonomy['primary_label'].astype(str).tolist()

    # GroupKFold on soundscape files
    sc_files  = sc_labels['filename'].unique()
    sc_groups = [f.split('_')[2] for f in sc_files]
    gkf       = GroupKFold(n_splits=d_cfg.get('n_folds', 5))
    fold_splits = list(gkf.split(sc_files, groups=sc_groups))
    _, val_idx  = fold_splits[fold]
    val_files   = set(sc_files[val_idx])

    sc_val_raw = sc_labels[sc_labels['filename'].isin(val_files)]
    sc_val_df  = build_ss_val_df(sc_val_raw, species_cols)
    print(f"Fold {fold}: val_sc_files={len(val_files)}, val_clips={len(sc_val_df)}")

    # Datasets
    audio_ds = TrainAudioKDDataset(
        df=train_df, audio_dir=d_cfg['audio_dir'],
        species_cols=species_cols,
        competitor_probs=competitor_probs,
        n_samples=n_samples,
    )
    val_ds   = SoundscapeValDataset(
        df=sc_val_df, ss_dir=d_cfg['soundscape_dir'],
        species_cols=species_cols, n_samples=n_samples,
    )
    print(f"  train_audio clips : {len(audio_ds)}")

    # Optional soundscape KD branch (competitor predictions on train_soundscapes)
    ss_npz = d_cfg.get('competitor_ss_npz')
    ss_kd_loader = None
    if ss_npz and os.path.isfile(ss_npz):
        ss_kd_ds = SoundscapeKDDataset(
            ss_dir=d_cfg['soundscape_dir'], ss_npz=ss_npz,
            n_samples=n_samples, exclude_files=val_files,
        )
        ss_kd_loader = DataLoader(ss_kd_ds, batch_size=bs, shuffle=True,
                                  num_workers=2, pin_memory=True, drop_last=True)
        print(f"  soundscape KD clips: {len(ss_kd_ds):,}")

    bs = t_cfg.get('batch_size', 32)
    audio_loader = DataLoader(audio_ds, batch_size=bs, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True,
                              persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=2, pin_memory=True,
                              persistent_workers=True)

    # ── Model ────────────────────────────────────────────────────────────────
    model = SEDModel(
        backbone       = m_cfg.get('backbone', 'tf_efficientnet_b0.ns_jft_in1k'),
        num_classes    = NUM_CLASSES,
        dropout        = m_cfg.get('dropout', 0.1),
        drop_path_rate = m_cfg.get('drop_path_rate', 0.0),
        gem_p_init     = m_cfg.get('gem_p_init', 3.0),
    ).to(device)

    mel_tf   = MelTransform(**{k: v for k, v in m_cfg.items()
                                if k in ('sr', 'n_mels', 'n_fft', 'hop_length',
                                         'fmin', 'fmax', 'top_db', 'power',
                                         'norm', 'mel_scale', 'peak_norm')}).to(device)
    spec_aug = SpecAug(
        freq_mask_param = m_cfg.get('freq_mask', 24),
        time_mask_param = m_cfg.get('time_mask', 32),
    ).to(device)

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = t_cfg.get('learning_rate', 1e-3),
        weight_decay = t_cfg.get('weight_decay', 1e-4),
    )
    epochs = t_cfg.get('epochs', 30)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    focal_gamma    = t_cfg.get('focal_gamma', 2.0)
    kd_weight      = t_cfg.get('kd_weight', 0.5)
    ss_kd_weight   = t_cfg.get('ss_kd_weight', 0.5)
    ss_oversample  = t_cfg.get('ss_oversample', 1)
    use_sumix_freq = t_cfg.get('use_sumix_freq', True)
    criterion      = FocalBCE(gamma=focal_gamma).to(device)

    best_auc       = 0.0
    best_state     = None
    history        = []
    oof_logits     = np.zeros((len(val_ds), NUM_CLASSES), dtype=np.float32)
    patience       = t_cfg.get('early_stopping_patience', 5)
    no_improve_cnt = 0

    # ── WandB ────────────────────────────────────────────────────────────────
    run = None
    if _WANDB_AVAILABLE:
        run = wandb.init(
            project = 'birdclef-2026',
            name    = f"{cfg['experiment']['name']}-fold{fold}",
            group   = cfg['experiment']['name'],
            tags    = ['distill-competitor', 'kd', f"fold{fold}"],
            config  = {**cfg, 'fold': fold},
            reinit  = True,
        )

    print(f"  Training fold {fold} | epochs={epochs} | early_stop={patience} | kd_w={kd_weight}")

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss    = 0.0
        ep_hard_l  = 0.0
        ep_kd_l    = 0.0
        n_steps    = 0
        t0         = time.time()
        ss_iter    = iter(ss_kd_loader) if ss_kd_loader else None

        for audio_wav, hard_label, soft_label in audio_loader:
            audio_wav  = audio_wav.to(device)
            hard_label = hard_label.to(device)
            soft_label = soft_label.to(device)

            # Audio MixUp (fixed lam=0.5, both hard and soft labels take max)
            audio_wav, hard_label, soft_label = audio_mixup(
                audio_wav, hard_label, soft_label
            )

            # Mel + SpecAug
            with torch.no_grad():
                mel = mel_tf(audio_wav)
            mel = spec_aug(mel)

            # SumixFreq (frequency-bin mixing, labels take max)
            if use_sumix_freq:
                mel, hard_label, soft_label = sumix_freq(mel, hard_label, soft_label)

            with torch.cuda.amp.autocast():
                out   = model(mel)
                logit = out['clipwise_logit']
                prob  = out['clipwise_prob']
                hard_loss = criterion(logit, hard_label)

            # BCE with soft labels must be in float32 (not safe inside autocast)
            kd_loss = F.binary_cross_entropy(
                prob.float().clamp(1e-7, 1 - 1e-7),
                soft_label.float().clamp(0.0, 1.0),
            )
            loss = hard_loss + kd_weight * kd_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            ep_loss   += loss.item()
            ep_hard_l += hard_loss.item()
            ep_kd_l   += kd_loss.item()
            n_steps   += 1

            # ── Soundscape KD branch (unlabeled soundscapes) ──────────────────
            if ss_iter is not None:
                for _ in range(ss_oversample):
                    try:
                        ss_wav, ss_soft = next(ss_iter)
                    except StopIteration:
                        ss_iter = iter(ss_kd_loader)
                        try:
                            ss_wav, ss_soft = next(ss_iter)
                        except StopIteration:
                            break

                    ss_wav  = ss_wav.to(device)
                    ss_soft = ss_soft.to(device)

                    with torch.no_grad():
                        ss_mel = mel_tf(ss_wav)
                    ss_mel = spec_aug(ss_mel)

                    with torch.cuda.amp.autocast():
                        ss_out  = model(ss_mel)
                        ss_prob = ss_out['clipwise_prob']
                    ss_loss = F.binary_cross_entropy(
                        ss_prob.float().clamp(1e-7, 1 - 1e-7),
                        ss_soft.float().clamp(0.0, 1.0),
                    ) * ss_kd_weight

                    optimizer.zero_grad()
                    scaler.scale(ss_loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()

        sched.step()
        ep_time = time.time() - t0

        # Validation
        model.eval()
        val_logits_ep = []
        val_labels_ep = []
        with torch.no_grad():
            for waves, labels in val_loader:
                mel = mel_tf(waves.to(device))
                out = model(mel)
                val_logits_ep.append(out['clipwise_logit'].cpu().numpy())
                val_labels_ep.append(labels.numpy())

        vl  = np.concatenate(val_logits_ep)
        vla = np.concatenate(val_labels_ep)
        vp  = 1.0 / (1.0 + np.exp(-vl))
        auc = macro_auc(vla, vp)

        avg_loss   = ep_loss   / max(n_steps, 1)
        avg_hard   = ep_hard_l / max(n_steps, 1)
        avg_kd     = ep_kd_l   / max(n_steps, 1)

        print(f"  Ep {ep:3d}/{epochs}  "
              f"loss={avg_loss:.4f}  hard={avg_hard:.4f}  kd={avg_kd:.4f}  "
              f"ss_auc={auc:.4f}  {ep_time:.0f}s")
        history.append({'epoch': ep, 'loss': avg_loss, 'hard_loss': avg_hard,
                        'kd_loss': avg_kd, 'val_auc': auc})

        if run is not None:
            run.log({'epoch': ep, 'train/loss': avg_loss, 'train/hard': avg_hard,
                     'train/kd': avg_kd, 'val/ss_auc': auc, 'val/best_auc': best_auc})

        if auc > best_auc:
            best_auc       = auc
            no_improve_cnt = 0
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            oof_logits     = vl
            torch.save({
                'state_dict':   best_state,
                'fold':         fold,
                'best_val_auc': best_auc,
                'epoch':        ep,
            }, out_dir / f'fold{fold}_best.pt')
            print(f"    ✓ New best AUC={best_auc:.4f}")
        else:
            no_improve_cnt += 1
            if no_improve_cnt >= patience:
                print(f"  Early stopping at ep {ep} (no improvement for {patience} epochs)")
                break

    if run is not None:
        run.finish()

    return {'fold': fold, 'best_auc': best_auc, 'history': history,
            'oof_logits': oof_logits}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--fold',   type=int, default=None,
                        help='Single fold (default: all folds)')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    cfg = load_config(args.config)
    print(f"Config   : {args.config}")
    print(f"Device   : {device}")
    print(f"Exp name : {cfg['experiment']['name']}")

    # ── Load competitor pseudo labels ─────────────────────────────────────────
    pseudo_npz = cfg['data'].get('competitor_pseudo_npz')
    if not pseudo_npz or not os.path.isfile(pseudo_npz):
        print(f"ERROR: competitor_pseudo_npz not found: {pseudo_npz}")
        print("Run scripts/gen_competitor_pseudo.py first.")
        sys.exit(1)

    print(f"Loading competitor pseudo labels from {pseudo_npz} …")
    npz = np.load(pseudo_npz, allow_pickle=True)
    filenames_list = npz['filenames'].tolist()
    probs_arr      = npz['probs'].astype(np.float32)
    competitor_probs = {fn: probs_arr[i] for i, fn in enumerate(filenames_list)}
    print(f"  {len(competitor_probs):,} files loaded")
    print(f"  mean max prob: {probs_arr.max(axis=1).mean():.4f}")

    # ── Run folds ─────────────────────────────────────────────────────────────
    n_folds     = cfg['data'].get('n_folds', 5)
    folds       = [args.fold] if args.fold is not None else list(range(n_folds))
    all_results = []
    all_oof     = {}

    for fold in folds:
        print(f"\n{'='*60}")
        print(f"  Fold {fold}/{n_folds-1}")
        print(f"{'='*60}")
        result = train_fold(fold, cfg, device, competitor_probs)
        all_results.append(result)
        all_oof[fold] = result['oof_logits']

    # ── Summary ───────────────────────────────────────────────────────────────
    out_dir = Path(cfg['output']['dir'])
    if len(all_results) == n_folds:
        mean_auc = np.mean([r['best_auc'] for r in all_results])
        print(f"\n{'='*60}")
        print("  Results (all folds):")
        for r in all_results:
            print(f"    Fold {r['fold']}: best_auc={r['best_auc']:.4f}")
        print(f"  Mean fold AUC : {mean_auc:.4f}")
        print(f"{'='*60}")

        with open(out_dir / 'result.json', 'w') as f:
            json.dump({'mean_fold_auc': mean_auc,
                       'folds': [{'fold': r['fold'], 'best_auc': r['best_auc']}
                                  for r in all_results]}, f, indent=2)

        np.savez_compressed(
            str(out_dir / 'oof_predictions.npz'),
            logits=np.concatenate(list(all_oof.values())),
        )
        print(f"  OOF saved → {out_dir / 'oof_predictions.npz'}")


if __name__ == '__main__':
    main()
