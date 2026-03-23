"""EfficientNet-B0 SED — Noisy Student training for BirdCLEF 2026.

5-fold cross-validation. Validation fold uses the same soundscape file_id split
as train_proto_ssm.py (GroupKFold on 66 labeled soundscape files) for fair comparison.

Training data per fold:
  - ALL train_audio clips (weak-labeled, from train.csv)
  - Pseudo-labeled soundscape windows (from pseudo_labels/ns_rK.csv)

Validation:
  - Labeled soundscape fold k → soundscape-level macro AUC (comparable to SSM OOF)

Usage:
  python train_sed_ns.py --config configs/sed_ns_b0_r1.yaml [--fold 0]
  CUDA_VISIBLE_DEVICES=1 python train_sed_ns.py --config configs/sed_ns_b0_r1.yaml --fold 0
"""

import argparse
import json
import math
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils.config import load_config

import soundfile as sf
import librosa

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

torch.backends.cudnn.benchmark = True


# ── Model ─────────────────────────────────────────────────────────────────────

class GEMFreqPool(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p_init))
        self.eps = eps

    def forward(self, x):
        p = self.p.clamp(min=1.0)
        return x.clamp(min=self.eps).pow(p).mean(dim=2).pow(1.0 / p)


class AttentionSEDHead(nn.Module):
    def __init__(self, feat_dim, num_classes, dropout=0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)
        )
        self.att_conv = nn.Conv1d(feat_dim, num_classes, 1)
        self.cls_conv = nn.Conv1d(feat_dim, num_classes, 1)

    def forward(self, x):
        # x: (B, feat_dim, T_frames)
        x = self.fc(x.permute(0, 2, 1)).permute(0, 2, 1)
        att = F.softmax(torch.tanh(self.att_conv(x)), dim=-1)
        cls = self.cls_conv(x)
        logit = (att * cls).sum(-1)
        return {'clipwise_logit': logit, 'clipwise_prob': torch.sigmoid(logit)}


class SEDModel(nn.Module):
    def __init__(self, backbone='tf_efficientnet_b0.ns_jft_in1k',
                 num_classes=234, in_channels=3, dropout=0.1, drop_path_rate=0.0,
                 gem_p_init=3.0):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=True, in_chans=in_channels,
            features_only=False, global_pool='', num_classes=0,
            drop_path_rate=drop_path_rate,
        )
        self.gem_pool = GEMFreqPool(p_init=gem_p_init)
        feat_dim      = self.backbone.num_features
        self.head     = AttentionSEDHead(feat_dim, num_classes, dropout)

    def forward(self, x):
        return self.head(self.gem_pool(self.backbone(x)))


# ── Mel transform ─────────────────────────────────────────────────────────────

class MelTransform(nn.Module):
    def __init__(self, sr=32_000, n_mels=224, n_fft=2048, hop_length=512,
                 fmin=0, fmax=16_000, top_db=80.0, power=2.0,
                 norm='slaney', mel_scale='htk', peak_norm=False):
        super().__init__()
        self.peak_norm = peak_norm
        self.mel = T.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=fmin, f_max=fmax,
            power=power, norm=norm, mel_scale=mel_scale,
        )
        self.db = T.AmplitudeToDB(stype='power', top_db=top_db)

    @torch.no_grad()
    def forward(self, waveforms):
        waveforms = torch.nan_to_num(waveforms.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if self.peak_norm:
            peak = waveforms.abs().amax(dim=1, keepdim=True).clamp(min=1e-7)
            waveforms = waveforms / peak
        mel = torch.nan_to_num(self.db(self.mel(waveforms)), nan=-80.0)
        B = mel.shape[0]
        flat = mel.reshape(B, -1)
        mn = flat.min(1, keepdim=True)[0].unsqueeze(-1)
        mx = flat.max(1, keepdim=True)[0].unsqueeze(-1)
        mel = torch.nan_to_num((mel - mn) / (mx - mn + 1e-7), nan=0.0)
        return mel.unsqueeze(1).repeat(1, 3, 1, 1)  # (B, 3, n_mels, T_frames)


# ── Augmentation ──────────────────────────────────────────────────────────────

class SpecAug(nn.Module):
    """SpecAugment: frequency and time masking on mel spectrograms."""
    def __init__(self, freq_mask_param=24, time_mask_param=32, n_freq=2, n_time=2):
        super().__init__()
        self.freq_masks = nn.ModuleList([
            T.FrequencyMasking(freq_mask_param, iid_masks=True) for _ in range(n_freq)
        ])
        self.time_masks = nn.ModuleList([
            T.TimeMasking(time_mask_param, iid_masks=True) for _ in range(n_time)
        ])

    def forward(self, x):
        # x: (B, 3, n_mels, T)
        for m in self.freq_masks:
            x = m(x)
        for m in self.time_masks:
            x = m(x)
        return x


def absmax_normalize(audio: np.ndarray) -> np.ndarray:
    """Normalize audio by absolute maximum (1st place BirdCLEF 2025).
    Ensures consistent amplitude scale before MixUp blending.
    """
    m = np.abs(audio).max()
    return audio / (m + 1e-8) if m > 1e-8 else audio


def sumix_freq(mel: torch.Tensor, labels: torch.Tensor) -> tuple:
    """SumixFreq (1st place BirdCLEF 2025): per-frequency-bin random selection.

    Different species occupy different frequency bands. Mixing at the frequency
    bin level creates more realistic multi-species spectrograms than waveform
    mixing, because each frequency bin comes fully from one recording.

    mel:    (B, 3, n_mels, T_frames) — already normalised mel spectrogram
    labels: (B, n_classes)
    Returns (mixed_mel, max_labels) — max labels = union of both clips' species.
    """
    B = mel.shape[0]
    if B < 2:
        return mel, labels
    idx  = torch.randperm(B, device=mel.device)
    # Binary mask: (1, 1, n_mels, 1) broadcast over batch, channels, time
    mask = (torch.rand(mel.shape[2], device=mel.device) > 0.5).view(1, 1, -1, 1)
    mixed = torch.where(mask, mel[idx], mel)
    return mixed, torch.max(labels, labels[idx])


def audio_mixup(x: torch.Tensor, y: torch.Tensor) -> tuple:
    """MixUp on raw audio with fixed lam=0.5 (1st place BirdCLEF 2025).

    Key insight from 1st place: variable lam near 0 or 1 suppresses meaningful
    signal. Constant 0.5 ensures both clips always contribute equally.
    Labels take union (max) because all species from both clips are present.
    """
    B = x.shape[0]
    idx = torch.randperm(B, device=x.device)
    mixed_x = 0.5 * x + 0.5 * x[idx]
    mixed_y = torch.max(y, y[idx])
    return mixed_x, mixed_y


# ── Dataset ───────────────────────────────────────────────────────────────────

SR          = 32_000
CLIP_SAMPLES = SR * 5
NUM_CLASSES = 234


def load_audio_clip(path: str, sr: int = SR, n_samples: int = None) -> np.ndarray:
    """Load audio, pad/trim to n_samples, absmax normalize."""
    if n_samples is None:
        n_samples = CLIP_SAMPLES  # read global at call time (not definition time)
    try:
        audio, orig_sr = sf.read(path, dtype='float32', always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if orig_sr != sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    except Exception:
        return np.zeros(n_samples, dtype=np.float32)
    if len(audio) < n_samples:
        audio = np.pad(audio, (0, n_samples - len(audio)))
    else:
        start = np.random.randint(0, len(audio) - n_samples + 1)
        audio = audio[start:start + n_samples]
    return absmax_normalize(audio.astype(np.float32))


def load_ss_clip(path: str, offset_sec: int, sr: int = SR,
                  n_samples: int = None) -> np.ndarray:
    """Load soundscape clip by end-time offset, absmax normalize."""
    if n_samples is None:
        n_samples = CLIP_SAMPLES  # read global at call time (not definition time)
    try:
        start_sample = max(0, offset_sec - n_samples // sr) * sr
        audio, orig_sr = sf.read(path, start=start_sample, frames=n_samples * 2,
                                  dtype='float32', always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if orig_sr != sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    except Exception:
        return np.zeros(n_samples, dtype=np.float32)
    if len(audio) < n_samples:
        audio = np.pad(audio, (0, n_samples - len(audio)))
    else:
        audio = audio[:n_samples]
    return absmax_normalize(audio.astype(np.float32))


def load_ss_clip_by_start(path: str, start_sec: int, sr: int = SR,
                           n_samples: int = None) -> np.ndarray:
    """Load soundscape clip by start time, absmax normalize."""
    if n_samples is None:
        n_samples = CLIP_SAMPLES  # read global at call time (not definition time)
    try:
        audio, orig_sr = sf.read(path, start=start_sec * sr, frames=n_samples * 2,
                                  dtype='float32', always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if orig_sr != sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    except Exception:
        return np.zeros(n_samples, dtype=np.float32)
    if len(audio) < n_samples:
        audio = np.pad(audio, (0, n_samples - len(audio)))
    else:
        audio = audio[:n_samples]
    return absmax_normalize(audio.astype(np.float32))


class TrainAudioDataset(Dataset):
    """Dataset for train_audio clips with weak per-clip labels."""

    def __init__(self, df: pd.DataFrame, audio_dir: str, species_cols: list,
                 augment: bool = True):
        self.df          = df.reset_index(drop=True)
        self.audio_dir   = audio_dir
        self.species_cols = species_cols
        self.augment     = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        path     = os.path.join(self.audio_dir, str(row['filename']))
        audio    = load_audio_clip(path)
        label    = np.zeros(NUM_CLASSES, dtype=np.float32)

        # Primary label
        if str(row.get('primary_label', '')) in self.species_cols:
            label[self.species_cols.index(str(row['primary_label']))] = 1.0

        # Secondary labels
        sec = str(row.get('secondary_labels', ''))
        if sec and sec not in ('[]', 'nan', ''):
            import re
            for sp in re.split(r"[;,\[\]'\s]+", sec):
                sp = sp.strip()
                if sp in self.species_cols:
                    label[self.species_cols.index(sp)] = 0.5  # soft secondary

        return torch.from_numpy(audio), torch.from_numpy(label)


class PseudoSoundscapeDataset(Dataset):
    """Pseudo-labeled soundscape dataset — 1st place BirdCLEF 2025 design.

    Key features:
    - Random interval selection: randomly picks a clip-sized interval from each
      soundscape instead of fixed stride, providing more augmentation.
    - Max-pool labels: pseudo label = max over all pseudo windows covered by the
      random interval, matching how 1st place handles multi-frame aggregation.
    - WeightedRandomSampler support: soundscapes with higher total confidence
      (sum of per-soundscape max class probs) are sampled more frequently,
      giving preference to high-quality pseudo labels.
    """

    def __init__(self, pseudo_df: pd.DataFrame, ss_dir: str, species_cols: list):
        self.ss_dir       = ss_dir
        self.species_cols = species_cols

        # Group by soundscape, build per-soundscape structure
        pseudo_df = pseudo_df.copy()
        pseudo_df['_fname']  = pseudo_df['row_id'].apply(
            lambda r: str(r).rsplit('_', 1)[0] + '.ogg')
        pseudo_df['_offset'] = pseudo_df['row_id'].apply(
            lambda r: int(str(r).rsplit('_', 1)[1]) if str(r).rsplit('_', 1)[-1].isdigit() else 5)

        self._soundscapes = []   # list of (path, offsets_array, probs_array)
        self._weights     = []

        for fname, grp in pseudo_df.groupby('_fname'):
            grp     = grp.sort_values('_offset').reset_index(drop=True)
            offsets = grp['_offset'].values                              # (N,) end-times
            probs   = grp[species_cols].values.astype(np.float32)        # (N, 234)
            path    = os.path.join(ss_dir, fname)
            # Weight = sum of per-window max class prob (1st place WeightedRandomSampler)
            weight  = float(probs.max(axis=1).sum())
            self._soundscapes.append((path, offsets, probs))
            self._weights.append(max(weight, 1e-6))

    def __len__(self):
        return len(self._soundscapes)

    def get_weights(self):
        """Return per-soundscape sampling weights for WeightedRandomSampler."""
        return self._weights

    def __getitem__(self, idx):
        path, offsets, probs = self._soundscapes[idx]
        clip_dur = CLIP_SAMPLES // SR

        # Random start: [0, max_start] where max_start keeps clip within soundscape
        max_offset_end = int(offsets.max())
        max_start = max(0, max_offset_end - clip_dur)
        start_sec = np.random.randint(0, max_start + 1)

        audio = load_ss_clip_by_start(path, start_sec)

        # Max-pool pseudo labels across all windows covered by [start_sec, start_sec+clip_dur]
        # offset is end-time: window i covers [offset-5, offset]
        clip_end = start_sec + clip_dur
        covered  = [(i, o) for i, o in enumerate(offsets)
                    if o > start_sec and (o - 5) < clip_end]
        if covered:
            label = probs[[i for i, _ in covered]].max(axis=0)
        else:
            # Fallback: nearest window
            nearest = int(np.argmin(np.abs(offsets - (start_sec + clip_dur // 2))))
            label   = probs[nearest]

        return torch.from_numpy(audio), torch.from_numpy(label.astype(np.float32))


class SoundscapeValDataset(Dataset):
    """Validation dataset: labeled soundscape clips for OOF AUC."""

    def __init__(self, df: pd.DataFrame, ss_dir: str, species_cols: list):
        """df: expanded (row_id, offset_sec, label_vector)."""
        self.df           = df.reset_index(drop=True)
        self.ss_dir       = ss_dir
        self.species_cols = species_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        audio = load_ss_clip(
            os.path.join(self.ss_dir, str(row['filename'])),
            int(row.get('end', 5))
        )
        label = np.array([row[sc] for sc in self.species_cols], dtype=np.float32)
        return torch.from_numpy(audio), torch.from_numpy(label)


# ── Loss ──────────────────────────────────────────────────────────────────────

class FocalBCE(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )
        pt  = torch.exp(-bce)
        return ((1 - pt) ** self.gamma * bce).mean()


# ── Build expanded soundscape validation DataFrame ────────────────────────────

def hhmmss_to_sec(t) -> int:
    """Convert HH:MM:SS or integer to seconds."""
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
    """Convert soundscape labels to per-row_id format with multi-hot label vectors."""
    sp2idx = {sp: i for i, sp in enumerate(species_cols)}
    rows = []
    for filename, grp in sc_labels.groupby('filename'):
        for _, row in grp.iterrows():
            end   = hhmmss_to_sec(row.get('end', 5))
            label = np.zeros(NUM_CLASSES, dtype=np.float32)
            # primary_label may be semicolon-separated
            for sp in str(row.get('primary_label', '')).split(';'):
                sp = sp.strip()
                if sp in sp2idx:
                    label[sp2idx[sp]] = 1.0
            row_dict = {'filename': filename, 'end': end}
            for j, sc in enumerate(species_cols):
                row_dict[sc] = label[j]
            rows.append(row_dict)
    return pd.DataFrame(rows)


# ── Training loop ─────────────────────────────────────────────────────────────

def macro_auc(y_true, y_score):
    keep = y_true.sum(axis=0) > 0
    if keep.sum() == 0:
        return 0.0
    return roc_auc_score(y_true[:, keep], y_score[:, keep], average='macro')


def train_fold(fold: int, cfg: dict, device: torch.device) -> dict:
    global CLIP_SAMPLES
    t_cfg   = cfg['training']
    d_cfg   = cfg['data']
    m_cfg   = cfg.get('model', {})
    clip_dur = m_cfg.get('clip_duration', 5)
    CLIP_SAMPLES = SR * clip_dur
    out_dir = Path(cfg['output']['dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    train_df    = pd.read_csv(d_cfg['train_csv'])
    sc_labels   = pd.read_csv(d_cfg['soundscape_labels_csv'])
    taxonomy    = pd.read_csv(d_cfg['taxonomy_csv'])
    species_cols = taxonomy['primary_label'].astype(str).tolist()

    # Soundscape GroupKFold (same split as SSM)
    sc_files = sc_labels['filename'].unique()
    gkf      = GroupKFold(n_splits=d_cfg.get('n_folds', 5))
    sc_groups = [f.split('_')[2] for f in sc_files]  # file_id as group
    fold_splits = list(gkf.split(sc_files, groups=sc_groups))
    _, val_idx  = fold_splits[fold]
    val_files   = set(sc_files[val_idx])
    train_sc_files = set(sc_files) - val_files

    # Build val soundscape df
    sc_val_raw = sc_labels[sc_labels['filename'].isin(val_files)]
    sc_val_df  = build_ss_val_df(sc_val_raw, species_cols)
    print(f"Fold {fold}: val_sc_files={len(val_files)}, val_rows={len(sc_val_df)}")

    # Train: ALL train_audio (no fold split — consistent with noisy student)
    audio_dir = d_cfg['audio_dir']
    audio_ds  = TrainAudioDataset(train_df, audio_dir, species_cols, augment=True)
    print(f"  train_audio clips: {len(audio_ds)}")

    # Pseudo soundscape train set
    pseudo_ds = None
    pseudo_csv = d_cfg.get('pseudo_labels_csv')
    if pseudo_csv and os.path.exists(pseudo_csv):
        pseudo_df = pd.read_csv(pseudo_csv)
        # Exclude val files from pseudo set
        def _fname(rid): return str(rid).rsplit('_', 1)[0] + '.ogg'
        mask = pseudo_df['row_id'].apply(lambda r: _fname(r) not in val_files)
        pseudo_df = pseudo_df[mask].reset_index(drop=True)
        pseudo_ds = PseudoSoundscapeDataset(pseudo_df, d_cfg['soundscape_dir'], species_cols)
        print(f"  pseudo soundscape windows: {len(pseudo_ds)}")

    # Val dataset
    val_ds = SoundscapeValDataset(sc_val_df, d_cfg['soundscape_dir'], species_cols)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    bs = t_cfg.get('batch_size', 32)
    audio_loader  = DataLoader(audio_ds,  batch_size=bs, shuffle=True,
                               num_workers=4, pin_memory=True, drop_last=True)
    pseudo_loader = None
    if pseudo_ds and len(pseudo_ds) > 0:
        from torch.utils.data import WeightedRandomSampler
        _w = pseudo_ds.get_weights()
        _sampler = WeightedRandomSampler(weights=_w, num_samples=len(pseudo_ds),
                                         replacement=True)
        pseudo_loader = DataLoader(pseudo_ds, batch_size=bs, sampler=_sampler,
                                   num_workers=2, pin_memory=True, drop_last=True)
    val_loader    = DataLoader(val_ds, batch_size=bs, shuffle=False,
                               num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
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

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = t_cfg.get('learning_rate', 1e-3),
        weight_decay = t_cfg.get('weight_decay',  1e-4),
    )
    epochs = t_cfg.get('epochs', 30)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    gamma           = t_cfg.get('focal_gamma', 2.0)
    mixup_a         = t_cfg.get('mixup_alpha', 0.15)          # 1st place: 0.15
    pseudo_mixup_a  = t_cfg.get('pseudo_mixup_alpha', 0.15)   # MixUp on pseudo data too
    pseudo_w        = t_cfg.get('pseudo_weight', 1.0)         # 1st place: equal weight
    use_sumix_freq  = t_cfg.get('use_sumix_freq', False)      # 1st place: SumixFreq
    criterion       = FocalBCE(gamma=gamma)

    best_auc        = 0.0
    best_state      = None
    history         = []
    oof_logits      = np.zeros((len(val_ds), NUM_CLASSES), dtype=np.float32)
    patience        = t_cfg.get('early_stopping_patience', 7)
    no_improve_cnt  = 0

    # ── WandB ─────────────────────────────────────────────────────────────────
    run = None
    if _WANDB_AVAILABLE:
        run = wandb.init(
            project = 'birdclef-2026',
            name    = f"{cfg['experiment']['name']}-fold{fold}",
            group   = cfg['experiment']['name'],
            tags    = ['sed-ns', f"round{cfg['experiment'].get('round', 1)}", f"fold{fold}"],
            config  = {**cfg, 'fold': fold},
            reinit  = True,
        )

    print(f"  Training fold {fold} for {epochs} epochs  early_stop={patience}  on {device}")

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        n_steps = 0

        # 1st place: combine labeled + pseudo, MixUp on RAW AUDIO with fixed lam=0.5
        pseudo_iter = iter(pseudo_loader) if pseudo_loader else None
        for audio_waves, audio_labels in audio_loader:
            audio_waves  = audio_waves.to(device)
            audio_labels = audio_labels.to(device)

            if pseudo_iter:
                try:
                    pseudo_waves, pseudo_labels = next(pseudo_iter)
                except StopIteration:
                    pseudo_iter = iter(pseudo_loader)
                    pseudo_waves, pseudo_labels = next(pseudo_iter)
                pseudo_waves  = pseudo_waves.to(device)
                pseudo_labels = pseudo_labels.to(device)

                # Concatenate → audio-level MixUp with fixed lam=0.5 (1st place)
                combined_audio  = torch.cat([audio_waves, pseudo_waves], dim=0)
                combined_labels = torch.cat([audio_labels, pseudo_labels], dim=0)
                combined_audio, combined_labels = audio_mixup(combined_audio, combined_labels)
            else:
                combined_audio, combined_labels = audio_mixup(audio_waves, audio_labels)

            # Mel + SpecAugment AFTER audio MixUp
            with torch.no_grad():
                combined_mel = mel_tf(combined_audio)
            combined_mel = spec_aug(combined_mel)

            # SumixFreq AFTER mel (1st place): per-freq-bin random selection
            if use_sumix_freq:
                combined_mel, combined_labels = sumix_freq(combined_mel, combined_labels)

            with torch.cuda.amp.autocast():
                out  = model(combined_mel)
                loss = criterion(out['clipwise_logit'], combined_labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            ep_loss += loss.item()
            n_steps += 1

        sched.step()
        avg_loss = ep_loss / max(n_steps, 1)

        # Validation
        model.eval()
        val_logits_ep = []
        val_labels_ep = []
        with torch.no_grad():
            for waves, labels in val_loader:
                mel  = mel_tf(waves.to(device))
                out  = model(mel)
                val_logits_ep.append(out['clipwise_logit'].cpu().numpy())
                val_labels_ep.append(labels.numpy())

        vl  = np.concatenate(val_logits_ep)
        vla = np.concatenate(val_labels_ep)
        vp  = 1.0 / (1.0 + np.exp(-vl))
        auc = macro_auc(vla, vp)

        print(f"  Ep {ep:3d}/{epochs}  loss={avg_loss:.4f}  ss_auc={auc:.4f}")
        history.append({'epoch': ep, 'loss': avg_loss, 'val_auc': auc})

        if run is not None:
            run.log({'epoch': ep, 'train/loss': avg_loss, 'val/ss_auc': auc,
                     'val/best_auc': best_auc})

        if auc > best_auc:
            best_auc       = auc
            no_improve_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            oof_logits = vl
            # Save checkpoint
            torch.save({
                'state_dict':   best_state,
                'fold':         fold,
                'best_val_auc': best_auc,
                'epoch':        ep,
            }, out_dir / f'fold{fold}_best.pt')
            print(f"    ✓ New best AUC={best_auc:.4f}")
            if run is not None:
                run.summary['best_val_auc'] = best_auc
                run.summary['best_epoch']   = ep
        else:
            no_improve_cnt += 1
            if no_improve_cnt >= patience:
                print(f"  Early stopping at epoch {ep} (no improvement for {patience} epochs)")
                break

    if run is not None:
        run.finish()

    return {'fold': fold, 'best_auc': best_auc, 'history': history,
            'oof_logits': oof_logits}


# ── Inference on all soundscapes ──────────────────────────────────────────────

def infer_all_soundscapes(cfg: dict, device: torch.device):
    """Run 5-fold ensemble inference on all soundscapes, save all_ss_probs.npz."""
    global CLIP_SAMPLES
    d_cfg   = cfg['data']
    m_cfg   = cfg.get('model', {})
    clip_dur = m_cfg.get('clip_duration', 5)
    CLIP_SAMPLES = SR * clip_dur
    out_dir = Path(cfg['output']['dir'])
    ss_dir  = Path(d_cfg['soundscape_dir'])
    taxonomy = pd.read_csv(d_cfg['taxonomy_csv'])
    species_cols = taxonomy['primary_label'].astype(str).tolist()

    ogg_files = sorted(ss_dir.glob('*.ogg'))
    print(f"\nInference on {len(ogg_files)} soundscapes ...")

    mel_tf = MelTransform(**{k: v for k, v in m_cfg.items()
                              if k in ('sr', 'n_mels', 'n_fft', 'hop_length',
                                       'fmin', 'fmax', 'top_db', 'power',
                                       'norm', 'mel_scale', 'peak_norm')}).to(device)

    # Load all fold models
    fold_models = []
    for fold in range(cfg['data'].get('n_folds', 5)):
        ckpt_path = out_dir / f'fold{fold}_best.pt'
        if not ckpt_path.exists():
            print(f"  Skip fold {fold}: {ckpt_path} not found")
            continue
        model = SEDModel(
            backbone       = m_cfg.get('backbone', 'tf_efficientnet_b0.ns_jft_in1k'),
            num_classes    = NUM_CLASSES,
            dropout        = 0.0,
        ).to(device)
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        fold_models.append(model)
    print(f"Loaded {len(fold_models)} fold models for ensemble")

    all_row_ids = []
    all_probs   = []

    for ogg_path in tqdm(ogg_files, desc='Soundscape inference'):
        ss_id = ogg_path.stem
        try:
            audio, _ = sf.read(str(ogg_path), dtype='float32', always_2d=False)
            if audio.ndim == 2: audio = audio.mean(axis=1)
        except Exception:
            continue

        STRIDE = SR * 5   # always 5s stride → 12 rows per 60s soundscape
        n_clips = min(len(audio) // STRIDE, 12)
        if n_clips == 0:
            continue

        for ci in range(n_clips):
            start = ci * STRIDE
            clip  = audio[start:start + CLIP_SAMPLES]
            if len(clip) < CLIP_SAMPLES:
                clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
            wav  = torch.from_numpy(clip[None]).to(device)
            with torch.no_grad():
                mel = mel_tf(wav)
                probs_acc = np.zeros(NUM_CLASSES, dtype=np.float32)
                for mdl in fold_models:
                    out = mdl(mel)
                    probs_acc += torch.sigmoid(out['clipwise_logit']).cpu().numpy()[0]
                probs_acc /= len(fold_models)

            offset = (ci + 1) * 5
            all_row_ids.append(f"{ss_id}_{offset}")
            all_probs.append(probs_acc)

    all_probs = np.stack(all_probs, axis=0)
    out_path  = out_dir / 'all_ss_probs.npz'
    np.savez_compressed(str(out_path),
                        row_ids=np.array(all_row_ids),
                        probs=all_probs)
    print(f"Saved {len(all_row_ids)} rows → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--fold',   type=int, default=None,
                        help='Single fold (default: all folds)')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--infer_all_ss', action='store_true',
                        help='After training, run inference on all soundscapes')
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Experiment: {cfg['experiment']['name']}")

    out_dir = Path(cfg['output']['dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    # If --infer_all_ss is passed standalone (no --fold), skip training entirely
    if args.infer_all_ss and args.fold is None:
        infer_all_soundscapes(cfg, device)
        return

    n_folds  = cfg['data'].get('n_folds', 5)
    folds    = [args.fold] if args.fold is not None else list(range(n_folds))

    all_results = []
    all_oof_logits = {}

    for fold in folds:
        print(f"\n{'='*60}")
        print(f"  Fold {fold}/{n_folds-1}")
        print(f"{'='*60}")
        result = train_fold(fold, cfg, device)
        all_results.append(result)
        all_oof_logits[fold] = result['oof_logits']

    # Summary
    if len(all_results) == n_folds:
        mean_auc = np.mean([r['best_auc'] for r in all_results])
        print(f"\n{'='*60}")
        print(f"  Results (all folds):")
        for r in all_results:
            print(f"    Fold {r['fold']}: best_auc={r['best_auc']:.4f}")
        print(f"  Mean fold AUC: {mean_auc:.4f}")

        result_path = out_dir / 'result.json'
        with open(result_path, 'w') as f:
            json.dump({'mean_fold_auc': mean_auc,
                       'folds': [{'fold': r['fold'], 'best_auc': r['best_auc']}
                                  for r in all_results]}, f, indent=2)

        # Save OOF predictions
        oof_path = out_dir / 'oof_predictions.npz'
        np.savez_compressed(str(oof_path),
                            logits=np.concatenate(list(all_oof_logits.values())))
        print(f"  OOF saved → {oof_path}")

    # Optionally infer all soundscapes
    if args.infer_all_ss or cfg.get('output', {}).get('infer_all_ss', False):
        infer_all_soundscapes(cfg, device)


if __name__ == '__main__':
    main()
