"""EfficientNet-B0 + Bidirectional SSM — Noisy Student training for BirdCLEF 2026.

Architecture:
  raw audio clip  ->  Mel(224)  ->  EfficientNet-B0 (global_pool='avg')  ->  (d_feat=1280)
  Stack T clips   ->  (B, T, 1280)  ->  Linear proj  ->  BiSSM x N  ->  (B, T, n_classes)

Training data per fold:
  - ALL train_audio clips  (weak-labeled, T=1, no temporal context - trains backbone)
  - Pseudo soundscape sequences (12-window sequences, T=12, trains SSM temporal layers)

Validation:
  - Labeled soundscape fold k -> per-window AUC across 12-window sequences

Usage:
  python train_ssm_ns.py --config configs/ssm_ns_b0_r1.yaml [--fold 0]
  CUDA_VISIBLE_DEVICES=1 python train_ssm_ns.py --config configs/ssm_ns_b0_r1.yaml --fold 0
"""

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as TAT
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


# ── Selective SSM (Mamba-style) ────────────────────────────────────────────────

class SelectiveSSM(nn.Module):
    """Simplified bidirectional Mamba-style SSM block."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj  = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d   = nn.Conv1d(d_model, d_model, d_conv,
                                  padding=d_conv - 1, groups=d_model)
        self.dt_proj  = nn.Linear(d_model, d_model, bias=True)
        self.B_proj   = nn.Linear(d_model, d_state, bias=False)
        self.C_proj   = nn.Linear(d_model, d_state, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.D        = nn.Parameter(torch.ones(d_model))
        A = torch.arange(1, d_state + 1, dtype=torch.float32
                         ).unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_sz, T, D = x.shape
        xz         = self.in_proj(x)
        x_ssm, z   = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)
        dt    = F.softplus(self.dt_proj(x_conv))
        B_mat = self.B_proj(x_conv)
        C_mat = self.C_proj(x_conv)
        A     = -torch.exp(self.A_log)
        y = self._scan(x_conv, dt, A, B_mat, C_mat)
        y = y * F.silu(z)
        return self.out_proj(y)

    def _scan(self, x, dt, A, B, C):
        batch, T, D = x.shape
        N = self.d_state
        h  = torch.zeros(batch, D, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            dt_t = dt[:, t, :, None]
            dA   = torch.exp(A[None] * dt_t)
            dB   = dt_t * B[:, t, None, :]
            h    = h * dA + x[:, t, :, None] * dB
            y_t  = (h * C[:, t, None, :]).sum(-1)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)
        return y + x * self.D[None, None, :]


# ── EfficientSSM model ─────────────────────────────────────────────────────────

class EfficientSSM(nn.Module):
    """EfficientNet-B0 frontend + Bidirectional SSM temporal model.

    forward(wavs, mel_tf, spec_aug=None):
      wavs: (B, T, CLIP_SAMPLES)  T=1 for train_audio, T=12 for soundscapes
      returns: logits (B, T, n_classes)
    """

    def __init__(
        self,
        backbone:       str   = 'tf_efficientnet_b0.ns_jft_in1k',
        d_model:        int   = 256,
        d_state:        int   = 16,
        n_ssm_layers:   int   = 2,
        n_classes:      int   = 234,
        n_windows:      int   = 12,
        dropout:        float = 0.1,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.n_classes = n_classes

        # EfficientNet-B0: global avg pool -> (B, d_feat=1280)
        self.backbone = timm.create_model(
            backbone, pretrained=True, in_chans=3,
            num_classes=0, global_pool='avg',
            drop_path_rate=drop_path_rate,
        )
        d_feat = self.backbone.num_features  # 1280 for B0

        self.input_proj = nn.Sequential(
            nn.Linear(d_feat, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)

        self.ssm_fwd   = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(n_ssm_layers)])
        self.ssm_bwd   = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(n_ssm_layers)])
        self.ssm_merge = nn.ModuleList([nn.Linear(2 * d_model, d_model) for _ in range(n_ssm_layers)])
        self.ssm_norm  = nn.ModuleList([nn.LayerNorm(d_model)           for _ in range(n_ssm_layers)])
        self.ssm_drop  = nn.Dropout(dropout)

        # Linear classification head
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, wavs, mel_tf, spec_aug=None):
        B, T, S = wavs.shape

        # Extract per-clip features in parallel
        with torch.no_grad():
            mel = mel_tf(wavs.reshape(B * T, S))       # (B*T, 3, n_mels, T_frames)
        if spec_aug is not None:
            mel = spec_aug(mel)
        feat = self.backbone(mel).reshape(B, T, -1)    # (B, T, d_feat)

        h = self.input_proj(feat) + self.pos_enc[:, :T, :]

        for fwd, bwd, merge, norm in zip(
            self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm
        ):
            residual = h
            h_f = fwd(h)
            h_b = bwd(h.flip(1)).flip(1)
            h   = merge(torch.cat([h_f, h_b], dim=-1))
            h   = self.ssm_drop(h)
            h   = norm(h + residual)

        return self.classifier(h)   # (B, T, n_classes)


# ── Mel + augmentation ─────────────────────────────────────────────────────────

class MelTransform(nn.Module):
    def __init__(self, sr=32_000, n_mels=224, n_fft=2048, hop_length=512,
                 fmin=0, fmax=16_000, top_db=80.0, power=2.0,
                 norm='slaney', mel_scale='htk', peak_norm=False):
        super().__init__()
        self.peak_norm = peak_norm
        self.mel = TAT.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=fmin, f_max=fmax,
            power=power, norm=norm, mel_scale=mel_scale,
        )
        self.db = TAT.AmplitudeToDB(stype='power', top_db=top_db)

    @torch.no_grad()
    def forward(self, waveforms):
        waveforms = torch.nan_to_num(waveforms.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if self.peak_norm:
            peak = waveforms.abs().amax(dim=1, keepdim=True).clamp(min=1e-7)
            waveforms = waveforms / peak
        mel  = torch.nan_to_num(self.db(self.mel(waveforms)), nan=-80.0)
        B    = mel.shape[0]
        flat = mel.reshape(B, -1)
        mn   = flat.min(1, keepdim=True)[0].unsqueeze(-1)
        mx   = flat.max(1, keepdim=True)[0].unsqueeze(-1)
        mel  = torch.nan_to_num((mel - mn) / (mx - mn + 1e-7), nan=0.0)
        return mel.unsqueeze(1).repeat(1, 3, 1, 1)


class SpecAug(nn.Module):
    def __init__(self, freq_mask_param=24, time_mask_param=32, n_freq=2, n_time=2):
        super().__init__()
        self.freq_masks = nn.ModuleList([
            TAT.FrequencyMasking(freq_mask_param, iid_masks=True) for _ in range(n_freq)
        ])
        self.time_masks = nn.ModuleList([
            TAT.TimeMasking(time_mask_param, iid_masks=True) for _ in range(n_time)
        ])

    def forward(self, x):
        for m in self.freq_masks:
            x = m(x)
        for m in self.time_masks:
            x = m(x)
        return x


# ── Audio loading helpers ──────────────────────────────────────────────────────

SR           = 32_000
CLIP_SAMPLES = SR * 5
NUM_CLASSES  = 234
N_WINDOWS    = 12


def load_audio_clip(path: str, sr: int = SR, n_samples: int = CLIP_SAMPLES) -> np.ndarray:
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
    return audio.astype(np.float32)


def load_ss_clip(path: str, offset_sec: int, sr: int = SR,
                 n_samples: int = CLIP_SAMPLES) -> np.ndarray:
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
    return audio.astype(np.float32)


# ── Datasets ───────────────────────────────────────────────────────────────────

class TrainAudioDataset(Dataset):
    """Single clips from train_audio with weak labels, returned as T=1 sequences."""

    def __init__(self, df: pd.DataFrame, audio_dir: str, species_cols: list):
        self.df        = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.sp2idx    = {sp: i for i, sp in enumerate(species_cols)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        path  = os.path.join(self.audio_dir, str(row['filename']))
        audio = load_audio_clip(path)
        label = np.zeros(NUM_CLASSES, dtype=np.float32)
        for sp in str(row.get('primary_label', '')).split(';'):
            sp = sp.strip()
            if sp in self.sp2idx:
                label[self.sp2idx[sp]] = 1.0
        sec = str(row.get('secondary_labels', ''))
        if sec and sec not in ('[]', 'nan', ''):
            for sp in re.split(r"[;,\[\]'\s]+", sec):
                sp = sp.strip()
                if sp in self.sp2idx:
                    label[self.sp2idx[sp]] = 0.5
        # (1, CLIP_SAMPLES), (1, n_classes)
        return torch.from_numpy(audio[None]), torch.from_numpy(label[None])


class SoundscapeSequenceDataset(Dataset):
    """Full 12-window soundscape sequences with pseudo labels.

    Groups pseudo-labeled rows by soundscape file.
    Returns: wavs (T, CLIP_SAMPLES), labels (T, n_classes)
    """

    def __init__(self, pseudo_df: pd.DataFrame, ss_dir: str, species_cols: list):
        self.ss_dir       = ss_dir
        self.species_cols = species_cols

        def _fname(rid):   return str(rid).rsplit('_', 1)[0] + '.ogg'
        def _offset(rid):
            parts = str(rid).rsplit('_', 1)
            return int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 5

        pseudo_df = pseudo_df.copy()
        pseudo_df['_fname']  = pseudo_df['row_id'].apply(_fname)
        pseudo_df['_offset'] = pseudo_df['row_id'].apply(_offset)

        self.sequences = []
        for fname, grp in pseudo_df.groupby('_fname'):
            grp = grp.sort_values('_offset').reset_index(drop=True)
            self.sequences.append({
                'path':    os.path.join(ss_dir, fname),
                'offsets': grp['_offset'].tolist(),
                'labels':  grp[species_cols].values.astype(np.float32),
            })

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq     = self.sequences[idx]
        offsets = seq['offsets']
        labels  = seq['labels']   # (T_actual, n_classes)
        T       = len(offsets)

        clips = np.stack([load_ss_clip(seq['path'], off) for off in offsets], axis=0)

        # Pad to N_WINDOWS so all batches have the same shape
        if T < N_WINDOWS:
            pad_c = np.zeros((N_WINDOWS - T, CLIP_SAMPLES), dtype=np.float32)
            pad_l = np.zeros((N_WINDOWS - T, NUM_CLASSES),  dtype=np.float32)
            clips  = np.concatenate([clips, pad_c], axis=0)
            labels = np.concatenate([labels, pad_l], axis=0)

        return torch.from_numpy(clips[:N_WINDOWS]), torch.from_numpy(labels[:N_WINDOWS])


class SoundscapeValSequenceDataset(Dataset):
    """Labeled soundscape sequences for OOF validation (variable T, batch_size=1)."""

    def __init__(self, df: pd.DataFrame, ss_dir: str, species_cols: list):
        self.ss_dir    = ss_dir
        self.sequences = []
        for fname, grp in df.groupby('filename'):
            grp = grp.sort_values('end').reset_index(drop=True)
            self.sequences.append({
                'path':    os.path.join(ss_dir, fname),
                'offsets': grp['end'].astype(int).tolist(),
                'labels':  grp[species_cols].values.astype(np.float32),
            })

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq     = self.sequences[idx]
        offsets = seq['offsets']
        labels  = seq['labels']
        T       = len(offsets)

        clips = np.stack([load_ss_clip(seq['path'], off) for off in offsets], axis=0)

        # Pad to N_WINDOWS (val uses batch_size=1, but pad for model consistency)
        if T < N_WINDOWS:
            pad_c = np.zeros((N_WINDOWS - T, CLIP_SAMPLES), dtype=np.float32)
            pad_l = np.zeros((N_WINDOWS - T, NUM_CLASSES),  dtype=np.float32)
            clips  = np.concatenate([clips, pad_c], axis=0)
            labels = np.concatenate([labels, pad_l], axis=0)

        # Return actual T so we only score real windows
        return torch.from_numpy(clips[:N_WINDOWS]), torch.from_numpy(labels[:T])


# ── Loss ───────────────────────────────────────────────────────────────────────

class FocalBCE(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt  = torch.exp(-bce)
        return ((1 - pt) ** self.gamma * bce).mean()


# ── Helpers ────────────────────────────────────────────────────────────────────

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
    """Build per-window multi-hot labels, aggregating multiple rows at the same end time."""
    sp2idx = {sp: i for i, sp in enumerate(species_cols)}
    rows = []
    for filename, grp in sc_labels.groupby('filename'):
        # First pass: collect all labels per (filename, end_sec)
        window_labels = {}  # end_sec -> np.array(n_classes)
        for _, row in grp.iterrows():
            end = hhmmss_to_sec(row.get('end', 5))
            if end not in window_labels:
                window_labels[end] = np.zeros(NUM_CLASSES, dtype=np.float32)
            for sp in str(row.get('primary_label', '')).split(';'):
                sp = sp.strip()
                if sp in sp2idx:
                    window_labels[end][sp2idx[sp]] = 1.0
        # One row per unique time window
        for end in sorted(window_labels.keys()):
            label = window_labels[end]
            row_dict = {'filename': filename, 'end': end}
            for j, sc in enumerate(species_cols):
                row_dict[sc] = label[j]
            rows.append(row_dict)
    return pd.DataFrame(rows)


def macro_auc(y_true, y_score):
    keep = y_true.sum(axis=0) > 0
    if keep.sum() == 0:
        return 0.0
    return roc_auc_score(y_true[:, keep], y_score[:, keep], average='macro')


# ── Training ───────────────────────────────────────────────────────────────────

def train_fold(fold: int, cfg: dict, device: torch.device) -> dict:
    global CLIP_SAMPLES
    t_cfg   = cfg['training']
    d_cfg   = cfg['data']
    m_cfg   = cfg.get('model', {})
    clip_dur = m_cfg.get('clip_duration', 5)
    CLIP_SAMPLES = SR * clip_dur
    out_dir = Path(cfg['output']['dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load labels ───────────────────────────────────────────────────────────
    train_df     = pd.read_csv(d_cfg['train_csv'])
    sc_labels    = pd.read_csv(d_cfg['soundscape_labels_csv'])
    taxonomy     = pd.read_csv(d_cfg['taxonomy_csv'])
    species_cols = taxonomy['primary_label'].astype(str).tolist()

    # GroupKFold on soundscape files (same split as SED NS for fair comparison)
    sc_files    = sc_labels['filename'].unique()
    gkf         = GroupKFold(n_splits=d_cfg.get('n_folds', 5))
    sc_groups   = [f.split('_')[2] for f in sc_files]
    fold_splits = list(gkf.split(sc_files, groups=sc_groups))
    _, val_idx  = fold_splits[fold]
    val_files   = set(sc_files[val_idx])

    sc_val_raw = sc_labels[sc_labels['filename'].isin(val_files)]
    sc_val_df  = build_ss_val_df(sc_val_raw, species_cols)
    print(f"Fold {fold}: val_sc_files={len(val_files)}, val_rows={len(sc_val_df)}")

    # Train audio (T=1 clips, trains backbone)
    audio_ds = TrainAudioDataset(train_df, d_cfg['audio_dir'], species_cols)
    print(f"  train_audio clips: {len(audio_ds)}")

    # Pseudo soundscape sequences (T=12, trains SSM temporal layers)
    pseudo_ds  = None
    pseudo_csv = d_cfg.get('pseudo_labels_csv')
    if pseudo_csv and os.path.exists(pseudo_csv):
        pseudo_df = pd.read_csv(pseudo_csv)
        def _fname(rid): return str(rid).rsplit('_', 1)[0] + '.ogg'
        mask      = pseudo_df['row_id'].apply(lambda r: _fname(r) not in val_files)
        pseudo_df = pseudo_df[mask].reset_index(drop=True)
        pseudo_ds = SoundscapeSequenceDataset(pseudo_df, d_cfg['soundscape_dir'], species_cols)
        print(f"  pseudo soundscape sequences: {len(pseudo_ds)}")

    val_ds = SoundscapeValSequenceDataset(sc_val_df, d_cfg['soundscape_dir'], species_cols)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    bs_audio  = t_cfg.get('batch_size', 16)
    bs_pseudo = t_cfg.get('pseudo_batch_size', 4)  # smaller: T=12 clips heavier
    audio_loader = DataLoader(audio_ds, batch_size=bs_audio, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    pseudo_loader = (DataLoader(pseudo_ds, batch_size=bs_pseudo, shuffle=True,
                                num_workers=2, pin_memory=True, drop_last=True)
                     if pseudo_ds and len(pseudo_ds) > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    mel_keys = ('sr', 'n_mels', 'n_fft', 'hop_length', 'fmin', 'fmax',
                'top_db', 'power', 'norm', 'mel_scale', 'peak_norm')
    mel_tf   = MelTransform(**{k: v for k, v in m_cfg.items() if k in mel_keys}).to(device)
    spec_aug = SpecAug(
        freq_mask_param = m_cfg.get('freq_mask', 24),
        time_mask_param = m_cfg.get('time_mask', 32),
    ).to(device)

    model = EfficientSSM(
        backbone       = m_cfg.get('backbone', 'tf_efficientnet_b0.ns_jft_in1k'),
        d_model        = m_cfg.get('d_model', 256),
        d_state        = m_cfg.get('d_state', 16),
        n_ssm_layers   = m_cfg.get('n_ssm_layers', 2),
        n_classes      = NUM_CLASSES,
        n_windows      = m_cfg.get('n_windows', N_WINDOWS),
        dropout        = m_cfg.get('dropout', 0.1),
        drop_path_rate = m_cfg.get('drop_path_rate', 0.0),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  EfficientSSM params: {n_params:,}")

    lr = t_cfg.get('learning_rate', 1e-3)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = lr,
        weight_decay = t_cfg.get('weight_decay', 1e-4),
    )

    epochs       = t_cfg.get('epochs', 40)
    warmup_eps   = t_cfg.get('warmup_epochs', 5)
    def _lr_lambda(ep):
        if ep < warmup_eps:
            return (ep + 1) / warmup_eps
        progress = (ep - warmup_eps) / max(epochs - warmup_eps, 1)
        return 1e-6 / lr + (1 - 1e-6 / lr) * 0.5 * (1 + math.cos(math.pi * progress))
    sched     = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    scaler    = torch.cuda.amp.GradScaler()
    criterion      = FocalBCE(gamma=t_cfg.get('focal_gamma', 2.0))
    pseudo_w       = t_cfg.get('pseudo_weight', 1.0)          # 1st place: equal weight
    pseudo_mixup_a = t_cfg.get('pseudo_mixup_alpha', 0.15)    # MixUp on pseudo sequences too

    best_auc       = 0.0
    best_state     = None
    history        = []
    oof_preds      = None
    patience       = t_cfg.get('early_stopping_patience', 7)
    no_improve_cnt = 0

    # ── WandB ─────────────────────────────────────────────────────────────────
    run = None
    if _WANDB_AVAILABLE:
        run = wandb.init(
            project = 'birdclef-2026',
            name    = f"{cfg['experiment']['name']}-fold{fold}",
            group   = cfg['experiment']['name'],
            tags    = ['ssm-ns', f"round{cfg['experiment'].get('round', 1)}", f"fold{fold}"],
            config  = {**cfg, 'fold': fold},
            reinit  = True,
        )

    print(f"  Training fold {fold} for {epochs} epochs  lr={lr:.1e}  warmup={warmup_eps}ep")

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        n_steps = 0
        t0 = time.time()

        pseudo_iter = iter(pseudo_loader) if pseudo_loader else None

        for wavs, labels in audio_loader:
            # wavs: (B, 1, CLIP_SAMPLES)   labels: (B, 1, n_classes)
            wavs   = wavs.to(device)
            labels = labels.to(device).squeeze(1)          # (B, n_classes)
            with torch.cuda.amp.autocast():
                logits = model(wavs, mel_tf, spec_aug).squeeze(1)  # (B, n_classes)
                loss   = criterion(logits, labels)

            # Pseudo sequences (T=12)
            if pseudo_iter:
                try:
                    p_wavs, p_labels = next(pseudo_iter)
                except StopIteration:
                    pseudo_iter = iter(pseudo_loader)
                    p_wavs, p_labels = next(pseudo_iter)
                p_wavs   = p_wavs.to(device)    # (Bp, T, CLIP_SAMPLES)
                p_labels = p_labels.to(device)  # (Bp, T, n_classes)

                if pseudo_mixup_a > 0 and p_wavs.shape[0] > 1:
                    # Pseudo × pseudo MixUp with max labels
                    lam = float(torch.distributions.Beta(pseudo_mixup_a, pseudo_mixup_a).sample())
                    idx = torch.randperm(p_wavs.shape[0], device=device)
                    p_wavs   = lam * p_wavs   + (1 - lam) * p_wavs[idx]
                    p_labels = torch.max(p_labels, p_labels[idx])  # union

                    # Cross-domain mix: blend labeled clips into pseudo sequences
                    # Broadcast labeled clip (B, CLIP) across T windows of pseudo (Bp, T, CLIP)
                    n_cross = min(p_wavs.shape[0], wavs.shape[0])
                    lam_c = float(torch.distributions.Beta(pseudo_mixup_a, pseudo_mixup_a).sample())
                    lab_clips = wavs[:n_cross, 0, :]                              # (n_cross, CLIP)
                    lab_clips = lab_clips.unsqueeze(1).expand(-1, p_wavs.shape[1], -1)  # (n_cross, T, CLIP)
                    p_wavs[:n_cross]   = lam_c * p_wavs[:n_cross] + (1 - lam_c) * lab_clips
                    lab_lbls = labels[:n_cross].unsqueeze(1).expand(-1, p_wavs.shape[1], -1)  # (n_cross, T, C)
                    p_labels[:n_cross] = torch.max(p_labels[:n_cross], lab_lbls)  # union

                with torch.cuda.amp.autocast():
                    p_logits = model(p_wavs, mel_tf, spec_aug)  # (Bp, T, n_classes)
                    p_loss   = criterion(p_logits, p_labels)
                loss = loss + pseudo_w * p_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ep_loss += loss.item()
            n_steps += 1

        sched.step()
        avg_loss = ep_loss / max(n_steps, 1)

        # Validation: full sequences
        model.eval()
        all_preds  = []
        all_labels = []
        with torch.no_grad():
            for wavs, labels in val_loader:
                # wavs: (1, N_WINDOWS, CLIP_SAMPLES)  labels: (1, T_real, n_classes)
                logits   = model(wavs.to(device), mel_tf)        # (1, N_WINDOWS, n_classes)
                T_real   = labels.shape[1]
                probs    = torch.sigmoid(logits[0, :T_real]).cpu().numpy()  # (T_real, n_classes)
                all_preds.append(probs)
                all_labels.append(labels[0].numpy())

        all_preds  = np.concatenate(all_preds,  axis=0)  # (N_val_windows, n_classes)
        all_labels = np.concatenate(all_labels, axis=0)
        auc        = macro_auc(all_labels, all_preds)

        elapsed = time.time() - t0
        print(f"  Ep {ep:3d}/{epochs}  loss={avg_loss:.4f}  ss_auc={auc:.4f}  ({elapsed:.0f}s)")
        history.append({'epoch': ep, 'loss': avg_loss, 'val_auc': auc})

        if run is not None:
            run.log({'epoch': ep, 'train/loss': avg_loss, 'val/ss_auc': auc,
                     'val/best_auc': best_auc})

        if auc > best_auc:
            best_auc       = auc
            no_improve_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            oof_preds  = all_preds
            torch.save({
                'state_dict':   best_state,
                'fold':         fold,
                'best_val_auc': best_auc,
                'epoch':        ep,
                'config':       cfg,
            }, out_dir / f'fold{fold}_best.pt')
            print(f"    New best AUC={best_auc:.4f}")
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

    if oof_preds is None:
        oof_preds = np.zeros((len(val_ds), NUM_CLASSES), dtype=np.float32)

    return {'fold': fold, 'best_auc': best_auc, 'history': history, 'oof_preds': oof_preds}


# ── All-soundscape inference ───────────────────────────────────────────────────

def infer_all_soundscapes(cfg: dict, device: torch.device):
    """5-fold ensemble on all soundscapes -> all_ss_probs.npz."""
    global CLIP_SAMPLES
    d_cfg   = cfg['data']
    m_cfg   = cfg.get('model', {})
    clip_dur = m_cfg.get('clip_duration', 5)
    CLIP_SAMPLES = SR * clip_dur
    out_dir = Path(cfg['output']['dir'])
    ss_dir  = Path(d_cfg['soundscape_dir'])
    taxonomy     = pd.read_csv(d_cfg['taxonomy_csv'])
    species_cols = taxonomy['primary_label'].astype(str).tolist()

    mel_keys = ('sr', 'n_mels', 'n_fft', 'hop_length', 'fmin', 'fmax',
                'top_db', 'power', 'norm', 'mel_scale', 'peak_norm')
    mel_tf = MelTransform(**{k: v for k, v in m_cfg.items() if k in mel_keys}).to(device)

    fold_models = []
    for fold in range(d_cfg.get('n_folds', 5)):
        ckpt_path = out_dir / f'fold{fold}_best.pt'
        if not ckpt_path.exists():
            print(f"  Skip fold {fold}: not found")
            continue
        model = EfficientSSM(
            backbone     = m_cfg.get('backbone', 'tf_efficientnet_b0.ns_jft_in1k'),
            d_model      = m_cfg.get('d_model', 256),
            d_state      = m_cfg.get('d_state', 16),
            n_ssm_layers = m_cfg.get('n_ssm_layers', 2),
            n_classes    = NUM_CLASSES,
            n_windows    = m_cfg.get('n_windows', N_WINDOWS),
            dropout      = 0.0,
        ).to(device)
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        fold_models.append(model)
    print(f"Loaded {len(fold_models)} fold models")

    ogg_files   = sorted(ss_dir.glob('*.ogg'))
    all_row_ids = []
    all_probs   = []

    for ogg_path in tqdm(ogg_files, desc='SSM inference'):
        ss_id = ogg_path.stem
        try:
            audio, _ = sf.read(str(ogg_path), dtype='float32', always_2d=False)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
        except Exception:
            continue
        STRIDE  = SR * 5   # always 5s stride → 12 rows per 60s soundscape
        n_clips = min(len(audio) // STRIDE, N_WINDOWS)
        if n_clips == 0:
            continue

        def _window(ci):
            start = ci * STRIDE
            seg   = audio[start:start + CLIP_SAMPLES]
            if len(seg) < CLIP_SAMPLES:
                seg = np.pad(seg, (0, CLIP_SAMPLES - len(seg)))
            return seg.astype(np.float32)

        clips = np.stack([_window(ci) for ci in range(n_clips)], axis=0)
        wavs  = torch.from_numpy(clips[None]).to(device)  # (1, T, CLIP_SAMPLES)
        with torch.no_grad():
            acc = np.zeros((n_clips, NUM_CLASSES), dtype=np.float32)
            for mdl in fold_models:
                logits = mdl(wavs, mel_tf)
                acc   += torch.sigmoid(logits[0]).cpu().numpy()
            acc /= len(fold_models)

        for ci in range(n_clips):
            all_row_ids.append(f"{ss_id}_{(ci+1)*5}")
            all_probs.append(acc[ci])

    all_probs = np.stack(all_probs, axis=0)
    out_path  = out_dir / 'all_ss_probs.npz'
    np.savez_compressed(str(out_path),
                        row_ids=np.array(all_row_ids), probs=all_probs)
    print(f"Saved {len(all_row_ids)} rows -> {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',       required=True)
    parser.add_argument('--fold',         type=int, default=None)
    parser.add_argument('--device',       default='cuda:1')
    parser.add_argument('--infer_all_ss', action='store_true')
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Config: {args.config}  Device: {device}")
    print(f"Experiment: {cfg['experiment']['name']}")

    out_dir = Path(cfg['output']['dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.infer_all_ss:
        infer_all_soundscapes(cfg, device)
        return

    n_folds = cfg['data'].get('n_folds', 5)
    folds   = [args.fold] if args.fold is not None else list(range(n_folds))

    all_results = []
    for fold in folds:
        print(f"\n{'='*60}\n  Fold {fold}/{n_folds-1}\n{'='*60}")
        result = train_fold(fold, cfg, device)
        all_results.append(result)

    if len(all_results) == n_folds:
        mean_auc = np.mean([r['best_auc'] for r in all_results])
        print(f"\nMean fold AUC: {mean_auc:.4f}")
        for r in all_results:
            print(f"  Fold {r['fold']}: {r['best_auc']:.4f}")
        with open(out_dir / 'result.json', 'w') as f:
            json.dump({'mean_fold_auc': mean_auc,
                       'folds': [{'fold': r['fold'], 'best_auc': r['best_auc']}
                                  for r in all_results]}, f, indent=2)

    if cfg.get('output', {}).get('infer_all_ss', False):
        infer_all_soundscapes(cfg, device)


if __name__ == '__main__':
    main()
