"""EfficientNet-B0 SED — Semi-Supervised Mean Teacher (train_audio + soundscapes).

Architecture:
  - Teacher: EMA of student (initialised from competitor SED for warm start)
  - Student: EfficientNet-B0 (initialised from competitor SED)

Loss per step:
  1. Labeled (train_audio)   : FocalBCE(student_logit, hard_label)
  2. Unlabeled (soundscapes) : consistency = BCE(student_prob_strong, teacher_prob_weak)
     Only applied where teacher max-prob >= confidence_threshold.

Total = labeled_loss + ramp(ep) * consistency_weight * consistency_loss

Augmentation strategy:
  - Weak  (teacher path): absmax normalize + mel (no aug)
  - Strong (student path): SpecAug + SumixFreq applied on mel

Ramp-up: consistency weight linearly increases from 0 → max over rampup_epochs.
Teacher EMA: θ_T ← α * θ_T + (1-α) * θ_S  every step.

Usage:
    CUDA_VISIBLE_DEVICES=1 python train_ssl_mean_teacher.py \
        --config configs/ssl_mean_teacher_b0_v1.yaml
"""

import argparse
import copy
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
from torch.utils.data import Dataset, DataLoader, IterableDataset

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


# ── Model (identical to train_distill_competitor.py) ──────────────────────────

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
                 num_classes=234, dropout=0.1, drop_path_rate=0.0, gem_p_init=3.0):
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


# ── Mel + Augmentation ────────────────────────────────────────────────────────

class MelTransform(nn.Module):
    def __init__(self, sr=SR, n_mels=224, n_fft=2048, hop_length=512,
                 fmin=0, fmax=16_000, top_db=80.0, power=2.0,
                 norm='slaney', mel_scale='htk', **_):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=fmin, f_max=fmax,
            power=power, norm=norm, mel_scale=mel_scale,
        )
        self.db = T.AmplitudeToDB(stype='power', top_db=top_db)

    @torch.no_grad()
    def forward(self, wav):
        wav = torch.nan_to_num(wav.float(), nan=0.0)
        mel = torch.nan_to_num(self.db(self.mel(wav)), nan=-80.0)
        B   = mel.shape[0]
        flat = mel.reshape(B, -1)
        mn = flat.min(1, keepdim=True)[0].unsqueeze(-1)
        mx = flat.max(1, keepdim=True)[0].unsqueeze(-1)
        mel = torch.nan_to_num((mel - mn) / (mx - mn + 1e-7), nan=0.0)
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


def sumix_freq(mel: torch.Tensor, labels: torch.Tensor) -> tuple:
    B = mel.shape[0]
    if B < 2: return mel, labels
    idx  = torch.randperm(B, device=mel.device)
    mask = (torch.rand(mel.shape[2], device=mel.device) > 0.5).view(1, 1, -1, 1)
    return torch.where(mask, mel[idx], mel), torch.max(labels, labels[idx])


def audio_mixup(x, y):
    B   = x.shape[0]
    idx = torch.randperm(B, device=x.device)
    return 0.5 * x + 0.5 * x[idx], torch.max(y, y[idx])


# ── EMA teacher update ────────────────────────────────────────────────────────

@torch.no_grad()
def update_teacher_ema(student: nn.Module, teacher: nn.Module, alpha: float):
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data.mul_(alpha).add_(s_param.data * (1.0 - alpha))
    for t_buf, s_buf in zip(teacher.buffers(), student.buffers()):
        t_buf.data.copy_(s_buf.data)


# ── Datasets ──────────────────────────────────────────────────────────────────

def load_audio_clip(path: str, n_samples: int) -> np.ndarray:
    try:
        audio, sr = sf.read(path, dtype='float32', always_2d=False)
        if audio.ndim == 2: audio = audio.mean(axis=1)
        if sr != SR: audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    except Exception: return np.zeros(n_samples, dtype=np.float32)
    if len(audio) < n_samples:
        audio = np.pad(audio, (0, n_samples - len(audio)))
    else:
        start = np.random.randint(0, len(audio) - n_samples + 1)
        audio = audio[start: start + n_samples]
    return absmax_normalize(audio.astype(np.float32))


def load_ss_clip(path: str, end_sec: int, n_samples: int) -> np.ndarray:
    try:
        start = max(0, end_sec - n_samples // SR) * SR
        audio, sr = sf.read(path, start=start, frames=n_samples * 2,
                             dtype='float32', always_2d=False)
        if audio.ndim == 2: audio = audio.mean(axis=1)
        if sr != SR: audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    except Exception: return np.zeros(n_samples, dtype=np.float32)
    if len(audio) < n_samples:
        audio = np.pad(audio, (0, n_samples - len(audio)))
    else:
        audio = audio[:n_samples]
    return absmax_normalize(audio.astype(np.float32))


class TrainAudioDataset(Dataset):
    def __init__(self, df, audio_dir, species_cols, n_samples):
        self.df          = df.reset_index(drop=True)
        self.audio_dir   = audio_dir
        self.species_cols = species_cols
        self.n_samples   = n_samples

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        audio = load_audio_clip(os.path.join(self.audio_dir, str(row['filename'])),
                                self.n_samples)
        hard  = np.zeros(NUM_CLASSES, dtype=np.float32)
        pl    = str(row.get('primary_label', ''))
        if pl in self.species_cols:
            hard[self.species_cols.index(pl)] = 1.0
        sec = str(row.get('secondary_labels', ''))
        if sec and sec not in ('[]', 'nan', ''):
            import re
            for sp in re.split(r"[;,\[\]'\s]+", sec):
                sp = sp.strip()
                if sp in self.species_cols:
                    hard[self.species_cols.index(sp)] = 0.5
        return torch.from_numpy(audio), torch.from_numpy(hard)


class SoundscapeUnlabeledDataset(IterableDataset):
    """Unlabeled soundscape clips — yields (audio,) with random 5s windows.

    Pre-loads all soundscape audio into RAM for fast random access.
    """

    def __init__(self, ss_dir: str, ogg_files: list, n_samples: int,
                 shuffle: bool = True):
        self.n_samples = n_samples
        self.shuffle   = shuffle
        print(f"  Pre-loading {len(ogg_files)} soundscapes for SSL …", flush=True)
        self._clips = []
        skipped = 0
        for f in ogg_files:
            try:
                audio, sr = sf.read(str(f), dtype='float32', always_2d=False)
                if audio.ndim == 2: audio = audio.mean(axis=1)
                if sr != SR: audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
            except Exception:
                skipped += 1
                continue
            # Split into fixed 5s clips
            for i in range(max(1, len(audio) // n_samples)):
                clip = audio[i * n_samples: (i + 1) * n_samples]
                if len(clip) < n_samples:
                    clip = np.pad(clip, (0, n_samples - len(clip)))
                m = np.abs(clip).max()
                if m > 1e-8: clip = clip / m
                self._clips.append(clip.astype(np.float32))
        print(f"  Soundscape clips: {len(self._clips):,} ({skipped} files skipped)",
              flush=True)

    def __iter__(self):
        idxs = list(range(len(self._clips)))
        if self.shuffle:
            import random; random.shuffle(idxs)
        for i in idxs:
            yield torch.from_numpy(self._clips[i])


class SoundscapeValDataset(Dataset):
    def __init__(self, df, ss_dir, species_cols, n_samples):
        self.df = df.reset_index(drop=True); self.ss_dir = ss_dir
        self.species_cols = species_cols; self.n_samples = n_samples

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        audio = load_ss_clip(os.path.join(self.ss_dir, str(row['filename'])),
                             int(row.get('end', 5)), self.n_samples)
        label = np.array([row[sc] for sc in self.species_cols], dtype=np.float32)
        return torch.from_numpy(audio), torch.from_numpy(label)


# ── Loss + helpers ────────────────────────────────────────────────────────────

class FocalBCE(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt  = torch.exp(-bce)
        return ((1 - pt) ** self.gamma * bce).mean()


def macro_auc(y_true, y_score):
    keep = y_true.sum(axis=0) > 0
    if keep.sum() == 0: return 0.0
    return roc_auc_score(y_true[:, keep], y_score[:, keep], average='macro')


def hhmmss_to_sec(t):
    if isinstance(t, (int, float)): return int(t)
    try:
        parts = str(t).split(':')
        if len(parts) == 3: return int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
        if len(parts) == 2: return int(parts[0])*60 + int(parts[1])
        return int(float(t))
    except: return 0


def build_ss_val_df(sc_labels, species_cols):
    sp2idx = {sp: i for i, sp in enumerate(species_cols)}
    rows = []
    for filename, grp in sc_labels.groupby('filename'):
        for _, row in grp.iterrows():
            end = hhmmss_to_sec(row.get('end', 5))
            label = np.zeros(NUM_CLASSES, dtype=np.float32)
            for sp in str(row.get('primary_label', '')).split(';'):
                sp = sp.strip()
                if sp in sp2idx: label[sp2idx[sp]] = 1.0
            rd = {'filename': filename, 'end': end}
            for j, sc in enumerate(species_cols): rd[sc] = label[j]
            rows.append(rd)
    return pd.DataFrame(rows)


def rampup(epoch, rampup_epochs):
    """Sigmoid ramp-up schedule for consistency weight."""
    if rampup_epochs == 0: return 1.0
    frac = min(epoch / rampup_epochs, 1.0)
    return float(np.exp(-5.0 * (1.0 - frac) ** 2))


# ── Training fold ─────────────────────────────────────────────────────────────

def train_fold(fold, cfg, device):
    t_cfg  = cfg['training']
    d_cfg  = cfg['data']
    m_cfg  = cfg.get('model', {})
    clip_dur = m_cfg.get('clip_duration', 5)
    n_samples = SR * clip_dur
    out_dir = Path(cfg['output']['dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    train_df     = pd.read_csv(d_cfg['train_csv'])
    sc_labels    = pd.read_csv(d_cfg['soundscape_labels_csv'])
    taxonomy     = pd.read_csv(d_cfg['taxonomy_csv'])
    species_cols = taxonomy['primary_label'].astype(str).tolist()

    sc_files  = sc_labels['filename'].unique()
    sc_groups = [f.split('_')[2] for f in sc_files]
    gkf       = GroupKFold(n_splits=d_cfg.get('n_folds', 5))
    fold_splits = list(gkf.split(sc_files, groups=sc_groups))
    _, val_idx  = fold_splits[fold]
    val_files   = set(sc_files[val_idx])
    train_sc_files = set(sc_files) - val_files

    sc_val_raw = sc_labels[sc_labels['filename'].isin(val_files)]
    sc_val_df  = build_ss_val_df(sc_val_raw, species_cols)
    print(f"Fold {fold}: val_sc={len(val_files)}, val_clips={len(sc_val_df)}")

    # Labeled dataset
    audio_ds = TrainAudioDataset(train_df, d_cfg['audio_dir'], species_cols, n_samples)
    print(f"  train_audio: {len(audio_ds):,} clips")

    # Unlabeled soundscape dataset (exclude val soundscapes)
    ss_dir  = Path(d_cfg['soundscape_dir'])
    ss_oggs = [ss_dir / f for f in train_sc_files if (ss_dir / f).exists()]
    # Also include extra unlabeled soundscapes (all non-val files)
    all_oggs = sorted(ss_dir.glob('*.ogg'))
    extra_oggs = [f for f in all_oggs if f.name not in val_files]
    unlabeled_oggs = sorted(set(ss_oggs + extra_oggs))
    print(f"  Unlabeled soundscapes: {len(unlabeled_oggs):,} files")

    unlabeled_ds = SoundscapeUnlabeledDataset(
        ss_dir=str(ss_dir), ogg_files=unlabeled_oggs,
        n_samples=n_samples, shuffle=True,
    )
    val_ds   = SoundscapeValDataset(sc_val_df, str(ss_dir), species_cols, n_samples)

    bs = t_cfg.get('batch_size', 32)
    audio_loader  = DataLoader(audio_ds, batch_size=bs, shuffle=True,
                               num_workers=4, pin_memory=True, drop_last=True)
    unlab_loader  = DataLoader(unlabeled_ds, batch_size=bs, num_workers=2,
                               pin_memory=True)
    val_loader    = DataLoader(val_ds, batch_size=bs, shuffle=False,
                               num_workers=2, pin_memory=True)

    # ── Models: student + teacher ─────────────────────────────────────────────
    def build_model():
        return SEDModel(
            backbone       = m_cfg.get('backbone', 'tf_efficientnet_b0.ns_jft_in1k'),
            num_classes    = NUM_CLASSES,
            dropout        = m_cfg.get('dropout', 0.1),
            drop_path_rate = m_cfg.get('drop_path_rate', 0.0),
            gem_p_init     = m_cfg.get('gem_p_init', 3.0),
        )

    student = build_model().to(device)
    teacher = build_model().to(device)

    # Warm-start both from competitor checkpoint
    competitor_ckpt = d_cfg.get('competitor_ckpt')
    if competitor_ckpt and os.path.isfile(competitor_ckpt):
        print(f"  Warm-start from: {competitor_ckpt}")
        ckpt  = torch.load(competitor_ckpt, map_location='cpu', weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        student.load_state_dict(state, strict=False)
        teacher.load_state_dict(state, strict=False)
        print(f"  Student + teacher initialised from competitor checkpoint")

    # Teacher has no gradient — EMA only
    for p in teacher.parameters():
        p.requires_grad_(False)

    mel_tf   = MelTransform(**{k: v for k, v in m_cfg.items()
                                if k in ('sr','n_mels','n_fft','hop_length',
                                         'fmin','fmax','top_db','power',
                                         'norm','mel_scale')}).to(device)
    spec_aug = SpecAug(
        freq_mask_param = m_cfg.get('freq_mask', 24),
        time_mask_param = m_cfg.get('time_mask', 32),
    ).to(device)

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr           = t_cfg.get('learning_rate', 1e-3),
        weight_decay = t_cfg.get('weight_decay', 1e-4),
    )
    epochs         = t_cfg.get('epochs', 30)
    sched          = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    scaler         = torch.cuda.amp.GradScaler()

    focal_gamma      = t_cfg.get('focal_gamma', 2.0)
    ema_alpha        = t_cfg.get('ema_alpha', 0.999)
    consistency_w    = t_cfg.get('consistency_weight', 1.0)
    rampup_epochs    = t_cfg.get('rampup_epochs', 5)
    confidence_thr   = t_cfg.get('confidence_threshold', 0.0)  # min teacher max-prob
    use_sumix_freq   = t_cfg.get('use_sumix_freq', True)
    unlab_oversample = t_cfg.get('unlabeled_oversample', 2)
    criterion        = FocalBCE(gamma=focal_gamma).to(device)

    best_auc     = 0.0
    history      = []
    oof_logits   = np.zeros((len(val_ds), NUM_CLASSES), dtype=np.float32)
    patience     = t_cfg.get('early_stopping_patience', 5)
    no_improve   = 0
    global_step  = 0

    run = None
    if _WANDB_AVAILABLE:
        run = wandb.init(
            project='birdclef-2026',
            name=f"{cfg['experiment']['name']}-fold{fold}",
            group=cfg['experiment']['name'],
            tags=['ssl', 'mean-teacher', f"fold{fold}"],
            config={**cfg, 'fold': fold}, reinit=True,
        )

    print(f"  Training fold {fold} | epochs={epochs} | "
          f"ema_α={ema_alpha} | consist_w={consistency_w}")

    for ep in range(1, epochs + 1):
        student.train(); teacher.eval()
        ep_sup  = 0.0
        ep_cons = 0.0
        n_steps = 0
        t0      = time.time()

        consist_coeff = rampup(ep - 1, rampup_epochs) * consistency_w
        unlab_iter    = iter(unlab_loader)

        for audio_wav, hard_label in audio_loader:
            audio_wav  = audio_wav.to(device)
            hard_label = hard_label.to(device)

            # MixUp on labeled audio
            audio_wav, hard_label = audio_mixup(audio_wav, hard_label)

            with torch.no_grad():
                mel_labeled = mel_tf(audio_wav)
            mel_labeled = spec_aug(mel_labeled)
            if use_sumix_freq:
                mel_labeled, hard_label = sumix_freq(mel_labeled, hard_label)

            with torch.cuda.amp.autocast():
                out      = student(mel_labeled)
                sup_loss = criterion(out['clipwise_logit'], hard_label)

            optimizer.zero_grad()
            scaler.scale(sup_loss).backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ep_sup  += sup_loss.item()
            n_steps += 1

            # ── Consistency loss on unlabeled soundscapes ─────────────────────
            for _ in range(unlab_oversample):
                try:
                    unlab_wav = next(unlab_iter)
                except StopIteration:
                    unlab_iter = iter(unlab_loader)
                    try: unlab_wav = next(unlab_iter)
                    except StopIteration: break

                unlab_wav = unlab_wav.to(device)

                # Teacher: weak path (just mel, no aug)
                with torch.no_grad():
                    mel_weak     = mel_tf(unlab_wav)
                    teacher_out  = teacher(mel_weak)
                    teacher_prob = teacher_out['clipwise_prob']  # (B, 234)

                # Filter: only use clips where teacher is confident
                if confidence_thr > 0.0:
                    confident = teacher_prob.max(dim=1)[0] >= confidence_thr
                    if confident.sum() == 0:
                        continue
                    teacher_prob = teacher_prob[confident]
                    unlab_wav    = unlab_wav[confident]

                # Student: strong path (SpecAug + SumixFreq)
                with torch.no_grad():
                    mel_strong = mel_tf(unlab_wav)
                mel_strong = spec_aug(mel_strong)
                if use_sumix_freq:
                    dummy = torch.zeros_like(teacher_prob)
                    mel_strong, _ = sumix_freq(mel_strong, dummy)

                with torch.cuda.amp.autocast():
                    student_out  = student(mel_strong)
                    student_prob = student_out['clipwise_prob']
                cons_loss = F.binary_cross_entropy(
                    student_prob.float().clamp(1e-7, 1 - 1e-7),
                    teacher_prob.detach().float().clamp(0.0, 1.0),
                ) * consist_coeff

                optimizer.zero_grad()
                scaler.scale(cons_loss).backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                ep_cons += cons_loss.item()

            # EMA update teacher
            update_teacher_ema(student, teacher, ema_alpha)
            global_step += 1

        sched.step()
        ep_time = time.time() - t0

        # Validation (using teacher for inference — typically better)
        teacher.eval()
        val_logits_ep, val_labels_ep = [], []
        with torch.no_grad():
            for wav, lbl in val_loader:
                mel = mel_tf(wav.to(device))
                out = teacher(mel)
                val_logits_ep.append(out['clipwise_logit'].cpu().numpy())
                val_labels_ep.append(lbl.numpy())

        vl  = np.concatenate(val_logits_ep)
        vla = np.concatenate(val_labels_ep)
        vp  = 1.0 / (1.0 + np.exp(-vl))
        auc = macro_auc(vla, vp)

        avg_sup  = ep_sup  / max(n_steps, 1)
        avg_cons = ep_cons / max(n_steps, 1)
        print(f"  Ep {ep:3d}/{epochs}  "
              f"sup={avg_sup:.4f}  cons={avg_cons:.4f}  "
              f"consist_coeff={consist_coeff:.3f}  "
              f"teacher_auc={auc:.4f}  {ep_time:.0f}s")
        history.append({'epoch': ep, 'sup_loss': avg_sup, 'cons_loss': avg_cons,
                        'val_auc': auc})

        if run is not None:
            run.log({'epoch': ep, 'train/sup': avg_sup, 'train/cons': avg_cons,
                     'val/ss_auc': auc, 'val/best_auc': best_auc,
                     'consist_coeff': consist_coeff})

        if auc > best_auc:
            best_auc = auc; no_improve = 0
            oof_logits = vl
            torch.save({
                'student_state_dict': {k: v.cpu().clone()
                                       for k, v in student.state_dict().items()},
                'teacher_state_dict': {k: v.cpu().clone()
                                       for k, v in teacher.state_dict().items()},
                'fold': fold, 'best_val_auc': best_auc, 'epoch': ep,
            }, out_dir / f'fold{fold}_best.pt')
            print(f"    ✓ New best teacher AUC={best_auc:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at ep {ep}")
                break

    if run is not None:
        run.finish()

    return {'fold': fold, 'best_auc': best_auc, 'history': history,
            'oof_logits': oof_logits}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--fold',   type=int, default=None)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    cfg    = load_config(args.config)
    print(f"Exp: {cfg['experiment']['name']}  Device: {device}")

    n_folds = cfg['data'].get('n_folds', 5)
    folds   = [args.fold] if args.fold is not None else list(range(n_folds))

    all_results = []; all_oof = {}
    for fold in folds:
        print(f"\n{'='*60}\n  Fold {fold}/{n_folds-1}\n{'='*60}")
        result = train_fold(fold, cfg, device)
        all_results.append(result)
        all_oof[fold] = result['oof_logits']

    out_dir = Path(cfg['output']['dir'])
    if len(all_results) == n_folds:
        mean_auc = np.mean([r['best_auc'] for r in all_results])
        print(f"\n{'='*60}")
        for r in all_results:
            print(f"  Fold {r['fold']}: best_auc={r['best_auc']:.4f}")
        print(f"  Mean fold AUC : {mean_auc:.4f}\n{'='*60}")
        with open(out_dir / 'result.json', 'w') as f:
            json.dump({'mean_fold_auc': mean_auc,
                       'folds': [{'fold': r['fold'], 'best_auc': r['best_auc']}
                                  for r in all_results]}, f, indent=2)
        np.savez_compressed(str(out_dir / 'oof_predictions.npz'),
                            logits=np.concatenate(list(all_oof.values())))


if __name__ == '__main__':
    main()
