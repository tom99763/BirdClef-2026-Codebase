"""Fine-tune competitor SED (AUC=0.9478) on our data.

Strategy:
  - Init : competitor_sed_fold0.pt (AUC=0.9478) — NOT training from scratch
  - Data1: train_audio  → FocalBCE on hard labels (pre-cached into RAM)
  - Data2: train_soundscapes → soft BCE on Perch pseudo labels (pre-cached)
  - LR   : very low (1-5e-5) with LLRD (backbone = 0.1× head)
  - Val  : GroupKFold on labeled soundscapes (same split as train_sed_ns.py)

Usage:
    CUDA_VISIBLE_DEVICES=1 python train_finetune_competitor.py \
        --config configs/finetune_comp_v1.yaml
"""

import argparse, json, os, re, sys, time
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils.config import load_config

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────
SR          = 32_000
NUM_CLASSES = 234

# ── Helpers ───────────────────────────────────────────────────────────────────
def absmax_normalize(x: np.ndarray) -> np.ndarray:
    m = np.abs(x).max()
    return x / m if m > 1e-8 else x

def macro_auc(labels, probs):
    keep = labels.sum(axis=0) > 0
    if keep.sum() == 0:
        return 0.0
    try:
        return roc_auc_score(labels[:, keep], probs[:, keep], average='macro')
    except Exception:
        return 0.0

def hhmmss_to_sec(s) -> int:
    parts = str(s).split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return int(float(s))

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
    """Load 5s soundscape clip by end-time offset."""
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


def build_ss_val_df(sc_labels: pd.DataFrame, species_cols: list) -> pd.DataFrame:
    """Expand soundscape labels to per-clip rows with multi-hot label vectors."""
    sp2idx = {sp: i for i, sp in enumerate(species_cols)}
    rows = []
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


# ── Datasets ──────────────────────────────────────────────────────────────────
class TrainAudioDataset(Dataset):
    """train_audio with hard labels, pre-cached to RAM."""

    def __init__(self, df: pd.DataFrame, audio_dir: str, species_cols: list, n_samples: int):
        self.df           = df.reset_index(drop=True)
        self.species_cols = species_cols
        self.n_samples    = n_samples

        print(f"  Pre-caching {len(self.df):,} train_audio clips into RAM …", flush=True)
        self._cache = []
        for i, row in enumerate(self.df.itertuples(index=False)):
            path = os.path.join(audio_dir, str(row.filename))
            clip = load_audio_clip(path, n_samples).astype(np.float16)
            self._cache.append(clip)
            if (i + 1) % 5000 == 0:
                print(f"    cached {i+1:,}/{len(self.df):,}", flush=True)
        print("  Pre-cache complete.", flush=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        audio = self._cache[idx].astype(np.float32)

        hard = np.zeros(NUM_CLASSES, dtype=np.float32)
        pl   = str(row.get('primary_label', ''))
        if pl in self.species_cols:
            hard[self.species_cols.index(pl)] = 1.0
        sec = str(row.get('secondary_labels', ''))
        if sec and sec not in ('[]', 'nan', ''):
            for sp in re.split(r"[;,\[\]'\s]+", sec):
                sp = sp.strip()
                if sp and sp in self.species_cols:
                    hard[self.species_cols.index(sp)] = 0.5

        return torch.from_numpy(audio), torch.from_numpy(hard)


class PerchSSDataset(Dataset):
    """Soundscape clips with Perch/SED pseudo labels, pre-cached to RAM.

    CSV format: row_id, {species_prob_cols...}[, primary_label, secondary_labels]
    row_id: {soundscape_stem}_{end_sec}
    """

    def __init__(self, csv_path: str, ss_dir: str, species_cols: list,
                 n_samples: int, exclude_files: set = None):
        self.n_samples = n_samples

        df = pd.read_csv(csv_path)
        prob_cols = [c for c in df.columns
                     if c not in ('row_id', 'primary_label', 'secondary_labels')]
        # Build mapping: our taxonomy order → prob column indices
        prob_col_set = set(prob_cols)
        col_map = [(species_cols.index(c), prob_cols.index(c))
                   for c in species_cols if c in prob_col_set]

        self._cache = []
        skipped = 0
        print(f"  Loading SS pseudo clips ({len(df):,} rows) …", flush=True)
        prob_arr = df[prob_cols].values.astype(np.float32)

        for i, row in enumerate(df.itertuples(index=False)):
            rid = str(row.row_id)
            parts = rid.rsplit('_', 1)
            if len(parts) != 2:
                skipped += 1; continue
            fname   = parts[0] + '.ogg'
            if exclude_files and fname in exclude_files:
                skipped += 1; continue
            end_sec = int(parts[1])
            path    = os.path.join(ss_dir, fname)

            soft = np.zeros(NUM_CLASSES, dtype=np.float32)
            for our_idx, csv_idx in col_map:
                soft[our_idx] = prob_arr[i, csv_idx]

            clip = load_ss_clip(path, end_sec, n_samples).astype(np.float16)
            self._cache.append((clip, soft))

            if (i + 1) % 10000 == 0:
                print(f"    loaded {i+1:,}/{len(df):,}", flush=True)

        print(f"  SS pseudo: {len(self._cache):,} clips (skipped {skipped})", flush=True)

    def __len__(self):
        return len(self._cache)

    def __getitem__(self, idx):
        clip, soft = self._cache[idx]
        return torch.from_numpy(clip.astype(np.float32)), torch.from_numpy(soft.copy())


class SoundscapeValDataset(Dataset):
    """Labeled soundscape clips for validation (same as train_sed_ns.py)."""

    def __init__(self, df: pd.DataFrame, ss_dir: str, species_cols: list, n_samples: int):
        self.df           = df.reset_index(drop=True)
        self.ss_dir       = ss_dir
        self.species_cols = species_cols
        self.n_samples    = n_samples

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        audio = load_ss_clip(
            os.path.join(self.ss_dir, str(row['filename'])),
            int(row.get('end', 5)), self.n_samples
        )
        label = np.array([row[sc] for sc in self.species_cols], dtype=np.float32)
        return torch.from_numpy(audio), torch.from_numpy(label)


# ── Model ─────────────────────────────────────────────────────────────────────
class GEMFreqPool(nn.Module):
    """Named gem_pool to match competitor checkpoint."""
    def __init__(self, p_init=3.0):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)))

    def forward(self, x):
        p   = self.p.clamp(1.0, 10.0)
        out = x.clamp(min=1e-6).pow(p).mean(dim=2).pow(1.0 / p)
        return out


class CompetitorHead(nn.Module):
    """Mirrors competitor head exactly:
      fc       : Linear(C, C)
      att_conv : Conv1d(C, num_classes, 1)  → attention weights over T
      cls_conv : Conv1d(C, num_classes, 1)  → class logits over T
    """
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.fc       = nn.Sequential(nn.Linear(in_features, in_features))
        self.att_conv = nn.Conv1d(in_features, num_classes, kernel_size=1)
        self.cls_conv = nn.Conv1d(in_features, num_classes, kernel_size=1)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, T) — comes directly from gem_pool
        x   = self.dropout(x)
        # fc applied per-timestep: (B, C, T) → permute → fc → permute back
        x   = self.fc(x.permute(0, 2, 1)).permute(0, 2, 1)   # (B, C, T)
        att = torch.softmax(self.att_conv(x), dim=-1)          # (B, classes, T)
        cls = self.cls_conv(x)                                  # (B, classes, T)
        clip_logit = (att * cls).sum(dim=-1)                   # (B, classes)
        return {'clipwise_logit': clip_logit,
                'clipwise_prob':  torch.sigmoid(clip_logit)}


class SEDModel(nn.Module):
    def __init__(self, backbone='tf_efficientnet_b0.ns_jft_in1k',
                 num_classes=NUM_CLASSES, dropout=0.1,
                 drop_path_rate=0.0, gem_p_init=3.0):
        super().__init__()
        # in_chans=3 to match competitor checkpoint architecture exactly
        self.backbone = timm.create_model(
            backbone, pretrained=False, in_chans=3,
            drop_path_rate=drop_path_rate,
            features_only=False, num_classes=0, global_pool='',
        )
        feat = self.backbone.num_features
        self.gem_pool = GEMFreqPool(p_init=gem_p_init)   # named gem_pool to match
        self.head     = CompetitorHead(feat, num_classes, dropout)

    def forward(self, x):
        # x: (B, 1, F, T) mel — replicate to 3 channels to match competitor
        x    = x.repeat(1, 3, 1, 1)       # (B, 3, F, T)
        feat = self.backbone(x)            # (B, C, F, T)
        feat = self.gem_pool(feat)         # (B, C, T)
        return self.head(feat)


# ── Mel + Augmentation ────────────────────────────────────────────────────────
class MelTransform(nn.Module):
    def __init__(self, sr=SR, n_mels=224, n_fft=2048, hop_length=512,
                 fmin=0, fmax=16000, top_db=80.0, power=2.0,
                 norm='slaney', mel_scale='htk'):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=fmin, f_max=fmax, power=power,
            norm=norm, mel_scale=mel_scale,
        )
        self.top_db = top_db
        self.amp2db = T.AmplitudeToDB(top_db=top_db)

    def forward(self, wav):
        mel = self.mel(wav)
        mel = self.amp2db(mel)
        mel = mel.unsqueeze(1)
        mel = (mel + self.top_db) / self.top_db
        return mel


class SpecAugment(nn.Module):
    def __init__(self, freq_mask=24, time_mask=32):
        super().__init__()
        self.fm = T.FrequencyMasking(freq_mask)
        self.tm = T.TimeMasking(time_mask)

    def forward(self, x):
        return self.tm(self.fm(x))


def sumix_freq(mel, labels_a, labels_b=None):
    B    = mel.shape[0]
    idx  = torch.randperm(B, device=mel.device)
    lam  = torch.rand(B, 1, 1, 1, device=mel.device)
    mix  = lam * mel + (1 - lam) * mel[idx]
    la   = torch.max(labels_a, labels_a[idx])
    if labels_b is not None:
        lb = torch.max(labels_b, labels_b[idx])
        return mix, la, lb
    return mix, la


def audio_mixup(wav, labels):
    idx = torch.randperm(wav.shape[0], device=wav.device)
    mix = 0.5 * wav + 0.5 * wav[idx]
    lbl = torch.max(labels, labels[idx])
    return mix, lbl


def focal_bce(logit, target, gamma=2.0):
    bce  = F.binary_cross_entropy_with_logits(logit, target, reduction='none')
    prob = torch.sigmoid(logit).detach()
    p_t  = prob * target + (1 - prob) * (1 - target)
    return (bce * (1 - p_t).pow(gamma)).mean()


# ── Training fold ─────────────────────────────────────────────────────────────
def train_fold(fold: int, cfg: dict, device: torch.device, out_dir: Path):
    d_cfg = cfg['data']
    m_cfg = cfg['model']
    t_cfg = cfg['training']

    n_samples        = int(m_cfg.get('clip_duration', 5) * SR)
    epochs           = t_cfg.get('epochs', 20)
    patience         = t_cfg.get('early_stopping_patience', 5)
    bs               = t_cfg.get('batch_size', 64)
    lr               = t_cfg.get('learning_rate', 2e-5)
    weight_decay     = t_cfg.get('weight_decay', 1e-4)
    focal_gamma      = t_cfg.get('focal_gamma', 2.0)
    use_sumix        = t_cfg.get('use_sumix_freq', True)
    ss_weight        = t_cfg.get('ss_weight', 0.0)
    ss_oversample    = t_cfg.get('ss_oversample', 1)
    backbone_lr_mult = t_cfg.get('backbone_lr_mult', 0.1)
    use_llrd         = t_cfg.get('use_llrd', True)
    warmup_epochs    = t_cfg.get('warmup_epochs', 2)
    competitor_ckpt  = d_cfg.get('competitor_ckpt', '')

    # ── Data ──────────────────────────────────────────────────────────────────
    train_df    = pd.read_csv(d_cfg['train_csv'])
    tax         = pd.read_csv(d_cfg['taxonomy_csv'])
    species     = tax['primary_label'].astype(str).tolist()
    sc_labels   = pd.read_csv(d_cfg['soundscape_labels_csv'])

    # Soundscape GroupKFold — identical to train_sed_ns.py
    sc_files  = sc_labels['filename'].unique()
    sc_groups = [f.split('_')[2] for f in sc_files]
    gkf       = GroupKFold(n_splits=d_cfg.get('n_folds', 5))
    splits    = list(gkf.split(sc_files, groups=sc_groups))
    _, val_idx = splits[fold]
    val_files  = set(sc_files[val_idx])

    sc_val_raw = sc_labels[sc_labels['filename'].isin(val_files)]
    sc_val_df  = build_ss_val_df(sc_val_raw, species)
    print(f"Fold {fold}: val_sc_files={len(val_files)}, val_rows={len(sc_val_df)}")

    # Datasets
    audio_ds = TrainAudioDataset(train_df, d_cfg['audio_dir'], species, n_samples)

    ss_loader = None
    perch_csv = d_cfg.get('perch_ss_csv', '')
    if perch_csv and os.path.isfile(perch_csv) and ss_weight > 0:
        ss_ds = PerchSSDataset(
            csv_path=perch_csv, ss_dir=d_cfg['soundscape_dir'],
            species_cols=species, n_samples=n_samples, exclude_files=val_files,
        )
        ss_loader = DataLoader(ss_ds, batch_size=bs, shuffle=True,
                               num_workers=2, pin_memory=True, drop_last=True,
                               persistent_workers=True)

    val_ds = SoundscapeValDataset(sc_val_df, d_cfg['soundscape_dir'], species, n_samples)

    audio_loader = DataLoader(audio_ds, batch_size=bs, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True,
                              persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=2, pin_memory=True, persistent_workers=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SEDModel(
        backbone       = m_cfg.get('backbone', 'tf_efficientnet_b0.ns_jft_in1k'),
        num_classes    = NUM_CLASSES,
        dropout        = m_cfg.get('dropout', 0.1),
        drop_path_rate = m_cfg.get('drop_path_rate', 0.0),
        gem_p_init     = m_cfg.get('gem_p_init', 3.0),
    ).to(device)

    if competitor_ckpt and os.path.isfile(competitor_ckpt):
        ckpt  = torch.load(competitor_ckpt, map_location='cpu', weights_only=False)
        state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        # in_chans=3 matches competitor exactly — direct load, no conversion needed
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Loaded ckpt: missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        print(f"  WARNING: ckpt not found ({competitor_ckpt}) — using timm pretrain!")

    # ── Optimiser (LLRD) ──────────────────────────────────────────────────────
    backbone_params = [p for n, p in model.named_parameters() if 'backbone' in n]
    head_params     = [p for n, p in model.named_parameters() if 'backbone' not in n]
    if use_llrd:
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': lr * backbone_lr_mult},
            {'params': head_params,     'lr': lr},
        ], weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                      weight_decay=weight_decay)

    def lr_lambda(ep):
        if ep < warmup_epochs:
            return (ep + 1) / warmup_epochs
        progress = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    mel_tf = MelTransform(
        sr=SR, n_mels=m_cfg.get('n_mels', 224), n_fft=m_cfg.get('n_fft', 2048),
        hop_length=m_cfg.get('hop_length', 512), fmin=m_cfg.get('fmin', 0),
        fmax=m_cfg.get('fmax', 16000), top_db=m_cfg.get('top_db', 80.0),
        power=m_cfg.get('power', 2.0), norm=m_cfg.get('norm', 'slaney'),
        mel_scale=m_cfg.get('mel_scale', 'htk'),
    ).to(device)

    spec_aug = SpecAugment(
        freq_mask=m_cfg.get('freq_mask', 24),
        time_mask=m_cfg.get('time_mask', 32),
    ).to(device)

    scaler = torch.amp.GradScaler('cuda')

    # ── wandb ─────────────────────────────────────────────────────────────────
    run = None
    if _WANDB_AVAILABLE:
        run = wandb.init(
            project = 'birdclef-2026',
            name    = f"{cfg['experiment']['name']}-fold{fold}",
            group   = cfg['experiment']['name'],
            tags    = ['finetune-competitor', f"fold{fold}"],
            config  = {**cfg, 'fold': fold},
            reinit  = 'finish_previous',
        )

    print(f"  Finetuning fold {fold} | epochs={epochs} | patience={patience} "
          f"| lr={lr} | ss_weight={ss_weight}", flush=True)

    best_auc       = 0.0
    best_state     = None
    no_improve_cnt = 0
    history        = []

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0; n_steps = 0
        t0      = time.time()
        ss_iter = iter(ss_loader) if ss_loader else None

        for wav, hard in audio_loader:
            wav  = wav.to(device)
            hard = hard.to(device)

            wav, hard = audio_mixup(wav, hard)

            with torch.no_grad():
                mel = mel_tf(wav)
            mel = spec_aug(mel)

            if use_sumix:
                mel, hard = sumix_freq(mel, hard)

            with torch.amp.autocast('cuda'):
                out  = model(mel)
                loss = focal_bce(out['clipwise_logit'], hard, gamma=focal_gamma)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            ep_loss += loss.item(); n_steps += 1

            # ── Perch SS branch ───────────────────────────────────────────────
            if ss_iter is not None and ss_weight > 0:
                for _ in range(ss_oversample):
                    try:
                        ss_wav, ss_soft = next(ss_iter)
                    except StopIteration:
                        ss_iter = iter(ss_loader)
                        try:
                            ss_wav, ss_soft = next(ss_iter)
                        except StopIteration:
                            break

                    ss_wav  = ss_wav.to(device)
                    ss_soft = ss_soft.to(device)

                    with torch.no_grad():
                        ss_mel = mel_tf(ss_wav)
                    ss_mel = spec_aug(ss_mel)

                    with torch.amp.autocast('cuda'):
                        ss_out  = model(ss_mel)
                        ss_prob = ss_out['clipwise_prob']

                    ss_loss = F.binary_cross_entropy(
                        ss_prob.float().clamp(1e-7, 1 - 1e-7),
                        ss_soft.float().clamp(0.0, 1.0),
                    ) * ss_weight

                    optimizer.zero_grad()
                    scaler.scale(ss_loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()

        sched.step()
        ep_time = time.time() - t0

        # Validation
        model.eval()
        val_logits, val_labels = [], []
        with torch.no_grad():
            for wav, lbl in val_loader:
                mel = mel_tf(wav.to(device))
                out = model(mel)
                val_logits.append(out['clipwise_logit'].cpu().numpy())
                val_labels.append(lbl.numpy())

        vl  = np.concatenate(val_logits)
        vla = np.concatenate(val_labels)
        vp  = 1.0 / (1.0 + np.exp(-vl))
        auc = macro_auc(vla, vp)
        avg_loss = ep_loss / max(n_steps, 1)

        print(f"  Ep {ep:3d}/{epochs}  loss={avg_loss:.4f}  ss_auc={auc:.4f}  {ep_time:.0f}s",
              flush=True)
        history.append({'epoch': ep, 'loss': avg_loss, 'val_auc': auc})

        if run:
            run.log({'epoch': ep, 'train/loss': avg_loss,
                     'val/ss_auc': auc, 'val/best_auc': best_auc})

        if auc > best_auc:
            best_auc       = auc
            no_improve_cnt = 0
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save({'state_dict': best_state, 'fold': fold,
                        'best_val_auc': best_auc, 'epoch': ep},
                       out_dir / f'fold{fold}_best.pt')
            print(f"    ✓ New best AUC={best_auc:.4f}", flush=True)
        else:
            no_improve_cnt += 1
            if no_improve_cnt >= patience:
                print(f"  Early stop at epoch {ep}")
                break

    if run:
        run.finish()

    return best_auc, history


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--fold',   type=int, default=None)
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    seed   = cfg['experiment'].get('seed', 42)
    torch.manual_seed(seed); np.random.seed(seed)

    out_dir = Path(cfg['output']['dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config   : {args.config}")
    print(f"Device   : {args.device}")
    print(f"Exp name : {cfg['experiment']['name']}")

    n_folds      = cfg['data'].get('n_folds', 5)
    fold_range   = [args.fold] if args.fold is not None else list(range(n_folds))
    fold_results = []

    for fold in fold_range:
        print(f"\n{'='*60}\n  Fold {fold}/{n_folds-1}\n{'='*60}")
        best_auc, history = train_fold(fold, cfg, device, out_dir)
        fold_results.append({'fold': fold, 'best_auc': best_auc})
        print(f"  Fold {fold} done — best AUC={best_auc:.4f}")

        # Auto-copy to sed_improved/ if above threshold
        if best_auc >= 0.9193:
            import shutil
            exp_name = cfg['experiment']['name'].replace('-', '_')
            dst = Path('sed_improved') / f"{exp_name}_fold{fold}_auc{best_auc:.4f}.pt"
            shutil.copy2(str(out_dir / f'fold{fold}_best.pt'), str(dst))
            print(f"  ✓ Copied to sed_improved/: {dst.name}")

    result = {
        'experiment':    cfg['experiment']['name'],
        'folds':         fold_results,
        'mean_fold_auc': float(np.mean([r['best_auc'] for r in fold_results])),
    }
    (out_dir / 'result.json').write_text(json.dumps(result, indent=2))
    print(f"\nMean fold AUC: {result['mean_fold_auc']:.4f}")
    print(f"Result: {out_dir}/result.json")


if __name__ == '__main__':
    main()
