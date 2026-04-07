"""Generate file-level soft pseudo labels from competitor SED on train_audio.

Loads competitor_sed_fold0.pt, runs inference on every 5s clip from all
train_audio files, then averages probabilities to produce a file-level
soft teacher signal for knowledge distillation.

Output (outputs/competitor_pseudo/train_audio_probs.npz):
  filenames : np.ndarray of str  — relative paths matching train.csv 'filename' col
  probs     : np.ndarray float32 — (N_files, 234)  mean prob over 5s clips

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/gen_competitor_pseudo.py \
        --audio_dir birdclef-2026/train_audio \
        --taxonomy  birdclef-2026/taxonomy.csv \
        --out       outputs/competitor_pseudo/train_audio_probs.npz \
        --batch_size 64
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import timm
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

SR          = 32_000
CLIP_SAMPLES = SR * 5   # competitor uses 5s clips


# ── Model (identical to train_sed_ns.py) ──────────────────────────────────────

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
        return torch.sigmoid(logit)


class SEDModel(nn.Module):
    def __init__(self, backbone='tf_efficientnet_b0.ns_jft_in1k',
                 num_classes=234, dropout=0.1):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=False, in_chans=3,
            features_only=False, global_pool='', num_classes=0,
        )
        self.gem_pool = GEMFreqPool()
        feat_dim      = self.backbone.num_features
        self.head     = AttentionSEDHead(feat_dim, num_classes, dropout)

    def forward(self, x):
        return self.head(self.gem_pool(self.backbone(x)))


# ── Mel transform ─────────────────────────────────────────────────────────────

class MelTransform(nn.Module):
    def __init__(self, sr=SR, n_mels=224, n_fft=2048, hop_length=512,
                 fmin=0, fmax=16_000, top_db=80.0):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=fmin, f_max=fmax, power=2.0,
            norm='slaney', mel_scale='htk',
        )
        self.db  = T.AmplitudeToDB(stype='power', top_db=top_db)

    @torch.no_grad()
    def forward(self, wav):
        wav = torch.nan_to_num(wav.float(), nan=0.0)
        mel = torch.nan_to_num(self.db(self.mel(wav)), nan=-80.0)
        B   = mel.shape[0]
        flat = mel.reshape(B, -1)
        mn = flat.min(1, keepdim=True)[0].unsqueeze(-1)
        mx = flat.max(1, keepdim=True)[0].unsqueeze(-1)
        mel = torch.nan_to_num((mel - mn) / (mx - mn + 1e-7), nan=0.0)
        return mel.unsqueeze(1).repeat(1, 3, 1, 1)


# ── Inference helpers ─────────────────────────────────────────────────────────

def load_audio_for_infer(path: str) -> np.ndarray:
    try:
        audio, orig_sr = sf.read(path, dtype='float32', always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if orig_sr != SR:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=SR)
        return audio.astype(np.float32)
    except Exception:
        return np.zeros(CLIP_SAMPLES, dtype=np.float32)


@torch.no_grad()
def infer_file(model, mel_tf, audio: np.ndarray, device,
               batch_size: int = 64) -> np.ndarray:
    """Run inference on all 5s clips from audio, return mean probability."""
    n_clips = max(1, len(audio) // CLIP_SAMPLES)
    clips   = []
    for i in range(n_clips):
        clip = audio[i * CLIP_SAMPLES: (i + 1) * CLIP_SAMPLES]
        if len(clip) < CLIP_SAMPLES:
            clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
        # absmax normalise
        m = np.abs(clip).max()
        if m > 1e-8:
            clip = clip / m
        clips.append(clip)

    all_probs = []
    for start in range(0, len(clips), batch_size):
        batch = torch.from_numpy(
            np.stack(clips[start: start + batch_size])
        ).to(device)                         # (B, CLIP_SAMPLES)
        mel   = mel_tf(batch)                # (B, 3, n_mels, T)
        probs = model(mel).cpu().numpy()     # (B, 234)
        all_probs.append(probs)

    return np.concatenate(all_probs, axis=0).mean(axis=0)  # (234,)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default=(
        "birdclef-2026/notebook resource/current_subs/weights/competitor_sed_fold0.pt"
    ))
    parser.add_argument('--audio_dir', default='birdclef-2026/train_audio')
    parser.add_argument('--taxonomy',  default='birdclef-2026/taxonomy.csv')
    parser.add_argument('--out',       default='outputs/competitor_pseudo/train_audio_probs.npz')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device',    default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Species list
    taxonomy    = pd.read_csv(args.taxonomy)
    species_cols = taxonomy['primary_label'].astype(str).tolist()
    num_classes  = len(species_cols)
    print(f"Species: {num_classes}")

    # Build model
    model = SEDModel(num_classes=num_classes).to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  Missing keys : {len(missing)}  Unexpected: {len(unexpected)}")
    if missing:
        print(f"  Missing (first 5): {missing[:5]}")
    model.eval()

    # Mel transform
    mel_tf = MelTransform().to(device)

    # Walk train_audio
    audio_dir = Path(args.audio_dir)
    ogg_files = sorted(audio_dir.rglob('*.ogg'))
    print(f"Found {len(ogg_files):,} .ogg files in {audio_dir}")

    os.makedirs(Path(args.out).parent, exist_ok=True)

    filenames_out = []
    probs_out     = []

    for ogg_path in tqdm(ogg_files, desc='Inferring'):
        rel = str(ogg_path.relative_to(audio_dir))   # e.g. 'species/XC12345.ogg'
        audio = load_audio_for_infer(str(ogg_path))
        prob  = infer_file(model, mel_tf, audio, device, args.batch_size)
        filenames_out.append(rel)
        probs_out.append(prob)

    probs_arr = np.stack(probs_out, axis=0).astype(np.float32)  # (N, 234)
    np.savez_compressed(
        args.out,
        filenames = np.array(filenames_out),
        probs     = probs_arr,
    )
    print(f"\nSaved {len(filenames_out):,} files → {args.out}")
    print(f"  probs shape: {probs_arr.shape}")
    print(f"  mean max prob: {probs_arr.max(axis=1).mean():.4f}")


if __name__ == '__main__':
    main()
