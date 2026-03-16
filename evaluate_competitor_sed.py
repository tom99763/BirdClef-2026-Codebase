"""Evaluate competitor SED model (best_fold0.pt, LB=0.862) on our holdout set.

Usage:
    python evaluate_competitor_sed.py
    python evaluate_competitor_sed.py --gpu 0
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchaudio.transforms as T
import librosa
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score


# ── Config (exact copy from competitor notebook) ────────────────────────────
@dataclass
class Config:
    sr: int = 32_000
    chunk_duration: float = 5.0
    n_mels: int = 224
    n_fft: int = 2048
    hop_length: int = 512
    fmin: int = 0
    fmax: int = 16_000
    top_db: float = 80.0
    power: float = 2.0
    norm: str = "slaney"
    mel_scale: str = "htk"
    backbone: str = "tf_efficientnet_b0.ns_jft_in1k"
    num_classes: int = 234
    in_channels: int = 3
    dropout: float = 0.1
    drop_path_rate: float = 0.0
    gem_p_init: float = 3.0

    @property
    def chunk_samples(self) -> int:
        return int(self.sr * self.chunk_duration)


# ── Model (exact copy from competitor notebook) ─────────────────────────────
class GEMFreqPool(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p_init))
        self.eps = eps

    def forward(self, x):
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=2)
        x = x.pow(1.0 / p)
        return x


class AttentionSEDHead(nn.Module):
    def __init__(self, feat_dim, num_classes, dropout=0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.att_conv = nn.Conv1d(feat_dim, num_classes, kernel_size=1)
        self.cls_conv = nn.Conv1d(feat_dim, num_classes, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        x = x.permute(0, 2, 1)
        att = torch.tanh(self.att_conv(x))
        att = F.softmax(att, dim=-1)
        cls = self.cls_conv(x)
        clipwise_logit = (att * cls).sum(dim=-1)
        clipwise_prob = torch.sigmoid(clipwise_logit)
        segmentwise_logit = cls.permute(0, 2, 1)
        return {
            "clipwise_prob": clipwise_prob,
            "segmentwise_logit": segmentwise_logit,
        }


class SEDModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.backbone, pretrained=False,
            in_chans=cfg.in_channels, features_only=False,
            global_pool="", num_classes=0,
            drop_path_rate=cfg.drop_path_rate,
        )
        feat_dim = self.backbone.num_features
        self.gem_pool = GEMFreqPool(p_init=cfg.gem_p_init)
        self.head = AttentionSEDHead(feat_dim, cfg.num_classes, cfg.dropout)

    def forward(self, x):
        features = self.backbone(x)
        pooled = self.gem_pool(features)
        return self.head(pooled)


# ── Mel Transform ────────────────────────────────────────────────────────────
class MelSpectrogramTransform(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=cfg.sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
            n_mels=cfg.n_mels, f_min=cfg.fmin, f_max=cfg.fmax,
            power=cfg.power, norm=cfg.norm, mel_scale=cfg.mel_scale,
        )
        self.db = T.AmplitudeToDB(stype="power", top_db=cfg.top_db)

    @torch.no_grad()
    def forward(self, waveforms):
        mel = self.db(self.mel(waveforms))
        B = mel.shape[0]
        mel_flat = mel.reshape(B, -1)
        mel_min = mel_flat.min(dim=1, keepdim=True)[0].unsqueeze(-1)
        mel_max = mel_flat.max(dim=1, keepdim=True)[0].unsqueeze(-1)
        mel = (mel - mel_min) / (mel_max - mel_min + 1e-7)
        mel = mel.unsqueeze(1).repeat(1, 3, 1, 1)
        return mel


def load_audio_clips(path, cfg):
    """Load audio file and split into 5-second clips (same logic as competitor)."""
    y, _ = librosa.load(path, sr=cfg.sr, mono=True)
    CHUNK = cfg.chunk_samples
    n_chunks = max(1, len(y) // CHUNK)
    padded_len = n_chunks * CHUNK
    if len(y) < padded_len:
        y = np.pad(y, (0, padded_len - len(y)))
    else:
        y = y[:padded_len]
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak
    chunks = y.reshape(n_chunks, CHUNK)
    return chunks  # (n_chunks, chunk_samples)


def predict_file(audio_path, model, mel_transform, cfg, device):
    """Return max-pooled probabilities over all clips in the file."""
    chunks = load_audio_clips(audio_path, cfg)
    chunks_t = torch.from_numpy(chunks).float().to(device)
    with torch.no_grad():
        mel = mel_transform(chunks_t)
        out = model(mel)
        probs = out["clipwise_prob"]  # (n_chunks, 234)
    # Max-pool over clips → single prediction per file
    return probs.max(dim=0).values.cpu().numpy()  # (234,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--checkpoint", default="models/sed_weights/best_fold0.pt")
    parser.add_argument("--holdout_csv", default="configs/holdout_val_files.csv")
    parser.add_argument("--audio_dir", default="birdclef-2026/train_audio")
    parser.add_argument("--sample_submission", default="birdclef-2026/sample_submission.csv")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = Config()

    # ── Species mapping ──────────────────────────────────────────────────────
    sample_sub = pd.read_csv(args.sample_submission, nrows=1)
    species_list = list(sample_sub.columns[1:])
    species_to_idx = {s: i for i, s in enumerate(species_list)}
    num_classes = len(species_list)
    print(f"Species: {num_classes}")

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"Loading: {args.checkpoint}")
    model = SEDModel(cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    print(f"  Epoch {ckpt['epoch']}, val_auc={ckpt['metrics'].get('macro_auc', '?'):.4f}")

    mel_transform = MelSpectrogramTransform(cfg)
    mel_transform.eval().to(device)

    # ── Load holdout set ─────────────────────────────────────────────────────
    holdout = pd.read_csv(args.holdout_csv)
    # unique files only (take first label per file)
    holdout_files = holdout.drop_duplicates("filename")[["filename", "primary_label"]].copy()
    holdout_files["primary_label"] = holdout_files["primary_label"].astype(str)
    print(f"Holdout files: {len(holdout_files)}  (unique: {holdout_files['filename'].nunique()})")

    # ── Inference ────────────────────────────────────────────────────────────
    all_preds = []
    all_labels = []
    skipped = 0

    for i, (_, row) in enumerate(holdout_files.iterrows()):
        audio_path = os.path.join(args.audio_dir, row["filename"])
        if not os.path.isfile(audio_path):
            skipped += 1
            continue
        try:
            probs = predict_file(audio_path, model, mel_transform, cfg, device)
        except Exception as e:
            print(f"  ERROR {row['filename']}: {e}")
            skipped += 1
            continue

        all_preds.append(probs)
        all_labels.append(str(row["primary_label"]))

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(holdout_files)}] processed")

    print(f"Done: {len(all_preds)} files  (skipped={skipped})")

    # ── Build y matrix ───────────────────────────────────────────────────────
    X = np.stack(all_preds).astype(np.float32)
    y = np.zeros((len(all_labels), num_classes), dtype=np.float32)
    for i, sp in enumerate(all_labels):
        if sp in species_to_idx:
            y[i, species_to_idx[sp]] = 1.0

    species_with_pos = np.where(y.sum(0) > 0)[0]
    print(f"Shape: X={X.shape}  species_with_pos={len(species_with_pos)}/234")

    # ── Score ─────────────────────────────────────────────────────────────────
    try:
        auc = roc_auc_score(
            y[:, species_with_pos],
            X[:, species_with_pos],
            average="macro",
        )
    except Exception as e:
        print(f"Scoring error: {e}")
        auc = None

    print(f"\n{'='*60}")
    print(f"  Competitor SED (best_fold0.pt, LB=0.862)")
    print(f"  Holdout ROC-AUC (macro, {len(species_with_pos)}/234 species): "
          f"{auc:.4f}" if auc else "  N/A")
    print(f"{'='*60}")
    print(f"\nBaseline (nohuman-label-pseudo, LB=0.849+PP): holdout=0.9453")
    print(f"Holdout ↔ LB gap ≈ 0.096  (individual recordings vs soundscape domain)")


if __name__ == "__main__":
    main()
