"""Visualize SED attention maps for worst-performing validation samples.

Usage:
    python scripts/visualize_sed_attention.py \
        --config configs/sed_ns_b0_20s_v2_r1.yaml \
        --fold 0 \
        --out_dir outputs/sed-ns-b0-20s-v2-r1/attention_maps \
        --n_worst 20
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Minimal model re-definition (matches train_sed_ns.py) ─────────────────────
import timm

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
        x = self.fc(x.permute(0, 2, 1)).permute(0, 2, 1)
        att = F.softmax(torch.tanh(self.att_conv(x)), dim=-1)  # (B, n_cls, T)
        cls = self.cls_conv(x)                                   # (B, n_cls, T)
        logit = (att * cls).sum(-1)
        return {
            'clipwise_logit': logit,
            'clipwise_prob':  torch.sigmoid(logit),
            'attention':      att,   # (B, n_cls, T) — for visualization
            'framewise':      cls,   # (B, n_cls, T)
        }


class SEDModel(nn.Module):
    def __init__(self, backbone='tf_efficientnet_b0.ns_jft_in1k',
                 num_classes=234, in_channels=3, dropout=0.1,
                 drop_path_rate=0.0, gem_p_init=3.0):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=False, in_chans=in_channels,
            features_only=False, global_pool='', num_classes=0,
            drop_path_rate=drop_path_rate,
        )
        self.gem_pool = GEMFreqPool(p_init=gem_p_init)
        feat_dim      = self.backbone.num_features
        self.head     = AttentionSEDHead(feat_dim, num_classes, dropout)

    def forward(self, x):
        return self.head(self.gem_pool(self.backbone(x)))


# ── Mel transform ──────────────────────────────────────────────────────────────
import torchaudio

class MelTransform(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        sr = 32000
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=m["n_fft"], hop_length=m["hop_length"],
            n_mels=m["n_mels"], f_min=m["fmin"], f_max=m["fmax"],
            power=m["power"], norm=m["norm"], mel_scale=m["mel_scale"],
        )
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=m["top_db"])

    def forward(self, x):
        x = torch.nan_to_num(x.float(), nan=0.0)
        mel = torch.nan_to_num(self.db(self.mel(x)), nan=-80.0)
        mel = (mel + 80.0) / 80.0
        return mel.unsqueeze(1).expand(-1, 3, -1, -1)


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_cfg(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(cfg, ckpt_path, device):
    m = cfg["model"]
    model = SEDModel(
        backbone       = m["backbone"],
        num_classes    = 234,
        dropout        = m["dropout"],
        drop_path_rate = m["drop_path_rate"],
        gem_p_init     = m["gem_p_init"],
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def load_soundscape_clips(soundscape_dir, labels_csv, taxonomy_csv, clip_dur=20, sr=32000):
    """Load labeled soundscape clips with ground truth."""
    import csv
    # Load taxonomy
    with open(taxonomy_csv) as f:
        rows = list(csv.DictReader(f))
    species_cols = [r["primary_label"] for r in rows]

    # Load labels
    clips = []
    with open(labels_csv) as f:
        for row in csv.DictReader(f):
            fn       = row["filename"]
            end_sec  = int(row["end_time"]) if "end_time" in row else 20
            path     = Path(soundscape_dir) / fn
            if not path.exists():
                continue
            labels = np.array([float(row.get(c, 0)) for c in species_cols], dtype=np.float32)
            if labels.sum() == 0:
                continue
            clips.append({
                "path":     str(path),
                "filename": fn,
                "end_sec":  end_sec,
                "labels":   labels,
            })
    return clips, species_cols


def load_clip(path, end_sec, clip_dur=20, sr=32000):
    n_samples = clip_dur * sr
    start = max(0, end_sec - clip_dur) * sr
    try:
        audio, orig_sr = sf.read(path, start=start, frames=n_samples * 2,
                                  dtype="float32", always_2d=False)
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
    peak = np.abs(audio).max()
    if peak > 1e-7:
        audio = audio / peak
    return audio.astype(np.float32)


def plot_attention_map(mel_np, att_np, cls_np, species_cols, gt_labels,
                       pred_probs, filename, end_sec, out_path, top_k=5):
    """
    mel_np:      (n_mels, T) mel spectrogram
    att_np:      (n_cls, T) attention weights
    cls_np:      (n_cls, T) framewise logits
    gt_labels:   (n_cls,) ground truth
    pred_probs:  (n_cls,) predicted probabilities
    """
    # Select top-k classes by pred_probs OR ground truth
    gt_idx   = np.where(gt_labels > 0)[0]
    pred_idx = np.argsort(pred_probs)[::-1][:top_k]
    show_idx = np.union1d(gt_idx, pred_idx)[:top_k + len(gt_idx)]
    if len(show_idx) == 0:
        return

    n_show = len(show_idx)
    fig = plt.figure(figsize=(14, 3 + 2 * n_show))
    gs  = gridspec.GridSpec(n_show + 1, 1, hspace=0.4)

    # Mel spectrogram
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(mel_np, origin="lower", aspect="auto", cmap="magma",
               interpolation="nearest")
    ax0.set_title(f"{filename} | t=[{end_sec-20}s, {end_sec}s]", fontsize=9)
    ax0.set_ylabel("Mel bin")
    ax0.set_xticks([])

    T = mel_np.shape[1]
    t_axis = np.linspace(0, 20, T)

    for row_i, cls_i in enumerate(show_idx):
        ax = fig.add_subplot(gs[row_i + 1])
        name  = species_cols[cls_i]
        gt    = gt_labels[cls_i]
        prob  = pred_probs[cls_i]
        color = "green" if gt > 0 else "red"

        # Attention weight (blue fill)
        ax.fill_between(t_axis, att_np[cls_i], alpha=0.4, color="steelblue", label="attention")
        # Framewise prob (orange line)
        fw_prob = 1.0 / (1.0 + np.exp(-cls_np[cls_i]))
        ax.plot(t_axis, fw_prob, color="darkorange", linewidth=1.2, label="frame prob")

        ax.set_ylim(0, 1)
        ax.set_xlim(0, 20)
        ax.set_ylabel("weight", fontsize=7)
        title_color = color
        ax.set_title(
            f"{name}  |  GT={'✓' if gt > 0 else '✗'}  pred={prob:.2f}",
            color=title_color, fontsize=8
        )
        if row_i == n_show - 1:
            ax.set_xlabel("Time (s)")
        else:
            ax.set_xticks([])
        if row_i == 0:
            ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  required=True)
    parser.add_argument("--fold",    type=int, default=0)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_worst", type=int, default=20,
                        help="Number of worst-performing samples to visualize")
    args = parser.parse_args()

    cfg    = load_cfg(args.config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    ckpt_path = Path(cfg["output"]["dir"]) / f"fold{args.fold}_best.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return
    print(f"Loading model from {ckpt_path}")
    model    = load_model(cfg, ckpt_path, device)
    mel_tf   = MelTransform(cfg).to(device)

    # Load validation clips
    clips, species_cols = load_soundscape_clips(
        cfg["data"]["soundscape_dir"],
        cfg["data"]["soundscape_labels_csv"],
        cfg["data"]["taxonomy_csv"],
        clip_dur = cfg["model"]["clip_duration"],
    )
    print(f"Loaded {len(clips)} labeled soundscape clips")

    # Run inference on all clips
    results = []
    with torch.no_grad():
        for clip in clips:
            audio = load_clip(clip["path"], clip["end_sec"],
                              clip_dur=cfg["model"]["clip_duration"])
            wave  = torch.tensor(audio[None], dtype=torch.float32).to(device)
            mel   = mel_tf(wave)
            out   = model(mel)
            prob  = out["clipwise_prob"][0].cpu().numpy()   # (234,)
            att   = out["attention"][0].cpu().numpy()        # (234, T)
            cls   = out["framewise"][0].cpu().numpy()        # (234, T)
            mel_np = mel[0, 0].cpu().numpy()                 # (n_mels, T)

            gt = clip["labels"]
            # Per-class AUC proxy: loss for positive classes
            pos_idx = np.where(gt > 0)[0]
            if len(pos_idx) > 0:
                sample_loss = -np.log(prob[pos_idx] + 1e-7).mean()
            else:
                sample_loss = 0.0

            results.append({
                "clip":       clip,
                "prob":       prob,
                "att":        att,
                "cls":        cls,
                "mel_np":     mel_np,
                "loss":       sample_loss,
            })

    # Sort by loss descending → worst samples first
    results.sort(key=lambda x: x["loss"], reverse=True)
    worst = results[:args.n_worst]
    print(f"Visualizing top {len(worst)} worst samples (highest loss)...")

    for rank, r in enumerate(worst):
        clip    = r["clip"]
        fn_safe = clip["filename"].replace("/", "_").replace(".ogg", "")
        out_path = out_dir / f"rank{rank+1:03d}_loss{r['loss']:.3f}_{fn_safe}.png"
        plot_attention_map(
            mel_np      = r["mel_np"],
            att_np      = r["att"],
            cls_np      = r["cls"],
            species_cols = species_cols,
            gt_labels   = clip["labels"],
            pred_probs  = r["prob"],
            filename    = clip["filename"],
            end_sec     = clip["end_sec"],
            out_path    = str(out_path),
        )
        print(f"  [{rank+1}/{len(worst)}] loss={r['loss']:.3f}  {clip['filename']}")

    print(f"\nDone. Saved {len(worst)} attention maps to {out_dir}")


if __name__ == "__main__":
    main()
