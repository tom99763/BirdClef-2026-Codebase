"""Evaluate a SED checkpoint on holdout with geographic species masking.

Geographic mask modes:
  none          — no masking (baseline, same as eval_sed_holdout.py)
  ss_hard       — soundscape species: 1.0, others: 0.0
  ss_soft       — soundscape species: 1.0, others: --soft_factor (default 0.1)
  sa_weighted   — continuous: pred *= max(in_soundscape, sa_fraction)

Usage:
    python scripts/eval_geo_holdout.py \
        --checkpoint checkpoints/sed-b0-v5/best_sed.pt \
        --config configs/sed_b0_v5.yaml \
        --run_name sed-b0-v5 \
        --mask_mode ss_soft --soft_factor 0.1
"""

import argparse
import json
import os
import sys

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio.transforms as T
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import load_config
from src.data.dataset import build_species_mapping
from src.model.sed_model import SEDModel

HOLDOUT_CSV = "configs/holdout_val_files.csv"
AUDIO_DIR   = "birdclef-2026/train_audio"
GEO_MASK    = "outputs/geo_mask.csv"
SR          = 32_000
CLIP_SAMPLES = SR * 5


class MelTransform(nn.Module):
    def __init__(self, n_mels=224):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=SR, n_fft=2048, hop_length=512,
            n_mels=n_mels, f_min=0, f_max=16000,
            power=2.0, norm="slaney", mel_scale="htk",
        )
        self.db = T.AmplitudeToDB(stype="power", top_db=80.0)

    @torch.no_grad()
    def forward(self, waveforms):
        peak = waveforms.abs().amax(dim=1, keepdim=True).clamp(min=1e-7)
        waveforms = waveforms / peak
        mel = self.db(self.mel(waveforms))
        B   = mel.shape[0]
        flat = mel.reshape(B, -1)
        mn  = flat.min(1, keepdim=True)[0].unsqueeze(-1)
        mx  = flat.max(1, keepdim=True)[0].unsqueeze(-1)
        mel = (mel - mn) / (mx - mn + 1e-7)
        return mel.unsqueeze(1).repeat(1, 3, 1, 1)


def build_geo_weights(geo_mask_path, species_to_idx, mask_mode, soft_factor):
    """Return a numpy array of shape (num_classes,) with per-species weights."""
    num_classes = len(species_to_idx)
    weights = np.ones(num_classes, dtype=np.float32)

    if mask_mode == "none":
        return weights

    if not os.path.exists(geo_mask_path):
        print(f"WARN: geo_mask not found at {geo_mask_path}, skipping geo filter")
        return weights

    geo = pd.read_csv(geo_mask_path)
    geo["primary_label"] = geo["primary_label"].astype(str)
    geo_lookup = dict(zip(geo["primary_label"], geo.itertuples(index=False)))

    for sp, idx in species_to_idx.items():
        sp_str = str(sp)
        if sp_str not in geo_lookup:
            continue
        row = geo_lookup[sp_str]

        if mask_mode == "ss_hard":
            weights[idx] = 1.0 if row.in_soundscape else 0.0
        elif mask_mode == "ss_soft":
            weights[idx] = 1.0 if row.in_soundscape else float(soft_factor)
        elif mask_mode == "sa_weighted":
            weights[idx] = float(row.geo_score)

    in_ss  = sum(1 for w in weights if w == 1.0)
    masked = sum(1 for w in weights if w < 0.5)
    print(f"  Geo weights — full(1.0): {in_ss}  masked(<0.5): {masked}  "
          f"mean: {weights.mean():.3f}")
    return weights


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--config",      required=True)
    p.add_argument("--run_name",    default=None)
    p.add_argument("--mask_mode",   default="none",
                   choices=["none", "ss_hard", "ss_soft", "sa_weighted"])
    p.add_argument("--soft_factor", type=float, default=0.1,
                   help="Weight for out-of-range species in ss_soft mode")
    p.add_argument("--geo_mask",    default=GEO_MASK)
    p.add_argument("--holdout_csv", default=HOLDOUT_CSV)
    p.add_argument("--audio_dir",   default=AUDIO_DIR)
    p.add_argument("--gpu",         default=None)
    p.add_argument("--batch_size",  type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config             = load_config(args.config)
    _, species_to_idx  = build_species_mapping(config.data.sample_submission_csv)
    num_classes        = len(species_to_idx)
    base_run           = args.run_name or config.experiment.name
    suffix             = f"_geo-{args.mask_mode}" + (
                             f"-sf{args.soft_factor:.2f}" if args.mask_mode == "ss_soft" else ""
                         )
    run_name           = base_run + suffix

    # ── Geo weights ───────────────────────────────────────────────────────────
    print(f"\nMask mode: {args.mask_mode}")
    geo_w = build_geo_weights(
        args.geo_mask, species_to_idx, args.mask_mode, args.soft_factor
    )

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading: {args.checkpoint}")
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    if any("gem_pool" in k for k in state):
        state = {k.replace("gem_pool", "freq_pool"): v for k, v in state.items()}

    model = SEDModel(
        backbone    = config.model.backbone,
        num_classes = num_classes,
        in_chans    = config.model.get("in_chans", 3),
        pretrained  = False,
        drop_rate   = config.model.get("dropout", 0.1),
        use_gem     = config.model.get("use_gem", True),
        gem_p_init  = config.model.get("gem_p_init", 3.0),
        n_mels      = config.mel.n_mels,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    mel_tf = MelTransform(n_mels=config.mel.n_mels).to(device)
    mel_tf.eval()
    print(f"  epoch={ckpt.get('epoch','?')}  "
          f"ss_val_auc={ckpt.get('metrics',{}).get('macro_auc','?')}")

    # ── Load holdout ──────────────────────────────────────────────────────────
    holdout     = pd.read_csv(args.holdout_csv)
    files       = holdout["filename"].unique()
    file_to_lbl = dict(zip(holdout["filename"], holdout["primary_label"].astype(str)))
    print(f"\nHoldout: {len(files)} files  ({holdout['primary_label'].nunique()} species)")

    y = np.zeros((len(files), num_classes), dtype=np.float32)
    for i, fname in enumerate(files):
        sp = file_to_lbl.get(fname, "")
        if sp in species_to_idx:
            y[i, species_to_idx[sp]] = 1.0
    species_with_pos = np.where(y.sum(0) > 0)[0]
    print(f"  Species with positives: {len(species_with_pos)}/234")

    # ── Inference ─────────────────────────────────────────────────────────────
    print("\nRunning holdout inference …")
    file_preds = []

    for fname in tqdm(files, desc="holdout", ncols=80):
        audio_path = os.path.join(args.audio_dir, fname)
        try:
            audio, _ = librosa.load(audio_path, sr=SR, mono=True)
            audio = audio.astype(np.float32)
        except Exception as e:
            print(f"  WARN: {fname}: {e}")
            file_preds.append(np.zeros(num_classes, dtype=np.float32))
            continue

        n_clips    = max(1, int(np.ceil(len(audio) / CLIP_SAMPLES)))
        clip_preds = []

        for b_start in range(0, n_clips, args.batch_size):
            batch_clips = []
            for ci in range(b_start, min(b_start + args.batch_size, n_clips)):
                clip = audio[ci * CLIP_SAMPLES: (ci + 1) * CLIP_SAMPLES]
                if len(clip) < CLIP_SAMPLES:
                    clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
                batch_clips.append(clip)
            batch_t = torch.from_numpy(np.stack(batch_clips)).to(device)
            with torch.no_grad():
                mel = mel_tf(batch_t)
                out = model(mel)
                prob = (out[0] if isinstance(out, tuple) else out).cpu().numpy()
            clip_preds.append(prob)

        clip_preds = np.concatenate(clip_preds, axis=0)
        file_preds.append(clip_preds.max(axis=0))

    file_preds = np.stack(file_preds).astype(np.float32)

    # ── Apply geo mask ────────────────────────────────────────────────────────
    if args.mask_mode != "none":
        file_preds = file_preds * geo_w[np.newaxis, :]

    # ── Score ─────────────────────────────────────────────────────────────────
    try:
        holdout_auc = float(roc_auc_score(
            y[:, species_with_pos],
            file_preds[:, species_with_pos],
            average="macro",
        ))
    except Exception as e:
        print(f"Scoring error: {e}")
        holdout_auc = None

    # Also score with geo mask applied only to out-of-soundscape species in scoring
    print(f"\n{'='*60}")
    print(f"  {run_name}")
    print(f"  mask_mode       : {args.mask_mode}"
          + (f"  soft_factor={args.soft_factor}" if args.mask_mode == "ss_soft" else ""))
    print(f"  Holdout ROC-AUC : {holdout_auc:.4f}" if holdout_auc else "  Holdout ROC-AUC : ERROR")
    print(f"  Files           : {len(files)}")
    print(f"{'='*60}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir  = os.path.join("outputs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sed_holdout_eval.json")
    result = {
        "run_name":      run_name,
        "base_run":      base_run,
        "checkpoint":    args.checkpoint,
        "mask_mode":     args.mask_mode,
        "soft_factor":   args.soft_factor,
        "epoch":         ckpt.get("epoch", "?"),
        "ss_val_auc":    ckpt.get("metrics", {}).get("macro_auc"),
        "holdout_auc":   round(holdout_auc, 6) if holdout_auc else None,
        "n_files":       int(len(files)),
        "species_w_pos": int(len(species_with_pos)),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {out_path}")

    return holdout_auc


if __name__ == "__main__":
    main()
