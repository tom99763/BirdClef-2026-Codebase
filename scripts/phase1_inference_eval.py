"""Phase 1 Inference Enhancement Evaluation

Evaluates TTA (temporal shifts), temporal smoothing, and TopN postprocessing
on the soundscape validation split using the best sed-b0-v5 checkpoint.

Writes results to: outputs/phase1_inference_eval.json

Usage:
    python scripts/phase1_inference_eval.py --checkpoint checkpoints/sed-b0-v5/best_sed.pt
    python scripts/phase1_inference_eval.py --checkpoint checkpoints/sed-b0-v5/best_sed.pt --gpu 1
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

SR = 32_000
CLIP_SAMPLES = SR * 5
SS_LABELS_CSV = "birdclef-2026/train_soundscapes_labels.csv"
SS_AUDIO_DIR  = "birdclef-2026/train_soundscapes"
CONFIG        = "configs/sed_b0_v5.yaml"
SS_VAL_FRAC   = 0.2


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
        B = mel.shape[0]
        flat = mel.reshape(B, -1)
        mn = flat.min(1, keepdim=True)[0].unsqueeze(-1)
        mx = flat.max(1, keepdim=True)[0].unsqueeze(-1)
        mel = (mel - mn) / (mx - mn + 1e-7)
        return mel.unsqueeze(1).repeat(1, 3, 1, 1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/sed-b0-v5/best_sed.pt")
    p.add_argument("--gpu", default=None)
    p.add_argument("--config", default=CONFIG)
    return p.parse_args()


def get_val_files():
    ss_df = pd.read_csv(SS_LABELS_CSV)
    files = sorted(ss_df["filename"].unique())
    n_val = max(1, int(len(files) * SS_VAL_FRAC))
    return files[-n_val:], ss_df


def build_labels(val_files, ss_df, species_to_idx, num_classes):
    valid = set(species_to_idx.keys())
    rows = ss_df[ss_df["filename"].isin(val_files)].copy()
    rows = rows.sort_values(["filename", "start"]).reset_index(drop=True)

    def to_clip(s):
        h, m, sec = map(int, s.split(":"))
        return (h*3600 + m*60 + sec) // 5

    rows["clip_idx"] = rows["start"].apply(to_clip)
    rows["clip_key"] = rows["filename"] + "_c" + rows["clip_idx"].astype(str)

    y_dict = {}
    for _, row in rows.iterrows():
        k = row["clip_key"]
        if k not in y_dict:
            y_dict[k] = np.zeros(num_classes, dtype=np.float32)
        for sp in str(row["primary_label"]).split(";"):
            sp = sp.strip()
            if sp in valid:
                y_dict[k][species_to_idx[sp]] = 1.0

    keys = sorted(y_dict.keys())
    y = np.stack([y_dict[k] for k in keys])
    return keys, y


def get_clip(audio, start, length=CLIP_SAMPLES):
    """Extract clip with zero-padding if out of bounds."""
    if start < 0:
        pad_left = -start
        clip = audio[: max(0, length - pad_left)]
        clip = np.concatenate([np.zeros(pad_left, dtype=np.float32), clip])
    else:
        clip = audio[start: start + length]
    if len(clip) < length:
        clip = np.pad(clip, (0, length - len(clip)))
    return clip[:length]


def predict_file(model, mel_tf, audio, device, use_tta=False):
    """Run model on one soundscape file. Returns (n_clips, num_classes)."""
    total_clips = max(1, int(np.ceil(len(audio) / CLIP_SAMPLES)))
    all_preds = []

    for i in range(total_clips):
        start = i * CLIP_SAMPLES
        if use_tta:
            shifts = [0, -CLIP_SAMPLES // 2, CLIP_SAMPLES // 2]
            clip_preds = []
            for sh in shifts:
                clip = get_clip(audio, start + sh)
                t = torch.from_numpy(clip).unsqueeze(0).to(device)
                with torch.no_grad():
                    mel = mel_tf(t)
                    out = model(mel)
                    prob = out[0] if isinstance(out, tuple) else out
                clip_preds.append(prob.cpu().numpy()[0])
            all_preds.append(np.mean(clip_preds, axis=0))
        else:
            clip = get_clip(audio, start)
            t = torch.from_numpy(clip).unsqueeze(0).to(device)
            with torch.no_grad():
                mel = mel_tf(t)
                out = model(mel)
                prob = out[0] if isinstance(out, tuple) else out
            all_preds.append(prob.cpu().numpy()[0])

    return np.stack(all_preds)   # (n_clips, C)


def temporal_smoothing(preds, kernel=None):
    """Apply Gaussian-like smoothing kernel across clips (2024 3rd place)."""
    if kernel is None:
        kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float32)
    result = np.zeros_like(preds)
    for c in range(preds.shape[1]):
        result[:, c] = np.convolve(preds[:, c], kernel, mode="same")
    return result


def topn_postproc(preds, n=1):
    """Multiply each segment prob by file-level top-N prob (2025 2nd place, +0.011 LB)."""
    if n == 1:
        file_top = preds.max(axis=0, keepdims=True)  # (1, C)
    else:
        file_top = np.sort(preds, axis=0)[-n:].mean(axis=0, keepdims=True)
    return preds * file_top


def roc_safe(y, preds):
    valid = np.where(y.sum(0) > 0)[0]
    if len(valid) == 0:
        return None
    try:
        return float(roc_auc_score(y[:, valid], preds[:, valid], average="macro"))
    except Exception:
        return None


def main():
    args = parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = load_config(args.config)
    target_species, species_to_idx = build_species_mapping(config.data.sample_submission_csv)
    num_classes = len(target_species)

    val_files, ss_df = get_val_files()
    clip_keys, y = build_labels(val_files, ss_df, species_to_idx, num_classes)
    clip_keys_set = set(clip_keys)
    print(f"Val files: {len(val_files)}  clips: {len(clip_keys)}  "
          f"species with pos: {(y.sum(0)>0).sum()}/234")

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    if any("gem_pool" in k for k in state):
        state = {k.replace("gem_pool", "freq_pool"): v for k, v in state.items()}
    model = SEDModel(
        backbone="tf_efficientnet_b0.ns_jft_in1k",
        num_classes=num_classes, in_chans=3, pretrained=False,
        drop_rate=0.1, use_gem=True, gem_p_init=3.0, n_mels=224,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    mel_tf = MelTransform(n_mels=224).to(device)
    mel_tf.eval()
    print(f"  epoch={ckpt.get('epoch','?')}  stored_auc={ckpt.get('metrics',{}).get('macro_auc','?')}")

    # Collect predictions per file for each variant
    file_preds_base = {}
    file_preds_tta  = {}

    print("\nRunning inference …")
    for fname in tqdm(val_files, desc="files", ncols=80):
        apath = os.path.join(SS_AUDIO_DIR, fname)
        try:
            audio, _ = librosa.load(apath, sr=SR, mono=True)
            audio = audio.astype(np.float32)
        except Exception as e:
            print(f"  WARN: {fname}: {e}")
            continue
        file_preds_base[fname] = predict_file(model, mel_tf, audio, device, use_tta=False)
        file_preds_tta[fname]  = predict_file(model, mel_tf, audio, device, use_tta=True)

    # Build aligned arrays for each variant
    def align(file_preds_dict, postproc_fn=None):
        """Align predictions to clip_keys ordering."""
        preds_list, keys_matched = [], []
        for fname in val_files:
            if fname not in file_preds_dict:
                continue
            fp = file_preds_dict[fname]
            if postproc_fn is not None:
                fp = postproc_fn(fp)
            n_clips = fp.shape[0]
            for i in range(n_clips):
                k = f"{fname}_c{i}"
                if k in clip_keys_set:
                    preds_list.append(fp[i])
                    keys_matched.append(k)
        if not preds_list:
            return None, None
        preds_arr = np.stack(preds_list)
        key_to_idx = {k: i for i, k in enumerate(clip_keys)}
        y_matched = np.stack([y[key_to_idx[k]] for k in keys_matched])
        return preds_arr, y_matched

    variants = {}

    # Baseline
    preds, ym = align(file_preds_base)
    variants["baseline"] = roc_safe(ym, preds)

    # TTA only
    preds, ym = align(file_preds_tta)
    variants["tta_only"] = roc_safe(ym, preds)

    # Baseline + TopN
    preds, ym = align(file_preds_base, postproc_fn=topn_postproc)
    variants["baseline+topn"] = roc_safe(ym, preds)

    # Baseline + smoothing
    preds, ym = align(file_preds_base, postproc_fn=temporal_smoothing)
    variants["baseline+smooth"] = roc_safe(ym, preds)

    # TTA + TopN
    preds, ym = align(file_preds_tta, postproc_fn=topn_postproc)
    variants["tta+topn"] = roc_safe(ym, preds)

    # TTA + smoothing
    preds, ym = align(file_preds_tta, postproc_fn=temporal_smoothing)
    variants["tta+smooth"] = roc_safe(ym, preds)

    # TTA + smoothing + TopN (all combined)
    def smooth_then_topn(fp):
        return topn_postproc(temporal_smoothing(fp))
    preds, ym = align(file_preds_tta, postproc_fn=smooth_then_topn)
    variants["tta+smooth+topn"] = roc_safe(ym, preds)

    # Print results
    print(f"\n{'='*60}")
    print(f"  Phase 1 Inference Variants — SS Val AUC")
    print(f"{'='*60}")
    baseline = variants["baseline"] or 0.0
    for name, auc in sorted(variants.items(), key=lambda x: -(x[1] or 0)):
        delta = f"  ({auc - baseline:+.4f})" if auc and name != "baseline" else ""
        auc_str = f"{auc:.4f}" if auc else "N/A"
        print(f"  {name:<28}  {auc_str}{delta}")
    print(f"{'='*60}")

    # Save JSON
    result = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": ckpt.get("epoch", "?"),
        "n_val_files": len(val_files),
        "n_clips": len(clip_keys),
        "variants": {k: round(v, 6) if v else None for k, v in variants.items()},
        "best_variant": max(variants, key=lambda k: variants[k] or 0),
        "best_auc": round(max(v for v in variants.values() if v), 6),
        "baseline_auc": round(baseline, 6),
        "best_gain": round((max(v for v in variants.values() if v) - baseline), 6),
    }
    out_path = "outputs/phase1_inference_eval.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nBest variant  : {result['best_variant']}  AUC={result['best_auc']:.4f}  "
          f"gain={result['best_gain']:+.4f}")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
