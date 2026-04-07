"""Ensemble holdout eval — test soup model combinations + weighting.

Tests which SED combination beats the current best (0.9954).

Usage:
    python scripts/eval_ensemble_soup.py --gpu 1
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import librosa
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import load_config
from src.data.dataset import build_species_mapping
from src.model.classifier import PerchClassifier
from src.model.sed_model import SEDModel

HOLDOUT_CSV  = "configs/holdout_val_files.csv"
CONFIG       = "configs/default.yaml"
AUDIO_DIR    = "birdclef-2026/train_audio"
SR           = 32_000
CLIP_SAMPLES = SR * 5

PERCH_RUNS = [
    ("nohuman-label-pseudo",           "label_head",     "embeddings_cache_nohuman_label"),
    ("nohuman-label-soundscape-train", "label_head",     "embeddings_cache_nohuman_label"),
    ("nohuman-embedding-soundscape",   "embedding_head", "embeddings_cache_nohuman"),
]

SED_CHECKPOINTS = [
    {"name": "sed-b0-v5",
     "path": "checkpoints/sed-b0-v5/best_sed.pt",
     "backbone": "tf_efficientnet_b0.ns_jft_in1k", "n_mels": 224},
    {"name": "competitor",
     "path": "models/sed_weights/best_fold0.pt",
     "backbone": "tf_efficientnet_b0.ns_jft_in1k", "n_mels": 224},
    {"name": "soup-b0-v6",
     "path": "checkpoints/sed-b0-v6/soup_sed.pt",
     "backbone": "tf_efficientnet_b0.ns_jft_in1k", "n_mels": 224},
    {"name": "soup-v2s-v1",
     "path": "checkpoints/sed-v2s-v1/soup_sed.pt",
     "backbone": "tf_efficientnetv2_s.in21k_ft_in1k", "n_mels": 224},
]


class MelTransform(nn.Module):
    def __init__(self, n_mels=224, peak_norm=True):
        super().__init__()
        self.peak_norm = peak_norm
        self.mel = T.MelSpectrogram(
            sample_rate=SR, n_fft=2048, hop_length=512,
            n_mels=n_mels, f_min=0, f_max=16000,
            power=2.0, norm="slaney", mel_scale="htk",
        )
        self.db = T.AmplitudeToDB(stype="power", top_db=80.0)

    @torch.no_grad()
    def forward(self, waveforms):
        waveforms = torch.nan_to_num(waveforms.float(), nan=0.0)
        if self.peak_norm:
            peak = waveforms.abs().amax(dim=1, keepdim=True).clamp(min=1e-7)
            waveforms = waveforms / peak
        mel = torch.nan_to_num(self.db(self.mel(waveforms)), nan=-80.0)
        B = mel.shape[0]; flat = mel.reshape(B, -1)
        mn = flat.min(1, keepdim=True)[0].unsqueeze(-1)
        mx = flat.max(1, keepdim=True)[0].unsqueeze(-1)
        mel = torch.nan_to_num((mel - mn) / (mx - mn + 1e-7), nan=0.0)
        return mel.unsqueeze(1).repeat(1, 3, 1, 1)


def load_sed(path, backbone, n_mels, num_classes, device):
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    # normalise key names (checkpoints may use gem_pool or freq_pool)
    if any("gem_pool" in k for k in state):
        state = {k.replace("gem_pool", "freq_pool"): v for k, v in state.items()}
    model = SEDModel(
        backbone=backbone, num_classes=num_classes,
        in_chans=3, pretrained=False,
        drop_rate=0.1, use_gem=True, gem_p_init=3.0, n_mels=n_mels,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    mel_tf = MelTransform(n_mels=n_mels).to(device)
    mel_tf.eval()
    ep  = ckpt.get("epoch", "?")
    auc = ckpt.get("metrics", {}).get("macro_auc", "?")
    print(f"  Loaded {os.path.basename(path)}  epoch={ep}  val_auc={auc}")
    return model, mel_tf


def predict_sed_files(model, mel_tf, files, num_classes, device, batch_size=16):
    preds = []
    for fname in tqdm(files, ncols=80):
        try:
            audio, _ = librosa.load(os.path.join(AUDIO_DIR, fname), sr=SR, mono=True)
            audio = audio.astype(np.float32)
        except Exception:
            preds.append(np.zeros(num_classes, dtype=np.float32))
            continue
        n_clips = max(1, int(np.ceil(len(audio) / CLIP_SAMPLES)))
        clip_preds = []
        for b in range(0, n_clips, batch_size):
            batch = []
            for ci in range(b, min(b + batch_size, n_clips)):
                clip = audio[ci * CLIP_SAMPLES:(ci + 1) * CLIP_SAMPLES]
                if len(clip) < CLIP_SAMPLES:
                    clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
                batch.append(clip)
            t = torch.from_numpy(np.stack(batch)).to(device)
            with torch.no_grad():
                out  = model(mel_tf(t))
                logit = out["clipwise_logit"] if isinstance(out, dict) else (out[0] if isinstance(out, tuple) else out)
                prob = torch.sigmoid(logit).cpu().numpy()
            clip_preds.append(prob)
        preds.append(np.concatenate(clip_preds, axis=0).max(axis=0))
    return np.stack(preds).astype(np.float32)


def load_perch_preds(run_name, mode, cache_name, holdout_csv, species_to_idx,
                     num_classes):
    holdout       = pd.read_csv(holdout_csv)
    holdout_files = set(holdout["filename"].unique())
    file_to_label = dict(zip(holdout["filename"], holdout["primary_label"].astype(str)))

    mcsv = f"outputs/{cache_name}/manifest.csv"
    mf   = pd.read_csv(mcsv)
    mf   = mf[mf["source_file"].isin(holdout_files) & (mf["split"] == "holdout")].copy()
    mf["primary_label"] = mf["source_file"].map(file_to_label)
    mf   = mf.dropna(subset=["primary_label"])

    embs, labs, fnames = [], [], []
    for _, row in mf.iterrows():
        if not os.path.isfile(row["npy_path"]):
            continue
        embs.append(np.load(row["npy_path"]))
        labs.append(str(row["primary_label"]))
        fnames.append(row["source_file"])

    X = np.stack(embs).astype(np.float32)

    run_cfg_path = os.path.join("outputs", run_name, "config.yaml")
    config       = load_config(CONFIG)
    run_config   = load_config(run_cfg_path) if os.path.isfile(run_cfg_path) else config

    ckpt_path = os.path.join("checkpoints", run_name, "best_head")
    model     = PerchClassifier(
        perch_dir=config.model.perch_dir, num_classes=num_classes, mode=mode,
        hidden_dim=run_config.model.hidden_dim, dropout=0.0, embedding_dim=X.shape[1],
    )
    model.load_head(ckpt_path)

    clip_preds = []
    for start in range(0, len(X), 512):
        batch  = tf.constant(X[start:start + 512])
        logits = model.head(batch, training=False)
        out    = logits[0] if isinstance(logits, tuple) else logits
        clip_preds.append(tf.sigmoid(out).numpy())
    clip_preds = np.concatenate(clip_preds, axis=0)
    del model; tf.keras.backend.clear_session()

    df    = pd.DataFrame({"fname": fnames, "label": labs, "idx": range(len(fnames))})
    files = df["fname"].unique()
    file_preds = np.zeros((len(files), num_classes), dtype=np.float32)
    y          = np.zeros((len(files), num_classes), dtype=np.float32)
    for i, fname in enumerate(files):
        rows = df[df["fname"] == fname]["idx"].tolist()
        file_preds[i] = clip_preds[rows].mean(axis=0)
        sp = df[df["fname"] == fname]["label"].iloc[0]
        if sp in species_to_idx:
            y[i, species_to_idx[sp]] = 1.0

    print(f"  [{run_name}] {len(files)} files  species_w_pos={(y.sum(0) > 0).sum()}")
    return file_preds, np.where(y.sum(0) > 0)[0], files


def score(y, preds, swp):
    try:
        return float(roc_auc_score(y[:, swp], preds[:, swp], average="macro"))
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", default=None)
    args = p.parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    config = load_config(CONFIG)
    _, species_to_idx = build_species_mapping(config.data.sample_submission_csv)
    num_classes = len(species_to_idx)

    holdout       = pd.read_csv(HOLDOUT_CSV)
    files_ref     = holdout["filename"].unique()
    file_to_label = dict(zip(holdout["filename"], holdout["primary_label"].astype(str)))
    y_ref = np.zeros((len(files_ref), num_classes), dtype=np.float32)
    for i, fname in enumerate(files_ref):
        sp = file_to_label.get(fname, "")
        if sp in species_to_idx:
            y_ref[i, species_to_idx[sp]] = 1.0
    swp_ref = np.where(y_ref.sum(0) > 0)[0]
    print(f"Holdout: {len(files_ref)} files  species_w_pos={len(swp_ref)}\n")

    all_preds = {}

    # ── Perch × 3 ─────────────────────────────────────────────────────────────
    for run_name, mode, cache_name in PERCH_RUNS:
        ckpt_path = os.path.join("checkpoints", run_name, "best_head")
        if not (os.path.isfile(ckpt_path + ".weights.h5") or os.path.isfile(ckpt_path)):
            print(f"[{run_name}] checkpoint not found — skipping"); continue
        print(f"\n[{run_name}]")
        fp, _, _ = load_perch_preds(run_name, mode, cache_name, HOLDOUT_CSV,
                                    species_to_idx, num_classes)
        all_preds[run_name] = fp

    # ── SED models ────────────────────────────────────────────────────────────
    for sed_cfg in SED_CHECKPOINTS:
        if not os.path.isfile(sed_cfg["path"]):
            print(f"\n[{sed_cfg['name']}] not found — skipping"); continue
        print(f"\n[{sed_cfg['name']}] Loading …")
        model, mel_tf = load_sed(sed_cfg["path"], sed_cfg["backbone"],
                                  sed_cfg["n_mels"], num_classes, device)
        print(f"  Inference on {len(files_ref)} files …")
        fp = predict_sed_files(model, mel_tf, files_ref, num_classes, device)
        all_preds[sed_cfg["name"]] = fp
        del model, mel_tf; torch.cuda.empty_cache()

    # ── Build combos ──────────────────────────────────────────────────────────
    lp   = all_preds.get("nohuman-label-pseudo")
    ls   = all_preds.get("nohuman-label-soundscape-train")
    emb  = all_preds.get("nohuman-embedding-soundscape")
    v5   = all_preds.get("sed-b0-v5")
    cmp  = all_preds.get("competitor")
    sv6  = all_preds.get("soup-b0-v6")
    sv2s = all_preds.get("soup-v2s-v1")

    perch3 = (lp + ls + emb) / 3 if all(x is not None for x in [lp, ls, emb]) else None

    results = []

    def add(name, *arrays_weights):
        # accepts alternating (array, weight) pairs or just arrays
        if any(a is None for a in arrays_weights):
            return
        preds_sum = sum(arrays_weights)
        results.append((name, preds_sum / len(arrays_weights)))

    # Baselines (already known)
    if perch3 is not None:
        results.append(("perch×3",                         perch3))
    if perch3 is not None and v5 is not None:
        results.append(("perch×3 + v5",                    (lp+ls+emb+v5)/4))
    if perch3 is not None and cmp is not None:
        results.append(("perch×3 + competitor",            (lp+ls+emb+cmp)/4))
    if perch3 is not None and v5 is not None and cmp is not None:
        results.append(("perch×3 + v5 + competitor  [CURRENT BEST]",
                                                            (lp+ls+emb+v5+cmp)/5))

    # Soup combos — equal weight
    if perch3 is not None and sv6 is not None:
        results.append(("perch×3 + soup-b0-v6",            (lp+ls+emb+sv6)/4))
    if perch3 is not None and sv2s is not None:
        results.append(("perch×3 + soup-v2s",              (lp+ls+emb+sv2s)/4))
    if perch3 is not None and sv6 is not None and sv2s is not None:
        results.append(("perch×3 + soup-b0-v6 + soup-v2s", (lp+ls+emb+sv6+sv2s)/5))
    if perch3 is not None and v5 is not None and sv6 is not None:
        results.append(("perch×3 + v5 + soup-b0-v6",       (lp+ls+emb+v5+sv6)/5))
    if perch3 is not None and v5 is not None and sv2s is not None:
        results.append(("perch×3 + v5 + soup-v2s",         (lp+ls+emb+v5+sv2s)/5))
    if perch3 is not None and v5 is not None and sv6 is not None and sv2s is not None:
        results.append(("perch×3 + v5 + soup-b0-v6 + soup-v2s",
                                                            (lp+ls+emb+v5+sv6+sv2s)/6))

    # Weighted combos — downweight soup relative to v5
    if perch3 is not None and v5 is not None and sv6 is not None and sv2s is not None:
        # v5 weight=2, soup weight=1 each
        results.append(("perch×3 + v5(w=2) + soups(w=1)",
                         (lp+ls+emb + 2*v5 + sv6 + sv2s) / 7))
        # v5 weight=3, soup weight=1 each
        results.append(("perch×3 + v5(w=3) + soups(w=1)",
                         (lp+ls+emb + 3*v5 + sv6 + sv2s) / 8))

    # With competitor
    if perch3 is not None and v5 is not None and cmp is not None and sv6 is not None:
        results.append(("perch×3 + v5 + competitor + soup-b0-v6",
                                                            (lp+ls+emb+v5+cmp+sv6)/6))
    if perch3 is not None and v5 is not None and cmp is not None and sv2s is not None:
        results.append(("perch×3 + v5 + competitor + soup-v2s",
                                                            (lp+ls+emb+v5+cmp+sv2s)/6))
    if all(x is not None for x in [perch3, v5, cmp, sv6, sv2s]):
        results.append(("perch×3 + v5 + competitor + both-soups",
                                                            (lp+ls+emb+v5+cmp+sv6+sv2s)/7))

    # ── Score & print ─────────────────────────────────────────────────────────
    scored = [(name, score(y_ref, prd, swp_ref)) for name, prd in results]
    best   = max((s for _, s in scored if s), default=None)

    print(f"\n{'='*76}")
    print(f"  {'Model':<50}  Holdout AUC")
    print(f"{'='*76}")
    for name, s in sorted(scored, key=lambda x: x[1] or 0, reverse=True):
        marker = " ★" if s and s == best else ""
        print(f"  {name:<50}  {f'{s:.6f}' if s else 'N/A':>11}{marker}")
    print(f"{'='*76}")
    print(f"\nReference best (before soup):  0.995395  (perch×3 + v5 + competitor)")

    import json
    out = {
        "scores":         {n: round(s, 6) for n, s in scored if s},
        "best":           best,
        "n_files":        int(len(files_ref)),
        "species_w_pos":  int(len(swp_ref)),
    }
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/ensemble_soup_eval.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved → outputs/ensemble_soup_eval.json")


if __name__ == "__main__":
    main()
