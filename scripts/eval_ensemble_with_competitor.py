"""One-off: ensemble holdout eval adding competitor SED (best_fold0.pt) to v3.

Does NOT modify evaluate_ensemble_v3_holdout.py.
After completion, triggers run_improvement_plan.sh Phase 1-5.

Usage:
    python scripts/eval_ensemble_with_competitor.py --gpu 1
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

HOLDOUT_CSV    = "configs/holdout_val_files.csv"
CONFIG         = "configs/default.yaml"
AUDIO_DIR      = "birdclef-2026/train_audio"
SR             = 32_000
CLIP_SAMPLES   = SR * 5

PERCH_RUNS = [
    ("nohuman-label-pseudo",          "label_head",     "embeddings_cache_nohuman_label"),
    ("nohuman-label-soundscape-train", "label_head",     "embeddings_cache_nohuman_label"),
    ("nohuman-embedding-soundscape",   "embedding_head", "embeddings_cache_nohuman"),
]

SED_CHECKPOINTS = [
    {"name": "sed-b0-v5",   "path": "checkpoints/sed-b0-v5/best_sed.pt",
     "backbone": "tf_efficientnet_b0.ns_jft_in1k", "n_mels": 224},
    {"name": "competitor",  "path": "models/sed_weights/best_fold0.pt",
     "backbone": "tf_efficientnet_b0.ns_jft_in1k", "n_mels": 224},
]


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


def load_sed(path, backbone, n_mels, num_classes, device):
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
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
    print(f"  Loaded  epoch={ep}  val_auc={auc}")
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
                mel = mel_tf(t)
                out = model(mel)
                prob = (out[0] if isinstance(out, tuple) else out).cpu().numpy()
            clip_preds.append(prob)
        clip_preds = np.concatenate(clip_preds, axis=0)
        preds.append(clip_preds.max(axis=0))
    return np.stack(preds).astype(np.float32)


def load_perch_preds(run_name, mode, cache_name, holdout_csv, species_to_idx,
                     num_classes, checkpoints_dir, outputs_dir):
    holdout      = pd.read_csv(holdout_csv)
    holdout_files = set(holdout["filename"].unique())
    file_to_label = dict(zip(holdout["filename"], holdout["primary_label"].astype(str)))

    mcsv = f"{outputs_dir}/{cache_name}/manifest.csv"
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

    run_cfg_path = os.path.join(outputs_dir, run_name, "config.yaml")
    config       = load_config("configs/default.yaml")
    run_config   = load_config(run_cfg_path) if os.path.isfile(run_cfg_path) else config

    ckpt_path = os.path.join(checkpoints_dir, run_name, "best_head")
    model     = PerchClassifier(
        perch_dir=config.model.perch_dir,
        num_classes=num_classes,
        mode=mode,
        hidden_dim=run_config.model.hidden_dim,
        dropout=0.0,
        embedding_dim=X.shape[1],
    )
    model.load_head(ckpt_path)

    clip_preds = []
    for start in range(0, len(X), 512):
        batch  = tf.constant(X[start:start + 512])
        logits = model.head(batch, training=False)
        out    = logits[0] if isinstance(logits, tuple) else logits
        clip_preds.append(tf.sigmoid(out).numpy())
    clip_preds = np.concatenate(clip_preds, axis=0)

    del model
    tf.keras.backend.clear_session()

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

    print(f"  [{run_name}] {len(files)} files  species_w_pos={( y.sum(0)>0).sum()}")
    return file_preds, y, np.where(y.sum(0) > 0)[0], files


def auc(y, preds, swp):
    try:
        return float(roc_auc_score(y[:, swp], preds[:, swp], average="macro"))
    except Exception:
        return None


def pp(preds, thr=0.02):
    out = preds.copy(); out[out < thr] = 0.0; return out


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

    all_preds = {}

    # ── Perch × 3 ─────────────────────────────────────────────────────────────
    for run_name, mode, cache_name in PERCH_RUNS:
        ckpt_path = os.path.join("checkpoints", run_name, "best_head")
        if not (os.path.isfile(ckpt_path + ".weights.h5") or os.path.isfile(ckpt_path)):
            print(f"[{run_name}] checkpoint not found — skipping"); continue
        print(f"\n[{run_name}]")
        fp, _, _, _ = load_perch_preds(
            run_name, mode, cache_name, HOLDOUT_CSV, species_to_idx,
            num_classes, "checkpoints", "outputs"
        )
        all_preds[run_name] = fp

    # ── SED models ────────────────────────────────────────────────────────────
    for sed_cfg in SED_CHECKPOINTS:
        if not os.path.isfile(sed_cfg["path"]):
            print(f"\n[{sed_cfg['name']}] not found — skipping"); continue
        print(f"\n[{sed_cfg['name']}] Loading SED …")
        model, mel_tf = load_sed(
            sed_cfg["path"], sed_cfg["backbone"], sed_cfg["n_mels"], num_classes, device
        )
        print(f"  Running inference on {len(files_ref)} files …")
        fp = predict_sed_files(model, mel_tf, files_ref, num_classes, device)
        all_preds[sed_cfg["name"]] = fp
        del model, mel_tf
        torch.cuda.empty_cache()

    # ── Build combos ──────────────────────────────────────────────────────────
    lp  = all_preds.get("nohuman-label-pseudo")
    ls  = all_preds.get("nohuman-label-soundscape-train")
    emb = all_preds.get("nohuman-embedding-soundscape")
    sed = all_preds.get("sed-b0-v5")
    cmp = all_preds.get("competitor")

    results = []
    for name, prd in all_preds.items():
        results.append((name, prd))

    # Perch × 3
    if lp is not None and ls is not None and emb is not None:
        results.append(("perch×3",       (lp + ls + emb) / 3))

    # Perch×3 + our SED
    if lp is not None and ls is not None and emb is not None and sed is not None:
        results.append(("perch×3 + our-SED",  (lp + ls + emb + sed) / 4))

    # Perch×3 + competitor SED
    if lp is not None and ls is not None and emb is not None and cmp is not None:
        results.append(("perch×3 + competitor", (lp + ls + emb + cmp) / 4))

    # Perch×3 + both SEDs
    if lp is not None and ls is not None and emb is not None and sed is not None and cmp is not None:
        results.append(("perch×3 + both-SED",  (lp + ls + emb + sed + cmp) / 5))
        results.append(("perch×3 + both-SED+PP", pp((lp + ls + emb + sed + cmp) / 5)))

    # ── Score ─────────────────────────────────────────────────────────────────
    scored = [(name, auc(y_ref, prd, swp_ref)) for name, prd in results]

    best = max((s for _, s in scored if s), default=None)
    print(f"\n{'='*72}")
    print(f"  {'Model':<42}  {'Holdout AUC':>11}")
    print(f"{'='*72}")
    for name, score in scored:
        s      = f"{score:.4f}" if score else "  N/A"
        marker = " ★" if score and score == best else ""
        print(f"  {name:<42}  {s:>11}{marker}")
    print(f"{'='*72}")
    print(f"\nReference: competitor SED alone = 0.9883")
    print(f"Reference: our ensemble v3 best  = 0.9870")

    import json
    out = {
        "scores": {name: round(s, 6) for name, s in scored if s},
        "best": best,
        "n_files": int(len(files_ref)),
        "species_w_pos": int(len(swp_ref)),
    }
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/ensemble_with_competitor.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved → outputs/ensemble_with_competitor.json")


if __name__ == "__main__":
    main()
