"""Soundscape validation eval: compare all models on the 13-file soundscape val set.

Uses the same file-level split as sed-b0-v5 (last 20% of 66 soundscape files).
Evaluates Perch models from cache + SED from raw audio.

Outputs a side-by-side table:  Soundscape Val AUC  vs  Holdout AUC  vs  LB

Usage:
    python evaluate_soundscape_val.py
    python evaluate_soundscape_val.py --all_soundscapes   # use all 66 files
"""

import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torchaudio.transforms as T
import librosa
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.utils.config import load_config
from src.data.dataset import build_species_mapping
from src.model.classifier import PerchClassifier
from src.model.sed_model import SEDModel


PERCH_RUNS = [
    ("nohuman-label-pseudo",           "label_head",     "embeddings_cache_nohuman_label"),
    ("nohuman-label-soundscape-train",  "label_head",     "embeddings_cache_nohuman_label"),
    ("nohuman-embedding-soundscape",    "embedding_head", "embeddings_cache_nohuman"),
]
CONFIG         = "configs/default.yaml"
SS_LABELS_CSV  = "birdclef-2026/train_soundscapes_labels.csv"
SS_AUDIO_DIR   = "birdclef-2026/train_soundscapes"
SED_CHECKPOINT = "checkpoints/sed-b0-v5/best_sed.pt"
SS_VAL_FRAC    = 0.2   # matches sed-b0-v5 config
SR             = 32_000
CLIP_SAMPLES   = SR * 5

# Known holdout AUCs for reference table
HOLDOUT_AUCS = {
    "nohuman-label-pseudo":           0.9453,
    "nohuman-label-soundscape-train":  0.9550,
    "nohuman-embedding-soundscape":    None,   # TBD
    "sed-b0-v5":                       None,   # TBD
}
LB_SCORES = {
    "nohuman-label-pseudo":           "0.849(+PP)",
    "nohuman-label-soundscape-train":  "0.839",
    "nohuman-embedding-soundscape":    "—",
    "sed-b0-v5":                       "—",
}


class MelTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=SR, n_fft=2048, hop_length=512,
            n_mels=224, f_min=0, f_max=16000,
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
        mel_min = flat.min(1, keepdim=True)[0].unsqueeze(-1)
        mel_max = flat.max(1, keepdim=True)[0].unsqueeze(-1)
        mel = (mel - mel_min) / (mel_max - mel_min + 1e-7)
        return mel.unsqueeze(1).repeat(1, 3, 1, 1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",          default=CONFIG)
    p.add_argument("--checkpoints_dir", default="checkpoints")
    p.add_argument("--outputs_dir",     default="outputs")
    p.add_argument("--sed_checkpoint",  default=SED_CHECKPOINT)
    p.add_argument("--all_soundscapes", action="store_true",
                   help="Eval on all 66 soundscape files instead of val-split 13")
    return p.parse_args()


def get_val_files(all_soundscapes=False):
    ss_df = pd.read_csv(SS_LABELS_CSV)
    files = sorted(ss_df["filename"].unique())
    if all_soundscapes:
        return files, ss_df
    n_val = max(1, int(len(files) * SS_VAL_FRAC))
    return files[-n_val:], ss_df


def build_soundscape_labels(val_files, ss_df, species_to_idx, num_classes):
    """Build (N_clips, num_classes) multi-label matrix for soundscape clips."""
    tax = pd.read_csv("birdclef-2026/taxonomy.csv")
    # primary_label in soundscapes = inat_taxon_id = primary_label in taxonomy
    valid_labels = set(species_to_idx.keys())

    rows = ss_df[ss_df["filename"].isin(val_files)].copy()
    rows = rows.sort_values(["filename", "start"]).reset_index(drop=True)

    # Derive clip index from start time (5-sec clips)
    def start_to_clip(s):
        h, m, sec = map(int, s.split(":"))
        total_s = h * 3600 + m * 60 + sec
        return total_s // 5

    rows["clip_idx"] = rows["start"].apply(start_to_clip)
    rows["clip_key"]  = rows["filename"] + "_c" + rows["clip_idx"].astype(str)

    y_dict = {}
    for _, row in rows.iterrows():
        key = row["clip_key"]
        if key not in y_dict:
            y_dict[key] = np.zeros(num_classes, dtype=np.float32)
        for taxon_id in str(row["primary_label"]).split(";"):
            sp = taxon_id.strip()
            if sp in valid_labels:
                y_dict[key][species_to_idx[sp]] = 1.0

    clip_keys = sorted(y_dict.keys())
    y = np.stack([y_dict[k] for k in clip_keys])
    return clip_keys, y


def load_perch_embeddings(cache_name, clip_keys_set):
    mcsv = f"outputs/{cache_name}/manifest.csv"
    mf   = pd.read_csv(mcsv)
    mf   = mf[mf["split"] == "soundscape"].copy()

    # Build clip_key from source_file + clip offset in npy_path
    def to_clip_key(row):
        fname = row["source_file"]
        npy   = row["npy_path"]
        # npy ends in _c{N}.npy
        tag = os.path.basename(npy)
        import re
        m = re.search(r"_c(\d+)\.npy$", tag)
        if m:
            return f"{fname}_c{m.group(1)}"
        return None

    mf["clip_key"] = mf.apply(to_clip_key, axis=1)
    mf = mf[mf["clip_key"].isin(clip_keys_set)].copy()
    print(f"  [{cache_name}] {len(mf)} clips matched")

    embs, keys = [], []
    for _, row in mf.iterrows():
        if os.path.isfile(row["npy_path"]):
            embs.append(np.load(row["npy_path"]))
            keys.append(row["clip_key"])
    return np.stack(embs).astype(np.float32), keys


def predict_perch(model, X, batch_size=512):
    preds = []
    for start in range(0, len(X), batch_size):
        batch  = tf.constant(X[start: start + batch_size])
        logits = model.head(batch, training=False)
        out    = logits[0] if isinstance(logits, tuple) else logits
        preds.append(tf.sigmoid(out).numpy())
    return np.concatenate(preds, axis=0)


def predict_sed_clips(sed_model, mel_tf, val_files, clip_keys_set, batch_size=16):
    """Run SED on each soundscape file, return {clip_key: pred} dict."""
    sed_model.eval()
    results = {}

    for fname in tqdm(val_files, desc="SED soundscape", ncols=80):
        audio_path = os.path.join(SS_AUDIO_DIR, fname)
        try:
            audio, _ = librosa.load(audio_path, sr=SR, mono=True)
            audio = audio.astype(np.float32)
        except Exception as e:
            print(f"  WARN: {fname}: {e}")
            continue

        total_clips = max(1, int(np.ceil(len(audio) / CLIP_SAMPLES)))
        clips = []
        for i in range(total_clips):
            clip = audio[i * CLIP_SAMPLES: (i + 1) * CLIP_SAMPLES]
            if len(clip) < CLIP_SAMPLES:
                clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
            clips.append(clip)

        clip_preds = []
        for b in range(0, len(clips), batch_size):
            t = torch.from_numpy(np.stack(clips[b: b + batch_size]))
            with torch.no_grad():
                mel = mel_tf(t)
                out = sed_model(mel)
            clip_prob = out[0] if isinstance(out, tuple) else out
            clip_preds.append(clip_prob.cpu().numpy())
        clip_preds = np.concatenate(clip_preds, axis=0)

        for i in range(len(clips)):
            key = f"{fname}_c{i}"
            if key in clip_keys_set:
                results[key] = clip_preds[i]

    return results


def roc_auc_score_safe(y, preds):
    valid = np.where(y.sum(0) > 0)[0]
    if len(valid) == 0:
        return None
    try:
        return roc_auc_score(y[:, valid], preds[:, valid], average="macro")
    except Exception as e:
        print(f"  Scoring error: {e}")
        return None


def main():
    args = parse_args()
    config  = load_config(args.config)
    target_species, species_to_idx = build_species_mapping(config.data.sample_submission_csv)
    num_classes = len(target_species)

    val_files, ss_df = get_val_files(args.all_soundscapes)
    mode_label = "all 66 files" if args.all_soundscapes else f"val-split {len(val_files)} files"
    print(f"\nSoundscape eval on: {mode_label}")
    print(f"Species: {num_classes}\n")

    clip_keys, y = build_soundscape_labels(val_files, ss_df, species_to_idx, num_classes)
    clip_keys_set = set(clip_keys)
    valid_species = np.where(y.sum(0) > 0)[0]
    print(f"Clips: {len(clip_keys)}  species with positives: {len(valid_species)}/234\n")

    results = {}   # run_name → ss_val_auc

    # ── Perch models ──────────────────────────────────────────────────────────
    for run_name, mode, cache_name in PERCH_RUNS:
        ckpt_path    = os.path.join(args.checkpoints_dir, run_name, "best_head")
        run_cfg_path = os.path.join(args.outputs_dir, run_name, "config.yaml")

        if not (os.path.isfile(ckpt_path + ".weights.h5") or os.path.isfile(ckpt_path)):
            print(f"[{run_name}] checkpoint not found — skipping")
            continue

        print(f"[{run_name}] loading embeddings …")
        X, matched_keys = load_perch_embeddings(cache_name, clip_keys_set)

        # Reorder y to match matched_keys
        key_to_idx = {k: i for i, k in enumerate(clip_keys)}
        y_matched  = np.stack([y[key_to_idx[k]] for k in matched_keys])

        run_config = load_config(run_cfg_path) if os.path.isfile(run_cfg_path) else config
        model = PerchClassifier(
            perch_dir=config.model.perch_dir,
            num_classes=num_classes,
            mode=mode,
            hidden_dim=run_config.model.hidden_dim,
            dropout=0.0,
            embedding_dim=X.shape[1],
        )
        model.load_head(ckpt_path)
        preds = predict_perch(model, X)
        del model
        tf.keras.backend.clear_session()

        score = roc_auc_score_safe(y_matched, preds)
        results[run_name] = score
        print(f"  ss_val_auc = {score:.4f}\n" if score else "  ss_val_auc = N/A\n")

    # ── SED model ─────────────────────────────────────────────────────────────
    sed_label = "sed-b0-v5"
    if os.path.isfile(args.sed_checkpoint):
        print(f"[{sed_label}] loading checkpoint …")
        ckpt = torch.load(args.sed_checkpoint, map_location="cpu", weights_only=False)
        print(f"  epoch={ckpt.get('epoch','?')}  training_val={ckpt.get('metrics',{}).get('macro_auc','?')}")

        sed_model = SEDModel(
            backbone="tf_efficientnet_b0.ns_jft_in1k",
            num_classes=num_classes,
            in_chans=3,
            pretrained=False,
            drop_rate=0.1,
            use_gem=True,
            gem_p_init=3.0,
            n_mels=224,
        )
        # Remap competitor checkpoint keys: gem_pool → freq_pool
        state = ckpt["model_state_dict"]
        if any("gem_pool" in k for k in state):
            state = {k.replace("gem_pool", "freq_pool"): v for k, v in state.items()}
        sed_model.load_state_dict(state)
        sed_model.eval()
        mel_tf = MelTransform()
        mel_tf.eval()

        pred_dict = predict_sed_clips(sed_model, mel_tf, val_files, clip_keys_set)

        # Align predictions with clip_keys
        matched_keys = [k for k in clip_keys if k in pred_dict]
        key_to_idx   = {k: i for i, k in enumerate(clip_keys)}
        preds_arr = np.stack([pred_dict[k] for k in matched_keys])
        y_matched = np.stack([y[key_to_idx[k]] for k in matched_keys])
        print(f"  [{sed_label}] {len(matched_keys)}/{len(clip_keys)} clips matched")

        score = roc_auc_score_safe(y_matched, preds_arr)
        results[sed_label] = score
        print(f"  ss_val_auc = {score:.4f}\n" if score else "  ss_val_auc = N/A\n")
    else:
        print(f"[{sed_label}] checkpoint not found ({args.sed_checkpoint}) — skipping\n")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*76}")
    print(f"  {'Model':<42}  {'SS Val':>7}  {'Holdout':>8}  {'LB':>10}")
    print(f"{'='*76}")
    for run_name in list(results.keys()):
        ss   = results[run_name]
        hold = HOLDOUT_AUCS.get(run_name)
        lb   = LB_SCORES.get(run_name, "—")
        ss_s   = f"{ss:.4f}"   if ss   else "  N/A"
        hold_s = f"{hold:.4f}" if hold else "  TBD"
        print(f"  {run_name:<42}  {ss_s:>7}  {hold_s:>8}  {lb:>10}")
    print(f"{'='*76}")
    print(f"  SS Val = soundscape val AUC ({mode_label})")
    print(f"  Holdout = individual-recording holdout AUC (7037 files)")
    print(f"\nNote: Holdout AUC is generally ~0.04 higher than SS Val due to domain gap.")

    # Save
    log_path = "outputs/soundscape_val_eval.log"
    with open(log_path, "w") as f:
        f.write(f"{'='*76}\n")
        f.write(f"  {'Model':<42}  {'SS Val':>7}  {'Holdout':>8}  {'LB':>10}\n")
        f.write(f"{'='*76}\n")
        for run_name in results:
            ss   = results[run_name]
            hold = HOLDOUT_AUCS.get(run_name)
            lb   = LB_SCORES.get(run_name, "—")
            ss_s   = f"{ss:.4f}"   if ss   else "  N/A"
            hold_s = f"{hold:.4f}" if hold else "  TBD"
            f.write(f"  {run_name:<42}  {ss_s:>7}  {hold_s:>8}  {lb:>10}\n")
        f.write(f"{'='*76}\n")
    print(f"\nSaved → {log_path}")


if __name__ == "__main__":
    main()
