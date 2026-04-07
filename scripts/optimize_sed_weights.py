"""Optimize SED ensemble weights using per-fold ONNX inference on train_soundscapes.

Uses the ground truth labels from train_soundscapes_labels.csv to find optimal
weights for each SED model in the ensemble.

Usage:
    python scripts/optimize_sed_weights.py
"""
import os
import sys
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio.transforms as T
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from itertools import product

# ── Config ────────────────────────────────────────────────────────────────────
SR = 32000
CLIP_SEC = 20
CLIP_SAMPLES = SR * CLIP_SEC
STRIDE = SR * 5  # 5s stride

TAXONOMY_CSV = 'birdclef-2026/taxonomy.csv'
LABELS_CSV   = 'birdclef-2026/train_soundscapes_labels.csv'
SS_DIR       = 'birdclef-2026/train_soundscapes'
WEIGHTS_DIR  = 'birdclef-2026/notebook resource/new direction/weights/sed'

# Models to evaluate (same as notebook config)
MODELS = [
    {'name': 'b0_r8_f2',  'onnx': f'{WEIGHTS_DIR}/sed_ns_b0_r8_fold2.onnx',  'backbone': 'tf_efficientnet_b0.ns_jft_in1k'},
    {'name': 'pvt_r5_f4', 'onnx': f'{WEIGHTS_DIR}/sed_ns_pvt_r5_fold4.onnx', 'backbone': 'pvt_v2_b0'},
    {'name': 'b0_r8_f3',  'onnx': f'{WEIGHTS_DIR}/sed_ns_b0_r8_fold3.onnx',  'backbone': 'tf_efficientnet_b0.ns_jft_in1k'},
]

N_MELS = 224
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 0
FMAX = 16000
TOP_DB = 80.0


class MelTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, f_min=FMIN, f_max=FMAX,
            power=2.0, norm='slaney', mel_scale='htk',
        )
        self.db = T.AmplitudeToDB(stype='power', top_db=TOP_DB)

    def forward(self, waveforms):
        waveforms = torch.nan_to_num(waveforms.float(), nan=0.0, posinf=0.0, neginf=0.0)
        mel = torch.nan_to_num(self.db(self.mel(waveforms)), nan=-80.0)
        B = mel.shape[0]
        flat = mel.reshape(B, -1)
        mn = flat.min(1, keepdim=True)[0].unsqueeze(-1)
        mx = flat.max(1, keepdim=True)[0].unsqueeze(-1)
        mel = torch.nan_to_num((mel - mn) / (mx - mn + 1e-7), nan=0.0)
        return mel.unsqueeze(1).expand(-1, 3, -1, -1)


def main():
    import onnxruntime as ort

    # Load taxonomy
    taxonomy = pd.read_csv(TAXONOMY_CSV)
    species_cols = taxonomy['primary_label'].astype(str).tolist()
    NUM_CLASSES = len(species_cols)
    label2idx = {s: i for i, s in enumerate(species_cols)}

    # Load ground truth
    labels_df = pd.read_csv(LABELS_CSV)

    def time_to_sec(t):
        parts = str(t).split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

    labels_df['end_sec'] = labels_df['end'].apply(time_to_sec)
    labels_df['row_id'] = labels_df['filename'].str.replace('.ogg', '') + '_' + labels_df['end_sec'].astype(str)

    gt_dict = {}
    for _, row in labels_df.iterrows():
        y = np.zeros(NUM_CLASSES, dtype=np.float32)
        for sp in str(row['primary_label']).split(';'):
            sp = sp.strip()
            if sp in label2idx:
                y[label2idx[sp]] = 1.0
        rid = row['row_id']
        if rid in gt_dict:
            gt_dict[rid] = np.maximum(gt_dict[rid], y)  # union of labels
        else:
            gt_dict[rid] = y

    print(f"GT labels: {len(gt_dict)} row_ids, {NUM_CLASSES} classes")

    # Load ONNX sessions
    mel_tf = MelTransform()
    mel_tf.eval()

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sessions = []
    for m in MODELS:
        if not os.path.isfile(m['onnx']):
            print(f"SKIP {m['name']}: {m['onnx']} not found")
            continue
        sess = ort.InferenceSession(m['onnx'], sess_opts, providers=['CPUExecutionProvider'])
        sessions.append((sess, m['name']))
        print(f"Loaded {m['name']}: {os.path.getsize(m['onnx'])/1e6:.1f}MB")

    if len(sessions) != len(MODELS):
        print("ERROR: Not all models loaded")
        return

    # Only run inference on soundscapes that have GT labels
    labeled_ss = set(labels_df['filename'].str.replace('.ogg', '').unique())
    ogg_files = sorted([f for f in Path(SS_DIR).glob('*.ogg') if f.stem in labeled_ss])
    print(f"\nRunning inference on {len(ogg_files)} labeled soundscapes (of {len(list(Path(SS_DIR).glob('*.ogg')))} total) with {len(sessions)} models...")

    all_preds = {name: [] for _, name in sessions}
    all_gt = []
    all_row_ids = []

    for ogg_path in tqdm(ogg_files, desc='Inference'):
        ss_id = ogg_path.stem
        try:
            audio, _ = sf.read(str(ogg_path), dtype='float32', always_2d=False)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
        except Exception:
            continue

        n_windows = min(len(audio) // STRIDE, 12)

        for w in range(n_windows):
            end_sec = (w + 1) * 5
            row_id = f"{ss_id}_{end_sec}"

            if row_id not in gt_dict:
                continue

            # Extract clip (20s or available)
            center = w * STRIDE + STRIDE // 2
            start = max(0, center - CLIP_SAMPLES // 2)
            end = start + CLIP_SAMPLES
            if end > len(audio):
                end = len(audio)
                start = max(0, end - CLIP_SAMPLES)

            clip = audio[start:end]
            if len(clip) < CLIP_SAMPLES:
                clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))

            # Compute mel once
            with torch.no_grad():
                mel = mel_tf(torch.from_numpy(clip).unsqueeze(0)).numpy()

            # Run each model
            for sess, name in sessions:
                probs = sess.run(['probs'], {'mel': mel})[0]  # (1, 234)
                all_preds[name].append(probs[0])

            all_gt.append(gt_dict[row_id])
            all_row_ids.append(row_id)

    # Convert to arrays
    Y_gt = np.array(all_gt)  # (N, 234)
    preds = {name: np.array(all_preds[name]) for name in all_preds}

    print(f"\nTotal samples: {len(Y_gt)}")
    print(f"Positive labels: {Y_gt.sum():.0f}")

    # Per-model AUC
    for name, P in preds.items():
        mask = Y_gt.sum(axis=0) > 0
        auc = roc_auc_score(Y_gt[:, mask], P[:, mask], average='macro')
        print(f"  {name} solo AUC: {auc:.5f}")

    # Grid search weights
    print("\n=== Weight optimization ===")
    best_auc = 0
    best_w = None
    names = [n for _, n in sessions]

    # Search: w1, w2, w3 in [0.1, 0.2, ..., 2.0]
    weight_range = np.arange(0.2, 2.1, 0.2)

    results = []
    for w1 in weight_range:
        for w2 in weight_range:
            for w3 in weight_range:
                total = w1 + w2 + w3
                ensemble = (w1 * preds[names[0]] + w2 * preds[names[1]] + w3 * preds[names[2]]) / total
                mask = Y_gt.sum(axis=0) > 0
                auc = roc_auc_score(Y_gt[:, mask], ensemble[:, mask], average='macro')
                results.append((auc, w1, w2, w3))
                if auc > best_auc:
                    best_auc = auc
                    best_w = (w1, w2, w3)

    results.sort(key=lambda x: -x[0])
    print(f"\nTop 10 weight combinations:")
    for auc, w1, w2, w3 in results[:10]:
        print(f"  w=[{w1:.1f}, {w2:.1f}, {w3:.1f}] → AUC={auc:.5f}")

    print(f"\n*** Best: {names[0]}={best_w[0]:.1f}, {names[1]}={best_w[1]:.1f}, {names[2]}={best_w[2]:.1f} → AUC={best_auc:.5f}")


if __name__ == '__main__':
    main()
