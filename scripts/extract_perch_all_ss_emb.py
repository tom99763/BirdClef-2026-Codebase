"""Extract Perch 1536-dim embeddings + 234-dim logits for ALL train soundscapes.

Saves outputs/perch_all_ss_emb.npz with:
  emb:       (N_total_windows, 1536) float32  — Perch embedding
  logits:    (N_total_windows, 234)  float32  — Perch species logits (target classes)
  row_ids:   (N_total_windows,)      str      — row_id format: {ss_id}_{offset_sec}
  file_list: (N_files,)              str      — soundscape filenames
  n_windows: (N_files,)              int      — windows per file

Required for SSM Noisy Student rounds 2-4 (pseudo-labeled unlabeled soundscapes).

Usage:
  python scripts/extract_perch_all_ss_emb.py [--output PATH] [--limit N]
  # Run in background with nohup:
  nohup python scripts/extract_perch_all_ss_emb.py > outputs/logs/perch_all_ss_emb.log 2>&1 &
"""

import argparse
import gc
import glob
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import build_species_mapping

PERCH_DIR    = "models/bird-vocalization-classifier-tensorflow2-perch_v2-v2"
SS_DIR       = "birdclef-2026/train_soundscapes"
OUTPUT_NPZ   = "outputs/perch_all_ss_emb.npz"
TAXONOMY_CSV = "birdclef-2026/taxonomy.csv"
SAMPLE_SUB   = "birdclef-2026/sample_submission.csv"
SR           = 32_000
CLIP_DUR     = 5
CLIP_SAMPLES = SR * CLIP_DUR
SAVE_EVERY   = 200    # flush accumulated arrays every N files


def load_audio(path, sr=SR):
    try:
        audio, orig_sr = sf.read(path, dtype='float32', always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if orig_sr != sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
        return audio.astype(np.float32)
    except Exception as e:
        print(f"  ERROR loading {path}: {e}")
        return None


def extract_file(filepath, sig, label_indices, n_perch_classes, max_windows=12):
    """Extract embeddings + logits for one soundscape file.
    Returns (row_ids, emb_arr, logit_arr) or ([], None, None) on error.
    """
    ss_id = Path(filepath).stem
    audio = load_audio(filepath)
    if audio is None or len(audio) < CLIP_SAMPLES:
        return [], None, None

    n_segs = min(len(audio) // CLIP_SAMPLES, max_windows)
    if n_segs == 0:
        return [], None, None

    clips = np.stack([
        audio[i * CLIP_SAMPLES:(i + 1) * CLIP_SAMPLES]
        for i in range(n_segs)
    ], axis=0).astype(np.float32)

    row_ids = [f"{ss_id}_{(i + 1) * CLIP_DUR}" for i in range(n_segs)]

    # Run Perch in small batches
    embs   = []
    logits = []
    batch_size = 8
    for start in range(0, n_segs, batch_size):
        batch = tf.constant(clips[start:start + batch_size], dtype=tf.float32)
        out   = sig(inputs=batch)
        emb   = out["embedding"].numpy()       # (B, 1536)
        lab   = out["label"].numpy()           # (B, N_perch)
        # Pad and gather target species
        lab_pad = np.concatenate([lab, np.zeros((len(lab), 1), np.float32)], axis=1)
        lab_234 = lab_pad[:, label_indices]    # (B, 234)
        embs.append(emb)
        logits.append(lab_234)

    emb_arr   = np.concatenate(embs,   axis=0).astype(np.float32)   # (n_segs, 1536)
    logit_arr = np.concatenate(logits, axis=0).astype(np.float32)   # (n_segs, 234)
    return row_ids, emb_arr, logit_arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',   default=OUTPUT_NPZ)
    parser.add_argument('--ss_dir',   default=SS_DIR)
    parser.add_argument('--perch_dir',default=PERCH_DIR)
    parser.add_argument('--limit',    type=int, default=None)
    parser.add_argument('--gpu',      default='1', help='GPU index')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    # ── Load Perch ─────────────────────────────────────────────────────────────
    print(f"Loading Perch from {args.perch_dir} ...")
    perch = tf.saved_model.load(args.perch_dir)
    sig   = perch.signatures["serving_default"]
    # Probe for label indices
    target_species, _ = build_species_mapping(SAMPLE_SUB)
    labels_csv   = os.path.join(args.perch_dir, "assets", "labels.csv")
    bc_labels    = pd.read_csv(labels_csv).reset_index()
    bc_labels.rename(columns={'inat2024_fsd50k': 'scientific_name', 'index': 'bc_index'}, inplace=True)

    taxonomy     = pd.read_csv(TAXONOMY_CSV)
    sp2sci       = dict(zip(taxonomy['primary_label'].astype(str),
                             taxonomy['scientific_name'].astype(str)))
    sci2bc       = dict(zip(bc_labels['scientific_name'], bc_labels['bc_index']))

    # Probe output shape
    dummy = sig(inputs=tf.zeros((1, CLIP_SAMPLES), dtype=tf.float32))
    n_perch_classes = dummy['label'].shape[1]
    print(f"Perch label classes: {n_perch_classes}")

    label_indices = []
    for sp in target_species:
        sci = sp2sci.get(str(sp), '')
        idx = sci2bc.get(sci, n_perch_classes)  # OOV → last index (will pad)
        label_indices.append(idx)
    label_indices = np.array(label_indices, dtype=np.int32)
    print(f"Mapped {(label_indices < n_perch_classes).sum()}/{len(label_indices)} species to Perch labels")

    # ── Find soundscape files ──────────────────────────────────────────────────
    ogg_files = sorted(glob.glob(os.path.join(args.ss_dir, '*.ogg')))
    if args.limit:
        ogg_files = ogg_files[:args.limit]
    n_files = len(ogg_files)
    print(f"\nProcessing {n_files} soundscape files → {args.output}")

    # Resume from checkpoint if partial output exists
    done_files = set()
    chunk_emb, chunk_logits, chunk_rids = [], [], []
    chunk_fnames, chunk_nwindows = [], []
    all_emb_rows = 0

    ckpt_path = args.output.replace('.npz', '_ckpt.npz')
    if os.path.exists(ckpt_path):
        try:
            ckpt = np.load(ckpt_path, allow_pickle=True)
            done_files = set(ckpt['file_list'].astype(str))
            all_emb_rows = len(ckpt['emb'])
            print(f"Resuming from checkpoint: {len(done_files)} files done")
        except Exception as e:
            print(f"Warning: could not load checkpoint ({e}), starting fresh")

    t0 = time.time()
    n_done = 0

    for fi, filepath in enumerate(ogg_files):
        fname = os.path.basename(filepath)
        if fname in done_files:
            n_done += 1
            continue

        row_ids, emb_arr, logit_arr = extract_file(filepath, sig, label_indices, n_perch_classes)
        if emb_arr is None:
            continue

        chunk_emb.append(emb_arr)
        chunk_logits.append(logit_arr)
        chunk_rids.extend(row_ids)
        chunk_fnames.append(fname)
        chunk_nwindows.append(len(row_ids))
        all_emb_rows += len(row_ids)
        n_done += 1

        if (fi + 1) % SAVE_EVERY == 0 or fi == n_files - 1:
            if chunk_emb:
                # Save checkpoint (append if exists)
                if os.path.exists(ckpt_path):
                    old = np.load(ckpt_path, allow_pickle=True)
                    combined_emb   = np.concatenate([old['emb'],       np.concatenate(chunk_emb)],   axis=0)
                    combined_log   = np.concatenate([old['logits'],    np.concatenate(chunk_logits)], axis=0)
                    combined_rids  = np.concatenate([old['row_ids'],   np.array(chunk_rids)])
                    combined_flist = np.concatenate([old['file_list'], np.array(chunk_fnames)])
                    combined_nw    = np.concatenate([old['n_windows'], np.array(chunk_nwindows)])
                else:
                    combined_emb   = np.concatenate(chunk_emb,   axis=0)
                    combined_log   = np.concatenate(chunk_logits, axis=0)
                    combined_rids  = np.array(chunk_rids)
                    combined_flist = np.array(chunk_fnames)
                    combined_nw    = np.array(chunk_nwindows)

                np.savez_compressed(ckpt_path,
                                    emb=combined_emb, logits=combined_log,
                                    row_ids=combined_rids, file_list=combined_flist,
                                    n_windows=combined_nw)
                chunk_emb, chunk_logits, chunk_rids = [], [], []
                chunk_fnames, chunk_nwindows = [], []
                gc.collect()

            elapsed = time.time() - t0
            rate    = n_done / elapsed if elapsed > 0 else 1
            eta     = (n_files - n_done) / rate / 60 if rate > 0 else 0
            print(f"  [{n_done}/{n_files}]  {all_emb_rows:,} windows  "
                  f"{rate:.1f} files/s  ETA={eta:.0f}min")

    # Copy checkpoint to final output
    if os.path.exists(ckpt_path):
        import shutil
        shutil.copy(ckpt_path, args.output)
        final = np.load(args.output, allow_pickle=True)
        print(f"\nDone! {len(final['emb']):,} windows from {len(final['file_list'])} files → {args.output}")
        print(f"Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
