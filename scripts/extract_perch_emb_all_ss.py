"""Extract Perch v2 embeddings (1536-dim) for ALL train_soundscapes files.

Uses the full Perch v2 SavedModel (GPU-compatible) to extract embeddings for
every 5s window of every soundscape file.  Skips files already present in the
existing perch_labeled_ss.npz cache to avoid redundant work.

Output: outputs/perch_emb_all_ss.npz
  emb       (N, 1536)  Perch v2 embeddings
  logits    (N, 234)   BirdCLEF-2026 logits (from perch_teacher_all_ss.csv)
  filenames (N,)       .ogg filename for each window
  row_ids   (N,)       e.g. "BC2026_Train_0001_S08_..._5"

N = 10658 files × 12 windows = 127,896 rows.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/extract_perch_emb_all_ss.py
    CUDA_VISIBLE_DEVICES=1 python scripts/extract_perch_emb_all_ss.py --batch_files 8
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.audio import load_audio

# ── Paths ──────────────────────────────────────────────────────────────────────
SS_DIR        = "birdclef-2026/train_soundscapes"
TAXONOMY_CSV  = "birdclef-2026/taxonomy.csv"
TEACHER_CSV   = "outputs/perch_teacher_all_ss.csv"
LABELED_NPZ   = "outputs/perch_labeled_ss.npz"
PERCH_DIR     = "models/bird-vocalization-classifier-tensorflow2-perch_v2-v2"
OUTPUT_NPZ    = "outputs/perch_emb_all_ss.npz"

SR            = 32_000
CLIP_DUR      = 5
CLIP_SAMPLES  = SR * CLIP_DUR    # 160_000
N_WINDOWS     = 12               # 12 × 5s = 60s per file


def build_row_ids(filename: str, n_windows: int = N_WINDOWS) -> list[str]:
    ss_id = re.sub(r"\.ogg$", "", filename, flags=re.IGNORECASE)
    return [f"{ss_id}_{(i + 1) * CLIP_DUR}" for i in range(n_windows)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",      default=OUTPUT_NPZ)
    parser.add_argument("--batch_files", type=int, default=8,
                        help="Number of files per inference batch")
    parser.add_argument("--save_every",  type=int, default=500,
                        help="Save checkpoint every N files")
    parser.add_argument("--limit",       type=int, default=None)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    # ── Load taxonomy → species list ────────────────────────────────────────────
    taxonomy     = pd.read_csv(TAXONOMY_CSV)
    species_list = taxonomy["primary_label"].astype(str).tolist()
    n_classes    = len(species_list)
    sp2idx       = {s: i for i, s in enumerate(species_list)}

    # ── Load teacher logits CSV (234-dim probabilities) ─────────────────────────
    print("Loading Perch teacher logits CSV ...")
    teacher_df = pd.read_csv(TEACHER_CSV, index_col="row_id")
    # teacher_df columns are species codes in some order
    teacher_species = list(teacher_df.columns)
    # Mapping: species_list order → teacher col index
    sp_to_teacher = {sp: teacher_species.index(sp) for sp in species_list
                     if sp in teacher_species}
    teacher_probs = teacher_df.values.astype(np.float32)
    teacher_ridx  = {rid: i for i, rid in enumerate(teacher_df.index)}
    print(f"  Teacher CSV: {len(teacher_df)} rows, {len(teacher_species)} species")

    # ── Load labeled-file embedding cache (to skip re-extraction) ───────────────
    cached_rids = set()
    cached_emb_dict  = {}   # row_id → embedding
    if os.path.exists(LABELED_NPZ):
        print(f"Loading existing labeled cache: {LABELED_NPZ}")
        lab = np.load(LABELED_NPZ, allow_pickle=True)
        for rid, emb in zip(lab["row_ids"], lab["emb"]):
            cached_rids.add(str(rid))
            cached_emb_dict[str(rid)] = emb
        print(f"  {len(cached_rids)} windows already cached")

    # ── Enumerate all soundscape files ──────────────────────────────────────────
    all_files = sorted(Path(SS_DIR).glob("*.ogg"))
    if args.limit:
        all_files = all_files[:args.limit]
    print(f"Total soundscape files: {len(all_files)}")

    # Check for existing partial output
    done_files = set()
    if os.path.exists(args.output):
        print(f"Resuming from existing output: {args.output}")
        existing = np.load(args.output, allow_pickle=True)
        done_files = set(str(f) for f in existing["filenames"])
        print(f"  Already processed: {len(done_files) // N_WINDOWS} files")

    files_to_process = [f for f in all_files if f.name not in done_files]
    print(f"Files to process: {len(files_to_process)}")

    if not files_to_process:
        print("All files already processed.")
        return

    # ── Load Perch v2 SavedModel ─────────────────────────────────────────────────
    print(f"Loading Perch v2 SavedModel from {PERCH_DIR} ...")
    perch_model = tf.saved_model.load(PERCH_DIR)
    perch_infer = perch_model.signatures["serving_default"]
    print("  Perch model loaded.")

    # Warm-up pass
    dummy = tf.constant(np.zeros((1, CLIP_SAMPLES), dtype=np.float32))
    out = perch_infer(inputs=dummy)
    emb_key = "embedding" if "embedding" in out else list(out.keys())[0]
    print(f"  Embedding key: '{emb_key}', shape: {out[emb_key].shape}")

    # ── Accumulate output arrays ─────────────────────────────────────────────────
    out_emb   = []
    out_logits = []
    out_fnames = []
    out_rids  = []

    # Include existing data
    if done_files:
        ex = np.load(args.output, allow_pickle=True)
        out_emb.append(ex["emb"])
        out_logits.append(ex["logits"])
        out_fnames.extend([str(f) for f in ex["filenames"]])
        out_rids.extend([str(r) for r in ex["row_ids"]])

    total = len(files_to_process)
    t0 = time.time()

    for batch_start in range(0, total, args.batch_files):
        batch_files = files_to_process[batch_start:batch_start + args.batch_files]

        # Build clips for the whole batch
        batch_clips = []
        batch_meta  = []  # (fname, row_id) per clip

        for fpath in batch_files:
            audio = load_audio(str(fpath), SR)
            if audio is None:
                print(f"  WARNING: could not load {fpath.name}, filling with zeros")
                audio = np.zeros(N_WINDOWS * CLIP_SAMPLES, dtype=np.float32)

            for i in range(N_WINDOWS):
                s = i * CLIP_SAMPLES
                clip = audio[s:s + CLIP_SAMPLES]
                if len(clip) < CLIP_SAMPLES:
                    clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
                batch_clips.append(clip)
                ss_id = re.sub(r"\.ogg$", "", fpath.name, flags=re.IGNORECASE)
                rid = f"{ss_id}_{(i + 1) * CLIP_DUR}"
                batch_meta.append((fpath.name, rid))

        # TF inference
        x = tf.constant(np.stack(batch_clips, axis=0), dtype=tf.float32)
        result = perch_infer(inputs=x)
        batch_emb = result[emb_key].numpy()  # (batch*N_WINDOWS, 1536)

        # Build logits from teacher CSV
        batch_logits = np.zeros((len(batch_meta), n_classes), dtype=np.float32)
        for ci, (fname, rid) in enumerate(batch_meta):
            # Check per-row-id cache first
            if rid in cached_rids:
                batch_emb[ci] = cached_emb_dict[rid]  # use cached embedding
            if rid in teacher_ridx:
                teacher_row = teacher_probs[teacher_ridx[rid]]
                for sp, ti in sp_to_teacher.items():
                    batch_logits[ci, sp2idx[sp]] = teacher_row[ti]

        out_emb.append(batch_emb)
        out_logits.append(batch_logits)
        out_fnames.extend([m[0] for m in batch_meta])
        out_rids.extend([m[1] for m in batch_meta])

        # Progress
        files_done = batch_start + len(batch_files)
        elapsed = time.time() - t0
        rate = files_done / elapsed if elapsed > 0 else 0
        eta  = (total - files_done) / rate if rate > 0 else 0
        print(f"  [{files_done}/{total}]  {elapsed:.0f}s elapsed  ETA {eta/60:.1f}min", flush=True)

        # Periodic save
        if files_done % args.save_every < args.batch_files:
            _save(args.output, out_emb, out_logits, out_fnames, out_rids)
            print(f"  Checkpoint saved → {args.output}")

    # ── Final save ───────────────────────────────────────────────────────────────
    _save(args.output, out_emb, out_logits, out_fnames, out_rids)
    total_rows = sum(e.shape[0] for e in out_emb)
    print(f"\nDone! Saved {total_rows:,} windows → {args.output}")


def _save(path, out_emb, out_logits, out_fnames, out_rids):
    emb_arr    = np.concatenate(out_emb,    axis=0).astype(np.float32)
    logits_arr = np.concatenate(out_logits, axis=0).astype(np.float32)
    np.savez_compressed(
        path,
        emb       = emb_arr,
        logits    = logits_arr,
        filenames = np.array(out_fnames),
        row_ids   = np.array(out_rids),
    )


if __name__ == "__main__":
    main()
