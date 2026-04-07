"""Build a unified Perch embedding cache for all 66 labeled soundscape files.

Merges the existing perch_cache_extended.npz (59 files) with the 7 missing
labeled files by running Perch v2 inference on their audio.

Teacher logits (234-dim) for all files are sourced from perch_teacher_all_ss.csv
(our trained nohuman head, probability outputs → converted back to logit space).

Output: outputs/perch_labeled_ss.npz
  emb      (N_windows, 1536)  raw Perch v2 embeddings, flat (window-level)
  logits   (N_windows, 234)   teacher logits (logit-space), flat
  labels   (N_windows, 234)   binary species labels, flat
  filenames (N_windows,)      .ogg filename per window
  row_ids   (N_windows,)      e.g. "BC2026_Train_0001_S08_..._5"
  n_windows (N_files,)        windows per file (for reshaping later)
  file_list (N_files,)        unique filenames in order

Usage:
    python scripts/extract_ss_labeled_embeddings.py
    python scripts/extract_ss_labeled_embeddings.py --output outputs/perch_labeled_ss.npz
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.audio import load_audio

# ── Paths ─────────────────────────────────────────────────────────────────────
SS_DIR        = "birdclef-2026/train_soundscapes"
LABELS_CSV    = "birdclef-2026/train_soundscapes_labels.csv"
TAXONOMY_CSV  = "birdclef-2026/taxonomy.csv"
TEACHER_CSV   = "outputs/perch_teacher_all_ss.csv"
EXISTING_NPZ  = "outputs/perch_cache_extended.npz"
PERCH_DIR     = "models/bird-vocalization-classifier-tensorflow2-perch_v2-v2"
OUTPUT_NPZ    = "outputs/perch_labeled_ss.npz"

SR            = 32_000
CLIP_DUR      = 5
CLIP_SAMPLES  = SR * CLIP_DUR   # 160_000
WINDOWS_PER_SEGMENT = 12        # max windows used per 60-second segment


# ── Helpers ───────────────────────────────────────────────────────────────────

def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Convert probability → logit (inverse sigmoid)."""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def build_label_matrix(
    df_file: pd.DataFrame,
    species_list: list[str],
) -> np.ndarray:
    """Convert label rows to binary matrix (T, n_classes).

    Args:
        df_file:      rows for one file, sorted by start time.
        species_list: ordered list of 234 species codes.
    Returns:
        labels: (T, n_classes) float32 binary array.
    """
    sp2idx = {s: i for i, s in enumerate(species_list)}
    T = len(df_file)
    labels = np.zeros((T, len(species_list)), dtype=np.float32)
    for t, row in enumerate(df_file.itertuples()):
        for sp in str(row.primary_label).split(";"):
            sp = sp.strip()
            if sp in sp2idx:
                labels[t, sp2idx[sp]] = 1.0
    return labels


def extract_embeddings_from_audio(
    filepath: str,
    perch_infer,
    n_windows: int,
) -> np.ndarray:
    """Run Perch v2 on first n_windows 5-second clips of the file.

    Returns:
        emb: (n_windows, 1536) float32
    """
    audio = load_audio(filepath, SR)
    if audio is None:
        raise RuntimeError(f"Failed to load audio: {filepath}")

    n_segs = min(n_windows, len(audio) // CLIP_SAMPLES)
    clips  = np.stack(
        [audio[i * CLIP_SAMPLES:(i + 1) * CLIP_SAMPLES] for i in range(n_segs)],
        axis=0,
    )  # (n_segs, 160000)

    out  = perch_infer(inputs=tf.constant(clips, dtype=tf.float32))
    embs = out["embedding"].numpy()                        # (n_segs, 1536)

    if n_segs < n_windows:
        # Pad with zeros if audio shorter than label count
        pad  = np.zeros((n_windows - n_segs, 1536), dtype=np.float32)
        embs = np.concatenate([embs, pad], axis=0)

    return embs


def build_row_ids(filename: str, n_windows: int) -> list[str]:
    """Generate row_ids matching perch_teacher_all_ss.csv naming."""
    ss_id = re.sub(r"\.ogg$", "", filename, flags=re.IGNORECASE)
    return [f"{ss_id}_{(i + 1) * CLIP_DUR}" for i in range(n_windows)]


# ── Main ──────────────────────────────────────────────────────────────────────

def main(output_path: str = OUTPUT_NPZ) -> None:
    # 1. Load metadata ─────────────────────────────────────────────────────────
    print("Loading metadata...")
    labels_df  = pd.read_csv(LABELS_CSV)
    taxonomy   = pd.read_csv(TAXONOMY_CSV)
    species_list = taxonomy["primary_label"].astype(str).tolist()  # 234 species
    n_classes    = len(species_list)

    # Sort labels by file + start time; drop duplicate windows (some files have
    # duplicated rows in the CSV — keep first occurrence per window)
    labels_df = (
        labels_df
        .sort_values(["filename", "start"])
        .drop_duplicates(subset=["filename", "start"])
        .reset_index(drop=True)
    )
    labeled_files = labels_df["filename"].unique().tolist()
    print(f"  Labeled files: {len(labeled_files)}")

    # 2. Load teacher logits from CSV ──────────────────────────────────────────
    print("Loading teacher logits...")
    teacher_df = pd.read_csv(TEACHER_CSV, index_col="row_id")
    # Columns are species codes; convert probs → logits
    teacher_logits_all = logit(teacher_df.values.astype(np.float32))
    teacher_row2idx    = {rid: i for i, rid in enumerate(teacher_df.index)}
    teacher_species    = list(teacher_df.columns)
    # Build mapping from teacher species order to our species_list order
    t2s = [teacher_species.index(sp) if sp in teacher_species else -1
           for sp in species_list]

    # 3. Load existing cache ───────────────────────────────────────────────────
    print("Loading existing Perch cache...")
    ex = np.load(EXISTING_NPZ, allow_pickle=True)
    ex_emb       = ex["emb_full"]          # (708, 1536)
    ex_logits    = ex["scores_full_raw"]   # (708, 234) already in logit space
    ex_filenames = [str(f) for f in ex["filenames"]]
    ex_row_ids   = [str(r) for r in ex["row_ids"]]
    ex_files_set = set(ex_filenames)

    missing_files = [f for f in labeled_files if f not in ex_files_set]
    print(f"  Cached: {len(set(ex_filenames))} files | Missing: {len(missing_files)} files")

    # 4. Extract embeddings for missing files ──────────────────────────────────
    new_embs, new_logits, new_fnames, new_rids = [], [], [], []

    if missing_files:
        print(f"Loading Perch v2 SavedModel from {PERCH_DIR} ...")
        perch_model = tf.saved_model.load(PERCH_DIR)
        perch_infer = perch_model.signatures["serving_default"]

        for fname in missing_files:
            filepath = os.path.join(SS_DIR, fname)
            if not os.path.exists(filepath):
                print(f"  WARNING: file not found: {filepath}  — skipping")
                continue

            # How many labeled windows does this file have?
            file_df   = labels_df[labels_df["filename"] == fname]
            n_windows = len(file_df)
            print(f"  Extracting {fname}  ({n_windows} windows) ...", end=" ", flush=True)

            embs    = extract_embeddings_from_audio(filepath, perch_infer, n_windows)
            row_ids = build_row_ids(fname, n_windows)

            # Get teacher logits from CSV (use all-zero fallback if missing)
            file_logits = np.zeros((n_windows, n_classes), dtype=np.float32)
            for t, rid in enumerate(row_ids):
                if rid in teacher_row2idx:
                    raw_row = teacher_logits_all[teacher_row2idx[rid]]
                    for s_idx, t_idx in enumerate(t2s):
                        if t_idx >= 0:
                            file_logits[t, s_idx] = raw_row[t_idx]

            new_embs.append(embs)
            new_logits.append(file_logits)
            new_fnames.extend([fname] * n_windows)
            new_rids.extend(row_ids)
            print("OK")

    # 5. Build unified flat arrays ─────────────────────────────────────────────
    print("Merging arrays...")

    # For logits from existing cache: already 234-dim in logit space
    # For logits from teacher CSV: already converted above
    # For logits from existing cache: need to ensure species order matches
    # (existing cache was built separately — assume same species ordering)

    all_emb    = [ex_emb]
    all_logits = [ex_logits]
    all_fnames = ex_filenames[:]
    all_rids   = ex_row_ids[:]

    if new_embs:
        all_emb.append(np.concatenate(new_embs, axis=0))
        all_logits.append(np.concatenate(new_logits, axis=0))
        all_fnames.extend(new_fnames)
        all_rids.extend(new_rids)

    emb_flat    = np.concatenate(all_emb, axis=0).astype(np.float32)
    logits_flat = np.concatenate(all_logits, axis=0).astype(np.float32)
    fnames_arr  = np.array(all_fnames)
    rids_arr    = np.array(all_rids)

    # 6. Build ground-truth label matrix ──────────────────────────────────────
    print("Building label matrix...")
    labels_flat = np.zeros((len(emb_flat), n_classes), dtype=np.float32)
    rid2idx     = {rid: i for i, rid in enumerate(all_rids)}

    for fname in labeled_files:
        file_df = labels_df[labels_df["filename"] == fname]
        row_ids = build_row_ids(fname, len(file_df))
        lmat    = build_label_matrix(file_df, species_list)
        for t, rid in enumerate(row_ids):
            if rid in rid2idx:
                labels_flat[rid2idx[rid]] = lmat[t]

    # 7. Build per-file summaries ──────────────────────────────────────────────
    file_list  = labeled_files
    n_windows_per_file = []
    for fname in file_list:
        count = (fnames_arr == fname).sum()
        n_windows_per_file.append(int(count))

    # 8. Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    np.savez_compressed(
        output_path,
        emb       = emb_flat,       # (N_windows, 1536)
        logits    = logits_flat,    # (N_windows, 234)
        labels    = labels_flat,    # (N_windows, 234)
        filenames = fnames_arr,     # (N_windows,)
        row_ids   = rids_arr,       # (N_windows,)
        file_list  = np.array(file_list),         # (66,)
        n_windows  = np.array(n_windows_per_file, dtype=np.int32),  # (66,)
    )
    total_windows = len(emb_flat)
    labeled_wins  = sum(n_windows_per_file)
    print(f"\nSaved → {output_path}")
    print(f"  Total windows : {total_windows}  (emb shape: {emb_flat.shape})")
    print(f"  Labeled files : {len(file_list)}  ({labeled_wins} windows have ground truth)")
    print(f"  Label density : {labels_flat.sum()/labels_flat.size*100:.2f}% positive")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=OUTPUT_NPZ)
    args = p.parse_args()
    main(args.output)
