"""Generate Noisy Student pseudo labels for unlabeled soundscapes.

Round 0: Use Perch teacher predictions (perch_teacher_all_ss.csv).
Round k: Ensemble of 5-fold SED probabilities + (optionally) 5-fold SSM probabilities.

Pipeline:
  1. Collect soft probabilities from each model
  2. Ensemble (simple average)
  3. Power transform (gamma=2.0, per BirdCLEF 2025 1st place)
  4. Per-class dynamic threshold (95th percentile)
  5. Save soft probs for high-confidence windows as pseudo_labels/ns_rK.csv

Output format (matches existing pseudo_labels/*.csv):
  row_id, <234 species cols>, primary_label, secondary_labels

Usage:
  # Round 0 (Perch only teacher):
  python scripts/gen_pseudo_ns.py --round 0 \
      --perch_csv outputs/perch_teacher_all_ss.csv \
      --out pseudo_labels/ns_r0.csv

  # Round k (SED + SSM ensemble):
  python scripts/gen_pseudo_ns.py --round 1 \
      --sed_dir outputs/sed-ns-b0-r1 \
      --ssm_dir outputs/proto-ssm-ns-r1 \
      --perch_csv outputs/perch_teacher_all_ss.csv \
      --out pseudo_labels/ns_r1.csv
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GAMMA          = 2.0    # Power transform exponent (BirdCLEF 2025 1st place)
PERCENTILE     = 95.0   # Per-class dynamic threshold percentile
MIN_THRESHOLD  = 0.05   # Floor threshold (avoid too many FP)
MAX_THRESHOLD  = 0.5    # Ceiling threshold (ensure some positive labels)
PERCH_W        = 0.50   # Weight for Perch probs in ensemble
SED_W          = 0.30   # Weight for SED probs in ensemble
SSM_W          = 0.20   # Weight for SSM probs in ensemble
NUM_CLASSES    = 234


def load_perch_probs(perch_csv: str) -> pd.DataFrame:
    """Load Perch teacher predictions from CSV."""
    print(f"Loading Perch probs: {perch_csv}")
    df = pd.read_csv(perch_csv)
    print(f"  {len(df):,} rows, {df.shape[1]} cols")
    return df


def load_oof_probs(exp_dir: str, n_folds: int = 5) -> np.ndarray | None:
    """Load OOF predictions from an experiment directory.
    Returns (N_windows, 234) probs averaged over folds, or None if not found.
    """
    oof_path = Path(exp_dir) / 'oof_predictions.npz'
    if not oof_path.exists():
        print(f"  OOF not found: {oof_path}")
        return None
    npz = np.load(str(oof_path))
    # Expected keys: probs (N, 234) or logits (N, 234)
    if 'probs' in npz:
        probs = npz['probs']
    elif 'logits' in npz:
        probs = 1.0 / (1.0 + np.exp(-npz['logits']))
    else:
        print(f"  Unexpected keys in {oof_path}: {list(npz.keys())}")
        return None
    print(f"  OOF probs: {probs.shape}")
    return probs.astype(np.float32)


def load_all_ss_probs_from_sed(sed_dir: str, row_ids: list) -> np.ndarray | None:
    """Load SED predictions for all soundscapes (inference, not OOF).
    Looks for sed_dir/all_ss_probs.npz with keys: row_ids, probs.
    """
    pred_path = Path(sed_dir) / 'all_ss_probs.npz'
    if not pred_path.exists():
        print(f"  SED all-ss preds not found: {pred_path}")
        return None
    npz = np.load(str(pred_path), allow_pickle=True)
    sed_rids = list(npz['row_ids'])
    sed_probs = npz['probs'].astype(np.float32)
    # Align to row_ids order
    rid2idx = {r: i for i, r in enumerate(sed_rids)}
    aligned = np.zeros((len(row_ids), NUM_CLASSES), dtype=np.float32)
    found = 0
    for i, rid in enumerate(row_ids):
        if rid in rid2idx:
            aligned[i] = sed_probs[rid2idx[rid]]
            found += 1
    print(f"  SED probs: {found}/{len(row_ids)} rows matched")
    return aligned


def power_transform(probs: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    """Power transform to sharpen predictions. Used in BirdCLEF 2025 1st place."""
    return np.power(np.clip(probs, 0, 1), gamma)


def dynamic_threshold(probs_pt: np.ndarray,
                      percentile: float = PERCENTILE,
                      min_thr: float = MIN_THRESHOLD,
                      max_thr: float = MAX_THRESHOLD) -> np.ndarray:
    """Per-class dynamic threshold at the given percentile of transformed probs."""
    thr = np.percentile(probs_pt, percentile, axis=0)  # (234,)
    thr = np.clip(thr, min_thr, max_thr)
    return thr


def get_species_cols(df: pd.DataFrame) -> list:
    """Extract species column names (all cols except row_id, primary_label, secondary_labels)."""
    non_species = {'row_id', 'primary_label', 'secondary_labels'}
    return [c for c in df.columns if c not in non_species]


def merge_5s_to_Ns(df: pd.DataFrame, species_cols: list,
                   clip_sec: int = 10, stride_sec: int = 5) -> pd.DataFrame:
    """Convert 5s Perch windows to N-second-aligned pseudo labels (generalised).

    For each N-second training clip (stride=stride_sec), the pseudo label is
    max across all 5s Perch windows that fall within the clip — matches
    BirdCLEF 2025 1st place:
    "the maximum probability for each label was taken across the segments
    within the interval."

    clip_sec=10  → each clip covers 2 Perch windows (prev + curr)
    clip_sec=20  → each clip covers 4 Perch windows

    Input row_id format:  {soundscape_id}_{end_sec}   (5s grid)
    Output row_id format: {soundscape_id}_{end_sec}   (stride_sec grid, start ≥ clip_sec)
    """
    rows_out = []
    df = df.copy()
    df['_fname']  = df['row_id'].apply(lambda r: str(r).rsplit('_', 1)[0])
    df['_offset'] = df['row_id'].apply(lambda r: int(str(r).rsplit('_', 1)[1]))

    for fname, grp in df.groupby('_fname'):
        grp    = grp.sort_values('_offset').reset_index(drop=True)
        offsets = grp['_offset'].tolist()
        probs   = grp[species_cols].values   # (N_perch, 234)
        off2row = {o: i for i, o in enumerate(offsets)}
        max_off = max(offsets)

        # Generate clip end-times with stride_sec step, starting at clip_sec
        for end in range(clip_sec, max_off + stride_sec + 1, stride_sec):
            # Perch 5s windows whose end-time falls within [end-clip_sec+5 .. end]
            window_offsets = range(end - clip_sec + 5, end + 1, 5)
            rows = [probs[off2row[o]] for o in window_offsets if o in off2row]
            if not rows:
                continue
            merged = np.max(rows, axis=0)   # max across overlapping Perch windows
            rows_out.append([f"{fname}_{end}"] + merged.tolist())

    out = pd.DataFrame(rows_out, columns=['row_id'] + species_cols)
    print(f"  merge_5s_to_Ns(clip={clip_sec}s): {len(df)} 5s rows → {len(out)} {clip_sec}s rows")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round',      type=int, required=True, help='Pseudo label round number')
    parser.add_argument('--clip_sec',   type=int, default=10, help='Student training clip duration (s)')
    parser.add_argument('--perch_csv',  default='outputs/perch_teacher_all_ss.csv')
    parser.add_argument('--sed_dir',    default=None, help='SED experiment dir (has all_ss_probs.npz)')
    parser.add_argument('--ssm_dir',    default=None, help='SSM experiment dir (has all_ss_probs.npz)')
    parser.add_argument('--out',        required=True, help='Output pseudo label CSV')
    parser.add_argument('--gamma',      type=float, default=GAMMA)
    parser.add_argument('--percentile', type=float, default=PERCENTILE)
    parser.add_argument('--min_thr',    type=float, default=MIN_THRESHOLD)
    parser.add_argument('--max_thr',    type=float, default=MAX_THRESHOLD)
    parser.add_argument('--perch_w',    type=float, default=PERCH_W)
    parser.add_argument('--sed_w',      type=float, default=SED_W)
    parser.add_argument('--ssm_w',      type=float, default=SSM_W)
    # Filter to unlabeled only (exclude 66 labeled soundscapes from pseudo set)
    parser.add_argument('--labeled_csv', default='birdclef-2026/train_soundscapes_labels.csv',
                        help='Path to soundscape labels CSV (to exclude labeled files)')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)

    # ── Load Perch teacher ───────────────────────────────────────────────────
    perch_df     = load_perch_probs(args.perch_csv)
    species_cols = get_species_cols(perch_df)
    assert len(species_cols) == NUM_CLASSES, f"Expected 234 species cols, got {len(species_cols)}"

    # Merge 5s Perch windows to match student clip duration (e.g. 10s, 20s)
    if args.clip_sec > 5:
        print(f"Merging 5s Perch windows → {args.clip_sec}s clips (max across overlapping windows)")
        perch_df = merge_5s_to_Ns(perch_df, species_cols, clip_sec=args.clip_sec)

    row_ids      = perch_df['row_id'].astype(str).tolist()
    perch_probs  = perch_df[species_cols].values.astype(np.float32)

    # ── Exclude labeled soundscapes ──────────────────────────────────────────
    labeled_df     = pd.read_csv(args.labeled_csv)
    labeled_files  = set(labeled_df['filename'].astype(str).unique())
    # row_id format: BC2026_Train_XXXX_SXX_YYYYMMDD_HHMMSS_T
    # filename format: BC2026_Train_XXXX_SXX_YYYYMMDD_HHMMSS.ogg
    def row_id_to_filename(rid):
        parts = rid.rsplit('_', 1)
        return parts[0] + '.ogg' if len(parts) == 2 else rid + '.ogg'

    mask_unlabeled = np.array([
        row_id_to_filename(rid) not in labeled_files for rid in row_ids
    ])
    print(f"Rows: total={len(row_ids)}, unlabeled={mask_unlabeled.sum()}, "
          f"labeled(excluded)={(~mask_unlabeled).sum()}")

    # ── Ensemble ─────────────────────────────────────────────────────────────
    w_total  = args.perch_w
    ensemble = args.perch_w * perch_probs.copy()

    if args.sed_dir:
        sed_probs = load_all_ss_probs_from_sed(args.sed_dir, row_ids)
        if sed_probs is not None:
            ensemble += args.sed_w * sed_probs
            w_total  += args.sed_w

    if args.ssm_dir:
        ssm_probs = load_all_ss_probs_from_sed(args.ssm_dir, row_ids)
        if ssm_probs is not None:
            ensemble += args.ssm_w * ssm_probs
            w_total  += args.ssm_w

    ensemble /= w_total
    sed_share = args.sed_w / w_total if args.sed_dir else 0.0
    ssm_share = args.ssm_w / w_total if args.ssm_dir else 0.0
    print(f"Ensemble weights: perch={args.perch_w/w_total:.2f}, "
          f"sed={sed_share:.2f}, ssm={ssm_share:.2f}")

    # ── Filter to unlabeled only ──────────────────────────────────────────────
    probs_unlab  = ensemble[mask_unlabeled]
    rids_unlab   = [r for r, m in zip(row_ids, mask_unlabeled) if m]
    print(f"Working with {len(rids_unlab):,} unlabeled windows")

    # ── Power transform ───────────────────────────────────────────────────────
    probs_pt = power_transform(probs_unlab, gamma=args.gamma)
    print(f"Power transform (gamma={args.gamma}): "
          f"mean={probs_pt.mean():.4f}, max={probs_pt.max():.4f}")

    # ── Dynamic threshold ─────────────────────────────────────────────────────
    thr = dynamic_threshold(probs_pt, args.percentile, args.min_thr, args.max_thr)
    print(f"Threshold (p{args.percentile}): "
          f"mean={thr.mean():.4f}, min={thr.min():.4f}, max={thr.max():.4f}")

    # ── Keep windows with at least one species above threshold ────────────────
    above = (probs_pt >= thr[None, :]).any(axis=1)
    probs_keep = probs_unlab[above]   # use ORIGINAL probs (before power transform) as soft labels
    rids_keep  = [r for r, a in zip(rids_unlab, above) if a]
    print(f"Windows above threshold: {above.sum():,} / {len(rids_unlab):,} "
          f"({100*above.mean():.1f}%)")

    # ── Assign primary and secondary labels ───────────────────────────────────
    primary_labels = [species_cols[i] for i in probs_keep.argmax(axis=1)]
    secondary_labels = []
    for i, row_probs in enumerate(probs_keep):
        above_thr = np.where(row_probs >= thr)[0]
        sec = [species_cols[j] for j in above_thr if species_cols[j] != primary_labels[i]]
        secondary_labels.append(';'.join(sec))

    # ── Build output DataFrame ────────────────────────────────────────────────
    out_df = pd.DataFrame(probs_keep, columns=species_cols)
    out_df.insert(0, 'row_id', rids_keep)
    out_df['primary_label']    = primary_labels
    out_df['secondary_labels'] = secondary_labels

    out_df.to_csv(args.out, index=False)
    print(f"\nSaved {len(out_df):,} pseudo-labeled windows → {args.out}")
    print(f"  Species with >0 positives: "
          f"{(out_df[species_cols].max() >= thr).sum()}")

    # ── Stats ─────────────────────────────────────────────────────────────────
    print(f"\n=== Pseudo Label Stats (Round {args.round}) ===")
    print(f"  Total windows    : {len(rids_unlab):,}")
    print(f"  Kept (≥1 species): {len(rids_keep):,} ({100*len(rids_keep)/len(rids_unlab):.1f}%)")
    top5 = pd.Series(primary_labels).value_counts().head(5)
    print(f"  Top-5 primary labels:\n{top5.to_string()}")


if __name__ == '__main__':
    main()
