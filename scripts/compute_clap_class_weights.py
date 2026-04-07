#!/usr/bin/env python3
"""
Compute dynamic per-class CLAP ensemble weights from zero-shot AUC.

Input : weights/clap/clap_per_class_auc.npy   (234,) — -1.0 for unknown classes
Output: weights/clap/clap_class_weights.npy    (234,) — final blend weights

Weight scheme:
  - AUC ≤ 0.5  → w = 0.0   (worse than random, exclude CLAP)
  - 0.5 < AUC  → w scales linearly from 0 to max_w as AUC goes 0.5 → 1.0
  - Unknown (AUC=-1) → fallback: 0.08 (Aves) or 0.18 (non-Aves)

Non-Aves classes (Amphibia/Insecta/Mammalia/Reptilia) get 2× higher max cap.

Usage:
  python scripts/compute_clap_class_weights.py \\
      --auc_npy   weights/clap/clap_per_class_auc.npy \\
      --taxonomy  birdclef-2026/taxonomy.csv \\
      --out       weights/clap/clap_class_weights.npy
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

_NONAVES = {"Amphibia", "Insecta", "Mammalia", "Reptilia"}

# Caps on CLAP weight per class type
MAX_W_AVES    = 0.15   # Aves:    AUC=1.0 → w=0.15
MAX_W_NONAVES = 0.30   # non-Aves: AUC=1.0 → w=0.30

# Fallback for classes not seen in eval set (AUC=-1)
FALLBACK_AVES    = 0.08
FALLBACK_NONAVES = 0.18


def auc_to_weight(auc: float, is_nonaves: bool) -> float:
    """Convert per-class zero-shot AUC to CLAP blend weight."""
    if auc < 0:
        # Unknown (not in eval set) — use conservative fallback
        return FALLBACK_NONAVES if is_nonaves else FALLBACK_AVES
    if auc <= 0.5:
        return 0.0   # worse than random → exclude
    max_w = MAX_W_NONAVES if is_nonaves else MAX_W_AVES
    # Linear scale: AUC=0.5 → 0, AUC=1.0 → max_w
    return float(max_w * (auc - 0.5) / 0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auc_npy",  default="weights/clap/clap_per_class_auc.npy")
    parser.add_argument("--taxonomy", default="birdclef-2026/taxonomy.csv")
    parser.add_argument("--out",      default="weights/clap/clap_class_weights.npy")
    args = parser.parse_args()

    auc_arr  = np.load(args.auc_npy)            # (234,)
    tax_df   = pd.read_csv(args.taxonomy).set_index("primary_label")
    sp_list  = sorted(tax_df.index.tolist())

    assert len(auc_arr) == len(sp_list), \
        f"AUC array length {len(auc_arr)} != taxonomy {len(sp_list)}"

    weights = np.zeros(len(sp_list), dtype=np.float32)
    for i, sp in enumerate(sp_list):
        cls_name  = str(tax_df.loc[sp]["class_name"]) if sp in tax_df.index else "Aves"
        is_nonaves = cls_name in _NONAVES
        weights[i] = auc_to_weight(float(auc_arr[i]), is_nonaves)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, weights)

    # ── Summary ───────────────────────────────────────────────────────────────
    active  = auc_arr >= 0
    unknown = auc_arr < 0
    zero_w  = (weights == 0) & active   # active but AUC ≤ 0.5
    print(f"Per-class CLAP weights computed → {out_path}")
    print(f"  Species total   : {len(sp_list)}")
    print(f"  Active (AUC≥0)  : {active.sum()}  "
          f"(mean AUC={auc_arr[active].mean():.4f})")
    print(f"  Unknown (AUC=-1): {unknown.sum()}  → fallback weights")
    print(f"  Zero weight     : {zero_w.sum()}  (AUC ≤ 0.5, excluded)")
    print(f"  Weights range   : [{weights.min():.4f}, {weights.max():.4f}]")
    print(f"  Weights mean    : {weights.mean():.4f}")

    # Top-5 / bottom-5 by weight
    sp_arr = np.array(sp_list)
    top5   = np.argsort(weights)[-5:][::-1]
    bot5   = np.argsort(weights)[:5]
    print(f"\nTop-5 CLAP weights:")
    for i in top5:
        name = tax_df.loc[sp_arr[i]]["common_name"] if sp_arr[i] in tax_df.index else sp_arr[i]
        print(f"  {weights[i]:.4f}  {name}  (AUC={auc_arr[i]:.4f})")
    print(f"\nBottom-5 CLAP weights (zero = excluded):")
    for i in bot5:
        name = tax_df.loc[sp_arr[i]]["common_name"] if sp_arr[i] in tax_df.index else sp_arr[i]
        print(f"  {weights[i]:.4f}  {name}  (AUC={auc_arr[i]:.4f})")


if __name__ == "__main__":
    main()
