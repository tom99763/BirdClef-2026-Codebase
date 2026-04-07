"""Generate Noisy Classmate pseudo labels by fusing multiple backbone chains.

Implements three innovations over simple ensemble averaging:
  Phase 2: Confidence-Aware Blending — per-sample entropy-weighted fusion
  Phase 3: Disagreement Mining — compute per-sample disagreement scores as training weights
  Phase 4: Soft Label Preservation — output raw soft probabilities alongside hard labels

Usage:
    python scripts/gen_noisy_classmate_pseudo.py \
        --chains b0:outputs/sed-ns-b0-20s-r12 pvt:outputs/sed-ns-pvt-20s-r8 \
        --weights 0.5 0.5 \
        --confidence_weighting \
        --disagreement_mining \
        --soft_labels \
        --percentile 95 --gamma 2.0 \
        --out pseudo_labels/noisy_classmate_r1.csv
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import entropy as scipy_entropy


def compute_entropy(probs):
    """Per-sample entropy. probs: (N, C). Returns (N,)."""
    # Treat each class as independent Bernoulli → sum of binary entropies
    eps = 1e-7
    p = np.clip(probs, eps, 1.0 - eps)
    binary_ent = -(p * np.log(p + eps) + (1.0 - p) * np.log(1.0 - p + eps))
    return np.nansum(binary_ent, axis=1)  # (N,)


def confidence_aware_blend(chain_probs_list, base_weights):
    """Phase 2: Per-sample confidence-weighted blending.

    Instead of fixed weights, each sample uses the model with lower entropy
    (= higher confidence) more heavily.

    Args:
        chain_probs_list: list of (N, C) arrays
        base_weights: list of floats (base blend weights, normalized)
    Returns:
        blended: (N, C) array
    """
    n_chains = len(chain_probs_list)
    N, C = chain_probs_list[0].shape

    # Compute per-sample entropy for each chain
    entropies = np.stack([compute_entropy(p) for p in chain_probs_list], axis=1)  # (N, n_chains)

    # Inverse entropy as confidence (lower entropy = higher confidence)
    inv_ent = 1.0 / (entropies + 1e-8)  # (N, n_chains)

    # Multiply by base weights
    for i, bw in enumerate(base_weights):
        inv_ent[:, i] *= bw

    # Normalize per sample
    w_sum = inv_ent.sum(axis=1, keepdims=True)
    sample_weights = inv_ent / (w_sum + 1e-8)  # (N, n_chains)

    # Weighted blend
    blended = np.zeros((N, C), dtype=np.float32)
    for i in range(n_chains):
        blended += sample_weights[:, i:i+1] * chain_probs_list[i]

    return blended, sample_weights


def compute_disagreement(chain_probs_list):
    """Phase 3: Compute per-sample disagreement score.

    Disagreement = mean variance across chains for each species, averaged.
    High disagreement = classmates disagree = high learning potential.

    Args:
        chain_probs_list: list of (N, C) arrays
    Returns:
        disagreement: (N,) array of disagreement scores in [0, 1]
    """
    stacked = np.stack(chain_probs_list, axis=0)  # (n_chains, N, C)
    # Per-species variance across chains
    var_per_species = np.var(stacked, axis=0)  # (N, C)
    # Mean variance across species → single disagreement score per sample
    disagreement = var_per_species.mean(axis=1)  # (N,)
    return disagreement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chains', nargs='+', required=True,
                        help='name:dir pairs, e.g. b0:outputs/sed-ns-b0-20s-r12')
    parser.add_argument('--weights', nargs='+', type=float, default=None,
                        help='Blend weights per chain (default: equal)')
    parser.add_argument('--percentile', type=float, default=95)
    parser.add_argument('--gamma', type=float, default=2.0)
    parser.add_argument('--nonaves_perch_only', action='store_true')
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--taxonomy_csv', type=str, default='birdclef-2026/taxonomy.csv')
    parser.add_argument('--perch_teacher', type=str, default='outputs/perch_teacher_aug_all_ss.csv')
    # Noisy Classmate innovations
    parser.add_argument('--confidence_weighting', action='store_true',
                        help='Phase 2: per-sample entropy-based confidence weighting')
    parser.add_argument('--disagreement_mining', action='store_true',
                        help='Phase 3: compute disagreement scores for sample weighting')
    parser.add_argument('--soft_labels', action='store_true',
                        help='Phase 4: save soft probabilities (pre-threshold) alongside hard labels')
    parser.add_argument('--disagreement_alpha', type=float, default=2.0,
                        help='Phase 3: max extra weight for high-disagreement samples (default: 2.0 → 3x max)')
    args = parser.parse_args()

    # Parse chains
    chain_data = []
    for c in args.chains:
        name, dirpath = c.split(':')
        corrected = Path(dirpath) / 'all_ss_probs_corrected.npz'
        original = Path(dirpath) / 'all_ss_probs.npz'
        npz_path = corrected if corrected.exists() else original
        if not npz_path.exists():
            print(f"ERROR: {npz_path} not found for chain {name}")
            return
        d = np.load(str(npz_path))
        chain_data.append({
            'name': name,
            'row_ids': d['row_ids'],
            'probs': d['probs'].astype(np.float32),
        })
        print(f"  Loaded {name}: {d['probs'].shape} from {npz_path}")

    # Verify all chains have same row_ids
    ref_ids = chain_data[0]['row_ids']
    for cd in chain_data[1:]:
        if not np.array_equal(ref_ids, cd['row_ids']):
            print(f"WARNING: row_ids mismatch between {chain_data[0]['name']} and {cd['name']}")
            common = set(ref_ids.tolist()) & set(cd['row_ids'].tolist())
            print(f"  Common: {len(common)} / {len(ref_ids)}")

    # Base weights
    weights = args.weights or [1.0] * len(chain_data)
    assert len(weights) == len(chain_data)
    total_w = sum(weights)
    norm_weights = [w / total_w for w in weights]

    chain_probs_list = [cd['probs'] for cd in chain_data]

    # ── Phase 2: Confidence-Aware Blending ──────────────────────────────────
    if args.confidence_weighting and len(chain_data) >= 2:
        print("\n=== Phase 2: Confidence-Aware Blending ===")
        blended, sample_weights = confidence_aware_blend(chain_probs_list, norm_weights)
        # Report statistics
        for i, cd in enumerate(chain_data):
            mean_w = sample_weights[:, i].mean()
            std_w = sample_weights[:, i].std()
            print(f"  {cd['name']}: mean_weight={mean_w:.4f} +/- {std_w:.4f}")
    else:
        print("\n=== Phase 1: Simple Average Blending ===")
        blended = np.zeros_like(chain_data[0]['probs'])
        for cd, w in zip(chain_data, norm_weights):
            blended += w * cd['probs']
            print(f"  {cd['name']}: weight={w:.3f}")

    # ── Phase 3: Disagreement Mining ────────────────────────────────────────
    disagreement_scores = None
    if args.disagreement_mining and len(chain_data) >= 2:
        print("\n=== Phase 3: Disagreement Mining ===")
        disagreement_scores = compute_disagreement(chain_probs_list)
        print(f"  Disagreement: mean={disagreement_scores.mean():.6f}, "
              f"max={disagreement_scores.max():.6f}, "
              f"p95={np.percentile(disagreement_scores, 95):.6f}")
        # Normalize to [0, 1]
        d_max = disagreement_scores.max()
        if d_max > 0:
            disagreement_norm = disagreement_scores / d_max
        else:
            disagreement_norm = disagreement_scores
        # Training weights: 1.0 + alpha * normalized_disagreement
        train_weights = 1.0 + args.disagreement_alpha * disagreement_norm
        print(f"  Training weights: min={train_weights.min():.2f}, "
              f"max={train_weights.max():.2f}, mean={train_weights.mean():.2f}")
        # How many samples have weight > 2.0 (significant disagreement)?
        n_high = (train_weights > 2.0).sum()
        print(f"  High-disagreement samples (weight > 2.0): {n_high} "
              f"({n_high / len(train_weights) * 100:.1f}%)")

    print(f"\nBlended probs shape: {blended.shape}")
    print(f"Blended mean: {blended.mean():.6f}, max: {blended.max():.6f}")

    # Load taxonomy
    taxonomy = pd.read_csv(args.taxonomy_csv)
    species_cols = taxonomy['primary_label'].astype(str).tolist()

    # ── Phase 4: Save soft labels ───────────────────────────────────────────
    if args.soft_labels:
        soft_out = args.out.replace('.csv', '_soft.npz')
        save_dict = {
            'row_ids': chain_data[0]['row_ids'],
            'soft_probs': blended,  # pre-gamma, pre-threshold
        }
        if disagreement_scores is not None:
            save_dict['disagreement'] = disagreement_scores
            save_dict['train_weights'] = train_weights
        np.savez_compressed(soft_out, **save_dict)
        print(f"\n=== Phase 4: Soft labels saved → {soft_out} ===")

    # Apply gamma sharpening (for hard pseudo labels)
    blended_sharp = blended.copy()
    if args.gamma != 1.0:
        blended_sharp = np.power(blended_sharp, args.gamma)
        print(f"Applied gamma={args.gamma}")

    # Threshold by percentile
    positive_vals = blended_sharp[blended_sharp > 0]
    if len(positive_vals) == 0:
        print("WARNING: no positive values after gamma")
        return
    threshold = np.percentile(positive_vals, args.percentile)
    print(f"Threshold (p{args.percentile}): {threshold:.6f}")

    # Build pseudo label DataFrame — compatible with existing gen_pseudo_ns.py format
    # Format: row_id, <234 species soft probs>, _nc_weight (optional)
    row_ids = chain_data[0]['row_ids']

    # Start with row_id column
    df = pd.DataFrame({'row_id': row_ids})

    # Add ALL 234 species columns with soft blended probs (pre-gamma for training)
    for j, sc in enumerate(species_cols):
        df[sc] = blended[:, j]

    # Add NC disagreement weights
    if disagreement_scores is not None:
        df['_nc_weight'] = train_weights

    # Filter by threshold: only keep rows where at least one species passes
    row_max = blended_sharp.max(axis=1)
    keep_mask = row_max >= threshold
    df = df[keep_mask].reset_index(drop=True)
    print(f"Kept {keep_mask.sum()} / {len(keep_mask)} rows above threshold")

    # Count positive labels (soft probs above threshold)
    species_data = df[species_cols].values
    n_pos = (species_data > 0.05).sum()
    print(f"Pseudo labels: {len(df)} rows, {n_pos} positive entries (>0.05)")

    if '_nc_weight' in df.columns:
        print(f"NC weights in CSV: mean={df['_nc_weight'].mean():.3f}, "
              f"max={df['_nc_weight'].max():.3f}")

    df.to_csv(args.out, index=False)
    print(f"Saved: {args.out} ({len(df)} rows)")


if __name__ == '__main__':
    main()
