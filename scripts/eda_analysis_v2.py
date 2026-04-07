#!/usr/bin/env python3
"""
BirdCLEF 2026 Comprehensive EDA Analysis v2
Generates full HTML report with all analysis sections.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
from scipy.stats import pearsonr, spearmanr
import warnings
import os
import json
import re
from collections import defaultdict

warnings.filterwarnings('ignore')

BASE = '/home/lab/BirdClef-2026-Codebase'
OUTPUTS = f'{BASE}/outputs'
DATA = f'{BASE}/birdclef-2026'
REPORTS = f'{BASE}/reports'

print("=== Loading base data ===")

# Load taxonomy
tax_df = pd.read_csv(f'{DATA}/taxonomy.csv')
# primary_label as string key
tax_df['label_str'] = tax_df['primary_label'].astype(str)
label2class = dict(zip(tax_df['label_str'], tax_df['class_name']))
label2sciname = dict(zip(tax_df['label_str'], tax_df['scientific_name']))
print(f"Taxonomy: {len(tax_df)} entries")
print("Class distribution:", tax_df['class_name'].value_counts().to_dict())

# Load sample submission for class order
sub_df = pd.read_csv(f'{DATA}/sample_submission.csv')
class_cols = sub_df.columns[1:].tolist()
n_classes = len(class_cols)
print(f"Classes: {n_classes}")

# Map class cols to taxonomy
class_names_arr = [label2class.get(str(c), 'Aves') for c in class_cols]
is_aves = np.array([cn == 'Aves' for cn in class_names_arr])
class_to_idx = {c: i for i, c in enumerate(class_cols)}

# Load ground truth labels
labels_df = pd.read_csv(f'{DATA}/train_soundscapes_labels.csv')
print(f"GT labels: {len(labels_df)} rows")

# Parse GT into binary matrix
# row_id format: filename + "_" + end_time_in_seconds
def parse_row_id(filename, end_str):
    """Convert filename + end time to row_id format."""
    # end time like 00:00:05 → 5
    parts = end_str.split(':')
    secs = int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
    fname = filename.replace('.ogg', '')
    return f"{fname}_{secs}"

gt_rows = {}
for _, row in labels_df.iterrows():
    rid = parse_row_id(row['filename'], row['end'])
    labels = str(row['primary_label']).split(';')
    gt_rows[rid] = labels

print(f"GT unique row_ids: {len(gt_rows)}")
print("Sample GT row_id:", list(gt_rows.keys())[0])

# Load B0 R8 to get all row_ids
d_b0r8 = np.load(f'{OUTPUTS}/sed-ns-b0-20s-r8/all_ss_probs.npz', mmap_mode='r')
all_row_ids = d_b0r8['row_ids'].tolist()
print(f"Total row_ids in npz: {len(all_row_ids)}")

# Find which row_ids have GT labels
gt_mask = np.array([rid in gt_rows for rid in all_row_ids])
gt_indices = np.where(gt_mask)[0]
print(f"Row_ids with GT: {len(gt_indices)}")

# Build GT binary matrix for GT windows only
gt_row_ids_subset = [all_row_ids[i] for i in gt_indices]
gt_matrix = np.zeros((len(gt_indices), n_classes), dtype=np.float32)
for j, rid in enumerate(gt_row_ids_subset):
    for label in gt_rows[rid]:
        label_str = str(label).strip()
        if label_str in class_to_idx:
            gt_matrix[j, class_to_idx[label_str]] = 1.0

print(f"GT matrix shape: {gt_matrix.shape}")
print(f"GT positive rate: {gt_matrix.mean():.4f}")
print(f"Classes with any GT positive: {(gt_matrix.sum(0) > 0).sum()}")

# Per-class positive counts
class_pos_counts = gt_matrix.sum(0)
print(f"\nClass positive count stats: min={class_pos_counts.min():.0f}, max={class_pos_counts.max():.0f}, mean={class_pos_counts.mean():.1f}")
print(f"Classes with 0 positives: {(class_pos_counts == 0).sum()}")
print(f"Classes with <5 positives: {(class_pos_counts < 5).sum()}")

# Helper: compute macro AUC (only classes with both pos and neg)
def compute_macro_auc(y_true, y_score, class_names_arr=None):
    """Compute per-class AUC and macro average."""
    n = y_true.shape[1]
    aucs = []
    valid_classes = []
    for i in range(n):
        if y_true[:, i].sum() > 0 and y_true[:, i].sum() < len(y_true):
            try:
                auc = roc_auc_score(y_true[:, i], y_score[:, i])
                aucs.append(auc)
                valid_classes.append(i)
            except:
                pass
    return np.array(aucs), valid_classes

print("\n=== Computing per-round AUCs ===")

# B0 rounds - only R8 has all_ss_probs.npz directly
b0_rounds_available = []
pvt_rounds_available = []

for r in range(1, 10):
    p = f'{OUTPUTS}/sed-ns-b0-20s-r{r}/all_ss_probs.npz'
    if os.path.exists(p):
        b0_rounds_available.append(r)
for r in range(1, 9):
    p = f'{OUTPUTS}/sed-ns-pvt-20s-r{r}/all_ss_probs.npz'
    if os.path.exists(p):
        pvt_rounds_available.append(r)

print(f"B0 rounds with npz: {b0_rounds_available}")
print(f"PVT rounds with npz: {pvt_rounds_available}")

# Compute GT AUC for available rounds
b0_gt_aucs = {}
pvt_gt_aucs = {}
pvt_gt_aucs_corrected = {}

for r in b0_rounds_available:
    d = np.load(f'{OUTPUTS}/sed-ns-b0-20s-r{r}/all_ss_probs.npz', mmap_mode='r')
    probs_gt = d['probs'][gt_indices]
    aucs, valid = compute_macro_auc(gt_matrix, probs_gt)
    b0_gt_aucs[r] = float(np.mean(aucs))
    print(f"B0 R{r} GT macro AUC: {b0_gt_aucs[r]:.4f} ({len(valid)} valid classes)")

for r in pvt_rounds_available:
    d = np.load(f'{OUTPUTS}/sed-ns-pvt-20s-r{r}/all_ss_probs.npz', mmap_mode='r')
    probs_gt = d['probs'][gt_indices]
    aucs, valid = compute_macro_auc(gt_matrix, probs_gt)
    pvt_gt_aucs[r] = float(np.mean(aucs))
    print(f"PVT R{r} GT macro AUC: {pvt_gt_aucs[r]:.4f} ({len(valid)} valid classes)")

    # Check corrected version
    pc = f'{OUTPUTS}/sed-ns-pvt-20s-r{r}/all_ss_probs_corrected.npz'
    if os.path.exists(pc):
        d2 = np.load(pc, mmap_mode='r')
        probs_gt2 = d2['probs'][gt_indices]
        aucs2, valid2 = compute_macro_auc(gt_matrix, probs_gt2)
        pvt_gt_aucs_corrected[r] = float(np.mean(aucs2))
        print(f"  PVT R{r} CORRECTED GT macro AUC: {pvt_gt_aucs_corrected[r]:.4f}")

print("\n=== Per-fold AUC from logs ===")

def extract_val_auc_from_log(log_path):
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        content = f.read()
    patterns = [
        r'Best val AUC[:\s]+([\d.]+)',
        r'best_val_auc[:\s]+([\d.]+)',
        r'val_auc[:\s]+([\d.]+)',
        r'Val AUC[:\s]+([\d.]+)',
        r'soundscape.*?auc[:\s]+([\d.]+)',
    ]
    for p in patterns:
        matches = re.findall(p, content, re.IGNORECASE)
        if matches:
            return float(matches[-1])
    return None

b0_fold_aucs = {}
pvt_fold_aucs = {}

for r in range(1, 10):
    aucs = []
    for f in range(5):
        log = f'{OUTPUTS}/logs/sed_ns_20s_r{r}_fold{f}.log'
        auc = extract_val_auc_from_log(log)
        aucs.append(auc)
    b0_fold_aucs[r] = aucs
    valid = [a for a in aucs if a is not None]
    if valid:
        print(f"B0 R{r}: {[f'{a:.4f}' if a else 'None' for a in aucs]} mean={np.mean(valid):.4f}")

for r in range(1, 9):
    aucs = []
    for f in range(5):
        log = f'{OUTPUTS}/logs/sed_ns_pvt_r{r}_fold{f}.log'
        auc = extract_val_auc_from_log(log)
        aucs.append(auc)
    pvt_fold_aucs[r] = aucs
    valid = [a for a in aucs if a is not None]
    if valid:
        print(f"PVT R{r}: {[f'{a:.4f}' if a else 'None' for a in aucs]} mean={np.mean(valid):.4f}")

# Confirmed LB data
print("\n=== LB Correlation Analysis ===")
lb_data = [
    # (model, round, val_auc_mean, lb_score, note)
    ('B0', 3, 0.9305, 0.926, 'fold1+3'),
    ('B0', 4, 0.9341, 0.927, 'fold1+3'),
    ('B0', 5, 0.9330, 0.931, 'fold1+3'),
    ('B0', 6, 0.9383, 0.931, 'fold1+3 full5'),
    ('B0+PVT', 6, 0.9383, 0.933, 'B0R6f3+PVTR4'),
    ('B0+PVT', 8, 0.9383, 0.937, 'B0R8f2+PVTR5f4+B0R8f3, sed_w=0.5'),
    ('B0+PVT', 8, 0.9383, 0.938, 'B0R8f2+PVTR5f4+B0R8f3, sed_w=0.7'),
]

# Compute B0 mean val AUCs
b0_mean_aucs = {}
for r, aucs in b0_fold_aucs.items():
    valid = [a for a in aucs if a is not None]
    if valid:
        b0_mean_aucs[r] = np.mean(valid)

pvt_mean_aucs = {}
for r, aucs in pvt_fold_aucs.items():
    valid = [a for a in aucs if a is not None]
    if valid:
        pvt_mean_aucs[r] = np.mean(valid)

print("B0 mean AUCs:", {r: f"{v:.4f}" for r, v in b0_mean_aucs.items()})
print("PVT mean AUCs:", {r: f"{v:.4f}" for r, v in pvt_mean_aucs.items()})

print("\n=== Per-class AUC analysis (B0 R8) ===")

d_b0r8_loaded = np.load(f'{OUTPUTS}/sed-ns-b0-20s-r8/all_ss_probs.npz', mmap_mode='r')
b0r8_probs_gt = d_b0r8_loaded['probs'][gt_indices].copy()

per_class_auc_b0r8 = {}
for i in range(n_classes):
    if class_pos_counts[i] > 0 and class_pos_counts[i] < len(gt_indices):
        try:
            auc = roc_auc_score(gt_matrix[:, i], b0r8_probs_gt[:, i])
            per_class_auc_b0r8[i] = auc
        except:
            pass

print(f"B0 R8: {len(per_class_auc_b0r8)} valid classes")
print(f"B0 R8 macro AUC: {np.mean(list(per_class_auc_b0r8.values())):.4f}")

# Per-class AUC for PVT R8
d_pvtr8 = np.load(f'{OUTPUTS}/sed-ns-pvt-20s-r8/all_ss_probs.npz', mmap_mode='r')
pvtr8_probs_gt = d_pvtr8['probs'][gt_indices].copy()

per_class_auc_pvtr8 = {}
for i in range(n_classes):
    if class_pos_counts[i] > 0 and class_pos_counts[i] < len(gt_indices):
        try:
            auc = roc_auc_score(gt_matrix[:, i], pvtr8_probs_gt[:, i])
            per_class_auc_pvtr8[i] = auc
        except:
            pass

print(f"PVT R8: {len(per_class_auc_pvtr8)} valid classes")
print(f"PVT R8 macro AUC: {np.mean(list(per_class_auc_pvtr8.values())):.4f}")

# Compare B0 vs PVT per class
common_classes = set(per_class_auc_b0r8.keys()) & set(per_class_auc_pvtr8.keys())
pvt_wins = {i: per_class_auc_pvtr8[i] - per_class_auc_b0r8[i] for i in common_classes}
pvt_better = [(i, pvt_wins[i]) for i in common_classes if pvt_wins[i] > 0]
b0_better = [(i, -pvt_wins[i]) for i in common_classes if pvt_wins[i] < 0]

pvt_better.sort(key=lambda x: -x[1])
b0_better.sort(key=lambda x: -x[1])

print(f"\nPVT better in {len(pvt_better)}/{len(common_classes)} classes")
print(f"B0 better in {len(b0_better)}/{len(common_classes)} classes")
print("Top 10 PVT wins:")
for i, diff in pvt_better[:10]:
    print(f"  {class_cols[i]} ({class_names_arr[i]}): PVT={per_class_auc_pvtr8[i]:.4f} B0={per_class_auc_b0r8[i]:.4f} +{diff:.4f}")

print("Top 10 B0 wins:")
for i, diff in b0_better[:10]:
    print(f"  {class_cols[i]} ({class_names_arr[i]}): B0={per_class_auc_b0r8[i]:.4f} PVT={per_class_auc_pvtr8[i]:.4f} +{diff:.4f}")

# Taxon breakdown
print("\n=== Taxon breakdown ===")
for taxon in ['Aves', 'Amphibia', 'Insecta', 'Mammalia', 'Reptilia']:
    taxon_indices = [i for i in common_classes if class_names_arr[i] == taxon]
    if taxon_indices:
        b0_auc_t = np.mean([per_class_auc_b0r8[i] for i in taxon_indices])
        pvt_auc_t = np.mean([per_class_auc_pvtr8[i] for i in taxon_indices])
        print(f"{taxon}: {len(taxon_indices)} classes, B0={b0_auc_t:.4f}, PVT={pvt_auc_t:.4f}, diff={pvt_auc_t-b0_auc_t:+.4f}")

print("\n=== Ensemble diversity analysis ===")

# Correlation between B0 and PVT on GT windows
corr = np.corrcoef(b0r8_probs_gt.flatten(), pvtr8_probs_gt.flatten())[0, 1]
print(f"B0 R8 vs PVT R8 overall prediction correlation: {corr:.4f}")

# Per-class correlation
per_class_corr = []
for i in range(n_classes):
    if class_pos_counts[i] > 1:
        c = np.corrcoef(b0r8_probs_gt[:, i], pvtr8_probs_gt[:, i])[0, 1]
        per_class_corr.append(c)
print(f"Per-class correlation: mean={np.mean(per_class_corr):.4f}, std={np.std(per_class_corr):.4f}")

# Ensemble analysis: try different fold combinations for B0 R8
print("\n=== Ensemble optimization analysis ===")

# Load per-fold predictions
# For B0 R8, we only have the ensemble all_ss_probs.npz
# Let's compute ensemble combinations with PVT rounds

# Best ensemble: B0 R8 + PVT R8 at different weights
for w_sed in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    ens_probs = w_sed * b0r8_probs_gt + (1 - w_sed) * pvtr8_probs_gt
    aucs, _ = compute_macro_auc(gt_matrix, ens_probs)
    print(f"B0R8 + PVT R8 ensemble (sed_w={w_sed}): GT macro AUC = {np.mean(aucs):.4f}")

# Try different PVT rounds
print("\nB0 R8 + PVT round comparison (w=0.5):")
for r in pvt_rounds_available:
    d = np.load(f'{OUTPUTS}/sed-ns-pvt-20s-r{r}/all_ss_probs.npz', mmap_mode='r')
    probs_gt_pvt = d['probs'][gt_indices]
    ens = 0.5 * b0r8_probs_gt + 0.5 * probs_gt_pvt
    aucs, _ = compute_macro_auc(gt_matrix, ens)
    print(f"  B0R8 + PVT R{r}: {np.mean(aucs):.4f}")

# Confidence distribution analysis
print("\n=== Prediction confidence distribution ===")
for r in pvt_rounds_available[-3:]:
    d = np.load(f'{OUTPUTS}/sed-ns-pvt-20s-r{r}/all_ss_probs.npz', mmap_mode='r')
    p = d['probs']
    print(f"PVT R{r}: mean={p.mean():.4f}, std={p.std():.4f}, p10={np.percentile(p, 10):.4f}, p90={np.percentile(p, 90):.4f}, p99={np.percentile(p, 99):.4f}")

d = np.load(f'{OUTPUTS}/sed-ns-b0-20s-r8/all_ss_probs.npz', mmap_mode='r')
p = d['probs']
print(f"B0 R8: mean={p.mean():.4f}, std={p.std():.4f}, p10={np.percentile(p, 10):.4f}, p90={np.percentile(p, 90):.4f}, p99={np.percentile(p, 99):.4f}")

print("\n=== Calibration analysis ===")

# Calibration: bin predictions and compare to actual positive rate
def calibration_analysis(probs, gt, n_bins=10):
    """Return mean predicted prob and actual positive rate per bin."""
    bins = np.linspace(0, 1, n_bins+1)
    bin_means = []
    bin_actuals = []
    bin_counts = []
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() > 0:
            bin_means.append(probs[mask].mean())
            bin_actuals.append(gt[mask].mean())
            bin_counts.append(mask.sum())
    return bin_means, bin_actuals, bin_counts

# GT-only calibration
b0_cal_means, b0_cal_actuals, b0_cal_counts = calibration_analysis(
    b0r8_probs_gt.flatten(), gt_matrix.flatten()
)
pvt_cal_means, pvt_cal_actuals, pvt_cal_counts = calibration_analysis(
    pvtr8_probs_gt.flatten(), gt_matrix.flatten()
)

print("B0 R8 calibration (mean_pred, actual_positive_rate):")
for m, a, c in zip(b0_cal_means, b0_cal_actuals, b0_cal_counts):
    print(f"  pred={m:.3f}, actual={a:.3f}, count={c}")

print("\nPVT R8 calibration (mean_pred, actual_positive_rate):")
for m, a, c in zip(pvt_cal_means, pvt_cal_actuals, pvt_cal_counts):
    print(f"  pred={m:.3f}, actual={a:.3f}, count={c}")

print("\n=== Isotonic calibration simulation ===")

# Apply isotonic calibration and check AUC improvement
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

def isotonic_calibration_auc(probs_gt, gt_matrix):
    """Cross-val isotonic calibration AUC."""
    n = probs_gt.shape[0]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cal_probs = np.zeros_like(probs_gt)

    for train_idx, val_idx in kf.split(np.arange(n)):
        ir = IsotonicRegression(out_of_bounds='clip')
        # Fit on all classes together
        train_probs = probs_gt[train_idx].flatten()
        train_gt = gt_matrix[train_idx].flatten()
        ir.fit(train_probs, train_gt)
        val_probs_flat = probs_gt[val_idx].flatten()
        cal_flat = ir.transform(val_probs_flat)
        cal_probs[val_idx] = cal_flat.reshape(probs_gt[val_idx].shape)

    aucs, _ = compute_macro_auc(gt_matrix, cal_probs)
    return np.mean(aucs), cal_probs

b0_cal_auc, b0_cal_probs = isotonic_calibration_auc(b0r8_probs_gt, gt_matrix)
pvt_cal_auc, pvt_cal_probs = isotonic_calibration_auc(pvtr8_probs_gt, gt_matrix)

b0_raw_aucs, _ = compute_macro_auc(gt_matrix, b0r8_probs_gt)
pvt_raw_aucs, _ = compute_macro_auc(gt_matrix, pvtr8_probs_gt)

print(f"B0 R8 raw AUC: {np.mean(b0_raw_aucs):.4f} → isotonic calibrated: {b0_cal_auc:.4f} (delta: {b0_cal_auc-np.mean(b0_raw_aucs):+.4f})")
print(f"PVT R8 raw AUC: {np.mean(pvt_raw_aucs):.4f} → isotonic calibrated: {pvt_cal_auc:.4f} (delta: {pvt_cal_auc-np.mean(pvt_raw_aucs):+.4f})")

print("\n=== Per-class threshold optimization ===")

# Optimize threshold per class (proxy metric: per-class AUC doesn't change with threshold,
# but we can show the optimal F1 threshold vs. fixed threshold effect on balanced accuracy)
# Since competition metric is AUC, threshold optimization doesn't directly help AUC.
# But let's show the confidence-weighted macro AUC upper bound

# Max possible if we had perfect ordering per class (= per-class AUC itself)
max_possible = np.mean([per_class_auc_b0r8[i] for i in per_class_auc_b0r8])
print(f"B0 R8 per-class AUC already represents the optimal threshold-agnostic performance")
print(f"B0 R8 macro AUC (GT): {max_possible:.4f}")

# Temporal consistency analysis
print("\n=== Temporal consistency analysis ===")

# Group GT windows by file, check consistency across overlapping segments
file_windows = defaultdict(list)
for j, rid in enumerate(gt_row_ids_subset):
    # rid format: BC2026_Train_XXXX_...._timestamp
    # Extract file prefix (everything except the last _timestamp)
    parts = rid.rsplit('_', 1)
    file_prefix = parts[0]
    file_windows[file_prefix].append(j)

# For files with multiple windows, compute std of predictions
consistency_scores_b0 = []
consistency_scores_pvt = []
for file_prefix, indices in file_windows.items():
    if len(indices) > 2:
        probs_b0_file = b0r8_probs_gt[indices]
        probs_pvt_file = pvtr8_probs_gt[indices]
        consistency_scores_b0.append(probs_b0_file.std(axis=0).mean())
        consistency_scores_pvt.append(probs_pvt_file.std(axis=0).mean())

print(f"Files with >2 GT windows: {len(consistency_scores_b0)}")
print(f"B0 R8 temporal std (avg per file): {np.mean(consistency_scores_b0):.4f}")
print(f"PVT R8 temporal std (avg per file): {np.mean(consistency_scores_pvt):.4f}")

# Non-Aves analysis
print("\n=== Non-Aves analysis ===")

aves_mask = np.array([class_names_arr[i] == 'Aves' for i in range(n_classes)])
non_aves_mask = ~aves_mask

# Only analyze classes with GT positives
valid_aves = [i for i in per_class_auc_b0r8 if aves_mask[i]]
valid_nonaves = [i for i in per_class_auc_b0r8 if non_aves_mask[i]]

print(f"Aves classes with GT: {len(valid_aves)}, Non-Aves: {len(valid_nonaves)}")

b0_aves_auc = np.mean([per_class_auc_b0r8[i] for i in valid_aves]) if valid_aves else 0
b0_nonaves_auc = np.mean([per_class_auc_b0r8[i] for i in valid_nonaves]) if valid_nonaves else 0
pvt_aves_auc = np.mean([per_class_auc_pvtr8[i] for i in valid_aves]) if valid_aves else 0
pvt_nonaves_auc = np.mean([per_class_auc_pvtr8[i] for i in valid_nonaves]) if valid_nonaves else 0

print(f"B0 R8: Aves AUC={b0_aves_auc:.4f}, Non-Aves AUC={b0_nonaves_auc:.4f}")
print(f"PVT R8: Aves AUC={pvt_aves_auc:.4f}, Non-Aves AUC={pvt_nonaves_auc:.4f}")

# Perch analysis on non-Aves
perch_df = pd.read_csv(f'{OUTPUTS}/perch_teacher_aug_all_ss.csv')
perch_df = perch_df.set_index('row_id')
perch_col_map = {}
for c in class_cols:
    if str(c) in perch_df.columns:
        perch_col_map[c] = str(c)

# Align perch to gt_indices
perch_probs_gt = np.zeros((len(gt_indices), n_classes), dtype=np.float32)
for j, rid in enumerate(gt_row_ids_subset):
    if rid in perch_df.index:
        row = perch_df.loc[rid]
        for i, c in enumerate(class_cols):
            if str(c) in perch_df.columns:
                perch_probs_gt[j, i] = row[str(c)]

per_class_auc_perch = {}
for i in range(n_classes):
    if class_pos_counts[i] > 0 and class_pos_counts[i] < len(gt_indices):
        try:
            auc = roc_auc_score(gt_matrix[:, i], perch_probs_gt[:, i])
            per_class_auc_perch[i] = auc
        except:
            pass

valid_aves_p = [i for i in per_class_auc_perch if aves_mask[i]]
valid_nonaves_p = [i for i in per_class_auc_perch if non_aves_mask[i]]
perch_aves_auc = np.mean([per_class_auc_perch[i] for i in valid_aves_p]) if valid_aves_p else 0
perch_nonaves_auc = np.mean([per_class_auc_perch[i] for i in valid_nonaves_p]) if valid_nonaves_p else 0
perch_macro_auc = np.mean(list(per_class_auc_perch.values()))
print(f"Perch: Aves AUC={perch_aves_auc:.4f}, Non-Aves AUC={perch_nonaves_auc:.4f}, overall={perch_macro_auc:.4f}")

# Ensemble with Perch for non-Aves
ens_nonaves_probs = b0r8_probs_gt.copy()
for i in range(n_classes):
    if non_aves_mask[i]:
        ens_nonaves_probs[:, i] = 0.5 * b0r8_probs_gt[:, i] + 0.5 * perch_probs_gt[:, i]

per_class_auc_hybrid = {}
for i in range(n_classes):
    if class_pos_counts[i] > 0 and class_pos_counts[i] < len(gt_indices):
        try:
            auc = roc_auc_score(gt_matrix[:, i], ens_nonaves_probs[:, i])
            per_class_auc_hybrid[i] = auc
        except:
            pass
hybrid_macro = np.mean(list(per_class_auc_hybrid.values()))
print(f"B0 R8 + Perch non-Aves hybrid macro AUC: {hybrid_macro:.4f}")

# Compare per-taxon detailed
print("\nPer-taxon detailed analysis:")
for taxon in ['Aves', 'Amphibia', 'Insecta', 'Mammalia', 'Reptilia']:
    idxs_b0 = [i for i in per_class_auc_b0r8 if class_names_arr[i] == taxon]
    idxs_pvt = [i for i in per_class_auc_pvtr8 if class_names_arr[i] == taxon]
    idxs_perch = [i for i in per_class_auc_perch if class_names_arr[i] == taxon]

    total_in_tax = sum(1 for cn in class_names_arr if cn == taxon)
    if idxs_b0:
        b0_t = np.mean([per_class_auc_b0r8[i] for i in idxs_b0])
        pvt_t = np.mean([per_class_auc_pvtr8[i] for i in idxs_pvt]) if idxs_pvt else 0
        perch_t = np.mean([per_class_auc_perch[i] for i in idxs_perch]) if idxs_perch else 0
        gt_positives_t = sum(class_pos_counts[i] for i in range(n_classes) if class_names_arr[i] == taxon)
        print(f"  {taxon}: total={total_in_tax}, GT_valid={len(idxs_b0)}, GT_pos={gt_positives_t:.0f}, B0={b0_t:.4f}, PVT={pvt_t:.4f}, Perch={perch_t:.4f}")

# LB projection for future rounds
print("\n=== LB Projection ===")

# Use linear regression on val_auc -> LB
# Confirmed points:
confirmed_lb = [
    (0.9305, 0.926),  # B0 R3
    (0.9341, 0.927),  # B0 R4
    (0.9330, 0.931),  # B0 R5
    (0.9383, 0.931),  # B0 R6
]

val_aucs_for_reg = np.array([x[0] for x in confirmed_lb])
lbs_for_reg = np.array([x[1] for x in confirmed_lb])

# Simple linear fit
from numpy.polynomial import polynomial as P
coeffs = np.polyfit(val_aucs_for_reg, lbs_for_reg, 1)
print(f"Linear fit: LB = {coeffs[0]:.4f} * val_auc + {coeffs[1]:.4f}")

# Project for future rounds
print("\nProjected LB for PVT rounds (assuming same linear relationship):")
for r, auc in pvt_mean_aucs.items():
    proj_lb = coeffs[0] * auc + coeffs[1]
    print(f"  PVT R{r}: val_auc={auc:.4f} → projected_LB={proj_lb:.4f}")

print("\nAll analysis complete! Saving results...")

# Collect all results for HTML report
results = {
    'b0_fold_aucs': b0_fold_aucs,
    'pvt_fold_aucs': pvt_fold_aucs,
    'b0_mean_aucs': b0_mean_aucs,
    'pvt_mean_aucs': pvt_mean_aucs,
    'b0_gt_aucs': b0_gt_aucs,
    'pvt_gt_aucs': pvt_gt_aucs,
    'pvt_gt_aucs_corrected': pvt_gt_aucs_corrected,
    'per_class_auc_b0r8': per_class_auc_b0r8,
    'per_class_auc_pvtr8': per_class_auc_pvtr8,
    'per_class_auc_perch': per_class_auc_perch,
    'pvt_better': pvt_better[:20],
    'b0_better': b0_better[:20],
    'n_classes': n_classes,
    'class_cols': class_cols,
    'class_names_arr': class_names_arr,
    'class_pos_counts': class_pos_counts.tolist(),
    'gt_matrix_shape': list(gt_matrix.shape),
    'corr_b0_pvt': float(corr),
    'b0_cal_means': b0_cal_means,
    'b0_cal_actuals': b0_cal_actuals,
    'b0_cal_counts': b0_cal_counts,
    'pvt_cal_means': pvt_cal_means,
    'pvt_cal_actuals': pvt_cal_actuals,
    'pvt_cal_counts': pvt_cal_counts,
    'b0_raw_macro_auc': float(np.mean(b0_raw_aucs)),
    'pvt_raw_macro_auc': float(np.mean(pvt_raw_aucs)),
    'b0_cal_macro_auc': float(b0_cal_auc),
    'pvt_cal_macro_auc': float(pvt_cal_auc),
    'b0_aves_auc': float(b0_aves_auc),
    'b0_nonaves_auc': float(b0_nonaves_auc),
    'pvt_aves_auc': float(pvt_aves_auc),
    'pvt_nonaves_auc': float(pvt_nonaves_auc),
    'perch_aves_auc': float(perch_aves_auc),
    'perch_nonaves_auc': float(perch_nonaves_auc),
    'perch_macro_auc': float(perch_macro_auc),
    'hybrid_macro_auc': float(hybrid_macro),
    'temporal_std_b0': float(np.mean(consistency_scores_b0)),
    'temporal_std_pvt': float(np.mean(consistency_scores_pvt)),
    'n_valid_aves': len(valid_aves),
    'n_valid_nonaves': len(valid_nonaves),
    'coeffs': coeffs.tolist(),
}

# Per-taxon
taxon_results = {}
for taxon in ['Aves', 'Amphibia', 'Insecta', 'Mammalia', 'Reptilia']:
    idxs_b0 = [i for i in per_class_auc_b0r8 if class_names_arr[i] == taxon]
    idxs_pvt = [i for i in per_class_auc_pvtr8 if class_names_arr[i] == taxon]
    idxs_perch = [i for i in per_class_auc_perch if class_names_arr[i] == taxon]
    total_in_tax = sum(1 for cn in class_names_arr if cn == taxon)
    gt_positives_t = sum(class_pos_counts[i] for i in range(n_classes) if class_names_arr[i] == taxon)
    taxon_results[taxon] = {
        'total': total_in_tax,
        'gt_valid': len(idxs_b0),
        'gt_positives': int(gt_positives_t),
        'b0_auc': float(np.mean([per_class_auc_b0r8[i] for i in idxs_b0])) if idxs_b0 else 0,
        'pvt_auc': float(np.mean([per_class_auc_pvtr8[i] for i in idxs_pvt])) if idxs_pvt else 0,
        'perch_auc': float(np.mean([per_class_auc_perch[i] for i in idxs_perch])) if idxs_perch else 0,
    }

results['taxon_results'] = taxon_results

# Ensemble AUC table
ens_results = {}
for w_sed in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    ens_probs = w_sed * b0r8_probs_gt + (1 - w_sed) * pvtr8_probs_gt
    aucs, _ = compute_macro_auc(gt_matrix, ens_probs)
    ens_results[f'b0r8_pvtr8_w{int(w_sed*10)}'] = float(np.mean(aucs))

ens_pvt_results = {}
for r in pvt_rounds_available:
    d = np.load(f'{OUTPUTS}/sed-ns-pvt-20s-r{r}/all_ss_probs.npz', mmap_mode='r')
    probs_gt_pvt = d['probs'][gt_indices]
    ens = 0.5 * b0r8_probs_gt + 0.5 * probs_gt_pvt
    aucs, _ = compute_macro_auc(gt_matrix, ens)
    ens_pvt_results[r] = float(np.mean(aucs))

results['ens_results'] = ens_results
results['ens_pvt_results'] = ens_pvt_results

# Save
import pickle
with open(f'{REPORTS}/eda_v2_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print(f"Results saved to {REPORTS}/eda_v2_results.pkl")
print("DONE!")
PYEOF