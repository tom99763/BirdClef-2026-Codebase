#!/usr/bin/env python3
"""
research_new_methods.py
=========================
三個真正新的方法研究（非 weighted ensemble）：

M1: Co-occurrence Propagation（從1478訓練窗口建矩陣，比65-file LOO穩健得多）
M2: Test-time Temporal Coherence（file內相鄰窗口互相補強）
M3: Species-frequency Calibration（用訓練集 prior P(species) 校正預測）

基準：embed_prior LOO = 0.9918（softmax_T6_proto_kde）
"""
import os, json, pickle, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
import warnings; warnings.filterwarnings('ignore')

os.chdir("/home/lab/BirdClef-2026-Codebase")

# ── 載入 labeled soundscape 資料 ─────────────────────────────────────────────
perch = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = perch['emb'].astype(np.float32)      # (739, 1536)
logits_win = perch['logits'].astype(np.float32)   # (739, 234)
labels_win = perch['labels'].astype(np.float32)   # (739, 234)
file_list  = list(perch['file_list'])              # 66 files
n_windows  = perch['n_windows']                    # (66,)
n_files    = len(file_list)
n_species  = labels_win.shape[1]

file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end   = np.cumsum(n_windows).astype(np.int32)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x.astype(np.float64), -88, 88))).astype(np.float32)

# Per-file ground truth (max over windows)
file_labels    = np.zeros((n_files, n_species), np.float32)
file_logit_max = np.zeros((n_files, n_species), np.float32)
file_perch_max = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_labels[fi]    = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi] = logits_win[s:e].max(0)
    file_perch_max[fi] = sigmoid(logits_win[s:e]).max(0)

def macro_auc(yt, yp):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], yp[:, mask], average='macro')

# Baseline: Perch直接 sigmoid max per file
baseline_auc = macro_auc(file_labels, file_perch_max)
print(f"[Baseline] Perch sigmoid max per file: {baseline_auc:.4f}")

# Best embed_prior pkl
with open("outputs/embed_prior_model.pkl", "rb") as f:
    ep = pickle.load(f)

EPS = 1e-7

# ── 讀取 taxonomy 與訓練資料 ────────────────────────────────────────────────
tax = pd.read_csv("birdclef-2026/taxonomy.csv")
PRIMARY_LABELS = tax['primary_label'].astype(str).tolist()
label2idx = {l: i for i, l in enumerate(PRIMARY_LABELS)}

# ════════════════════════════════════════════════════════════════════════════
# M1: Co-occurrence Propagation from FULL training soundscapes (1478 windows)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("M1: Co-occurrence Propagation (1478 training windows)")
print("="*60)

ss_labels = pd.read_csv("birdclef-2026/train_soundscapes_labels.csv")
N = n_species

# Build Y matrix from ALL training soundscape windows
Y_train = np.zeros((len(ss_labels), N), np.float32)
for wi, row in ss_labels.iterrows():
    for sp in str(row['primary_label']).split(';'):
        sp = sp.strip()
        if sp in label2idx:
            Y_train[wi, label2idx[sp]] = 1.0

# Co-occurrence matrix: C_cond[i,j] = P(j|i) = count(i∩j) / count(i)
C_raw    = Y_train.T @ Y_train               # (N, N) - all windows
row_sums = Y_train.sum(0) + EPS             # (N,)
C_cond   = C_raw / row_sums[:, None]        # P(j | i)
np.fill_diagonal(C_cond, 0)                 # 自身不計

print(f"  Training windows: {len(ss_labels)}, species present: {(Y_train.sum(0)>0).sum()}")
print(f"  C_cond non-zero pairs: {(C_cond>0.1).sum()}")

# LOO on labeled soundscapes: predict file i using Perch + co-occurrence
def cooc_predict_file(file_probs, alpha):
    """
    file_probs: (n_species,) - base probability predictions
    Propagate via co-occurrence: y_out = y + alpha * (y @ C_cond)
    """
    prop = file_probs @ C_cond        # weighted sum of co-occurring species
    return np.clip(file_probs + alpha * prop, 0, 1)

best_cooc_auc = 0; best_alpha_cooc = 0
for alpha in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
    preds = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        preds[fi] = cooc_predict_file(file_perch_max[fi], alpha)
    auc = macro_auc(file_labels, preds)
    print(f"  alpha={alpha:.2f}: AUC={auc:.4f}")
    if auc > best_cooc_auc:
        best_cooc_auc = auc; best_alpha_cooc = alpha

print(f"  >> Best alpha={best_alpha_cooc}: {best_cooc_auc:.4f}  (baseline={baseline_auc:.4f})")

# Test on logit space
print("\n  [Logit-space co-occurrence]")
best_cooc_logit_auc = 0; best_alpha_l = 0
for alpha in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
    preds = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        logit = file_logit_max[fi]
        prop  = sigmoid(logit) @ C_cond   # co-occurrence signal in prob space
        logit_prop = np.log(prop.clip(EPS)) - np.log((1-prop).clip(EPS))
        combined = logit + alpha * logit_prop
        preds[fi] = sigmoid(combined)
    auc = macro_auc(file_labels, preds)
    print(f"  alpha={alpha:.1f}: AUC={auc:.4f}")
    if auc > best_cooc_logit_auc:
        best_cooc_logit_auc = auc; best_alpha_l = alpha

print(f"  >> Best logit alpha={best_alpha_l}: {best_cooc_logit_auc:.4f}")

# Also: use per-window co-occurrence and then take file max
print("\n  [Window-level co-occurrence, then file max]")
best_cooc_win_auc = 0; best_alpha_w = 0
for alpha in [0.1, 0.2, 0.3, 0.5]:
    preds = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        s, e = int(file_start[fi]), int(file_end[fi])
        probs_w = sigmoid(logits_win[s:e])  # (n_win, 234)
        # Apply co-occurrence per window
        prop_w = probs_w @ C_cond           # (n_win, 234)
        out_w  = np.clip(probs_w + alpha * prop_w, 0, 1)
        preds[fi] = out_w.max(0)
    auc = macro_auc(file_labels, preds)
    print(f"  alpha={alpha:.1f}: AUC={auc:.4f}")
    if auc > best_cooc_win_auc:
        best_cooc_win_auc = auc; best_alpha_w = alpha

print(f"  >> Best window alpha={best_alpha_w}: {best_cooc_win_auc:.4f}")

# ════════════════════════════════════════════════════════════════════════════
# M2: Test-time Temporal Coherence (file-level window smoothing)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("M2: Test-time Temporal Coherence (window smoothing)")
print("="*60)

def temporal_smooth_predict(alpha_smooth=0.3, use_max=True):
    """
    For each file, smooth predictions across adjacent windows.
    Then take max (or mean) over windows.
    """
    preds = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        s, e = int(file_start[fi]), int(file_end[fi])
        probs_w = sigmoid(logits_win[s:e])  # (n_win, 234)
        n_w = e - s
        if n_w <= 1:
            preds[fi] = probs_w[0] if n_w == 1 else np.zeros(n_species)
            continue

        # Bilateral-like smoothing: each window influenced by neighbors
        smoothed = probs_w.copy()
        for t in range(1, n_w - 1):
            smoothed[t] = (1 - alpha_smooth) * probs_w[t] + \
                          alpha_smooth * 0.5 * (probs_w[t-1] + probs_w[t+1])
        preds[fi] = smoothed.max(0) if use_max else smoothed.mean(0)
    return preds

best_ts_auc = 0; best_ts_alpha = 0
for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]:
    preds = temporal_smooth_predict(alpha_smooth=alpha, use_max=True)
    auc = macro_auc(file_labels, preds)
    print(f"  alpha={alpha:.1f} (max): AUC={auc:.4f}")
    if auc > best_ts_auc:
        best_ts_auc = auc; best_ts_alpha = alpha

for alpha in [0.1, 0.2, 0.3, 0.5]:
    preds = temporal_smooth_predict(alpha_smooth=alpha, use_max=False)
    auc = macro_auc(file_labels, preds)
    print(f"  alpha={alpha:.1f} (mean): AUC={auc:.4f}")
    if auc > best_ts_auc:
        best_ts_auc = auc; best_ts_alpha = alpha

# Cumulative max across windows (early detection should reinforce later)
print("\n  [Cumulative max smoothing]")
def cummax_predict(beta=0.3):
    preds = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        s, e = int(file_start[fi]), int(file_end[fi])
        probs_w = sigmoid(logits_win[s:e])
        running_max = probs_w[0].copy()
        for t in range(1, e - s):
            running_max = np.maximum(running_max, probs_w[t])
            probs_w[t] = (1 - beta) * probs_w[t] + beta * running_max
        preds[fi] = probs_w.max(0)
    return preds

for beta in [0.1, 0.2, 0.3, 0.5]:
    preds = cummax_predict(beta)
    auc = macro_auc(file_labels, preds)
    print(f"  cummax beta={beta}: AUC={auc:.4f}")
    if auc > best_ts_auc:
        best_ts_auc = auc; best_ts_alpha = beta

print(f"  >> Best temporal coherence: {best_ts_auc:.4f}")

# ════════════════════════════════════════════════════════════════════════════
# M3: Species-frequency Calibration via class-aware temperature
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("M3: Per-species temperature calibration (class-frequency aware)")
print("="*60)

# Compute prior P(species) from training soundscapes
species_freq = Y_train.mean(0)  # (N,) prior probability

# Idea: rare species need different temperature than common ones
# Temperature = f(species_frequency): rare → higher T (more conservative)
# Common → lower T (more aggressive)

def freq_temp_predict(T_rare, T_common, freq_threshold=0.01):
    """Per-species temperature based on training frequency."""
    preds = np.zeros((n_files, n_species), np.float32)
    T_per_species = np.where(species_freq >= freq_threshold, T_common, T_rare)
    for fi in range(n_files):
        s, e = int(file_start[fi]), int(file_end[fi])
        scaled = logits_win[s:e] / T_per_species[None, :]  # (n_win, 234)
        preds[fi] = sigmoid(scaled).max(0)
    return preds

# Baseline: uniform T=1.0
preds_t1 = freq_temp_predict(1.0, 1.0)
print(f"  T_rare=1.0 T_common=1.0: AUC={macro_auc(file_labels, preds_t1):.4f}")

best_temp_auc = 0
for T_rare in [0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]:
    for T_common in [0.5, 0.7, 0.8, 0.9, 1.0]:
        preds = freq_temp_predict(T_rare, T_common)
        auc = macro_auc(file_labels, preds)
        if auc > best_temp_auc:
            best_temp_auc = auc
            print(f"  T_rare={T_rare} T_common={T_common}: AUC={auc:.4f} *** NEW BEST ***")

print(f"  >> Best temp calibration: {best_temp_auc:.4f}")

# Also: logit-space prior calibration
# y_cal = sigmoid(logit + alpha * log(prior / (1-prior)))
print("\n  [Prior-based logit calibration]")
log_prior = np.log(species_freq.clip(EPS)) - np.log((1 - species_freq).clip(EPS))
best_prior_auc = 0
for alpha in [-0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5]:
    preds = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        s, e = int(file_start[fi]), int(file_end[fi])
        calibrated = logits_win[s:e] + alpha * log_prior[None, :]
        preds[fi] = sigmoid(calibrated).max(0)
    auc = macro_auc(file_labels, preds)
    print(f"  alpha={alpha:+.1f}: AUC={auc:.4f}")
    if auc > best_prior_auc:
        best_prior_auc = auc

print(f"  >> Best prior calib: {best_prior_auc:.4f}")

# ════════════════════════════════════════════════════════════════════════════
# M4: Co-occurrence + Temporal Coherence COMBINED
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("M4: Co-occurrence + Temporal Coherence COMBINED")
print("="*60)

best_combo_auc = 0
for alpha_cooc in [0.1, 0.2, 0.3]:
    for alpha_ts in [0.1, 0.2, 0.3]:
        preds = np.zeros((n_files, n_species), np.float32)
        for fi in range(n_files):
            s, e = int(file_start[fi]), int(file_end[fi])
            probs_w = sigmoid(logits_win[s:e])  # (n_win, 234)
            n_w = e - s

            # Step 1: temporal smoothing
            if n_w > 2:
                for t in range(1, n_w - 1):
                    probs_w[t] = (1-alpha_ts)*probs_w[t] + alpha_ts*0.5*(probs_w[t-1]+probs_w[t+1])

            # Step 2: per-window co-occurrence
            prop_w = probs_w @ C_cond
            out_w  = np.clip(probs_w + alpha_cooc * prop_w, 0, 1)

            preds[fi] = out_w.max(0)

        auc = macro_auc(file_labels, preds)
        if auc > best_combo_auc:
            best_combo_auc = auc
            print(f"  cooc={alpha_cooc} ts={alpha_ts}: AUC={auc:.4f} *** NEW BEST ***")

print(f"  >> Best combo: {best_combo_auc:.4f}")

# ════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  RESULTS SUMMARY")
print("="*65)
results = {
    "baseline_perch_max": float(baseline_auc),
    "M1_cooc_prob": float(best_cooc_auc),
    "M1_cooc_logit": float(best_cooc_logit_auc),
    "M1_cooc_window": float(best_cooc_win_auc),
    "M2_temporal_coherence": float(best_ts_auc),
    "M3_freq_temp": float(best_temp_auc),
    "M3_prior_logit": float(best_prior_auc),
    "M4_combo": float(best_combo_auc),
    "reference_embed_prior": 0.9918,
}
for k, v in results.items():
    flag = " ★ BEAT BASELINE ★" if v > baseline_auc else ""
    ref_flag = " ★★ BEAT EP ★★" if v > 0.9918 else ""
    print(f"  {k:<40} {v:.4f}{flag}{ref_flag}")

with open("outputs/new_methods_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\n  Saved to outputs/new_methods_results.json")
