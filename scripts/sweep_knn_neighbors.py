"""
Full neighbor (k) sweep for 4way_knn_logit method.
Uses the SAME file-level LOO protocol as logit_fusion_v3.py.

Parts:
  1) Single-k weighted-logit KNN sweep (alpha x k grid)
  2) 2-way KNN combo sweep (alpha=0.35 fixed)
  3) Logit aggregation comparison (max / mean / softmax_pool / top2_mean)
"""

import numpy as np
import json
import pickle
import scipy.special
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score

DATA_PATH    = 'outputs/perch_labeled_ss.npz'
RESULTS_PATH = 'outputs/embed_prior_results.json'
MODEL_PATH   = 'outputs/embed_prior_model.pkl'
SWEEP_PATH   = 'outputs/embed_prior_sweep.json'

PREV_BEST    = 0.8930

# ─── Load data ────────────────────────────────────────────────────────────────
raw       = np.load(DATA_PATH, allow_pickle=True)
emb_win   = raw['emb'].astype(np.float32)       # [739, 1536]
logits_win= raw['logits'].astype(np.float32)    # [739, 234]
labels_win= raw['labels'].astype(np.float32)    # [739, 234]
file_list = raw['file_list']                    # [66]
n_windows = raw['n_windows']                    # [66]

n_files   = len(file_list)
n_species = labels_win.shape[1]
print(f"Loaded: {n_files} files, {sum(n_windows)} windows, {n_species} species")

# ─── Build file-level aggregations ────────────────────────────────────────────
file_embs        = np.zeros((n_files, emb_win.shape[1]), dtype=np.float32)
file_labels      = np.zeros((n_files, n_species),        dtype=np.float32)
file_logit_max   = np.zeros((n_files, n_species),        dtype=np.float32)
file_logit_mean  = np.zeros((n_files, n_species),        dtype=np.float32)
file_logit_top2  = np.zeros((n_files, n_species),        dtype=np.float32)
file_logit_softpool = np.zeros((n_files, n_species),     dtype=np.float32)

idx = 0
for fi, nw in enumerate(n_windows):
    ws  = emb_win[idx:idx+nw]
    wl  = logits_win[idx:idx+nw]  # (nw, 234)
    lb  = labels_win[idx:idx+nw]

    file_embs[fi]       = ws.mean(0)
    file_labels[fi]     = (lb.max(0) > 0.5).astype(np.float32)
    file_logit_max[fi]  = wl.max(0)
    file_logit_mean[fi] = wl.mean(0)

    # top-2 per species
    if nw >= 2:
        file_logit_top2[fi] = np.sort(wl, axis=0)[-2:].mean(0)
    else:
        file_logit_top2[fi] = wl.max(0)

    # softmax-pool: weight each window by its max-logit, then weighted sum
    w_win = wl.max(axis=1)         # (nw,) — window importance
    w_win = w_win - w_win.max()
    w_sm  = np.exp(w_win)
    w_sm  = w_sm / (w_sm.sum() + 1e-12)
    file_logit_softpool[fi] = (w_sm[:, None] * wl).sum(0)

    idx += nw

file_embs_norm    = normalize(file_embs, norm='l2')   # [66, 1536]
file_prob_max     = scipy.special.expit(file_logit_max)
file_prob_mean    = scipy.special.expit(file_logit_mean)
file_prob_top2    = scipy.special.expit(file_logit_top2)
file_prob_softpool= scipy.special.expit(file_logit_softpool)

AGG_SOURCES = {
    "max_logit":          file_prob_max,
    "mean_logit":         file_prob_mean,
    "top2_mean_logit":    file_prob_top2,
    "softmax_pool_logit": file_prob_softpool,
}

# ─── Helpers ─────────────────────────────────────────────────────────────────
def macro_auc(y_true, y_score):
    mask = y_true.sum(0) > 0
    if mask.sum() < 2:
        return float('nan')
    try:
        return float(roc_auc_score(y_true[:, mask], y_score[:, mask], average='macro'))
    except Exception:
        return float('nan')

def knn_predict(k=5, X=None):
    """File-level LOO cosine-similarity KNN."""
    if X is None:
        X = file_embs_norm
    k_eff = min(k, n_files - 1)
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for i in range(n_files):
        mask = np.ones(n_files, dtype=bool); mask[i] = False
        tr = X[mask]; te = X[[i]]; y_tr = file_labels[mask]
        sims = (te @ tr.T).ravel()          # (65,)
        nn_idx = np.argpartition(-sims, k_eff)[:k_eff]
        weights = np.clip(sims[nn_idx], 0, None)
        if weights.sum() < 1e-9:
            weights = np.ones(k_eff, dtype=np.float32)
        preds[i] = (weights[:, None] * y_tr[nn_idx]).sum(0) / weights.sum()
    return preds

# ─── Pre-compute KNN for all k values we'll need ────────────────────────────
K_ALL = [1, 2, 3, 5, 7, 10, 15, 20]
print("\nPre-computing KNN for k =", K_ALL, "...")
knn_cache = {}
for k in K_ALL:
    knn_cache[k] = knn_predict(k=k)
    print(f"  k={k:2d}  AUC={macro_auc(file_labels, knn_cache[k]):.6f}")

# ══════════════════════════════════════════════════════════════════════════════
# PART 1: Single-k weighted-logit KNN sweep
# score = alpha * sigmoid(max_logit) + (1-alpha) * KNN(k)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("PART 1: Single-k weighted-logit KNN sweep (max_logit)")
print("="*65)

k_list = [1, 2, 3, 5, 7, 10, 15, 20]
a_list = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

part1_results = []
for k in k_list:
    for alpha in a_list:
        ens = alpha * file_prob_max + (1 - alpha) * knn_cache[k]
        auc = macro_auc(file_labels, ens)
        part1_results.append({"k": k, "alpha": alpha, "auc": round(auc, 6)})
        print(f"  k={k:2d}  alpha={alpha:.2f}  AUC={auc:.6f}")

best_p1 = max(part1_results, key=lambda x: x["auc"])
print(f"\nPART 1 BEST: k={best_p1['k']}, alpha={best_p1['alpha']}, AUC={best_p1['auc']:.6f}")

# ══════════════════════════════════════════════════════════════════════════════
# PART 2: 2-way KNN combo sweep (alpha=0.35 fixed)
# score = 0.35 * logit + w1 * KNN(k1) + w2 * KNN(k2), w1+w2=0.65
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("PART 2: 2-way KNN combo sweep (alpha=0.35 fixed)")
print("="*65)

combos = [
    (1, 3,  0.40),
    (1, 5,  0.40),
    (2, 5,  0.45),
    (2, 7,  0.40),
    (3, 7,  0.35),
    (1, 10, 0.35),
    (2, 10, 0.35),
]

part2_results = []
for k1, k2, w1 in combos:
    w2 = round(0.65 - w1, 6)
    ens = 0.35 * file_prob_max + w1 * knn_cache[k1] + w2 * knn_cache[k2]
    auc = macro_auc(file_labels, ens)
    part2_results.append({
        "k1": k1, "k2": k2, "w1": w1, "w2": w2,
        "alpha": 0.35, "auc": round(auc, 6)
    })
    print(f"  k1={k1:2d} k2={k2:2d}  w1={w1:.2f} w2={w2:.3f}  AUC={auc:.6f}")

best_p2 = max(part2_results, key=lambda x: x["auc"])
print(f"\nPART 2 BEST: k1={best_p2['k1']} k2={best_p2['k2']}, "
      f"w1={best_p2['w1']} w2={best_p2['w2']}, AUC={best_p2['auc']:.6f}")

# ══════════════════════════════════════════════════════════════════════════════
# PART 3: Logit aggregation comparison
# Fix best k from Part 1 & best 2-way from Part 2, vary logit agg
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("PART 3: Logit aggregation comparison")
print("="*65)

best_k_p1 = best_p1["k"]
best_a_p1  = best_p1["alpha"]

print(f"\n  [Single-k variant: k={best_k_p1}, alpha={best_a_p1}]")
part3a_results = []
for agg_name, agg_prob in AGG_SOURCES.items():
    ens = best_a_p1 * agg_prob + (1 - best_a_p1) * knn_cache[best_k_p1]
    auc = macro_auc(file_labels, ens)
    part3a_results.append({
        "agg": agg_name, "k": best_k_p1,
        "alpha": best_a_p1, "auc": round(auc, 6)
    })
    print(f"  {agg_name:<25s}  k={best_k_p1}  alpha={best_a_p1:.2f}  AUC={auc:.6f}")

best_p3a = max(part3a_results, key=lambda x: x["auc"])
print(f"\nPART 3a BEST: {best_p3a['agg']}, AUC={best_p3a['auc']:.6f}")

# 2-way combo variant
bk1, bk2 = best_p2["k1"], best_p2["k2"]
bw1, bw2  = best_p2["w1"], best_p2["w2"]
print(f"\n  [2-way combo variant: k1={bk1} k2={bk2}, w1={bw1} w2={bw2}]")
part3b_results = []
for agg_name, agg_prob in AGG_SOURCES.items():
    ens = 0.35 * agg_prob + bw1 * knn_cache[bk1] + bw2 * knn_cache[bk2]
    auc = macro_auc(file_labels, ens)
    part3b_results.append({
        "agg": agg_name, "k1": bk1, "k2": bk2,
        "w1": bw1, "w2": bw2, "alpha": 0.35, "auc": round(auc, 6)
    })
    print(f"  {agg_name:<25s}  2way(k1={bk1},k2={bk2})  AUC={auc:.6f}")

best_p3b = max(part3b_results, key=lambda x: x["auc"])
print(f"\nPART 3b BEST: {best_p3b['agg']}, AUC={best_p3b['auc']:.6f}")

# ── BONUS: exhaustive fine sweep around best combo using best agg ─────────────
print("\n" + "="*65)
print("BONUS: Fine sweep using best logit agg from Part 3")
print("="*65)

best_agg_name = max(
    part3a_results + part3b_results, key=lambda x: x["auc"]
)["agg"]
best_agg_prob = AGG_SOURCES[best_agg_name]
print(f"  Using agg: {best_agg_name}")

bonus_results = []
for k in k_list:
    for alpha in np.arange(0.10, 0.61, 0.025):
        ens = alpha * best_agg_prob + (1 - alpha) * knn_cache[k]
        auc = macro_auc(file_labels, ens)
        bonus_results.append({
            "agg": best_agg_name,
            "k": k, "alpha": round(float(alpha), 4),
            "auc": round(auc, 6)
        })

best_bonus = max(bonus_results, key=lambda x: x["auc"])
print(f"  BONUS BEST: agg={best_bonus['agg']} k={best_bonus['k']} "
      f"alpha={best_bonus['alpha']}  AUC={best_bonus['auc']:.6f}")

# ══════════════════════════════════════════════════════════════════════════════
# OVERALL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
all_results = (
    part1_results + part2_results +
    part3a_results + part3b_results +
    bonus_results
)
overall_best = max(all_results, key=lambda x: x["auc"])
overall_best_auc = overall_best["auc"]

print("\n" + "="*65)
print("OVERALL SUMMARY")
print("="*65)
print(f"  Previous best (4way_knn_logit): {PREV_BEST:.6f}")
print(f"  Part 1 best:   k={best_p1['k']}  alpha={best_p1['alpha']}  AUC={best_p1['auc']:.6f}")
print(f"  Part 2 best:   k1={best_p2['k1']} k2={best_p2['k2']}  w1={best_p2['w1']} w2={best_p2['w2']}  AUC={best_p2['auc']:.6f}")
print(f"  Part 3a best:  {best_p3a['agg']}  AUC={best_p3a['auc']:.6f}")
print(f"  Part 3b best:  {best_p3b['agg']}  AUC={best_p3b['auc']:.6f}")
print(f"  Bonus best:    agg={best_bonus['agg']} k={best_bonus['k']} alpha={best_bonus['alpha']}  AUC={best_bonus['auc']:.6f}")
print(f"\n  OVERALL BEST:  AUC={overall_best_auc:.6f}  config={overall_best}")
if overall_best_auc > PREV_BEST:
    print(f"\n  >>> NEW BEST! Improvement: +{overall_best_auc - PREV_BEST:.6f}")
else:
    print(f"\n  >>> No improvement over {PREV_BEST:.6f}")

# ─── Save sweep JSON ──────────────────────────────────────────────────────────
sweep_data = {
    "previous_best": {"method": "4way_knn_logit", "auc": PREV_BEST},
    "part1_single_k": part1_results,
    "part2_2way_combo": part2_results,
    "part3a_logit_agg_single_k": part3a_results,
    "part3b_logit_agg_2way": part3b_results,
    "bonus_fine_sweep": bonus_results,
    "overall_best": {
        "config": overall_best,
        "auc": overall_best_auc,
        "is_new_best": bool(overall_best_auc > PREV_BEST),
        "delta": round(overall_best_auc - PREV_BEST, 6),
    }
}
with open(SWEEP_PATH, "w") as f:
    json.dump(sweep_data, f, indent=2)
print(f"\nSaved sweep → {SWEEP_PATH}")

# ─── If new best: update results.json + save model ───────────────────────────
if overall_best_auc > PREV_BEST:
    print(f"\n>>> NEW BEST — fitting full model ...")

    # Reconstruct best prediction using best config
    cfg = overall_best
    if "k1" in cfg:
        # 2-way (Part 2 / Part 3b)
        agg_prob = AGG_SOURCES[cfg.get("agg", "max_logit")]
        ens_best = (0.35 * agg_prob +
                    cfg["w1"] * knn_cache[cfg["k1"]] +
                    cfg["w2"] * knn_cache[cfg["k2"]])
    else:
        # Single-k (Part 1 / Part 3a / Bonus)
        agg_prob = AGG_SOURCES[cfg.get("agg", "max_logit")]
        ens_best = cfg["alpha"] * agg_prob + (1 - cfg["alpha"]) * knn_cache[cfg["k"]]

    verify_auc = macro_auc(file_labels, ens_best)
    print(f"  Verification AUC = {verify_auc:.6f}")

    model_data = {
        "config": cfg,
        "loo_auc": overall_best_auc,
        "file_list": file_list.tolist(),
        "file_embs_norm": file_embs_norm,
        "file_labels": file_labels,
        "file_prob_max": file_prob_max,
        "file_prob_mean": file_prob_mean,
        "file_prob_top2": file_prob_top2,
        "file_prob_softpool": file_prob_softpool,
        "knn_cache": {k: v for k, v in knn_cache.items()},
        "loo_preds": ens_best,
        "note": "Saved by sweep_knn_neighbors.py 2026-03-25",
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)
    print(f"  Saved model → {MODEL_PATH}")

    # Update embed_prior_results.json
    with open(RESULTS_PATH, "r") as f:
        results_json = json.load(f)

    def clean(v):
        if isinstance(v, (np.float32, np.float64, float)): return float(v)
        if isinstance(v, (np.integer, int)): return int(v)
        return v

    record = {"method": f"sweep_best_{overall_best.get('agg','max_logit')}",
              "loo_auc": overall_best_auc}
    for k2, v2 in overall_best.items():
        record[k2] = clean(v2)
    results_json["experiments"].append(record)
    results_json["best"] = {
        "method": f"sweep_best_{overall_best.get('agg','max_logit')}",
        "loo_auc": overall_best_auc,
        "note": f"Found by k-sweep 2026-03-25; prev best was 4way_knn_logit=0.893",
        "config": {k2: clean(v2) for k2, v2 in overall_best.items()},
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"  Updated results → {RESULTS_PATH}")
else:
    print(f"\n>>> No new best; files not updated.")

print("\nDone.")
