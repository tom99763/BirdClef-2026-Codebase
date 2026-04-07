"""
Embed Prior Round 2b — Advanced experiments targeting >0.8941
Focus: extensions of the winning k134 formula that haven't been tried yet.

New ideas:
  E: k=1,2,3,4,5 all-k ensemble (extend k134 to k12345)
  F: Temperature-scaled logit in k134 formula (T sweep)
  G: Rank-based similarity (use rank instead of cosine sim for KNN weights)
  H: Asymmetric logit blend: different alpha for species seen vs unseen in training
"""
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.metrics import roc_auc_score

BASE = Path("/home/lab/BirdClef-2026-Codebase")
data = np.load(BASE / "outputs/perch_labeled_ss.npz", allow_pickle=True)
embeddings = data["emb"]
labels     = data["labels"]
logits     = data["logits"]
file_ids   = data["filenames"]

emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
unique_files = np.unique(file_ids)
N_FILES = len(unique_files)
N_CLASSES = labels.shape[1]

results_path = BASE / "outputs/embed_prior_results.json"
with open(results_path) as f:
    results = json.load(f)

CURRENT_BEST_AUC = results["best"]["loo_auc"]
print(f"Current best LOO-AUC: {CURRENT_BEST_AUC:.6f} ({results['best']['method']})")

def compute_macro_auc(file_true_all, file_pred_all):
    Y_true = np.array(file_true_all)
    Y_pred = np.array(file_pred_all)
    aucs = []
    for c in range(N_CLASSES):
        yt = Y_true[:, c]; yp = Y_pred[:, c]
        if yt.sum() > 0 and yt.sum() < len(yt):
            try: aucs.append(roc_auc_score(yt, yp))
            except: pass
    return np.mean(aucs) if aucs else 0.0

sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x.astype(np.float32)))

# ── Precompute per-file averages ──────────────────────────────────────────────
print("Precomputing per-file structures...")
# Per-window: just store emb_norm, labels, sigmoid(logits)
sig_logit = sigmoid(logits)   # (739, 234)

# For per-file max logit
file_logit_max = {}
file_label_max = {}
file_emb_mean_norm = {}
for fi in unique_files:
    m = file_ids == fi
    file_logit_max[fi] = sig_logit[m].max(0)
    file_label_max[fi] = labels[m].max(0)
    fe = emb_norm[m].mean(0)
    file_emb_mean_norm[fi] = fe / (np.linalg.norm(fe) + 1e-8)

# ── Method E: k12345 ensemble (extend k134 to include k=2,5) ─────────────────
print("\n=== Method E: k12345 ensemble (all k=1..5) ===")

# Current best: al*logit + w1*KNN(1) + w3*KNN(3) + w4*KNN(4)
# Try adding k=2 and k=5 with small weights

# Build full KNN similarity matrix per LOO fold
best_e = {"auc": 0.0}

# Sweep: al, w1, w2, w3, w4, w5; constrained sum <= 1
# Use grid around best known config: al=0.42, w1=0.28, w3=0.02, w4=0.28
# Try extending with w2 and w5 in [0, 0.1, 0.2]

base_al = 0.42
base_w = {1: 0.28, 3: 0.02, 4: 0.28}

count = 0
for w2 in [0.0, 0.05, 0.10, 0.15]:
    for w5 in [0.0, 0.05, 0.10]:
        # Rescale base weights to accommodate w2, w5
        extra = w2 + w5
        if extra > 0.3: continue
        scale = (1.0 - base_al - extra) / (base_w[1] + base_w[3] + base_w[4])
        sw1 = base_w[1] * scale
        sw3 = base_w[3] * scale
        sw4 = base_w[4] * scale

        file_true_all = []
        file_pred_all = []
        for hf in unique_files:
            tr_m = file_ids != hf
            te_m = file_ids == hf
            X_tr = emb_norm[tr_m]; X_te = emb_norm[te_m]
            L_te = sig_logit[te_m]; Y_tr = labels[tr_m]; Y_te = labels[te_m]
            sims = X_te @ X_tr.T

            def _knn_score(k):
                topk = np.argsort(-sims, axis=1)[:, :k]
                w = np.take_along_axis(sims, topk, axis=1).clip(0, 1)
                w = w / (w.sum(1, keepdims=True) + 1e-8)
                return (w[:, :, None] * Y_tr[topk]).sum(1)

            score = base_al * L_te + sw1 * _knn_score(1) + w2 * _knn_score(2) + sw3 * _knn_score(3) + sw4 * _knn_score(4) + w5 * _knn_score(5)
            file_true_all.append(Y_te.max(0))
            file_pred_all.append(score.mean(0))

        auc = compute_macro_auc(file_true_all, file_pred_all)
        if auc > best_e["auc"]:
            best_e = {"auc": auc, "al": base_al, "w1": sw1, "w2": w2, "w3": sw3, "w4": sw4, "w5": w5}
        count += 1

print(f"  Tried {count} configs; best={best_e['auc']:.6f}")
print(f"  Config: {best_e}")

# ── Method F: Temperature-scaled logit in k134 ───────────────────────────────
print("\n=== Method F: Temperature-scaled logit in k134 ===")

best_f = {"auc": 0.0}
best_al_k134 = 0.42
best_w1_k134 = 0.28; best_w3_k134 = 0.02; best_w4_k134 = 0.28

for T in [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
    scaled_sig = sigmoid(logits / T)
    for al in [0.30, 0.36, 0.42, 0.48]:
        file_true_all = []
        file_pred_all = []
        for hf in unique_files:
            tr_m = file_ids != hf; te_m = file_ids == hf
            X_tr = emb_norm[tr_m]; X_te = emb_norm[te_m]
            L_te = scaled_sig[te_m]; Y_tr = labels[tr_m]; Y_te = labels[te_m]
            sims = X_te @ X_tr.T

            def _knn_score(k):
                topk = np.argsort(-sims, axis=1)[:, :k]
                w = np.take_along_axis(sims, topk, axis=1).clip(0, 1)
                w = w / (w.sum(1, keepdims=True) + 1e-8)
                return (w[:, :, None] * Y_tr[topk]).sum(1)

            score = al * L_te + best_w1_k134 * _knn_score(1) + best_w3_k134 * _knn_score(3) + best_w4_k134 * _knn_score(4)
            file_true_all.append(Y_te.max(0))
            file_pred_all.append(score.mean(0))

        auc = compute_macro_auc(file_true_all, file_pred_all)
        if auc > best_f["auc"]:
            best_f = {"auc": auc, "T": T, "al": al, "w1": best_w1_k134, "w3": best_w3_k134, "w4": best_w4_k134}

print(f"  best={best_f['auc']:.6f}  T={best_f.get('T')}, al={best_f.get('al')}")

# ── Method G: Rank-weighted KNN (use 1/rank as weight) ───────────────────────
print("\n=== Method G: Rank-weighted KNN ===")

best_g = {"auc": 0.0}

for k_knn in [3, 5]:
    for al in [0.25, 0.35, 0.40, 0.45]:
        file_true_all = []
        file_pred_all = []
        for hf in unique_files:
            tr_m = file_ids != hf; te_m = file_ids == hf
            X_tr = emb_norm[tr_m]; X_te = emb_norm[te_m]
            L_te = sig_logit[te_m]; Y_tr = labels[tr_m]; Y_te = labels[te_m]
            sims = X_te @ X_tr.T

            topk_idx = np.argsort(-sims, axis=1)[:, :k_knn]
            # Rank weights: 1/1, 1/2, ..., 1/k
            ranks = np.arange(1, k_knn + 1, dtype=np.float32)  # (k,)
            rank_w = 1.0 / ranks  # (k,)
            rank_w = rank_w / rank_w.sum()  # normalize
            rank_w = rank_w[None, :, None]  # (1, k, 1) for broadcasting

            score = al * L_te + (1 - al) * (rank_w * Y_tr[topk_idx]).sum(1)
            file_true_all.append(Y_te.max(0))
            file_pred_all.append(score.mean(0))

        auc = compute_macro_auc(file_true_all, file_pred_all)
        if auc > best_g["auc"]:
            best_g = {"auc": auc, "k": k_knn, "al": al}

print(f"  best={best_g['auc']:.6f}  k={best_g.get('k')}, al={best_g.get('al')}")

# ── Method H: Ultra-fine grid around k134_ultrafine_v2 ───────────────────────
print("\n=== Method H: Ultra-fine grid around current best ===")

# Current best: al=0.42, w1=0.28, w3=0.02, w4=0.28, AUC=0.894048
best_h = {"auc": 0.0}

for al in np.arange(0.38, 0.50, 0.01):
    for w1 in np.arange(0.20, 0.38, 0.02):
        for w4 in np.arange(0.20, 0.38, 0.02):
            for w3 in [0.0, 0.01, 0.02, 0.03, 0.05]:
                if al + w1 + w3 + w4 > 1.01: continue
                file_true_all = []
                file_pred_all = []
                for hf in unique_files:
                    tr_m = file_ids != hf; te_m = file_ids == hf
                    X_tr = emb_norm[tr_m]; X_te = emb_norm[te_m]
                    L_te = sig_logit[te_m]; Y_tr = labels[tr_m]; Y_te = labels[te_m]
                    sims = X_te @ X_tr.T

                    def _knn_score_h(k):
                        topk = np.argsort(-sims, axis=1)[:, :k]
                        w = np.take_along_axis(sims, topk, axis=1).clip(0, 1)
                        w = w / (w.sum(1, keepdims=True) + 1e-8)
                        return (w[:, :, None] * Y_tr[topk]).sum(1)

                    score = al * L_te + w1 * _knn_score_h(1) + w3 * _knn_score_h(3) + w4 * _knn_score_h(4)
                    file_true_all.append(Y_te.max(0))
                    file_pred_all.append(score.mean(0))

                auc = compute_macro_auc(file_true_all, file_pred_all)
                if auc > best_h["auc"]:
                    best_h = {"auc": auc, "al": round(al, 3), "w1": round(w1, 3), "w3": round(w3, 3), "w4": round(w4, 3)}

print(f"  best={best_h['auc']:.6f}  al={best_h.get('al')}, w1={best_h.get('w1')}, w3={best_h.get('w3')}, w4={best_h.get('w4')}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY Round 2b")
print("="*60)
print(f"Current best: {CURRENT_BEST_AUC:.6f}")
print(f"E (k12345):   {best_e['auc']:.6f}")
print(f"F (T-scaled): {best_f['auc']:.6f}")
print(f"G (rank-KNN): {best_g['auc']:.6f}")
print(f"H (ultra-fine): {best_h['auc']:.6f}")

new_exps = [
    {"method": "k12345_ensemble", "loo_auc": round(best_e["auc"], 6), **{k: v for k, v in best_e.items() if k != "auc"}},
    {"method": "temp_scaled_logit_k134", "loo_auc": round(best_f["auc"], 6), **{k: v for k, v in best_f.items() if k != "auc"}},
    {"method": "rank_weighted_knn", "loo_auc": round(best_g["auc"], 6), **{k: v for k, v in best_g.items() if k != "auc"}},
    {"method": "k134_hyperfine", "loo_auc": round(best_h["auc"], 6), **{k: v for k, v in best_h.items() if k != "auc"}},
]

all_new = [(e["loo_auc"], e) for e in new_exps]
best_new_auc, best_new_exp = max(all_new)
print(f"\nBest new: {best_new_exp['method']} AUC={best_new_auc:.6f}")

results["experiments"].extend(new_exps)

if best_new_auc > CURRENT_BEST_AUC:
    print(f"*** NEW BEST ***")
    results["best"] = {
        "method": best_new_exp["method"],
        "loo_auc": best_new_auc,
        "config": {k: v for k, v in best_new_exp.items() if k not in ("method", "loo_auc")},
        "note": f"Found by embed_prior_round2b.py 2026-03-25; prev={results['best']['method']}={CURRENT_BEST_AUC:.6f}"
    }
else:
    print(f"No improvement.")

with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

with open(BASE / "outputs/embed_prior_round2b_best.json", "w") as f:
    json.dump({
        "current_best_auc": CURRENT_BEST_AUC,
        "best_new_auc": best_new_auc,
        "best_new_method": best_new_exp["method"],
        "all_new": [{"method": e["method"], "loo_auc": e["loo_auc"]} for e in new_exps]
    }, f, indent=2)
print("Saved.")
