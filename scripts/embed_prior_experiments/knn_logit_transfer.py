"""
KNN Logit Transfer + Pyro Bayesian Temperature Calibration
目標：超越 k134_ultrafine_v2 (LOO-AUC=0.8940)

新概念：
1. KNN Logit Transfer - 用鄰居的 logits（soft signal）而非 binary labels
2. Pyro Bayesian Temperature - 推斷 per-species temperature，得到校準 logit
"""
import numpy as np
import json
import pickle
import os
import sys
from sklearn.metrics import roc_auc_score

# ── Load data ─────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data = np.load(os.path.join(ROOT, "outputs", "perch_labeled_ss.npz"))
emb    = data["emb"].astype(np.float32)          # [739, 1536]
lab    = data["labels"].astype(np.float32)       # [739, 234]
logits = data["logits"].astype(np.float32)       # [739, 234]
fnames = data["filenames"]                       # [739] — filename per window
unique_files = data["file_list"]                 # [66] unique filenames
# Build file_id index (string filenames)
fid = fnames
print(f"Loaded: {emb.shape}, {lab.shape}, {logits.shape}, {len(unique_files)} files")

RESULTS_PATH = os.path.join(ROOT, "outputs", "embed_prior_results.json")
with open(RESULTS_PATH) as f:
    results = json.load(f)
BEST_AUC = results["best"]["loo_auc"]
print(f"Current best: {BEST_AUC:.6f} ({results['best']['method']})")

# ── Helpers ───────────────────────────────────────────────────────────────────
def l2norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x.clip(-20, 20)))

def compute_loo_auc(score_fn):
    scores_all, true_all = [], []
    for hf in unique_files:
        tr = fid != hf; te = fid == hf
        X_tr, Y_tr, L_tr = emb[tr], lab[tr], logits[tr]
        X_te, Y_te, L_te = emb[te], lab[te], logits[te]
        s = score_fn(X_tr, Y_tr, L_tr, X_te, L_te)
        if s.ndim == 2:
            file_score = s.mean(axis=0)
        else:
            file_score = s
        scores_all.append(file_score)
        true_all.append(Y_te.max(0))
    S = np.array(scores_all)   # [66, 234]
    T = np.array(true_all)     # [66, 234]
    aucs = []
    for c in range(234):
        if 0 < T[:, c].sum() < 66:
            try:
                aucs.append(roc_auc_score(T[:, c], S[:, c]))
            except Exception:
                pass
    return np.mean(aucs)

# ── Experiment 1: KNN Logit Transfer ─────────────────────────────────────────
print("\n=== Experiment 1: KNN Logit Transfer ===")
# Use neighbor LOGITS instead of neighbor binary labels
# score = sum(w_i * sigmoid(logit_neighbor_i))  where w_i = cosine similarity

def knn_logit_transfer_fn(k, alpha_logit):
    def fn(X_tr, Y_tr, L_tr, X_te, L_te):
        tr_n = l2norm(X_tr); te_n = l2norm(X_te)
        sims = te_n @ tr_n.T  # [n_te, n_tr]
        topk = np.argsort(-sims, axis=1)[:, :k]
        w = np.take_along_axis(sims, topk, axis=1).clip(0)
        w /= (w.sum(1, keepdims=True) + 1e-8)
        neighbor_probs = sigmoid(L_tr[topk])  # [n_te, k, 234]
        klt_score = (w[:, :, None] * neighbor_probs).sum(1)  # [n_te, 234]
        # max logit aggregation
        logit_max_score = sigmoid(L_te.max(axis=0))  # [234]
        # combine
        return alpha_logit * logit_max_score + (1 - alpha_logit) * klt_score.mean(0)
    return fn

best_klt_auc = 0
best_klt_cfg = {}
for k in [1, 2, 3, 4, 5]:
    for alpha in [0.30, 0.35, 0.40, 0.42, 0.45]:
        auc = compute_loo_auc(knn_logit_transfer_fn(k, alpha))
        print(f"  KLT k={k} alpha={alpha:.2f}: {auc:.6f}")
        if auc > best_klt_auc:
            best_klt_auc = auc
            best_klt_cfg = {"k": k, "alpha": alpha}

print(f"Best KLT: AUC={best_klt_auc:.6f} cfg={best_klt_cfg}")

# ── Experiment 2: k134 KLT (logit transfer for k=1,3,4) ─────────────────────
print("\n=== Experiment 2: k134 KLT combo ===")

def k134_klt_fn(al, w1, w3, w4):
    def fn(X_tr, Y_tr, L_tr, X_te, L_te):
        tr_n = l2norm(X_tr); te_n = l2norm(X_te)
        sims = te_n @ tr_n.T
        def klt(k):
            topk = np.argsort(-sims, axis=1)[:, :k]
            w = np.take_along_axis(sims, topk, axis=1).clip(0)
            w /= (w.sum(1, keepdims=True) + 1e-8)
            return (w[:, :, None] * sigmoid(L_tr[topk])).sum(1).mean(0)  # [234]
        logit_max = sigmoid(L_te.max(0))
        return al * logit_max + w1 * klt(1) + w3 * klt(3) + w4 * klt(4)
    return fn

# Use best-known weights from label-based k134, try KLT version
configs = [
    (0.42, 0.28, 0.02, 0.28),
    (0.40, 0.30, 0.02, 0.28),
    (0.38, 0.30, 0.04, 0.28),
    (0.35, 0.32, 0.02, 0.31),
    (0.42, 0.25, 0.05, 0.28),
]
best_k134klt_auc = 0
best_k134klt_cfg = {}
for al, w1, w3, w4 in configs:
    auc = compute_loo_auc(k134_klt_fn(al, w1, w3, w4))
    print(f"  k134-KLT al={al} w1={w1} w3={w3} w4={w4}: {auc:.6f}")
    if auc > best_k134klt_auc:
        best_k134klt_auc = auc
        best_k134klt_cfg = {"al": al, "w1": w1, "w3": w3, "w4": w4}

print(f"Best k134-KLT: AUC={best_k134klt_auc:.6f} cfg={best_k134klt_cfg}")

# ── Experiment 3: Mixed label + logit transfer KNN ───────────────────────────
print("\n=== Experiment 3: Mixed label+logit KNN ===")
# Hybrid: score = alpha*logit_max + beta*KNN_labels + gamma*KNN_logit_transfer

def hybrid_label_klt_fn(al, bl, gl, k):
    def fn(X_tr, Y_tr, L_tr, X_te, L_te):
        tr_n = l2norm(X_tr); te_n = l2norm(X_te)
        sims = te_n @ tr_n.T
        topk = np.argsort(-sims, axis=1)[:, :k]
        w = np.take_along_axis(sims, topk, axis=1).clip(0)
        w /= (w.sum(1, keepdims=True) + 1e-8)
        knn_label = (w[:, :, None] * Y_tr[topk]).sum(1).mean(0)          # [234]
        knn_logit = (w[:, :, None] * sigmoid(L_tr[topk])).sum(1).mean(0)  # [234]
        logit_max = sigmoid(L_te.max(0))
        return al * logit_max + bl * knn_label + gl * knn_logit
    return fn

best_hybrid_auc = 0
best_hybrid_cfg = {}
for k in [1, 3, 4]:
    for al in [0.40, 0.42]:
        for ratio in [0.3, 0.5, 0.7]:  # ratio of (bl vs gl)
            bl = (1 - al) * ratio
            gl = (1 - al) * (1 - ratio)
            auc = compute_loo_auc(hybrid_label_klt_fn(al, bl, gl, k))
            if auc > best_hybrid_auc:
                best_hybrid_auc = auc
                best_hybrid_cfg = {"al": al, "bl": bl, "gl": gl, "k": k}

print(f"Best Hybrid label+KLT: AUC={best_hybrid_auc:.6f} cfg={best_hybrid_cfg}")

# ── Experiment 4: Pyro Bayesian Temperature Calibration ──────────────────────
print("\n=== Experiment 4: Pyro Bayesian Temperature Calibration ===")
try:
    import torch
    import pyro
    import pyro.distributions as dist_pyro
    from pyro.infer import SVI, Trace_ELBO
    from pyro.optim import ClippedAdam

    def bayesian_temp_cal_fn(n_steps=150, k_knn=3, al_knn=0.58):
        def fn(X_tr, Y_tr, L_tr, X_te, L_te):
            pyro.clear_param_store()
            L_t = torch.tensor(L_tr, dtype=torch.float32)
            Y_t = torch.tensor(Y_tr, dtype=torch.float32)

            def model(logits, labels):
                log_T = pyro.sample("log_T", dist_pyro.Normal(
                    torch.zeros(234), 0.5 * torch.ones(234)).to_event(1))
                T = torch.exp(log_T).clamp(0.1, 10.0)
                scaled = logits / T[None, :]
                with pyro.plate("data", logits.shape[0]):
                    pyro.sample("obs", dist_pyro.Bernoulli(logits=scaled).to_event(1), obs=labels)

            def guide(logits, labels):
                loc = pyro.param("log_T_loc", torch.zeros(234))
                scale = pyro.param("log_T_scale", 0.1 * torch.ones(234),
                                   constraint=dist_pyro.constraints.positive)
                pyro.sample("log_T", dist_pyro.Normal(loc, scale).to_event(1))

            svi = SVI(model, guide, ClippedAdam({"lr": 0.05}), loss=Trace_ELBO())
            for _ in range(n_steps):
                svi.step(L_t, Y_t)

            T_est = torch.exp(pyro.param("log_T_loc")).detach().numpy().clip(0.1, 10.0)
            logit_max_cal = L_te.max(0) / T_est   # calibrated
            logit_score = sigmoid(logit_max_cal)   # [234]

            # Combine with KNN labels
            tr_n = l2norm(X_tr); te_n = l2norm(X_te)
            sims = te_n @ tr_n.T
            topk = np.argsort(-sims, axis=1)[:, :k_knn]
            w = np.take_along_axis(sims, topk, axis=1).clip(0)
            w /= (w.sum(1, keepdims=True) + 1e-8)
            knn_score = (w[:, :, None] * Y_tr[topk]).sum(1).mean(0)  # [234]

            return al_knn * knn_score + (1 - al_knn) * logit_score
        return fn

    pyro_auc = compute_loo_auc(bayesian_temp_cal_fn(n_steps=150, k_knn=3, al_knn=0.58))
    print(f"  Pyro BayesTemp (k=3, al=0.58): {pyro_auc:.6f}")

    # Try variation with k134 structure
    def pyro_k134_fn(n_steps=100):
        def fn(X_tr, Y_tr, L_tr, X_te, L_te):
            pyro.clear_param_store()
            L_t = torch.tensor(L_tr, dtype=torch.float32)
            Y_t = torch.tensor(Y_tr, dtype=torch.float32)

            def model(logits, labels):
                log_T = pyro.sample("log_T", dist_pyro.Normal(
                    torch.zeros(234), 0.5*torch.ones(234)).to_event(1))
                T = torch.exp(log_T).clamp(0.1, 10.0)
                with pyro.plate("data", logits.shape[0]):
                    pyro.sample("obs", dist_pyro.Bernoulli(
                        logits=logits/T[None,:]).to_event(1), obs=labels)

            def guide(logits, labels):
                loc = pyro.param("log_T_loc", torch.zeros(234))
                scale = pyro.param("log_T_scale", 0.1*torch.ones(234),
                                   constraint=dist_pyro.constraints.positive)
                pyro.sample("log_T", dist_pyro.Normal(loc, scale).to_event(1))

            svi = SVI(model, guide, ClippedAdam({"lr": 0.05}), loss=Trace_ELBO())
            for _ in range(n_steps):
                svi.step(L_t, Y_t)

            T_est = torch.exp(pyro.param("log_T_loc")).detach().numpy().clip(0.1, 10.0)
            logit_score = sigmoid(L_te.max(0) / T_est)  # [234]

            tr_n = l2norm(X_tr); te_n = l2norm(X_te)
            sims = te_n @ tr_n.T
            def knn_lab(k):
                topk = np.argsort(-sims, axis=1)[:, :k]
                w = np.take_along_axis(sims, topk, axis=1).clip(0)
                w /= (w.sum(1, keepdims=True) + 1e-8)
                return (w[:, :, None] * Y_tr[topk]).sum(1).mean(0)

            # k134 structure but with calibrated logit
            return 0.42 * logit_score + 0.28 * knn_lab(1) + 0.02 * knn_lab(3) + 0.28 * knn_lab(4)
        return fn

    pyro_k134_auc = compute_loo_auc(pyro_k134_fn(n_steps=100))
    print(f"  Pyro BayesTemp + k134 (100 steps): {pyro_k134_auc:.6f}")

    pyro_best = max(pyro_auc, pyro_k134_auc)
    print(f"  Best Pyro: {pyro_best:.6f}")
except ImportError as e:
    print(f"  Pyro not available: {e}")
    pyro_auc = pyro_k134_auc = pyro_best = 0

# ── Summary & Update ──────────────────────────────────────────────────────────
print("\n=== 結果摘要 ===")
all_results = [
    ("knn_logit_transfer", best_klt_auc, best_klt_cfg),
    ("k134_knn_logit_transfer", best_k134klt_auc, best_k134klt_cfg),
    ("hybrid_label_klt", best_hybrid_auc, best_hybrid_cfg),
    ("pyro_bayes_temp", pyro_auc if pyro_auc > 0 else 0, {"n_steps": 150, "k": 3}),
    ("pyro_bayes_temp_k134", pyro_k134_auc if pyro_k134_auc > 0 else 0, {"n_steps": 100}),
]
all_results.sort(key=lambda x: -x[1])

overall_best_method, overall_best_auc, overall_best_cfg = all_results[0]
print(f"{'方法':<35} {'LOO-AUC':>10}")
print("-" * 50)
for name, auc, cfg in all_results:
    marker = " ← NEW BEST" if auc > BEST_AUC else ""
    print(f"  {name:<33} {auc:>10.6f}{marker}")
print(f"\n  Baseline best: {BEST_AUC:.6f} (k134_ultrafine_v2)")

# Update results JSON
new_experiments = [
    {"method": name, "loo_auc": round(auc, 6), **cfg}
    for name, auc, cfg in all_results if auc > 0
]
results["experiments"].extend(new_experiments)

if overall_best_auc > BEST_AUC:
    print(f"\nNEW BEST: {overall_best_method} AUC={overall_best_auc:.6f}")
    results["best"] = {
        "method": overall_best_method,
        "loo_auc": overall_best_auc,
        "config": overall_best_cfg,
        "note": f"knn_logit_transfer experiments 2026-03-25; prev={results['best']['method']}={BEST_AUC}"
    }

    # Refit on all data and save model
    def best_model_predict(X_train, Y_train, L_train, X_test, L_test):
        """Best full model predict"""
        if overall_best_method in ("k134_knn_logit_transfer",):
            al = overall_best_cfg.get("al", 0.42)
            w1 = overall_best_cfg.get("w1", 0.28)
            w3 = overall_best_cfg.get("w3", 0.02)
            w4 = overall_best_cfg.get("w4", 0.28)
            tr_n = l2norm(X_train); te_n = l2norm(X_test)
            sims = te_n @ tr_n.T
            def klt(k):
                topk = np.argsort(-sims, axis=1)[:, :k]
                w = np.take_along_axis(sims, topk, axis=1).clip(0)
                w /= (w.sum(1, keepdims=True) + 1e-8)
                return (w[:, :, None] * sigmoid(L_train[topk])).sum(1).mean(0)
            return al * sigmoid(L_test.max(0)) + w1 * klt(1) + w3 * klt(3) + w4 * klt(4)
        return None

    model_data = {
        "method": overall_best_method,
        "loo_auc": overall_best_auc,
        "config": overall_best_cfg,
        "file_embs_norm": l2norm(emb),
        "file_labels": lab,
        "file_logits": logits,
        # file-level aggregated data
        "file_ids": fid,
        "unique_files": unique_files,
    }
    pkl_path = os.path.join(ROOT, "outputs", "embed_prior_model.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"  Model saved → {pkl_path}")
else:
    print(f"\n未超越 best ({BEST_AUC:.6f})，記錄實驗結果。")

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results updated → {RESULTS_PATH}")
