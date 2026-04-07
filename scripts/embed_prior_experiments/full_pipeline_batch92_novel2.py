"""
Batch 92 — Novel Improvement Methods (Non-Ensemble)
Focus: fundamentally different signal sources to break 0.991782 ceiling
Uses EXACT base implementation from batch88 (confirmed correct).

Experiments:
1. site_prior: geographic site prior from labeled data
2. logit_kde: KDE in PCA(Perch logits) space
3. isotonic_calib: per-class isotonic calibration (LOO)
4. spectral_diffusion: label diffusion via embedding affinity graph
5. genus_prior: hierarchical genus-level aggregation (no taxonomy column check)
6. hard_pos_knn: nearest-neighbor among positives only
7. joint_embed_logit_kde: joint embedding+logit KDE
8. window_entropy_weight: weight file score by window entropy
"""
import json, pickle, sys, re, time
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from numpy.linalg import norm

ROOT = Path("/home/lab/BirdClef-2026-Codebase")
RESULTS_PATH = ROOT / "outputs" / "embed_prior_results.json"
MODEL_PATH   = ROOT / "outputs" / "embed_prior_model.pkl"
DATA_PATH    = ROOT / "outputs" / "perch_labeled_ss.npz"

# ─── Load data ────────────────────────────────────────────────────────────────
DATA = np.load(DATA_PATH)
labels_win  = DATA["labels"].astype(np.float32)
logit_win   = DATA["logits"].astype(np.float32)
n_windows   = DATA["n_windows"]
file_list   = DATA["file_list"]
n_files     = len(n_windows)
n_species   = labels_win.shape[1]
file_start  = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end    = np.cumsum(n_windows).astype(np.int32)
win_file_id = np.zeros(739, np.int32)
for fi in range(n_files):
    win_file_id[file_start[fi]:file_end[fi]] = fi
EPS = 1e-8

with open(MODEL_PATH, "rb") as f:
    ep = pickle.load(f)

ew_ica = ep["emb_win_ica_norm"]
ew_pca = ep["emb_win_pca_norm"]
ew_std = ep["emb_win_std_norm"]
ew_nmf = ep["emb_win_nmf_norm"]
file_labels = ep["file_labels"]
cfg = ep["config"]

# Raw embeddings for new experiments
emb_raw = DATA["emb"].astype(np.float32)

print(f"Loaded: ICA{ew_ica.shape} NMF{ew_nmf.shape}")

# ─── Load JSON ───────────────────────────────────────────────────────────────
res = json.load(open(RESULTS_PATH))
best = res["best"]
best_loo = best["loo_auc"]
print(f"[batch92] Current best: {best['method']} LOO={best_loo:.6f}")

# ─── Helpers (EXACT copy from batch88) ───────────────────────────────────────
def macro_auc(s):
    aucs = []
    for si in range(n_species):
        y = file_labels[:, si]
        if y.sum() > 0 and y.sum() < n_files:
            aucs.append(roc_auc_score(y, s[:, si]))
    return float(np.mean(aucs)) if aucs else 0.0

def wl_loo(ew, k_neg, wmp, wma):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            pw = tr[pm]; ps = te @ pw.T
            pp = pw.mean(0); pp /= norm(pp) + EPS
            sp = wmp * ps.max(1) + (1 - wmp) * (te @ pp)
            if nm.any() and k_neg > 0:
                nw = tr[nm]; ns = te @ nw.T; k2 = min(k_neg, ns.shape[1])
                if k2 > 0:
                    tn = nw[np.argsort(-ns, axis=1)[:, :k2]].mean(1)
                    tn /= norm(tn, axis=1, keepdims=True) + EPS
                    ws[:, si] = (sp - (te * tn).sum(1) + 1) / 2
                else:
                    ws[:, si] = (sp + 1) / 2
            else:
                ws[:, si] = (sp + 1) / 2
        out[fi] = wma * ws.max(0) + (1 - wma) * ws.mean(0)
    return out

def make_logit_pred(T, agg="max"):
    sig = 1.0 / (1.0 + np.exp(np.clip(-logit_win / T, -88, 88)))
    if agg == "max":
        return np.stack([sig[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)])
    elif agg == "mean":
        return np.stack([sig[file_start[fi]:file_end[fi]].mean(0) for fi in range(n_files)])

def make_softmax_pred(T, agg="max"):
    sm_r = logit_win / T; sm_r -= sm_r.max(1, keepdims=True)
    sm_e = np.exp(sm_r); smp = sm_e / (sm_e.sum(1, keepdims=True) + EPS)
    if agg == "max":
        return np.stack([smp[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)])

def compute_subspace(ew_sp, n_comp=2, wma_ss=0.92):
    ss = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew_sp[win_file_id == fi]; tr = ew_sp[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        dim = te.shape[1]
        for si in range(n_species):
            pm = tl[:, si] > 0.5
            if not pm.any(): ws[:, si] = 0.5; continue
            pos = tr[pm]; k = min(n_comp, len(pos) - 1, dim - 1)
            if k < 1:
                pp = pos.mean(0); pp /= norm(pp) + EPS
                ws[:, si] = np.clip((te @ pp + 1) / 2, 0, 1); continue
            try:
                pca_sp = SklearnPCA(n_components=k); pca_sp.fit(pos)
                te_r = pca_sp.inverse_transform(pca_sp.transform(te))
                err = norm(te - te_r, axis=1)
                ws[:, si] = np.clip(1 - err / (norm(te, axis=1) + EPS), 0, 1)
            except Exception:
                ws[:, si] = 0.5
        ss[fi] = wma_ss * ws.max(0) + (1 - wma_ss) * ws.mean(0)
    return ss

def proto_kde_loo_ica(bw=0.08):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew_ica[win_file_id == fi]; tr = ew_ica[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        sims = te @ tr.T
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pos_idx = np.where(tl[:, si] > 0.5)[0]
            if len(pos_idx) == 0: ws[:, si] = 0.5; continue
            pos_wins = tr[pos_idx]
            centroid = pos_wins.mean(0); centroid /= (norm(centroid) + EPS)
            proto_w = np.clip(pos_wins @ centroid, 0, None)
            proto_w = proto_w / (proto_w.sum() + EPS)
            kern = np.exp((sims[:, pos_idx] - 1.0) / (bw**2 + EPS))
            ws[:, si] = np.clip((kern * proto_w[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

# ─── Pre-compute base ─────────────────────────────────────────────────────────
print("Pre-computing base components...", flush=True)
t0 = time.time()
s_ica = wl_loo(ew_ica, cfg["ica100"]["k_neg"], cfg["ica100"]["w_max_pos"], cfg["ica100"]["w_max_agg"])
s_std = wl_loo(ew_std, cfg["std_pca80"]["k_neg"], cfg["std_pca80"]["w_max_pos"], cfg["std_pca80"]["w_max_agg"])
s_pca = wl_loo(ew_pca, cfg["pca80"]["k_neg"], cfg["pca80"]["w_max_pos"], cfg["pca80"]["w_max_agg"])
s_nmf = wl_loo(ew_nmf, cfg["nmf"]["k_neg"], cfg["nmf"]["w_max_pos"], cfg["nmf"]["w_max_agg"])
uh_b  = cfg["w_ica100"] * s_ica + cfg["w_std"] * s_std + cfg["w_pca80"] * s_pca
uh_nmf = cfg["nmf"]["uh_scale"] * uh_b + cfg["nmf"]["w_nmf"] * s_nmf
pT8  = make_logit_pred(cfg["logit_temperature"])
pmt  = (pT8 + make_logit_pred(10.0)) / 2
sm6  = make_softmax_pred(cfg["softmax_temp"])
ss2  = compute_subspace(ew_pca, n_comp=2, wma_ss=0.92)
print(f"  Base done ({time.time()-t0:.0f}s)", flush=True)

w_uh = 1 - cfg["w_logit"] - cfg["w_multit"] - cfg["w_subspace"] - cfg["w_softmax"]
base_cur = w_uh * uh_nmf + cfg["w_logit"] * pT8 + cfg["w_multit"] * pmt + cfg["w_subspace"] * ss2 + cfg["w_softmax"] * sm6
kde_ref  = proto_kde_loo_ica(bw=0.08)
final_cur = 0.96 * base_cur + 0.04 * kde_ref

auc_base  = macro_auc(base_cur)
auc_final = macro_auc(final_cur)
print(f"Base AUC:  {auc_base:.6f} (should ~0.991359)")
print(f"Final AUC: {auc_final:.6f} (expected {best_loo:.6f})")

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Site prior (geographic)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[EXP1] Site prior...", flush=True)

def parse_site(fn):
    m = re.search(r'_S(\d+)_', fn)
    return m.group(1) if m else 'UNK'

file_sites = np.array([parse_site(f) for f in file_list])

def site_prior_loo():
    scores = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        tr_mask = np.ones(n_files, bool); tr_mask[fi] = False
        same_site = file_sites[tr_mask] == file_sites[fi]
        if same_site.sum() > 0:
            scores[fi] = file_labels[tr_mask][same_site].mean(0)
        else:
            scores[fi] = file_labels[tr_mask].mean(0)
    return scores

s_site = site_prior_loo()
a_site = macro_auc(s_site)
print(f"  site_prior alone: LOO={a_site:.6f}")

for w in [0.01, 0.02, 0.03, 0.04, 0.05]:
    cand = (1 - w) * final_cur + w * s_site
    a = macro_auc(cand)
    delta = a - auc_final
    if abs(delta) < 0.002 or delta > 0:
        print(f"  site_prior blend w={w}: LOO={a:.6f} delta={delta:+.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Perch logit-space proto-KDE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[EXP2] Logit-space proto-KDE...", flush=True)

# Build logit embedding: PCA of sigmoid(logit) features
logit_sig = 1.0 / (1.0 + np.exp(np.clip(-logit_win / 8.0, -88, 88)))
scaler_l = StandardScaler()
logit_scaled = scaler_l.fit_transform(logit_sig)
for n_comp in [30, 50, 80]:
    pca_l = SklearnPCA(n_components=n_comp)
    logit_pca = pca_l.fit_transform(logit_scaled).astype(np.float32)
    logit_pca_n = logit_pca / (norm(logit_pca, axis=1, keepdims=True) + EPS)

    def logit_kde_loo(bw, ew_l=logit_pca_n):
        out = np.zeros((n_files, n_species), np.float32)
        for fi in range(n_files):
            te = ew_l[win_file_id == fi]; tr = ew_l[win_file_id != fi]
            tl = labels_win[win_file_id != fi]
            sims = te @ tr.T
            ws = np.zeros((len(te), n_species), np.float32)
            for si in range(n_species):
                pos_idx = np.where(tl[:, si] > 0.5)[0]
                if len(pos_idx) == 0: ws[:, si] = 0.5; continue
                pos_wins = tr[pos_idx]
                centroid = pos_wins.mean(0); centroid /= (norm(centroid) + EPS)
                proto_w = np.clip(pos_wins @ centroid, 0, None)
                proto_w = proto_w / (proto_w.sum() + EPS)
                kern = np.exp((sims[:, pos_idx] - 1.0) / (bw**2 + EPS))
                ws[:, si] = np.clip((kern * proto_w[None, :]).sum(1), 0, None)
            for si in range(n_species):
                mx = ws[:, si].max()
                if mx > EPS: ws[:, si] /= mx
            out[fi] = ws.max(0)
        return out

    for bw in [0.05, 0.08, 0.12]:
        s_lkde = logit_kde_loo(bw)
        for w in [0.02, 0.04]:
            cand = (1 - w) * final_cur + w * s_lkde
            a = macro_auc(cand)
            delta = a - auc_final
            if abs(delta) < 0.002 or delta > 0:
                print(f"  logit_kde n_comp={n_comp} bw={bw} w={w}: LOO={a:.6f} delta={delta:+.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Isotonic calibration (LOO, per-class)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[EXP3] Isotonic calibration...", flush=True)

def isotonic_calib_loo(base_scores):
    calib = np.zeros_like(base_scores)
    for si in range(n_species):
        if file_labels[:, si].sum() == 0: continue
        y = file_labels[:, si]; s = base_scores[:, si]
        for fi in range(n_files):
            tr_mask = np.ones(n_files, bool); tr_mask[fi] = False
            if y[tr_mask].sum() == 0:
                calib[fi, si] = s[fi]; continue
            ir = IsotonicRegression(out_of_bounds='clip')
            try:
                ir.fit(s[tr_mask], y[tr_mask])
                calib[fi, si] = ir.predict([s[fi]])[0]
            except:
                calib[fi, si] = s[fi]
    return calib

print("  calibrating base...", flush=True)
s_iso_base = isotonic_calib_loo(base_cur)
a_iso_base = macro_auc(s_iso_base)
print(f"  iso_calib(base) LOO={a_iso_base:.6f} delta={a_iso_base-auc_final:+.6f}")

# Apply KDE on top of calibrated base
for w in [0.02, 0.04, 0.06]:
    cand = (1 - w) * s_iso_base + w * kde_ref
    a = macro_auc(cand)
    print(f"  iso_base + kde w={w}: LOO={a:.6f} delta={a-auc_final:+.6f}")

# Blend iso with final
for w in [0.05, 0.10, 0.20]:
    cand = (1 - w) * final_cur + w * s_iso_base
    a = macro_auc(cand)
    delta = a - auc_final
    if abs(delta) < 0.003 or delta > 0:
        print(f"  blend(final+iso_base) w={w}: LOO={a:.6f} delta={delta:+.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Spectral diffusion label propagation
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[EXP4] Spectral diffusion...", flush=True)

# File mean embeddings (ICA)
file_emb_ica = np.array([ew_ica[win_file_id==fi].mean(0) for fi in range(n_files)])
file_emb_ica_n = file_emb_ica / (norm(file_emb_ica, axis=1, keepdims=True) + EPS)

# Affinity via cosine similarity
cosine_aff = file_emb_ica_n @ file_emb_ica_n.T
np.fill_diagonal(cosine_aff, 0)

for sigma in [0.3, 0.5, 0.7, 1.0]:
    # Soft affinity: exp((sim - 1) / sigma^2)
    W = np.exp((cosine_aff - 1.0) / (sigma**2 + EPS))
    np.fill_diagonal(W, 0)
    W = np.clip(W, 0, None)

    def diffusion_loo(n_steps=1):
        scores = np.zeros((n_files, n_species), np.float32)
        for fi in range(n_files):
            idx = [j for j in range(n_files) if j != fi]
            W_loo = W[np.ix_(idx, idx)]
            D = W_loo.sum(1, keepdims=True) + EPS
            T = W_loo / D

            # Seed with OOF file_labels
            s = file_labels[idx].astype(np.float32)
            for _ in range(n_steps):
                s = T @ s

            # Score for fi from train affinities
            w_fi = W[fi, idx]
            w_fi = w_fi / (w_fi.sum() + EPS)
            scores[fi] = w_fi @ s
        return scores

    s_diff = diffusion_loo(n_steps=1)
    for w in [0.02, 0.04, 0.06]:
        cand = (1 - w) * final_cur + w * s_diff
        a = macro_auc(cand)
        delta = a - auc_final
        if abs(delta) < 0.002 or delta > 0:
            print(f"  diffusion sigma={sigma} w={w}: LOO={a:.6f} delta={delta:+.6f}")

# 2 steps
for sigma in [0.3, 0.5]:
    W = np.exp((cosine_aff - 1.0) / (sigma**2 + EPS))
    np.fill_diagonal(W, 0)
    W = np.clip(W, 0, None)
    def diffusion_loo2(n_steps=2):
        scores = np.zeros((n_files, n_species), np.float32)
        for fi in range(n_files):
            idx = [j for j in range(n_files) if j != fi]
            W_loo = W[np.ix_(idx, idx)]
            D = W_loo.sum(1, keepdims=True) + EPS
            T = W_loo / D
            s = file_labels[idx].astype(np.float32)
            for _ in range(n_steps):
                s = T @ s
            w_fi = W[fi, idx]; w_fi = w_fi / (w_fi.sum() + EPS)
            scores[fi] = w_fi @ s
        return scores
    s_diff2 = diffusion_loo2(2)
    for w in [0.02, 0.04, 0.06]:
        cand = (1 - w) * final_cur + w * s_diff2
        a = macro_auc(cand)
        delta = a - auc_final
        if abs(delta) < 0.002 or delta > 0:
            print(f"  diffusion2 sigma={sigma} w={w}: LOO={a:.6f} delta={delta:+.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5: Per-file window entropy weighting
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[EXP5] Window entropy-weighted aggregation...", flush=True)

# Hypothesis: windows with high confidence (low entropy) should contribute more
def entropy_weighted_agg(bw_ica=0.08):
    """Use window entropy to weight KDE contributions."""
    sig = 1.0 / (1.0 + np.exp(np.clip(-logit_win / 8.0, -88, 88)))
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew_ica[win_file_id == fi]; tr = ew_ica[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        te_sig = sig[win_file_id == fi]  # (T, C)

        # Window entropy: high entropy = uncertain window
        te_ent = -(te_sig * np.log(te_sig + EPS) + (1-te_sig) * np.log(1-te_sig + EPS)).mean(1)  # (T,)
        te_conf = np.exp(-te_ent)  # (T,) - higher for more confident windows
        te_conf = te_conf / (te_conf.sum() + EPS)

        sims = te @ tr.T
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pos_idx = np.where(tl[:, si] > 0.5)[0]
            if len(pos_idx) == 0: ws[:, si] = 0.5; continue
            pos_wins = tr[pos_idx]
            centroid = pos_wins.mean(0); centroid /= (norm(centroid) + EPS)
            proto_w = np.clip(pos_wins @ centroid, 0, None)
            proto_w = proto_w / (proto_w.sum() + EPS)
            kern = np.exp((sims[:, pos_idx] - 1.0) / (bw_ica**2 + EPS))
            ws[:, si] = np.clip((kern * proto_w[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        # Entropy-weighted: sum(conf[t] * ws[t, :])
        out[fi] = (te_conf[:, None] * ws).sum(0)
    return out

s_ent = entropy_weighted_agg(bw_ica=0.08)
for w in [0.02, 0.04, 0.06]:
    cand = (1 - w) * final_cur + w * s_ent
    a = macro_auc(cand)
    delta = a - auc_final
    print(f"  entropy_kde w={w}: LOO={a:.6f} delta={delta:+.6f}")

# Replace kde_ref entirely
for w in [0.02, 0.03, 0.04, 0.05]:
    cand = (1 - w) * base_cur + w * s_ent
    a = macro_auc(cand)
    delta = a - auc_final
    if abs(delta) < 0.002 or delta > 0:
        print(f"  base + entropy_kde w={w}: LOO={a:.6f} delta={delta:+.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 6: Proto-KDE in STD space
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[EXP6] Proto-KDE in alternative spaces...", flush=True)

def proto_kde_in_space(ew, bw=0.08):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        sims = te @ tr.T
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pos_idx = np.where(tl[:, si] > 0.5)[0]
            if len(pos_idx) == 0: ws[:, si] = 0.5; continue
            pos_wins = tr[pos_idx]
            centroid = pos_wins.mean(0); centroid /= (norm(centroid) + EPS)
            proto_w = np.clip(pos_wins @ centroid, 0, None)
            proto_w = proto_w / (proto_w.sum() + EPS)
            kern = np.exp((sims[:, pos_idx] - 1.0) / (bw**2 + EPS))
            ws[:, si] = np.clip((kern * proto_w[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

for space_name, ew_sp in [("std", ew_std), ("pca", ew_pca), ("nmf", ew_nmf)]:
    for bw in [0.05, 0.08, 0.12]:
        s_sp = proto_kde_in_space(ew_sp, bw)
        for w in [0.02, 0.04]:
            cand = (1 - w) * final_cur + w * s_sp
            a = macro_auc(cand)
            delta = a - auc_final
            if delta > 0:
                print(f"  IMPROVED! kde_{space_name} bw={bw} w={w}: LOO={a:.6f} delta={delta:+.6f}")
            elif abs(delta) < 0.0005:
                print(f"  TIE kde_{space_name} bw={bw} w={w}: LOO={a:.6f} delta={delta:+.6f}")

# Also try ICA with finer bw sweep
for bw in [0.04, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12]:
    s_kde_bw = proto_kde_in_space(ew_ica, bw)
    cand = 0.96 * base_cur + 0.04 * s_kde_bw
    a = macro_auc(cand)
    delta = a - auc_final
    if delta > 0:
        print(f"  IMPROVED! kde_ica bw={bw}: LOO={a:.6f} delta={delta:+.6f}")
    elif abs(delta) < 0.0005:
        print(f"  TIE kde_ica bw={bw}: LOO={a:.6f} delta={delta:+.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 7: Dual-space proto-KDE (ICA + STD blend)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[EXP7] Dual-space KDE blend...", flush=True)

kde_ica = proto_kde_in_space(ew_ica, 0.08)
kde_std = proto_kde_in_space(ew_std, 0.08)
kde_pca = proto_kde_in_space(ew_pca, 0.08)
kde_nmf = proto_kde_in_space(ew_nmf, 0.08)

for a1, a2 in [(0.7, 0.3), (0.8, 0.2), (0.6, 0.4), (0.5, 0.5)]:
    for s1, s2 in [("ica+std", kde_ica, kde_std), ("ica+pca", kde_ica, kde_pca), ("ica+nmf", kde_ica, kde_nmf)]:
        kde_blend = a1 * s2[1] + a2 * s2[2]  # wrong indexing, fix:
        pass

for s1n, s1 in [("ica", kde_ica), ("std", kde_std), ("pca", kde_pca), ("nmf", kde_nmf)]:
    for s2n, s2 in [("ica", kde_ica), ("std", kde_std), ("pca", kde_pca), ("nmf", kde_nmf)]:
        if s1n >= s2n: continue
        for a1 in [0.5, 0.6, 0.7]:
            kde_b = a1 * s1 + (1-a1) * s2
            for w in [0.03, 0.04, 0.05]:
                cand = (1-w) * base_cur + w * kde_b
                a = macro_auc(cand)
                delta = a - auc_final
                if delta > 0:
                    print(f"  IMPROVED! kde_{s1n}+{s2n} a1={a1} w={w}: LOO={a:.6f} delta={delta:+.6f}")
                elif abs(delta) < 0.0003:
                    print(f"  TIE kde_{s1n}+{s2n} a1={a1} w={w}: LOO={a:.6f} delta={delta:+.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 8: Proto-KDE with varying w_kde and base_scale
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[EXP8] Fine-tune kde weight and scale...", flush=True)

for scale in [0.94, 0.95, 0.96, 0.97, 0.98]:
    for w_k in [0.02, 0.03, 0.04, 0.05, 0.06]:
        if abs(scale + w_k - 1.0) > 0.01: continue
        cand = scale * base_cur + w_k * kde_ica
        a = macro_auc(cand)
        delta = a - auc_final
        if delta > 0:
            print(f"  IMPROVED! scale={scale} w_k={w_k}: LOO={a:.6f} delta={delta:+.6f}")
        elif abs(delta) < 0.0003:
            print(f"  TIE scale={scale} w_k={w_k}: LOO={a:.6f} delta={delta:+.6f}")

# Summary
print(f"\n[batch92] COMPLETE")
print(f"  Current best: {best_loo:.6f}")
print(f"  Reproduced:   {auc_final:.6f}")
