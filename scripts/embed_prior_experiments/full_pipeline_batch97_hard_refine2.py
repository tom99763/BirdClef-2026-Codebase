"""
Batch 97 — Fisher Hard KDE Refinement Round 2
===============================================
Current best: fhard_k40_bw7_w03 LOO=0.992139

Batch96 showed k=40, bw=0.07 beats k=30, bw=0.06.
Also: stacking fh35 on best_ref w=0.02/0.03 gives 0.992126.

1. k_sweep_bw7   — k sweep around 40 with bw=0.07: k=33,35,37,40,42,45,48,50
2. w_sweep_k40bw7 — w sweep for k=40, bw=0.07: w=0.01..0.08
3. bw_sweep_k40  — bw sweep for k=40: bw=0.060,0.065,0.070,0.075,0.080
4. stack_fh35    — Fine-tune stack_fh35: k=33,35,37 x w=0.01,0.02,0.03 on current best
5. stack_fh40bw7 — Stack fh40bw7 on current best (self-stack)
6. combine_fh30_fh40 — Combine fh30(bw=0.06) + fh40(bw=0.07) in single blend
7. k2d_bw7       — 2D grid around k=40, bw=0.07: k in {36,38,40,42} x w in {0.02,0.03,0.04}
8. stack_multilevel — 3-level stack: fin_ref → fh30_bw6 → fh40_bw7
"""
import numpy as np
import json
import pickle
import copy
import time
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA as SklearnPCA
from numpy.linalg import norm

ROOT = Path("/home/lab/BirdClef-2026-Codebase")
RESULTS_PATH = ROOT / "outputs" / "embed_prior_results.json"
MODEL_PATH   = ROOT / "outputs" / "embed_prior_model.pkl"

DATA = np.load(ROOT / "outputs" / "perch_labeled_ss.npz")
labels_win  = DATA["labels"].astype(np.float32)
logit_win   = DATA["logits"].astype(np.float32)
n_windows   = DATA["n_windows"]
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

ew_ica  = ep["emb_win_ica_norm"]
ew_pca  = ep["emb_win_pca_norm"]
ew_std  = ep["emb_win_std_norm"]
ew_nmf  = ep["emb_win_nmf_norm"]
file_labels = ep["file_labels"]
cfg = ep["config"]

print(f"[batch97] ICA{ew_ica.shape}", flush=True)
res = json.load(open(RESULTS_PATH))
best = res["best"]
best_loo = best["loo_auc"]
print(f"[batch97] Current best: {best['method']} LOO={best_loo:.6f}", flush=True)

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

def make_lp(T):
    sig = 1.0 / (1.0 + np.exp(np.clip(-logit_win / T, -88, 88)))
    return np.stack([sig[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)])

def make_sp(T):
    sm_r = logit_win / T; sm_r -= sm_r.max(1, keepdims=True)
    sm_e = np.exp(sm_r); smp = sm_e / (sm_e.sum(1, keepdims=True) + EPS)
    return np.stack([smp[file_start[fi]:file_end[fi]].max(0) for fi in range(n_files)])

def compute_subspace(ew_sp, n_comp=2, wma_ss=0.92):
    ss = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew_sp[win_file_id == fi]; tr = ew_sp[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32); dim = te.shape[1]
        for si in range(n_species):
            pm = tl[:, si] > 0.5
            if not pm.any(): ws[:, si] = 0.5; continue
            pos = tr[pm]; k = min(n_comp, len(pos) - 1, dim - 1)
            if k < 1:
                pp = pos.mean(0); pp /= norm(pp) + EPS
                ws[:, si] = np.clip((te @ pp + 1) / 2, 0, 1); continue
            try:
                p = SklearnPCA(n_components=k); p.fit(pos)
                te_r = p.inverse_transform(p.transform(te))
                err = norm(te - te_r, axis=1)
                ws[:, si] = np.clip(1 - err / (norm(te, axis=1) + EPS), 0, 1)
            except:
                ws[:, si] = 0.5
        ss[fi] = wma_ss * ws.max(0) + (1 - wma_ss) * ws.mean(0)
    return ss

def proto_kde_loo(ew, bw=0.08):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        sims = te @ tr.T; ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pos_idx = np.where(tl[:, si] > 0.5)[0]
            if len(pos_idx) == 0: ws[:, si] = 0.5; continue
            pw = tr[pos_idx]; centroid = pw.mean(0); centroid /= (norm(centroid) + EPS)
            proto_w = np.clip(pw @ centroid, 0, None); proto_w /= (proto_w.sum() + EPS)
            kern = np.exp((sims[:, pos_idx] - 1.0) / (bw**2 + EPS))
            ws[:, si] = np.clip((kern * proto_w[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

def fisher_kde_loo(ew, bw=0.06):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            pos_wins = tr[pm]; neg_wins = tr[nm] if nm.any() else tr[~pm]
            mu_p = pos_wins.mean(0); mu_n = neg_wins.mean(0)
            var_p = pos_wins.var(0) + EPS; var_n = neg_wins.var(0) + EPS
            fisher = np.sqrt(np.clip((mu_p - mu_n)**2 / (var_p + var_n), 0, None))
            w_dim = fisher / (norm(fisher) + EPS)
            te_w = te * w_dim[None, :]; te_w /= norm(te_w, axis=1, keepdims=True) + EPS
            tr_w = pos_wins * w_dim[None, :]; tr_w /= norm(tr_w, axis=1, keepdims=True) + EPS
            centroid = tr_w.mean(0); centroid /= norm(centroid) + EPS
            proto_w = np.clip(tr_w @ centroid, 0, None); proto_w /= (proto_w.sum() + EPS)
            sims_w = te_w @ tr_w.T
            kern = np.exp((sims_w - 1.0) / (bw**2 + EPS))
            ws[:, si] = np.clip((kern * proto_w[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

def fisher_hard_kde_loo(ew, bw=0.06, top_k=30):
    out = np.zeros((n_files, n_species), np.float32)
    for fi in range(n_files):
        te = ew[win_file_id == fi]; tr = ew[win_file_id != fi]
        tl = labels_win[win_file_id != fi]
        ws = np.zeros((len(te), n_species), np.float32)
        for si in range(n_species):
            pm = tl[:, si] > 0.5; nm = tl[:, si] < 0.1
            if not pm.any(): ws[:, si] = 0.5; continue
            pos_wins = tr[pm]; neg_wins = tr[nm] if nm.any() else tr[~pm]
            mu_p = pos_wins.mean(0); mu_n = neg_wins.mean(0)
            var_p = pos_wins.var(0) + EPS; var_n = neg_wins.var(0) + EPS
            fisher_raw = np.sqrt(np.clip((mu_p - mu_n)**2 / (var_p + var_n), 0, None))
            top_idx = np.argsort(-fisher_raw)[:top_k]
            w_dim = np.zeros(len(fisher_raw), np.float32)
            w_dim[top_idx] = 1.0 / np.sqrt(float(top_k))
            te_w = te * w_dim[None, :]; te_w /= norm(te_w, axis=1, keepdims=True) + EPS
            tr_w = pos_wins * w_dim[None, :]; tr_w /= norm(tr_w, axis=1, keepdims=True) + EPS
            centroid = tr_w.mean(0); centroid /= norm(centroid) + EPS
            proto_w = np.clip(tr_w @ centroid, 0, None); proto_w /= (proto_w.sum() + EPS)
            sims_w = te_w @ tr_w.T
            kern = np.exp((sims_w - 1.0) / (bw**2 + EPS))
            ws[:, si] = np.clip((kern * proto_w[None, :]).sum(1), 0, None)
        for si in range(n_species):
            mx = ws[:, si].max()
            if mx > EPS: ws[:, si] /= mx
        out[fi] = ws.max(0)
    return out

# ── Pre-compute base ──────────────────────────────────────────────────────────
print("Pre-computing base...", flush=True)
t0 = time.time()
s_ica = wl_loo(ew_ica, cfg["ica100"]["k_neg"], cfg["ica100"]["w_max_pos"], cfg["ica100"]["w_max_agg"])
s_std = wl_loo(ew_std, cfg["std_pca80"]["k_neg"], cfg["std_pca80"]["w_max_pos"], cfg["std_pca80"]["w_max_agg"])
s_pca = wl_loo(ew_pca, cfg["pca80"]["k_neg"], cfg["pca80"]["w_max_pos"], cfg["pca80"]["w_max_agg"])
s_nmf = wl_loo(ew_nmf, cfg["nmf"]["k_neg"], cfg["nmf"]["w_max_pos"], cfg["nmf"]["w_max_agg"])
uh_b  = cfg["w_ica100"] * s_ica + cfg["w_std"] * s_std + cfg["w_pca80"] * s_pca
uh_nmf = cfg["nmf"]["uh_scale"] * uh_b + cfg["nmf"]["w_nmf"] * s_nmf
pT8   = make_lp(cfg["logit_temperature"]); pmt = (pT8 + make_lp(10.0)) / 2
sm6   = make_sp(cfg["softmax_temp"]); ss2 = compute_subspace(ew_pca, n_comp=2, wma_ss=0.92)
kde08 = proto_kde_loo(ew_ica, bw=0.08)
print(f"  done ({time.time()-t0:.0f}s)", flush=True)

w_uh = 1 - cfg["w_logit"] - cfg["w_multit"] - cfg["w_subspace"] - cfg["w_softmax"]
base_cur  = w_uh * uh_nmf + cfg["w_logit"] * pT8 + cfg["w_multit"] * pmt + cfg["w_subspace"] * ss2 + cfg["w_softmax"] * sm6
final_ref = 0.96 * base_cur + 0.04 * kde08

# Reference chain
f06 = fisher_kde_loo(ew_ica, bw=0.06)
fin_ref = (1 - 0.05) * final_ref + 0.05 * f06
fh30 = fisher_hard_kde_loo(ew_ica, bw=0.06, top_k=30)
best_ref = (1 - 0.03) * fin_ref + 0.03 * fh30  # batch95 best = 0.992112
auc_bestref = macro_auc(best_ref)
print(f"  best_ref (k30 bw6 w3): {auc_bestref:.6f}", flush=True)

fh40_bw7 = fisher_hard_kde_loo(ew_ica, bw=0.07, top_k=40)
cur_best = (1 - 0.03) * fin_ref + 0.03 * fh40_bw7
auc_cur = macro_auc(cur_best)
print(f"  current best (k40 bw7 w3): {auc_cur:.6f} (expected 0.992139)", flush=True)

results = {}
new_best_loo = best_loo
new_best_method = None

def reg(name, auc):
    delta = auc - best_loo
    mark = ""
    if auc > best_loo:
        mark = f" *** NEW BEST ***"
    elif auc > best_loo - 0.0003:
        mark = " (near-best)"
    print(f"  {name}: {auc:.6f} {delta:+.6f}{mark}", flush=True)
    results[name] = {"loo_auc": auc, "method": name, "batch": "batch97"}
    return auc

# ─────────────────────────────────────────────────────────────────────────────
# EXP1: k sweep around 40 with bw=0.07 (on fin_ref, w=0.03)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP1] k sweep with bw=0.07 (on fin_ref, w=0.03)...", flush=True)
t1 = time.time()
for k in [33, 35, 37, 40, 42, 45, 48, 50, 55, 60]:
    fh = fisher_hard_kde_loo(ew_ica, bw=0.07, top_k=k)
    s = (1 - 0.03) * fin_ref + 0.03 * fh
    reg(f"fh_bw7_k{k:02d}_w03", macro_auc(s))
print(f"  EXP1 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP2: w sweep for k=40, bw=0.07
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP2] w sweep for k=40, bw=0.07...", flush=True)
t1 = time.time()
fh40_bw7 = fisher_hard_kde_loo(ew_ica, bw=0.07, top_k=40)
for w_int in range(1, 9):
    w = w_int * 0.01
    s = (1 - w) * fin_ref + w * fh40_bw7
    reg(f"fh40bw7_w{w_int:02d}", macro_auc(s))
print(f"  EXP2 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP3: bw fine sweep for k=40: bw=0.060 to 0.080
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP3] bw fine sweep for k=40, w=0.03...", flush=True)
t1 = time.time()
for bw_x10 in [60, 62, 64, 65, 66, 68, 70, 72, 74, 75, 76, 78, 80]:
    bw = bw_x10 * 0.001
    fh = fisher_hard_kde_loo(ew_ica, bw=bw, top_k=40)
    s = (1 - 0.03) * fin_ref + 0.03 * fh
    reg(f"fh40_bw{bw_x10}_w03", macro_auc(s))
print(f"  EXP3 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP4: 2D fine grid k x w for bw=0.07
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP4] 2D k x w grid for bw=0.07...", flush=True)
t1 = time.time()
fh_cache = {}
for k in [35, 37, 40, 42, 45]:
    fh_cache[k] = fisher_hard_kde_loo(ew_ica, bw=0.07, top_k=k)
for k in [35, 37, 40, 42, 45]:
    for w_int in [2, 3, 4, 5]:
        w = w_int * 0.01
        s = (1 - w) * fin_ref + w * fh_cache[k]
        reg(f"fh_bw7_k{k}_w{w_int:02d}", macro_auc(s))
print(f"  EXP4 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP5: Stack fh35 on current best (0.992139)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP5] Stack variations on current best (fh40_bw7)...", flush=True)
t1 = time.time()
fh40_bw7 = fisher_hard_kde_loo(ew_ica, bw=0.07, top_k=40)
cur_best_sig = (1 - 0.03) * fin_ref + 0.03 * fh40_bw7
for k, bw in [(35, 0.06), (35, 0.07), (40, 0.06), (40, 0.07), (30, 0.06)]:
    fhk = fisher_hard_kde_loo(ew_ica, bw=bw, top_k=k)
    for w_int in [1, 2, 3]:
        w = w_int * 0.01
        s = (1 - w) * cur_best_sig + w * fhk
        reg(f"stack_k{k}bw{int(bw*100)}_w{w_int:02d}", macro_auc(s))
print(f"  EXP5 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP6: Combine fh30(bw=0.06) + fh40(bw=0.07) jointly
# final = (1-w1-w2)*fin_ref + w1*fh30_bw6 + w2*fh40_bw7
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP6] Joint blend of fh30_bw6 + fh40_bw7...", flush=True)
t1 = time.time()
fh30_bw6 = fh30  # already computed
fh40_bw7 = fisher_hard_kde_loo(ew_ica, bw=0.07, top_k=40)
for w1_int, w2_int in [(2, 2), (2, 3), (3, 2), (3, 3), (2, 1), (1, 2), (1, 3), (3, 1)]:
    w1, w2 = w1_int * 0.01, w2_int * 0.01
    s = (1 - w1 - w2) * fin_ref + w1 * fh30_bw6 + w2 * fh40_bw7
    reg(f"joint_fh30w{w1_int}_fh40w{w2_int}", macro_auc(s))
print(f"  EXP6 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP7: 3-level stack: fin_ref → fh30_bw6 → fh40_bw7
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP7] 3-level stack...", flush=True)
t1 = time.time()
fh30_bw6 = fh30
fh40_bw7 = fisher_hard_kde_loo(ew_ica, bw=0.07, top_k=40)
for w1, w2 in [(0.03, 0.03), (0.03, 0.02), (0.02, 0.03), (0.04, 0.02), (0.02, 0.04)]:
    lv1 = (1 - w1) * fin_ref + w1 * fh30_bw6
    lv2 = (1 - w2) * lv1 + w2 * fh40_bw7
    reg(f"threelv_w1{int(w1*100):02d}_w2{int(w2*100):02d}", macro_auc(lv2))
print(f"  EXP7 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXP8: Add soft Fisher f06 as an extra layer on top of fh40_bw7 result
# ─────────────────────────────────────────────────────────────────────────────
print("\n[EXP8] Add soft Fisher on top of fh40_bw7...", flush=True)
t1 = time.time()
fh40_bw7 = fisher_hard_kde_loo(ew_ica, bw=0.07, top_k=40)
cur_best_sig = (1 - 0.03) * fin_ref + 0.03 * fh40_bw7
for f_bw, f_tag in [(0.05, "f05"), (0.06, "f06"), (0.07, "f07"), (0.08, "f08")]:
    fsoft = fisher_kde_loo(ew_ica, bw=f_bw)
    for w_int in [1, 2, 3, 5]:
        w = w_int * 0.01
        s = (1 - w) * cur_best_sig + w * fsoft
        reg(f"fh40_plus_soft{f_tag}_w{w_int:02d}", macro_auc(s))
print(f"  EXP8 done ({time.time()-t1:.0f}s)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60, flush=True)
print("[batch97] SUMMARY", flush=True)
print(f"  Previous best: {best_loo:.6f} ({best['method']})", flush=True)

if results:
    all_sorted = sorted(results.items(), key=lambda x: -x[1]["loo_auc"])
    top_tag, top_v = all_sorted[0]
    top_auc = top_v["loo_auc"]
    if top_auc > best_loo:
        print(f"  *** NEW BEST: {top_tag} LOO={top_auc:.6f} (+{top_auc-best_loo:.6f}) ***", flush=True)
        new_best_loo = top_auc
        new_best_method = top_tag
    else:
        print(f"  Best this batch: {top_auc:.6f} ({top_tag}) — no improvement", flush=True)
    print("\n  Top-10:", flush=True)
    for tag, v in all_sorted[:10]:
        d = v["loo_auc"] - best_loo
        print(f"    {tag}: {v['loo_auc']:.6f} ({d:+.6f})", flush=True)

# ── Save ─────────────────────────────────────────────────────────────────────
res2 = json.load(open(RESULTS_PATH))
if isinstance(res2.get("experiments"), list):
    res2["experiments"].extend(list(results.values()))
elif isinstance(res2.get("experiments"), dict):
    res2["experiments"].update(results)
else:
    res2["experiments"] = list(results.values())

if new_best_loo > best_loo:
    res2["best"] = {"method": new_best_method, "loo_auc": new_best_loo, "batch": "batch97"}
    ep2 = copy.deepcopy(ep)
    ep2["loo_auc"] = new_best_loo
    ep2["method"] = new_best_method
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(ep2, f)
    print(f"\n  SAVED new best: {new_best_method} LOO={new_best_loo:.6f}", flush=True)

json.dump(res2, open(RESULTS_PATH, "w"), indent=2)
print(f"\nSaved {len(results)} experiments to JSON.", flush=True)
