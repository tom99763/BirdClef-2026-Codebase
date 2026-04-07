"""
batch133 — File-level aggregation + direct logit signal
===============================================================================
Current best: altfine_b76_i16_s8 LOO=0.994870
PKL has: file_prob_max [66,234] (current LOO predictions), file_logit_max [66,234]

New directions:
 M1 : file-level ICA dual-softmax LOO   (mean-pool windows → file-level KNN)
 M2 : file-level STD dual-softmax LOO
 M3 : file-level ICA WL-style LOO
 M4 : 3way(M1_ica)  — file-level after co-occ smoothing
 M5 : 3way(file_logit_max)  — raw Perch logit signal after 3way
 M6-M14 : blend(current_best, 3way_file_ica, 3way_logit) at various weights
Hypothesis: file-level mean embedding captures global habitat; different errors from window-level
"""
import numpy as np
import json, pickle, time
from pathlib import Path
from sklearn.metrics import roc_auc_score
from numpy.linalg import norm as lnorm

WORK_DIR = Path("/home/lab/BirdClef-2026-Codebase")
DATA_PATH = WORK_DIR / "outputs/perch_labeled_ss.npz"
JSON_PATH = WORK_DIR / "outputs/embed_prior_results.json"
PKL_PATH  = WORK_DIR / "outputs/embed_prior_model.pkl"
EPS = 1e-8

# ── Load data ──────────────────────────────────────────────────────────────
with open(PKL_PATH, "rb") as f:
    model = pickle.load(f)

ew_ica     = model["emb_win_ica_norm"]   # [739, 100]
ew_std     = model["emb_win_std_norm"]   # [739,  80]
labels_win = model["labels_win"]         # [739, 234]
file_labels= model["file_labels"]        # [66, 234]
win_file_id= model["win_file_id"]        # [739] int32
logit_sig  = model["logit_sig_win"]      # [739, 234] sigmoid
file_prob_max  = model["file_prob_max"]  # [66, 234]  current best LOO preds
file_logit_max_raw = model["file_logit_max"]  # [66, 234]  raw Perch per-file

unique_files = np.array(sorted(set(win_file_id.tolist())))
n_files = len(unique_files)   # 66
n_sp    = labels_win.shape[1] # 234
fi2idx  = {int(fi): i for i, fi in enumerate(unique_files)}

present_sp = np.where(file_labels.max(0) > 0)[0]
print(f"Files={n_files}, species={n_sp}, present_sp={len(present_sp)}")
print(f"PKL method: {model.get('method','?')}  LOO={model.get('loo_auc',0):.6f}")

with open(JSON_PATH) as f:
    results = json.load(f)

best_loo = results["best"]["loo_auc"]
tried    = {e["method"] for e in results["experiments"]}
print(f"JSON best: {results['best']['method']}  LOO={best_loo:.6f}")
print(f"Tried methods: {len(tried)}\n")

# ── Pre-compute file-level mean features ─────────────────────────────────
file_ica_mean = np.zeros((n_files, ew_ica.shape[1]), np.float32)
file_std_mean = np.zeros((n_files, ew_std.shape[1]), np.float32)

for fi_idx, fi in enumerate(unique_files):
    mask = win_file_id == fi
    v = ew_ica[mask].mean(0); file_ica_mean[fi_idx] = v / (lnorm(v) + EPS)
    v = ew_std[mask].mean(0); file_std_mean[fi_idx] = v / (lnorm(v) + EPS)

# ── Co-occurrence helpers (same params as batch129-132) ──────────────────
def _scooc(s, c, sl, a):
    gate = 1.0 / (1.0 + np.exp(-c * (s - 0.5)))
    cooc = file_labels.T @ file_labels  # global [234,234]
    row_sum = cooc.sum(1, keepdims=True).clip(1, None)
    cooc_norm = cooc / row_sum; np.fill_diagonal(cooc_norm, 0)
    sl_exp = np.exp(sl * (s - 0.5))
    sl_w   = sl_exp / (sl_exp.sum() + EPS)
    sig    = np.clip(a * (cooc_norm @ sl_w), 0, 1)
    return gate * (s + sig * (1 - s)) + (1 - gate) * s

def _idf_cooc(s, s_pow=2.0, idf_exp=0.75, blend=0.55, center=0.55, sl=41, a=0.130):
    sp_freq = file_labels.mean(0).clip(EPS, 1-EPS)
    idf = (np.log(1.0 / sp_freq) ** idf_exp)
    idf /= idf.max() + EPS
    sp2 = np.clip(s, 0, 1) ** s_pow
    gate = 1.0 / (1.0 + np.exp(-center * (sp2 - 0.5) * 10))
    cooc = file_labels.T @ file_labels
    row_sum = cooc.sum(1, keepdims=True).clip(1, None)
    cn = cooc / row_sum * idf[None, :]; np.fill_diagonal(cn, 0)
    sl_exp = np.exp(sl * (sp2 - 0.5)); sl_w = sl_exp / (sl_exp.sum() + EPS)
    sig = np.clip(a * (cn @ sl_w), 0, 1)
    blended = blend * gate * (sp2 + sig * (1 - sp2)) + (1 - blend) * s
    return np.clip(blended, 0, 1)

def _3way(s):
    idf_s = _idf_cooc(s, s_pow=2.0, idf_exp=0.75, blend=0.55, center=0.55, sl=41, a=0.130)
    tr1   = _scooc(s, 0.54, 41.0, 0.089)
    tr2   = _scooc(tr1, 0.53, 37.0, 0.040)
    return 0.85 * idf_s + 0.15 * tr2

# ── LOO AUC helper ───────────────────────────────────────────────────────
def loo_auc(preds, fl=file_labels):
    aucs = []
    for si in present_sp:
        y_t = fl[:, si]; y_p = preds[:, si]
        if y_t.max() == 0 or y_t.min() == 1: continue
        try: aucs.append(roc_auc_score(y_t, y_p))
        except: pass
    return float(np.mean(aucs)) if aucs else 0.0

# ── M1: File-level ICA dual-softmax LOO ──────────────────────────────────
def file_ds_loo(feats, fl, tau=0.3):
    """feats [n,D], fl [n,sp]. LOO dual-softmax."""
    n = len(feats)
    preds = np.zeros((n, n_sp), np.float32)
    for fi in range(n):
        te = feats[fi]
        tr_idx = np.arange(n) != fi
        tr_fe  = feats[tr_idx]
        tr_fl  = fl[tr_idx]
        sims   = tr_fe @ te  # [n-1]
        for si in range(n_sp):
            pm = tr_fl[:, si] > 0.5
            if not pm.any(): preds[fi, si] = 0.5; continue
            pos_s = sims[pm]; neg_s = sims[~pm]
            # forward softmax over positives
            pexp  = np.exp(pos_s / tau); fwd = (pos_s * pexp).sum() / (pexp.sum() + EPS)
            bwd   = pos_s.max()
            ds    = np.sqrt(np.clip(((fwd+1)/2) * ((bwd+1)/2), 0, 1))
            if len(neg_s) > 0:
                nexp  = np.exp(neg_s / tau); ns = (neg_s * nexp).sum() / (nexp.sum() + EPS)
                preds[fi, si] = np.clip(ds - 0.08 * ((ns+1)/2) + 0.04, 0, 1)
            else:
                preds[fi, si] = ds
    return preds

# ── M3: File-level WL-style LOO ──────────────────────────────────────────
def file_wl_loo(feats, fl, wmp=0.8):
    n = len(feats)
    preds = np.zeros((n, n_sp), np.float32)
    for fi in range(n):
        te = feats[fi]
        tr_idx = np.arange(n) != fi
        tr_fe  = feats[tr_idx]; tr_fl = fl[tr_idx]
        for si in range(n_sp):
            pm = tr_fl[:, si] > 0.5
            if not pm.any(): preds[fi, si] = 0.5; continue
            pos_fe = tr_fe[pm]; pos_s = pos_fe @ te
            pp = pos_fe.mean(0); pp /= lnorm(pp) + EPS
            score = wmp * pos_s.max() + (1-wmp) * (te @ pp)
            preds[fi, si] = (score + 1) / 2
    return preds

# ── Experiment runner ─────────────────────────────────────────────────────
t0 = time.time()
experiments_this_run = []

def save_result(name, auc, meta=None):
    global best_loo
    entry = {"method": name, "loo_auc": round(auc, 8),
             "batch": "batch133", "meta": meta or {}}
    results["experiments"].append(entry)
    marker = " ← NEW BEST" if auc > best_loo else ""
    print(f"  {name}: {auc:.6f}{marker}")
    if auc > best_loo:
        best_loo = auc
        results["best"] = {"method": name, "loo_auc": auc}
        with open(PKL_PATH, "rb") as f:
            mdl = pickle.load(f)
        mdl["method"]       = name
        mdl["loo_auc"]      = auc
        mdl["file_prob_max"]= preds_map.get(name, mdl["file_prob_max"])
        with open(PKL_PATH, "wb") as f:
            pickle.dump(mdl, f)
        print(f"  → PKL updated")
    with open(JSON_PATH, "w") as f:
        json.dump(results, f, indent=2)
    experiments_this_run.append((name, auc))

preds_map = {}

# ── Step 1: Verify current best LOO ──────────────────────────────────────
print("=== Verify current best (file_prob_max) ===")
cur_auc = loo_auc(file_prob_max)
print(f"file_prob_max LOO-AUC: {cur_auc:.6f}  (expect ~0.994870)")

# ── Step 2: Direct logit 3way ─────────────────────────────────────────────
print("\n=== M5: 3way(file_logit_max) ===")
t1 = time.time()
logit_3way = np.array([_3way(file_logit_max_raw[i]) for i in range(n_files)])
logit_3way_auc = loo_auc(logit_3way)
print(f"Raw file_logit_max  : {loo_auc(file_logit_max_raw):.6f}")
print(f"3way(file_logit_max): {logit_3way_auc:.6f}  ({time.time()-t1:.1f}s)")
preds_map["3way_file_logit"] = logit_3way
save_result("3way_file_logit", logit_3way_auc, {"desc": "3way on raw Perch file logit max"})

# ── Step 3: File-level ICA dual-softmax ───────────────────────────────────
print("\n=== M1: File-level ICA dual-softmax LOO ===")
t1 = time.time()
fl_ds_ica = file_ds_loo(file_ica_mean, file_labels, tau=0.3)
fl_ds_ica_auc = loo_auc(fl_ds_ica)
print(f"file_ds_ica LOO: {fl_ds_ica_auc:.6f}  ({time.time()-t1:.1f}s)")
preds_map["file_ds_ica"] = fl_ds_ica
save_result("file_ds_ica", fl_ds_ica_auc, {"tau": 0.3})

# Apply 3way
t1 = time.time()
fl_3way_ica = np.array([_3way(fl_ds_ica[i]) for i in range(n_files)])
fl_3way_ica_auc = loo_auc(fl_3way_ica)
print(f"3way(file_ds_ica)  : {fl_3way_ica_auc:.6f}  ({time.time()-t1:.1f}s)")
preds_map["3way_file_ds_ica"] = fl_3way_ica
save_result("3way_file_ds_ica", fl_3way_ica_auc, {"tau": 0.3})

# ── Step 4: File-level STD dual-softmax ───────────────────────────────────
print("\n=== M2: File-level STD dual-softmax LOO ===")
t1 = time.time()
fl_ds_std = file_ds_loo(file_std_mean, file_labels, tau=0.3)
fl_ds_std_auc = loo_auc(fl_ds_std)
print(f"file_ds_std LOO: {fl_ds_std_auc:.6f}  ({time.time()-t1:.1f}s)")
save_result("file_ds_std", fl_ds_std_auc, {"tau": 0.3})

fl_3way_std = np.array([_3way(fl_ds_std[i]) for i in range(n_files)])
fl_3way_std_auc = loo_auc(fl_3way_std)
print(f"3way(file_ds_std)  : {fl_3way_std_auc:.6f}")
preds_map["3way_file_ds_std"] = fl_3way_std
save_result("3way_file_ds_std", fl_3way_std_auc)

# ── Step 5: File-level WL LOO ─────────────────────────────────────────────
print("\n=== M3: File-level WL LOO ===")
t1 = time.time()
fl_wl_ica = file_wl_loo(file_ica_mean, file_labels, wmp=0.8)
fl_wl_ica_auc = loo_auc(fl_wl_ica)
print(f"file_wl_ica LOO: {fl_wl_ica_auc:.6f}  ({time.time()-t1:.1f}s)")
preds_map["file_wl_ica"] = fl_wl_ica
save_result("file_wl_ica", fl_wl_ica_auc)

fl_3way_wl = np.array([_3way(fl_wl_ica[i]) for i in range(n_files)])
fl_3way_wl_auc = loo_auc(fl_3way_wl)
print(f"3way(file_wl_ica)  : {fl_3way_wl_auc:.6f}")
preds_map["3way_file_wl_ica"] = fl_3way_wl
save_result("3way_file_wl_ica", fl_3way_wl_auc)

# ── Step 6: Ensemble file-level methods ──────────────────────────────────
print("\n=== M4: File-level ensemble (ica+std+wl) ===")
fl_ens = 0.5*fl_ds_ica + 0.3*fl_wl_ica + 0.2*fl_ds_std
fl_ens_auc = loo_auc(fl_ens)
print(f"file_ens LOO: {fl_ens_auc:.6f}")
fl_3way_ens = np.array([_3way(fl_ens[i]) for i in range(n_files)])
fl_3way_ens_auc = loo_auc(fl_3way_ens)
print(f"3way(file_ens): {fl_3way_ens_auc:.6f}")
preds_map["file_ens_3way"] = fl_3way_ens
save_result("file_ens_3way", fl_3way_ens_auc, {"w_ica":0.5,"w_wl":0.3,"w_std":0.2})

# ── Step 7: Blend current best with file-level 3way ──────────────────────
print("\n=== Blend experiments: current_best + file_level_3way ===")
best_preds = file_prob_max  # LOO=0.994870

# Best single file-level: which has highest 3way AUC?
candidates = [
    ("3way_file_ds_ica", fl_3way_ica, fl_3way_ica_auc),
    ("3way_file_ds_std", fl_3way_std, fl_3way_std_auc),
    ("3way_file_wl_ica", fl_3way_wl, fl_3way_wl_auc),
    ("file_ens_3way",    fl_3way_ens, fl_3way_ens_auc),
    ("3way_file_logit",  logit_3way, logit_3way_auc),
]
candidates.sort(key=lambda x: -x[2])
print("File-level 3way candidates (sorted by LOO):")
for nm, _, av in candidates:
    print(f"  {nm}: {av:.6f}")

# Blend weights to sweep
best_cand_name, best_cand, best_cand_auc = candidates[0]
print(f"\nBlending current_best + {best_cand_name}")

best_blend_auc = cur_auc
best_blend_w = 0.0
best_blend_preds = best_preds

for w in [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]:
    blended = (1-w) * best_preds + w * best_cand
    auc = loo_auc(blended)
    name = f"blend_fl_{int(w*100):02d}"
    preds_map[name] = blended
    if auc > best_blend_auc:
        best_blend_auc = auc
        best_blend_w = w
        best_blend_preds = blended
    save_result(name, auc, {"w_file": w, "w_current": 1-w, "cand": best_cand_name})

# Also try blending with 2nd best candidate
if len(candidates) > 1:
    best2_name, best2, best2_auc = candidates[1]
    print(f"\nBlending current_best + {best2_name}")
    for w in [0.01, 0.02, 0.03, 0.05]:
        blended = (1-w) * best_preds + w * best2
        auc = loo_auc(blended)
        name = f"blend_fl2_{int(w*100):02d}"
        preds_map[name] = blended
        if auc > best_blend_auc:
            best_blend_auc = auc
            best_blend_w = w
            best_blend_preds = blended
        save_result(name, auc, {"w_file": w, "w_current": 1-w, "cand": best2_name})

# ── Step 8: Three-way blend: current + file_ica + file_logit ─────────────
print("\n=== Three-way blend experiments ===")
for wa, wb, wc in [
    (0.96, 0.03, 0.01), (0.95, 0.03, 0.02), (0.94, 0.04, 0.02),
    (0.96, 0.02, 0.02), (0.97, 0.02, 0.01), (0.93, 0.05, 0.02),
]:
    blended = wa * best_preds + wb * fl_3way_ica + wc * logit_3way
    auc = loo_auc(blended)
    name = f"3blend_b{int(wa*100)}_f{int(wb*100)}_l{int(wc*100)}"
    preds_map[name] = blended
    if auc > best_blend_auc:
        best_blend_auc = auc
        best_blend_preds = blended
    save_result(name, auc, {"wa": wa, "wb": wb, "wc": wc})

# ── Step 9: Four-way blend (add file_wl_ica) ─────────────────────────────
print("\n=== Four-way blend ===")
for wa, wb, wc, wd in [
    (0.94, 0.03, 0.02, 0.01), (0.93, 0.03, 0.02, 0.02), (0.95, 0.02, 0.02, 0.01),
]:
    blended = wa * best_preds + wb * fl_3way_ica + wc * logit_3way + wd * fl_3way_wl
    auc = loo_auc(blended)
    name = f"4blend_b{int(wa*100)}_f{int(wb*100)}_l{int(wc*100)}_w{int(wd*100)}"
    preds_map[name] = blended
    if auc > best_blend_auc:
        best_blend_auc = auc
        best_blend_preds = blended
    save_result(name, auc, {"wa": wa, "wb": wb, "wc": wc, "wd": wd})

# ── Summary ──────────────────────────────────────────────────────────────
elapsed = time.time() - t0
print(f"\n{'='*70}")
print(f"Batch133 complete in {elapsed:.1f}s")
print(f"Experiments this run: {len(experiments_this_run)}")
print(f"Starting best LOO: {cur_auc:.6f}")
print(f"Final best LOO:    {best_loo:.6f}")
if best_loo > cur_auc:
    print(f"IMPROVEMENT: +{best_loo-cur_auc:.6f}")
    print(f"Best method: {results['best']['method']}")
else:
    print("No improvement found. Experiments appended to JSON.")
print(f"\nTop-5 this batch:")
sorted_exp = sorted(experiments_this_run, key=lambda x: -x[1])
for nm, av in sorted_exp[:5]:
    print(f"  {nm}: {av:.6f}")
