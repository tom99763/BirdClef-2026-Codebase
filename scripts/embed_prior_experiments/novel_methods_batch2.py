"""
Deep fine-tuning around the new best: ls_geo0.60_win0.40_a0.75_b1.40 (0.9148)
Explore:
  - w_geo variations: 0.50~0.75
  - a variations: 0.65~0.90
  - b variations: 1.2~2.0
  - k_win: 1, 2, 3
  - k_geo: 3, 4, 5
  - T_geo: 0.15, 0.18, 0.20, 0.25
  - 3-way logspace with win_k1 and geo_k4
"""
import numpy as np, pickle, json, os, warnings
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')

os.chdir("/home/lab/BirdClef-2026-Codebase")

perch = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win    = perch['emb'].astype(np.float32)
logits_win = perch['logits'].astype(np.float32)
labels_win = perch['labels'].astype(np.float32)
file_list  = list(perch['file_list'])
n_windows  = perch['n_windows']
n_files    = len(file_list)
n_species  = labels_win.shape[1]
file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end   = np.cumsum(n_windows).astype(np.int32)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

file_embs      = np.zeros((n_files, emb_win.shape[1]), np.float32)
file_labels    = np.zeros((n_files, n_species), np.float32)
file_logit_max = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_embs[fi]      = emb_win[s:e].mean(0)
    file_labels[fi]    = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi] = logits_win[s:e].max(0)
file_embs_norm = normalize(file_embs, norm='l2').astype(np.float32)
emb_win_norm   = normalize(emb_win, norm='l2').astype(np.float32)

win_file_id = np.zeros(len(emb_win), np.int32)
for fi in range(n_files):
    win_file_id[int(file_start[fi]):int(file_end[fi])] = fi

RESULTS_PATH = "outputs/embed_prior_results.json"
with open(RESULTS_PATH) as f:
    results_db = json.load(f)

tried_methods = set(e['method'] for e in results_db.get('experiments', []))
best_auc = results_db.get('best', {}).get('loo_auc', 0.0)
print(f"Current best: {best_auc:.6f}")

with open("outputs/embed_prior_attn.pkl", "rb") as f:
    pkl_attn = pickle.load(f)
X_pkl = pkl_attn['X_combined_n'].astype(np.float32)
fl_pkl = pkl_attn['file_labels'].astype(np.float32)

def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

EPS = 1e-7

def append_result(name, auc, **kw):
    results_db['experiments'].append({'method': name, 'loo_auc': round(auc, 6), **kw})
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results_db, f, indent=2)

def maybe_update_best(name, auc, **kw):
    global best_auc
    if auc > best_auc:
        results_db['best'] = {'method': name, 'loo_auc': auc, **kw}
        best_auc = auc
        with open(RESULTS_PATH, 'w') as f:
            json.dump(results_db, f, indent=2)
        print(f"  *** NEW BEST: {name} = {auc:.4f} ***")
        return True
    return False

def attn_knn_loo_pkl(k=4, T=0.2):
    preds = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        tr = np.array([j for j in range(n_files) if j != i])
        sims = (X_pkl[[i]] @ X_pkl[tr].T).ravel()
        top  = np.argsort(-sims)[:k]
        logit = sims[top] / T; logit -= logit.max()
        w = np.exp(logit); w /= w.sum()
        preds[i] = (w[:, None] * fl_pkl[tr[top]]).sum(0)
    return preds

def window_knn_loo(k=1):
    preds = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        te_s, te_e = int(file_start[i]), int(file_end[i])
        X_te = emb_win_norm[te_s:te_e]
        tr_mask = win_file_id != i
        X_tr = emb_win_norm[tr_mask]
        tr_fi = win_file_id[tr_mask]
        sims = X_te @ X_tr.T
        top_idx = np.argsort(-sims, 1)[:, :k]
        wp = np.zeros((te_e - te_s, n_species), np.float32)
        for wi in range(te_e - te_s):
            w = sims[wi, top_idx[wi]].clip(0); ws = w.sum()
            w = w/ws if ws > 1e-8 else np.ones(k)/k
            wp[wi] = (w[:, None] * file_labels[tr_fi[top_idx[wi]]]).sum(0)
        preds[i] = wp.mean(0)
    return preds

# Pre-compute all needed predictions
print("Computing KNN predictions...", flush=True)
y_geo_k3  = attn_knn_loo_pkl(k=3, T=0.2)
y_geo_k4  = attn_knn_loo_pkl(k=4, T=0.2)
y_geo_k5  = attn_knn_loo_pkl(k=5, T=0.2)
y_geo_k4_T015 = attn_knn_loo_pkl(k=4, T=0.15)
y_geo_k4_T018 = attn_knn_loo_pkl(k=4, T=0.18)
y_geo_k4_T025 = attn_knn_loo_pkl(k=4, T=0.25)
y_win_k1  = window_knn_loo(k=1)
y_win_k2  = window_knn_loo(k=2)
y_win_k3  = window_knn_loo(k=3)

geo_variants = {
    'geo_k3_T0.20': y_geo_k3,
    'geo_k4_T0.20': y_geo_k4,
    'geo_k5_T0.20': y_geo_k5,
    'geo_k4_T0.15': y_geo_k4_T015,
    'geo_k4_T0.18': y_geo_k4_T018,
    'geo_k4_T0.25': y_geo_k4_T025,
}
win_variants = {
    'win_k1': y_win_k1,
    'win_k2': y_win_k2,
    'win_k3': y_win_k3,
}
print("Done computing.\n")

# ── A) Ultra-fine sweep around new best ───────────────────────────────────
print("="*60)
print("A) Ultra-fine: w_geo × k_geo × k_win × a × b")
print("="*60)

best_so_far = best_auc
new_bests = []

for geo_name, y_geo in geo_variants.items():
    for win_name, y_win in win_variants.items():
        for w_g in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
            y_blend = w_g * y_geo + (1-w_g) * y_win
            log_p = np.log(y_blend.clip(EPS))
            for a in [0.65, 0.70, 0.72, 0.75, 0.78, 0.80, 0.85, 0.90]:
                for b in [1.2, 1.3, 1.35, 1.4, 1.45, 1.5, 1.6, 1.7, 1.8, 2.0]:
                    name = f"ls2_{geo_name}_{win_name}_wg{w_g:.2f}_a{a:.2f}_b{b:.2f}"
                    if name in tried_methods:
                        continue
                    pred = sigmoid(a * file_logit_max + b * log_p)
                    auc = macro_auc(file_labels, pred)
                    append_result(name, auc, geo=geo_name, win=win_name,
                                  w_g=w_g, a=a, b=b)
                    if maybe_update_best(name, auc, geo=geo_name, win=win_name,
                                         w_g=w_g, a=a, b=b):
                        new_bests.append((auc, name))

print(f"\n  Best in A): {best_auc:.6f}")
if new_bests:
    print(f"  New bests found: {len(new_bests)}")
    for a, n in sorted(new_bests, reverse=True)[:5]:
        print(f"    {a:.4f} {n}")

# ── B) 3-way logspace: a*lmx + b*log(geo) + c*log(win) ─────────────────
print("\n" + "="*60)
print("B) 3-way: a*logit_max + b*log(geo_k4) + c*log(win_k1)")
print("="*60)

log_geo4 = np.log(y_geo_k4.clip(EPS))
log_win1 = np.log(y_win_k1.clip(EPS))
log_win2 = np.log(y_win_k2.clip(EPS))

best_3way = best_auc; best_3way_params = {}
for a in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
    for b in [0.6, 0.8, 1.0, 1.2, 1.4]:
        for c in [0.3, 0.4, 0.5, 0.6, 0.8]:
            for win_log, wn in [(log_win1, 'w1'), (log_win2, 'w2')]:
                name = f"ls3way_geo4_{wn}_a{a:.2f}_b{b:.2f}_c{c:.2f}"
                if name in tried_methods:
                    continue
                pred = sigmoid(a * file_logit_max + b * log_geo4 + c * win_log)
                auc = macro_auc(file_labels, pred)
                append_result(name, auc, a=a, b=b, c=c, win=wn)
                if auc > best_3way:
                    best_3way = auc; best_3way_params = {'a': a, 'b': b, 'c': c, 'name': name}
                maybe_update_best(name, auc, a=a, b=b, c=c, win=wn)

print(f"  Best 3-way: {best_3way_params.get('name', '?')} = {best_3way:.4f}")

# ── C) Logit_max + logit_mean fusion + geo KNN ─────────────────────────────
print("\n" + "="*60)
print("C) Mixed logit (max + mean) + geo blend")
print("="*60)

file_logit_mean = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_logit_mean[fi] = logits_win[s:e].mean(0)

best_C = best_auc
for w_max in [0.5, 0.6, 0.7, 0.8]:
    w_mean = 1 - w_max
    logit_blend = w_max * file_logit_max + w_mean * file_logit_mean
    for w_g in [0.55, 0.60, 0.65]:
        y_blend = w_g * y_geo_k4 + (1-w_g) * y_win_k1
        log_p = np.log(y_blend.clip(EPS))
        for a in [0.65, 0.70, 0.75, 0.80]:
            for b in [1.2, 1.4, 1.5, 1.6]:
                name = f"ls_lmix{w_max:.1f}_{w_g:.2f}_a{a:.2f}_b{b:.2f}"
                if name in tried_methods:
                    continue
                pred = sigmoid(a * logit_blend + b * log_p)
                auc = macro_auc(file_labels, pred)
                append_result(name, auc, w_max=w_max, w_g=w_g, a=a, b=b)
                if auc > best_C: best_C = auc
                maybe_update_best(name, auc, w_max=w_max, w_g=w_g, a=a, b=b)

print(f"  Best from C): {best_C:.4f}")

# ── D) Window geo KNN: window emb + geo features ──────────────────────────
print("\n" + "="*60)
print("D) Logspace with geo k=4 and multiple win variants")
print("="*60)

# geo k4 T0.18 blended with win k1
for y_g, gn in [(y_geo_k4_T018, 'gT018')]:
    for y_w, wn in [(y_win_k1, 'w1')]:
        for wg in [0.55, 0.60, 0.65, 0.70]:
            y_b = wg * y_g + (1-wg) * y_w
            lp = np.log(y_b.clip(EPS))
            for a in [0.70, 0.75, 0.80]:
                for b in [1.3, 1.4, 1.5, 1.6]:
                    name = f"ls_{gn}_{wn}_wg{wg:.2f}_a{a:.2f}_b{b:.2f}"
                    if name in tried_methods: continue
                    pred = sigmoid(a * file_logit_max + b * lp)
                    auc = macro_auc(file_labels, pred)
                    append_result(name, auc, a=a, b=b)
                    maybe_update_best(name, auc, a=a, b=b)

print(f"  Current best: {best_auc:.6f}")

# ── E) Logspace + 4-way ensemble ──────────────────────────────────────────
print("\n" + "="*60)
print("E) 4-way: geo_k4 + geo_k5 + win_k1 + win_k2")
print("="*60)

best_E = best_auc
for wg4 in [0.30, 0.35, 0.40]:
    for wg5 in [0.20, 0.25, 0.30]:
        for ww1 in [0.20, 0.25, 0.30]:
            ww2 = 1 - wg4 - wg5 - ww1
            if ww2 < 0.05 or ww2 > 0.40: continue
            y_4 = wg4*y_geo_k4 + wg5*y_geo_k5 + ww1*y_win_k1 + ww2*y_win_k2
            lp = np.log(y_4.clip(EPS))
            for a in [0.65, 0.70, 0.75, 0.80]:
                for b in [1.3, 1.4, 1.5, 1.6]:
                    name = f"ls4way_g4_{wg4:.2f}g5_{wg5:.2f}w1_{ww1:.2f}w2_{ww2:.2f}_a{a:.2f}_b{b:.2f}"
                    if name in tried_methods: continue
                    pred = sigmoid(a * file_logit_max + b * lp)
                    auc = macro_auc(file_labels, pred)
                    append_result(name, auc, wg4=wg4, wg5=wg5, ww1=ww1, a=a, b=b)
                    if auc > best_E: best_E = auc
                    maybe_update_best(name, auc, wg4=wg4, wg5=wg5, ww1=ww1, a=a, b=b)

print(f"  Best 4-way: {best_E:.4f}")

# ── FINAL SUMMARY ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
with open(RESULTS_PATH) as f:
    final_db = json.load(f)
final_best = final_db.get('best', {})
print(f"  Final best: {final_best}")

# Show top-10 experiments
exps = [(e.get('loo_auc', 0), e['method']) for e in final_db.get('experiments', []) if 'loo_auc' in e]
exps.sort(reverse=True)
print("\n  Top 10 overall:")
for auc, m in exps[:10]:
    print(f"    {auc:.6f}  {m}")
print("done")
