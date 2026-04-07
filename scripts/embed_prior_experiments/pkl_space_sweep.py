"""
Full pipeline sweep using the actual pkl X_combined_n spaces.
These are the correct feature spaces that give 0.9246 for v7-geo-knn.

Target: > 0.9246 full pipeline LOO-CV AUC
"""
import numpy as np, pickle, re, os
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')

os.chdir("/home/lab/BirdClef-2026-Codebase")

# ── Load data ──────────────────────────────────────────────────────────────
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

win_file_id = np.zeros(len(emb_win), np.int32)
for fi in range(n_files):
    win_file_id[int(file_start[fi]):int(file_end[fi])] = fi
emb_win_norm = normalize(emb_win, norm='l2').astype(np.float32)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

file_embs      = np.zeros((n_files, emb_win.shape[1]), np.float32)
file_labels    = np.zeros((n_files, n_species), np.float32)
file_logit_max = np.zeros((n_files, n_species), np.float32)
file_prob_mean = np.zeros((n_files, n_species), np.float32)
for fi in range(n_files):
    s, e = int(file_start[fi]), int(file_end[fi])
    file_embs[fi]      = emb_win[s:e].mean(0)
    file_labels[fi]    = (labels_win[s:e].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi] = logits_win[s:e].max(0)
    file_prob_mean[fi] = sigmoid(logits_win[s:e]).mean(0)
file_embs_norm = normalize(file_embs, norm='l2').astype(np.float32)

# ── Load SED ──────────────────────────────────────────────────────────────
sed_npz = np.load("outputs/sed-ns-b0-20s-r1/all_ss_probs_corrected.npz", allow_pickle=True)
sed_row_ids = sed_npz['row_ids']
sed_probs_all = sed_npz['probs'].astype(np.float32)
sed_by_file = {}
for i, rid in enumerate(sed_row_ids):
    fname_base = '_'.join(str(rid).split('_')[:-1])
    if fname_base not in sed_by_file:
        sed_by_file[fname_base] = []
    sed_by_file[fname_base].append(i)
file_sed_max  = np.zeros((n_files, n_species), np.float32)
file_sed_mean = np.zeros((n_files, n_species), np.float32)
for fi, fname in enumerate(file_list):
    fname_base = fname.replace('.ogg','').replace('.flac','')
    if fname_base in sed_by_file:
        idxs = sed_by_file[fname_base]
        wp = sed_probs_all[idxs]
        file_sed_max[fi] = wp.max(0)
        file_sed_mean[fi] = wp.mean(0)

print(f"Files={n_files}, species={n_species}")

EPS = 1e-7

def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

def vlom_blend(a, b, w_a=0.5, w_b=0.5):
    w_sum = w_a + w_b; w_a /= w_sum; w_b /= w_sum
    la = np.log(a.clip(EPS) / (1-a).clip(EPS))
    lb = np.log(b.clip(EPS) / (1-b).clip(EPS))
    return sigmoid(w_a * la + w_b * lb)

def attn_knn_loo(X, k=10, T=0.2, labels=None):
    fl = file_labels if labels is None else labels
    preds = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        tr = np.array([j for j in range(n_files) if j != i])
        sims = (X[[i]] @ X[tr].T).ravel()
        top  = np.argsort(-sims)[:k]
        logit = sims[top] / T; logit -= logit.max()
        w = np.exp(logit); w /= w.sum()
        preds[i] = (w[:, None] * fl[tr[top]]).sum(0)
    return preds

def window_knn_loo(k=1):
    preds = np.zeros((n_files, n_species), np.float32)
    for i in range(n_files):
        te_s, te_e = int(file_start[i]), int(file_end[i])
        X_te = emb_win_norm[te_s:te_e]
        tr_mask = win_file_id != i
        X_tr = emb_win_norm[tr_mask]; tr_wi = np.where(tr_mask)[0]
        Y_tr = file_labels[win_file_id[tr_wi]]
        sims = X_te @ X_tr.T; top_idx = np.argsort(-sims, 1)[:, :k]
        wp = np.zeros((te_e - te_s, n_species), np.float32)
        for wi in range(te_e - te_s):
            w = sims[wi, top_idx[wi]].clip(0); ws = w.sum()
            w = w/ws if ws > 1e-8 else np.ones(k)/k
            wp[wi] = (w[:, None] * Y_tr[top_idx[wi]]).sum(0)
        preds[i] = wp.mean(0)
    return preds

def full_pipeline_auc(y_ep, lam=0.25, proto_w=0.5, sed_w=0.5, sed_agg='max'):
    sed_p = file_sed_max if sed_agg == 'max' else file_sed_mean
    proto_p = sigmoid(file_logit_max)
    base = vlom_blend(proto_p, sed_p, w_a=proto_w, w_b=sed_w)
    bl = np.log(base.clip(EPS)) - np.log((1-base).clip(EPS))
    el = np.log(y_ep.clip(EPS)) - np.log((1-y_ep).clip(EPS))
    return macro_auc(file_labels, sigmoid(bl + lam * el))

def ep_only_auc(y_ep):
    return macro_auc(file_labels, y_ep)

# ── Load all pkl spaces ────────────────────────────────────────────────────
pkls = {
    'attn':     'outputs/embed_prior_attn.pkl',
    'combined': 'outputs/embed_prior_combined.pkl',
    'cosine':   'outputs/embed_prior_cosine.pkl',
    'model':    'outputs/embed_prior_model.pkl',
    'blend':    'outputs/embed_prior_blend.pkl',
    'logspace': 'outputs/embed_prior_logspace.pkl',
}
pkl_data = {}
for name, path in pkls.items():
    try:
        with open(path, 'rb') as f:
            d = pickle.load(f)
        pkl_data[name] = d
        X = d.get('X_combined_n', None)
        if X is not None:
            print(f"  pkl[{name}]: X_combined_n shape={X.shape}")
        else:
            print(f"  pkl[{name}]: no X_combined_n, keys={list(d.keys())[:5]}")
    except Exception as e:
        print(f"  pkl[{name}]: error={e}")

# The main space to optimize on is 'attn' (v7-geo-knn)
X_attn = pkl_data['attn']['X_combined_n'].astype(np.float32)
fl_attn = pkl_data['attn']['file_labels'].astype(np.float32)

# ── Reference (v7-geo-knn): k=10, T=0.2, λ=0.25 ──────────────────────────
y_ref = attn_knn_loo(X_attn, k=10, T=0.2, labels=fl_attn)
ref_full = full_pipeline_auc(y_ref, lam=0.25)
ref_ep   = ep_only_auc(y_ref)
print(f"\nReference v7-geo-knn:")
print(f"  EP-only AUC: {ref_ep:.4f} (expected ~0.8758)")
print(f"  Full pipeline: {ref_full:.4f} (expected 0.9246)")
print(f"\nTarget: full pipeline > {ref_full:.4f}")

BEST = ref_full
results = []

def record(name, auc, params=None):
    global BEST
    mk = " ★ NEW BEST" if auc > BEST else ""
    if auc > BEST:
        BEST = auc
    results.append({'name': name, 'auc': auc, 'params': params or {}})
    return mk

# ── A) λ sweep on attn space ──────────────────────────────────────────────
print("\n" + "="*60)
print("A) λ sweep (X_attn, k=10, T=0.2, 50/50)")
print("="*60)
for lam in [0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    auc = full_pipeline_auc(y_ref, lam=lam)
    mk = record(f"lam{lam:.2f}", auc, {'lam': lam, 'space': 'attn', 'k': 10, 'T': 0.2})
    print(f"  λ={lam:.2f}: {auc:.4f}{mk}")

# ── B) k sweep on attn space ──────────────────────────────────────────────
print("\n" + "="*60)
print("B) k sweep (X_attn, T=0.2, λ=0.25, 50/50)")
print("="*60)
for k in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
    y_k = attn_knn_loo(X_attn, k=k, T=0.2, labels=fl_attn)
    auc = full_pipeline_auc(y_k, lam=0.25)
    ep_a = ep_only_auc(y_k)
    mk = record(f"k{k}", auc, {'k': k, 'lam': 0.25, 'T': 0.2, 'space': 'attn'})
    print(f"  k={k}: full={auc:.4f} ep={ep_a:.4f}{mk}")

# ── C) k+λ joint sweep on attn space ─────────────────────────────────────
print("\n" + "="*60)
print("C) k+λ joint sweep (X_attn, T=0.2)")
print("="*60)
grid_kl = []
for k in [2, 3, 4, 5, 6, 7, 8, 10]:
    y_k = attn_knn_loo(X_attn, k=k, T=0.2, labels=fl_attn)
    for lam in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        auc = full_pipeline_auc(y_k, lam=lam)
        grid_kl.append((auc, k, lam))
        record(f"C_k{k}_lam{lam:.2f}", auc, {'k': k, 'lam': lam, 'T': 0.2, 'space': 'attn'})
grid_kl.sort(reverse=True)
print("  Top 15:")
for auc, k, lam in grid_kl[:15]:
    mk = "★" if auc > ref_full else ""
    print(f"    k={k} λ={lam:.2f}: {auc:.4f} {mk}")

best_k_C, best_lam_C = grid_kl[0][1], grid_kl[0][2]
print(f"\n  Best from C: k={best_k_C}, λ={best_lam_C:.2f}, auc={grid_kl[0][0]:.4f}")

# ── D) T sweep on attn space ──────────────────────────────────────────────
print("\n" + "="*60)
print(f"D) T sweep (X_attn, k={best_k_C}, λ={best_lam_C:.2f})")
print("="*60)
for T in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.50]:
    y_T = attn_knn_loo(X_attn, k=best_k_C, T=T, labels=fl_attn)
    auc = full_pipeline_auc(y_T, lam=best_lam_C)
    mk = record(f"D_T{T:.2f}", auc, {'T': T, 'k': best_k_C, 'lam': best_lam_C, 'space': 'attn'})
    print(f"  T={T:.2f}: {auc:.4f}{mk}")

# ── E) VLOM weight × λ on attn space ─────────────────────────────────────
print("\n" + "="*60)
print(f"E) VLOM weights (X_attn, k={best_k_C})")
print("="*60)
y_best_k = attn_knn_loo(X_attn, k=best_k_C, T=0.2, labels=fl_attn)
grid_vlom = []
for pw in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    for lam in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        auc = full_pipeline_auc(y_best_k, lam=lam, proto_w=pw, sed_w=1-pw)
        grid_vlom.append((auc, pw, lam))
        record(f"E_pw{pw:.2f}_lam{lam:.2f}", auc,
               {'pw': pw, 'lam': lam, 'k': best_k_C, 'T': 0.2, 'space': 'attn'})
grid_vlom.sort(reverse=True)
print("  Top 15:")
for auc, pw, lam in grid_vlom[:15]:
    mk = "★" if auc > ref_full else ""
    print(f"    proto={pw:.2f} λ={lam:.2f}: {auc:.4f} {mk}")

# ── F) Combined spaces ────────────────────────────────────────────────────
print("\n" + "="*60)
print("F) Combined embed spaces (attn + combined pkl)")
print("="*60)
if 'combined' in pkl_data and 'X_combined_n' in pkl_data['combined']:
    X_comb = pkl_data['combined']['X_combined_n'].astype(np.float32)
    fl_comb = pkl_data['combined']['file_labels'].astype(np.float32)
    for k in [3, 5, 7, 10]:
        y_comb = attn_knn_loo(X_comb, k=k, T=0.2, labels=fl_comb)
        for lam in [0.20, 0.25, 0.30, 0.35]:
            auc = full_pipeline_auc(y_comb, lam=lam)
            mk = record(f"F_combined_k{k}_lam{lam:.2f}", auc,
                        {'k': k, 'lam': lam, 'space': 'combined'})
    # Ensemble: attn + combined predictions
    for wa in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_comb_k3 = attn_knn_loo(X_comb, k=3, T=0.2, labels=fl_comb)
        y_ens = wa * y_best_k + (1-wa) * y_comb_k3
        for lam in [0.20, 0.25, 0.30]:
            auc = full_pipeline_auc(y_ens, lam=lam)
            mk = record(f"F_ens_wa{wa:.1f}_lam{lam:.2f}", auc,
                        {'wa': wa, 'lam': lam})
    results_F = [r for r in results if r['name'].startswith('F_')]
    results_F.sort(key=lambda x: -x['auc'])
    print("  Top 10 from F:")
    for r in results_F[:10]:
        mk = " ★" if r['auc'] > ref_full else ""
        print(f"    {r['name']}: {r['auc']:.4f}{mk}")

# ── G) Window KNN + attn ensemble with best λ ─────────────────────────────
print("\n" + "="*60)
print("G) Attn-KNN + Window KNN ensemble (best config)")
print("="*60)
print("  Computing window KNN predictions...")
y_win1 = window_knn_loo(k=1)
y_win3 = window_knn_loo(k=3)

for wa in [0.7, 0.75, 0.80, 0.85, 0.90, 0.95]:
    for wk, y_w in [(1, y_win1), (3, y_win3)]:
        y_ens = wa * y_best_k + (1-wa) * y_w
        for lam in [0.20, 0.25, 0.30, 0.35]:
            auc = full_pipeline_auc(y_ens, lam=lam)
            mk = record(f"G_wa{wa:.2f}_wk{wk}_lam{lam:.2f}", auc,
                        {'wa': wa, 'wk': wk, 'lam': lam, 'k': best_k_C})

results_G = [r for r in results if r['name'].startswith('G_')]
results_G.sort(key=lambda x: -x['auc'])
print("  Top 10 from G:")
for r in results_G[:10]:
    mk = " ★" if r['auc'] > ref_full else ""
    print(f"    {r['name']}: {r['auc']:.4f}{mk}")

# ── H) 3-way VLOM with embed_prior ────────────────────────────────────────
print("\n" + "="*60)
print("H) 3-way VLOM (ProtoSSM + SED + EmbedPrior)")
print("="*60)
proto_p = sigmoid(file_logit_max)
for ep_w in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    for pw_ratio in [0.5, 0.6]:  # proto:sed ratio
        pw = (1-ep_w) * pw_ratio
        sw = (1-ep_w) * (1-pw_ratio)
        w_sum = pw + sw + ep_w
        a, b, c = pw/w_sum, sw/w_sum, ep_w/w_sum
        la = np.log(proto_p.clip(EPS)) - np.log((1-proto_p).clip(EPS))
        lb = np.log(file_sed_max.clip(EPS)) - np.log((1-file_sed_max).clip(EPS))
        lc = np.log(y_best_k.clip(EPS)) - np.log((1-y_best_k).clip(EPS))
        full = sigmoid(a*la + b*lb + c*lc)
        auc = macro_auc(file_labels, full)
        mk = record(f"H_ew{ep_w:.2f}_pr{pw_ratio:.1f}", auc,
                    {'ep_w': ep_w, 'pw_ratio': pw_ratio})

results_H = [r for r in results if r['name'].startswith('H_')]
results_H.sort(key=lambda x: -x['auc'])
print("  Top 10 from H:")
for r in results_H[:10]:
    mk = " ★" if r['auc'] > ref_full else ""
    print(f"    {r['name']}: {r['auc']:.4f}{mk}")

# ── I) Blend pkl predictions ──────────────────────────────────────────────
print("\n" + "="*60)
print("I) Blend different pkl predictions")
print("="*60)
# attn k=best_k_C + model pkl
if 'model' in pkl_data and 'X_combined_n' in pkl_data['model']:
    X_model = pkl_data['model']['X_combined_n'].astype(np.float32)
    fl_model = pkl_data['model']['file_labels'].astype(np.float32)
    for k_m in [3, 5]:
        y_model_k = attn_knn_loo(X_model, k=k_m, T=0.2, labels=fl_model)
        for wa in [0.6, 0.7, 0.8]:
            y_ens = wa * y_best_k + (1-wa) * y_model_k
            for lam in [0.25, 0.30]:
                auc = full_pipeline_auc(y_ens, lam=lam)
                mk = record(f"I_attn+model_km{k_m}_wa{wa:.1f}_lam{lam:.2f}", auc, {})

results_I = [r for r in results if r['name'].startswith('I_')]
results_I.sort(key=lambda x: -x['auc'])
print("  Top 5 from I:")
for r in results_I[:5]:
    mk = " ★" if r['auc'] > ref_full else ""
    print(f"    {r['name']}: {r['auc']:.4f}{mk}")

# ── J) Ultra-fine sweep around best configuration ─────────────────────────
print("\n" + "="*60)
print("J) Ultra-fine sweep around best config so far")
print("="*60)
best_so_far = sorted(results, key=lambda x: -x['auc'])[0]
print(f"  Best so far: {best_so_far['name']} = {best_so_far['auc']:.4f}")
best_params = best_so_far['params']
best_k_J = best_params.get('k', best_k_C)
best_lam_J = best_params.get('lam', best_lam_C)

grid_J = []
for k in [max(1, best_k_J-2), max(1, best_k_J-1), best_k_J, best_k_J+1, best_k_J+2]:
    y_kJ = attn_knn_loo(X_attn, k=k, T=0.2, labels=fl_attn)
    for lam in [best_lam_J - 0.05, best_lam_J, best_lam_J + 0.05, best_lam_J + 0.10]:
        if lam <= 0: continue
        for pw in [0.45, 0.50, 0.55, 0.60]:
            auc = full_pipeline_auc(y_kJ, lam=lam, proto_w=pw, sed_w=1-pw)
            grid_J.append((auc, k, lam, pw))
            record(f"J_k{k}_lam{lam:.2f}_pw{pw:.2f}", auc,
                   {'k': k, 'lam': lam, 'pw': pw})

grid_J.sort(reverse=True)
print("  Top 15:")
for auc, k, lam, pw in grid_J[:15]:
    mk = "★" if auc > ref_full else ""
    print(f"    k={k} λ={lam:.2f} pw={pw:.2f}: {auc:.4f} {mk}")

# ─── SUMMARY ───────────────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"SUMMARY — Methods vs reference {ref_full:.4f}")
print("="*70)
all_sorted = sorted(results, key=lambda x: -x['auc'])
better = [r for r in all_sorted if r['auc'] > ref_full]
print(f"  # methods beating {ref_full:.4f}: {len(better)}")
for r in better[:30]:
    print(f"    {r['name']:50s} {r['auc']:.4f}  (+{r['auc']-ref_full:.4f})")

print(f"\n  Top 5 overall:")
for r in all_sorted[:5]:
    print(f"    {r['name']:50s} {r['auc']:.4f}")

print("\ndone")
