"""
Evaluate full pipeline AUC for logspace_geo5_win1 method.
Formula: sigmoid(a * logit_max + b * log(w_geo * geo_k5 + w_win * win_k1))
New best LOO-AUC = 0.9164 (embed prior only)

Full pipeline integrates with VLOM base (ProtoSSM + SED).
"""
import numpy as np, pickle, re, os
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')

os.chdir("/home/lab/BirdClef-2026-Codebase")

# Load data
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

emb_win_norm = normalize(emb_win, norm='l2').astype(np.float32)
win_file_id  = np.zeros(len(emb_win), np.int32)
for fi in range(n_files):
    win_file_id[int(file_start[fi]):int(file_end[fi])] = fi

# Load SED predictions
sed_npz = np.load("outputs/sed-ns-b0-20s-r1/all_ss_probs_corrected.npz", allow_pickle=True)
sed_row_ids   = sed_npz['row_ids']
sed_probs_all = sed_npz['probs'].astype(np.float32)
sed_by_file   = {}
for i, rid in enumerate(sed_row_ids):
    fname_base = '_'.join(str(rid).split('_')[:-1])
    if fname_base not in sed_by_file:
        sed_by_file[fname_base] = []
    sed_by_file[fname_base].append(i)

file_sed_max = np.zeros((n_files, n_species), np.float32)
for fi, fname in enumerate(file_list):
    fname_base = fname.replace('.ogg', '').replace('.flac', '')
    if fname_base in sed_by_file:
        file_sed_max[fi] = sed_probs_all[sed_by_file[fname_base]].max(0)

def macro_auc(yt, ys):
    mask = yt.sum(0) > 0
    return roc_auc_score(yt[:, mask], ys[:, mask], average='macro')

EPS = 1e-7

def vlom_blend(a, b, w_a=0.5, w_b=0.5):
    w_s = w_a + w_b; w_a /= w_s; w_b /= w_s
    la = np.log(a.clip(EPS) / (1-a).clip(EPS))
    lb = np.log(b.clip(EPS) / (1-b).clip(EPS))
    return sigmoid(w_a * la + w_b * lb)

# Base scores
base_probs = vlom_blend(sigmoid(file_logit_max), file_sed_max, 0.5, 0.5)
base_auc   = macro_auc(file_labels, base_probs)
base_logit = np.log(base_probs.clip(EPS)) - np.log((1-base_probs).clip(EPS))
print(f"Base (VLOM 50/50) AUC: {base_auc:.4f}")

# Load pkl
with open("outputs/embed_prior_logspace_geo5_win1.pkl", "rb") as f:
    ep = pickle.load(f)

X_ref    = ep['X_combined_n'].astype(np.float32)  # (66, 39) pkl space
fl       = ep['file_labels'].astype(np.float32)
emb_ref  = ep['emb_win_norm'].astype(np.float32)  # (739, 1536) window embeddings
wfi      = ep['win_file_id'].astype(np.int32)
A        = ep.get('logspace_a', 0.70)
B        = ep.get('logspace_b', 1.45)
k_geo    = ep.get('k_geo', 5)
T_geo    = ep.get('T_geo', 0.2)
k_win    = ep.get('k_win', 1)
w_geo    = ep.get('w_geo', 0.50)
w_win    = 1.0 - w_geo

print(f"\nPKL params: a={A}, b={B}, k_geo={k_geo}, T_geo={T_geo}, k_win={k_win}, w_geo={w_geo}, w_win={w_win}")

# LOO geo-KNN in pkl X_combined_n space
print("Computing geo-KNN LOO (pkl space)...")
y_geo = np.zeros((n_files, n_species), np.float32)
for i in range(n_files):
    tr = np.array([j for j in range(n_files) if j != i])
    sims = (X_ref[[i]] @ X_ref[tr].T).ravel()
    top = np.argsort(-sims)[:k_geo]
    logit_s = sims[top] / T_geo; logit_s -= logit_s.max()
    w = np.exp(logit_s); w /= w.sum()
    y_geo[i] = (w[:, None] * fl[tr[top]]).sum(0)

# LOO window-KNN
print("Computing window-KNN LOO...")
y_win = np.zeros((n_files, n_species), np.float32)
for i in range(n_files):
    te_s, te_e = int(file_start[i]), int(file_end[i])
    X_te = emb_win_norm[te_s:te_e]
    tr_mask = win_file_id != i
    X_tr = emb_win_norm[tr_mask]; tr_fi = win_file_id[tr_mask]
    sims = X_te @ X_tr.T; top_idx = np.argsort(-sims, 1)[:, :k_win]
    wp = np.zeros((te_e - te_s, n_species), np.float32)
    for wi in range(te_e - te_s):
        w = sims[wi, top_idx[wi]].clip(0); ws = w.sum()
        w = w/ws if ws > 1e-8 else np.ones(k_win)/k_win
        wp[wi] = (w[:, None] * file_labels[tr_fi[top_idx[wi]]]).sum(0)
    y_win[i] = wp.mean(0)

# Verify LOO-AUC (should match ~0.9164)
y_blend = w_geo * y_geo + w_win * y_win
ep_probs_loo = sigmoid(A * file_logit_max + B * np.log(y_blend.clip(EPS)))
ep_auc = macro_auc(file_labels, ep_probs_loo)
print(f"\nEP-only LOO-AUC: {ep_auc:.4f} (expected ~0.9164)")

# Full pipeline evaluation: different LAMBDA values
print("\n=== Full pipeline sweep (varying LAMBDA) ===")
print(f"{'Method':30s}  {'EP_only':8s}  {'Full_AUC':8s}  {'Delta_base':10s}")
print("-"*65)

# Method 1: Use logspace output as replace for base (LAMBDA approach with logit-delta)
# ep_logit_delta = logit(ep_probs) - logit_max (relative to raw Perch)
ep_logit_delta = np.log(ep_probs_loo.clip(EPS)) - np.log((1-ep_probs_loo).clip(EPS)) - file_logit_max

best_lam, best_auc = 0, 0
for lam in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 1.0]:
    full = sigmoid(base_logit + lam * ep_logit_delta)
    full_auc = macro_auc(file_labels, full)
    marker = " ←" if full_auc > best_auc else ""
    print(f"  logit_delta lam={lam:.2f}:             {ep_auc:.4f}    {full_auc:.4f}    {full_auc-base_auc:+.4f}{marker}")
    if full_auc > best_auc:
        best_auc = full_auc; best_lam = lam

print(f"\nBest: lam={best_lam}, full_AUC={best_auc:.4f}")

# Method 2: Additive in logit space: base_logit + LAMBDA * logit(ep_probs)
print("\n--- Method 2: base_logit + LAMBDA * logit(ep_probs) ---")
ep_logit = np.log(ep_probs_loo.clip(EPS)) - np.log((1-ep_probs_loo).clip(EPS))
best2_lam, best2_auc = 0, 0
for lam in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    full = sigmoid(base_logit + lam * ep_logit)
    full_auc = macro_auc(file_labels, full)
    marker = " ←" if full_auc > best2_auc else ""
    print(f"  lam={lam:.2f}: {full_auc:.4f} ({full_auc-base_auc:+.4f}){marker}")
    if full_auc > best2_auc:
        best2_auc = full_auc; best2_lam = lam

# Method 3: Logspace full: sigmoid(a * base_logit + b * log(y_blend)) — same structure as v9
print("\n--- Method 3: sigmoid(a * base_logit + b * log(blended_knn)) ---")
best3_auc = 0
for a in [0.60, 0.70, 0.80, 0.90, 1.00]:
    for b in [1.20, 1.30, 1.40, 1.45, 1.50, 1.60, 1.70]:
        full = sigmoid(a * base_logit + b * np.log(y_blend.clip(EPS)))
        full_auc = macro_auc(file_labels, full)
        if full_auc > best3_auc:
            best3_auc = full_auc
            print(f"  a={a:.2f} b={b:.2f}: {full_auc:.4f} ({full_auc-base_auc:+.4f}) ← new best")

print(f"\n=== Summary ===")
print(f"  Base AUC:          {base_auc:.4f}")
print(f"  EP-only LOO-AUC:   {ep_auc:.4f}")
print(f"  Best Method 1 (logit_delta): lam={best_lam}, AUC={best_auc:.4f} ({best_auc-base_auc:+.4f})")
print(f"  Best Method 2 (additive):    lam={best2_lam}, AUC={best2_auc:.4f} ({best2_auc-base_auc:+.4f})")
print(f"  Best Method 3 (logspace):    AUC={best3_auc:.4f} ({best3_auc-base_auc:+.4f})")
print(f"  v7-geo-knn ref:    0.9246")
