"""
eval_pantanal_improvements.py
──────────────────────────────────────────────────────────────────────────────
本地評估 pantanal-distill 後處理方法（使用 perch_labeled_ss.npz 作 proxy val）

方法清單：
  0. Baseline          : sigmoid(Perch logits)
  1. GMM Cluster Prior : PCA(64) + GMM(32) 對 127k windows 做聚類先驗
  2. Co-occurrence Prop: P(j|i) 共現傳播（來自 train_soundscapes_labels.csv）
  3. Taxonomy Sibling  : 同屬種類相互提升
  4. Combine 1+3       : GMM + Sibling
  5. Combine 2+3       : Co-occur + Sibling
  6. Combine 1+2+3     : 全部組合

評估指標：window-level macro AUC（跳過無正樣本的類別）
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
from sklearn.cluster import SpectralClustering
from scipy.special import expit as sigmoid
from itertools import product
import warnings
warnings.filterwarnings('ignore')

ROOT = Path('birdclef-2026')
OUT  = Path('outputs')
EPS  = 1e-7

# ══════════════════════════════════════════════════════════════════════════════
# 1. 載入資料
# ══════════════════════════════════════════════════════════════════════════════
print('Loading data...')
lab = np.load(OUT / 'perch_labeled_ss.npz')
lab_emb       = lab['emb']        # (739, 1536)
lab_logits    = lab['logits']     # (739, 234)
lab_labels    = lab['labels'].astype(np.float32)  # (739, 234)
lab_filenames = lab['filenames']  # (739,)
lab_n_windows = lab['n_windows']  # (66,)

all_ss = np.load(OUT / 'perch_emb_all_ss.npz')
all_emb = all_ss['emb']           # (127896, 1536)

taxonomy = pd.read_csv(ROOT / 'taxonomy.csv')
SPECIES  = taxonomy['primary_label'].tolist()
N_CLASS  = len(SPECIES)           # 234
sp2idx   = {s: i for i, s in enumerate(SPECIES)}

ss_labels_df = pd.read_csv(ROOT / 'train_soundscapes_labels.csv')

print(f'  Labeled SS: {lab_emb.shape}, All SS: {all_emb.shape}')
print(f'  Species: {N_CLASS}, Files: {len(lab_n_windows)}')

# ══════════════════════════════════════════════════════════════════════════════
# 2. 評估指標
# ══════════════════════════════════════════════════════════════════════════════
def macro_auc(y_true, y_score):
    valid = y_true.sum(0) > 0
    if valid.sum() == 0:
        return 0.0
    return roc_auc_score(y_true[:, valid], y_score[:, valid], average='macro')

valid_cols = lab_labels.sum(0) > 0
print(f'  Valid classes: {valid_cols.sum()}/234')

# ══════════════════════════════════════════════════════════════════════════════
# 3. Baseline
# ══════════════════════════════════════════════════════════════════════════════
base_probs   = sigmoid(lab_logits)
baseline_auc = macro_auc(lab_labels, base_probs)
print(f'\n[0] Baseline AUC: {baseline_auc:.4f}')

results = {'baseline': baseline_auc}

# ══════════════════════════════════════════════════════════════════════════════
# 4. 方法一：GMM Cluster Prior
# ══════════════════════════════════════════════════════════════════════════════
print('\n[1] GMM Cluster Prior...')
print('  Normalizing embeddings...')
all_emb_norm = normalize(all_emb, norm='l2')
lab_emb_norm = normalize(lab_emb, norm='l2')

PCA_DIM      = 64
N_COMPONENTS = 32

print(f'  PCA({PCA_DIM}) on {len(all_emb_norm)} windows...')
pca = PCA(n_components=PCA_DIM, random_state=42)
all_pca = pca.fit_transform(all_emb_norm)
lab_pca = pca.transform(lab_emb_norm)
print(f'  Variance explained: {pca.explained_variance_ratio_.sum():.3f}')

print(f'  GMM({N_COMPONENTS} components)...')
gmm = GaussianMixture(
    n_components=N_COMPONENTS,
    covariance_type='diag',
    random_state=42,
    max_iter=100,
    n_init=3,
    verbose=0,
)
gmm.fit(all_pca)
print(f'  GMM converged: {gmm.converged_}')

lab_posteriors = gmm.predict_proba(lab_pca)   # (739, 32)

def build_cluster_profiles(posteriors, labels, alpha=3.0):
    K = posteriors.shape[1]
    numerator   = posteriors.T @ labels         # (K, 234)
    denominator = posteriors.sum(axis=0) + alpha # (K,)
    return numerator / denominator[:, None]

cluster_profiles = build_cluster_profiles(lab_posteriors, lab_labels, alpha=3.0)
cluster_prior    = lab_posteriors @ cluster_profiles   # (739, 234)

best_gmm_auc, best_gmm_lam = baseline_auc, None
for lam in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
    base_logit  = np.log(base_probs.clip(EPS) / (1 - base_probs).clip(EPS))
    prior_logit = np.log(cluster_prior.clip(EPS) / (1 - cluster_prior).clip(EPS))
    blended     = sigmoid(base_logit + lam * prior_logit)
    auc         = macro_auc(lab_labels, blended)
    print(f'  lambda={lam:.2f} → AUC={auc:.4f} ({auc-baseline_auc:+.4f})')
    if auc > best_gmm_auc:
        best_gmm_auc, best_gmm_lam = auc, lam

print(f'  Best GMM lambda={best_gmm_lam}  AUC={best_gmm_auc:.4f} ({best_gmm_auc-baseline_auc:+.4f})')
results['gmm'] = best_gmm_auc

# 儲存 GMM artifacts
gmm_artifacts = dict(pca=pca, gmm=gmm, cluster_profiles=cluster_profiles,
                     best_lambda=best_gmm_lam, n_components=N_COMPONENTS, pca_dim=PCA_DIM)
save_path = Path('weights/gmm_cluster_prior.pkl')
save_path.parent.mkdir(parents=True, exist_ok=True)
with open(save_path, 'wb') as f:
    pickle.dump(gmm_artifacts, f)
print(f'  Saved → {save_path}')

def apply_gmm(probs, emb_norm, lam):
    pca_proj   = pca.transform(normalize(emb_norm, norm='l2'))
    posteriors = gmm.predict_proba(pca_proj)
    prior      = posteriors @ cluster_profiles
    bl = np.log(probs.clip(EPS) / (1-probs).clip(EPS))
    pl = np.log(prior.clip(EPS)  / (1-prior).clip(EPS))
    return sigmoid(bl + lam * pl)

# ══════════════════════════════════════════════════════════════════════════════
# 5. 方法二：Co-occurrence Propagation
# ══════════════════════════════════════════════════════════════════════════════
print('\n[2] Co-occurrence Propagation...')

# 從 train_soundscapes_labels.csv 建立共現矩陣
file_species = {}
for _, row in ss_labels_df.iterrows():
    fn = row['filename']
    if fn not in file_species:
        file_species[fn] = set()
    for sp in str(row['primary_label']).split(';'):
        sp = sp.strip()
        if sp in sp2idx:
            file_species[fn].add(sp2idx[sp])

cooccur     = np.zeros((N_CLASS, N_CLASS), dtype=np.float32)
sp_count    = np.zeros(N_CLASS, dtype=np.float32)
for fn, sp_set in file_species.items():
    for i in sp_set:
        sp_count[i] += 1
        for j in sp_set:
            cooccur[i, j] += 1

cooccur_prob = cooccur / (sp_count[:, None] + EPS)
np.fill_diagonal(cooccur_prob, 0.0)
print(f'  Active species (in train SS): {(sp_count > 0).sum()}')

def cooccur_propagation(preds, anchor_threshold=0.4, propagation_weight=0.12, top_k=5):
    result = preds.copy()
    for n in range(len(preds)):
        p = preds[n]
        anchor_mask = p >= anchor_threshold
        if anchor_mask.sum() == 0:
            continue
        if anchor_mask.sum() > top_k:
            top_idx = np.argsort(p)[::-1][:top_k]
            anchor_mask = np.zeros(N_CLASS, dtype=bool)
            anchor_mask[top_idx] = True
        anchor_probs = p[anchor_mask][:, None]
        delta = (anchor_probs * cooccur_prob[anchor_mask, :]).sum(0)
        result[n] = np.clip(p + propagation_weight * delta * (1.0 - p), 0.0, 1.0)
    return result

best_cooc_auc, best_cooc_params = baseline_auc, None
for anchor_thr, pw in product([0.3, 0.4, 0.5], [0.05, 0.10, 0.15, 0.20]):
    out = cooccur_propagation(base_probs, anchor_thr, pw)
    auc = macro_auc(lab_labels, out)
    if auc > best_cooc_auc:
        best_cooc_auc   = auc
        best_cooc_params = dict(anchor_threshold=anchor_thr, propagation_weight=pw)
    print(f'  anchor={anchor_thr:.1f} pw={pw:.2f} → AUC={auc:.4f} ({auc-baseline_auc:+.4f})')

print(f'  Best: {best_cooc_params}  AUC={best_cooc_auc:.4f} ({best_cooc_auc-baseline_auc:+.4f})')
results['cooccur'] = best_cooc_auc

# ══════════════════════════════════════════════════════════════════════════════
# 6. 方法三：Taxonomy Sibling Boost
# ══════════════════════════════════════════════════════════════════════════════
print('\n[3] Taxonomy Sibling Boost...')

# 解析 genus 和 class
taxonomy['genus'] = taxonomy['scientific_name'].str.split().str[0]
genus_list = taxonomy['genus'].tolist()
class_list = taxonomy['class_name'].tolist()

def build_sibling_matrix(genus_list, class_list, genus_boost=0.4, class_boost=0.05):
    N = len(genus_list)
    mat = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if genus_list[i] == genus_list[j]:
                mat[i, j] = genus_boost
            elif class_list[i] == class_list[j]:
                mat[i, j] = class_boost
    return mat

sibling_matrix = build_sibling_matrix(genus_list, class_list, 0.4, 0.05)

def taxonomy_sibling_boost(preds, sibling_matrix, anchor_thr=0.45, boost_strength=0.08, top_k=8):
    result = preds.copy()
    for n in range(len(preds)):
        p = preds[n]
        anchor_mask = p >= anchor_thr
        if anchor_mask.sum() == 0:
            continue
        if anchor_mask.sum() > top_k:
            top_idx = np.argsort(p)[::-1][:top_k]
            anchor_mask = np.zeros(N_CLASS, dtype=bool)
            anchor_mask[top_idx] = True
        anchor_probs = p[anchor_mask][:, None]  # (n_anchors, 1)
        boost = (anchor_probs * sibling_matrix[anchor_mask, :]).sum(0)  # (234,)
        result[n] = np.clip(p + boost_strength * boost * (1.0 - p), 0.0, 1.0)
    return result

best_sib_auc, best_sib_params = baseline_auc, None
for anchor_thr, boost in product([0.35, 0.40, 0.45, 0.50], [0.04, 0.06, 0.08, 0.10, 0.15]):
    out = taxonomy_sibling_boost(base_probs, sibling_matrix, anchor_thr, boost)
    auc = macro_auc(lab_labels, out)
    if auc > best_sib_auc:
        best_sib_auc    = auc
        best_sib_params = dict(anchor_thr=anchor_thr, boost_strength=boost)
    if boost in [0.06, 0.10]:
        print(f'  anchor={anchor_thr:.2f} boost={boost:.2f} → AUC={auc:.4f} ({auc-baseline_auc:+.4f})')

print(f'  Best: {best_sib_params}  AUC={best_sib_auc:.4f} ({best_sib_auc-baseline_auc:+.4f})')
# alias for apply_all compatibility
best_sib_params_call = best_sib_params
results['sibling'] = best_sib_auc

# ══════════════════════════════════════════════════════════════════════════════
# 7. 方法四：Class-aware Temporal Smoothing
# ══════════════════════════════════════════════════════════════════════════════
print('\n[4] Class-aware Temporal Smoothing...')

aves_mask     = np.array([c == 'Aves'     for c in class_list], dtype=bool)
insecta_mask  = np.array([c == 'Insecta'  for c in class_list], dtype=bool)
amphibia_mask = np.array([c == 'Amphibia' for c in class_list], dtype=bool)

def class_aware_smooth(preds, n_windows_per_file, alpha_aves=0.15, alpha_insecta=0.40, alpha_amphibia=0.35):
    result = preds.copy()
    idx = 0
    for nw in n_windows_per_file:
        chunk = preds[idx:idx+nw].copy()   # (nw, 234)
        out   = chunk.copy()
        # Aves: max-pool boost (event-based)
        if alpha_aves > 0:
            file_max = chunk[:, aves_mask].max(0, keepdims=True)  # (1, n_aves)
            out[:, aves_mask] = chunk[:, aves_mask] + alpha_aves * (file_max - chunk[:, aves_mask])
        # Insecta: avg-pool smooth (texture-based)
        if alpha_insecta > 0:
            file_mean = chunk[:, insecta_mask].mean(0, keepdims=True)
            out[:, insecta_mask] = (1 - alpha_insecta) * chunk[:, insecta_mask] + alpha_insecta * file_mean
        # Amphibia: avg-pool smooth (chorus-based)
        if alpha_amphibia > 0:
            file_mean = chunk[:, amphibia_mask].mean(0, keepdims=True)
            out[:, amphibia_mask] = (1 - alpha_amphibia) * chunk[:, amphibia_mask] + alpha_amphibia * file_mean
        result[idx:idx+nw] = np.clip(out, 0.0, 1.0)
        idx += nw
    return result

best_smooth_auc, best_smooth_params = baseline_auc, None
for a_aves, a_ins, a_amp in product([0.0, 0.10, 0.15, 0.20],
                                     [0.0, 0.30, 0.40, 0.50],
                                     [0.0, 0.25, 0.35, 0.45]):
    out = class_aware_smooth(base_probs, lab_n_windows, a_aves, a_ins, a_amp)
    auc = macro_auc(lab_labels, out)
    if auc > best_smooth_auc:
        best_smooth_auc    = auc
        best_smooth_params = dict(alpha_aves=a_aves, alpha_insecta=a_ins, alpha_amphibia=a_amp)

print(f'  Best: {best_smooth_params}  AUC={best_smooth_auc:.4f} ({best_smooth_auc-baseline_auc:+.4f})')
results['class_smooth'] = best_smooth_auc

# ══════════════════════════════════════════════════════════════════════════════
# 8. 組合實驗
# ══════════════════════════════════════════════════════════════════════════════
print('\n[5] Combinations...')

# 最佳單方法參數
gmm_lam  = best_gmm_lam  or 0.15
cooc_p   = best_cooc_params or dict(anchor_threshold=0.4, propagation_weight=0.12)
sib_p    = best_sib_params  or dict(anchor_threshold=0.45, boost_strength=0.08)
sm_p     = best_smooth_params or dict(alpha_aves=0.15, alpha_insecta=0.40, alpha_amphibia=0.35)

def apply_all(probs, do_gmm=True, do_cooc=True, do_sib=True, do_smooth=True):
    out = probs.copy()
    if do_gmm  and gmm_lam is not None:
        out = apply_gmm(out, lab_emb_norm, gmm_lam)
    if do_cooc:
        out = cooccur_propagation(out, **cooc_p)
    if do_sib:
        out = taxonomy_sibling_boost(out, sibling_matrix, **sib_p)
    if do_smooth:
        out = class_aware_smooth(out, lab_n_windows, **sm_p)
    return out

combos = {
    'GMM+Sibling'       : dict(do_gmm=True,  do_cooc=False, do_sib=True,  do_smooth=False),
    'GMM+Smooth'        : dict(do_gmm=True,  do_cooc=False, do_sib=False, do_smooth=True),
    'Cooc+Sibling'      : dict(do_gmm=False, do_cooc=True,  do_sib=True,  do_smooth=False),
    'Sibling+Smooth'    : dict(do_gmm=False, do_cooc=False, do_sib=True,  do_smooth=True),
    'GMM+Sibling+Smooth': dict(do_gmm=True,  do_cooc=False, do_sib=True,  do_smooth=True),
    'All'               : dict(do_gmm=True,  do_cooc=True,  do_sib=True,  do_smooth=True),
}

for name, kwargs in combos.items():
    out = apply_all(base_probs, **kwargs)
    auc = macro_auc(lab_labels, out)
    delta = auc - baseline_auc
    print(f'  {name:<22} AUC={auc:.4f} ({delta:+.4f})')
    results[name] = auc

# ══════════════════════════════════════════════════════════════════════════════
# 9. 最終摘要
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('SUMMARY (window-level macro AUC)')
print('='*60)
for name, auc in results.items():
    delta = auc - baseline_auc
    marker = ' ←BEST' if auc == max(results.values()) else ''
    print(f'  {name:<25} {auc:.4f}  ({delta:+.4f}){marker}')

best_method = max(results, key=results.get)
best_auc    = results[best_method]
print(f'\nBest method: {best_method}  AUC={best_auc:.4f}  (+{best_auc-baseline_auc:.4f})')

# 儲存 sibling matrix
np.save('weights/sibling_matrix.npy', sibling_matrix)
import json
with open('weights/taxonomy_sibling_config.json', 'w') as f:
    json.dump(dict(best_sib_params or {}, genus_boost=0.4, class_boost=0.05), f, indent=2)
print('\nSaved: weights/sibling_matrix.npy, taxonomy_sibling_config.json')
