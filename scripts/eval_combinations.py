"""
eval_combinations.py — 只跑組合階段（已知各方法最佳參數）
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
from scipy.special import expit as sigmoid
import warnings
warnings.filterwarnings('ignore')

ROOT = Path('birdclef-2026')
OUT  = Path('outputs')
EPS  = 1e-7

# ── 載入資料 ──────────────────────────────────────────────────────────────────
lab = np.load(OUT / 'perch_labeled_ss.npz')
lab_emb       = lab['emb']
lab_logits    = lab['logits']
lab_labels    = lab['labels'].astype(np.float32)
lab_n_windows = lab['n_windows']

all_ss  = np.load(OUT / 'perch_emb_all_ss.npz')
all_emb = all_ss['emb']

taxonomy     = pd.read_csv(ROOT / 'taxonomy.csv')
SPECIES      = taxonomy['primary_label'].tolist()
N_CLASS      = len(SPECIES)
sp2idx       = {s: i for i, s in enumerate(SPECIES)}
ss_labels_df = pd.read_csv(ROOT / 'train_soundscapes_labels.csv')

taxonomy['genus'] = taxonomy['scientific_name'].str.split().str[0]
genus_list = taxonomy['genus'].tolist()
class_list = taxonomy['class_name'].tolist()

# ── 指標 ──────────────────────────────────────────────────────────────────────
def macro_auc(y_true, y_score):
    valid = y_true.sum(0) > 0
    return roc_auc_score(y_true[:, valid], y_score[:, valid], average='macro')

base_probs   = sigmoid(lab_logits)
baseline_auc = macro_auc(lab_labels, base_probs)
print(f'Baseline AUC: {baseline_auc:.4f}')

# ── 重建各方法（使用已知最佳參數）──────────────────────────────────────────────

# 1. GMM（載入已儲存的 artifact）
print('Loading GMM artifact...')
with open('weights/gmm_cluster_prior.pkl', 'rb') as f:
    gmm_art = pickle.load(f)
pca              = gmm_art['pca']
gmm              = gmm_art['gmm']
cluster_profiles = gmm_art['cluster_profiles']
GMM_LAM          = 0.40  # best from sweep

lab_emb_norm  = normalize(lab_emb, norm='l2')
lab_pca       = pca.transform(lab_emb_norm)
lab_posteriors = gmm.predict_proba(lab_pca)
cluster_prior  = lab_posteriors @ cluster_profiles

def apply_gmm(probs, lam=GMM_LAM):
    bl = np.log(probs.clip(EPS) / (1-probs).clip(EPS))
    pl = np.log(cluster_prior.clip(EPS) / (1-cluster_prior).clip(EPS))
    return sigmoid(bl + lam * pl)

# 2. Co-occurrence
file_species = {}
for _, row in ss_labels_df.iterrows():
    fn = row['filename']
    if fn not in file_species: file_species[fn] = set()
    for sp in str(row['primary_label']).split(';'):
        sp = sp.strip()
        if sp in sp2idx: file_species[fn].add(sp2idx[sp])

cooccur  = np.zeros((N_CLASS, N_CLASS), dtype=np.float32)
sp_count = np.zeros(N_CLASS, dtype=np.float32)
for fn, sp_set in file_species.items():
    for i in sp_set:
        sp_count[i] += 1
        for j in sp_set: cooccur[i, j] += 1
cooccur_prob = cooccur / (sp_count[:, None] + EPS)
np.fill_diagonal(cooccur_prob, 0.0)

def apply_cooccur(preds, anchor_thr=0.5, pw=0.20, top_k=5):
    result = preds.copy()
    for n in range(len(preds)):
        p = preds[n]
        anchor = p >= anchor_thr
        if not anchor.any(): continue
        if anchor.sum() > top_k:
            idx = np.argsort(p)[::-1][:top_k]
            anchor = np.zeros(N_CLASS, dtype=bool); anchor[idx] = True
        delta = (p[anchor][:, None] * cooccur_prob[anchor, :]).sum(0)
        result[n] = np.clip(p + pw * delta * (1-p), 0, 1)
    return result

# 3. Taxonomy Sibling
sibling = np.zeros((N_CLASS, N_CLASS), dtype=np.float32)
for i in range(N_CLASS):
    for j in range(N_CLASS):
        if i == j: continue
        if genus_list[i] == genus_list[j]:   sibling[i,j] = 0.4
        elif class_list[i] == class_list[j]: sibling[i,j] = 0.05

def apply_sibling(preds, anchor_thr=0.35, boost=0.15, top_k=8):
    result = preds.copy()
    for n in range(len(preds)):
        p = preds[n]
        anchor = p >= anchor_thr
        if not anchor.any(): continue
        if anchor.sum() > top_k:
            idx = np.argsort(p)[::-1][:top_k]
            anchor = np.zeros(N_CLASS, dtype=bool); anchor[idx] = True
        b = (p[anchor][:, None] * sibling[anchor, :]).sum(0)
        result[n] = np.clip(p + boost * b * (1-p), 0, 1)
    return result

# 4. Class-aware Temporal Smooth
aves_mask     = np.array([c == 'Aves'     for c in class_list])
insecta_mask  = np.array([c == 'Insecta'  for c in class_list])
amphibia_mask = np.array([c == 'Amphibia' for c in class_list])

def apply_smooth(preds, a_aves=0.20, a_ins=0.0, a_amp=0.45):
    result = preds.copy()
    idx = 0
    for nw in lab_n_windows:
        c = preds[idx:idx+nw].copy()
        o = c.copy()
        if a_aves > 0:
            o[:, aves_mask]     = c[:, aves_mask]     + a_aves * (c[:, aves_mask].max(0) - c[:, aves_mask])
        if a_ins > 0:
            o[:, insecta_mask]  = (1-a_ins)  * c[:, insecta_mask]  + a_ins  * c[:, insecta_mask].mean(0)
        if a_amp > 0:
            o[:, amphibia_mask] = (1-a_amp) * c[:, amphibia_mask] + a_amp * c[:, amphibia_mask].mean(0)
        result[idx:idx+nw] = np.clip(o, 0, 1)
        idx += nw
    return result

# ── 評估所有組合 ───────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('RESULTS')
print('='*60)

# 各方法 standalone
steps = {
    'Baseline'          : base_probs,
    'GMM(lam=0.40)'     : apply_gmm(base_probs),
    'Co-occur(pw=0.20)' : apply_cooccur(base_probs),
    'Sibling(b=0.15)'   : apply_sibling(base_probs),
    'ClassSmooth'       : apply_smooth(base_probs),
}

# 組合（疊加在 baseline 上）
def chain(*fns):
    out = base_probs.copy()
    for fn in fns: out = fn(out)
    return out

combos = {
    'GMM→Cooc'         : chain(apply_gmm, apply_cooccur),
    'GMM→Sibling'      : chain(apply_gmm, apply_sibling),
    'GMM→Smooth'       : chain(apply_gmm, apply_smooth),
    'Cooc→Sibling'     : chain(apply_cooccur, apply_sibling),
    'Cooc→Smooth'      : chain(apply_cooccur, apply_smooth),
    'Sibling→Smooth'   : chain(apply_sibling, apply_smooth),
    'GMM→Cooc→Sib'     : chain(apply_gmm, apply_cooccur, apply_sibling),
    'GMM→Sib→Smooth'   : chain(apply_gmm, apply_sibling, apply_smooth),
    'Cooc→Sib→Smooth'  : chain(apply_cooccur, apply_sibling, apply_smooth),
    'ALL'              : chain(apply_gmm, apply_cooccur, apply_sibling, apply_smooth),
}

all_results = {**steps, **combos}
for name, preds in all_results.items():
    auc   = macro_auc(lab_labels, preds)
    delta = auc - baseline_auc
    marker = ' ←BEST' if name != 'Baseline' and auc == max(macro_auc(lab_labels, p) for p in all_results.values()) else ''
    print(f'  {name:<22} AUC={auc:.4f}  ({delta:+.4f}){marker}')

# GMM lambda sweep 在組合中
print('\n[GMM lambda sweep in GMM→Sib→Smooth]')
best_combo_auc, best_lam = 0, 0
for lam in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
    out = apply_sibling(apply_smooth(apply_gmm(base_probs, lam)))
    auc = macro_auc(lab_labels, out)
    print(f'  lam={lam:.2f} → AUC={auc:.4f} ({auc-baseline_auc:+.4f})')
    if auc > best_combo_auc: best_combo_auc, best_lam = auc, lam

print(f'\nBest GMM lam in combo: {best_lam}  AUC={best_combo_auc:.4f}')

print('\n⚠ 注意：GMM cluster profiles 用了與 eval 相同的 labeled SS windows')
print('  → GMM 的 AUC 提升有 in-sample 偏差，實際 LB 效益需要 LOO 確認')
