"""Verify co-occurrence LOO correctness: leaked vs proper."""
import numpy as np, pickle, os
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')
os.chdir("/home/lab/BirdClef-2026-Codebase")

perch = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win=perch['emb'].astype(np.float32); logits_win=perch['logits'].astype(np.float32)
labels_win=perch['labels'].astype(np.float32); file_list=list(perch['file_list'])
n_windows=perch['n_windows']; n_files=len(file_list); n_species=labels_win.shape[1]
file_start=np.concatenate([[0],np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end=np.cumsum(n_windows).astype(np.int32)
def sigmoid(x): return 1./(1.+np.exp(-np.clip(x,-88,88)))
file_labels=np.zeros((n_files,n_species),np.float32)
file_logit_max=np.zeros((n_files,n_species),np.float32)
for fi in range(n_files):
    s,e=int(file_start[fi]),int(file_end[fi])
    file_labels[fi]=(labels_win[s:e].max(0)>0.5).astype(np.float32)
    file_logit_max[fi]=logits_win[s:e].max(0)
emb_win_norm=normalize(emb_win,norm='l2').astype(np.float32)
win_file_id=np.zeros(len(emb_win),np.int32)
for fi in range(n_files): win_file_id[int(file_start[fi]):int(file_end[fi])]=fi
EPS=1e-7
def macro_auc(yt,ys):
    mask=yt.sum(0)>0; return roc_auc_score(yt[:,mask],ys[:,mask],average='macro')

with open("outputs/embed_prior_logspace_geo5_win1.pkl","rb") as f: ep=pickle.load(f)
X_ref=ep['X_combined_n'].astype(np.float32); fl=ep['file_labels'].astype(np.float32)

print("Computing LOO geo+win...", flush=True)
y_geo=np.zeros((n_files,n_species),np.float32)
for i in range(n_files):
    tr=np.array([j for j in range(n_files) if j!=i])
    sims=(X_ref[[i]]@X_ref[tr].T).ravel(); top=np.argsort(-sims)[:5]
    ls=sims[top]/0.2; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
    y_geo[i]=(w[:,None]*fl[tr[top]]).sum(0)
y_win=np.zeros((n_files,n_species),np.float32)
for i in range(n_files):
    te_s,te_e=int(file_start[i]),int(file_end[i]); X_te=emb_win_norm[te_s:te_e]
    tr_mask=(win_file_id!=i); X_tr=emb_win_norm[tr_mask]; tr_fi=win_file_id[tr_mask]
    sims=X_te@X_tr.T; top_idx=np.argsort(-sims,1)[:,:1]
    wp=np.zeros((te_e-te_s,n_species),np.float32)
    for wi in range(te_e-te_s):
        ww=sims[wi,top_idx[wi]].clip(0); ws=ww.sum(); ww=ww/ws if ws>1e-8 else np.ones(1)
        wp[wi]=(ww[:,None]*file_labels[tr_fi[top_idx[wi]]]).sum(0)
    y_win[i]=wp.mean(0)
y_blend=0.5*y_geo+0.5*y_win
ls2_ep=macro_auc(file_labels,sigmoid(0.7*file_logit_max+1.45*np.log(y_blend.clip(EPS))))
print(f"LS2 EP-only: {ls2_ep:.4f}")

# Leaked (batch3 method - uses all 66 files for co-occurrence)
print("\n--- LEAKED (incorrect LOO) ---")
C_all=(file_labels.T@file_labels)/(n_files+EPS); np.fill_diagonal(C_all,0)
row_sum_all=file_labels.sum(0)+EPS
C_cond_all=C_all/row_sum_all[:,None]
for alpha in [0.1,0.2,0.3,0.5]:
    y_c=y_blend+alpha*(y_blend@C_cond_all)
    auc=macro_auc(file_labels,sigmoid(0.7*file_logit_max+1.45*np.log(y_c.clip(EPS))))
    print(f"  alpha={alpha}: {auc:.4f}")

# Proper LOO (exclude test file from co-occurrence matrix)
print("\n--- PROPER LOO (correct) ---")
for alpha in [0.1,0.2,0.3,0.5]:
    preds=np.zeros((n_files,n_species),np.float32)
    for i in range(n_files):
        tr=np.array([j for j in range(n_files) if j!=i])
        C_tr=(fl[tr].T@fl[tr])/(len(tr)+EPS); np.fill_diagonal(C_tr,0)
        C_cond_tr=C_tr/(fl[tr].sum(0)+EPS)[:,None]
        y_b=y_blend[i]
        preds[i]=(y_b+alpha*(y_b@C_cond_tr)).clip(EPS,1-EPS)
    auc=macro_auc(file_labels,sigmoid(0.7*file_logit_max+1.45*np.log(preds.clip(EPS))))
    print(f"  alpha={alpha}: {auc:.4f}")

# Also test: proper LOO co-occurrence as standalone EP signal
print("\n--- PROPER LOO co-occurrence as EP (standalone, no logit_max) ---")
for alpha in [0.2, 0.5, 1.0, 2.0]:
    preds=np.zeros((n_files,n_species),np.float32)
    for i in range(n_files):
        tr=np.array([j for j in range(n_files) if j!=i])
        C_tr=(fl[tr].T@fl[tr])/(len(tr)+EPS); np.fill_diagonal(C_tr,0)
        C_cond_tr=C_tr/(fl[tr].sum(0)+EPS)[:,None]
        y_b=y_blend[i]
        preds[i]=(y_b+alpha*(y_b@C_cond_tr)).clip(EPS,1-EPS)
    auc=macro_auc(file_labels,preds)
    print(f"  standalone alpha={alpha}: {auc:.4f}")

print("\nDone.")
