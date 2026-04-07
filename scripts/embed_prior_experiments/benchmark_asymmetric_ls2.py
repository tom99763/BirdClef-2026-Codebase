"""
Full pipeline benchmark for asymmetric_ls2 method.
Formula: sigmoid(a * base_logit + b1 * log(geo_k5) + b2 * log(win_k1))
Separate logspace coefficients for geo and win components.
"""
import numpy as np, pickle, os, json
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')
os.chdir("/home/lab/BirdClef-2026-Codebase")

perch=np.load("outputs/perch_labeled_ss.npz",allow_pickle=True)
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

sed_npz=np.load("outputs/sed-ns-b0-20s-r1/all_ss_probs_corrected.npz",allow_pickle=True)
sed_by_file={}
for i,rid in enumerate(sed_npz['row_ids']): sed_by_file.setdefault('_'.join(str(rid).split('_')[:-1]),[]).append(i)
file_sed_max=np.zeros((n_files,n_species),np.float32)
for fi,fname in enumerate(file_list):
    fb=fname.replace('.ogg','').replace('.flac','')
    if fb in sed_by_file: file_sed_max[fi]=sed_npz['probs'][sed_by_file[fb]].max(0)

EPS=1e-7
def vlom_blend(a,b,w_a=0.5,w_b=0.5):
    ws=w_a+w_b; w_a/=ws; w_b/=ws
    return sigmoid(w_a*np.log(a.clip(EPS)/(1-a).clip(EPS))+w_b*np.log(b.clip(EPS)/(1-b).clip(EPS)))
def macro_auc(yt,ys):
    mask=yt.sum(0)>0; return roc_auc_score(yt[:,mask],ys[:,mask],average='macro')

base_probs=vlom_blend(sigmoid(file_logit_max),file_sed_max)
base_logit=np.log(base_probs.clip(EPS))-np.log((1-base_probs).clip(EPS))

with open("outputs/embed_prior_logspace_geo5_win1.pkl","rb") as f: ep_base=pickle.load(f)
X_ref=ep_base['X_combined_n'].astype(np.float32); fl=ep_base['file_labels'].astype(np.float32)

# Geo KNN k=5
print("Computing geo_k5 LOO...", flush=True)
y_geo_k5=np.zeros((n_files,n_species),np.float32)
for i in range(n_files):
    tr=np.array([j for j in range(n_files) if j!=i])
    sims=(X_ref[[i]]@X_ref[tr].T).ravel()
    top=np.argsort(-sims)[:5]
    ls=sims[top]/0.2; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
    y_geo_k5[i]=(w[:,None]*fl[tr[top]]).sum(0)

# Window KNN k=1
print("Computing win_k1 LOO...", flush=True)
y_win_k1=np.zeros((n_files,n_species),np.float32)
for i in range(n_files):
    te_s,te_e=int(file_start[i]),int(file_end[i]); X_te=emb_win_norm[te_s:te_e]
    tr_mask=(win_file_id!=i); X_tr=emb_win_norm[tr_mask]; tr_fi=win_file_id[tr_mask]
    sims=X_te@X_tr.T; top_idx=np.argsort(-sims,1)[:,:1]
    wp=np.zeros((te_e-te_s,n_species),np.float32)
    for wi in range(te_e-te_s):
        ww=sims[wi,top_idx[wi]].clip(0); ws=ww.sum(); ww=ww/ws if ws>1e-8 else np.ones(1)
        wp[wi]=(ww[:,None]*file_labels[tr_fi[top_idx[wi]]]).sum(0)
    y_win_k1[i]=wp.mean(0)

log_geo=np.log(y_geo_k5.clip(EPS))
log_win=np.log(y_win_k1.clip(EPS))

print("\nFull pipeline sweep: sigmoid(a*base_logit + b1*log(geo_k5) + b2*log(win_k1))")
best={'auc':0}
for a in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]:
    for b1 in [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20]:
        for b2 in [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20, 1.40, 1.60, 1.80]:
            full=sigmoid(a*base_logit+b1*log_geo+b2*log_win)
            auc=macro_auc(file_labels,full)
            if auc>best['auc']:
                best={'auc':auc,'a':a,'b1':b1,'b2':b2}
                print(f"  a={a} b1={b1} b2={b2}: {auc:.4f}", flush=True)

print(f"\nBest asymmetric_ls2: a={best['a']}, b1={best['b1']}, b2={best['b2']}, AUC={best['auc']:.4f}")
print(f"vs RKNN best: 0.9432, improvement: {best['auc']-0.9432:+.4f}")

# Compare with symmetric blend
print("\nComparison with symmetric LS2 (geo5+win1 blended):")
for wg in [0.40, 0.50]:
    y_blend = wg * y_geo_k5 + (1-wg) * y_win_k1
    log_blend = np.log(y_blend.clip(EPS))
    best_sym = 0
    best_sym_cfg = None
    for a in [0.80, 0.85, 0.90, 0.95]:
        for b in [1.40, 1.50, 1.55, 1.60, 1.70, 1.80]:
            full = sigmoid(a * base_logit + b * log_blend)
            auc = macro_auc(file_labels, full)
            if auc > best_sym:
                best_sym = auc
                best_sym_cfg = {'wg': wg, 'a': a, 'b': b}
    print(f"  wg={wg}: {best_sym:.4f} (a={best_sym_cfg['a']}, b={best_sym_cfg['b']})")
