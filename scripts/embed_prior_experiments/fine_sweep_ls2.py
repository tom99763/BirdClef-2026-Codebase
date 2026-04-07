"""Fine sweep for logspace_geo5_win1 full pipeline parameters."""
import numpy as np, pickle, os
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')
os.chdir("/home/lab/BirdClef-2026-Codebase")

perch = np.load("outputs/perch_labeled_ss.npz", allow_pickle=True)
emb_win = perch['emb'].astype(np.float32); logits_win = perch['logits'].astype(np.float32)
labels_win = perch['labels'].astype(np.float32); file_list = list(perch['file_list'])
n_windows = perch['n_windows']; n_files = len(file_list); n_species = labels_win.shape[1]
file_start = np.concatenate([[0], np.cumsum(n_windows[:-1])]).astype(np.int32)
file_end = np.cumsum(n_windows).astype(np.int32)
def sigmoid(x): return 1.0/(1.0+np.exp(-np.clip(x,-88,88)))
file_labels = np.zeros((n_files,n_species),np.float32)
file_logit_max = np.zeros((n_files,n_species),np.float32)
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

with open("outputs/embed_prior_logspace_geo5_win1.pkl","rb") as f: ep=pickle.load(f)
X_ref=ep['X_combined_n'].astype(np.float32); fl=ep['file_labels'].astype(np.float32)
k_geo=ep['k_geo']; T_geo=ep['T_geo']; k_win=ep['k_win']; w_geo=ep['w_geo']

print("Recomputing LOO KNN...", flush=True)
y_geo=np.zeros((n_files,n_species),np.float32)
for i in range(n_files):
    tr=np.array([j for j in range(n_files) if j!=i])
    sims=(X_ref[[i]]@X_ref[tr].T).ravel(); top=np.argsort(-sims)[:k_geo]
    ls=sims[top]/T_geo; ls-=ls.max(); w=np.exp(ls); w/=w.sum(); y_geo[i]=(w[:,None]*fl[tr[top]]).sum(0)
y_win=np.zeros((n_files,n_species),np.float32)
for i in range(n_files):
    te_s,te_e=int(file_start[i]),int(file_end[i]); X_te=emb_win_norm[te_s:te_e]
    tr_mask=(win_file_id!=i); X_tr=emb_win_norm[tr_mask]; tr_fi=win_file_id[tr_mask]
    sims=X_te@X_tr.T; top_idx=np.argsort(-sims,1)[:,:k_win]
    wp=np.zeros((te_e-te_s,n_species),np.float32)
    for wi in range(te_e-te_s):
        ww=sims[wi,top_idx[wi]].clip(0); ws=ww.sum(); ww=ww/ws if ws>1e-8 else np.ones(k_win)/k_win
        wp[wi]=(ww[:,None]*file_labels[tr_fi[top_idx[wi]]]).sum(0)
    y_win[i]=wp.mean(0)
y_blend=w_geo*y_geo+(1-w_geo)*y_win
log_yb=np.log(y_blend.clip(EPS))
print("Done.", flush=True)

# Fine sweep method 3: sigmoid(a*base_logit + b*log(blended_knn))
print("\nFine sweep: sigmoid(a*base_logit + b*log(blended_knn))")
best={'auc':0}
for a in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]:
    for b in [1.10, 1.20, 1.30, 1.35, 1.40, 1.45, 1.50, 1.55, 1.60, 1.70, 1.80]:
        full=sigmoid(a*base_logit+b*log_yb)
        auc=macro_auc(file_labels,full)
        if auc>best['auc']:
            best={'auc':auc,'a':a,'b':b}
            print(f"  a={a:.2f} b={b:.2f}: {auc:.4f}", flush=True)

print(f"\nBest: a={best['a']:.2f}, b={best['b']:.2f}, AUC={best['auc']:.4f}")
print(f"v7-geo-knn=0.9246, v14-win070-lam35=0.9399")
