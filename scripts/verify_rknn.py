"""Verify Reciprocal KNN full-pipeline AUC."""
import numpy as np, pickle, os
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
with open("outputs/embed_prior_logspace_geo5_win1.pkl","rb") as f: ep=pickle.load(f)
X_ref=ep['X_combined_n'].astype(np.float32); fl=ep['file_labels'].astype(np.float32)
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

print("Reciprocal KNN verification...", flush=True)
for k in [5, 8, 10, 15]:
    y_rknn=np.zeros((n_files,n_species),np.float32)
    fallback=0
    for i in range(n_files):
        tr=np.array([j for j in range(n_files) if j!=i])
        sims_i=(X_ref[[i]]@X_ref[tr].T).ravel()
        top_i_set=set(tr[np.argsort(-sims_i)[:k]])
        mutual=[]; mutual_sims=[]
        for tj in top_i_set:
            tr2=np.array([jj for jj in range(n_files) if jj!=tj])
            sims_j=(X_ref[[tj]]@X_ref[tr2].T).ravel()
            top_j_set=set(tr2[np.argsort(-sims_j)[:k]])
            if i in top_j_set:
                mutual.append(tj); mutual_sims.append(sims_i[np.where(tr==tj)[0][0]])
        if len(mutual)==0:
            fallback+=1
            top=np.argsort(-sims_i)[:5]; ls=sims_i[top]/0.2; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
            y_rknn[i]=(w[:,None]*fl[tr[top]]).sum(0)
        else:
            ma=np.array(mutual); ms=np.array(mutual_sims)
            ls=ms/0.2; ls-=ls.max(); w=np.exp(ls); w/=w.sum()
            y_rknn[i]=(w[:,None]*fl[ma]).sum(0)
    ep_only=macro_auc(file_labels,sigmoid(0.7*file_logit_max+1.45*np.log(y_rknn.clip(EPS))))
    best_full=0; best_wg=0
    for wg in [0.4,0.5,0.6,0.7]:
        yb=wg*y_rknn+(1-wg)*y_win
        full=macro_auc(file_labels,sigmoid(0.9*base_logit+1.55*np.log(yb.clip(EPS))))
        if full>best_full: best_full=full; best_wg=wg
    marker="*** BEAT ***" if best_full>0.9408 else ""
    print(f"  k={k:2d}: EP={ep_only:.4f} best_FULL={best_full:.4f} wg={best_wg} fallback={fallback}/{n_files} {marker}")
print("Done.")
