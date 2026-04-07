"""Fine-grain k-sweep v2: faster vectorized version."""
import numpy as np, json, pickle, scipy.special
import sys
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score

raw = np.load('outputs/perch_labeled_ss.npz', allow_pickle=True)
emb_win   = raw['emb'].astype(np.float32)
logits_win= raw['logits'].astype(np.float32)
labels_win= raw['labels'].astype(np.float32)
file_list = raw['file_list']
n_windows = raw['n_windows']
n_files = len(file_list); n_species = labels_win.shape[1]

file_embs = np.zeros((n_files, emb_win.shape[1]), dtype=np.float32)
file_labels = np.zeros((n_files, n_species), dtype=np.float32)
file_logit_max = np.zeros((n_files, n_species), dtype=np.float32)
file_logit_top2 = np.zeros((n_files, n_species), dtype=np.float32)
idx = 0
for fi, nw in enumerate(n_windows):
    wl = logits_win[idx:idx+nw]
    file_embs[fi] = emb_win[idx:idx+nw].mean(0)
    file_labels[fi] = (labels_win[idx:idx+nw].max(0) > 0.5).astype(np.float32)
    file_logit_max[fi] = wl.max(0)
    if nw >= 2: file_logit_top2[fi] = np.sort(wl,axis=0)[-2:].mean(0)
    else: file_logit_top2[fi] = wl.max(0)
    idx += nw

file_embs_norm = normalize(file_embs, norm='l2')
file_prob_max = scipy.special.expit(file_logit_max)
file_prob_top2 = scipy.special.expit(file_logit_top2)

def macro_auc(y_true, y_score):
    mask = y_true.sum(0) > 0
    try: return float(roc_auc_score(y_true[:, mask], y_score[:, mask], average='macro'))
    except: return float('nan')

def knn_predict(k):
    k_eff = min(k, n_files-1)
    preds = np.zeros((n_files, n_species), dtype=np.float32)
    for i in range(n_files):
        mask = np.ones(n_files, dtype=bool); mask[i]=False
        tr=file_embs_norm[mask]; te=file_embs_norm[[i]]; y_tr=file_labels[mask]
        sims=(te@tr.T).ravel()
        nn_idx=np.argpartition(-sims,k_eff)[:k_eff]
        w=np.clip(sims[nn_idx],0,None)
        if w.sum()<1e-9: w=np.ones(k_eff,dtype=np.float32)
        preds[i]=(w[:,None]*y_tr[nn_idx]).sum(0)/w.sum()
    return preds

print('Computing KNN k=1,2,3,4...')
sys.stdout.flush()
knn1=knn_predict(1); knn2=knn_predict(2); knn3=knn_predict(3); knn4=knn_predict(4)
print('Done'); sys.stdout.flush()

PREV = 0.8930

# ── Vectorized sweep: pre-stack all knn arrays, then enumerate weight vectors ──
# For 4-way search (logit + k1 + k3 + k4): 4 components, weights sum to 1
# Use coarser grid but fully exhaustive

def sweep_4way(sources, step=0.05):
    """sources: list of (n_files, n_species) arrays. Sweep all weight combos sum=1."""
    n = len(sources)
    best_auc = 0.0; best_w = None
    vals = np.arange(0.0, 1.0 + step/2, step)
    for w0 in vals:
        for w1 in vals:
            if w0+w1 > 1.0+step/2: continue
            for w2 in vals:
                if w0+w1+w2 > 1.0+step/2: continue
                w3 = round(1.0 - w0 - w1 - w2, 6)
                if abs(w3) < 1e-7: w3 = 0.0
                if w3 < 0: continue
                ens = w0*sources[0] + w1*sources[1] + w2*sources[2] + w3*sources[3]
                auc = macro_auc(file_labels, ens)
                if auc > best_auc:
                    best_auc = auc
                    best_w = [round(w0,4),round(w1,4),round(w2,4),round(w3,4)]
    return best_auc, best_w

# Coarse sweep 0.05 step
print('\n--- Coarse 4-way sweeps (step=0.05) ---'); sys.stdout.flush()

# (logit_max, knn1, knn3, knn4)
auc_134, w_134 = sweep_4way([file_prob_max, knn1, knn3, knn4], 0.05)
print(f'max+k1+k3+k4: {auc_134:.6f}  w=[al,w1,w3,w4]={w_134}'); sys.stdout.flush()

# (logit_max, knn1, knn2, knn3, knn4) - 5-way
def sweep_5way(sources, step=0.05):
    n = len(sources)
    best_auc = 0.0; best_w = None
    vals = np.arange(0.0, 1.0 + step/2, step)
    for w0 in vals:
        for w1 in vals:
            if w0+w1 > 1.0+step/2: continue
            for w2 in vals:
                if w0+w1+w2 > 1.0+step/2: continue
                for w3 in vals:
                    if w0+w1+w2+w3 > 1.0+step/2: continue
                    w4 = round(1.0-w0-w1-w2-w3, 6)
                    if abs(w4) < 1e-7: w4 = 0.0
                    if w4 < 0: continue
                    ens = w0*sources[0]+w1*sources[1]+w2*sources[2]+w3*sources[3]+w4*sources[4]
                    auc = macro_auc(file_labels, ens)
                    if auc > best_auc:
                        best_auc = auc
                        best_w = [round(w0,4),round(w1,4),round(w2,4),round(w3,4),round(w4,4)]
    return best_auc, best_w

auc_1234, w_1234 = sweep_5way([file_prob_max, knn1, knn2, knn3, knn4], 0.05)
print(f'max+k1+k2+k3+k4: {auc_1234:.6f}  w=[al,w1,w2,w3,w4]={w_1234}'); sys.stdout.flush()

# top2_logit versions
auc_t134, w_t134 = sweep_4way([file_prob_top2, knn1, knn3, knn4], 0.05)
print(f'top2+k1+k3+k4: {auc_t134:.6f}  w={w_t134}'); sys.stdout.flush()

auc_t1234, w_t1234 = sweep_5way([file_prob_top2, knn1, knn2, knn3, knn4], 0.05)
print(f'top2+k1+k2+k3+k4: {auc_t1234:.6f}  w={w_t1234}'); sys.stdout.flush()

print('\n--- Summary ---')
results = [
    ('max+k1+k3+k4', auc_134, w_134, [file_prob_max,knn1,knn3,knn4], 'max'),
    ('max+k1+k2+k3+k4', auc_1234, w_1234, [file_prob_max,knn1,knn2,knn3,knn4], 'max'),
    ('top2+k1+k3+k4', auc_t134, w_t134, [file_prob_top2,knn1,knn3,knn4], 'top2'),
    ('top2+k1+k2+k3+k4', auc_t1234, w_t1234, [file_prob_top2,knn1,knn2,knn3,knn4], 'top2'),
]
for name, auc, w, _, _ in sorted(results, key=lambda x: -x[1]):
    mark = ' *** NEW BEST ***' if auc > PREV else ''
    print(f'  {name}: {auc:.6f} (delta={auc-PREV:+.6f}){mark}  w={w}')

winner = max(results, key=lambda x: x[1])
winner_name, winner_auc, winner_w, winner_srcs, winner_agg = winner
print(f'\nWINNER: {winner_name}  AUC={winner_auc:.6f}')
sys.stdout.flush()

if winner_auc > PREV:
    print('Saving NEW BEST...')
    ens_best = sum(w*s for w,s in zip(winner_w, winner_srcs))
    verify = macro_auc(file_labels, ens_best)
    print(f'Verify AUC={verify:.6f}')

    # Fine-tune around winner: step=0.025
    print('Fine-tuning around winner (step=0.025)...'); sys.stdout.flush()
    if len(winner_srcs) == 4:
        auc_fine, w_fine = sweep_4way(winner_srcs, 0.025)
    else:
        auc_fine, w_fine = sweep_5way(winner_srcs, 0.025)
    print(f'Fine-tuned: {auc_fine:.6f}  w={w_fine}')
    if auc_fine > winner_auc:
        winner_auc = auc_fine; winner_w = w_fine
        ens_best = sum(w*s for w,s in zip(winner_w, winner_srcs))
        print(f'Improved to {winner_auc:.6f}')

    model_data = {
        'method': winner_name, 'logit_agg': winner_agg, 'loo_auc': winner_auc,
        'weights': winner_w, 'file_list': file_list.tolist(),
        'file_embs_norm': file_embs_norm, 'file_labels': file_labels,
        'file_prob_max': file_prob_max, 'file_prob_top2': file_prob_top2,
        'knn_cache': {1:knn1,2:knn2,3:knn3,4:knn4}, 'loo_preds': ens_best,
        'note': 'Saved by sweep_knn_fine2.py 2026-03-25',
    }
    with open('outputs/embed_prior_model.pkl','wb') as f: pickle.dump(model_data,f)
    print('Saved: outputs/embed_prior_model.pkl')

    with open('outputs/embed_prior_results.json') as f: rj=json.load(f)
    def cv(v):
        if isinstance(v,(float,np.float32,np.float64)): return float(v)
        if isinstance(v,(int,np.integer)): return int(v)
        return v
    rec={'method':winner_name,'loo_auc':round(winner_auc,6),'logit_agg':winner_agg,'weights':winner_w}
    rj['experiments'].append(rec)
    rj['best']={'method':winner_name,'loo_auc':round(winner_auc,6),'logit_agg':winner_agg,
                'weights':winner_w,'note':'Found by sweep_knn_fine2.py 2026-03-25; prev=0.893'}
    with open('outputs/embed_prior_results.json','w') as f: json.dump(rj,f,indent=2)
    print('Updated: outputs/embed_prior_results.json')
else:
    print(f'No improvement over {PREV:.4f}.')

print('Done.')
