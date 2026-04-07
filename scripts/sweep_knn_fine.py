"""Fine-grain k-sweep: 4-way (k=1,3,4) and 4-way (k=1,2,3,4) combos."""
import numpy as np, json, pickle, scipy.special
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
knn1=knn_predict(1); knn2=knn_predict(2); knn3=knn_predict(3); knn4=knn_predict(4)
print('Done')

PREV = 0.8930

# Ultra-fine sweep 3-way k=1,3,4 + max_logit
print('\nUltra-fine k134+max_logit (0.01 step):')
best_b=0.0; best_cfg_b={}
for al in np.arange(0.25, 0.50, 0.01):
    for w1 in np.arange(0.05, 0.35, 0.01):
        for w3 in np.arange(0.05, 0.25, 0.01):
            w4 = round(1 - al - w1 - w3, 6)
            if w4 < 0: continue
            ens = al*file_prob_max + w1*knn1 + w3*knn3 + w4*knn4
            auc = macro_auc(file_labels, ens)
            if auc > best_b:
                best_b=auc; best_cfg_b={'al':round(float(al),4),'w1':round(float(w1),4),'w3':round(float(w3),4),'w4':round(float(w4),4)}
print(f'k134+max: {best_b:.6f}  {best_cfg_b}')

# Fine 4-way k=1,2,3,4 + max_logit
print('\nFine k1234+max_logit (0.025 step):')
best_c=0.0; best_cfg_c={}
for al in np.arange(0.20, 0.51, 0.025):
    for w1 in np.arange(0.0, 0.41, 0.025):
        for w2 in np.arange(0.0, 0.31, 0.025):
            for w3 in np.arange(0.0, 0.31, 0.025):
                w4 = round(1-al-w1-w2-w3, 6)
                if w4 < 0: continue
                ens = al*file_prob_max + w1*knn1 + w2*knn2 + w3*knn3 + w4*knn4
                auc = macro_auc(file_labels, ens)
                if auc > best_c:
                    best_c=auc; best_cfg_c={'al':round(float(al),4),'w1':round(float(w1),4),'w2':round(float(w2),4),'w3':round(float(w3),4),'w4':round(float(w4),4)}
print(f'k1234+max: {best_c:.6f}  {best_cfg_c}')

# Fine 4-way k=1,2,3,4 + top2_logit
print('\nFine k1234+top2_logit (0.025 step):')
best_d=0.0; best_cfg_d={}
for al in np.arange(0.20, 0.51, 0.025):
    for w1 in np.arange(0.0, 0.41, 0.025):
        for w2 in np.arange(0.0, 0.31, 0.025):
            for w3 in np.arange(0.0, 0.31, 0.025):
                w4 = round(1-al-w1-w2-w3, 6)
                if w4 < 0: continue
                ens = al*file_prob_top2 + w1*knn1 + w2*knn2 + w3*knn3 + w4*knn4
                auc = macro_auc(file_labels, ens)
                if auc > best_d:
                    best_d=auc; best_cfg_d={'al':round(float(al),4),'w1':round(float(w1),4),'w2':round(float(w2),4),'w3':round(float(w3),4),'w4':round(float(w4),4)}
print(f'k1234+top2: {best_d:.6f}  {best_cfg_d}')

print()
for name, auc, cfg in [('k134+max',best_b,best_cfg_b),('k1234+max',best_c,best_cfg_c),('k1234+top2',best_d,best_cfg_d)]:
    mark = ' *** NEW BEST ***' if auc > PREV else ''
    print(f'  {name}: {auc:.6f} (delta={auc-PREV:+.6f}){mark}  cfg={cfg}')

all_c = [(best_b,best_cfg_b,'k134_fine_max','max'),(best_c,best_cfg_c,'k1234_fine_max','max'),(best_d,best_cfg_d,'k1234_fine_top2','top2')]
winner_auc,winner_cfg,winner_name,winner_agg = max(all_c, key=lambda x:x[0])
print(f'\nWINNER: {winner_name}  AUC={winner_auc:.6f}')

if winner_auc > PREV:
    print('Saving NEW BEST...')
    agg_prob = file_prob_max if winner_agg=='max' else file_prob_top2
    ens_best = winner_cfg['al'] * agg_prob
    knn_map = {1:knn1,2:knn2,3:knn3,4:knn4}
    for key,k_idx in [('w1',1),('w2',2),('w3',3),('w4',4)]:
        if key in winner_cfg: ens_best = ens_best + winner_cfg[key] * knn_map[k_idx]
    verify = macro_auc(file_labels, ens_best)
    print(f'Verify AUC={verify:.6f}')

    model_data = {'config':winner_cfg,'method':winner_name,'logit_agg':winner_agg,'loo_auc':winner_auc,
                  'file_list':file_list.tolist(),'file_embs_norm':file_embs_norm,'file_labels':file_labels,
                  'file_prob_max':file_prob_max,'file_prob_top2':file_prob_top2,
                  'knn_cache':{1:knn1,2:knn2,3:knn3,4:knn4},'loo_preds':ens_best,
                  'note':'Saved by sweep_knn_fine.py 2026-03-25'}
    with open('outputs/embed_prior_model.pkl','wb') as f: pickle.dump(model_data,f)
    print('Saved: outputs/embed_prior_model.pkl')

    with open('outputs/embed_prior_results.json') as f: rj=json.load(f)
    def cv(v):
        if isinstance(v,(float,np.float32,np.float64)): return float(v)
        if isinstance(v,(int,np.integer)): return int(v)
        return v
    rec={'method':winner_name,'loo_auc':round(winner_auc,6),'logit_agg':winner_agg}
    rec.update({k:cv(v) for k,v in winner_cfg.items()})
    rj['experiments'].append(rec)
    rj['best']={'method':winner_name,'loo_auc':round(winner_auc,6),'logit_agg':winner_agg,
                'config':{k:cv(v) for k,v in winner_cfg.items()},
                'note':'Found by sweep_knn_fine.py 2026-03-25; prev=4way_knn_logit=0.893'}
    with open('outputs/embed_prior_results.json','w') as f: json.dump(rj,f,indent=2)
    print('Updated: outputs/embed_prior_results.json')
else:
    print(f'No improvement over {PREV:.4f}.')

print('Done.')
