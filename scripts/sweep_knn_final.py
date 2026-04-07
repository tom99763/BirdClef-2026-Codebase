"""Final k-sweep: fine-grain around the best found config (k=1,3,4 + max_logit AUC=0.893809)."""
import numpy as np, json, pickle, scipy.special, sys
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

print('Computing KNN k=1,2,3,4,5...'); sys.stdout.flush()
knn1=knn_predict(1); knn2=knn_predict(2); knn3=knn_predict(3)
knn4=knn_predict(4); knn5=knn_predict(5)
print('Done'); sys.stdout.flush()

PREV = 0.8930

# ── Fine sweep around (al=0.35, w1=0.15, w3=0.10, w4=0.40) ──────────────────
# Step=0.025
print('\nFine sweep k134+max_logit (step=0.025)...'); sys.stdout.flush()
best=0.0; best_cfg={}
for al in np.arange(0.225, 0.475, 0.025):
    for w1 in np.arange(0.0, 0.375, 0.025):
        for w3 in np.arange(0.0, 0.275, 0.025):
            w4 = round(1-al-w1-w3, 6)
            if w4 < -1e-7: continue
            w4 = max(w4, 0.0)
            ens = al*file_prob_max + w1*knn1 + w3*knn3 + w4*knn4
            auc = macro_auc(file_labels, ens)
            if auc > best:
                best=auc; best_cfg={'al':round(float(al),4),'w1':round(float(w1),4),'w3':round(float(w3),4),'w4':round(float(w4),4)}
print(f'k134+max: {best:.6f}  {best_cfg}'); sys.stdout.flush()

# ── Also try top2 version ──────────────────────────────────────────────────────
print('Fine sweep k134+top2_logit (step=0.025)...'); sys.stdout.flush()
best_t=0.0; best_cfg_t={}
for al in np.arange(0.225, 0.475, 0.025):
    for w1 in np.arange(0.0, 0.375, 0.025):
        for w3 in np.arange(0.0, 0.275, 0.025):
            w4 = round(1-al-w1-w3, 6)
            if w4 < -1e-7: continue
            w4 = max(w4, 0.0)
            ens = al*file_prob_top2 + w1*knn1 + w3*knn3 + w4*knn4
            auc = macro_auc(file_labels, ens)
            if auc > best_t:
                best_t=auc; best_cfg_t={'al':round(float(al),4),'w1':round(float(w1),4),'w3':round(float(w3),4),'w4':round(float(w4),4)}
print(f'k134+top2: {best_t:.6f}  {best_cfg_t}'); sys.stdout.flush()

# ── 4-way sweep k=1,2,4 (skip k3) ────────────────────────────────────────────
print('Fine sweep k124+max_logit (step=0.05)...'); sys.stdout.flush()
best_124=0.0; best_cfg_124={}
for al in np.arange(0.20, 0.51, 0.05):
    for w1 in np.arange(0.0, 0.41, 0.05):
        for w2 in np.arange(0.0, 0.31, 0.05):
            w4 = round(1-al-w1-w2, 6)
            if w4 < -1e-7: continue
            w4 = max(w4, 0.0)
            ens = al*file_prob_max + w1*knn1 + w2*knn2 + w4*knn4
            auc = macro_auc(file_labels, ens)
            if auc > best_124:
                best_124=auc; best_cfg_124={'al':round(float(al),4),'w1':round(float(w1),4),'w2':round(float(w2),4),'w4':round(float(w4),4)}
print(f'k124+max: {best_124:.6f}  {best_cfg_124}'); sys.stdout.flush()

# ── Also: 4-way (k=1,4,5) ─────────────────────────────────────────────────────
print('Fine sweep k145+max_logit (step=0.05)...'); sys.stdout.flush()
best_145=0.0; best_cfg_145={}
for al in np.arange(0.20, 0.51, 0.05):
    for w1 in np.arange(0.0, 0.41, 0.05):
        for w4 in np.arange(0.0, 0.51, 0.05):
            w5 = round(1-al-w1-w4, 6)
            if w5 < -1e-7: continue
            w5 = max(w5, 0.0)
            ens = al*file_prob_max + w1*knn1 + w4*knn4 + w5*knn5
            auc = macro_auc(file_labels, ens)
            if auc > best_145:
                best_145=auc; best_cfg_145={'al':round(float(al),4),'w1':round(float(w1),4),'w4':round(float(w4),4),'w5':round(float(w5),4)}
print(f'k145+max: {best_145:.6f}  {best_cfg_145}'); sys.stdout.flush()

# ── Summary ───────────────────────────────────────────────────────────────────
print('\n--- SUMMARY ---')
all_r = [
    ('k134+max(fine)', best, best_cfg, [file_prob_max,knn1,knn3,knn4]),
    ('k134+top2(fine)', best_t, best_cfg_t, [file_prob_top2,knn1,knn3,knn4]),
    ('k124+max', best_124, best_cfg_124, [file_prob_max,knn1,knn2,knn4]),
    ('k145+max', best_145, best_cfg_145, [file_prob_max,knn1,knn4,knn5]),
    # Coarse result from earlier
    ('k134+max(coarse)', 0.893809, {'al':0.35,'w1':0.15,'w3':0.10,'w4':0.40}, None),
]
for name, auc, cfg, _ in sorted(all_r, key=lambda x:-x[1]):
    mark = ' *** NEW BEST ***' if auc > PREV else ''
    print(f'  {name}: {auc:.6f} (delta={auc-PREV:+.6f}){mark}  {cfg}')
sys.stdout.flush()

winner_name, winner_auc, winner_cfg, winner_srcs = max(all_r[:-1], key=lambda x: x[1])
print(f'\nWINNER: {winner_name}  AUC={winner_auc:.6f}')

if winner_auc > PREV:
    w_vals = list(winner_cfg.values())
    ens_best = sum(w*s for w,s in zip(w_vals, winner_srcs))
    verify = macro_auc(file_labels, ens_best)
    print(f'Verify AUC={verify:.6f}')

    model_data = {
        'method': winner_name, 'loo_auc': winner_auc,
        'config': winner_cfg, 'weights_order': list(winner_cfg.keys()),
        'file_list': file_list.tolist(), 'file_embs_norm': file_embs_norm,
        'file_labels': file_labels, 'file_prob_max': file_prob_max,
        'file_prob_top2': file_prob_top2,
        'knn_cache': {1:knn1,2:knn2,3:knn3,4:knn4,5:knn5},
        'loo_preds': ens_best,
        'note': 'Saved by sweep_knn_final.py 2026-03-25',
    }
    with open('outputs/embed_prior_model.pkl','wb') as f: pickle.dump(model_data,f)
    print('Saved: outputs/embed_prior_model.pkl')

    with open('outputs/embed_prior_results.json') as f: rj=json.load(f)
    def cv(v):
        if isinstance(v,(float,np.float32,np.float64)): return float(v)
        if isinstance(v,(int,np.integer)): return int(v)
        return v
    rec={'method':winner_name,'loo_auc':round(winner_auc,6)}
    rec.update({k:cv(v) for k,v in winner_cfg.items()})
    rj['experiments'].append(rec)
    rj['best']={'method':winner_name,'loo_auc':round(winner_auc,6),
                'config':{k:cv(v) for k,v in winner_cfg.items()},
                'note':'Found by sweep_knn_final.py 2026-03-25; prev=4way_knn_logit=0.893'}
    with open('outputs/embed_prior_results.json','w') as f: json.dump(rj,f,indent=2)
    print('Updated: outputs/embed_prior_results.json')
else:
    print(f'No improvement. Best coarse: 0.893809')
    # Still save the coarse best if it beats PREV
    coarse_best_auc = 0.893809
    coarse_best_cfg = {'al':0.35,'w1':0.15,'w3':0.10,'w4':0.40}
    if coarse_best_auc > PREV:
        print('Saving coarse best (0.893809) instead...')
        ens_best = coarse_best_cfg['al']*file_prob_max + coarse_best_cfg['w1']*knn1 + coarse_best_cfg['w3']*knn3 + coarse_best_cfg['w4']*knn4
        verify = macro_auc(file_labels, ens_best)
        print(f'Verify AUC={verify:.6f}')
        model_data = {
            'method': 'k134_coarse_max', 'loo_auc': coarse_best_auc,
            'config': coarse_best_cfg,
            'file_list': file_list.tolist(), 'file_embs_norm': file_embs_norm,
            'file_labels': file_labels, 'file_prob_max': file_prob_max,
            'file_prob_top2': file_prob_top2,
            'knn_cache': {1:knn1,2:knn2,3:knn3,4:knn4,5:knn5},
            'loo_preds': ens_best,
            'note': 'Saved by sweep_knn_final.py 2026-03-25 (coarse best)',
        }
        with open('outputs/embed_prior_model.pkl','wb') as f: pickle.dump(model_data,f)
        print('Saved: outputs/embed_prior_model.pkl')

        with open('outputs/embed_prior_results.json') as f: rj=json.load(f)
        def cv(v):
            if isinstance(v,(float,np.float32,np.float64)): return float(v)
            if isinstance(v,(int,np.integer)): return int(v)
            return v
        rec={'method':'k134_coarse_max','loo_auc':round(coarse_best_auc,6)}
        rec.update({k:cv(v) for k,v in coarse_best_cfg.items()})
        rj['experiments'].append(rec)
        rj['best']={'method':'k134_coarse_max','loo_auc':round(coarse_best_auc,6),
                    'config':{k:cv(v) for k,v in coarse_best_cfg.items()},
                    'note':'Found by coarse sweep 2026-03-25; prev=4way_knn_logit=0.893'}
        with open('outputs/embed_prior_results.json','w') as f: json.dump(rj,f,indent=2)
        print('Updated: outputs/embed_prior_results.json')

print('Done.')
