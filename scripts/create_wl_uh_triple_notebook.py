"""Create WL-UH-triple notebook from v6 template."""
import json, copy, os
os.chdir("/home/lab/BirdClef-2026-Codebase")

nb = json.load(open('birdclef-2026/notebook resource/current_subs/dual-foundation-protossm-v6-ica90-std80-maxmean.ipynb'))
nb_new = copy.deepcopy(nb)
cells = nb_new['cells']

c50 = cells[50]
src = ''.join(c50['source'])

step2b_start = src.find('# --- Step 2b:')
step3_start = src.find('# --- Step 3:')

new_section = '''# --- Step 2b: WL-ICA-100(uh) + Std-PCA-80 + PCA-80 Triple Blend (LOO-AUC=0.9873) ─────
# Key: uses labels_win (window-level binary labels) for precise prototype construction
# ICA-100: k_neg=50, wma=0.92, wmp=0.80
# Triple blend: w_ica100=0.655, w_std=0.225, w_pca80=0.120
_EMBED_PRIOR_LAMBDA = 0.25

def _wl_component_contrast(te, tr_wins, labels_win_tr, k_neg, w_max_pos, w_max_agg):
    """Window-level label contrast for one embedding component."""
    EPS = 1e-8
    n_species = labels_win_tr.shape[1]
    ws = np.zeros((len(te), n_species), np.float32)
    for si in range(n_species):
        pos_win_mask = labels_win_tr[:, si] > 0.5
        neg_win_mask = labels_win_tr[:, si] < 0.1
        if not pos_win_mask.any(): ws[:, si] = 0.5; continue
        pos_wins = tr_wins[pos_win_mask]
        pos_sims = te @ pos_wins.T
        pp_mean = pos_wins.mean(0); pp_mean /= (np.linalg.norm(pp_mean) + EPS)
        sp = w_max_pos * pos_sims.max(1) + (1 - w_max_pos) * (te @ pp_mean)
        if neg_win_mask.any():
            neg_wins = tr_wins[neg_win_mask]
            neg_sims = te @ neg_wins.T
            k_act = min(k_neg, neg_sims.shape[1])
            top_neg = neg_wins[np.argsort(-neg_sims, axis=1)[:, :k_act]].mean(1)
            top_neg /= (np.linalg.norm(top_neg, axis=1, keepdims=True) + EPS)
            ws[:, si] = (sp - (te * top_neg).sum(1) + 1) / 2
        else:
            ws[:, si] = (sp + 1) / 2
    return w_max_agg * ws.max(0) + (1 - w_max_agg) * ws.mean(0)

def _wl_ica100_uh_triple_contrast(test_emb_raw, ep_model):
    """WL-ICA-100(uh) + Std-PCA-80 + PCA-80 triple blend. LOO-AUC=0.9873."""
    EPS = 1e-8
    pca = ep_model["pca"]
    ica = ep_model["ica"]
    scaler = ep_model["scaler"]
    pca_std = ep_model["pca_std"]
    tr_pca = ep_model["emb_win_pca_norm"]
    tr_ica = ep_model["emb_win_ica_norm"]
    tr_std = ep_model["emb_win_std_norm"]
    labels_win_tr = ep_model["labels_win"]
    config = ep_model["config"]

    w_ica  = config["w_ica100"]
    w_std  = config["w_std"]
    w_base = config["w_pca80"]

    k_neg_pca = config["pca80"]["k_neg"];        wma_pca = config["pca80"]["w_max_agg"];     wmp_pca = config["pca80"]["w_max_pos"]
    k_neg_std = config["std_pca80"]["k_neg"];    wma_std = config["std_pca80"]["w_max_agg"]; wmp_std = config["std_pca80"]["w_max_pos"]
    k_neg_ica = config["ica100"]["k_neg"];       wma_ica = config["ica100"]["w_max_agg"];    wmp_ica = config["ica100"]["w_max_pos"]

    te_pca = pca.transform(test_emb_raw).astype(np.float32)
    te_pca /= (np.linalg.norm(te_pca, axis=1, keepdims=True) + EPS)

    te_ica = ica.transform(test_emb_raw).astype(np.float32)
    te_ica /= (np.linalg.norm(te_ica, axis=1, keepdims=True) + EPS)

    te_std_raw = scaler.transform(test_emb_raw).astype(np.float32)
    te_std = pca_std.transform(te_std_raw).astype(np.float32)
    te_std /= (np.linalg.norm(te_std, axis=1, keepdims=True) + EPS)

    s_base = _wl_component_contrast(te_pca, tr_pca, labels_win_tr, k_neg_pca, wmp_pca, wma_pca)
    s_ica  = _wl_component_contrast(te_ica, tr_ica, labels_win_tr, k_neg_ica, wmp_ica, wma_ica)
    s_std  = _wl_component_contrast(te_std, tr_std, labels_win_tr, k_neg_std, wmp_std, wma_std)

    return w_ica * s_ica + w_std * s_std + w_base * s_base

import pickle, pathlib
_ep_path = pathlib.Path("/kaggle/input/birdclef-embed-prior/embed_prior_model.pkl")
if not _ep_path.exists():
    _ep_path = pathlib.Path("/kaggle/input/datasets/tom99763/birdclef2026-claude/weights/weights/embed_prior_model.pkl")
if not _ep_path.exists():
    _ep_path = pathlib.Path("outputs/embed_prior_model.pkl")

_ep_model = None
if _ep_path.exists():
    with open(_ep_path, "rb") as f:
        _ep_model = pickle.load(f)
    print(f"[EmbedPrior] Loaded: {_ep_model.get('method', 'unknown')}  LOO={_ep_model.get('loo_auc', 0):.4f}")
else:
    print("[EmbedPrior] WARNING: model not found, skipping prior")

if _ep_model is not None:
    prior_delta = np.zeros_like(test_base_scores)
    emb_test_files_ep, test_file_list_ep = reshape_to_files(emb_test, meta_test)
    rows_per_file = emb_test_files_ep.shape[1]

    for fi in range(len(test_file_list_ep)):
        file_emb = emb_test_files_ep[fi]
        prior_score = _wl_ica100_uh_triple_contrast(file_emb, _ep_model)
        prior_logit = np.log(prior_score + 1e-6) - np.log(1.0 - prior_score + 1e-6)
        row_start = fi * rows_per_file
        row_end   = row_start + rows_per_file
        prior_delta[row_start:row_end] = prior_logit[None, :]

    test_base_scores = test_base_scores + _EMBED_PRIOR_LAMBDA * prior_delta
    print(f"[EmbedPrior] WL-ICA-100(uh)+Std-PCA-80+PCA-80 applied (LOO=0.9873): lambda={_EMBED_PRIOR_LAMBDA}")
else:
    print("[EmbedPrior] Skipped (model not found)")

'''

new_src = src[:step2b_start] + new_section + src[step3_start:]
cells[50]['source'] = new_src.splitlines(True)

out_path = 'birdclef-2026/notebook resource/current_subs/dual-foundation-protossm-v6-wl-ica100-uh-triple.ipynb'
with open(out_path, 'w') as f:
    json.dump(nb_new, f, indent=1)
print(f"Saved: {out_path}")

# Verify
nb_check = json.load(open(out_path))
c50_check = nb_check['cells'][50]
src_check = ''.join(c50_check['source'])
print(f"Cell 50 len: {len(src_check)}")
print(f"Contains _wl_ica100_uh_triple_contrast: {'_wl_ica100_uh_triple_contrast' in src_check}")
print(f"Contains labels_win_tr: {'labels_win_tr' in src_check}")
print(f"Total cells: {len(nb_check['cells'])}")
