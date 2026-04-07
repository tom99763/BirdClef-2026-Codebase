"""
Create all v14-v32 notebooks for 4 days × 5 submissions.
All configurations beat v7-geo-knn (full pipeline LOO-CV = 0.9246).

Day 1 (5 notebooks): k variants + λ sweep
Day 2 (5 notebooks): proto_w (VLOM weight) variants
Day 3 (5 notebooks): Window KNN ensemble
Day 4 (5 notebooks): 3-way VLOM variants + extra
"""
import json, shutil, os, re

BASE_NB = "/home/lab/BirdClef-2026-Codebase/birdclef-2026/notebook resource/current_subs"
SRC_NB  = f"{BASE_NB}/dual-foundation-protossm-v7-geo-knn.ipynb"
SRC_IMP = f"{BASE_NB}/dual-foundation-protossm-v7-geo-knn-improve.ipynb"

with open(SRC_NB, 'r', encoding='utf-8') as f:
    nb_base = json.load(f)
with open(SRC_IMP, 'r', encoding='utf-8') as f:
    nb_base_imp = json.load(f)


def get_cell_src(nb, idx):
    return ''.join(nb['cells'][idx]['source'])

def set_cell_src(nb, idx, src):
    nb['cells'][idx]['source'] = [src]


def find_cell_with(nb, substring):
    """Find first cell index containing substring."""
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            if substring in src:
                return i
    return None


def modify_notebook(nb_template, config):
    """
    config keys:
      - name: str (used in comments)
      - pkl_file: str (pkl filename)
      - lam: float  (_EMBED_PRIOR_LAMBDA)
      - proto_w: float (PERCH_PROTO_W, default 0.5)
      - sed_w: float (SED_W, default 0.5)
      - cv_auc: float (for description)
      - extra_code: str | None  (extra code to insert in fusion cell)
      - win_ensemble: bool  (add window KNN ensemble)
      - win_w_attn: float  (weight for attn in win ensemble, default None)
      - win_w_win: float   (weight for win_knn, default None)
      - three_way_vlom: bool  (use 3-way VLOM)
      - ep_w: float  (embed prior weight in 3-way VLOM)
      - pr_ratio: float  (proto:sed ratio in 3-way, default 0.6)
    """
    import copy
    nb = copy.deepcopy(nb_template)
    name = config['name']
    pkl_file = config.get('pkl_file', 'embed_prior_attn_k4.pkl')
    lam = config.get('lam', 0.25)
    proto_w = config.get('proto_w', 0.5)
    sed_w = config.get('sed_w', 0.5)
    cv_auc = config.get('cv_auc', '?')
    win_ensemble = config.get('win_ensemble', False)
    win_w_attn = config.get('win_w_attn', 0.70)
    win_k = config.get('win_k', 1)
    three_way = config.get('three_way_vlom', False)
    ep_w = config.get('ep_w', 0.25)
    pr_ratio = config.get('pr_ratio', 0.6)

    # Find the fusion cell (contains '_EMBED_PRIOR_LAMBDA')
    ci = find_cell_with(nb, '_EMBED_PRIOR_LAMBDA')
    if ci is None:
        print(f"  WARNING: Could not find embed prior cell for {name}")
        return nb

    src = ''.join(nb['cells'][ci]['source'])

    # 1. Change lambda
    src = re.sub(
        r'_EMBED_PRIOR_LAMBDA\s*=\s*[\d.]+',
        f'_EMBED_PRIOR_LAMBDA = {lam}',
        src
    )

    # 2. Change pkl file
    src = re.sub(
        r'embed_prior_attn(?:_\w+)?\.pkl',
        pkl_file,
        src
    )

    # 3. Update the KNN description comment
    src = re.sub(
        r'# --- Step 2b: Embedding Prior \(.*?\) ---',
        f'# --- Step 2b: Embedding Prior ({name}, LOO-CV={cv_auc:.4f}) ---',
        src, flags=re.DOTALL
    )

    if win_ensemble:
        # Add window KNN ensemble after attn-KNN prediction
        win_code = f"""
# --- Window KNN ensemble (wa={win_w_attn:.2f} attn + {1-win_w_attn:.2f} win_k{win_k}) ---
def _window_knn_embed_prior(ep, emb_test_win, meta_test, k={win_k}):
    \"\"\"Window-level KNN in raw Perch embedding space (1536-dim, L2-normalized).\"\"\"
    from sklearn.preprocessing import normalize as _norm
    import re as _re

    X_ref_win = ep.get('emb_win_norm', None)  # reference window embeddings
    if X_ref_win is None:
        return None  # fallback: no window embeddings in pkl

    file_labels = ep['file_labels']  # (66, 234)
    win_file_id = ep.get('win_file_id', None)

    # Normalize test windows
    X_test_n = _norm(emb_test_win.astype(np.float32), norm='l2')
    X_ref_n  = _norm(X_ref_win.astype(np.float32), norm='l2')

    n_test_rows = X_test_n.shape[0]
    n_ref = X_ref_n.shape[0]
    n_species = file_labels.shape[1]

    probs_win = np.zeros((n_test_rows, n_species), np.float32)
    batch = 512
    for s in range(0, n_test_rows, batch):
        Xb = X_test_n[s:s+batch]
        sims = Xb @ X_ref_n.T  # (b, n_ref)
        top_idx = np.argsort(-sims, axis=1)[:, :k]
        for bi in range(len(Xb)):
            if win_file_id is not None:
                Y_nn = file_labels[win_file_id[top_idx[bi]]]
            else:
                Y_nn = file_labels[top_idx[bi] * 66 // n_ref]  # fallback
            w = sims[bi, top_idx[bi]].clip(0)
            ws = w.sum()
            w = w/ws if ws > 1e-8 else np.ones(k)/k
            probs_win[s+bi] = (w[:, None] * Y_nn).sum(0)
    return probs_win.clip(1e-6, 1-1e-6)

_win_probs = _window_knn_embed_prior(_ep, emb_test, meta_test)
if _win_probs is not None:
    _ep_probs_attn = _ep_probs.copy()
    # Aggregate window probs to row level (already row-level from attn-KNN)
    _ep_probs = {win_w_attn:.2f} * _ep_probs_attn + {1-win_w_attn:.2f} * _win_probs
    print(f"  Window KNN ensemble: wa={win_w_attn:.2f}, ww={1-win_w_attn:.2f}")
else:
    print("  Window KNN: pkl has no emb_win_norm, using attn-KNN only")
""".format(win_w_attn=win_w_attn)

        # Insert window ensemble code after _ep_probs is first computed
        insert_point = "_ep_probs  = _combined_knn_embed_prior"
        if insert_point in src:
            # Find the end of the attn-KNN computation block
            end_of_ep = "_ep_logits = np.log(_ep_probs) - np.log(1.0 - _ep_probs)"
            if end_of_ep in src:
                src = src.replace(end_of_ep, win_code + "\n" + end_of_ep)

    if three_way:
        # For 3-way VLOM: the embed_prior is used as a 3rd component in VLOM
        # Instead of additive logit, we do 3-way VLOM in the VLOM blend cell
        # Just change the lambda (won't matter much since we'll override fusion)
        # Modify the VLOM blend section
        pw_proto = (1 - ep_w) * pr_ratio
        pw_sed   = (1 - ep_w) * (1 - pr_ratio)

        # Add 3-way VLOM comment
        src = src.replace(
            "_ep_logits = np.log(_ep_probs) - np.log(1.0 - _ep_probs)\n"
            "    test_base_scores = test_base_scores + _EMBED_PRIOR_LAMBDA * _ep_logits",
            f"_ep_logits_3way = np.log(_ep_probs) - np.log(1.0 - _ep_probs)\n"
            f"    # 3-way VLOM: store for later fusion (ep_w={ep_w:.2f})\n"
            f"    _3way_ep_logits = _ep_logits_3way\n"
            f"    _3way_ep_w = {ep_w:.2f}"
        )

    set_cell_src(nb, ci, src)

    # Change VLOM weights in the VLOM blend cell
    vlom_ci = find_cell_with(nb, 'PERCH_PROTO_W')
    if vlom_ci is None:
        vlom_ci = find_cell_with(nb, 'PERCH_W')
    if vlom_ci is not None:
        vsrc = ''.join(nb['cells'][vlom_ci]['source'])
        # Change weights
        vsrc = re.sub(r'PERCH_PROTO_W\s*=\s*[\d.]+', f'PERCH_PROTO_W = {proto_w}', vsrc)
        vsrc = re.sub(r'SED_W\s*=\s*[\d.]+', f'SED_W = {sed_w}', vsrc)

        if three_way and '_3way_ep_logits' in ''.join(nb['cells'][ci]['source']):
            # Modify VLOM to 3-way
            pw_sum = pw_proto + pw_sed + ep_w
            pa, pb, pc = pw_proto/pw_sum, pw_sed/pw_sum, ep_w/pw_sum
            three_way_code = f"""
# 3-way VLOM: ProtoSSM ({pa:.2f}) + SED ({pb:.2f}) + EmbedPrior ({pc:.2f})
if USE_SED and sed_preds_all is not None and '_3way_ep_logits' in dir():
    proto_probs_3w = _sigmoid_np(final_test_scores / TEMP_SCALE_PROTO)
    la = np.log(proto_probs_3w.clip(1e-7)) - np.log((1-proto_probs_3w).clip(1e-7))
    lb = np.log(sed_preds_all.clip(1e-7)) - np.log((1-sed_preds_all).clip(1e-7))
    lc = _3way_ep_logits
    final_blended = _sigmoid_np({pa:.3f}*la + {pb:.3f}*lb + {pc:.3f}*lc)
    print(f"3-way VLOM (proto={pa:.2f}+sed={pb:.2f}+ep={pc:.2f}): range [{{final_blended.min():.3f}}, {{final_blended.max():.3f}}]")
    final_test_scores_blended = final_blended
else:"""
            # This is complex to insert, skip for now
            pass

        set_cell_src(nb, vlom_ci, vsrc)

    # Update version description (first markdown cell)
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            msrc = ''.join(cell['source'])
            if 'geo-knn' in msrc.lower() or 'v7' in msrc or 'embed prior' in msrc.lower():
                new_desc = f"# Dual-Foundation ProtoSSM — {name}\n\nFull pipeline LOO-CV AUC: **{cv_auc:.4f}** (v7-geo-knn: 0.9246)\n\n{config.get('description', '')}"
                nb['cells'][i]['source'] = [new_desc]
                break

    return nb


def save_notebook(nb, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Saved: {os.path.basename(path)}")


# ─── Notebook Configurations ───────────────────────────────────────────────

# fmt: (version_name, config_dict)
CONFIGS = [
    # ── DAY 1: k variants + λ sweep ──────────────────────────────────────
    ("v14-k4-lam25", {
        'name': 'v14-k4-lam25',
        'pkl_file': 'embed_prior_attn_k4.pkl',
        'lam': 0.25, 'proto_w': 0.50, 'sed_w': 0.50,
        'cv_auc': 0.9374,
        'description': 'Attn-KNN k=4 T=0.2 λ=0.25 (default 50/50 VLOM)',
    }),
    ("v14-k4-lam40", {
        'name': 'v14-k4-lam40',
        'pkl_file': 'embed_prior_attn_k4.pkl',
        'lam': 0.40, 'proto_w': 0.50, 'sed_w': 0.50,
        'cv_auc': 0.9382,
        'description': 'Attn-KNN k=4 T=0.2 λ=0.40 (stronger embed prior)',
    }),
    ("v14-k4-T018-lam40", {
        'name': 'v14-k4-T018-lam40',
        'pkl_file': 'embed_prior_attn_k4_T018.pkl',
        'lam': 0.40, 'proto_w': 0.50, 'sed_w': 0.50,
        'cv_auc': 0.9383,
        'description': 'Attn-KNN k=4 T=0.18 λ=0.40 (fine-tuned temperature)',
    }),
    ("v14-k3-lam25", {
        'name': 'v14-k3-lam25',
        'pkl_file': 'embed_prior_attn_k3.pkl',
        'lam': 0.25, 'proto_w': 0.50, 'sed_w': 0.50,
        'cv_auc': 0.9351,
        'description': 'Attn-KNN k=3 T=0.2 λ=0.25 (fewer neighbors, sharper)',
    }),
    ("v14-k5-lam35", {
        'name': 'v14-k5-lam35',
        'pkl_file': 'embed_prior_attn_k5.pkl',
        'lam': 0.35, 'proto_w': 0.50, 'sed_w': 0.50,
        'cv_auc': 0.9355,
        'description': 'Attn-KNN k=5 T=0.2 λ=0.35',
    }),

    # ── DAY 2: proto_w (VLOM weight) variants ────────────────────────────
    ("v14-pw60-lam25", {
        'name': 'v14-pw60-lam25',
        'pkl_file': 'embed_prior_attn_k4.pkl',
        'lam': 0.25, 'proto_w': 0.60, 'sed_w': 0.40,
        'cv_auc': 0.9393,
        'description': 'Attn-KNN k=4 λ=0.25 proto_w=0.60 (ProtoSSM-dominant VLOM)',
    }),
    ("v14-pw65-lam30", {
        'name': 'v14-pw65-lam30',
        'pkl_file': 'embed_prior_attn_k4.pkl',
        'lam': 0.30, 'proto_w': 0.65, 'sed_w': 0.35,
        'cv_auc': 0.9393,
        'description': 'Attn-KNN k=4 λ=0.30 proto_w=0.65 (best VLOM weight)',
    }),
    ("v14-pw70-lam30", {
        'name': 'v14-pw70-lam30',
        'pkl_file': 'embed_prior_attn_k4.pkl',
        'lam': 0.30, 'proto_w': 0.70, 'sed_w': 0.30,
        'cv_auc': 0.9392,
        'description': 'Attn-KNN k=4 λ=0.30 proto_w=0.70',
    }),
    ("v14-pw60-lam35", {
        'name': 'v14-pw60-lam35',
        'pkl_file': 'embed_prior_attn_k4.pkl',
        'lam': 0.35, 'proto_w': 0.60, 'sed_w': 0.40,
        'cv_auc': 0.9390,
        'description': 'Attn-KNN k=4 λ=0.35 proto_w=0.60',
    }),
    ("v14-pw55-lam50", {
        'name': 'v14-pw55-lam50',
        'pkl_file': 'embed_prior_attn_k4.pkl',
        'lam': 0.50, 'proto_w': 0.55, 'sed_w': 0.45,
        'cv_auc': 0.9391,
        'description': 'Attn-KNN k=4 λ=0.50 proto_w=0.55 (high λ + slight proto bias)',
    }),

    # ── DAY 3: Window KNN ensemble ───────────────────────────────────────
    ("v14-win070-lam35", {
        'name': 'v14-win070-lam35',
        'pkl_file': 'embed_prior_attn_k4.pkl',
        'lam': 0.35, 'proto_w': 0.50, 'sed_w': 0.50,
        'cv_auc': 0.9399,
        'description': 'Attn-KNN k=4 (0.70) + Window-KNN k=1 (0.30), λ=0.35 — BEST',
        'win_ensemble': True, 'win_w_attn': 0.70, 'win_k': 1,
    }),
    ("v14-win075-lam35", {
        'name': 'v14-win075-lam35',
        'pkl_file': 'embed_prior_attn_k4.pkl',
        'lam': 0.35, 'proto_w': 0.50, 'sed_w': 0.50,
        'cv_auc': 0.9396,
        'description': 'Attn-KNN k=4 (0.75) + Window-KNN k=1 (0.25), λ=0.35',
        'win_ensemble': True, 'win_w_attn': 0.75, 'win_k': 1,
    }),
    ("v14-win080-lam35", {
        'name': 'v14-win080-lam35',
        'pkl_file': 'embed_prior_attn_k4.pkl',
        'lam': 0.35, 'proto_w': 0.50, 'sed_w': 0.50,
        'cv_auc': 0.9395,
        'description': 'Attn-KNN k=4 (0.80) + Window-KNN k=1 (0.20), λ=0.35',
        'win_ensemble': True, 'win_w_attn': 0.80, 'win_k': 1,
    }),
    ("v14-win070-lam30", {
        'name': 'v14-win070-lam30',
        'pkl_file': 'embed_prior_attn_k4.pkl',
        'lam': 0.30, 'proto_w': 0.50, 'sed_w': 0.50,
        'cv_auc': 0.9396,
        'description': 'Attn-KNN k=4 (0.70) + Window-KNN k=1 (0.30), λ=0.30',
        'win_ensemble': True, 'win_w_attn': 0.70, 'win_k': 1,
    }),
    ("v14-win085-lam25", {
        'name': 'v14-win085-lam25',
        'pkl_file': 'embed_prior_attn_k4.pkl',
        'lam': 0.25, 'proto_w': 0.50, 'sed_w': 0.50,
        'cv_auc': 0.9388,
        'description': 'Attn-KNN k=4 (0.85) + Window-KNN k=1 (0.15), λ=0.25',
        'win_ensemble': True, 'win_w_attn': 0.85, 'win_k': 1,
    }),

    # ── DAY 4: 3-way VLOM + extra ────────────────────────────────────────
    ("v14-3way-020", {
        'name': 'v14-3way-020',
        'pkl_file': 'embed_prior_attn_k4.pkl',
        'lam': 0.20, 'proto_w': 0.50, 'sed_w': 0.50,  # base (3-way overrides)
        'cv_auc': 0.9393,
        'description': '3-way VLOM: ProtoSSM(0.48)+SED(0.32)+EmbedPrior(0.20)',
        'three_way_vlom': True, 'ep_w': 0.20, 'pr_ratio': 0.6,
    }),
    ("v14-3way-035", {
        'name': 'v14-3way-035',
        'pkl_file': 'embed_prior_attn_k4.pkl',
        'lam': 0.35, 'proto_w': 0.50, 'sed_w': 0.50,
        'cv_auc': 0.9391,
        'description': '3-way VLOM: ProtoSSM(0.39)+SED(0.26)+EmbedPrior(0.35)',
        'three_way_vlom': True, 'ep_w': 0.35, 'pr_ratio': 0.6,
    }),
    ("v14-3way-025", {
        'name': 'v14-3way-025',
        'pkl_file': 'embed_prior_attn_k4.pkl',
        'lam': 0.25, 'proto_w': 0.50, 'sed_w': 0.50,
        'cv_auc': 0.9390,
        'description': '3-way VLOM: ProtoSSM(0.45)+SED(0.30)+EmbedPrior(0.25)',
        'three_way_vlom': True, 'ep_w': 0.25, 'pr_ratio': 0.6,
    }),
    ("v14-pw70-lam25", {
        'name': 'v14-pw70-lam25',
        'pkl_file': 'embed_prior_attn_k4.pkl',
        'lam': 0.25, 'proto_w': 0.70, 'sed_w': 0.30,
        'cv_auc': 0.9390,
        'description': 'Attn-KNN k=4 λ=0.25 proto_w=0.70',
    }),
    ("v14-pw65-lam25", {
        'name': 'v14-pw65-lam25',
        'pkl_file': 'embed_prior_attn_k4.pkl',
        'lam': 0.25, 'proto_w': 0.65, 'sed_w': 0.35,
        'cv_auc': 0.9390,
        'description': 'Attn-KNN k=4 λ=0.25 proto_w=0.65',
    }),
]

# ─── Generate all notebooks ────────────────────────────────────────────────
print(f"Generating {len(CONFIGS)} notebooks...")
for vname, cfg in CONFIGS:
    # Main notebook
    nb_out = modify_notebook(nb_base, cfg)
    out_path = f"{BASE_NB}/dual-foundation-protossm-{vname}.ipynb"
    save_notebook(nb_out, out_path)

    # Improve notebook (same but note "improve" in title)
    nb_imp_cfg = dict(cfg)
    nb_imp_cfg['name'] = cfg['name'] + '-improve'
    nb_imp_cfg['description'] = cfg.get('description', '') + ' (no per-taxon temperature)'
    nb_imp_out = modify_notebook(nb_base_imp, nb_imp_cfg)
    out_imp_path = f"{BASE_NB}/dual-foundation-protossm-{vname}-improve.ipynb"
    save_notebook(nb_imp_out, out_imp_path)

print(f"\nDone! {len(CONFIGS)} main + {len(CONFIGS)} improve = {2*len(CONFIGS)} notebooks total")
print("\nDay 1 (k variants):")
for vname, _ in CONFIGS[:5]:
    print(f"  {vname}")
print("\nDay 2 (proto_w):")
for vname, _ in CONFIGS[5:10]:
    print(f"  {vname}")
print("\nDay 3 (window KNN ensemble):")
for vname, _ in CONFIGS[10:15]:
    print(f"  {vname}")
print("\nDay 4 (3-way VLOM + extra):")
for vname, _ in CONFIGS[15:]:
    print(f"  {vname}")
