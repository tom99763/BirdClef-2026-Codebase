"""Update window-ensemble notebooks to use win pkl with proper window KNN code."""
import json, re, os

BASE = "/home/lab/BirdClef-2026-Codebase/birdclef-2026/notebook resource/current_subs"

configs = {
    'v14-win070-lam35': (0.70, 0.30, 0.35),
    'v14-win075-lam35': (0.75, 0.25, 0.35),
    'v14-win080-lam35': (0.80, 0.20, 0.35),
    'v14-win070-lam30': (0.70, 0.30, 0.30),
    'v14-win085-lam25': (0.85, 0.15, 0.25),
}

WIN_KNN_FUNC = (
    "\n# --- Window-KNN helper for v14-win ensemble ---\n"
    "def _window_knn_embed_prior_v14(ep, emb_test_arr, k=1):\n"
    '    """Window-level KNN in raw 1536-dim L2-normalized Perch embedding space."""\n'
    "    emb_ref = ep.get('emb_win_norm', None)\n"
    "    if emb_ref is None:\n"
    "        print('  win-KNN: pkl missing emb_win_norm, skipped')\n"
    "        return None\n"
    "    wfi = ep['win_file_id']; fl = ep['file_labels']\n"
    "    n_cls = fl.shape[1]\n"
    "    X_te = emb_test_arr.astype(np.float32)\n"
    "    nrm = np.linalg.norm(X_te, axis=1, keepdims=True); nrm[nrm<1e-8]=1.0\n"
    "    X_te = X_te / nrm\n"
    "    X_ref = emb_ref.astype(np.float32)\n"
    "    n_te = X_te.shape[0]; out = np.zeros((n_te, n_cls), np.float32)\n"
    "    BSZ = 512\n"
    "    for s in range(0, n_te, BSZ):\n"
    "        Xb = X_te[s:s+BSZ]\n"
    "        sims = Xb @ X_ref.T\n"
    "        top = np.argsort(-sims, 1)[:, :k]\n"
    "        for bi in range(len(Xb)):\n"
    "            fids = wfi[top[bi]]; Ynn = fl[fids]\n"
    "            w = sims[bi, top[bi]].clip(0); ws = w.sum()\n"
    "            w = w/ws if ws>1e-8 else np.ones(k)/k\n"
    "            out[s+bi] = (w[:,None]*Ynn).sum(0)\n"
    "    return out.clip(1e-6, 1-1e-6)\n"
)


def update_win_nb(fpath, w_a, w_w, lam):
    with open(fpath, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        if '_EMBED_PRIOR_LAMBDA' not in src:
            continue

        # 1) Change pkl reference
        src = src.replace('embed_prior_attn_k4.pkl', 'embed_prior_attn_k4_win.pkl')

        # 2) Find insert point (just after _ep_probs is assigned)
        patterns = [
            '_ep_probs  = _combined_knn_embed_prior(_ep, emb_test, meta_test)',
            '_ep_probs = _combined_knn_embed_prior(_ep, emb_test, meta_test)',
        ]
        matched = None
        for pat in patterns:
            if pat in src:
                matched = pat; break

        if matched is None:
            print(f"  WARNING: insert point not found in {os.path.basename(fpath)}")
        else:
            ENSEMBLE_ADDON = (
                "\n_ep_probs_attn = _ep_probs.copy()\n"
                + WIN_KNN_FUNC
                + f"_win_probs = _window_knn_embed_prior_v14(_ep, emb_test, k=1)\n"
                f"if _win_probs is not None:\n"
                f"    _ep_probs = {w_a} * _ep_probs_attn + {w_w} * _win_probs\n"
                f"    print(f'  win-ens: attn={w_a}, win={w_w}')\n"
                f"else:\n"
                f"    print('  win-ens fallback: attn-only')\n"
            )
            src = src.replace(matched, matched + ENSEMBLE_ADDON)

        nb['cells'][i]['source'] = [src]
        break

    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Updated: {os.path.basename(fpath)}")


for key, (w_a, w_w, lam) in configs.items():
    for suffix in ['', '-improve']:
        fname = f"{BASE}/dual-foundation-protossm-{key}{suffix}.ipynb"
        if os.path.exists(fname):
            update_win_nb(fname, w_a, w_w, lam)
        else:
            print(f"  NOT FOUND: {fname}")

print("done")
