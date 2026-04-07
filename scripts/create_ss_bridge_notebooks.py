"""
Create SS Bridge notebooks.
Best config: alpha=0.4, wg=0.4, a=1.0, b=1.5, AUC=0.9440
"""
import json, os, copy, shutil
os.chdir("/home/lab/BirdClef-2026-Codebase")

BASE_NB = "birdclef-2026/notebook resource/current_subs/dual-foundation-protossm-rknn-wg040-a095-b170.ipynb"
OUT_DIR = "birdclef-2026/notebook resource/current_subs"

with open(BASE_NB) as f:
    base_nb = json.load(f)

# ── Best config ────────────────────────────────────────────────────────────────
CONFIGS = [
    # (name_suffix, alpha, wg, a, b, cv_auc)
    ("ss-bridge-a020-wg040-a085-b190", 0.20, 0.40, 0.85, 1.90, "0.9440"),
    ("ss-bridge-a040-wg040-a100-b150", 0.40, 0.40, 1.00, 1.50, "0.9440"),
    ("ss-bridge-a015-wg035-a090-b150", 0.15, 0.35, 0.90, 1.50, "0.9439"),
]

# ── SS Bridge embed prior functions (replaces the RKNN cell block) ─────────────
SS_BRIDGE_CODE_TEMPLATE = '''
# --- Step 2b: SS Bridge RKNN Embed Prior (full-pipeline CV={cv_auc}) ---
# Uses 127,896 soundscape windows as "bridge" to enhance file-level similarity.
# For each test file, compute bridge_sim[j] = test_emb_avg_norm @ train_ss_sig[j]
# then combine with X_ref similarity for enhanced RKNN.
#   sim_combined = (1-{alpha}) × geo_sim + {alpha} × bridge_sim_normalized
#   RKNN k=5 on sim_combined
#   sigmoid({a} × vlom_logit + {b} × log({wg} × rknn_k5 + {ww} × win_k1))
_SSBRIDGE_ALPHA = {alpha}
_SSBRIDGE_A = {a}
_SSBRIDGE_B = {b}
_SSBRIDGE_WG = {wg}   # weight for SS-bridge RKNN
_SSBRIDGE_WW = {ww}   # weight for window KNN


def _build_geo_features_rknn(ep, test_emb, meta_df):
    """Build X_combined_n for test rows (same as geo-KNN)."""
    import re as _re
    SITES = ep['SITES']; site2idx = ep['site2idx']
    n_rows = len(test_emb); EPS = 1e-7
    _dt_re = _re.compile(r'_(\\d{{4}})(\\d{{2}})(\\d{{2}})_')
    test_months = np.zeros(n_rows, dtype=np.float32)
    test_days   = np.zeros(n_rows, dtype=np.float32)
    _dpm = [0,31,28,31,30,31,30,31,31,30,31,30,31]
    for ri, fn in enumerate(meta_df['filename'].values):
        m = _dt_re.search(str(fn))
        if m:
            mo, dy = int(m.group(2)), int(m.group(3))
            test_months[ri] = mo; test_days[ri] = sum(_dpm[:mo]) + dy
        else:
            test_months[ri] = 6; test_days[ri] = 152
    test_sites = meta_df['site'].values
    test_hours = meta_df['hour_utc'].values.astype(float)
    te_norm = test_emb / (np.linalg.norm(test_emb, axis=1, keepdims=True) + 1e-8)
    X_pca   = (te_norm - ep['pca_mean']) @ ep['pca_components'].T
    X_pca_s = (X_pca / ep['pca_std']).astype(np.float32)
    site_idxs = np.array([site2idx.get(str(s), -1) for s in test_sites])
    site_oh   = np.zeros((n_rows, len(SITES)), dtype=np.float32)
    valid = site_idxs >= 0; site_oh[valid, site_idxs[valid]] = 1.0
    hour_enc  = np.stack([np.sin(2*np.pi*test_hours/24), np.cos(2*np.pi*test_hours/24)], 1).astype(np.float32)
    month_enc = np.stack([np.sin(2*np.pi*(test_months-1)/12), np.cos(2*np.pi*(test_months-1)/12)], 1).astype(np.float32)
    day_enc   = np.stack([np.sin(2*np.pi*(test_days-1)/365), np.cos(2*np.pi*(test_days-1)/365)], 1).astype(np.float32)
    X_combined = np.concatenate([X_pca_s, site_oh, hour_enc, month_enc, day_enc], axis=1)
    nrm = np.linalg.norm(X_combined, axis=1, keepdims=True); nrm[nrm<1e-8]=1.0
    return (X_combined / nrm).astype(np.float32)


def _ss_bridge_embed_prior(ep, test_emb, meta_df, k=5, T=0.2, alpha={alpha}):
    """SS Bridge RKNN: uses soundscape-window signatures to enhance file similarity."""
    # 1. Standard geo+PCA features for test rows
    X_te = _build_geo_features_rknn(ep, test_emb, meta_df)
    X_ref = ep['X_combined_n']        # (66, 39) training files in geo space
    file_labels = ep['file_labels']   # (66, 234)
    n_rows = len(X_te); n_cls = file_labels.shape[1]; n_train = len(X_ref)
    EPS = 1e-7

    # 2. SS-bridge signatures: precomputed weighted avg of top-M soundscape windows
    #    train_ss_signatures: (66, 1536) — each training file's SS neighborhood
    train_ss_sig = ep['train_ss_signatures'].astype(np.float32)  # (66, 1536)

    # 3. Precompute training-training similarities for RKNN reciprocal check
    sim_ref = X_ref @ X_ref.T  # (66, 66)
    np.fill_diagonal(sim_ref, -np.inf)
    top_k_train_ref = np.argsort(-sim_ref, axis=1)[:, :k]

    # 4. Precompute bridge similarity matrix among training files (already in pkl)
    sim_bridge_train = ep['sim_bridge_n']  # (66, 66) normalized bridge similarities

    # 5. Combined training similarity
    sim_combined_train = (1-alpha) * sim_ref + alpha * sim_bridge_train
    np.fill_diagonal(sim_combined_train, -np.inf)
    top_k_train = np.argsort(-sim_combined_train, axis=1)[:, :k]
    kth_sim_train = sim_combined_train[np.arange(n_train), top_k_train[:, -1]]

    out = np.zeros((n_rows, n_cls), np.float32)
    BSZ = 256
    for s in range(0, n_rows, BSZ):
        Xb_geo = X_te[s:s+BSZ]         # (nb, 39) geo features for test batch
        Xb_emb = test_emb[s:s+BSZ]     # (nb, 1536) raw embeddings for bridge
        nb = len(Xb_geo)

        # Standard geo similarity
        sims_geo = Xb_geo @ X_ref.T    # (nb, 66)

        # Bridge similarity: test_emb_avg_norm @ train_ss_signatures
        Xb_emb_n = Xb_emb / (np.linalg.norm(Xb_emb, axis=1, keepdims=True) + 1e-8)
        sims_bridge_raw = Xb_emb_n @ train_ss_sig.T  # (nb, 66)
        # Row-normalize bridge similarity (consistent with training normalization)
        bnorm = np.linalg.norm(sims_bridge_raw, axis=1, keepdims=True).clip(1e-8)
        sims_bridge = sims_bridge_raw / bnorm  # (nb, 66)

        # Combined similarity
        sims_comb = (1-alpha) * sims_geo + alpha * sims_bridge  # (nb, 66)

        for bi in range(nb):
            sims_i = sims_comb[bi]
            top_i = np.argsort(-sims_i)[:k]
            mutual = []; mutual_sims = []
            for tj in top_i:
                if sims_i[tj] >= kth_sim_train[tj]:
                    mutual.append(tj); mutual_sims.append(sims_i[tj])
            if len(mutual) == 0:
                top5 = top_i[:5]; ls = sims_i[top5]/T; ls -= ls.max()
                w = np.exp(ls); w /= w.sum()
                out[s+bi] = (w[:,None] * file_labels[top5]).sum(0)
            else:
                ma = np.array(mutual); ms = np.array(mutual_sims)
                ls = ms/T; ls -= ls.max(); w = np.exp(ls); w /= w.sum()
                out[s+bi] = (w[:,None] * file_labels[ma]).sum(0)
    return out.clip(EPS, 1-EPS)


def _win_knn_bridge(ep, test_emb, k=1):
    """Window-KNN k=1 in raw 1536-dim L2-normalized Perch space."""
    emb_ref = ep.get('emb_win_norm', None)
    if emb_ref is None: return None
    wfi = ep['win_file_id']; fl = ep['file_labels']
    n_cls = fl.shape[1]; X_te = test_emb.astype(np.float32)
    nrm = np.linalg.norm(X_te, 1, keepdims=True); nrm[nrm<1e-8]=1.0; X_te=X_te/nrm
    X_ref = emb_ref.astype(np.float32); n_te = X_te.shape[0]
    out = np.zeros((n_te, n_cls), np.float32); BSZ = 512
    for s in range(0, n_te, BSZ):
        Xb = X_te[s:s+BSZ]; sims = Xb @ X_ref.T
        top = np.argsort(-sims, 1)[:, :k]
        for bi in range(len(Xb)):
            fids = wfi[top[bi]]; Ynn = fl[fids]
            w = sims[bi, top[bi]].clip(0); ws = w.sum()
            w = w/ws if ws>1e-8 else np.ones(k)/k
            out[s+bi] = (w[:,None]*Ynn).sum(0)
    return out.clip(1e-7, 1-1e-7)


# Load embed prior pkl
_EMBED_PRIOR_PATH = "/kaggle/input/birdclef-2026-dual-foundation/weights/embed_prior_ss_bridge.pkl"
with open(_EMBED_PRIOR_PATH, "rb") as _f:
    _ep = pickle.load(_f)
print(f"Loaded embed prior: {{_ep.get('method','?')}} LOO-AUC={{_ep.get('loo_auc',0):.4f}}")

# --- Run SS Bridge RKNN inference ---
y_bridge_rknn = _ss_bridge_embed_prior(
    ep=_ep, test_emb=emb_test_files.reshape(-1, emb_test_files.shape[-1]),
    meta_df=meta_test, k=5, T=0.2, alpha=_SSBRIDGE_ALPHA
)
print(f"SS Bridge RKNN shape: {{y_bridge_rknn.shape}}, mean: {{y_bridge_rknn.mean():.4f}}")

y_win_bridge = _win_knn_bridge(ep=_ep, test_emb=emb_test_files.reshape(-1, emb_test_files.shape[-1]), k=1)
if y_win_bridge is None:
    y_win_bridge = y_bridge_rknn
print(f"Win K1 shape: {{y_win_bridge.shape}}, mean: {{y_win_bridge.mean():.4f}}")

# Blend RKNN and win
EPS_e = 1e-7
y_ep_blended = _SSBRIDGE_WG * y_bridge_rknn + _SSBRIDGE_WW * y_win_bridge  # (n, 234)
log_ep = np.log(y_ep_blended.clip(EPS_e))

print(f"SS Bridge RKNN embed prior computed. Blending with VLOM base...")
'''

def get_fusion_code(src, new_code):
    """Replace the embed prior step 2b section."""
    # Find the RKNN section start
    markers = [
        "# --- Step 2b: Reciprocal KNN",
        "_RKNN_A =",
    ]
    start_idx = -1
    for marker in markers:
        idx = src.find(marker)
        if idx >= 0:
            start_idx = idx
            break
    if start_idx < 0:
        return None, "marker not found"

    # Find the end of the RKNN section (starts blending with test_base_scores)
    end_markers = [
        "# --- Blend: VLOM",
        "# Blend embed prior",
        "test_base_scores",
        "final_test_scores_blended",
    ]
    end_idx = -1
    for marker in end_markers:
        idx = src.find(marker, start_idx + 100)
        if idx >= 0:
            end_idx = idx
            break

    if end_idx < 0:
        return None, "end marker not found"

    # Replace RKNN code with SS Bridge code
    new_src = src[:start_idx] + new_code + src[end_idx:]
    return new_src, None


def build_fusion_note(alpha, wg, a, b):
    """Also build the final blend step that uses log_ep"""
    # Find and return the blend code
    return f"""
# --- Final Blend: VLOM base + SS Bridge RKNN prior ---
test_base_probs = sigmoid(test_base_logits)
vlom_logit = np.log(test_base_probs.clip(EPS_e)) - np.log((1 - test_base_probs).clip(EPS_e))
# Formula: sigmoid({a} × vlom_logit + {b} × log({wg}×rknn + {1-wg:.1f}×win))
final_test_scores_blended = sigmoid({a} * vlom_logit + {b} * log_ep)
print(f"Final scores shape: {{final_test_scores_blended.shape}}")
"""


notebooks_created = 0
for name_suffix, alpha, wg, a, b, cv_auc in CONFIGS:
    nb = copy.deepcopy(base_nb)
    cells = nb['cells']

    # Find cell 51 (score fusion cell)
    target_cell = None
    for i, cell in enumerate(cells):
        src = ''.join(cell['source'])
        if '_RKNN_A =' in src or 'Reciprocal KNN' in src:
            target_cell = i
            break

    if target_cell is None:
        print(f"  WARNING: Could not find RKNN cell in {name_suffix}")
        continue

    # Generate new SS bridge code
    ww = 1.0 - wg
    new_ep_code = SS_BRIDGE_CODE_TEMPLATE.format(
        cv_auc=cv_auc, alpha=alpha, a=a, b=b, wg=wg, ww=ww
    )

    # Get original cell source
    orig_src = ''.join(cells[target_cell]['source'])

    # Replace RKNN step 2b with SS bridge code
    new_src, err = get_fusion_code(orig_src, new_ep_code)
    if err:
        print(f"  Error replacing {name_suffix}: {err}")
        # Try direct approach: find and replace entire step 2b block
        # Find line with _RKNN_A
        lines = orig_src.split('\n')
        rknn_start = -1
        for li, line in enumerate(lines):
            if '_RKNN_A =' in line or '# --- Step 2b: Reciprocal KNN' in line:
                rknn_start = li
                break

        if rknn_start == -1:
            print(f"  SKIP: Cannot find RKNN start in cell {target_cell}")
            continue

        # Find where blend begins
        blend_start = -1
        for li in range(rknn_start + 5, len(lines)):
            if ('test_base_scores' in lines[li] and 'blend' in lines[li].lower()) or \
               ('# --- Blend' in lines[li]) or \
               ('print(f"SS Bridge' in lines[li]):
                pass
            if 'final_test_scores_blended' in lines[li] and 'sigmoid' in lines[li]:
                blend_start = li
                break
            if 'log_ep' in lines[li] and 'vlom' in lines[li].lower():
                blend_start = li
                break

        if blend_start == -1:
            print(f"  SKIP: Cannot find blend start")
            continue

        new_lines = lines[:rknn_start] + new_ep_code.split('\n') + lines[blend_start:]
        new_src = '\n'.join(new_lines)

    # Update cell source
    cells[target_cell]['source'] = [new_src]

    # Update the pkl filename reference in PKL loading cell
    for i, cell in enumerate(cells):
        src = ''.join(cell['source'])
        if 'embed_prior_rknn_k5_win1.pkl' in src:
            new_cell_src = src.replace('embed_prior_rknn_k5_win1.pkl', 'embed_prior_ss_bridge.pkl')
            cells[i]['source'] = [new_cell_src]
        elif 'embed_prior_logspace_geo5_win1.pkl' in src and '_EMBED_PRIOR_PATH' not in src:
            new_cell_src = src.replace('embed_prior_logspace_geo5_win1.pkl', 'embed_prior_ss_bridge.pkl')
            cells[i]['source'] = [new_cell_src]

    # Update description cell
    desc_cell_idx = None
    for i, cell in enumerate(cells):
        src = ''.join(cell['source'])
        if 'Embed Prior: LOO-AUC' in src or 'RKNN' in src[:100]:
            desc_cell_idx = i
            break

    if desc_cell_idx is not None:
        desc = f"""## Embed Prior: SS Bridge Weighted RKNN (CV={cv_auc})

**Method**: Soundscape Bridge RKNN
- Use 127,896 soundscape windows as "bridge" to enhance file-level similarity
- Bridge formula: sim_bridge[i,j] = Σ_m (sim(i,ss_m) × sim(j,ss_m)) for top-100 of j
- Combined similarity: (1-{alpha}) × geo_sim + {alpha} × bridge_sim_normalized
- RKNN k=5 on combined similarity
- Final: sigmoid({a} × vlom_logit + {b} × log({wg}×rknn + {1-wg:.1f}×win))

**Full Pipeline CV AUC**: {cv_auc}
"""
        cells[desc_cell_idx]['source'] = [desc]

    # Save notebook
    out_name = f"dual-foundation-protossm-{name_suffix}.ipynb"
    out_path = f"{OUT_DIR}/{out_name}"
    with open(out_path, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f"Created: {out_name}")
    notebooks_created += 1

print(f"\nTotal notebooks created: {notebooks_created}")
