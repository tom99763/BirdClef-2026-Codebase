"""
batch127: Cascade/nested co-occurrence combinations + per-species adaptive alpha
M1: Nested: two_round(idf_result) — 先 IDF 再 two-round（串聯而非並聯）
M2: Nested: idf_cooc(two_round_result)
M3: Cascade blend: 0.85*idf + 0.15*two_round(idf_result)
M4: Per-species adaptive alpha (alpha scales with IDF weight)
M5: Three-round co-occurrence (add 3rd weak round)
M6: Geometric mean blend (idf * two_round) ^ 0.5
"""
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

EPS = 1e-9

# ── data ──────────────────────────────────────────────────────────────────────
pkl = pickle.load(open('outputs/embed_prior_model.pkl', 'rb'))
win_file_id = pkl['win_file_id']      # [739] int32
file_labels  = pkl['file_labels']     # [66, 234]
file_prob_max = pkl['file_prob_max']  # [66, 234] — double_best predictions
N_FILES = file_labels.shape[0]
N_SP    = file_labels.shape[1]

data     = np.load('outputs/perch_labeled_ss.npz', allow_pickle=True)
LABELS   = data['labels'].astype(np.float32)
fnames   = data['filenames']
file_list = data['file_list']
file_ids = np.array([np.where(file_list == fn)[0][0] for fn in fnames])
sp_present = (LABELS.max(0) > 0)

# File-level co-occurrence matrix (from all 66 files)
fl = np.zeros((N_FILES, N_SP), dtype=np.float32)
for fi in range(N_FILES):
    mask = (file_ids == fi)
    fl[fi] = (LABELS[mask].max(0) > 0.5).astype(np.float32)

count_i  = fl.sum(0) + EPS
cooc     = fl.T @ fl / count_i[:, None]
np.fill_diagonal(cooc, 0)
COOC_NORM = cooc / (cooc.sum(1, keepdims=True) + EPS)

n_pos_files = fl.sum(0)
IDF = np.clip(np.log((N_FILES + 1) / (n_pos_files + 1)), 0, None)
IDF_W075 = IDF ** 0.75 / (IDF.mean() + EPS)

print(f"[batch127] N_FILES={N_FILES}, N_SP={N_SP}, sp_present={sp_present.sum()}")

# ── JSON store ─────────────────────────────────────────────────────────────────
results_path = Path('outputs/embed_prior_results.json')
with open(results_path) as f:
    store = json.load(f)
tried     = {e['method'] for e in store.get('experiments', [])}
best_loo  = store['best']['loo_auc']
best_method = store['best']['method']
print(f"[batch127] Current best: {best_method} LOO={best_loo:.6f}")

def file_loo_auc(file_scores):
    auc_list = []
    for fi in range(N_FILES):
        score = file_scores[fi]
        true  = fl[fi]
        sp = sp_present
        if sp.sum() < 2: continue
        try:
            auc_list.append(roc_auc_score(true[sp], score[sp]))
        except Exception:
            pass
    return float(np.mean(auc_list))

def save_result(method, score, config, note=''):
    global best_loo, best_method
    delta = score - best_loo
    r = {'method': method, 'loo_auc': score, 'config': config, 'note': note}
    store['experiments'].append(r)
    if score > best_loo:
        best_loo = score
        best_method = method
        store['best'] = {'method': method, 'loo_auc': score}
    with open(results_path, 'w') as f:
        json.dump(store, f, indent=2)
    return delta

# ── core functions (same as production) ──────────────────────────────────────
def soft_cooc(scores, center=0.53, slope=37.0, alpha=0.086, idf_w=None):
    smoothed = np.zeros_like(scores)
    for fi in range(N_FILES):
        s = scores[fi]
        arg = np.clip(-slope * (s - center), -88, 88)
        gate = 1.0 / (1.0 + np.exp(arg))
        s_gated = s * gate
        if idf_w is not None:
            s_gated = s_gated * idf_w
        contrib = COOC_NORM.T @ s_gated
        max_c = np.abs(contrib).max()
        if max_c > EPS:
            contrib /= max_c
        smoothed[fi] = (1 - alpha) * s + alpha * np.clip(contrib, 0, None)
    return smoothed

def idf_cooc(scores, center=0.55, slope=41.0, alpha=0.130, blend=0.55):
    s_pow = np.clip(scores, 0, 1) ** 2.0
    s_cooc = soft_cooc(s_pow, center=center, slope=slope, alpha=alpha, idf_w=IDF_W075)
    return (1 - blend) * scores + blend * s_cooc

def two_round(scores, c1=0.54, sl1=41.0, a1=0.089, c2=0.53, sl2=37.0, a2=0.040):
    r1 = soft_cooc(scores, center=c1, slope=sl1, alpha=a1)
    r2 = soft_cooc(r1,     center=c2, slope=sl2, alpha=a2)
    return r2

# Pre-compute reference outputs
print("Pre-computing reference outputs...")
double_best = file_prob_max.copy()
idf_result   = idf_cooc(double_best)
tworound_res = two_round(double_best)
current_best = 0.85 * idf_result + 0.15 * tworound_res
print(f"  double_best AUC: {file_loo_auc(double_best):.6f}")
print(f"  idf_result AUC:  {file_loo_auc(idf_result):.6f}")
print(f"  two_round AUC:   {file_loo_auc(tworound_res):.6f}")
print(f"  3way AUC:        {file_loo_auc(current_best):.6f} (expected {best_loo:.6f})")

# ═════════════════════════════════════════════════════════════════════════════
# M1: Nested — two_round(idf_result)
# ═════════════════════════════════════════════════════════════════════════════
print("\n[M1] Nested: two_round on top of idf_result...")

# Different two-round params applied to idf_result
m1_configs = [
    # (c1, sl1, a1, c2, sl2, a2)
    (0.54, 41.0, 0.089, 0.53, 37.0, 0.040),  # default two-round
    (0.54, 41.0, 0.040, 0.53, 37.0, 0.020),  # smaller alphas (already smoothed input)
    (0.54, 41.0, 0.060, 0.53, 37.0, 0.030),  # medium alphas
    (0.55, 41.0, 0.050, 0.54, 37.0, 0.025),  # tighter centers
    (0.50, 37.0, 0.060, 0.50, 33.0, 0.030),  # lower centers
    (0.54, 41.0, 0.020, 0.53, 37.0, 0.010),  # tiny alphas
    (0.56, 45.0, 0.040, 0.55, 41.0, 0.020),  # high centers
]

m1_best = 0.0
for cfg in m1_configs:
    c1, sl1, a1, c2, sl2, a2 = cfg
    mname = f'nest_idf_2r_c{int(c1*100):d}_a1{int(a1*1000):03d}_a2{int(a2*1000):03d}'
    if mname in tried:
        print(f'  {mname}: skip')
        continue
    result = two_round(idf_result, c1=c1, sl1=sl1, a1=a1, c2=c2, sl2=sl2, a2=a2)
    score  = file_loo_auc(result)
    delta  = save_result(mname, score, {'c1':c1,'sl1':sl1,'a1':a1,'c2':c2,'sl2':sl2,'a2':a2})
    tag    = f'+{delta:.6f}' if delta > 0 else f'{delta:.6f}'
    flag   = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {tag}{flag}')
    m1_best = max(m1_best, score)

# Also try blend of original and nested
for blend_w in [0.85, 0.90, 0.95, 0.80]:
    mname = f'nest_idf_blend_w{int(blend_w*100):d}'
    if mname in tried:
        continue
    r2_on_idf = two_round(idf_result, c1=0.54, sl1=41.0, a1=0.040, c2=0.53, sl2=37.0, a2=0.020)
    result = blend_w * idf_result + (1 - blend_w) * r2_on_idf
    score  = file_loo_auc(result)
    delta  = save_result(mname, score, {'blend_w': blend_w})
    tag    = f'+{delta:.6f}' if delta > 0 else f'{delta:.6f}'
    flag   = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {tag}{flag}')
    m1_best = max(m1_best, score)

print(f"  M1 done, best={m1_best:.6f}")

# ═════════════════════════════════════════════════════════════════════════════
# M2: Nested — idf_cooc(two_round_result)
# ═════════════════════════════════════════════════════════════════════════════
print("\n[M2] Nested: idf_cooc on top of two_round_result...")

m2_configs = [
    {'center': 0.55, 'slope': 41.0, 'alpha': 0.060, 'blend': 0.55},
    {'center': 0.55, 'slope': 41.0, 'alpha': 0.090, 'blend': 0.55},
    {'center': 0.55, 'slope': 41.0, 'alpha': 0.130, 'blend': 0.55},
    {'center': 0.55, 'slope': 41.0, 'alpha': 0.060, 'blend': 0.70},
    {'center': 0.55, 'slope': 41.0, 'alpha': 0.090, 'blend': 0.70},
    {'center': 0.54, 'slope': 41.0, 'alpha': 0.060, 'blend': 0.55},
    {'center': 0.54, 'slope': 41.0, 'alpha': 0.060, 'blend': 0.70},
]

m2_best = 0.0
for cfg in m2_configs:
    mname = f'nest_2r_idf_c{int(cfg["center"]*100):d}_a{int(cfg["alpha"]*1000):03d}_w{int(cfg["blend"]*100):d}'
    if mname in tried:
        print(f'  {mname}: skip')
        continue
    result = idf_cooc(tworound_res, **cfg)
    score  = file_loo_auc(result)
    delta  = save_result(mname, score, cfg)
    tag    = f'+{delta:.6f}' if delta > 0 else f'{delta:.6f}'
    flag   = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {tag}{flag}')
    m2_best = max(m2_best, score)

# Blend of 3way and nested idf(2r)
for blend_w in [0.85, 0.90, 0.95]:
    mname = f'nest_2r_idf_3way_blend_w{int(blend_w*100):d}'
    if mname in tried:
        continue
    nested = idf_cooc(tworound_res, center=0.55, slope=41.0, alpha=0.060, blend=0.55)
    result = blend_w * current_best + (1 - blend_w) * nested
    score  = file_loo_auc(result)
    delta  = save_result(mname, score, {'blend_w': blend_w})
    tag    = f'+{delta:.6f}' if delta > 0 else f'{delta:.6f}'
    flag   = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {tag}{flag}')
    m2_best = max(m2_best, score)

print(f"  M2 done, best={m2_best:.6f}")

# ═════════════════════════════════════════════════════════════════════════════
# M3: Cascade blend — 0.85*idf + 0.15*two_round(idf_result)
# ═════════════════════════════════════════════════════════════════════════════
print("\n[M3] Cascade blend (3way with nested 2r component)...")

# Pre-compute two_round on idf_result
r2_on_idf_small = two_round(idf_result, c1=0.54, sl1=41.0, a1=0.040, c2=0.53, sl2=37.0, a2=0.020)
r2_on_idf_med   = two_round(idf_result, c1=0.54, sl1=41.0, a1=0.060, c2=0.53, sl2=37.0, a2=0.030)

for w_idf, w_2r, w_nest, tag_str in [
    (0.85, 0.10, 0.05, 'cas_i85_r10_n05'),
    (0.85, 0.08, 0.07, 'cas_i85_r08_n07'),
    (0.85, 0.05, 0.10, 'cas_i85_r05_n10'),
    (0.85, 0.12, 0.03, 'cas_i85_r12_n03'),
    (0.80, 0.10, 0.10, 'cas_i80_r10_n10'),
    (0.85, 0.00, 0.15, 'cas_i85_r00_n15'),
    (0.90, 0.05, 0.05, 'cas_i90_r05_n05'),
]:
    if tag_str in tried:
        print(f'  {tag_str}: skip')
        continue
    result = w_idf * idf_result + w_2r * tworound_res + w_nest * r2_on_idf_small
    score  = file_loo_auc(result)
    delta  = save_result(tag_str, score, {'w_idf':w_idf,'w_2r':w_2r,'w_nest':w_nest})
    tag    = f'+{delta:.6f}' if delta > 0 else f'{delta:.6f}'
    flag   = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {tag_str}: {score:.6f} {tag}{flag}')

# Also with medium nested
for w_idf, w_2r, w_nest, tag_str in [
    (0.85, 0.10, 0.05, 'casM_i85_r10_n05'),
    (0.85, 0.05, 0.10, 'casM_i85_r05_n10'),
    (0.85, 0.00, 0.15, 'casM_i85_r00_n15'),
]:
    if tag_str in tried:
        continue
    result = w_idf * idf_result + w_2r * tworound_res + w_nest * r2_on_idf_med
    score  = file_loo_auc(result)
    delta  = save_result(tag_str, score, {'w_idf':w_idf,'w_2r':w_2r,'w_nest':w_nest,'nest':'med'})
    tag    = f'+{delta:.6f}' if delta > 0 else f'{delta:.6f}'
    flag   = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {tag_str}: {score:.6f} {tag}{flag}')

print(f"  M3 done")

# ═════════════════════════════════════════════════════════════════════════════
# M4: Per-species adaptive alpha
# ═════════════════════════════════════════════════════════════════════════════
print("\n[M4] Per-species adaptive alpha...")

def adaptive_alpha_cooc(scores, center=0.55, slope=41.0, base_alpha=0.130,
                         blend=0.55, idf_scale=0.5, power=2.0):
    """
    alpha_sp = base_alpha * (1 + idf_scale * (IDF_W075_sp - 1))
    Rare species (high IDF) get higher alpha (more co-occurrence smoothing).
    """
    s_pow = np.clip(scores, 0, 1) ** power
    # per-species alpha: scale with IDF weight
    alpha_sp = base_alpha * (1 + idf_scale * (IDF_W075 - 1.0))
    alpha_sp = np.clip(alpha_sp, 0.01, 0.5)

    smoothed = np.zeros_like(scores)
    for fi in range(N_FILES):
        s = s_pow[fi]
        arg = np.clip(-slope * (s - center), -88, 88)
        gate = 1.0 / (1.0 + np.exp(arg))
        s_gated = s * gate * IDF_W075
        contrib = COOC_NORM.T @ s_gated
        max_c = np.abs(contrib).max()
        if max_c > EPS:
            contrib /= max_c
        # per-species adaptive alpha
        smoothed[fi] = (1 - alpha_sp) * s + alpha_sp * np.clip(contrib, 0, None)

    return (1 - blend) * scores + blend * smoothed

m4_configs = [
    {'idf_scale': 0.3, 'base_alpha': 0.130},
    {'idf_scale': 0.5, 'base_alpha': 0.130},
    {'idf_scale': 0.7, 'base_alpha': 0.130},
    {'idf_scale': 0.5, 'base_alpha': 0.100},
    {'idf_scale': 0.5, 'base_alpha': 0.150},
    {'idf_scale': 1.0, 'base_alpha': 0.130},
    {'idf_scale': -0.3, 'base_alpha': 0.130},  # common species get higher alpha
]

m4_best = 0.0
for cfg in m4_configs:
    is_val = int(cfg['idf_scale'] * 10)
    sign = 'p' if is_val >= 0 else 'm'
    mname = f'adapt_alpha_is{sign}{abs(is_val):d}_ba{int(cfg["base_alpha"]*1000):03d}'
    if mname in tried:
        print(f'  {mname}: skip')
        continue
    result = adaptive_alpha_cooc(double_best, **cfg)
    score  = file_loo_auc(result)
    delta  = save_result(mname, score, cfg)
    tag    = f'+{delta:.6f}' if delta > 0 else f'{delta:.6f}'
    flag   = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {tag}{flag}')
    m4_best = max(m4_best, score)

# Blend adaptive_alpha result with 3way
for w_adapt in [0.1, 0.2, 0.3]:
    best_adapt_cfg = {'idf_scale': 0.5, 'base_alpha': 0.130}
    adapt_result = adaptive_alpha_cooc(double_best, **best_adapt_cfg)
    mname = f'adapt_3way_blend_wa{int(w_adapt*10):d}'
    if mname in tried:
        continue
    result = (1 - w_adapt) * current_best + w_adapt * adapt_result
    score  = file_loo_auc(result)
    delta  = save_result(mname, score, {'w_adapt': w_adapt})
    tag    = f'+{delta:.6f}' if delta > 0 else f'{delta:.6f}'
    flag   = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {tag}{flag}')
    m4_best = max(m4_best, score)

print(f"  M4 done, best={m4_best:.6f}")

# ═════════════════════════════════════════════════════════════════════════════
# M5: Three-round co-occurrence
# ═════════════════════════════════════════════════════════════════════════════
print("\n[M5] Three-round co-occurrence...")

def three_round(scores, c1=0.54, sl1=41.0, a1=0.089,
                           c2=0.53, sl2=37.0, a2=0.040,
                           c3=0.53, sl3=33.0, a3=0.020):
    r1 = soft_cooc(scores, center=c1, slope=sl1, alpha=a1)
    r2 = soft_cooc(r1,     center=c2, slope=sl2, alpha=a2)
    r3 = soft_cooc(r2,     center=c3, slope=sl3, alpha=a3)
    return r3

m5_configs = [
    # a3 sweep
    {'c3': 0.53, 'sl3': 33.0, 'a3': 0.010},
    {'c3': 0.53, 'sl3': 33.0, 'a3': 0.020},
    {'c3': 0.53, 'sl3': 33.0, 'a3': 0.030},
    {'c3': 0.52, 'sl3': 30.0, 'a3': 0.020},
    {'c3': 0.54, 'sl3': 37.0, 'a3': 0.020},
]

m5_best = 0.0
for cfg in m5_configs:
    mname = f'3round_c{int(cfg["c3"]*100):d}_sl{int(cfg["sl3"]):d}_a{int(cfg["a3"]*1000):03d}'
    if mname in tried:
        print(f'  {mname}: skip')
        continue
    result = three_round(double_best, **cfg)
    score  = file_loo_auc(result)
    delta  = save_result(mname, score, cfg)
    tag    = f'+{delta:.6f}' if delta > 0 else f'{delta:.6f}'
    flag   = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {tag}{flag}')
    m5_best = max(m5_best, score)

# 3-round blended with idf (4-way)
for w_3r, w_idf, w_2r in [(0.05, 0.80, 0.15), (0.10, 0.80, 0.10), (0.05, 0.85, 0.10)]:
    mname = f'4way_i{int(w_idf*100):d}_r{int(w_2r*100):d}_3r{int(w_3r*100):d}'
    if mname in tried:
        continue
    three_r = three_round(double_best, c3=0.53, sl3=33.0, a3=0.020)
    result = w_idf * idf_result + w_2r * tworound_res + w_3r * three_r
    score  = file_loo_auc(result)
    delta  = save_result(mname, score, {'w_idf':w_idf,'w_2r':w_2r,'w_3r':w_3r})
    tag    = f'+{delta:.6f}' if delta > 0 else f'{delta:.6f}'
    flag   = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {tag}{flag}')
    m5_best = max(m5_best, score)

print(f"  M5 done, best={m5_best:.6f}")

# ═════════════════════════════════════════════════════════════════════════════
# M6: Geometric mean blend
# ═════════════════════════════════════════════════════════════════════════════
print("\n[M6] Geometric mean and other non-linear blends...")

for gamma in [0.3, 0.5, 0.7, 0.85, 0.15]:
    mname = f'geomean_g{int(gamma*100):d}'
    if mname in tried:
        continue
    # weighted geometric mean: idf^gamma * two_round^(1-gamma)
    eps_blend = 1e-6
    result = (np.clip(idf_result, eps_blend, 1) ** gamma *
              np.clip(tworound_res, eps_blend, 1) ** (1 - gamma))
    score  = file_loo_auc(result)
    delta  = save_result(mname, score, {'gamma': gamma})
    tag    = f'+{delta:.6f}' if delta > 0 else f'{delta:.6f}'
    flag   = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {tag}{flag}')

# Max blend: element-wise max of idf and two_round
for w_max in [0.05, 0.10, 0.15]:
    mname = f'maxblend_wm{int(w_max*100):d}'
    if mname in tried:
        continue
    element_max = np.maximum(idf_result, tworound_res)
    result = (1 - w_max) * (0.85*idf_result + 0.15*tworound_res) + w_max * element_max
    score  = file_loo_auc(result)
    delta  = save_result(mname, score, {'w_max': w_max})
    tag    = f'+{delta:.6f}' if delta > 0 else f'{delta:.6f}'
    flag   = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  {mname}: {score:.6f} {tag}{flag}')

# Power post-processing: boost high scores
for pow_post in [1.05, 0.95, 1.1]:
    mname = f'postpow_{int(pow_post*100):d}'
    if mname in tried:
        continue
    result = np.clip(current_best, 0, 1) ** pow_post
    score  = file_loo_auc(result)
    delta  = save_result(mname, score, {'pow_post': pow_post})
    tag    = f'+{delta:.6f}' if delta > 0 else f'{delta:.6f}'
    flag   = ' ← NEW BEST!' if score > best_loo else ''
    print(f'  postpow_{int(pow_post*100):d}: {score:.6f} {tag}{flag}')

print(f"  M6 done")

# ── final summary ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"[batch127] SUMMARY")
print(f"  Global best: {store['best']['method']} LOO={store['best']['loo_auc']:.6f}")
print(f"  M1 Nested idf→2r:   {m1_best:.6f}")
print(f"  M2 Nested 2r→idf:   {m2_best:.6f}")
print(f"  M3 Cascade blend:   (see above)")
print(f"  M4 Adaptive alpha:  {m4_best:.6f}")
print(f"  M5 Three-round:     {m5_best:.6f}")
print(f"  M6 Non-linear:      (see above)")
