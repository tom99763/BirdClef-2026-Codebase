"""
create_stacker_v3_notebooks.py — Generate all 9 stacker-v3 Kaggle inference notebooks.

Run:
    python scripts/create_stacker_v3_notebooks.py
"""

import json
from pathlib import Path

NB_DIR = Path("/home/lab/BirdClef-2026-Codebase/birdclef-2026/notebook resource/current_subs 2")

# ─── Shared notebook cells ────────────────────────────────────────────────────

CELL_INSTALL = """\
# Cell 0 — Install dependencies
!pip install -q --no-deps /kaggle/input/notebooks/ashok205/tf-wheels/tf_wheels/tensorboard-2.20.0-py3-none-any.whl
!pip install -q --no-deps /kaggle/input/notebooks/ashok205/tf-wheels/tf_wheels/tensorflow-2.20.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"""

CELL_IMPORTS = """\
# Cell 1 — Imports
import os, gc, re, time, json, warnings, pickle
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''   # CPU-only on Kaggle

from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio
import torchaudio.transforms as T

import onnxruntime as ort
from tqdm.auto import tqdm

tf.experimental.numpy.experimental_enable_numpy_behavior()
_WALL_START = time.time()
print('Imports OK')"""

CELL_PATHS = """\
# Cell 2 — Paths and constants
BASE        = Path('/kaggle/input/birdclef-2026')
MODEL_DIR   = Path('/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1')
DATASET_DIR = Path('/kaggle/input/datasets/tom99763/birdclef2026-claude/weights/weights')
STACKER_DIR = Path('/kaggle/input/datasets/tom99763/birdclef2026-claude/weights/stacker_weights')
AUDIO_DIR   = BASE / 'test_soundscapes'

SR          = 32_000
WIN_SECS    = 5
WIN_SAMPLES = SR * WIN_SECS
FILE_SECS   = 60
N_WIN       = 12
N_CLASSES   = 234
N_MODELS    = 5    # perch_raw | perch_prior_fused | mlp_probe | proto_ssm | sed_csebbs
FEAT_DIM    = N_MODELS * N_CLASSES   # 1170
EPS         = 1e-6
CONTEXT_K   = 1
CONTEXT_SIZE = 2 * CONTEXT_K + 1   # 3
CTX_FEAT_DIM = CONTEXT_SIZE * FEAT_DIM   # 3510
TEMPERATURE = 1.5

DEVICE = torch.device('cpu')

print(f'BASE exists: {BASE.exists()}')
print(f'DATASET_DIR exists: {DATASET_DIR.exists()}  STACKER_DIR exists: {STACKER_DIR.exists()}')"""

CELL_TAXONOMY = """\
# Cell 3 — Taxonomy and test file discovery
taxonomy   = pd.read_csv(BASE / 'taxonomy.csv')
sample_sub = pd.read_csv(BASE / 'sample_submission.csv')

PRIMARY_LABELS = sample_sub.columns[1:].tolist()
assert len(PRIMARY_LABELS) == N_CLASSES

test_paths = sorted(AUDIO_DIR.glob('*.ogg')) if AUDIO_DIR.exists() else []
print(f'Test soundscapes found: {len(test_paths)}')

meta_rows = []
for p in test_paths:
    stem = p.stem
    for w in range(N_WIN):
        meta_rows.append({'filename': p.name, 'row_id': f'{stem}_{(w+1)*5}'})
meta_test = pd.DataFrame(meta_rows)
print(f'Total row_ids: {len(meta_test)}')"""

CELL_PERCH_SETUP = """\
# Cell 4 — Load Perch v2 and build label mapping
birdclassifier = tf.saved_model.load(str(MODEL_DIR))
infer_fn = birdclassifier.signatures['serving_default']

bc_labels = (
    pd.read_csv(MODEL_DIR / 'assets' / 'labels.csv')
    .reset_index()
    .rename(columns={'index': 'bc_index', 'inat2024_fsd50k': 'scientific_name'})
)
NO_LABEL_INDEX = len(bc_labels)

taxonomy['scientific_name_lookup'] = taxonomy['scientific_name'].str.strip().str.lower()
bc_labels['sci_lower'] = bc_labels['scientific_name'].str.strip().str.lower()

MAPPED_BC_INDICES = []
for label in PRIMARY_LABELS:
    row = taxonomy[taxonomy['primary_label'] == label]
    if row.empty:
        MAPPED_BC_INDICES.append(NO_LABEL_INDEX); continue
    sci = row.iloc[0]['scientific_name'].strip().lower()
    match = bc_labels[bc_labels['sci_lower'] == sci]
    MAPPED_BC_INDICES.append(int(match.iloc[0]['bc_index']) if not match.empty else NO_LABEL_INDEX)

MAPPED_BC_INDICES = np.array(MAPPED_BC_INDICES, dtype=np.int32)
print(f'Perch coverage: {(MAPPED_BC_INDICES < NO_LABEL_INDEX).sum()} / {N_CLASSES} species')


def read_audio_60s(path: Path) -> np.ndarray:
    audio, sr_in = sf.read(str(path), dtype='float32', always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr_in != SR:
        audio = torchaudio.functional.resample(torch.from_numpy(audio), sr_in, SR).numpy()
    target = SR * FILE_SECS
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    return audio[:target]


def perch_infer_file(audio: np.ndarray):
    logits_win, embs_win = [], []
    for w in range(N_WIN):
        clip    = audio[w * WIN_SAMPLES:(w+1) * WIN_SAMPLES].astype(np.float32)
        out     = infer_fn(tf.constant(clip[None], dtype=tf.float32))
        raw     = out['output_0'].numpy().squeeze(0)
        emb     = out['output_1'].numpy().squeeze(0) if 'output_1' in out else np.zeros(1536, np.float32)
        mapped  = np.zeros(N_CLASSES, dtype=np.float32)
        valid   = MAPPED_BC_INDICES < NO_LABEL_INDEX
        mapped[valid] = raw[MAPPED_BC_INDICES[valid]]
        logits_win.append(mapped)
        embs_win.append(emb)
    return np.stack(logits_win), np.stack(embs_win)  # (12,234), (12,1536)

print('Perch helpers OK.')"""

CELL_PRIOR = """\
# Cell 5 — Prior-fused Perch scores
_prior_path = STACKER_DIR / 'prior_tables.pkl'
_prior_tables = None
if _prior_path.exists():
    with open(_prior_path, 'rb') as fh:
        _prior_tables = pickle.load(fh)
    print('prior_tables loaded')
else:
    print('prior_tables.pkl not found - using perch_raw as fallback')


def apply_prior_fused(perch_raw_12: np.ndarray, audio_path: Path) -> np.ndarray:
    if _prior_tables is None:
        return perch_raw_12.copy()
    stem  = audio_path.stem
    parts = stem.split('_')
    site  = parts[0] if parts else ''
    hour  = -1
    if len(parts) >= 3:
        try: hour = int(parts[2][:2])
        except (ValueError, IndexError): pass
    eps       = 1e-6
    log_prior = np.zeros(N_CLASSES, dtype=np.float32)
    global_p  = _prior_tables.get('global_p', None)
    site_p    = _prior_tables.get('site_p', None)
    site_to_i = _prior_tables.get('site_to_i', {})
    hour_p    = _prior_tables.get('hour_p', None)
    hour_to_i = _prior_tables.get('hour_to_i', {})
    if global_p  is not None: log_prior += np.log(np.clip(global_p, eps, 1.0))
    if site_p    is not None and site in site_to_i:
        log_prior += np.log(np.clip(site_p[site_to_i[site]], eps, 1.0))
    if hour_p    is not None and hour in hour_to_i:
        log_prior += np.log(np.clip(hour_p[hour_to_i[hour]], eps, 1.0))
    return (perch_raw_12 + log_prior[None, :]).astype(np.float32)

print('Prior helper OK.')"""

CELL_MLP_PROBE = """\
# Cell 6 — MLP probe from Perch embeddings
_pca_model, _probe_models = None, None
if (STACKER_DIR / 'pca_model.pkl').exists():
    with open(STACKER_DIR / 'pca_model.pkl', 'rb') as fh:
        _pca_model = pickle.load(fh)
    print(f'PCA loaded: n_components={_pca_model.n_components}')
if (STACKER_DIR / 'mlp_probes.pkl').exists():
    with open(STACKER_DIR / 'mlp_probes.pkl', 'rb') as fh:
        _probe_models = pickle.load(fh)
    print(f'Probes loaded: {len(_probe_models)} classes')


def apply_mlp_probe(embs_12: np.ndarray) -> np.ndarray:
    if _pca_model is None or _probe_models is None:
        return np.zeros((N_WIN, N_CLASSES), dtype=np.float32)
    z   = _pca_model.transform(embs_12)
    out = np.zeros((N_WIN, N_CLASSES), dtype=np.float32)
    for c_idx, clf in _probe_models.items():
        if c_idx >= N_CLASSES: continue
        try:
            p = clf.predict_proba(z)[:, 1]
            out[:, c_idx] = np.log(np.clip(p, EPS, 1-EPS) / np.clip(1-p, EPS, 1-EPS))
        except Exception: pass
    return out

print('MLP probe helper OK.')"""

CELL_PROTO = """\
# Cell 7 — ProtoSSM TFLite inference
_proto_interps = {}
for fold_idx in range(5):
    p = DATASET_DIR / f'proto_ssm_v4_full_fold{fold_idx}.tflite'
    if p.exists():
        interp = tf.lite.Interpreter(model_path=str(p))
        interp.allocate_tensors()
        _proto_interps[fold_idx] = interp
USE_PROTO = len(_proto_interps) > 0
print(f'ProtoSSM folds loaded: {len(_proto_interps)}')


def proto_ssm_infer_file(emb_seq: np.ndarray) -> np.ndarray:
    '''emb_seq: (12,1536) -> (234,) logits avg over folds.'''
    if not USE_PROTO:
        return np.zeros(N_CLASSES, dtype=np.float32)
    preds = []
    for interp in _proto_interps.values():
        inp = interp.get_input_details()
        out = interp.get_output_details()
        data = emb_seq[None].astype(np.float32)
        interp.resize_tensor_input(inp[0]['index'], data.shape)
        interp.allocate_tensors()
        interp.set_tensor(inp[0]['index'], data)
        interp.invoke()
        preds.append(interp.get_tensor(out[0]['index']).squeeze())
    return np.mean(preds, axis=0).astype(np.float32)

print('ProtoSSM helper OK.')"""

CELL_SED = """\
# Cell 8 — SED ONNX sessions + BranchEns->cSEBBs
_mel_eff = T.MelSpectrogram(
    sample_rate=SR, n_fft=2048, hop_length=512, n_mels=224,
    f_min=0, f_max=16000, power=2.0, norm='slaney', mel_scale='htk',
)
_db_eff = T.AmplitudeToDB(top_db=80)
_sess_opts = ort.SessionOptions()
_sess_opts.intra_op_num_threads = 4
_providers = ['CPUExecutionProvider']

_sed_sessions = []
for name in ['best_sed_b0_v5.onnx', 'competitor_sed_fold0.onnx']:
    p = DATASET_DIR / name
    if p.exists():
        _sed_sessions.append(ort.InferenceSession(str(p), sess_options=_sess_opts,
                                                   providers=_providers))
print(f'SED sessions: {len(_sed_sessions)}')


def _make_mel_eff(clip: np.ndarray) -> np.ndarray:
    wav = torch.from_numpy(clip).unsqueeze(0)
    mel = _db_eff(_mel_eff(wav))
    mel = mel - mel.min()
    mx  = mel.max()
    if mx > 0: mel = mel / mx
    return mel.repeat(3, 1, 1)[None].numpy()


def apply_branchens_csebbs(probs_12: np.ndarray) -> np.ndarray:
    eps = 1e-7
    p = np.clip(probs_12.astype(np.float32), eps, 1.0 - eps)
    T_len, C = p.shape
    H  = -(p * np.log(p) + (1.0-p)*np.log(1.0-p)).mean(axis=1)
    w  = np.exp(-H / 0.1); w = w / w.sum() * T_len
    wl = np.log(p / (1.0-p)) * w[:, None]

    def _lse(wl_in, beta):
        out = np.zeros_like(wl_in)
        for t in range(T_len):
            win = wl_in[max(0,t-1):min(T_len,t+2)]
            mx  = win.max(axis=0)
            out[t] = mx + (1/beta)*np.log(np.exp(beta*(win-mx)).sum(axis=0))
        return 1.0/(1.0+np.exp(-out))

    def _anchor(lp, nw, alpha):
        anc = nw*(1.0-np.prod(1.0-lp,axis=0))+(1.0-nw)*lp.max(axis=0)
        return (1.0-alpha)*lp+alpha*anc[None,:]

    ens = np.clip(0.55*_anchor(_lse(wl,5.15),0.40,0.38)+0.45*_anchor(_lse(wl,6.0),0.30,0.40),eps,1-eps)
    out  = ens.copy()
    diff = np.abs(np.diff(ens, axis=0))
    for t in range(T_len-1):
        cols = np.where(diff[t]>0.06)[0]
        if len(cols):
            seg=ens[max(0,t-2):min(T_len,t+3)]
            out[t,cols]=seg[:,cols].mean(axis=0)
    return out.astype(np.float32)


def safe_logit(p, eps=EPS):
    p = np.clip(p.astype(np.float32), eps, 1.0-eps)
    return np.log(p/(1.0-p))


def sed_csebbs_infer_file(audio: np.ndarray) -> np.ndarray:
    sed_wins = []
    for w in range(N_WIN):
        clip = audio[w*WIN_SAMPLES:(w+1)*WIN_SAMPLES].astype(np.float32)
        mel  = _make_mel_eff(clip)
        if _sed_sessions:
            preds = [s.run(None,{s.get_inputs()[0].name:mel})[0].squeeze(0) for s in _sed_sessions]
            sed_wins.append(np.mean(preds,axis=0).astype(np.float32))
        else:
            sed_wins.append(np.full(N_CLASSES, 0.5, np.float32))
    return safe_logit(apply_branchens_csebbs(np.stack(sed_wins)))

print('SED+cSEBBs helpers OK.')"""

CELL_BUILD_FEATURES = """\
# Cell 7b — Feature building for test set
def build_context_row(X_file_12: np.ndarray, t: int, k: int) -> np.ndarray:
    T_total, F = X_file_12.shape
    W   = 2*k+1
    out = np.zeros(W*F, dtype=np.float32)
    for slot, offset in enumerate(range(-k, k+1)):
        j = t+offset
        if 0<=j<T_total:
            out[slot*F:(slot+1)*F] = X_file_12[j]
    return out


def build_test_features(audio_path: Path, feat_mean: np.ndarray, feat_std: np.ndarray):
    audio = read_audio_60s(audio_path)
    perch_logits, perch_embs = perch_infer_file(audio)       # (12,234), (12,1536)
    prior_fused  = apply_prior_fused(perch_logits, audio_path)
    mlp_probe_l  = apply_mlp_probe(perch_embs)
    proto_1d     = proto_ssm_infer_file(perch_embs)
    proto_12     = np.tile(proto_1d, (N_WIN, 1))
    sed_logits   = sed_csebbs_infer_file(audio)
    X_raw = np.concatenate([perch_logits, prior_fused, mlp_probe_l, proto_12, sed_logits], axis=1)
    X_norm = ((X_raw - feat_mean) / feat_std).astype(np.float32)
    del audio; gc.collect()
    return X_raw, X_norm   # (12,1170) each

print('Feature building helper OK.')"""

CELL_LOAD_META = """\
# Cell 8b — Load stacker meta + normalisation stats
_meta_path = STACKER_DIR / 'stacker_meta_v3.json'
if not _meta_path.exists():
    _meta_path = STACKER_DIR / 'stacker_meta.json'
with open(_meta_path) as fh:
    stacker_meta = json.load(fh)

TEMPERATURE = stacker_meta.get('temperature', 1.5)
print(f'Best arch: {stacker_meta.get("best_arch", "unknown")}')
print(f'OOF AUCs: {stacker_meta.get("oof_aucs", {})}')

_norm_path = STACKER_DIR / 'stacker_norm.npz'
if not _norm_path.exists():
    _norm_path = STACKER_DIR / 'stacker_feature_stats.npz'
_norm = np.load(_norm_path)
FEAT_MEAN = _norm['mean'].astype(np.float32)   # (1,1170)
FEAT_STD  = _norm['std'].astype(np.float32)    # (1,1170)
print(f'Norm stats: mean {FEAT_MEAN.shape}, std {FEAT_STD.shape}')"""

CELL_BUILD_SUBMISSION = """\
# Cell 9 — Build submission.csv
probs_all = []
for path in tqdm(test_paths, desc='Inference'):
    X_raw, X_norm = build_test_features(path, FEAT_MEAN, FEAT_STD)
    logits_12 = run_stacker(X_raw, X_norm)          # (12,234) logits
    probs_all.append(logits_12)

stacked_logits = np.concatenate(probs_all, axis=0).astype(np.float32)  # (N*12,234)
scaled = stacked_logits / TEMPERATURE
probs  = 1.0 / (1.0 + np.exp(-np.clip(scaled, -30, 30)))

assert len(probs) == len(meta_test), f'{len(probs)} != {len(meta_test)}'

submission = pd.DataFrame(probs, columns=PRIMARY_LABELS)
submission.insert(0, 'row_id', meta_test['row_id'].values)
assert submission.columns.tolist() == sample_sub.columns.tolist()
submission.to_csv('/kaggle/working/submission.csv', index=False)

wall_time = time.time() - _WALL_START
print(f'submission.csv: {len(submission)} rows  wall={wall_time:.1f}s')
submission.head(3)"""


# ─── Architecture-specific stacker cells ──────────────────────────────────────

STACKER_CELLS = {
    "lgbm": """\
# Cell 8-LGBM — Load and run LGBM stacker
import lightgbm as lgb

_lgbm_models = None
_lgbm_path   = STACKER_DIR / 'stacker_lgbm.pkl'
if _lgbm_path.exists():
    with open(_lgbm_path, 'rb') as fh:
        _lgbm_models = pickle.load(fh)
    print(f'LGBM loaded: {len(_lgbm_models)} models')
else:
    print('WARNING: stacker_lgbm.pkl not found')


def run_stacker(X_raw: np.ndarray, X_norm: np.ndarray) -> np.ndarray:
    '''X_raw/X_norm: (12,1170) -> (12,234) logits'''
    if _lgbm_models is None:
        return X_raw[:, :N_CLASSES]
    out = np.zeros((N_WIN, N_CLASSES), dtype=np.float32)
    X_ctx = np.stack([build_context_row(X_norm, t, CONTEXT_K) for t in range(N_WIN)])
    for c, m in enumerate(_lgbm_models):
        try:
            p = m.predict_proba(X_ctx)[:, 1]
            out[:, c] = safe_logit(p)
        except Exception:
            pass
    return out

print('LGBM stacker helper OK.')""",

    "xgb": """\
# Cell 8-XGB — Load and run XGBoost stacker
from xgboost import XGBClassifier

_xgb_models = None
_xgb_path   = STACKER_DIR / 'stacker_xgb.pkl'
if _xgb_path.exists():
    with open(_xgb_path, 'rb') as fh:
        _xgb_models = pickle.load(fh)
    print(f'XGB loaded: {len(_xgb_models)} models')
else:
    print('WARNING: stacker_xgb.pkl not found')


def run_stacker(X_raw: np.ndarray, X_norm: np.ndarray) -> np.ndarray:
    '''X_raw/X_norm: (12,1170) -> (12,234) logits'''
    if _xgb_models is None:
        return X_raw[:, :N_CLASSES]
    out = np.zeros((N_WIN, N_CLASSES), dtype=np.float32)
    X_ctx = np.stack([build_context_row(X_norm, t, CONTEXT_K) for t in range(N_WIN)])
    for c, m in enumerate(_xgb_models):
        try:
            p = m.predict_proba(X_ctx)[:, 1]
            out[:, c] = safe_logit(p)
        except Exception:
            pass
    return out

print('XGB stacker helper OK.')""",

    "mlp": """\
# Cell 8-MLP — Load and run MLP stacker via ONNX
_mlp_sess = None
_mlp_path = STACKER_DIR / 'stacker_mlp.onnx'
if _mlp_path.exists():
    _mlp_sess = ort.InferenceSession(str(_mlp_path), providers=['CPUExecutionProvider'])
    print('MLP ONNX loaded.')
else:
    print('WARNING: stacker_mlp.onnx not found')


def _ctx_to_mlp_input(X_norm_12: np.ndarray) -> np.ndarray:
    '''
    For each of 12 windows, build context features -> shape (12, 3, 5, 234).
    MLP ONNX input: (B, CONTEXT_SIZE=3, N_MODELS=5, N_CLASSES=234).
    '''
    ctx_rows = np.stack([build_context_row(X_norm_12, t, CONTEXT_K) for t in range(N_WIN)])
    # ctx_rows: (12, 3510) -> (12, 3, 1170) -> (12, 3, 5, 234)
    return ctx_rows.reshape(N_WIN, CONTEXT_SIZE, N_MODELS, N_CLASSES)


def run_stacker(X_raw: np.ndarray, X_norm: np.ndarray) -> np.ndarray:
    '''X_raw/X_norm: (12,1170) -> (12,234) logits'''
    if _mlp_sess is None:
        return X_raw[:, :N_CLASSES]
    inp = _ctx_to_mlp_input(X_norm).astype(np.float32)   # (12,3,5,234)
    name = _mlp_sess.get_inputs()[0].name
    out  = _mlp_sess.run(None, {name: inp})[0]            # (12,234)
    return out.astype(np.float32)

print('MLP stacker helper OK.')""",

    "bigru": """\
# Cell 8-BiGRU — Load and run BiGRU stacker via ONNX
_bigru_sess = None
_bigru_path = STACKER_DIR / 'stacker_bigru.onnx'
if _bigru_path.exists():
    _bigru_sess = ort.InferenceSession(str(_bigru_path), providers=['CPUExecutionProvider'])
    print('BiGRU ONNX loaded.')
else:
    print('WARNING: stacker_bigru.onnx not found')


def run_stacker(X_raw: np.ndarray, X_norm: np.ndarray) -> np.ndarray:
    '''X_norm: (12,1170) -> (12,234) logits. BiGRU processes full file sequence.'''
    if _bigru_sess is None:
        return X_raw[:, :N_CLASSES]
    inp  = X_norm[None].astype(np.float32)   # (1,12,1170)
    name = _bigru_sess.get_inputs()[0].name
    out  = _bigru_sess.run(None, {name: inp})[0].squeeze(0)  # (12,234)
    return out.astype(np.float32)

print('BiGRU stacker helper OK.')""",

    "tcn": """\
# Cell 8-TCN — Load and run TCN stacker via ONNX
_tcn_sess = None
_tcn_path = STACKER_DIR / 'stacker_tcn.onnx'
if _tcn_path.exists():
    _tcn_sess = ort.InferenceSession(str(_tcn_path), providers=['CPUExecutionProvider'])
    print('TCN ONNX loaded.')
else:
    print('WARNING: stacker_tcn.onnx not found')


def run_stacker(X_raw: np.ndarray, X_norm: np.ndarray) -> np.ndarray:
    '''X_norm: (12,1170) -> (12,234) logits.'''
    if _tcn_sess is None:
        return X_raw[:, :N_CLASSES]
    inp  = X_norm[None].astype(np.float32)
    name = _tcn_sess.get_inputs()[0].name
    out  = _tcn_sess.run(None, {name: inp})[0].squeeze(0)
    return out.astype(np.float32)

print('TCN stacker helper OK.')""",

    "transformer": """\
# Cell 8-Transformer — Load and run Transformer stacker via ONNX
_tfm_sess = None
_tfm_path = STACKER_DIR / 'stacker_transformer.onnx'
if _tfm_path.exists():
    _tfm_sess = ort.InferenceSession(str(_tfm_path), providers=['CPUExecutionProvider'])
    print('Transformer ONNX loaded.')
else:
    print('WARNING: stacker_transformer.onnx not found')


def run_stacker(X_raw: np.ndarray, X_norm: np.ndarray) -> np.ndarray:
    '''X_norm: (12,1170) -> (12,234) logits.'''
    if _tfm_sess is None:
        return X_raw[:, :N_CLASSES]
    inp  = X_norm[None].astype(np.float32)
    name = _tfm_sess.get_inputs()[0].name
    out  = _tfm_sess.run(None, {name: inp})[0].squeeze(0)
    return out.astype(np.float32)

print('Transformer stacker helper OK.')""",

    "ssm": """\
# Cell 8-SSM — Load and run SSM stacker via ONNX
_ssm_sess = None
_ssm_path = STACKER_DIR / 'stacker_ssm.onnx'
if _ssm_path.exists():
    _ssm_sess = ort.InferenceSession(str(_ssm_path), providers=['CPUExecutionProvider'])
    print('SSM ONNX loaded.')
else:
    print('WARNING: stacker_ssm.onnx not found')


def run_stacker(X_raw: np.ndarray, X_norm: np.ndarray) -> np.ndarray:
    '''X_norm: (12,1170) -> (12,234) logits.'''
    if _ssm_sess is None:
        return X_raw[:, :N_CLASSES]
    inp  = X_norm[None].astype(np.float32)
    name = _ssm_sess.get_inputs()[0].name
    out  = _ssm_sess.run(None, {name: inp})[0].squeeze(0)
    return out.astype(np.float32)

print('SSM stacker helper OK.')""",

    "ft_transformer": """\
# Cell 8-FT-Transformer — Load and run FT-Transformer stacker via ONNX
_ft_sess = None
_ft_path = STACKER_DIR / 'stacker_ft_transformer.onnx'
if _ft_path.exists():
    _ft_sess = ort.InferenceSession(str(_ft_path), providers=['CPUExecutionProvider'])
    print('FT-Transformer ONNX loaded.')
else:
    print('WARNING: stacker_ft_transformer.onnx not found')


def run_stacker(X_raw: np.ndarray, X_norm: np.ndarray) -> np.ndarray:
    '''X_norm: (12,1170) -> (12,234) logits.'''
    if _ft_sess is None:
        return X_raw[:, :N_CLASSES]
    inp  = X_norm[None].astype(np.float32)
    name = _ft_sess.get_inputs()[0].name
    out  = _ft_sess.run(None, {name: inp})[0].squeeze(0)
    return out.astype(np.float32)

print('FT-Transformer stacker helper OK.')""",

    "cnn1d": """\
# Cell 8-CNN1D — Load and run CNN1D stacker via ONNX
_cnn1d_sess = None
_cnn1d_path = STACKER_DIR / 'stacker_cnn1d.onnx'
if _cnn1d_path.exists():
    _cnn1d_sess = ort.InferenceSession(str(_cnn1d_path), providers=['CPUExecutionProvider'])
    print('CNN1D ONNX loaded.')
else:
    print('WARNING: stacker_cnn1d.onnx not found')


def run_stacker(X_raw: np.ndarray, X_norm: np.ndarray) -> np.ndarray:
    '''X_norm: (12,1170) -> (12,234) logits.'''
    if _cnn1d_sess is None:
        return X_raw[:, :N_CLASSES]
    inp  = X_norm[None].astype(np.float32)
    name = _cnn1d_sess.get_inputs()[0].name
    out  = _cnn1d_sess.run(None, {name: inp})[0].squeeze(0)
    return out.astype(np.float32)

print('CNN1D stacker helper OK.')""",
}

ARCH_NAMES = {
    "lgbm"          : "LGBM",
    "xgb"           : "XGBoost",
    "mlp"           : "MLP",
    "bigru"         : "BiGRU",
    "tcn"           : "TCN",
    "transformer"   : "Transformer",
    "ssm"           : "SSM",
    "ft_transformer": "FT-Transformer",
    "cnn1d"         : "CNN1D",
}

CONTEXT_ARCHS = {"lgbm", "xgb", "mlp"}   # use context features


def make_notebook(arch_key: str) -> dict:
    arch_name = ARCH_NAMES[arch_key]
    use_ctx   = arch_key in CONTEXT_ARCHS

    cells = []

    def code_cell(src):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": src,
        })

    def md_cell(src):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": src,
        })

    # Cell 0 — install
    code_cell(CELL_INSTALL)

    # Cell 1 — markdown title
    md_cell(
        f"# BirdCLEF+ 2026 — Stacker V3 ({arch_name})\n\n"
        f"Feature layout (5 × 234 = 1170 dims):\n"
        f"```\nX = [perch_raw | perch_prior_fused | mlp_probe | proto_ssm | sed_csebbs]\n"
        f"     [0:234]     [234:468]           [468:702]   [702:936]   [936:1170]\n```\n\n"
        f"This notebook runs **{arch_name}** stacker."
    )

    # Cell 2 — imports
    code_cell(CELL_IMPORTS)

    # Cell 3 — paths
    code_cell(CELL_PATHS)

    # Cell 4 — taxonomy
    code_cell(CELL_TAXONOMY)

    # Cell 5 — Perch setup
    code_cell(CELL_PERCH_SETUP)

    # Cell 6 — Prior
    code_cell(CELL_PRIOR)

    # Cell 7 — MLP probe
    code_cell(CELL_MLP_PROBE)

    # Cell 8 — ProtoSSM
    code_cell(CELL_PROTO)

    # Cell 9 — SED
    code_cell(CELL_SED)

    # Cell 10 — Feature building (with context helper for flat models)
    feat_cell = CELL_BUILD_FEATURES
    code_cell(feat_cell)

    # Cell 11 — Load meta
    code_cell(CELL_LOAD_META)

    # Cell 12 — Architecture-specific stacker loader
    code_cell(STACKER_CELLS[arch_key])

    # Cell 13 — Build submission
    code_cell(CELL_BUILD_SUBMISSION)

    # Cell 14 — Diagnostics
    diag_cell = f"""\
# Cell 14 — Diagnostics
print('=== Submission Diagnostics ===')
print(f'Stacker arch : {arch_key}')
print(f'Temperature  : {{TEMPERATURE}}')
print(f'Prob stats   : min={{probs.min():.4f}}  max={{probs.max():.4f}}  mean={{probs.mean():.4f}}')
print(f'Classes>0.5  : {{(probs>0.5).sum(axis=1).mean():.2f}} per row (mean)')

print('\\nOOF AUC reference (from training):')
for k, v in stacker_meta.get('oof_aucs', {{}}).items():
    marker = ' <- used' if k == '{arch_key}' else ''
    print(f'  {{k:<22}}: {{v:.4f}}{{marker}}')

print(f'\\nDone. submission.csv -> /kaggle/working/')"""
    code_cell(diag_cell)

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12.0",
            },
        },
        "cells": cells,
    }
    return nb


# ─── Generate notebooks ───────────────────────────────────────────────────────

for arch_key in STACKER_CELLS:
    arch_name = ARCH_NAMES[arch_key]
    nb_path = NB_DIR / f"stacker_{arch_key}.ipynb"
    nb = make_notebook(arch_key)
    with open(nb_path, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
    print(f"Written: {nb_path.name}")

print("\nAll 9 notebooks created.")
