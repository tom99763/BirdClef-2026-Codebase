"""Create SSM-integrated submission notebooks from the base notebook.

Generates two notebooks in current_subs/:
  - lgbm-infer-branchens-ssm-light.ipynb
  - lgbm-infer-branchens-ssm-full.ipynb

Each notebook integrates 5-fold ProtoSSM TFLite models with fold selection.
"""

import copy
import json
from pathlib import Path

BASE_NB = Path('birdclef-2026/notebook resource/current_subs/lgbm-infer-event-smooth-branchens-csebbs-postp.ipynb')
OUT_DIR = Path('birdclef-2026/notebook resource/current_subs')


def make_code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines,
    }


def make_md_cell(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines,
    }


# ── SSM config additions (injected into cell 3) ───────────────────────────────
SSM_CONFIG_PATCH = {
    'light': """
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ProtoSSM — v4-light (5-fold) integration
# Upload the weights/ folder from current_subs to your Kaggle dataset.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SSM_VARIANT    = 'light'                # 'light' | 'full'
SSM_TFLITE_DIR = f'{DATASET_DIR}'      # folder containing proto_ssm_v4_*.tflite

# ── Fold selection (default: all 5 folds) ─────────────────────────────────────
# Change to e.g. [0, 1, 2] to use only those folds.
SSM_FOLDS = [0, 1, 2, 3, 4]

# ── Blend weight for SSM output ───────────────────────────────────────────────
# SSM is added on top of the Perch+SED blend.
# Increase if you trust SSM more; set to 0 to disable.
SSM_W = 0.30
""",
    'full': """
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ProtoSSM — v4-full (5-fold) integration
# Upload the weights/ folder from current_subs to your Kaggle dataset.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SSM_VARIANT    = 'full'                 # 'light' | 'full'
SSM_TFLITE_DIR = f'{DATASET_DIR}'      # folder containing proto_ssm_v4_*.tflite

# ── Fold selection (default: all 5 folds) ─────────────────────────────────────
# Change to e.g. [0, 1, 2] to use only those folds.
SSM_FOLDS = [0, 1, 2, 3, 4]

# ── Blend weight for SSM output ───────────────────────────────────────────────
# SSM is added on top of the Perch+SED blend.
# Increase if you trust SSM more; set to 0 to disable.
SSM_W = 0.30
""",
}


# ── SSM loader cell ────────────────────────────────────────────────────────────
SSM_LOADER_CELL_SOURCE = """\
# ── Load ProtoSSM TFLite models ───────────────────────────────────────────────
# Input 0 'emb'         : (12, 1536) Perch embeddings
# Input 1 'perch_logits': (12, 234)  raw Perch species logits (before sigmoid)
# Output  'species_logits': (12, 234) SSM logits (apply sigmoid for probs)

_ssm_entries = []
_SSM_LOCK = threading.Lock()

for _fold in SSM_FOLDS:
    _tflite_path = os.path.join(SSM_TFLITE_DIR,
                                f'proto_ssm_v4_{SSM_VARIANT}_fold{_fold}.tflite')
    if not os.path.isfile(_tflite_path):
        print(f'  SKIP SSM fold{_fold}: {_tflite_path} not found')
        continue
    try:
        _interp = tf.lite.Interpreter(model_path=_tflite_path, num_threads=2)
        _interp.allocate_tensors()
        # Verify expected I/O shapes
        _inp = _interp.get_input_details()
        _out = _interp.get_output_details()
        # Find input indices by name
        _emb_idx    = next(d['index'] for d in _inp if 'emb' in d['name'])
        _logit_idx  = next(d['index'] for d in _inp if 'perch' in d['name'])
        _out_idx    = _out[0]['index']
        _ssm_entries.append({
            'interp':     _interp,
            'emb_idx':    _emb_idx,
            'logit_idx':  _logit_idx,
            'out_idx':    _out_idx,
            'fold':       _fold,
        })
        print(f'  SSM fold{_fold} loaded  ({os.path.getsize(_tflite_path)//1024} KB)')
    except Exception as e:
        print(f'  ERROR loading SSM fold{_fold}: {e}')

USE_SSM = len(_ssm_entries) > 0
print(f'{len(_ssm_entries)} SSM models loaded  USE_SSM={USE_SSM}  SSM_W={SSM_W}')


def ssm_predict_12windows(emb_1536, raw_logits_234):
    \"\"\"Run ProtoSSM ensemble over selected folds.
    emb_1536:      (12, 1536) Perch embeddings
    raw_logits_234: (12, 234) Perch raw logits (before sigmoid)
    Returns: ssm_probs (12, 234) averaged across folds
    \"\"\"
    if not _ssm_entries:
        return np.zeros((N_WINDOWS, NUM_CLASSES), dtype=np.float32)
    acc = np.zeros((N_WINDOWS, NUM_CLASSES), dtype=np.float32)
    emb_f32   = emb_1536.astype(np.float32)
    logit_f32 = raw_logits_234.astype(np.float32)
    with _SSM_LOCK:
        for entry in _ssm_entries:
            interp = entry['interp']
            interp.set_tensor(entry['emb_idx'],   emb_f32)
            interp.set_tensor(entry['logit_idx'], logit_f32)
            interp.invoke()
            logits = interp.get_tensor(entry['out_idx'])  # (12, 234)
            probs  = 1.0 / (1.0 + np.exp(-np.clip(logits, -88, 88)))
            acc   += probs
    return (acc / len(_ssm_entries)).astype(np.float32)
"""


# ── Patch for predict_file: add SSM inference after Perch block ───────────────
# We'll find the VLOM blend section and replace it with SSM-aware version.

OLD_BLEND_SNIPPET = """\
    if USE_SED and USE_VLOM_BLEND:
        preds = vlom_blend(perch_probs, sed_probs, w_a=PERCH_W, w_b=SED_W)
    elif USE_SED:
        preds = (PERCH_W * perch_probs + SED_W * sed_probs) / (PERCH_W + SED_W)
    else:
        preds = perch_probs"""

NEW_BLEND_SNIPPET = """\
    if USE_SED and USE_VLOM_BLEND:
        base_preds = vlom_blend(perch_probs, sed_probs, w_a=PERCH_W, w_b=SED_W)
    elif USE_SED:
        base_preds = (PERCH_W * perch_probs + SED_W * sed_probs) / (PERCH_W + SED_W)
    else:
        base_preds = perch_probs

    if USE_SSM:
        # ProtoSSM uses raw_scores (Perch logits) + emb_1536 (already available)
        try:
            ssm_probs = ssm_predict_12windows(emb_1536, raw_scores)
        except Exception as e:
            print(f'  ERROR SSM {ogg_path.name}: {e}')
            ssm_probs = np.zeros((N_WINDOWS, NUM_CLASSES), dtype=np.float32)
        w_base = PERCH_W + (SED_W if USE_SED else 0)
        preds  = (w_base * base_preds + SSM_W * ssm_probs) / (w_base + SSM_W)
    else:
        preds = base_preds"""


def patch_predict_file_cell(cell_source):
    """Patch the predict_file cell to add SSM inference."""
    src = ''.join(cell_source)
    if OLD_BLEND_SNIPPET not in src:
        raise ValueError("Could not find VLOM blend snippet to patch in predict_file cell")
    patched = src.replace(OLD_BLEND_SNIPPET, NEW_BLEND_SNIPPET)
    return list(patched)  # return as list of chars (notebook format accepts either)


def create_ssm_notebook(variant: str):
    with open(BASE_NB) as f:
        nb = json.load(f)

    cells = copy.deepcopy(nb['cells'])

    # ── 1. Update title (cell 0, markdown) ────────────────────────────────────
    title_src = ''.join(cells[0]['source'])
    title_src = title_src.replace(
        'BirdCLEF 2026 — v3 LGBM Infer + Event Smooth + BranchEns→cSEBBs PostProc',
        f'BirdCLEF 2026 — LGBM + BranchEns→cSEBBs + ProtoSSM-v4-{variant} PostProc',
    )
    cells[0]['source'] = [title_src]

    # ── 2. Append SSM config to cell 3 (config) ───────────────────────────────
    config_src = ''.join(cells[3]['source'])
    config_src += SSM_CONFIG_PATCH[variant]
    cells[3]['source'] = [config_src]

    # ── 3. Patch cell 11 (predict_file) — add SSM blend ──────────────────────
    cells[11]['source'] = patch_predict_file_cell(cells[11]['source'])

    # ── 4. Insert SSM loader cell after cell 8 (SED loader) ──────────────────
    ssm_loader = make_code_cell([SSM_LOADER_CELL_SOURCE])

    # SSM loader goes after cell 8 (SED), before cell 9 (VAD)
    new_cells = cells[:9] + [ssm_loader] + cells[9:]

    # ── 5. Update save submission cell (now index 13 due to inserted cell) ────
    # Update print statements to mention SSM
    save_cell_idx = 13  # was 12, shifted by 1
    save_src = ''.join(new_cells[save_cell_idx]['source'])
    save_src += f"\nprint(f'ProtoSSM-v4-{variant}: folds={{SSM_FOLDS}}  SSM_W={{SSM_W}}')\n"
    new_cells[save_cell_idx]['source'] = [save_src]

    nb_out = copy.deepcopy(nb)
    nb_out['cells'] = new_cells

    out_name = f'lgbm-infer-branchens-ssm-{variant}.ipynb'
    out_path = OUT_DIR / out_name
    with open(out_path, 'w') as f:
        json.dump(nb_out, f, indent=1, ensure_ascii=False)
    print(f'Created: {out_path}')
    return out_path


if __name__ == '__main__':
    create_ssm_notebook('light')
    create_ssm_notebook('full')
    print('\nDone. Both notebooks created in:')
    print(f'  {OUT_DIR}')
    print('\nweights/ folder contents:')
    for p in sorted((OUT_DIR / 'weights').glob('*.tflite')):
        print(f'  {p.name}  ({p.stat().st_size // 1024} KB)')
