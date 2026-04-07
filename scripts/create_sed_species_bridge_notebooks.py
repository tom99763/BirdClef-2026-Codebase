"""
Create SED-species Bridge notebooks from SS Bridge template.
New method: sed_species_bridge (CV=0.9444, NEW BEST)

Key differences from SS Bridge:
1. Use embed_prior_sed_species_bridge.pkl (species-weighted signatures)
2. Update params: alpha=0.5, wg=0.45, ww=0.55, a=0.85, b=1.70
3. Fix _RKNN_ACTIVE connection (set _y_blend_rknn = y_ep_blended)
"""
import json, os, copy, shutil

SUBS_DIR = "birdclef-2026/notebook resource/current_subs"
BASE_NB = f"{SUBS_DIR}/dual-foundation-protossm-ss-bridge-a040-wg040-a100-b150.ipynb"

with open(BASE_NB) as f:
    base = json.load(f)

def make_notebook(cfg):
    """
    cfg keys:
      pkl_name, alpha, wg, a, b, cv, note
    """
    nb = copy.deepcopy(base)
    cell = nb['cells'][51]
    src = ''.join(cell['source'])
    lines = src.split('\n')

    # --- Update SS Bridge parameters ---
    # Replace _SSBRIDGE_ALPHA, _SSBRIDGE_A, _SSBRIDGE_B, _SSBRIDGE_WG, _SSBRIDGE_WW
    new_lines = []
    for line in lines:
        if line.startswith('_SSBRIDGE_ALPHA'):
            new_lines.append(f"_SSBRIDGE_ALPHA = {cfg['alpha']}")
        elif line.startswith('_SSBRIDGE_A '):
            new_lines.append(f"_SSBRIDGE_A = {cfg['a']}")
        elif line.startswith('_SSBRIDGE_B '):
            new_lines.append(f"_SSBRIDGE_B = {cfg['b']}")
        elif line.startswith('_SSBRIDGE_WG'):
            new_lines.append(f"_SSBRIDGE_WG = {cfg['wg']}   # weight for SS-bridge RKNN")
        elif line.startswith('_SSBRIDGE_WW'):
            ww = round(1.0 - cfg['wg'], 2)
            new_lines.append(f"_SSBRIDGE_WW = {ww}   # weight for window KNN")
        # Update PKL file path
        elif '_EMBED_PRIOR_PATH' in line and 'embed_prior_ss_bridge.pkl' in line:
            new_lines.append(f'_EMBED_PRIOR_PATH = "/kaggle/input/birdclef-2026-dual-foundation/weights/{cfg["pkl_name"]}"')
        # Update the markdown comment about method
        elif 'Full-pipeline CV AUC: 0.9440' in line or 'embed_prior_ss_bridge.pkl' in line:
            new_lines.append(line.replace('0.9440', str(cfg['cv']))
                               .replace('embed_prior_ss_bridge.pkl', cfg['pkl_name']))
        # Bridge formula comment
        elif '#   sim_combined = (1-0.4)' in line:
            new_lines.append(f"#   sim_combined = (1-{cfg['alpha']}) × geo_sim + {cfg['alpha']} × bridge_sim_normalized")
        elif '#   sigmoid(1.0 × vlom_logit' in line:
            ww = round(1.0 - cfg['wg'], 2)
            new_lines.append(f"#   sigmoid({cfg['a']} × vlom_logit + {cfg['b']} × log({cfg['wg']} × rknn_k5 + {ww} × win_k1))")
        # Fix _RKNN_ACTIVE connection: after log_ep line, add bridge-to-rknn linkage
        elif 'log_ep = np.log(y_ep_blended.clip(EPS_e))' in line:
            new_lines.append(line)
            # Connect SS Bridge output to the RKNN correction block
            new_lines.append('')
            new_lines.append('# Connect SS Bridge output to final VLOM logspace correction')
            new_lines.append('_RKNN_ACTIVE = True')
            new_lines.append('_RKNN_A = _SSBRIDGE_A')
            new_lines.append('_RKNN_B = _SSBRIDGE_B')
            new_lines.append('_y_blend_rknn = y_ep_blended')
        else:
            new_lines.append(line)

    cell['source'] = '\n'.join(new_lines)

    # Update markdown cell 49 (method description)
    for ci, c in enumerate(nb['cells']):
        if c['cell_type'] == 'markdown':
            csrc = ''.join(c['source'])
            if 'SS Bridge' in csrc and 'CV=0.9440' in csrc:
                ww = round(1.0 - cfg['wg'], 2)
                new_src = f"""## Embed Prior: SED-Species Bridge RKNN (CV={cfg['cv']})

**Method**: SED-Species Soundscape Bridge RKNN
- Uses 127,896 soundscape windows as "bridge", weighted by Perch sim × (1 + β × SED species score)
- Bridge formula: w_m = perch_sim(j,m) × (1 + {cfg.get('beta',0.5)} × max_SED(j.species, window_m))
- Combined similarity: (1-{cfg['alpha']}) × geo_sim + {cfg['alpha']} × sed_species_bridge_n
- RKNN k=5 on sim_combined
- sigmoid({cfg['a']} × vlom_logit + {cfg['b']} × log({cfg['wg']} × rknn_k5 + {ww} × win_k1))
- **LOO-CV AUC: {cfg['cv']}** (vs SS Bridge 0.9440, RKNN 0.9432)
"""
                nb['cells'][ci]['source'] = new_src
                break

    return nb

# Configs to create (best first)
configs = [
    {
        'pkl_name': 'embed_prior_sed_species_bridge.pkl',
        'alpha': 0.50,
        'wg': 0.45,
        'a': 0.85,
        'b': 1.70,
        'beta': 0.50,
        'cv': 0.9444,
        'name': 'sed-species-bridge-b050-a050-wg045-a085-b170',
    },
    {
        'pkl_name': 'embed_prior_sed_species_bridge.pkl',
        'alpha': 0.35,
        'wg': 0.35,
        'a': 0.85,
        'b': 2.10,
        'beta': 0.20,
        'cv': 0.9444,
        'name': 'sed-species-bridge-b020-a035-wg035-a085-b210',
    },
    {
        'pkl_name': 'embed_prior_sed_species_bridge.pkl',
        'alpha': 0.35,
        'wg': 0.45,
        'a': 1.00,
        'b': 1.70,
        'beta': 0.30,
        'cv': 0.9444,
        'name': 'sed-species-bridge-b030-a035-wg045-a100-b170',
    },
]

for cfg in configs:
    nb = make_notebook(cfg)
    out_path = f"{SUBS_DIR}/dual-foundation-protossm-{cfg['name']}.ipynb"
    with open(out_path, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f"Created: {cfg['name']} (CV={cfg['cv']})")

print(f"\nCreated {len(configs)} SED-species bridge notebooks.")
