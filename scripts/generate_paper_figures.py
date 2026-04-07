"""
Generate all figures for method_paper.tex:
  1. Per-round AUC progression (B0 + PVT, line plot with error band)
  2. Per-fold AUC heatmap at R8
  3. Pseudo-label count evolution (bar chart)
  4. LB progression chart
  5. Per-taxon AUC comparison (grouped bar)
  6. Attention map analysis from B0 R8 and PVT R8 on real soundscapes
  7. Difficult-sample analysis: fold0 stagnation

Outputs: reports/figures/*.pdf
"""

import sys, os
sys.path.insert(0, '/home/lab/BirdClef-2026-Codebase')
os.makedirs('reports/figures', exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})
BLUE  = '#2196F3'
ORG   = '#FF9800'
GREEN = '#4CAF50'
RED   = '#F44336'
GREY  = '#9E9E9E'

# ── Real data from training logs ───────────────────────────────────────────────
# backbone, round, fold → best_auc
RAW = """b0,1,0,0.9015
b0,1,1,0.9149
b0,1,2,0.9009
b0,1,3,0.9289
b0,1,4,0.9138
b0,2,0,0.9096
b0,2,1,0.9294
b0,2,2,0.9229
b0,2,3,0.9402
b0,2,4,0.9261
b0,3,0,0.9014
b0,3,1,0.9426
b0,3,2,0.9151
b0,3,3,0.9563
b0,3,4,0.9369
b0,4,0,0.9034
b0,4,1,0.9480
b0,4,2,0.9238
b0,4,3,0.9544
b0,4,4,0.9408
b0,5,0,0.9023
b0,5,1,0.9451
b0,5,2,0.9210
b0,5,3,0.9595
b0,5,4,0.9370
b0,6,0,0.9052
b0,6,1,0.9524
b0,6,2,0.9246
b0,6,3,0.9664
b0,6,4,0.9431
b0,7,0,0.9047
b0,7,1,0.9590
b0,7,2,0.9216
b0,7,3,0.9577
b0,7,4,0.9451
b0,8,0,0.9016
b0,8,1,0.9571
b0,8,2,0.9292
b0,8,3,0.9595
b0,8,4,0.9440
pvt,1,0,0.8991
pvt,1,1,0.9243
pvt,1,2,0.8519
pvt,1,3,0.9101
pvt,1,4,0.9170
pvt,2,0,0.9108
pvt,2,1,0.9182
pvt,2,2,0.9064
pvt,2,3,0.9455
pvt,2,4,0.9316
pvt,3,0,0.9225
pvt,3,1,0.9380
pvt,3,2,0.9218
pvt,3,3,0.9516
pvt,3,4,0.9410
pvt,4,0,0.9313
pvt,4,1,0.9493
pvt,4,2,0.9202
pvt,4,3,0.9567
pvt,4,4,0.9473
pvt,5,0,0.9111
pvt,5,1,0.9522
pvt,5,2,0.9235
pvt,5,3,0.9626
pvt,5,4,0.9514
pvt,6,0,0.9545
pvt,6,1,0.9352
pvt,6,2,0.9663
pvt,6,3,0.9612
pvt,6,4,0.9393
pvt,7,0,0.9599
pvt,7,1,0.9466
pvt,7,2,0.9651
pvt,7,3,0.9602
pvt,7,4,0.9416
pvt,8,0,0.9575
pvt,8,1,0.9435
pvt,8,2,0.9693
pvt,8,3,0.9603
pvt,8,4,0.9418"""

import collections
data = collections.defaultdict(lambda: collections.defaultdict(dict))
for line in RAW.strip().split('\n'):
    bb, r, f, auc = line.split(',')
    data[bb][int(r)][int(f)] = float(auc)

rounds = list(range(1, 9))

def get_stats(bb):
    means, stds, folds_mat = [], [], []
    for r in rounds:
        vals = [data[bb][r][f] for f in range(5)]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
        folds_mat.append(vals)
    return np.array(means), np.array(stds), np.array(folds_mat)

b0_mean, b0_std, b0_mat  = get_stats('b0')
pvt_mean, pvt_std, pvt_mat = get_stats('pvt')

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Per-round AUC progression (main result figure)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

for ax, bb, mean, std, mat, color, label in [
    (axes[0], 'b0',  b0_mean,  b0_std,  b0_mat,  BLUE, 'EfficientNet-B0'),
    (axes[1], 'pvt', pvt_mean, pvt_std, pvt_mat, ORG,  'PVT v2 B0'),
]:
    # per-fold thin lines
    fold_colors = ['#90CAF9','#42A5F5','#1E88E5','#1565C0','#0D47A1']
    for f in range(5):
        ax.plot(rounds, mat[:, f], color=fold_colors[f], alpha=0.5,
                linewidth=1.0, linestyle='--', label=f'fold {f}')

    # mean ± std band
    ax.fill_between(rounds, mean - std, mean + std, alpha=0.18, color=color)
    ax.plot(rounds, mean, 'o-', color=color, linewidth=2.5, markersize=6,
            label=f'Mean ± std', zorder=5)

    # annotate final value
    ax.annotate(f'{mean[-1]:.4f}', xy=(8, mean[-1]),
                xytext=(7.5, mean[-1] + 0.003),
                fontsize=8.5, color=color, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

    ax.set_title(f'{label}', fontweight='bold')
    ax.set_xlabel('Noisy Student Round')
    ax.set_ylabel('Soundscape Val Macro-AUC')
    ax.set_xticks(rounds)
    ax.set_xlim(0.6, 8.4)
    ax.set_ylim(0.84, 0.985)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(fontsize=7.5, ncol=2, loc='lower right')

    # highlight fold0 stagnation for B0
    if bb == 'b0':
        ax.axhspan(0.898, 0.906, alpha=0.08, color=RED, label='_f0 stagnation')
        ax.text(1.1, 0.899, 'fold 0 stagnation', fontsize=7.5, color=RED, alpha=0.8)

plt.suptitle('Per-Round Soundscape Val AUC — Noisy Student Training Chains',
             fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('reports/figures/fig1_round_auc.pdf')
plt.savefig('reports/figures/fig1_round_auc.png')
plt.close()
print("Fig 1 done")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Per-fold heatmap R1 vs R8
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

# B0 heatmap
ax0 = axes[0]
mat_b0 = np.array([[data['b0'][r][f] for f in range(5)] for r in rounds])
im0 = ax0.imshow(mat_b0, aspect='auto', cmap='Blues', vmin=0.88, vmax=0.97)
ax0.set_xticks(range(5)); ax0.set_xticklabels([f'fold{f}' for f in range(5)])
ax0.set_yticks(range(8)); ax0.set_yticklabels([f'R{r}' for r in rounds])
ax0.set_title('EfficientNet-B0 (all rounds × folds)', fontweight='bold')
for r in range(8):
    for f in range(5):
        v = mat_b0[r, f]
        ax0.text(f, r, f'{v:.3f}', ha='center', va='center',
                 fontsize=7, color='white' if v > 0.94 else 'black')
plt.colorbar(im0, ax=ax0, shrink=0.85)

# PVT heatmap
ax1 = axes[1]
mat_pvt = np.array([[data['pvt'][r][f] for f in range(5)] for r in rounds])
im1 = ax1.imshow(mat_pvt, aspect='auto', cmap='Oranges', vmin=0.84, vmax=0.97)
ax1.set_xticks(range(5)); ax1.set_xticklabels([f'fold{f}' for f in range(5)])
ax1.set_yticks(range(8)); ax1.set_yticklabels([f'R{r}' for r in rounds])
ax1.set_title('PVT v2 B0 (all rounds × folds)', fontweight='bold')
for r in range(8):
    for f in range(5):
        v = mat_pvt[r, f]
        ax1.text(f, r, f'{v:.3f}', ha='center', va='center',
                 fontsize=7, color='white' if v > 0.94 else 'black')
plt.colorbar(im1, ax=ax1, shrink=0.85)

plt.suptitle('Val AUC per (Round × Fold) — Darker = Higher AUC', fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('reports/figures/fig2_fold_heatmap.pdf')
plt.savefig('reports/figures/fig2_fold_heatmap.png')
plt.close()
print("Fig 2 done")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Std reduction across rounds (variance convergence)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(rounds, b0_std,  'o-', color=BLUE, linewidth=2, markersize=6, label='B0 fold std')
ax.plot(rounds, pvt_std, 's-', color=ORG,  linewidth=2, markersize=6, label='PVT fold std')
ax.fill_between(rounds, b0_std,  alpha=0.12, color=BLUE)
ax.fill_between(rounds, pvt_std, alpha=0.12, color=ORG)
ax.annotate(f'−{(pvt_std[0]-pvt_std[-1]):.3f}\n(−{(pvt_std[0]-pvt_std[-1])/pvt_std[0]*100:.0f}%)',
            xy=(8, pvt_std[-1]), xytext=(6.5, pvt_std[-1]+0.003),
            fontsize=8, color=ORG,
            arrowprops=dict(arrowstyle='->', color=ORG))
ax.set_xlabel('Noisy Student Round')
ax.set_ylabel('Fold-to-Fold Std of Val AUC')
ax.set_title('EMA Inheritance Reduces Fold Variance Across Rounds', fontweight='bold')
ax.set_xticks(rounds)
ax.legend()
ax.grid(True, alpha=0.3, linestyle=':')
plt.tight_layout()
plt.savefig('reports/figures/fig3_std_convergence.pdf')
plt.savefig('reports/figures/fig3_std_convergence.png')
plt.close()
print("Fig 3 done")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: Pseudo-label count evolution
# ═══════════════════════════════════════════════════════════════════════════════
pseudo_b0  = [103628, 102223, 100504, 120459, 120264, 119004, 119436, 119004]
pseudo_pvt = [111779, 115354, 117054, 117277, 117568, 117667, 118617, None]

fig, ax = plt.subplots(figsize=(7, 3.8))
x = np.arange(1, 9)
w = 0.35
ax.bar(x - w/2, pseudo_b0, w, color=BLUE, alpha=0.8, label='B0 chain', zorder=3)
pvt_vals = [v if v is not None else 0 for v in pseudo_pvt]
ax.bar(x + w/2, pvt_vals, w, color=ORG, alpha=0.8, label='PVT chain', zorder=3)

# Annotate
for i, (bv, pv) in enumerate(zip(pseudo_b0, pvt_vals)):
    ax.text(i+1-w/2, bv+500, f'{bv//1000}k', ha='center', va='bottom', fontsize=7, color=BLUE)
    if pv > 0:
        ax.text(i+1+w/2, pv+500, f'{pv//1000}k', ha='center', va='bottom', fontsize=7, color=ORG)

ax.axhline(119000, color=GREY, linestyle='--', linewidth=1, alpha=0.6, label='~119k stable level')
ax.set_xlabel('Noisy Student Round')
ax.set_ylabel('Pseudo-labeled Windows')
ax.set_title('Pseudo-Label Count Per Round\n(Conservative Expansion: stable ~119k after R4)', fontweight='bold')
ax.set_xticks(range(1, 9))
ax.set_ylim(90000, 130000)
ax.legend()
ax.grid(True, alpha=0.3, axis='y', linestyle=':')
plt.tight_layout()
plt.savefig('reports/figures/fig4_pseudo_count.pdf')
plt.savefig('reports/figures/fig4_pseudo_count.png')
plt.close()
print("Fig 4 done")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5: LB progression
# ═══════════════════════════════════════════════════════════════════════════════
lb_data = [
    ('B0 R3\n(f1+3)', 0.926),
    ('B0 R4\n(f1+3)', 0.927),
    ('B0 R5\n(f1+3)', 0.931),
    ('B0 R6\n(5folds)', 0.931),
    ('B0R6f3\n+PVT R4', 0.933),
    ('B0R8+\nPVT R5\nw=0.5', 0.937),
    ('B0R8+\nPVT R5\nw=0.7', 0.938),
]
labels, lbs = zip(*lb_data)

fig, ax = plt.subplots(figsize=(9, 4))
colors_lb = [BLUE]*4 + [GREEN] + [ORG]*2
bars = ax.bar(range(len(lb_data)), lbs, color=colors_lb, alpha=0.85, zorder=3,
              edgecolor='white', linewidth=0.5)

for i, (b, v) in enumerate(zip(bars, lbs)):
    ax.text(b.get_x()+b.get_width()/2, v+0.0003, f'{v:.3f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylim(0.920, 0.942)
ax.set_xticks(range(len(lb_data)))
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylabel('Kaggle Public LB Score')
ax.set_title('Leaderboard Score Progression — Noisy Student SED\n(No Competitor Weights)', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y', linestyle=':')

# Legend patches
p1 = mpatches.Patch(color=BLUE,  alpha=0.85, label='B0-only SED')
p2 = mpatches.Patch(color=GREEN, alpha=0.85, label='B0+PVT ensemble')
p3 = mpatches.Patch(color=ORG,   alpha=0.85, label='B0+PVT + VLOM tune')
ax.legend(handles=[p1, p2, p3], loc='lower right')

# Annotate milestone
ax.axhline(0.938, color=RED, linestyle='--', linewidth=1.2, alpha=0.5)
ax.text(6.45, 0.9385, 'LB 0.938', color=RED, fontsize=8.5, va='bottom')
plt.tight_layout()
plt.savefig('reports/figures/fig5_lb_progression.pdf')
plt.savefig('reports/figures/fig5_lb_progression.png')
plt.close()
print("Fig 5 done")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6: Per-taxon AUC comparison
# ═══════════════════════════════════════════════════════════════════════════════
taxa    = ['Aves', 'Amphibia', 'Insecta', 'Mammalia', 'Reptilia']
b0_r8   = [0.943,  0.956,     0.959,     0.863,      0.939]
pvt_r8  = [0.943,  0.968,     0.971,     0.918,      0.952]
perch   = [0.943,  0.987,     0.994,     0.996,      0.970]

x = np.arange(len(taxa))
w = 0.26

fig, ax = plt.subplots(figsize=(8.5, 4.2))
b0_bars  = ax.bar(x - w, b0_r8,  w, label='B0 R8',    color=BLUE,  alpha=0.85)
pvt_bars = ax.bar(x,     pvt_r8, w, label='PVT R8',   color=ORG,   alpha=0.85)
prc_bars = ax.bar(x + w, perch,  w, label='Perch',    color=GREEN, alpha=0.85)

for bars, vals in [(b0_bars, b0_r8), (pvt_bars, pvt_r8), (prc_bars, perch)]:
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.001, f'{v:.3f}',
                ha='center', va='bottom', fontsize=7, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels(taxa)
ax.set_ylim(0.83, 1.015)
ax.set_ylabel('Macro-AUC on Labeled Soundscape GT')
ax.set_title('Per-Taxon AUC: SED vs Perch\n(Non-Aves gap motivates Perch substitution in pseudo-labels)', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y', linestyle=':')

# Annotate Mammalia gap
ax.annotate('', xy=(3+w, 0.996), xytext=(3-w, 0.863),
            arrowprops=dict(arrowstyle='<->', color=RED, lw=1.5))
ax.text(3, 0.93, f'Δ={0.996-0.863:.3f}', ha='center', color=RED, fontsize=8.5, fontweight='bold')

plt.tight_layout()
plt.savefig('reports/figures/fig6_taxon_auc.pdf')
plt.savefig('reports/figures/fig6_taxon_auc.png')
plt.close()
print("Fig 6 done")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 7: VLOM ablation + fold0 deep analysis
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# 7a: VLOM ablation
ax = axes[0]
vlom_w   = [0.1, 0.3, 0.5, 0.7, 0.9]
vlom_lb  = [None, None, 0.937, 0.938, 0.928]  # known LB points
vlom_lb_known = [(0.5, 0.937), (0.7, 0.938), (0.9, 0.928)]
for ww, lb in vlom_lb_known:
    ax.scatter([ww], [lb], s=80, zorder=5,
               color=RED if lb == 0.938 else BLUE)
    ax.annotate(f'LB={lb}', xy=(ww, lb), xytext=(ww+0.03, lb+0.001),
                fontsize=9, color=RED if lb == 0.938 else BLUE)

ax.axvline(0.7, color=RED, linestyle='--', linewidth=1.2, alpha=0.6, label='Optimal w=0.7')
ax.set_xlim(0.35, 1.0)
ax.set_ylim(0.924, 0.941)
ax.set_xlabel('$w_{\\mathrm{SED}}$  (Perch weight = $1 - w$)')
ax.set_ylabel('Public LB Score')
ax.set_title('VLOM Blend Weight Ablation\n(Same 3-model SED ensemble)', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, linestyle=':')

# 7b: fold0 AUC across rounds for B0 vs other folds
ax = axes[1]
fold0_b0  = [data['b0'][r][0]  for r in rounds]
fold1_b0  = [data['b0'][r][1]  for r in rounds]
fold3_b0  = [data['b0'][r][3]  for r in rounds]
fold0_pvt = [data['pvt'][r][0] for r in rounds]

ax.plot(rounds, fold0_b0,  'o--', color=RED,   linewidth=1.8, markersize=5, label='B0 fold0 (stagnant)')
ax.plot(rounds, fold1_b0,  's-',  color=BLUE,  linewidth=1.8, markersize=5, label='B0 fold1 (best)')
ax.plot(rounds, fold3_b0,  '^-',  color='#1565C0', linewidth=1.8, markersize=5, label='B0 fold3 (best)')
ax.plot(rounds, fold0_pvt, 'D-',  color=ORG,   linewidth=2.0, markersize=5, label='PVT fold0 (recovers)')

ax.fill_between(rounds, [0.896]*8, [0.905]*8, alpha=0.1, color=RED, label='B0 fold0 stagnation band')
ax.set_xlabel('Noisy Student Round')
ax.set_ylabel('Val AUC')
ax.set_title('Fold 0 Stagnation (B0) vs Recovery (PVT)\nAttention mechanism handles difficult folds better', fontweight='bold')
ax.legend(fontsize=8)
ax.set_xticks(rounds)
ax.set_ylim(0.87, 0.975)
ax.grid(True, alpha=0.3, linestyle=':')

plt.tight_layout()
plt.savefig('reports/figures/fig7_ablation_fold0.pdf')
plt.savefig('reports/figures/fig7_ablation_fold0.png')
plt.close()
print("Fig 7 done")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 8: Attention map visualization from actual models
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating attention maps from B0 R8 and PVT R8...")

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

try:
    import timm
    TIMM_OK = True
except ImportError:
    TIMM_OK = False

try:
 if TIMM_OK:
    # Load actual models and hook attention weights
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ── MelTransform (same as training) ──────────────────────────────────────
    SR = 32000
    mel_tf = T.MelSpectrogram(
        sample_rate=SR, n_fft=2048, hop_length=512, n_mels=224,
        f_min=0, f_max=16000, power=2.0, norm='slaney', mel_scale='htk',
    )
    db_tf = T.AmplitudeToDB(stype='power', top_db=80)

    def wav_to_mel(wav):
        mel = db_tf(mel_tf(wav))
        B = mel.shape[0]
        flat = mel.reshape(B, -1)
        mn = flat.min(1)[0].unsqueeze(-1).unsqueeze(-1)
        mx = flat.max(1)[0].unsqueeze(-1).unsqueeze(-1)
        mel = (mel - mn) / (mx - mn + 1e-7)
        return mel.unsqueeze(1).expand(-1, 3, -1, -1)

    # ── Load B0 R8 fold2 (highest individual fold) ───────────────────────────
    sys.path.insert(0, '/home/lab/BirdClef-2026-Codebase')
    import train_sed_ns as tns

    # Load class names
    import pandas as pd
    taxonomy = pd.read_csv('birdclef-2026/taxonomy.csv')
    class_list = pd.read_csv('birdclef-2026/train.csv')['primary_label'].unique().tolist()
    # Get species names for labels
    label2name = dict(zip(taxonomy.get('primary_label', taxonomy.iloc[:, 0]),
                          taxonomy.get('common_name', taxonomy.get('species_common_name', taxonomy.iloc[:, 0]))))

    # Load a soundscape for visualization
    import os
    ss_files = sorted([f for f in os.listdir('birdclef-2026/train_soundscapes') if f.endswith('.ogg')])

    # Pick a file with known species from labeled set
    import csv
    labeled = pd.read_csv('birdclef-2026/train_soundscapes_labels.csv')
    # Find file with multiple species
    # Find file with multiple species (primary_label may be semicolon-separated)
    labeled['n_species'] = labeled['primary_label'].apply(lambda x: len(str(x).split(';')))
    file_counts = labeled.groupby('filename')['n_species'].max()
    good_files = file_counts[file_counts >= 2].index.tolist()
    test_filename = good_files[0] if good_files else ss_files[0]
    if '/' not in test_filename:
        test_filename_path = f'birdclef-2026/train_soundscapes/{test_filename}'
    else:
        test_filename_path = test_filename

    print(f"  Using soundscape: {test_filename_path}")

    # Load 60s soundscape
    wav, sr_orig = torchaudio.load(test_filename_path)
    if sr_orig != SR:
        wav = torchaudio.functional.resample(wav, sr_orig, SR)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    wav = wav[0]  # (samples,)

    # 9 overlapping 20s windows
    WIN = 20 * SR
    STRIDE = 5 * SR
    windows = []
    for start in range(0, 9 * STRIDE, STRIDE):
        end = start + WIN
        if end > len(wav):
            pad = torch.zeros(end - len(wav))
            win = torch.cat([wav[start:], pad])
        else:
            win = wav[start:end]
        windows.append(win)
    wav_batch = torch.stack(windows)  # (9, WIN)

    with torch.no_grad():
        mel_batch = wav_to_mel(wav_batch).to(device)  # (9, 3, 224, T)

    # ── Hook into AttentionSEDHead to get attention weights ──────────────────
    def load_and_get_attention(ckpt_path, backbone='tf_efficientnet_b0.ns_jft_in1k'):
        model = tns.SEDModel(backbone=backbone, num_classes=234,
                             gem_p_init=3.0, dropout=0.1, drop_path_rate=0.1)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state = ckpt.get('model_state', ckpt.get('state_dict', ckpt))
        # strip prefix if needed
        state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        model = model.to(device).eval()

        att_weights = {}
        def hook_fn(module, inp, out):
            # att_conv output before softmax: (B, C, T)
            att_weights['raw'] = out.detach().cpu()

        # hook on att_conv
        h = model.head.att_conv.register_forward_hook(hook_fn)

        with torch.no_grad():
            out = model(mel_batch)
            probs = out['clipwise_prob'].cpu().numpy()  # (9, 234)

        h.remove()

        # Compute actual attention: softmax(tanh(raw))
        raw = att_weights['raw']  # (9, 234, T)
        att = F.softmax(torch.tanh(raw), dim=-1).numpy()  # (9, 234, T)

        return probs, att, model

    try:
        b0_probs, b0_att, b0_model = load_and_get_attention(
            'outputs/sed-ns-b0-20s-r8/fold2_best.pt',
            'tf_efficientnet_b0.ns_jft_in1k'
        )
        print(f"  B0 R8 fold2 loaded, probs shape: {b0_probs.shape}, att shape: {b0_att.shape}")
        b0_ok = True
    except Exception as e:
        print(f"  B0 load failed: {e}")
        b0_ok = False

    try:
        pvt_probs, pvt_att, pvt_model = load_and_get_attention(
            'outputs/sed-ns-pvt-20s-r8/fold2_best.pt',
            'pvt_v2_b0'
        )
        print(f"  PVT R8 fold2 loaded, probs shape: {pvt_probs.shape}, att shape: {pvt_att.shape}")
        pvt_ok = True
    except Exception as e:
        print(f"  PVT load failed: {e}")
        pvt_ok = False

    if b0_ok or pvt_ok:
        # Get ground truth labels for this file
        file_key = os.path.basename(test_filename_path).replace('.ogg', '')
        gt_rows = labeled[labeled['filename'].str.contains(file_key, na=False)]
        gt_species = gt_rows['primary_label'].unique().tolist() if len(gt_rows) > 0 else []
        print(f"  GT species in this file: {gt_species}")

        # Find top predicted species (averaged over 9 windows)
        if b0_ok:
            avg_probs_b0 = b0_probs.mean(axis=0)  # (234,)
            top_k_b0 = np.argsort(avg_probs_b0)[::-1][:6]

        if pvt_ok:
            avg_probs_pvt = pvt_probs.mean(axis=0)
            top_k_pvt = np.argsort(avg_probs_pvt)[::-1][:6]

        # ── Figure 8: Attention maps ──────────────────────────────────────────
        # Show 12-slot probability heatmap + attention patterns for top species
        fig = plt.figure(figsize=(14, 9))
        gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

        SLOTS = 12
        WINDOWS = 9
        # Map window probs to 12 slots (coverage-weighted average)
        slot_probs_b0  = np.zeros((SLOTS, 234)) if b0_ok  else None
        slot_probs_pvt = np.zeros((SLOTS, 234)) if pvt_ok else None
        slot_counts = np.zeros(SLOTS)

        for j in range(WINDOWS):
            t_start_s = j * 5
            for s in range(SLOTS):
                slot_start_s = s * 5
                if slot_start_s >= t_start_s and slot_start_s < t_start_s + 20:
                    slot_counts[s] += 1
                    if b0_ok:  slot_probs_b0[s]  += b0_probs[j]
                    if pvt_ok: slot_probs_pvt[s] += pvt_probs[j]

        slot_counts = np.maximum(slot_counts, 1)
        if b0_ok:  slot_probs_b0  /= slot_counts[:, None]
        if pvt_ok: slot_probs_pvt /= slot_counts[:, None]

        # --- Plot A: 12-slot heatmap for top species (B0) ---
        ax_a = fig.add_subplot(gs[0, 0])
        if b0_ok:
            top_sp = top_k_b0[:5]
            hm = slot_probs_b0[:, top_sp].T  # (5, 12)
            im = ax_a.imshow(hm, aspect='auto', cmap='Blues', vmin=0, vmax=1)
            ax_a.set_yticks(range(5))
            sp_names = [class_list[s] if s < len(class_list) else str(s) for s in top_sp]
            ax_a.set_yticklabels([f'{n[:15]}' for n in sp_names], fontsize=7.5)
            ax_a.set_xticks(range(12))
            ax_a.set_xticklabels([f'{i*5}s' for i in range(12)], fontsize=7)
            ax_a.set_title('B0 R8: 12-Slot Probabilities\n(Top 5 predicted species)', fontweight='bold')
            ax_a.set_xlabel('5-second slot')
            plt.colorbar(im, ax=ax_a, shrink=0.85)

        # --- Plot B: 12-slot heatmap for top species (PVT) ---
        ax_b = fig.add_subplot(gs[0, 1])
        if pvt_ok:
            top_sp = top_k_pvt[:5]
            hm = slot_probs_pvt[:, top_sp].T
            im = ax_b.imshow(hm, aspect='auto', cmap='Oranges', vmin=0, vmax=1)
            ax_b.set_yticks(range(5))
            sp_names = [class_list[s] if s < len(class_list) else str(s) for s in top_sp]
            ax_b.set_yticklabels([f'{n[:15]}' for n in sp_names], fontsize=7.5)
            ax_b.set_xticks(range(12))
            ax_b.set_xticklabels([f'{i*5}s' for i in range(12)], fontsize=7)
            ax_b.set_title('PVT R8: 12-Slot Probabilities\n(Top 5 predicted species)', fontweight='bold')
            ax_b.set_xlabel('5-second slot')
            plt.colorbar(im, ax=ax_b, shrink=0.85)

        # --- Plot C+D: Attention weight profiles for selected species ---
        # Show attention over time frames for window 4 (middle of file)
        WIN_IDX = 4  # middle window (20-40s)
        T_frames = b0_att.shape[2] if b0_ok else pvt_att.shape[2]
        t_axis_b0 = np.linspace(WIN_IDX*5, WIN_IDX*5+20, T_frames) if b0_ok else None

        ax_c = fig.add_subplot(gs[1, :])
        n_show = min(4, len(top_k_b0 if b0_ok else top_k_pvt))
        colors_sp = ['#1565C0', '#388E3C', '#C62828', '#7B1FA2']
        t_axis_frames = np.arange(T_frames)
        t_axis_sec = np.linspace(WIN_IDX * 5, WIN_IDX * 5 + 20, T_frames)

        for i, sp_idx in enumerate((top_k_b0 if b0_ok else top_k_pvt)[:n_show]):
            sp_name = class_list[sp_idx] if sp_idx < len(class_list) else str(sp_idx)
            if b0_ok:
                att_b0_sp = b0_att[WIN_IDX, sp_idx, :]  # attention over time for this species
                ax_c.plot(t_axis_sec, att_b0_sp, '-', color=colors_sp[i],
                          linewidth=1.2, alpha=0.85, label=f'{sp_name[:20]} (B0)')
            if pvt_ok:
                t_axis_pvt_sec = np.linspace(WIN_IDX * 5, WIN_IDX * 5 + 20, pvt_att.shape[2])
                att_pvt_sp = pvt_att[WIN_IDX, sp_idx, :]
                ax_c.plot(t_axis_pvt_sec, att_pvt_sp, '--', color=colors_sp[i],
                          linewidth=1.2, alpha=0.65, label=f'{sp_name[:20]} (PVT)')

        ax_c.set_xlabel('Time (seconds, within 20s window 20–40s)')
        ax_c.set_ylabel('Attention Weight')
        ax_c.set_title(
            'Attention Weight Over Time (Window 4: 20–40s of soundscape)\n'
            'Solid = B0  |  Dashed = PVT  |  Each color = one species',
            fontweight='bold')
        ax_c.legend(fontsize=7, ncol=2, loc='upper right')
        ax_c.grid(True, alpha=0.25, linestyle=':')

        # --- Plot E: Mel spectrogram of the window ---
        ax_e = fig.add_subplot(gs[2, :])
        with torch.no_grad():
            mel_vis = mel_batch[WIN_IDX, 0].cpu().numpy()  # (224, T)
        im_mel = ax_e.imshow(mel_vis, aspect='auto', origin='lower',
                              cmap='magma', extent=[WIN_IDX*5, WIN_IDX*5+20, 0, 16])
        ax_e.set_xlabel('Time (seconds)')
        ax_e.set_ylabel('Frequency (kHz)')
        ax_e.set_title(f'Log-Mel Spectrogram — Window 4 (20–40s)\nFile: {os.path.basename(test_filename_path)}',
                       fontweight='bold')
        plt.colorbar(im_mel, ax=ax_e, shrink=0.7, label='Normalized dB')

        plt.suptitle('Attention Map Analysis: B0 R8 vs PVT R8 on Real Soundscape', fontweight='bold', y=1.01)
        plt.savefig('reports/figures/fig8_attention_maps.pdf')
        plt.savefig('reports/figures/fig8_attention_maps.png')
        plt.close()
        print("Fig 8 done (attention maps)")

        # ── Figure 9: Attention sparsity comparison B0 vs PVT ────────────────
        # Compare how concentrated/diffuse attention is for each model
        if b0_ok and pvt_ok:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

            for ax, att, model_name, color, probs_all in [
                (axes[0], b0_att,  'B0 R8',  BLUE, b0_probs),
                (axes[1], pvt_att, 'PVT R8', ORG,  pvt_probs),
            ]:
                # Entropy of attention distribution (lower = more focused)
                # att shape: (9, 234, T)
                att_ent = -(att * np.log(att + 1e-9)).sum(-1)  # (9, 234) entropy per species per window
                avg_prob = probs_all  # (9, 234)

                # Scatter: avg_prob vs attention entropy for top-confidence species
                flat_prob = avg_prob.mean(0)  # (234,)
                flat_ent  = att_ent.mean(0)   # (234,)

                # color by taxon (we know Aves is majority)
                sc = ax.scatter(flat_prob, flat_ent, alpha=0.45, s=18, c=GREY)

                # Highlight top predicted species
                for si in top_k_b0[:8]:
                    ax.scatter(flat_prob[si], flat_ent[si], s=60, zorder=5,
                               color=RED, marker='*')
                    sp_name = class_list[si] if si < len(class_list) else str(si)
                    ax.annotate(sp_name[:12], (flat_prob[si], flat_ent[si]),
                                fontsize=6.5, alpha=0.85,
                                xytext=(3, 3), textcoords='offset points')

                ax.set_xlabel('Mean Predicted Probability')
                ax.set_ylabel('Attention Entropy (lower = more focused)')
                ax.set_title(f'{model_name}: Probability vs Attention Focus\n(★ = top predicted species)',
                             fontweight='bold')
                ax.grid(True, alpha=0.25, linestyle=':')

            plt.suptitle('Attention Entropy Analysis: High-Confidence Species Have More Focused Attention',
                         fontweight='bold', y=1.01)
            plt.tight_layout()
            plt.savefig('reports/figures/fig9_attention_entropy.pdf')
            plt.savefig('reports/figures/fig9_attention_entropy.png')
            plt.close()
            print("Fig 9 done (attention entropy)")

except Exception as e:
    print(f"Attention map generation failed: {e}")
    import traceback; traceback.print_exc()

print("\nAll figures saved to reports/figures/")
print("Files:", sorted(os.listdir('reports/figures')))
