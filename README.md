# BirdCLEF 2026 — Codebase

Kaggle competition: multi-label bird/amphibian/insect species classification from 5-second audio segments.
**Metric**: Macro-averaged ROC-AUC over 234 species. **Test data**: Soundscape recordings (Pantanal, Brazil).

---

## Current Best Results (2026-03-22)

| Model / Ensemble | Holdout AUC | LB | Notes |
|-----------------|-------------|-----|-------|
| **LGBM + R46.08 event smooth** | 0.8140 OOF | **0.926** ⭐ | Current best |
| LGBM probe (ptmap-lgbm) | — | **0.925** | LGBM per-class probe, no post-proc |
| v3-ensemble (Perch 70/30 + SED VLOM) | — | **0.921** | Bayes PCA64+LogReg + SED 50/50 |

> **Only nohuman models evaluated from 2026-03-15 onwards.**
> **Submit threshold**: individual SED soundscape val AUC > 0.9193 before new SED ensemble submission.

---

## LB Score History

| Date | Submission | LB | Notes |
|------|-----------|-----|-------|
| 2026-03-22 | LGBM + R46.08 event smooth | **0.926** ⭐ | lmax_pre_aves→SoftRich→cSEBBs OOF=0.8140 |
| 2026-03-21 | lgbm-infer / ptmap-lgbm | **0.925** | LGBM probe breakthrough |
| 2026-03-21 | lgbm-4fold (our SED only) | 0.908 | Replaced competitor SED → -0.013 |
| 2026-03-20 | v3-ensemble | 0.921 | Perch 70% Bayes + SED 50/50 VLOM |
| 2026-03-17 | v9-asl-soup ensemble | 0.892 | First competitive submission |

---

## Architecture Overview

### 1. Perch Embedding Probe (current best: LB 0.926)

```
Audio (60s) → 12×5s clips → Perch v2 TFLite →
  ├── 14795-dim logits → gather 234 → Bayesian prior fusion
  └── 1536-dim embedding → PCA(64) → LGBM probe (74-dim features)
  → (1-α)×base + α×lgbm_pred  [α=0.40]
  → lmax_pre_aves(α=0.1) → SoftRich(α=0.40) → cSEBBs → submission
```

### 2. Noisy Student Pipeline (10s clips, 1st-place inspired, 2026-03-23)

**Goal**: Train SED + EfficientSSM students independently over 4 rounds. Both take **raw audio** as input with no Perch dependency at inference.

#### Design Decisions (BirdCLEF 2025 1st place — Babych)

| Technique | Implementation |
|-----------|---------------|
| **Multi-iterative pseudo-labeling** | 4 rounds × 5 folds; each round uses previous round's predictions |
| **MixUp alpha=0.15** | Applied to labeled and pseudo data |
| **Cross-domain MixUp** | labeled + pseudo **concatenated into one batch** → single MixUp pass |
| **Max-of-labels** | `label = max(label_a, label_b)` — union of species; correct for audio mixing |
| **Stochastic Depth** | `drop_path_rate=0.1` in EfficientNet-B0 backbone |
| **Pseudo weight = 1.0** | Equal weighting of labeled and pseudo losses |
| **Power transform (γ=2.0)** | Applied before 95th-percentile threshold for pseudo label generation |
| **LR Warmup (SSM)** | 5-epoch linear warmup → stabilizes SSM state matrices |

#### SED Student (`train_sed_ns.py`, configs: `sed_ns_b0_10s_r{1-4}.yaml`)

```
train_audio/ + pseudo_labels/ns_r0.csv (round 1)
  → 10s clips (CLIP_SAMPLES = SR×10), inference: STRIDE=SR×5 → 12 row_ids per soundscape
  → MelSpec(n_mels=224, n_fft=2048, hop=512, fmin=0, fmax=16000) + AmplitudeToDB
  → SpecAugment (freq_mask=24, time_mask=32)
  → EfficientNet-B0 (tf_efficientnet_b0.ns_jft_in1k, drop_path=0.1)
  → GEMFreqPool → AttentionSEDHead → sigmoid
  → Focal BCE (γ=2.0), CosineAnnealingLR, 30 epochs, early_stop=7
  → Cross-domain MixUp: concat(labeled_mel, pseudo_mel) → MixUp → single forward pass
  → Label mixing: max(label_a, label_b)
  → Validation: labeled soundscape OOF (ss_auc = macro ROC-AUC on soundscape hold-out)
```

#### EfficientSSM Student (`train_ssm_ns.py`, configs: `ssm_ns_b0_10s_r{1-4}.yaml`)

```
train_audio/ (T=1) + pseudo soundscape sequences (T=12)
  → 10s clips → Mel → EfficientNet-B0(global_pool='avg') → (B, T, 1280)
  → Linear(1280→256) + LayerNorm
  → 2× BidirectionalSSM(d_model=256, d_state=16)
  → Linear head (nn.Linear(256, 234))         ← replaced cosine head (unstable)
  → Focal BCE (γ=2.0)
  → AdamW lr=1e-3, 5-ep linear warmup + CosineAnnealing, 30 epochs, early_stop=7
  → Cross-domain MixUp: pseudo×pseudo (max labels) + labeled clips broadcast across T
  → Validation: labeled soundscape sequences, per-window AUC
```

**Why linear head (not cosine)**: EMA prototypes drift during training, temperature can diverge →
causes 0.9→0.6 AUC oscillation in later epochs. Linear head is stable.

**Why LR warmup for SSM**: SSM state matrices (A, B, C, dt) are sensitive to large gradients at init.
5-epoch linear warmup prevents early destabilization.

#### Round 0 Pseudo Labels (Teacher)

```
train.py → Perch head fine-tuned on train_soundscapes (wandb: perch-head-retrain-r1)
  → extract_perch_teacher_all_ss.py → outputs/perch_teacher_all_ss.csv
  → gen_pseudo_ns.py --round 0 → pseudo_labels/ns_r0.csv
```

#### Orchestration

```
scripts/master_ns_chain.sh
  1. Wait for Perch head training (train.py) to finish
  2. extract_perch_teacher_all_ss.py  → outputs/perch_teacher_all_ss.csv
  3. gen_pseudo_ns.py --round 0       → pseudo_labels/ns_r0.csv
  4. Launch in parallel:
     ├── auto_sed_ns_10s_full.sh   SED  r1→r4, GPU1 → sed_10s_r{k}.csv
     └── auto_ssm_ns_10s_full.sh   SSM  r1→r4, GPU1 → ssm_10s_r{k}.csv
```

Skip logic: fold checkpoint exists → skip; old incomplete checkpoints must be deleted before launch.

---

## Currently Running (2026-03-23)

| Process | Status | Notes |
|---------|--------|-------|
| `train.py` (perch-head-retrain-r1) | 🔄 ep≈25/80, best=0.9555 | wandb: perch-head-retrain-r1, GPU1 |
| `master_ns_chain.sh` | ⏳ Waiting for Perch | Will auto-launch SED+SSM after Perch completes |

Monitor: `python3 scripts/monitor_experiments.py --excel`

---

## Project Structure

```
BirdClef-2026-Codebase/
│
├── configs/
│   ├── exp_nohuman_label_soundscape_train.yaml  # Perch head retrain config (teacher)
│   ├── sed_ns_b0_10s_r{1-4}.yaml               # SED NS 10s rounds 1-4
│   └── ssm_ns_b0_10s_r{1-4}.yaml               # EfficientSSM NS 10s rounds 1-4
│
├── train.py                # Perch head fine-tuning (teacher for round 0), early_stop=10
├── train_sed_ns.py         # SED Noisy Student — cross-domain MixUp, max labels
├── train_ssm_ns.py         # EfficientSSM NS — linear head, 5-ep warmup, cross-domain mix
│
├── scripts/
│   ├── master_ns_chain.sh              # Wait Perch → ns_r0 → launch SED+SSM chains
│   ├── auto_sed_ns_10s_full.sh         # SED chain r1→r4
│   ├── auto_ssm_ns_10s_full.sh         # SSM chain r1→r4
│   ├── extract_perch_teacher_all_ss.py # Perch teacher predictions for all soundscapes
│   ├── gen_pseudo_ns.py                # Power transform (γ=2.0) + 95th-pct → pseudo CSV
│   └── monitor_experiments.py         # Status print + Excel update (15-min cron)
│
├── pseudo_labels/
│   ├── ns_r0.csv              # Round 0: Perch teacher only (shared init for both chains)
│   ├── sed_10s_r{1-3}.csv     # SED-only pseudo labels (generated per round)
│   └── ssm_10s_r{1-3}.csv     # SSM-only pseudo labels (generated per round)
│
├── outputs/
│   ├── logs/
│   │   ├── perch_head_retrain.log
│   │   ├── master_ns_chain.log
│   │   ├── sed_ns_10s_r{N}_fold{F}.log
│   │   └── ssm_ns_10s_r{N}_fold{F}.log
│   ├── sed-ns-b0-10s-r{1-4}/   # SED NS checkpoints + all_ss_probs.npz
│   └── ssm-ns-b0-10s-r{1-4}/   # SSM NS checkpoints + all_ss_probs.npz
│
└── reports/
    └── ns_chain_progress.xlsx   # Auto-updated every 15min (latest + history sheets)
```

---

## Key Technical Findings

### What Works

| Technique | Effect | Evidence |
|-----------|--------|----------|
| Human voice removal (Silero VAD) | +0.039 LB | Ablation confirmed |
| LGBM probe (vs LogReg) | LB 0.925 | Better than LogReg; 74-dim with interaction features |
| Bayesian prior fusion | ~+0.02 LB | Site/hour priors on Perch logits |
| SoftRich (α=0.40) | OOF 0.8164 | Cross-file richness normalization |
| cSEBBs (cp_blend=0.60) | OOF 0.8164 | Change-point segment boosting |
| Linear head (SSM) | Stable training | Cosine head causes 0.9→0.6 oscillation |
| LR warmup (SSM) | Prevents collapse | SSM state matrices sensitive to init LR |
| Cross-domain MixUp (1st place) | Better generalization | labeled×pseudo in same batch |
| Max-of-labels MixUp (1st place) | Correct multi-label target | Union of species present in mix |

### What Didn't Work

| Technique | Result | Notes |
|-----------|--------|-------|
| Cosine/prototypical head (SSM) | AUC 0.9→0.6 oscillation | EMA prototype drift + temperature divergence |
| Old SED approach | -0.013 LB | Wrong data, wrong mel, no SumixFreq |
| ASL + secondary_weight=1.0 | Noisy gradients | Unreliable XC secondary labels |

---

## Quick Start

```bash
# Monitor all experiments
python3 scripts/monitor_experiments.py --excel

# Check Perch head training
grep "Epoch" outputs/logs/perch_head_retrain.log | tail -5

# Check master chain
tail -5 outputs/logs/master_ns_chain.log

# All training on GPU1
export CUDA_VISIBLE_DEVICES=1
```

---

## Constraints

- All training: **GPU1** (`CUDA_VISIBLE_DEVICES=1`)
- Only **nohuman models** evaluated/submitted
- **No competitor model weights** — only models trained ourselves
- Submit only when SED soundscape val AUC > **0.9193**
- Current LB anchor: **0.926**
