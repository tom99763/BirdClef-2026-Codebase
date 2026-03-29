# BirdCLEF 2026 — Codebase

Kaggle competition: multi-label bird/amphibian/insect species classification from 5-second audio segments.
**Metric**: Macro-averaged ROC-AUC over 234 species. **Test data**: Soundscape recordings (Pantanal, Brazil).

---

## Current Best Results (2026-03-29)

| Model / Ensemble | Holdout AUC | LB | Notes |
|-----------------|-------------|-----|-------|
| **dual-foundation + KNN Embedding Prior** | — | **0.929** ⭐ | Plan C KNN prior (lambda=0.25, k=5, cosine) on 66 labeled soundscapes. CURRENT BEST. |
| dual-foundation-protossm-v3-sed-fusion | — | **0.927** | MLP probe + TFLite heads + ProtoSSM + ResidualSSM + VLOM |
| dual-foundation (LGBM replaces MLP) | — | **0.926** | LGBM probe → -0.001 vs MLP; MLP is better |
| LGBM + R46.08 event smooth | 0.8140 OOF | **0.926** | lmax_pre_aves→SoftRich→cSEBBs |
| LGBM probe (ptmap-lgbm) | — | **0.925** | LGBM per-class probe, no post-proc |
| v3-ensemble (Perch 70/30 + SED VLOM) | — | **0.921** | Bayes PCA64+LogReg + SED 50/50 |

**Embed Prior best**: `wk4d_ab19_ia31_sa33` LOO=0.995956 (4-comp wKNN+3way, batch=168) — ready to integrate into next submission.

> **Only nohuman models evaluated from 2026-03-15 onwards.**
> **Submit threshold**: individual SED soundscape val AUC > 0.9193 before new SED ensemble submission.

---

## LB Score History

| Date | Submission | LB | Notes |
|------|-----------|-----|-------|
| 2026-03-25 | dual-foundation + KNN Embedding Prior (Plan C) | **0.929** ⭐ | cosine-KNN on 66 labeled ss, lambda=0.25, +0.002 over 0.927. NEW BEST |
| 2026-03-25 | dual-foundation (LGBM replaces MLP probe) | **0.926** | MLP probe 優於 LGBM；MLP 版本仍是最佳 |
| 2026-03-24 | dual-foundation-protossm-v3-sed-fusion | **0.927** ⭐ | MLP probe + TFLite heads blend + 2-way OOF + ProtoSSM + ResidualSSM (before VLOM) + VLOM 50/50 |
| 2026-03-22 | LGBM + R46.08 event smooth | **0.926** | lmax_pre_aves→SoftRich→cSEBBs OOF=0.8140 |
| 2026-03-21 | lgbm-infer / ptmap-lgbm | **0.925** | LGBM probe breakthrough |
| 2026-03-21 | lgbm-4fold (our SED only) | 0.908 | Replaced competitor SED → -0.013 |
| 2026-03-20 | v3-ensemble | 0.921 | Perch 70% Bayes + SED 50/50 VLOM |
| 2026-03-17 | v9-asl-soup ensemble | 0.892 | First competitive submission |

---

## Architecture Overview

### 1. Dual-Foundation Architecture (current best: LB 0.927)

Full pipeline for `dual-foundation-protossm-v3-sed-fusion.ipynb`:

```
Audio (60s soundscape) → 12 × 5s clips → Perch v2 TFLite
  ├── 14795-dim logits → gather 234 → Bayesian prior fusion (site/hour priors)
  └── 1536-dim embedding → PCA(64) → LGBM probe (74-dim features, per-class LGBMClassifier, scale_pos_weight)
  → (1-α)×base_probs + α×lgbm_pred  [α = OOF-optimized]
  → TFLite heads blend (HEAD_BLEND_ALPHA=0.30):
      label_head_pseudo × 0.30
    + label_head_soundscape_train × 0.30
    + embedding_head_nohuman × 0.30
  → 2-way OOF grid search → ENSEMBLE_WEIGHT_PROTO (ProtoSSM optimal weight)
  → ProtoSSM v2 (BiSSM, d_model=128, d_state=16; prototypes + metadata)
  → ResidualSSM correction applied to first_pass (ProtoSSM+probe, BEFORE VLOM blend)
  → VLOM blend: ProtoSSM(0.50) + SED ONNX(0.50)
  → final_test_scores_blended → submission
```

**Key design decision — ResidualSSM position**: Corrector targets `first_pass` (ProtoSSM+probe),
not the final VLOM output. Applying it after VLOM caused distribution mismatch (LB 0.923 regression).

### 2. Noisy Student Pipeline (20s clips, 1st-place inspired, 2026-03-23)

**Goal**: Train SED + EfficientSSM students independently over 4 rounds. Both take **raw audio** as input with no Perch dependency at inference.

#### 1st-place Techniques Adopted (Babych 2025, BirdCLEF)

| Technique | Implementation | Where |
|-----------|---------------|-------|
| **20s clip duration** | `clip_duration: 20` in configs; covers 4 Perch 5s windows | All NS configs |
| **SumixFreq** | Per-freq-bin random selection between two mel spectrograms | `train_sed_ns.py`, `train_ssm_ns.py` |
| **Sliding window inference** | Pad audio ±15s; each 5s window averaged from 4 overlapping 20s chunks | `infer_all_soundscapes()` in `train_sed_ns.py` |
| **Babych smoothing kernel** | `[0.1, 0.2, 0.4, 0.2, 0.1]` temporal smoothing | Submission notebook |
| **absmax normalization** | Normalize each clip to `[-1, 1]` before mel | `load_audio_clip`, `load_ss_clip` |
| **Cross-domain MixUp (lam=0.5, ratio=1.0)** | Every labeled sample 1-to-1 mixed with one pseudo sample, fixed λ=0.5 | `train_sed_ns.py` |
| **Max-of-labels** | `label = max(label_a, label_b)` — union of species | Both students |
| **Stochastic Depth** | `drop_path_rate=0.15` in EfficientNet-B0 backbone | R2-R4 configs |
| **WeightedRandomSampler** | Pseudo soundscapes weighted by sum of max label probs | `train_sed_ns.py` |
| **Power transform (per-round γ) → soft labels** | Saves γ-transformed probs as training targets; γ: R1=1.0, R2=1.54, R3=1.82 | `gen_pseudo_ns.py` |
| **LR warmup (SSM)** | 5-epoch linear warmup → stabilizes SSM state matrices | `train_ssm_ns.py` |

---

## Noisy Student Training — Step-by-Step

### Prerequisites

```bash
export CUDA_VISIBLE_DEVICES=1   # All training on GPU1
```

### Step 0 — Train Perch Head (Teacher)

Fine-tune a linear head on top of frozen Perch v2 embeddings using labeled soundscape data:

```bash
python3 train.py --config configs/exp_nohuman_label_soundscape_train.yaml
# Outputs: weights/label_head_soundscape_train.tflite (or .pt)
# WandB run: perch-head-retrain-r1
# ~80 epochs; monitor: tail -f outputs/logs/perch_head_retrain.log
```

### Step 1 — Extract Teacher Predictions

Run the fine-tuned Perch head over all training soundscapes to generate 5s-window predictions:

```bash
python3 scripts/extract_perch_teacher_all_ss.py \
    --output outputs/perch_teacher_aug_all_ss.csv
# Output: CSV with columns [row_id, species_1, ..., species_234]
# row_ids are in 5s format: <filename>_5, _10, ..., _60
```

### Step 2 — Generate Round-0 Pseudo Labels

Merge 5s Perch windows into 20s-aligned pseudo labels for student training:

```bash
python3 scripts/gen_pseudo_ns.py \
    --round    0 \
    --clip_sec 20 \
    --perch_csv outputs/perch_teacher_aug_all_ss.csv \
    --perch_w  1.0 \
    --out      pseudo_labels/ns_r0.csv
# Output: pseudo_labels/ns_r0.csv
# row_ids: <filename>_20, _25, _30, ..., _65 (20s windows, 5s stride)
# Power transform (γ=2.0) + 95th-percentile threshold applied
```

### Step 3 — Run Noisy Student Chains (R1→R4)

Launch SED and SSM chains in parallel. Each chain trains 5 folds per round and
auto-generates pseudo labels for the next round.

**Option A: Automated (recommended)**

```bash
# Launch both chains in parallel (background)
nohup bash scripts/auto_sed_ns_20s_full.sh \
    > outputs/logs/auto_sed_ns_20s_full.log 2>&1 &
echo "SED PID: $!"

nohup bash scripts/auto_ssm_ns_20s_full.sh \
    > outputs/logs/auto_ssm_ns_20s_full.log 2>&1 &
echo "SSM PID: $!"
```

**Option B: Full automated chain (waits for Perch head to finish first)**

```bash
nohup bash scripts/master_ns_chain.sh \
    > outputs/logs/master_ns_chain.log 2>&1 &
```

**Option C: Manual single fold (debug/testing)**

```bash
# SED, round 1, fold 0
python3 train_sed_ns.py \
    --config configs/sed_ns_b0_20s_r1.yaml \
    --fold   0 \
    --device cuda:0

# SSM, round 1, fold 0
python3 train_ssm_ns.py \
    --config configs/ssm_ns_b0_20s_r1.yaml \
    --fold   0 \
    --device cuda:0
```

#### What Each Chain Does (R1→R4 loop)

```
For R in 1 2 3 4:
  1. Train 5 folds:  train_{sed,ssm}_ns.py --config *_r{R}.yaml --fold {0..4}
     → outputs/{sed,ssm}-ns-b0-20s-r{R}/fold{F}_best.pt
     Skip logic: checkpoint exists → skip fold
  2. Infer all soundscapes: train_{sed,ssm}_ns.py --infer_all_ss
     → outputs/{sed,ssm}-ns-b0-20s-r{R}/all_ss_probs.npz
  3. [NEW 2026-03-24] Run Residual Corrector on SED inferences:
     python3 scripts/train_sed_residual_corrector.py
         --sed_dir outputs/sed-ns-b0-20s-r{R}
         --teacher outputs/perch_teacher_aug_all_ss.csv
         --round R
     → outputs/sed-ns-b0-20s-r{R}/all_ss_probs_corrected.npz
  4. Generate pseudo labels (R < 4 only) using corrected SED + Perch teacher:
     gen_pseudo_ns.py --round R --clip_sec 20
         --sed_dir ... (corrected)  --perch_csv ... --perch_w {teacher_weight}
         --percentile {threshold} --gamma {gamma}
         --out pseudo_labels/sed_20s_r{R}.csv
     Teacher weight schedule: R1=0.50, R2=0.30, R3=0.10
     Threshold percentile:    R1=92,   R2=93,   R3=94
     Power gamma schedule:    R1=1.00, R2=1.54, R3=1.82  ← 1st place per-round schedule
  5. Update next round config: sed -i pseudo_labels_csv → new CSV
```

**Pseudo label strategy (2026-03-25 update)**:

| Round | Teacher Weight | Threshold Pct | Gamma (γ) | Rationale |
|-------|---------------|---------------|-----------|-----------|
| R0 | 1.00 (Perch only) | 95 | 2.0 | Pure teacher; no student yet |
| R1 | 0.50 | 92 | **1.00** | Raw probs saved as soft labels; student bootstrapping |
| R2 | 0.30 | 93 | **1.54** | Moderate sharpening; reduce teacher weight |
| R3 | 0.10 | 94 | **1.82** | Strong sharpening; student dominant |
| R4 | — (inference only) | — | — | Final round; no next pseudo labels needed |

> **Key insight (1st-place power transform)**: Apply γ>1 to probabilities before saving as soft labels.
> Compresses low-confidence values toward 0 while preserving high-confidence signals.
> γ increases each round as the student becomes more reliable.
>
> **Key insight (Perch teacher retention)**: Keep teacher blend throughout all rounds.
> Dropping it causes confirmation bias — student self-reinforces its own errors.
> Even at R3, teacher_w=0.10 anchors label quality.

### Step 4 — Residual Corrector (2026-03-24)

After `infer_all_ss` for each SED round, train and apply a **BiSSM Temporal Residual Corrector**:

```bash
# Train and apply corrector for round R
python3 scripts/train_sed_residual_corrector.py \
    --sed_dir  outputs/sed-ns-b0-20s-r{R} \
    --teacher  outputs/perch_teacher_aug_all_ss.csv \
    --round    {R}
# Reads:   all_ss_probs.npz + perch_teacher (ground truth for residuals)
# Trains:  BiSSM(input=234, d_model=128, d_state=16) on teacher_probs - SED_probs residuals
# Outputs: all_ss_probs_corrected.npz (same format as all_ss_probs.npz)
```

**Architecture**:
```
SED probs (234-d per frame)
  → input_proj: Linear(234→128) + LayerNorm + GELU
  → BiSSM(d_model=128, d_state=16)   [forward + backward passes]
  → output_proj: Linear(128→234)     [residual delta]
  → corrected = SED_probs + delta    [additive correction in probability space]
```

**Training target**: `teacher_probs - SED_R_probs` (learn the gap between Perch teacher and SED student).

### Step 5 — Monitor Training

```bash
# Tail individual fold logs
tail -f outputs/logs/sed_ns_20s_r1_fold0.log
tail -f outputs/logs/ssm_ns_20s_r1_fold0.log

# Check master chain progress
tail -f outputs/logs/master_ns_chain.log

# Monitor all experiments (updates Excel)
python3 scripts/monitor_experiments.py --excel
```

### Step 6 — Inference / Submission

After round 4 completes (or latest available round), use the submission notebook:

```
birdclef-2026/notebook resource/current_subs/ns_perch_sed_ssm_submission.ipynb
```

Update `SED_DIR` and `SSM_DIR` in CONFIG cell to point to the latest round's weights dir.
Upload weights folder to Kaggle dataset and submit.

**1st-place inference settings (already in notebook):**
- `CLIP_SEC = 20` — matches 20s training clips
- `OVERLAP_INFERENCE = True` — overlapping 20s windows, 5s stride
- `BABYCH_SMOOTH_KERNEL = True` — temporal smoothing kernel `[0.1, 0.2, 0.4, 0.2, 0.1]`
- `BranchEns→cSEBBs` — SED temporal postprocessing (OOF AUC 0.8045)

---

### Config Files

| Config | Rounds | Clip | Batch | Notes |
|--------|--------|------|-------|-------|
| `sed_ns_b0_20s_r{1-4}.yaml` | 4 | 20s | 16 | SED, SumixFreq, drop_path=0.15, early_stop=3 |
| `ssm_ns_b0_20s_r{1-4}.yaml` | 4 | 20s | 8 (pseudo=2) | SSM, 5-ep warmup, SumixFreq |

Round 1 configs use `pseudo_labels/ns_r0.csv`. Rounds 2-4 are auto-updated by the chain scripts.

### Key Architecture Details

#### SED Student (`train_sed_ns.py`)

```
train_audio/ clips + pseudo_labels/*_r{N}.csv
  → 20s clips (CLIP_SAMPLES = SR×20), absmax normalized
  → MelSpec(n_mels=224, n_fft=2048, hop=512, fmin=0, fmax=16000, power=2, slaney/htk)
  → SpecAugment (freq_mask=24, time_mask=64)
  → SumixFreq: per-freq-bin random selection between two labeled mels  ← 1st place
  → Cross-domain MixUp (λ=0.5, ratio=1.0): every labeled sample 1-to-1 with one pseudo sample
  → EfficientNet-B0 (drop_path_rate=0.15) → GEMFreqPool(p=3) → AttentionSEDHead
  → Focal BCE (γ=2.0), lr=1e-3, CosineAnnealing, 30 epochs, early_stop=3
  → Validation: soundscape OOF macro ROC-AUC
```

#### EfficientSSM Student (`train_ssm_ns.py`)

```
train_audio/ (T=1) + pseudo soundscape sequences (T=12)
  → 20s clips, absmax normalized
  → Mel → EfficientNet-B0(global_pool='avg') → (B, T, 1280)
  → Linear(1280→256) + LayerNorm + GELU
  → 2× BidirectionalSSM(d_model=256, d_state=16)
  → Linear classifier (256→234)   ← stable; cosine head causes AUC oscillation
  → Focal BCE (γ=2.0)
  → AdamW lr=1e-3, 5-ep linear warmup + CosineAnnealing, 40 epochs, early_stop=7
  → Cross-domain MixUp: pseudo×pseudo (max labels) + labeled broadcast across T
  → SumixFreq on labeled mel (pre_mel path)
```

**Why linear head (not cosine)**: EMA prototypes drift → temperature diverges → 0.9→0.6 AUC oscillation.

**Why LR warmup for SSM**: SSM state matrices (A, B, C, dt) sensitive to large gradients at init.

#### Round 0 Pseudo Labels (Teacher Pipeline)

```
Perch head fine-tune (train.py)
  → extract_perch_teacher_all_ss.py → outputs/perch_teacher_aug_all_ss.csv (5s format)
  → gen_pseudo_ns.py --clip_sec 20  → pseudo_labels/ns_r0.csv (20s format)
     Power transform p[i]^2.0 → 95th-percentile threshold per species
```

---

## Currently Running (2026-03-29)

| Process | Status | Notes |
|---------|--------|-------|
| `auto_sed_ns_20s_full.sh` | **R2 fold0 training** | GPU1; R1 all done; R2 uses Aves-only pseudo labels |

R1 complete → Residual Corrector → gen_pseudo R1 (γ=1.00, Aves-only, 94,258 rows) → R2-R4 training

**R1 fold AUCs**: fold0=0.9433, fold1=0.9037, fold2=0.9420, fold3=0.9478, fold4=0.9152

Monitor:
```bash
tail -f outputs/logs/auto_sed_ns_20s_full.log
tail -f outputs/logs/sed_ns_20s_r2_fold0.log
```

---

## Project Structure

```
BirdClef-2026-Codebase/
│
├── configs/
│   ├── exp_nohuman_label_soundscape_train.yaml  # Perch head retrain config (teacher)
│   ├── sed_ns_b0_20s_r{1-4}.yaml               # SED NS 20s rounds 1-4
│   └── ssm_ns_b0_20s_r{1-4}.yaml               # EfficientSSM NS 20s rounds 1-4
│
├── train.py                # Perch head fine-tuning (teacher for round 0)
├── train_sed_ns.py         # SED Noisy Student — SumixFreq, cross-domain MixUp, absmax norm
├── train_ssm_ns.py         # EfficientSSM NS — linear head, 5-ep warmup, SumixFreq
│
├── scripts/
│   ├── master_ns_chain.sh                  # Full pipeline: wait Perch → ns_r0 → launch SED+SSM
│   ├── auto_sed_ns_20s_full.sh             # SED chain r1→r4 (20s clips) + Residual Corrector
│   ├── auto_ssm_ns_20s_full.sh             # SSM chain r1→r4 (20s clips)
│   ├── train_sed_residual_corrector.py     # BiSSM Temporal Residual Corrector for SED outputs
│   ├── extract_perch_teacher_all_ss.py     # Perch teacher predictions for all soundscapes
│   ├── gen_pseudo_ns.py                    # Power transform (γ=2.0) + threshold → pseudo CSV
│   └── monitor_experiments.py             # Status print + Excel update (15-min cron)
│
├── pseudo_labels/
│   ├── ns_r0.csv              # Round 0: Perch teacher, 20s format (_20, _25, ..., _65)
│   ├── sed_20s_r{1-3}.csv     # SED pseudo labels (corrected SED + teacher blend)
│   └── ssm_20s_r{1-3}.csv     # SSM-only pseudo labels (generated per round)
│
├── outputs/
│   ├── logs/
│   │   ├── auto_sed_ns_20s_full.log
│   │   ├── auto_ssm_ns_20s_full.log
│   │   ├── sed_ns_20s_r{N}_fold{F}.log
│   │   └── ssm_ns_20s_r{N}_fold{F}.log
│   ├── sed-ns-b0-20s-r{1-4}/
│   │   ├── fold{F}_best.pt
│   │   ├── all_ss_probs.npz           # Raw SED inference on all soundscapes
│   │   └── all_ss_probs_corrected.npz # After BiSSM Residual Corrector
│   └── ssm-ns-b0-20s-r{1-4}/   # SSM NS checkpoints + all_ss_probs.npz
│
├── birdclef-2026/notebook resource/current_subs/
│   ├── dual-foundation-protossm-v3-sed-fusion.ipynb  # Current best submission (LB 0.927)
│   └── ns_perch_sed_ssm_submission.ipynb             # Submission: TFLite Perch + NS-SED + NS-SSM
│
└── reports/
    ├── design_report_20260324.html      # Architecture decisions 2026-03-24
    └── ns_chain_progress.xlsx          # Auto-updated every 15min (latest + history sheets)
```

---

## Key Technical Findings

### What Works

| Technique | Effect | Evidence |
|-----------|--------|----------|
| Human voice removal (Silero VAD) | +0.039 LB | Ablation confirmed |
| LGBM probe (vs LogReg) | LB 0.925 | Better than LogReg; 74-dim with interaction features |
| Per-class LGBMClassifier + scale_pos_weight | LB +0.001 | Handles class imbalance; replaces MLP probe |
| TFLite heads blend (3 heads, alpha=0.30) | Part of LB 0.927 | label_head_pseudo + soundscape_train + embedding_nohuman |
| 2-way OOF optimization | Robust weight search | Grid search LGBM+heads → find best ProtoSSM weight |
| ResidualSSM on first_pass (before VLOM) | LB 0.927 | Must target pre-VLOM distribution; after-VLOM causes mismatch |
| Global seed fixing (SEED=42) | Reproducibility | random/np/torch + torch.use_deterministic_algorithms |
| Perch teacher retention in pseudo pipeline | Prevents confirmation bias | Even at R3, teacher_w=0.10 anchors label quality |
| **Aves-only pseudo labels** (`--aves_only`) | Removes 73% Amphibia contamination | Non-Aves species have insufficient training data (~12.9 clips/class) for reliable SED pseudo labeling; zeroing non-Aves columns in gen_pseudo_ns.py fixes pipeline |
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
| ResidualSSM applied after VLOM blend | LB 0.923 (-0.004) | Distribution mismatch: VLOM output has different statistics than pure ProtoSSM |
| MLP probe | Lower than LGBM | Replaced by per-class LGBMClassifier |
| Cosine/prototypical head (SSM) | AUC 0.9→0.6 oscillation | EMA prototype drift + temperature divergence |
| Old SED approach | -0.013 LB | Wrong data, wrong mel, no SumixFreq |
| ASL + secondary_weight=1.0 | Noisy gradients | Unreliable XC secondary labels |
| Embedding-space LoRA fine-tuning | AUC 0.4933→0.4914 (↓ from baseline 0.5209) | Domain mismatch: train_audio (point recordings, 97.9% Aves) vs soundscape (66.8% Amphibia). Perch has no discriminative power for non-Aves taxa. |
| MLP head on Perch embeddings (train_audio) | Best AUC=0.4998 | Same domain mismatch. ProtoHead (0.9585) is Perch's ceiling for this task — uses labeled soundscape embeddings directly. |
| Non-Aves pseudo labels in NS pipeline | Contaminated R1 labels | sed_20s_r1 had 73.6% Amphibia (73,178/99,487 rows), but SED has only ~12.9 clips/species for Amphibia → unreliable labels. Fixed with `--aves_only` flag: 94,258 rows, 100% Aves. |

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
- Current LB anchor: **0.927**
