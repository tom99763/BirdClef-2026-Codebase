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

### 2. Noisy Student Pipeline (20s clips, 1st-place inspired, 2026-03-23)

**Goal**: Train SED + EfficientSSM students independently over 4 rounds. Both take **raw audio** as input with no Perch dependency at inference.

#### 1st-place Techniques Adopted (Babych 2025, BirdCLEF)

| Technique | Implementation | Where |
|-----------|---------------|-------|
| **20s clip duration** | `clip_duration: 20` in configs; covers 4 Perch 5s windows | All NS configs |
| **SumixFreq** | Per-freq-bin random selection between two mel spectrograms | `train_sed_ns.py`, `train_ssm_ns.py` |
| **Overlapping window inference** | 20s window, 5s stride; each 5s frame averaged from multiple windows | Submission notebook |
| **Babych smoothing kernel** | `[0.1, 0.2, 0.4, 0.2, 0.1]` temporal smoothing | Submission notebook |
| **absmax normalization** | Normalize each clip to `[-1, 1]` before mel | `load_audio_clip`, `load_ss_clip` |
| **Cross-domain MixUp (lam=0.5)** | labeled + pseudo in same batch, fixed λ=0.5 | Both students |
| **Max-of-labels** | `label = max(label_a, label_b)` — union of species | Both students |
| **Stochastic Depth** | `drop_path_rate=0.1` in EfficientNet-B0 backbone | Both students |
| **Power transform (γ=2.0) + 95th-pct threshold** | Controls pseudo label confidence | `gen_pseudo_ns.py` |
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
    --output outputs/perch_teacher_all_ss.csv
# Output: CSV with columns [row_id, species_1, ..., species_234]
# row_ids are in 5s format: <filename>_5, _10, ..., _60
```

### Step 2 — Generate Round-0 Pseudo Labels

Merge 5s Perch windows into 20s-aligned pseudo labels for student training:

```bash
python3 scripts/gen_pseudo_ns.py \
    --round    0 \
    --clip_sec 20 \
    --perch_csv outputs/perch_teacher_all_ss.csv \
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
  3. Generate pseudo labels (R < 4 only):
     gen_pseudo_ns.py --round R --clip_sec 20 --{sed,ssm}_dir ... --out pseudo_labels/{sed,ssm}_20s_r{R}.csv
  4. Update next round config: sed -i pseudo_labels_csv → new CSV
```

### Step 4 — Monitor Training

```bash
# Tail individual fold logs
tail -f outputs/logs/sed_ns_20s_r1_fold0.log
tail -f outputs/logs/ssm_ns_20s_r1_fold0.log

# Check master chain progress
tail -f outputs/logs/master_ns_chain.log

# Monitor all experiments (updates Excel)
python3 scripts/monitor_experiments.py --excel
```

### Step 5 — Inference / Submission

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
| `sed_ns_b0_20s_r{1-4}.yaml` | 4 | 20s | 16 | SED, SumixFreq, early_stop=7 |
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
  → Cross-domain MixUp (λ=0.5): concat(labeled, pseudo) → single forward pass
  → EfficientNet-B0 → GEMFreqPool(p=3) → AttentionSEDHead
  → Focal BCE (γ=2.0), lr=1e-3, CosineAnnealing, 30 epochs, early_stop=7
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
  → extract_perch_teacher_all_ss.py → outputs/perch_teacher_all_ss.csv (5s format)
  → gen_pseudo_ns.py --clip_sec 20  → pseudo_labels/ns_r0.csv (20s format)
     Power transform p[i]^2.0 → 95th-percentile threshold per species
```

---

## Currently Running (2026-03-23)

| Process | Status | Notes |
|---------|--------|-------|
| `auto_sed_ns_20s_full.sh` | 🔄 r1 fold0 training | GPU1, 20s clips, SumixFreq |
| `auto_ssm_ns_20s_full.sh` | 🔄 r1 fold0 training | GPU1, 20s clips, SumixFreq |

Monitor: `python3 scripts/monitor_experiments.py --excel`

```bash
tail -f outputs/logs/auto_sed_ns_20s_full.log
tail -f outputs/logs/auto_ssm_ns_20s_full.log
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
│   ├── master_ns_chain.sh              # Full pipeline: wait Perch → ns_r0 → launch SED+SSM
│   ├── auto_sed_ns_20s_full.sh         # SED chain r1→r4 (20s clips)
│   ├── auto_ssm_ns_20s_full.sh         # SSM chain r1→r4 (20s clips)
│   ├── extract_perch_teacher_all_ss.py # Perch teacher predictions for all soundscapes
│   ├── gen_pseudo_ns.py                # Power transform (γ=2.0) + 95th-pct → pseudo CSV
│   └── monitor_experiments.py         # Status print + Excel update (15-min cron)
│
├── pseudo_labels/
│   ├── ns_r0.csv              # Round 0: Perch teacher, 20s format (_20, _25, ..., _65)
│   ├── sed_20s_r{1-3}.csv     # SED-only pseudo labels (generated per round)
│   └── ssm_20s_r{1-3}.csv     # SSM-only pseudo labels (generated per round)
│
├── outputs/
│   ├── logs/
│   │   ├── auto_sed_ns_20s_full.log
│   │   ├── auto_ssm_ns_20s_full.log
│   │   ├── sed_ns_20s_r{N}_fold{F}.log
│   │   └── ssm_ns_20s_r{N}_fold{F}.log
│   ├── sed-ns-b0-20s-r{1-4}/   # SED NS checkpoints + all_ss_probs.npz
│   └── ssm-ns-b0-20s-r{1-4}/   # SSM NS checkpoints + all_ss_probs.npz
│
├── birdclef-2026/notebook resource/current_subs/
│   └── ns_perch_sed_ssm_submission.ipynb   # Submission: TFLite Perch + NS-SED + NS-SSM
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
