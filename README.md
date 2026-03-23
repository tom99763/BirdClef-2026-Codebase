# BirdCLEF 2026 — Codebase

Kaggle competition: multi-label bird/amphibian/insect species classification from 5-second audio segments.
**Metric**: Macro-averaged ROC-AUC over 234 species. **Test data**: Soundscape recordings (Pantanal, Brazil).

---

## Current Best Results (2026-03-22)

| Model / Ensemble | Holdout AUC | LB | Notes |
|-----------------|-------------|-----|-------|
| **LGBM + R46.08 event smooth** | 0.8140 OOF | **0.926** ⭐ | Current best — R46.08 post-proc on LGBM ensemble |
| LGBM probe (ptmap-lgbm / lgbm-infer) | — | **0.925** | LGBM per-class probe, no post-proc |
| v3-ensemble (Perch 70/30 + SED VLOM) | — | **0.921** | Bayesian probe PCA64+LogReg α=0.40 + SED 50/50 |
| v9-asl-soup ensemble | — | 0.892 | First VLOM blend submission |
| *Competitor SED (reference)* | *~0.90 soundscape* | — | *Our SED soundscape val AUC target* |

> **Key gap**: Replacing competitor SED with our own → -0.013 LB (0.921→0.908). SED is the top priority.
> Only nohuman models evaluated from 2026-03-15 onwards.

---

## LB Score History

| Date | Submission | LB | Notes |
|------|-----------|-----|-------|
| 2026-03-22 | LGBM + R46.08 event smooth | **0.926** ⭐ | lmax_pre_aves→SoftRich→cSEBBs OOF=0.8140 |
| 2026-03-21 | lgbm-infer / ptmap-lgbm | **0.925** | LGBM probe breakthrough; PT-MAP ineffective |
| 2026-03-21 | lgbm-4fold (our SED only) | 0.908 | Replaced competitor SED → -0.013. Our SED weak. |
| 2026-03-21 | kNN+LGBM (competitor SED) | 0.923 | kNN probe change -0.002 vs 0.925 baseline |
| 2026-03-20 | v3-ensemble | 0.921 | Perch 70% Bayes + SED 50/50 VLOM + 3 tricks |
| 2026-03-17 | v9-asl-soup ensemble | 0.892 | First competitive submission |

---

## Architecture Overview

### 1. Perch Embedding Probe (current best: LB 0.926)

```
Audio (60s) → 12×5s clips → Perch v2 TFLite →
  ├── 14795-dim logits → gather 234 → Bayesian prior fusion
  └── 1536-dim embedding → PCA(64) → LGBM probe (74-dim features)
  → final = (1-α)×base + α×lgbm_pred   [α=0.40 blend weight]
  → lmax_pre_aves(α=0.1) → SoftRich(α=0.40) → cSEBBs → submission
```

Key techniques:
- **Bayesian prior fusion**: site/hour joint priors fused into logits
- **Texture smooth** (avg-neighbor, α=0.35): for Amphibia/Insecta classes
- **Event smooth** (local-max, α=0.15): for Aves classes — preserves transient peaks
- **LGBM probe**: 74-dim features (PCA32 + raw/prior/base + seq + 3 interactions)

### 2. Post-Processing Pipeline (R51, OOF AUC 0.8164)

```
SED logits → lmax_pre_aves(α=0.1, Aves-only idx 72-233)
           → SoftRich(alpha=0.40)
           → cSEBBs(cp_blend=0.60, cp_thr=0.05)
           → OOF AUC = 0.8164
```

### 3. SED — EfficientNet-B0 (v1, 5th-place inspired, running 2026-03-22)

**Complete rewrite** from scratch based on 5th place BirdCLEF 2025 solution.
Old approach was fundamentally wrong (wrong data, wrong mel, missing key augmentations).

```
train_audio/ (35k recordings) → AudioClipDataset (torchaudio, map-style)
  → GPU mel: MelSpec(n_fft=2048, hop=512, n_mels=128, fmin=50, fmax=15000) + AmplitudeToDB
  → FilterAugment (freq-band gain ramps, DCASE 2021, p=0.5)
  → SpecAugment (2 time masks, 2 freq masks)
  → SumixFreq (batch-level spectrogram mixup, key 5th-place trick)
  → EfficientNet-B0 (tf_efficientnet_b0.ns_jft_in1k, 3-ch replication)
  → GEMFreqPool(p=3.0) → AttentionSEDHead → sigmoid
  → Focal BCE (γ=2.0)
  → OneCycleLR (lr=1e-3, 30 epochs, 10% warmup)
```

**Data split**: `train_folds.csv` (pre-computed 5-fold stratified by primary_label)
**Validation**: ALL 66 labeled soundscape windows (best test-domain proxy)
**Config**: `configs/sed_b0_v1.yaml`

What changed vs old SED:
| Old | New |
|-----|-----|
| Generator-based IterableDataset | Map-style Dataset + DataLoader shuffle |
| n_fft=1024, hop=320, n_mels=128 | n_fft=2048, hop=512, n_mels=128, fmin=50, fmax=15000 |
| No SumixFreq | SumixFreq (batch-level mixup) |
| No FilterAugment | FilterAugment (DCASE 2021) |
| Dual loss (clip+frame BCE) | Single Focal BCE (γ=2.0) |
| Soundscape-based validation | All 66 labeled soundscape windows |
| Custom fold splits | Pre-computed train_folds.csv (stratified) |
| train on soundscapes too | train_audio/ only (soundscapes = validation) |

### 4. ProtoSSM — Prototypical State Space Model (v1)

Temporal model on **Perch v2 embeddings** from labeled soundscapes.

```
perch_labeled_ss.npz (66 labeled soundscape files, 12 windows each)
  → Perch embeddings (B, 12, 1536) + teacher logits + site/hour prior
  → Linear(1536→128) + LayerNorm + GELU
  → 2× BidirectionalSelectiveSSM(d_model=128, d_state=16)
  → Prototypical cosine head (234 learnable prototypes)
  → Gated fusion with Perch teacher logits (per-class α)
  → Focal BCE + 0.3×MSE distillation + 0.1×taxonomic BCE
```

**Config**: `configs/proto_ssm_v1.yaml`
**Parameters**: ~390K

---

### 5. Noisy Student Pipeline (NEW — 2026-03-22)

**Goal**: Train SED + EfficientSSM without Perch dependency (true student models).
Both models take **raw audio** as input. 4 rounds × 5 folds × 2 models.

#### SED Student (`train_sed_ns.py`)

```
train_audio/ + pseudo_labels/ns_rK.csv
  → AudioClipDataset (same as SED v1)
  → Same mel + FilterAugment + SpecAugment + SumixFreq
  → EfficientNet-B0 → GEMFreqPool → AttentionSEDHead → sigmoid
  → Focal BCE, OneCycleLR lr=1e-3, 30 epochs, early_stop=7
  → Validation: labeled soundscape OOF macro AUC
```

#### EfficientSSM Student (`train_ssm_ns.py`)

```
train_audio/ (T=1) + pseudo soundscape sequences (T=12)
  → raw clip (CLIP_SAMPLES) → Mel(128) → EfficientNet-B0(global_pool='avg') → (d_feat=1280)
  → stack T clips → (B, T, 1280) → Linear(1280→256) + LayerNorm
  → 2× BidirectionalSelectiveSSM(d_model=256, d_state=16)
  → Prototypical cosine head (234 prototypes, EMA-smoothed)
  → Focal BCE, AdamW lr=1e-3, proto_temp lr×0.01, 40 epochs, early_stop=7
  → Validation: labeled soundscape sequences, per-window AUC (uses EMA prototypes)
```

**Cosine head stabilization** (2026-03-23):
- **EMA prototypes**: `proto_ema` buffer updated after each step (`momentum=0.99`); val uses EMA, train uses learnable prototypes → eliminates AUC oscillation
- **proto_temp lr**: `lr × 0.01` (separate param group) → prevents temperature from overshooting
- Result: ep1→ep4 monotonically increasing (0.69→0.86→0.88→0.90), vs prior cosine head which oscillated ±0.15

**Pseudo label generation** (2026-03-23 update):
- **SED chain**: pure SED predictions only → `pseudo_labels/sed_r{k}.csv` (`perch_w=0, sed_w=1.0`)
- **SSM chain**: pure SSM predictions only → `pseudo_labels/ssm_r{k}.csv` (`perch_w=0, ssm_w=1.0`)
- Matches BirdCLEF 2025 1st place (Babych): Perch only used for round 0; student-only predictions for round 1+
- Pipeline: sigmoid probs → power transform (γ=2.0) → per-class 95th percentile threshold → CSV

**Orchestrators** (fully independent, no cross-chain dependency):
- `scripts/auto_sed_ns_full.sh` — SED r1→r4, generates `sed_r{k}.csv` pseudo labels
- `scripts/auto_ssm_ns_full.sh` — SSM r1→r4, generates `ssm_r{k}.csv` pseudo labels
- Skip logic: fold checkpoint exists → skip immediately; process running → wait (SED chain only waits for SED processes, SSM for SSM)

**Configs**: `configs/sed_ns_b0_r{1-4}.yaml`, `configs/ssm_ns_b0_r{1-4}.yaml`
- r2 SED: `pseudo_labels/sed_r1.csv`, r2 SSM: `pseudo_labels/ssm_r1.csv` (independent chains)

**WandB**: project=`birdclef-2026`, tags=[model, round, fold]
**Round 0**: `pseudo_labels/ns_r0.csv` (Perch teacher only, no student)

---

## Currently Running Experiments (2026-03-23)

| Experiment | Config | Status | GPU | Notes |
|-----------|--------|--------|-----|-------|
| **SED NS r1 fold3+** | `configs/sed_ns_b0_r1.yaml` | 🔄 Running | GPU1 | fold0-2 done (best: ~0.9482/0.9305/~) |
| **SSM NS r1 fold0** | `configs/ssm_ns_b0_r1.yaml` | 🔄 Running | GPU1 | ep19 AUC=0.5511, best=0.9415@ep16 |

Two fully independent chains (launched 2026-03-23):
```bash
nohup bash scripts/auto_sed_ns_full.sh > outputs/logs/auto_sed_ns_full.log 2>&1 &
nohup bash scripts/auto_ssm_ns_full.sh > outputs/logs/auto_ssm_ns_full.log 2>&1 &
```

Monitor: `tail -f outputs/logs/auto_sed_ns_full.log outputs/logs/auto_ssm_ns_full.log`

---

## Project Structure

```
BirdClef-2026-Codebase/
│
├── configs/
│   ├── sed_b0_v1.yaml              # SED v1 (5th-place inspired)
│   ├── proto_ssm_v1.yaml           # ProtoSSM v1 (Perch-based temporal model)
│   ├── sed_ns_b0_r{1-4}.yaml       # SED Noisy Student rounds 1-4
│   └── ssm_ns_b0_r{1-4}.yaml       # EfficientSSM Noisy Student rounds 1-4
│
├── birdclef-2026/
│   └── notebook resource/current_subs/  # Submission notebooks (LB 0.926)
│       ├── lgbm-infer-branchens-ssm-full.ipynb    # Full SSM blend
│       ├── lgbm-infer-branchens-ssm-light.ipynb   # Light SSM blend
│       └── lgbm-branchens-csebbs-protossm-v4-full-postpro.ipynb  # Full postproc + SSM
│
├── event_smooth/                   # Post-processing experiments (R46→R51)
│
├── src/
│   ├── data/
│   │   ├── dataset.py              # CachedEmbeddingDataset, SoundscapeDataset
│   │   └── augment.py              # Mixup, time masking, gain
│   ├── model/
│   │   ├── proto_ssm.py            # ProtoSSM, SelectiveSSM (Perch-based)
│   │   └── classifier.py           # PerchClassifier (label/embedding head)
│   └── utils/config.py             # YAML config loader
│
├── train_sed_ns.py                 # SED Noisy Student (EfficientNet-B0, raw audio)
│                                   # AudioClipDataset + PseudoSoundscapeDataset
│                                   # Focal BCE, OneCycleLR, early_stop=7, wandb
│
├── train_ssm_ns.py                 # EfficientSSM Noisy Student (raw audio)
│                                   # EfficientNet-B0 + BiSSM, T=1/12 sequences
│                                   # AdamW lr=1e-3, early_stop=7, wandb
│
├── train_proto_ssm.py              # ProtoSSM 5-fold training (Perch-based)
│
├── scripts/
│   ├── gen_pseudo_ns.py                  # Generate pseudo labels (model-only or +Perch)
│   ├── auto_sed_ns_full.sh               # SED chain r1→r4 (independent, pure SED pseudo)
│   ├── auto_ssm_ns_full.sh               # SSM chain r1→r4 (independent, pure SSM pseudo)
│   ├── extract_perch_all_ss_emb.py       # Extract Perch emb for all soundscapes
│   ├── extract_ss_labeled_embeddings.py  # Build perch_labeled_ss.npz
│   ├── monitor_experiments.py            # Status + Excel update (15-min cron)
│   └── eval_smooth_experiments.py        # Post-processing sweep
│
├── pseudo_labels/
│   ├── ns_r0.csv                   # Round 0: Perch teacher only (shared init)
│   ├── sed_r{1-4}.csv              # SED chain pseudo labels (pure SED predictions)
│   └── ssm_r{1-4}.csv              # SSM chain pseudo labels (pure SSM predictions)
│
├── outputs/
│   ├── logs/
│   │   ├── sed_ns_r1_fold{0-4}.log # SED NS training logs
│   │   └── ssm_ns_r1_fold{0-4}.log # SSM NS training logs
│   ├── sed-ns-b0-r{1-4}/           # SED NS checkpoints per round
│   └── ssm-ns-b0-r{1-4}/           # SSM NS checkpoints per round
│
└── reports/
    └── exp_results.xlsx            # Experiment log (auto-updated by monitor)
```

---

## Key Technical Findings

### What Works

| Technique | Effect | Evidence |
|-----------|--------|----------|
| Human voice removal (Silero VAD) | +0.039 LB | Ablation confirmed |
| LGBM probe (vs LogReg) | LB 0.925 | Better than LogReg; 74-dim with interaction features |
| Bayesian prior fusion | ~+0.02 LB | Site/hour priors on Perch logits |
| 3-model Perch ensemble | +0.0327 holdout | 0.9453 → 0.9780 |
| VLOM blend (Perch+SED) | LB 0.892→0.921 | Geometric-RMS blend beats linear |
| SoftRich (α=0.40) | OOF 0.8164 | Cross-file richness normalization |
| cSEBBs (cp_blend=0.60) | OOF 0.8140→0.8164 | Change-point rich-segment boosting |
| SumixFreq (batch mixup) | Key aug | 5th place: batch-level spectrogram mixup |
| FilterAugment (DCASE 2021) | Key aug | Frequency-band gain ramps |

### What Didn't Work

| Technique | Result | Notes |
|-----------|--------|-------|
| Old SED training approach | -0.013 LB | Wrong data (soundscape-centric), wrong mel, no SumixFreq |
| Dual clip+frame loss | Bug-prone | Silently dropped frame supervision; now replaced with single focal |
| ASL + secondary_weight=1.0 | Noisy gradients | Amplifies unreliable XC secondary labels |
| PT-MAP (few-shot meta-learning) | No gain | lgbm-infer = ptmap-lgbm = 0.925 |
| Gaussian smooth on probs | Weaker | Logit-space smoothing consistently better |
| Pseudo labels in SED | Breaks diversity | Collapses Perch teacher signal diversity |

### SED Root Cause Analysis (2026-03-22)

The old SED approach had multiple fundamental problems:
1. **Wrong primary data**: Trained mostly on soundscape windows (66 files) instead of `train_audio/` (35k recordings)
2. **Wrong mel params**: n_fft=1024, hop=320 → underresolved; should be n_fft=2048, hop=512, fmin=50, fmax=15000
3. **Missing SumixFreq**: The key batch-level spectrogram mixup from 5th place was missing
4. **Dual loss complexity**: clip+frame BCE was complex and bug-prone; 5th place uses single focal BCE
5. **Generator dataset**: IterableDataset can't be properly shuffled; map-style Dataset is correct

---

## Running Experiments

```bash
# Launch fully independent NS chains (r1→r4, GPU1)
nohup bash scripts/auto_sed_ns_full.sh > outputs/logs/auto_sed_ns_full.log 2>&1 &
nohup bash scripts/auto_ssm_ns_full.sh > outputs/logs/auto_ssm_ns_full.log 2>&1 &

# Single fold (SED NS)
CUDA_VISIBLE_DEVICES=1 python train_sed_ns.py --config configs/sed_ns_b0_r1.yaml --fold 0 --device cuda:0

# Single fold (SSM NS)
CUDA_VISIBLE_DEVICES=1 python train_ssm_ns.py --config configs/ssm_ns_b0_r1.yaml --fold 0 --device cuda:0

# Generate pseudo labels (SED-only, for next round)
python scripts/gen_pseudo_ns.py --round 1 --sed_dir outputs/sed-ns-b0-r1 --perch_w 0 --sed_w 1.0 --out pseudo_labels/sed_r1.csv

# Monitor progress (Excel + status)
python3 scripts/monitor_experiments.py --excel
```

---

## Submission Rules

- Only submit when **individual SED soundscape val AUC > 0.9193** (v5 benchmark), OR as part of ensemble with competitor SED
- Only evaluate/submit **nohuman models** (non-nohuman results discarded from 2026-03-15)
- All training must run on **GPU1** (`CUDA_VISIBLE_DEVICES=1`)
- **No unlabeled data** in current experiments — semi-supervised learning planned for later
- Current LB anchor: **0.926** — only submit if expected to beat this
