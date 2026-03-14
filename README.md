# BirdClef 2026 — Perch Fine-tuning Codebase

A clean, experiment-friendly codebase for the [BirdClef 2026](https://www.kaggle.com/competitions/birdclef-2026) Kaggle competition.
Uses Google's **Perch v2** bird vocalization model as a frozen backbone, with BirdCLEF 2025 top-10 techniques integrated for maximum performance.

---

## Competition Overview

| | |
|---|---|
| **Task** | Multi-label species classification from 5-second audio segments |
| **Test data** | Continuous soundscape recordings (Pantanal, Brazil) |
| **Target classes** | 234 species — birds, amphibians, reptiles, insects |
| **Metric** | **Macro-averaged ROC-AUC** (species with no positive labels are excluded) |
| **Audio format** | OGG Vorbis, 32 kHz mono |

---

## Project Structure

```
BirdClef-2026-Codebase/
│
├── configs/
│   ├── default.yaml                  # Perch pipeline — baseline config
│   ├── debug.yaml                    # Perch pipeline — quick sanity-check
│   ├── birdclef25_improvements.yaml  # Perch + all BirdCLEF 2025 techniques
│   ├── exp_focal_isolated.yaml       # Ablation: FocalLoss only
│   ├── exp_adamw_classweights.yaml   # Ablation: AdamW + sqrt class weights only
│   ├── exp_soundscape_train.yaml     # Ablation: soundscapes in training only
│   ├── pseudo_label_round1.yaml      # Pseudo-label retraining config
│   ├── sed_default.yaml              # SED pipeline — full training config
│   └── sed_debug.yaml                # SED pipeline — quick sanity-check
│
├── src/
│   ├── data/
│   │   ├── dataset.py        # ClipDataset + SoundscapeDataset + CachedEmbeddingDataset + class weights
│   │   ├── augment.py        # Noise, gain, Mixup, time masking, background noise
│   │   └── mel_dataset.py    # MelClipDataset + MelSoundscapeDataset  [SED]
│   ├── model/
│   │   ├── classifier.py     # Perch backbone + MLP classification head
│   │   ├── losses.py         # FocalBCELoss (TF) + power_transform  [BirdCLEF25]
│   │   └── sed_model.py      # SEDModel (PyTorch) + FocalBCELossTorch  [SED]
│   ├── metrics/
│   │   └── kaggle_metric.py  # Official BirdCLEF 2026 scorer (macro ROC-AUC)
│   └── utils/
│       ├── audio.py          # Audio loading, cropping utilities
│       ├── metrics.py        # competition_roc_auc() + padded_cmap() (reference)
│       ├── config.py         # YAML loader with dot-notation access
│       └── model_soup.py     # Checkpoint weight averaging  [BirdCLEF25]
│
├── scripts/
│   │
│   │  ── Baseline experiments ────────────────────────────────────────────
│   ├── run_baseline.sh               # Reference baseline run
│   ├── run_birdclef25_base.sh        # Train with all BirdCLEF25 improvements
│   ├── exp_focal_loss.sh             # FocalLoss vs BCE, gamma sweep
│   ├── exp_class_weights.sh          # Sqrt / linear / none weighting
│   ├── exp_augmentation_25.sh        # Time masking strength sweep
│   ├── exp_pseudo_label.sh           # 2-round pseudo-labeling pipeline
│   ├── exp_model_soup.sh             # Multi-seed model soup experiment
│   └── run_birdclef25_pipeline.sh    # Full Perch+BirdCLEF25 end-to-end pipeline
│
├── train.py                  # Perch pipeline training (validates with ROC-AUC)
├── train_sed.py              # SED pipeline training  [SED]
├── inference.py              # Perch inference + TTA → submission.csv
├── inference_sed.py          # SED inference + TTA + ensemble  [SED]
├── pseudo_label.py           # Pseudo-label generation  [BirdCLEF25]
├── extract_embeddings.py     # One-time offline Perch embedding extraction
├── evaluate_final.py         # Official ROC-AUC evaluation on all checkpoints
├── orchestrate.py            # Auto-pipeline: Phase 1 → 2a → 2b → 3 → eval
├── generate_report.py        # HTML technical report from result.json files
├── analyze_results.py        # Cross-experiment comparison
└── requirements.txt
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Perch pipeline** requires TensorFlow ≥ 2.20.
> **SED pipeline** requires PyTorch ≥ 2.0 and timm ≥ 0.9.
> Both can be installed together; they run independently.

### 2. Data layout

Place the competition data and Perch model so the paths match the config:

```
BirdClef-2026-Codebase/
├── birdclef-2026/
│   ├── train_audio/                  # 35,549 individual recordings (.ogg)
│   ├── train_soundscapes/            # 8–10 long soundscape recordings (.ogg)
│   ├── train.csv
│   ├── taxonomy.csv
│   ├── train_soundscapes_labels.csv
│   └── sample_submission.csv
└── models/
    └── bird-vocalization-classifier-tensorflow2-perch_v2-v2/
        ├── saved_model.pb
        ├── assets/
        │   ├── labels.csv
        │   └── perch_v2_ebird_classes.csv
        └── variables/
```

All paths are configurable in `configs/default.yaml`.

---

## Quick Start

### Step 0 — Pre-extract embeddings (one-time, ~20 min, strongly recommended)

Running Perch on every batch is the main training bottleneck. Pre-extracting embeddings once reduces each epoch from ~20 minutes to **~40 seconds**.

```bash
# Extract train and soundscape embeddings in parallel across two GPUs
CUDA_VISIBLE_DEVICES=0 python extract_embeddings.py \
    --config configs/default.yaml --split train --batch_size 64 &

CUDA_VISIBLE_DEVICES=1 python extract_embeddings.py \
    --config configs/default.yaml --split soundscapes --batch_size 64 &

wait
```

Embeddings are saved to `outputs/embeddings_cache/` (~855 MB). All subsequent training runs detect the cache automatically — no config change needed.

### Step 1 — Run experiments

#### Option A: Auto-pipeline (recommended)

Start the two Phase 1 experiments and the orchestrator simultaneously. The orchestrator watches for Phase 1 completion and then automatically runs all subsequent phases:

```bash
# Phase 1: two experiments in parallel
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/default.yaml experiment.name=baseline &
CUDA_VISIBLE_DEVICES=1 python train.py --config configs/birdclef25_improvements.yaml experiment.name=birdclef25-base &

# Orchestrator: waits for Phase 1, then auto-runs Phase 2a → 2b → Phase 3
python orchestrate.py
```

Full auto-pipeline schedule:

```
Phase 1  : baseline (GPU 0) + birdclef25-base (GPU 1)         [parallel, already running]
Phase 2a : focal-isolated (GPU 0) + adamw-classweights (GPU 1) [parallel ablations]
Phase 2b : soundscape-in-train (GPU 0)                         [sequential ablation]
           ↑ all 5 ablations finish here
derive_best_config()  →  reads all 5 scores, builds best_derived_v1.yaml
Phase 3  : best-derived-v1 (GPU 0) + SED (GPU 1)              [parallel]
evaluate_final.py     →  official ROC-AUC score for every checkpoint
generate_report.py    →  reports/experiment_report.html
```

#### Option B: Manual

```bash
# Baseline (GPU 0)
CUDA_VISIBLE_DEVICES=0 bash scripts/run_baseline.sh

# BirdCLEF 2025 improvements (GPU 1)
CUDA_VISIBLE_DEVICES=1 bash scripts/run_birdclef25_base.sh
```

### Step 2 — Official evaluation

After all experiments finish, score every checkpoint with the official Kaggle metric:

```bash
python evaluate_final.py --config configs/default.yaml
```

Output is written to `outputs/<run>/result.json` under the key `kaggle_roc_auc`.

### Step 3 — Generate HTML report

```bash
python generate_report.py --output reports/experiment_report.html
```

### Select GPU for any script

All training and inference scripts accept a `--gpu` flag:
```bash
python train.py --config configs/default.yaml --gpu 0
python train_sed.py --config configs/sed_default.yaml --gpu 1
python inference.py --checkpoint checkpoints/best/best_head --gpu 0
python extract_embeddings.py --config configs/default.yaml --gpu 0
```

### Generate submission (Perch + TTA)
```bash
python inference.py \
    --config configs/birdclef25_improvements.yaml \
    --checkpoint checkpoints/birdclef25-base/best_head \
    --tta
```

### Generate submission (SED + TTA)
```bash
python inference_sed.py \
    --config configs/sed_default.yaml \
    --checkpoint checkpoints/sed-v1/best_sed \
    --tta
```

### Ensemble Perch + SED
```bash
python inference_sed.py \
    --config configs/sed_default.yaml \
    --checkpoint checkpoints/sed-v1/best_sed \
    --tta \
    --ensemble_with submission_perch.csv \
    --output submission_ensemble.csv
```

---

## Competition Metric

The official BirdCLEF 2026 metric is **macro-averaged ROC-AUC** over species that have at least one positive label in the solution:

```python
# Official scorer (src/metrics/kaggle_metric.py)
scored_columns = [col for col in solution if solution[col].sum() > 0]
score = sklearn.metrics.roc_auc_score(
    solution[scored_columns], submission[scored_columns], average="macro"
)
```

**Key properties:**
- Species with **no positive labels** in the validation set are completely excluded from scoring
- Predictions are **not** required to sum to 1 per row (unlike padded cMAP)
- Predictions should be **sigmoid probabilities** in [0, 1]

Training validation in `train.py` uses the same `competition_roc_auc()` function to ensure the best checkpoint selection aligns with the competition metric. `padded_cmap()` is kept in `src/utils/metrics.py` for cross-reference only.

---

## Model Architecture

```
Raw audio (batch × 160,000 samples @ 32 kHz)
        │
        ▼
┌──────────────────────────────┐
│  Google Perch v2 (frozen)    │  TF SavedModel — 14,795 species pre-trained
│  bird-vocalization-classifier│  Input: 5-second mono waveform
└──────────────────────────────┘
        │  embedding (1,536-dim)   ← tf.stop_gradient in embedding_head mode
        ▼
┌──────────────────────────────┐
│  Classification Head         │  Only this is trained
│  Dense(512) → ReLU           │
│  → Dropout(0.3)              │
│  → Dense(234)                │
└──────────────────────────────┘
        │  logits (234 classes)
        ▼
   FocalBCELoss / BinaryCrossEntropy + label smoothing
```

**Two modes** (set via `model.mode` in config):

| Mode | What trains | Speed | Use when |
|---|---|---|---|
| `embedding_head` | Head only (Perch frozen) | Fast | Default; good for quick experiments |
| `full_finetune` | Entire model | Slow | After finding good head hparams |

---

## BirdCLEF 2025 Top-10 Techniques

The following techniques from BirdCLEF 2025 prize-winning solutions are integrated:

| Technique | Source | Where | Expected Gain |
|---|---|---|---|
| **FocalLoss** | 2nd & 5th place | `src/model/losses.py` | Better rare-class learning |
| **Sqrt class weighting** | 2nd place | `src/data/dataset.py` | Rare species oversampling |
| **Time masking** | Universal | `src/data/augment.py` | Regularisation, domain robustness |
| **Soundscapes in training** | Domain adaptation | `train.py` | Closes train/test domain gap |
| **Background noise inject** | Multiple teams | `src/data/augment.py` | Soundscape robustness |
| **Pseudo-labeling + PowerTransform** | 1st place | `pseudo_label.py` | **+5–8% ROC-AUC** |
| **TTA temporal shifts** | 2nd place | `inference.py --tta` | **+1.2% AUC at inference** |
| **Model Soup** | 3rd place | `src/utils/model_soup.py` | +0.3–1% ROC-AUC |

### FocalLoss

Reduces the loss contribution from easy (confident) negatives, forcing the model to focus on hard / rare-class samples:

```
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
  γ = 0  → standard BCE
  γ = 2  → BirdCLEF 2025 2nd-place default
```

Enable in config:
```yaml
training:
  loss: "focal"
  focal_gamma: 2.0
  focal_alpha: 0.25
```

### Sqrt Inverse-Frequency Class Weighting

Rare species recordings are sampled with probability proportional to `1/sqrt(freq)`, so the model sees rare species more often per epoch:

```yaml
training:
  class_weight_mode: "sqrt"   # "none" | "sqrt" | "linear"
```

### Time Masking (SpecAugment on waveform)

Randomly zeroes out up to N contiguous waveform windows per clip:

```yaml
augmentation:
  time_masking: true
  time_mask_ratio: 0.1   # Max 10% of clip per mask
  time_mask_n: 2         # 2 independent masks
```

### Soundscapes in Training

Mixes labeled soundscape segments into the training set to close the domain gap between clean individual recordings (training) and noisy ambient soundscapes (test):

```yaml
training:
  use_soundscapes_in_train: true
```

Works with both the raw audio path and the embedding cache path. The `soundscape-in-train` ablation experiment measures its isolated contribution.

### Background Noise Injection

Mixes in real ambient noise at a random SNR. Requires a directory of noise audio files:

```yaml
data:
  noise_dir: "/path/to/noise_files"   # .ogg / .wav / .flac

augmentation:
  background_noise: true
  snr_db_range: [5.0, 30.0]
```

### Pseudo-labeling + PowerTransform (1st place — biggest gain)

The **PowerTransform** sharpens confident predictions before using them as pseudo-labels, enabling multiple stable rounds:

```
p_sharp = p ^ power
  power = 1.0 → unchanged
  power = 2.0 → BirdCLEF 2025 1st-place default (sharpens confident, suppresses uncertain)
```

### TTA Temporal Shifts (2nd place)

At inference time, also predict clips starting 2.5 seconds later and average:

```bash
python inference.py --checkpoint ... --tta
```

### Model Soup

Average weights of multiple checkpoints (different seeds, pseudo-label rounds, or hyperparameter variants):

```bash
python -m src.utils.model_soup \
    --checkpoints ckpt_a/best_head ckpt_b/best_head ckpt_c/best_head \
    --output soup/best_head \
    --config configs/birdclef25_improvements.yaml
```

---

## Ablation Experiment Design

The auto-pipeline runs isolated ablations to measure each technique's contribution independently before combining them:

| Experiment | Config | Change vs Baseline |
|---|---|---|
| `baseline` | `configs/default.yaml` | — reference point |
| `birdclef25-base` | `configs/birdclef25_improvements.yaml` | All BirdCLEF25 techniques combined |
| `focal-isolated` | `configs/exp_focal_isolated.yaml` | FocalLoss(γ=2) only |
| `adamw-classweights` | `configs/exp_adamw_classweights.yaml` | AdamW + sqrt class weights only |
| `soundscape-in-train` | `configs/exp_soundscape_train.yaml` | `use_soundscapes_in_train=true` only |

`derive_best_config()` in `orchestrate.py` reads all 5 scores and builds `configs/best_derived_v1.yaml` that combines only the techniques that improved ROC-AUC by >0.002.

---

## Pseudo-labeling Workflow

### Step 1: Run the pseudo-label experiment (2 rounds)

```bash
bash scripts/exp_pseudo_label.sh
```

This script runs the full pipeline automatically:

```
[Stage 0]  Train base model
              ↓
[Round 1]  pseudo_label.py generate  (power=2.0, threshold=0.5)
              → pseudo_labels/round1_pseudo.csv
              ↓
[Round 1]  python train.py  (retrain with pseudo-labels)
              ↓
[Round 2]  pseudo_label.py generate
              → pseudo_labels/round2_pseudo.csv
              ↓
[Round 2]  python train.py  (retrain again)
              ↓
[Soup]     model_soup.py   (average stage0 + r1 + r2 checkpoints)
              ↓
[Inference] inference.py --tta → submission_pseudo_r2.csv
```

### Step 2: Manual pseudo-label generation (optional)

```bash
# Generate Round 1 pseudo-labels from your checkpoint
python pseudo_label.py generate \
    --config configs/birdclef25_improvements.yaml \
    --checkpoint checkpoints/birdclef25-base/best_head \
    --soundscapes_dir birdclef-2026/train_soundscapes \
    --output pseudo_labels/round1_pseudo.csv \
    --threshold 0.5 \
    --power 2.0

# Retrain using Round 1 pseudo-labels
python train.py --config configs/pseudo_label_round1.yaml \
    experiment.name="pseudo-r1"

# Repeat for Round 2
python pseudo_label.py generate \
    --config configs/pseudo_label_round1.yaml \
    --checkpoint checkpoints/pseudo-r1/best_head \
    --output pseudo_labels/round2_pseudo.csv \
    --power 2.0

python train.py --config configs/pseudo_label_round1.yaml \
    experiment.name="pseudo-r2"
```

**PowerTransform `power` parameter guidance:**

| Power | Effect |
|---|---|
| `1.0` | No transform (raw probabilities) |
| `1.5` | Light sharpening |
| `2.0` | BirdCLEF 2025 1st-place default |
| `3.0` | Aggressive sharpening (only very confident labels survive) |

---

## Model Soup

Average checkpoints from multiple seeds or pseudo-label rounds.

```bash
bash scripts/exp_model_soup.sh
```

To manually soup any set of checkpoints:

```bash
python -m src.utils.model_soup \
    --checkpoints \
        checkpoints/birdclef25-base/best_head \
        checkpoints/pseudo-r1/best_head \
        checkpoints/pseudo-r2/best_head \
    --output checkpoints/soup-final/best_head \
    --config configs/birdclef25_improvements.yaml
```

---

## SED Pipeline (BirdCLEF 2025 1st & 5th place)

The SED (Sound Event Detection) pipeline is a **separate, parallel** experiment that complements the Perch pipeline. Combining both via ensemble is the highest-ROI approach.

### Architecture

```
Waveform (160,000 samples @ 32 kHz)
        │
        ▼
Mel Spectrogram (1 × 128 mel × 501 frames)   ← 5s clip, 100 frames/sec
        │
        ▼
┌──────────────────────────────────────┐
│  CNN Backbone (timm)                 │  EfficientNetV2-S / EfficientNet-B3-NS
│  ImageNet-21k pretrained             │  in_chans=1 (adapted from 3-channel)
└──────────────────────────────────────┘
        │  Feature map (B, C, H', W')
        ▼
  mean(dim=2) — pool frequency axis
        │  Time sequence (B, T', C)
        ▼
┌──────────────────────────────────────┐
│  Attention Pooling                   │
│  att  = softmax(Linear(x))           │  (B, T', num_classes) weights
│  cls  = sigmoid(Linear(x))           │  (B, T', num_classes) frame preds
│  out  = sum(att × cls, dim=T')       │  (B, num_classes) clip prediction
└──────────────────────────────────────┘
        │
        ▼
   FocalBCELoss (PyTorch)
```

**Key differences from the Perch pipeline:**

| | Perch Pipeline | SED Pipeline |
|---|---|---|
| **Framework** | TensorFlow | PyTorch |
| **Input** | Raw waveform → Perch → 1,536-dim embedding | Raw waveform → mel spectrogram |
| **Backbone** | Perch v2 (frozen, 14k-species pre-trained) | EfficientNetV2-S (ImageNet-21k) |
| **Output** | Clip-level prediction | Frame-level + clip-level prediction |
| **Training** | Head only (embedding_head mode) | Full model fine-tuning |
| **Config** | `configs/birdclef25_improvements.yaml` | `configs/sed_default.yaml` |

### Backbone Options

| Backbone | Source | Speed | Quality |
|---|---|---|---|
| `tf_efficientnet_b0_ns` | 5th place (lightweight) | Fast | Good |
| `tf_efficientnet_b3_ns` | 5th place (noisy-student) | Medium | Very Good |
| `tf_efficientnetv2_b3` | 5th place | Medium | Very Good |
| `tf_efficientnetv2_s_in21k` | 2nd & 5th place **(recommended)** | Medium | Best single |
| `eca_nfnet_l0` | 2nd place | Slow | Best single |

### SED Experiment Steps

#### Step 1: Sanity check
```bash
python train_sed.py --config configs/sed_debug.yaml
```

#### Step 2: Full SED training
```bash
bash scripts/run_sed_experiment.sh
```

#### Step 3: Ensemble SED + Perch
```bash
python inference_sed.py \
    --config configs/sed_default.yaml \
    --checkpoint checkpoints/sed-efficientnetv2s/best_sed \
    --tta \
    --ensemble_with submission_perch.csv \
    --output submission_ensemble.csv
```

### Mel Spectrogram Parameters

```yaml
mel:
  n_fft: 1024         # FFT window
  hop_length: 320     # 320 @ 32kHz → 100 frames/sec, 5s → 501 frames
  n_mels: 128         # Mel bins
  fmin: 20.0          # Hz (below bird vocalisation range)
  fmax: 16000.0       # Hz (Nyquist for 32 kHz)
```

---

## Configuration Reference

All hyperparameters live in YAML files under `configs/`. Key sections:

```yaml
experiment:
  name: "my-run"          # WandB run name and output directory name
  seed: 42

data:
  min_rating: 0.0         # Filter low-quality recordings (0 = keep all, 3.0 = recommended)
  use_secondary_labels: true
  noise_dir: null         # Path to background noise files (enables background_noise aug)

audio:
  n_clips_per_file: 3     # Random crops per recording per epoch

training:
  epochs: 50
  batch_size: 256          # 256 recommended when using embedding cache
  optimizer: "adamw"      # "adam" or "adamw"
  learning_rate: 1.0e-3
  scheduler: "cosine"     # Cosine annealing with warm-up
  warmup_epochs: 3
  mixup_alpha: 0.3        # 0 = disabled
  label_smoothing: 0.05
  use_soundscapes_in_train: false  # Mix soundscape segments into training
  # BirdCLEF 2025
  loss: "focal"           # "bce" or "focal"
  focal_gamma: 2.0
  focal_alpha: 0.25
  class_weight_mode: "sqrt"  # "none" | "sqrt" | "linear"

model:
  mode: "embedding_head"
  hidden_dim: 512
  dropout: 0.3

augmentation:
  enabled: true
  noise_level: 0.005
  gain_range: [0.7, 1.3]
  # BirdCLEF 2025
  time_masking: true
  time_mask_ratio: 0.1
  time_mask_n: 2
  background_noise: false  # needs data.noise_dir
  snr_db_range: [5.0, 30.0]
```

---

## Training Data

| Source | Files | Usage |
|---|---|---|
| `train_audio/` | 35,549 individual recordings | **Training** — random 5-second clips |
| `train_soundscapes/` + labels | 8–10 soundscape recordings | **Validation** + optionally training |

Validation uses soundscapes (not individual recordings) because the test set consists of continuous soundscape recordings — this makes the validation metric directly comparable to the leaderboard score. Setting `use_soundscapes_in_train: true` also mixes them into training to help close the domain gap.

---

## WandB Integration

Set `wandb.enabled: true` in your config:

```yaml
wandb:
  enabled: true
  project: "birdclef-2026"
  entity: "your-username"
```

Each run logs:
- `train/loss` per epoch
- `val/roc_auc` per epoch (official competition metric)
- `val/best_roc_auc` whenever a new best is achieved
- `lr` per epoch

---

## Pre-extracting Embeddings (strongly recommended)

Running Perch on every batch is the main training bottleneck. Pre-extracting once gives a **~100× speedup** for `embedding_head` training:

| Mode | Batch speed | Epoch time | 50-epoch total |
|---|---|---|---|
| Raw audio (no cache) | ~1.4 batch/s | ~20 min/epoch | **~16 hours** |
| Cached embeddings + `@tf.function` | ~11 batch/s | ~40 sec/epoch | **~35 min** |

### How it works

`extract_embeddings.py` runs Perch once on all clips and saves 1536-dim float32 vectors as `.npy` files. The training script automatically detects the cache via `manifest.csv` and switches to a `@tf.function`-compiled head-only training path.

### Structure

```
outputs/embeddings_cache/
├── manifest.csv          # index: npy_path, source_file, clip_idx, label, split
├── train/                # one .npy per clip from train_audio/
│   ├── XC12345.ogg_c0.npy
│   ├── XC12345.ogg_c1.npy
│   └── ...               # ~106,647 files, ~640 MB
└── soundscape/           # one .npy per labeled soundscape segment
    └── ...               # ~739 files, ~5 MB
```

Each `.npy` is a `float32` array of shape `(1536,)` — the Perch v2 embedding for one 5-second clip.

### Extraction commands

```bash
# Recommended: run both splits in parallel across two GPUs
CUDA_VISIBLE_DEVICES=0 python extract_embeddings.py \
    --config configs/default.yaml --split train --batch_size 64 &

CUDA_VISIBLE_DEVICES=1 python extract_embeddings.py \
    --config configs/default.yaml --split soundscapes --batch_size 64 &

wait

# Or extract everything on a single GPU
python extract_embeddings.py --config configs/default.yaml --batch_size 64
```

### Cache auto-detection

`train.py` checks for `outputs/embeddings_cache/manifest.csv` at startup. If found and `model.mode == "embedding_head"`, it automatically:
- Loads embeddings from `.npy` files (skips Perch backbone entirely)
- Uses `@tf.function`-compiled `train_epoch_cached` path (graph mode)
- Does Mixup in TF ops inside the compiled graph

No config change needed — the switch is transparent.

---

## Key Files Reference

### Perch Pipeline (TensorFlow)

| File | Purpose |
|---|---|
| `train.py` | Training loop; validates with `competition_roc_auc`; saves `result.json` |
| `inference.py` | Processes soundscapes → `submission.csv` (supports `--tta`) |
| `extract_embeddings.py` | One-time offline embedding extraction; appends to existing manifest |
| `evaluate_final.py` | Scores all checkpoints with the official Kaggle ROC-AUC metric |
| `orchestrate.py` | Full auto-pipeline: Phase 1 → 2a → 2b → 3 → eval → report |
| `generate_report.py` | Generates Bootstrap+Chart.js HTML report from `result.json` files |
| `pseudo_label.py` | Generates pseudo-labels with PowerTransform |
| `src/model/classifier.py` | `PerchClassifier` — Perch backbone + MLP head |
| `src/model/losses.py` | `FocalBCELoss` (TF) + `power_transform` |
| `src/data/dataset.py` | `ClipDataset`, `SoundscapeDataset`, `CachedEmbeddingDataset`, `compute_class_weights` |
| `src/metrics/kaggle_metric.py` | Official BirdCLEF 2026 scorer (`score()` = macro ROC-AUC) |
| `src/utils/metrics.py` | `competition_roc_auc()` (training validation) + `padded_cmap()` (reference) |

### SED Pipeline (PyTorch)

| File | Purpose |
|---|---|
| `train_sed.py` | SED training loop (PyTorch) |
| `inference_sed.py` | SED inference + TTA + Perch ensemble |
| `src/model/sed_model.py` | `SEDModel` — EfficientNet + attention pooling; `FocalBCELossTorch` |
| `src/data/mel_dataset.py` | `MelClipDataset`, `MelSoundscapeDataset`, `compute_mel` |

---

## Performance Notes

### Training speed (embedding_head mode on RTX 4090 24 GB)

| Setup | Batch/s | Epoch time | 50-epoch total |
|---|---|---|---|
| Raw audio, eager | ~1.4 | ~20 min | ~16 hours |
| Cached embeddings + `@tf.function` | ~11 | ~40 sec | ~35 min |

### Progress bars

Both the epoch loop and batch loop show tqdm progress with live loss / ROC-AUC:

```
Epochs:  42%|████████████      | 21/50 [14:02, loss=0.0312, roc_auc=0.8821, best=0.8956, lr=3.2e-4, t=40s]
  train: 78%|████████████      | 325/416 [00:30, 11.2batch/s, loss=0.0318]
```

### GPU selection

All scripts support `--gpu` (sets `CUDA_VISIBLE_DEVICES`):
```bash
python train.py --config configs/default.yaml --gpu 0
python train.py --config configs/default.yaml --gpu 1
```

---

## Notes

- **Perch coverage**: Perch v2 covers 203 out of 234 target species by scientific name matching. The remaining 31 species will always receive near-zero predictions unless the head learns to infer them from co-occurring species embeddings.
- **Domain gap**: Individual recordings (`train_audio`) are clean, close-mic recordings; test soundscapes are ambient recordings with noise and multiple simultaneous species. Using soundscapes for validation (not individual recordings) is critical for an honest ROC-AUC estimate. `use_soundscapes_in_train: true` directly addresses this gap during training.
- **Label quality**: The `rating` column in `train.csv` (0–5) reflects crowd-sourced quality scores. Filtering with `data.min_rating=3` is a common way to remove noisy labels at the cost of fewer rare-species examples.
- **Metric alignment**: Training validation and checkpoint selection use `competition_roc_auc()` which matches the official Kaggle scorer exactly. `evaluate_final.py` runs the full official `score()` function on inference outputs after all experiments complete.
- **Pseudo-label rounds**: Each round requires a fresh full training run. The PowerTransform (`power=2.0`) is critical — without it, pseudo-label confidence values are too flat and label quality degrades across rounds.
- **Model Soup best practices**: Only average checkpoints that are individually strong (above baseline). Averaging with a weak checkpoint hurts performance.
- **Embedding manifest race condition**: Two parallel extraction runs each appended to the shared manifest. The manifest append+dedup logic in `extract_embeddings.py` prevents overwrite when running splits in parallel.
