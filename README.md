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
| **Metric** | Padded class-wise Mean Average Precision (padded cMAP) |
| **Audio format** | OGG Vorbis, 32 kHz mono |

---

## Project Structure

```
BirdClef-2026-Codebase/
│
├── configs/
│   ├── default.yaml                  # Perch pipeline — full training config
│   ├── debug.yaml                    # Perch pipeline — quick sanity-check
│   ├── birdclef25_improvements.yaml  # Perch + BirdCLEF 2025 techniques
│   ├── pseudo_label_round1.yaml      # Pseudo-label retraining config
│   ├── sed_default.yaml              # SED pipeline — full training config
│   └── sed_debug.yaml                # SED pipeline — quick sanity-check
│
├── src/
│   ├── data/
│   │   ├── dataset.py        # ClipDataset + SoundscapeDataset + class weights
│   │   ├── augment.py        # Noise, gain, Mixup, time masking, background noise
│   │   └── mel_dataset.py    # MelClipDataset + MelSoundscapeDataset  [SED]
│   ├── model/
│   │   ├── classifier.py     # Perch backbone + MLP classification head
│   │   ├── losses.py         # FocalBCELoss (TF) + power_transform  [BirdCLEF25]
│   │   └── sed_model.py      # SEDModel (PyTorch) + FocalBCELossTorch  [SED]
│   └── utils/
│       ├── audio.py          # Audio loading, cropping utilities
│       ├── metrics.py        # Padded cMAP (competition metric)
│       ├── config.py         # YAML loader with dot-notation access
│       └── model_soup.py     # Checkpoint weight averaging  [BirdCLEF25]
│
├── scripts/
│   │
│   │  ── Baseline experiments ────────────────────────────────────────────
│   ├── run_baseline.sh               # Reference baseline run
│   ├── exp_lr_sweep.sh               # Learning rate sweep
│   ├── exp_mixup.sh                  # Mixup alpha sweep
│   ├── exp_architecture.sh           # Hidden dim & dropout sweep
│   ├── exp_data_quality.sh           # Rating filter, secondary labels
│   ├── exp_clips_per_file.sh         # Number of random crops per recording
│   ├── run_strong.sh                 # Final run with best hyperparameters
│   │
│   │  ── BirdCLEF 2025 improvements ─────────────────────────────────────
│   ├── run_birdclef25_base.sh        # Train with all BirdCLEF25 improvements
│   ├── exp_focal_loss.sh             # FocalLoss vs BCE, gamma sweep
│   ├── exp_class_weights.sh          # Sqrt / linear / none weighting
│   ├── exp_augmentation_25.sh        # Time masking strength sweep
│   ├── exp_pseudo_label.sh           # 2-round pseudo-labeling pipeline
│   ├── exp_model_soup.sh             # Multi-seed model soup experiment
│   ├── run_birdclef25_pipeline.sh    # Full Perch+BirdCLEF25 end-to-end pipeline
│   │
│   │  ── SED experiments ────────────────────────────────────────────────
│   ├── run_sed_experiment.sh         # Full SED experiment + ensemble
│   └── exp_sed_backbone.sh           # Backbone comparison sweep (5 backbones)
│
├── train.py                  # Perch pipeline training
├── train_sed.py              # SED pipeline training  [SED]
├── inference.py              # Perch inference + TTA → submission.csv
├── inference_sed.py          # SED inference + TTA + ensemble  [SED]
├── pseudo_label.py           # Pseudo-label generation  [BirdCLEF25]
├── extract_embeddings.py     # Pre-extract Perch embeddings to disk
├── analyze_results.py        # Compare all experiment results
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
>
> On Kaggle, TensorFlow ships as 2.19 — the starter notebook includes a wheel upgrade step.

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

### Perch pipeline — sanity check
```bash
python train.py --config configs/debug.yaml
```

### Perch pipeline — full training
```bash
python train.py --config configs/birdclef25_improvements.yaml
```

### SED pipeline — sanity check
```bash
python train_sed.py --config configs/sed_debug.yaml
```

### SED pipeline — full training
```bash
python train_sed.py --config configs/sed_default.yaml
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

## Model Architecture

```
Raw audio (batch × 160,000 samples @ 32 kHz)
        │
        ▼
┌──────────────────────────────┐
│  Google Perch v2 (frozen)    │  TF SavedModel — 14,795 species pre-trained
│  bird-vocalization-classifier│  Input: 5-second mono waveform
└──────────────────────────────┘
        │  embedding (1,280-dim)   ← tf.stop_gradient in embedding_head mode
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

The following techniques from BirdCLEF 2025 prize-winning solutions are integrated and enabled by `configs/birdclef25_improvements.yaml`:

| Technique | Source | Where | Expected Gain |
|---|---|---|---|
| **FocalLoss** | 2nd & 5th place | `src/model/losses.py` | Better rare-class learning |
| **Sqrt class weighting** | 2nd place | `src/data/dataset.py` | Rare species oversampling |
| **Time masking** | Universal | `src/data/augment.py` | Regularisation, domain robustness |
| **Background noise inject** | Multiple teams | `src/data/augment.py` | Soundscape robustness |
| **Pseudo-labeling + PowerTransform** | 1st place | `pseudo_label.py` | **+5–8% cMAP** |
| **TTA temporal shifts** | 2nd place | `inference.py --tta` | **+1.2% AUC at inference** |
| **Model Soup** | 3rd place | `src/utils/model_soup.py` | +0.3–1% cMAP |

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

Randomly zeroes out up to N contiguous waveform windows per clip, teaching the model to identify species from partial audio:

```yaml
augmentation:
  time_masking: true
  time_mask_ratio: 0.1   # Max 10% of clip per mask
  time_mask_n: 2         # 2 independent masks
```

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

See full pseudo-label workflow below.

### TTA Temporal Shifts (2nd place)

At inference time, also predict clips starting 2.5 seconds later and average with the original predictions:

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

## Experiment Workflow

### Phase 1 — Baseline sweeps (original pipeline)

Establish a solid baseline and find the best base hyperparameters.

```bash
# Reference point
bash scripts/run_baseline.sh

# Hyperparameter sweeps
bash scripts/exp_lr_sweep.sh
bash scripts/exp_mixup.sh
bash scripts/exp_architecture.sh
bash scripts/exp_data_quality.sh
bash scripts/exp_clips_per_file.sh

# Analyze all results
python analyze_results.py

# Train the best combination
bash scripts/run_strong.sh
```

---

### Phase 2 — BirdCLEF 2025 improvements

Evaluate each BirdCLEF 2025 technique in isolation, then combine the best settings.

#### Step 2-1: FocalLoss sweep

Compares BCE vs FocalLoss with different gamma values.

```bash
bash scripts/exp_focal_loss.sh
```

Runs: `bce_baseline`, `focal_gamma1`, `focal_gamma2`, `focal_gamma3`, `focal_gamma2_no_alpha`

#### Step 2-2: Class weighting sweep

Compares uniform vs sqrt vs linear inverse-frequency sampling.

```bash
bash scripts/exp_class_weights.sh
```

Runs: `class_weight_none`, `class_weight_sqrt`, `class_weight_linear`

#### Step 2-3: Augmentation sweep

Compares different time masking strengths.

```bash
bash scripts/exp_augmentation_25.sh
```

Runs: `aug_baseline_25`, `aug_time_mask_2x10`, `aug_time_mask_3x15`, `aug_time_mask_1x5`

#### Step 2-4: Analyze and train best BirdCLEF25 base model

```bash
# Compare all Phase 2 runs
python analyze_results.py

# Train the best BirdCLEF25 config
bash scripts/run_birdclef25_base.sh
```

---

### Phase 3 — Pseudo-labeling (biggest gain)

Implements the multi-round noisy student approach from BirdCLEF 2025 1st place.

#### Step 3-1: Run the pseudo-label experiment (2 rounds)

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

#### Step 3-2: Manual pseudo-label generation (optional)

To control each step individually:

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

### Phase 4 — Model Soup

Average checkpoints from multiple seeds or pseudo-label rounds.

```bash
bash scripts/exp_model_soup.sh
```

This script runs two variants:

**Option A — 3-seed soup:** trains 3 independent runs with seeds 42/123/777, then averages.

**Option B — Pseudo-round soup:** if pseudo-label checkpoints exist, soups them automatically.

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
| **Input** | Raw waveform → Perch → 1280-dim embedding | Raw waveform → mel spectrogram |
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

This runs:
1. Debug sanity check (3 epochs, 200 files)
2. Full training (50 epochs, EfficientNetV2-S)
3. Inference with TTA → `submission_sed.csv`
4. Ensemble with Perch submission (if available) → `submission_ensemble.csv`

#### Step 3: Backbone sweep (optional)
```bash
bash scripts/exp_sed_backbone.sh
```

Runs 5 backbones (B0-NS, B3-NS, V2-B3, V2-S, NFNet-L0) sequentially and compares cMAP.

#### Step 4: Ensemble SED + Perch
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

### Phase 5 — Final inference

```bash
# With TTA (recommended)
python inference.py \
    --config configs/birdclef25_improvements.yaml \
    --checkpoint checkpoints/soup-final/best_head \
    --output submission_final.csv \
    --tta

# Without TTA (faster)
python inference.py \
    --config configs/birdclef25_improvements.yaml \
    --checkpoint checkpoints/soup-final/best_head \
    --output submission_final.csv
```

---

### Full one-command pipeline

Run all phases (base training → 2 pseudo-label rounds → soup → TTA inference):

```bash
bash scripts/run_birdclef25_pipeline.sh
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
  batch_size: 64
  optimizer: "adamw"      # "adam" or "adamw"
  learning_rate: 1.0e-3
  scheduler: "cosine"     # Cosine annealing with warm-up
  warmup_epochs: 3
  mixup_alpha: 0.3        # 0 = disabled
  label_smoothing: 0.05
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
| `train_soundscapes/` + labels | 8–10 soundscape recordings | **Validation** — segment-level, matches test format |

Validation uses soundscapes (not individual recordings) because the test set consists of continuous soundscape recordings — this makes the validation metric directly comparable to the leaderboard score.

---

## WandB Integration

Set `wandb.enabled: true` in your config and optionally set your username:

```yaml
wandb:
  enabled: true
  project: "birdclef-2026"
  entity: "your-username"
```

Each run logs:
- `train/loss` per epoch
- `val/padded_cmap` per epoch
- `val/best_padded_cmap` whenever a new best is achieved
- `lr` per epoch

---

## Analyzing Results

```bash
# Print ranked comparison + sensitivity analysis across all runs
python analyze_results.py

# Also auto-update run_strong.sh with the best hyperparameters
python analyze_results.py --update-strong
```

Sample output:
```
EXPERIMENT RANKINGS
──────────────────────────────────────────────────────────────────
Rank  Run name                    cMAP   Best ep  Progress
   1  pseudo-r2                  0.8210       28  ████████████████████░░░░
   2  pseudo-r1                  0.7950       31  ████████████████████░░░░
   3  birdclef25-base            0.7680       35  ██████████████████░░░░░░
   4  focal_gamma2               0.7450       33  ██████████████████░░░░░░
   5  baseline                   0.7112       35  ████████████████░░░░░░░░

HYPERPARAMETER SENSITIVITY
──────────────────────────────────────────────────────────────────
pseudo_labeling    0.0530  ████████████░  round2 → 0.8210
focal_loss         0.0340  ████████░░░░░  gamma2 → 0.7450
class_weight_mode  0.0122  ███░░░░░░░░░░  sqrt   → 0.7380
...
```

---

## Pre-extracting Embeddings (optional, for faster iteration)

Running Perch on every batch during training is the bottleneck.
Pre-extract all embeddings once and cache to disk:

```bash
python extract_embeddings.py --config configs/default.yaml
```

This writes `.npy` files to `outputs/embeddings_cache/` and a `manifest.csv` index.
Subsequent training runs can load from cache instead of re-running Perch each epoch.

---

## Key Files Reference

### Perch Pipeline (TensorFlow)

| File | Purpose |
|---|---|
| `train.py` | Training loop; saves `result.json` per run |
| `inference.py` | Processes soundscapes → `submission.csv` (supports `--tta`) |
| `pseudo_label.py` | Generates pseudo-labels with PowerTransform |
| `extract_embeddings.py` | One-time offline embedding extraction |
| `src/model/classifier.py` | `PerchClassifier` — Perch backbone + MLP head |
| `src/model/losses.py` | `FocalBCELoss` (TF) + `power_transform` |
| `src/data/dataset.py` | `ClipDataset`, `SoundscapeDataset`, `compute_class_weights` |

### SED Pipeline (PyTorch)

| File | Purpose |
|---|---|
| `train_sed.py` | SED training loop (PyTorch) |
| `inference_sed.py` | SED inference + TTA + Perch ensemble |
| `src/model/sed_model.py` | `SEDModel` — EfficientNet + attention pooling; `FocalBCELossTorch` |
| `src/data/mel_dataset.py` | `MelClipDataset`, `MelSoundscapeDataset`, `compute_mel` |
| `configs/sed_default.yaml` | SED full training config |
| `configs/sed_debug.yaml` | SED debug config |

### Shared

| File | Purpose |
|---|---|
| `analyze_results.py` | Cross-experiment comparison and sensitivity analysis |
| `src/data/augment.py` | Noise, gain, Mixup, time masking, background noise |
| `src/utils/metrics.py` | `padded_cmap()` — the competition metric |
| `src/utils/model_soup.py` | Checkpoint weight averaging |

---

## Notes

- **Perch coverage**: Perch v2 covers 203 out of 234 target species by scientific name matching. The remaining 31 species will always receive near-zero predictions unless the head learns to infer them from co-occurring species embeddings.
- **Domain gap**: Individual recordings (`train_audio`) are clean, close-mic recordings; test soundscapes are ambient recordings with noise and multiple simultaneous species. Using soundscapes for validation (not individual recordings) is critical for an honest cMAP estimate.
- **Label quality**: The `rating` column in `train.csv` (0–5) reflects crowd-sourced quality scores. Filtering with `data.min_rating=3` is a common way to remove noisy labels at the cost of fewer rare-species examples.
- **Pseudo-label rounds**: Each round requires a fresh full training run. The PowerTransform (`power=2.0`) is critical — without it, pseudo-label confidence values are too flat and label quality degrades across rounds.
- **Model Soup best practices**: Only average checkpoints that are individually strong (above baseline). Averaging with a weak checkpoint hurts performance.
