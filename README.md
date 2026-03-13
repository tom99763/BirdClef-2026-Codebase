# BirdClef 2026 — Perch Fine-tuning Codebase

A clean, experiment-friendly codebase for the [BirdClef 2026](https://www.kaggle.com/competitions/birdclef-2026) Kaggle competition.
Uses Google's **Perch v2** bird vocalization model as a frozen backbone and trains a lightweight classification head on the 234 target species.

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
│   ├── default.yaml          # Full training configuration
│   └── debug.yaml            # Quick sanity-check run (200 files, 3 epochs)
│
├── src/
│   ├── data/
│   │   ├── dataset.py        # ClipDataset (train_audio) + SoundscapeDataset (soundscapes)
│   │   └── augment.py        # Gaussian noise, random gain, Mixup
│   ├── model/
│   │   └── classifier.py     # Perch backbone + MLP classification head
│   └── utils/
│       ├── audio.py          # Audio loading, cropping utilities
│       ├── metrics.py        # Padded cMAP (competition metric)
│       └── config.py         # YAML loader with dot-notation access
│
├── scripts/                  # Shell scripts for running experiments
│   ├── run_baseline.sh       # Reference baseline run
│   ├── exp_lr_sweep.sh       # Learning rate sweep
│   ├── exp_mixup.sh          # Mixup alpha sweep
│   ├── exp_architecture.sh   # Hidden dim & dropout sweep
│   ├── exp_data_quality.sh   # Rating filter, secondary labels, soundscape data
│   ├── exp_clips_per_file.sh # Number of random crops per recording
│   └── run_strong.sh         # Final run with best hyperparameters
│
├── train.py                  # Main training script
├── inference.py              # Inference & submission generation
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

> **Note:** Kaggle requires TensorFlow ≥ 2.20.
> The default Kaggle environment ships with 2.19 — the starter notebook includes a wheel upgrade step.

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

### Sanity check (fast, no WandB)
```bash
python train.py --config configs/debug.yaml
```

### Full training run
```bash
python train.py --config configs/default.yaml
```

### Override any config value from the CLI
```bash
python train.py --config configs/default.yaml \
    training.learning_rate=5e-4 \
    model.dropout=0.5 \
    experiment.name="my_experiment"
```

### Generate submission
```bash
python inference.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/my_experiment/best_head
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
   BinaryCrossEntropy + label smoothing
```

**Two modes** (set via `model.mode` in config):

| Mode | What trains | Speed | Use when |
|---|---|---|---|
| `embedding_head` | Head only (Perch frozen) | Fast | Default; good for quick experiments |
| `full_finetune` | Entire model | Slow | After finding good head hparams |

---

## Training Data

| Source | Files | Usage |
|---|---|---|
| `train_audio/` | 35,549 individual recordings | **Training** — random 5-second clips |
| `train_soundscapes/` + labels | 8–10 soundscape recordings | **Validation** — segment-level, matches test format |

Validation uses soundscapes (not individual recordings) because the test set consists of continuous soundscape recordings — this makes the validation metric directly comparable to the leaderboard score.

---

## Configuration

All hyperparameters live in YAML files under `configs/`.
Key sections:

```yaml
experiment:
  name: "my-run"          # Used as WandB run name and output directory name
  seed: 42

training:
  epochs: 50
  batch_size: 64
  learning_rate: 1.0e-3
  scheduler: "cosine"     # Cosine annealing with warm-up
  warmup_epochs: 3
  mixup_alpha: 0.3        # 0 = disabled
  label_smoothing: 0.05

model:
  mode: "embedding_head"
  hidden_dim: 512
  dropout: 0.3

data:
  min_rating: 0.0         # Filter low-quality recordings (0 = keep all)
  use_secondary_labels: true

audio:
  n_clips_per_file: 3     # Random crops per recording per epoch
```

---

## Experiment Workflow

### Step 1 — Run sweeps

```bash
bash scripts/run_baseline.sh
bash scripts/exp_lr_sweep.sh
bash scripts/exp_mixup.sh
bash scripts/exp_architecture.sh
bash scripts/exp_data_quality.sh
bash scripts/exp_clips_per_file.sh
```

Each run saves its result to `outputs/<run_name>/result.json` (updated every epoch).

### Step 2 — Analyze results

```bash
# Print ranked comparison + sensitivity analysis
python analyze_results.py

# Also auto-update run_strong.sh with the best hyperparameters found
python analyze_results.py --update-strong
```

Sample output:
```
EXPERIMENT RANKINGS
──────────────────────────────────────────────────────────────────
Rank  Run name                    cMAP   Best ep  Progress
   1  lr_sweep_3e-3              0.7234       31  █████████████████████░░░
   2  mixup_alpha_0.4            0.7198       28  ████████████████████░░░░
   3  baseline                   0.7112       35  ██████████████████░░░░░░

HYPERPARAMETER SENSITIVITY
──────────────────────────────────────────────────────────────────
learning_rate      0.0122  ████████░░░░░  3e-3 → 0.7234
mixup_alpha        0.0086  ██████░░░░░░░  0.4  → 0.7198
...
```

### Step 3 — Run the strong model

Update `scripts/run_strong.sh` with the best settings (or use `--update-strong`), then:

```bash
bash scripts/run_strong.sh
```

---

## WandB Integration

Set `wandb.enabled: true` in your config and optionally set your username:

```yaml
wandb:
  enabled: true
  project: "birdclef-2026"
  entity: "your-username"    # optional
```

Each run logs:
- `train/loss` per epoch
- `val/padded_cmap` per epoch
- `val/best_padded_cmap` whenever a new best is achieved
- `lr` per epoch

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

| File | Purpose |
|---|---|
| `train.py` | Main training loop; saves `result.json` per run |
| `inference.py` | Processes test soundscapes → `submission.csv` |
| `extract_embeddings.py` | One-time offline embedding extraction |
| `analyze_results.py` | Cross-experiment comparison and sensitivity analysis |
| `src/model/classifier.py` | `PerchClassifier` — backbone + head |
| `src/data/dataset.py` | `ClipDataset`, `SoundscapeDataset` |
| `src/utils/metrics.py` | `padded_cmap()` — the competition metric |

---

## Notes

- **Perch coverage**: Perch v2 covers 203 out of 234 target species by scientific name matching.  The remaining 31 species will always receive near-zero predictions unless the head learns to infer them from co-occurring species embeddings.
- **Domain gap**: Individual recordings (`train_audio`) are clean, close-mic recordings; test soundscapes are ambient recordings with noise and multiple simultaneous species.  Using soundscapes for validation (not individual recordings) is critical for an honest cMAP estimate.
- **Label quality**: The `rating` column in `train.csv` (0–5) reflects crowd-sourced quality scores.  Filtering with `data.min_rating=3` is a common way to remove noisy labels at the cost of fewer rare-species examples.
