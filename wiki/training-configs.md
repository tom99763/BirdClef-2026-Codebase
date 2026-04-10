---
tags: [config, hyperparameters, training]
last-updated: 2026-04-08
---

# Training Configurations

All training uses YAML configs in `configs/`. This page documents the key hyperparameters and their rationale.

## Config Structure

```yaml
experiment:
  name: sed-ns-b0-20s-r1
  seed: 42
  round: 1

data:
  train_csv:              birdclef-2026/train.csv
  soundscape_labels_csv:  birdclef-2026/train_soundscapes_labels.csv
  taxonomy_csv:           birdclef-2026/taxonomy.csv
  audio_dir:              birdclef-2026/train_audio
  soundscape_dir:         birdclef-2026/train_soundscapes
  pseudo_labels_csv:      pseudo_labels/ns_r0_perch_aug.csv
  n_folds:                5

model:
  backbone:         tf_efficientnet_b0.ns_jft_in1k
  clip_duration:    20
  n_mels:           224
  n_fft:            2048
  hop_length:       512
  fmin:             0
  fmax:             16000
  top_db:           80.0
  power:            2.0
  norm:             slaney
  mel_scale:        htk
  peak_norm:        false
  dropout:          0.1
  drop_path_rate:   0.1
  gem_p_init:       3.0
  freq_mask:        24
  time_mask:        64

training:
  epochs:             30
  batch_size:         16
  learning_rate:      1.0e-3
  weight_decay:       1.0e-4
  focal_gamma:        2.0
  mixup_alpha:        0.15
  pseudo_weight:      1.0
  pseudo_mixup_alpha: 0.15
  use_sumix_freq:     true
  ema_decay:          0.999
  early_stopping_patience: 3

output:
  dir:          outputs/sed-ns-b0-20s-r1
```

## Key Hyperparameters

### Model

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| clip_duration | 20s | Covers 4 Perch 5s windows; BirdCLEF 2025 1st-place insight |
| n_mels | 224 | Matches backbone input height for EfficientNet/PVT |
| drop_path_rate | 0.1-0.15 | Stochastic depth regularization; added from R2 onward |
| gem_p_init | 3.0 | Learnable GeM pooling, between avg and max |
| freq_mask | 24 | SpecAugment frequency masking width |
| time_mask | 64 | Wider than typical (32) due to 20s clips |

### Training

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| batch_size | 16 | Smaller due to 20s clips (2x memory vs 10s) |
| learning_rate | 1e-3 | AdamW with cosine annealing to 1e-6 |
| focal_gamma | 2.0 | Standard focal loss; down-weights easy negatives |
| mixup_alpha | 0.15 | MixUp interpolation parameter (BirdCLEF 2025 1st place) |
| pseudo_weight | 1.0 | Pseudo samples weighted equally with labeled |
| use_sumix_freq | true | SumixFreq augmentation (BirdCLEF 2025 1st place) |
| ema_decay | 0.999 | EMA tracking; inherited across rounds |
| epochs | 25-30 | Avoid overconfidence; forum insight |
| early_stopping_patience | 3-4 | Stop on soundscape val AUC plateau |

### NC-Specific (R10+)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| nc_distill_beta | 0.3 | 30% KLD soft distillation (may reduce to 0.1-0.15) |
| nc_temperature | 2.0 | Temperature for soft target smoothing |

## Round Inheritance

Each round inherits from the previous round's EMA checkpoint via `prev_round_dir`:

```yaml
training:
  prev_round_dir: outputs/sed-ns-b0-20s-r1
```

The training script loads `fold{N}_ema.pt` (or `fold{N}_best.pt` as fallback) and continues training.

## Pseudo Label Schedule

Controlled by `scripts/auto_sed_ns_20s_full.sh`:

| Round | perch_w | sed_w | threshold_pct | gamma |
|-------|---------|-------|---------------|-------|
| R0 | 1.00 | 0.00 | 95 | 2.0 |
| R1 | 0.50 | 0.50 | 92 | 1.0 |
| R2 | 0.30 | 0.70 | 93 | 1.54 |
| R3 | 0.10 | 0.90 | 94 | 1.82 |
| R4+ | 0.00 | 1.00 | 95 | 2.0 |

The Perch teacher weight decreases monotonically as the SED student improves. Gamma increases to sharpen pseudo labels in later rounds.

## Config File Naming Convention

- `configs/sed_ns_b0_20s_r{N}.yaml` -- B0 NS round N
- `configs/sed_ns_pvt_20s_r{N}.yaml` -- PVT NS round N
- `configs/sed_ns_cnxt_20s_r{N}.yaml` -- ConvNeXt NS round N

## Training Data Per Fold

1. **ALL train_audio clips** (~24,000) -- no fold split, always included
2. **Pseudo-labeled soundscapes** -- from `pseudo_labels_csv`, excluding val fold files
3. **Labeled soundscapes** (training split) -- from `soundscape_labels_csv`, excluding val fold

Validation: labeled soundscape clips from the held-out fold (GroupKFold by file ID).

## Related Pages

- [[architecture-sed-model]] -- Model architecture reference
- [[augmentation]] -- Augmentation pipeline details
- [[noisy-student]] -- How configs evolve across rounds
