---
tags: [training, noisy-student, pseudo-labels]
last-updated: 2026-04-08
---

# Noisy Student (NS) Framework

Multi-round self-training where a teacher generates pseudo labels for unlabeled soundscapes, then a student trains on labeled + pseudo-labeled data, becomes the new teacher, and the cycle repeats.

## Motivation

The test set consists of soundscape recordings from the Pantanal region. Training data (`train_audio/`) contains isolated point recordings with very different acoustic characteristics. NS bridges this domain gap by iteratively training on pseudo-labeled soundscapes.

## Pipeline (per round)

```
Round R:
  1. Train 5 folds:  train_sed_ns.py --config sed_ns_{arch}_20s_r{R}.yaml --fold {0..4}
  2. Infer all soundscapes: --infer_all_ss -> all_ss_probs.npz
  3. Residual Corrector: train_sed_residual_corrector.py -> all_ss_probs_corrected.npz
  4. Generate pseudo labels: gen_pseudo_ns.py -> pseudo_labels/sed_20s_r{R}.csv
  5. Update next round config -> repeat
```

## Round Progression

### B0 Chain

```
Perch Teacher -> R0 pseudo -> B0 R1 -> R1 pseudo -> B0 R2 -> ... -> B0 R12
```

### PVT Chain (independent, seeded from B0 R4 pseudo)

```
B0 R4 pseudo -> PVT R1 -> PVT R1 pseudo -> PVT R2 -> ... -> PVT R8
```

## Pseudo Label Generation

Implemented in `scripts/gen_pseudo_ns.py`. Pipeline:

1. **Collect** soft probabilities from Perch teacher + SED 5-fold ensemble
2. **Ensemble** with weighted average (weights decay Perch over rounds)
3. **Merge** 5s Perch windows to 20s clips (max-pool across overlapping windows)
4. **Dynamic threshold** per-class at the Nth percentile
5. **Save** soft probs for windows with at least one species above threshold

### Per-Round Pseudo Label Schedule

| Round | Perch Weight | SED Weight | Threshold Pct | Gamma | Notes |
|-------|-------------|-----------|---------------|-------|-------|
| R0 | 1.00 | 0.00 | 95 | 2.0 | Perch-only teacher |
| R1 | 0.50 | 0.50 | 92 | 1.00 | Bootstrap, raw soft labels |
| R2 | 0.30 | 0.70 | 93 | 1.54 | Reduce teacher weight |
| R3 | 0.10 | 0.90 | 94 | 1.82 | Student dominant |
| R4+ | 0.00 | 1.00 | 95 | 2.00 | Pure self-training |

### Key Parameters

- **Gamma (power transform)**: Sharpens predictions by raising probs to the power of gamma. From BirdCLEF 2025 1st place. Compresses low-confidence scores, preserves high-confidence signals.
- **Dynamic threshold**: Per-class threshold at the Nth percentile of transformed probabilities. Floor=0.05, ceiling=0.50.
- **Non-Aves Perch-only**: For non-Aves species (Amphibia, Insecta, Mammalia, Reptilia), the ensemble is replaced with pure Perch teacher probs because SED has poor discriminability for these taxa.
- **Labeled exclusion**: The 66 labeled soundscapes are excluded from pseudo labeling to prevent data leakage.

## Perch Teacher

The initial teacher is a linear head fine-tuned on frozen Perch v2 embeddings. It generates R0 pseudo labels for all unlabeled soundscapes.

```bash
python3 train.py --config configs/exp_nohuman_label_soundscape_train.yaml
python3 scripts/extract_perch_teacher_all_ss.py --output outputs/perch_teacher_aug_all_ss.csv
```

## Residual Corrector

After each round's inference, a [[residual-corrector]] (BiSSM) learns the residual between SED predictions and Perch teacher predictions, then applies a correction with weight alpha=0.40. This corrected npz is used for pseudo label generation.

## Training Details

Each fold trains on:
- **ALL** `train_audio/` clips (~24,000) with ground-truth labels
- **Pseudo-labeled** soundscape windows (~117,000-120,000) from the previous round
- **Labeled** soundscapes (training split only, excluding val fold)

Cross-domain MixUp (lambda=0.5) pairs each labeled clip with a pseudo-labeled clip 1:1.

## Automation Scripts

| Script | Purpose |
|--------|---------|
| `scripts/auto_sed_ns_20s_full.sh` | B0 R1-R4 sequential pipeline |
| `scripts/auto_sed_ns_20s_r5r8.sh` | B0 R5-R8 extension |
| `scripts/auto_sed_ns_20s_r9r15.sh` | B0 R9-R15 extension |
| `scripts/auto_sed_ns_pvt_20s_r1r4.sh` | PVT R1-R4 pipeline |
| `scripts/auto_sed_ns_pvt_20s_r5r8.sh` | PVT R5-R8 pipeline |

## Val AUC by Round

### EfficientNet-B0

| Round | f0 | f1 | f2 | f3 | f4 | Mean |
|-------|------|------|------|------|------|------|
| R1 | 0.9015 | 0.9149 | 0.9009 | 0.9289 | 0.9138 | 0.9120 |
| R2 | 0.9096 | 0.9294 | 0.9229 | 0.9402 | 0.9261 | 0.9256 |
| R3 | 0.9014 | 0.9426 | 0.9151 | 0.9563 | 0.9369 | 0.9305 |
| R4 | 0.9034 | 0.9480 | 0.9238 | 0.9544 | 0.9408 | 0.9341 |
| R5 | 0.9023 | 0.9451 | 0.9210 | 0.9595 | 0.9370 | 0.9330 |
| R6 | 0.9052 | 0.9524 | 0.9246 | 0.9664 | 0.9431 | **0.9383** |
| R7 | 0.9047 | 0.9590 | 0.9216 | 0.9577 | 0.9451 | 0.9376 |
| R8 | 0.9016 | 0.9571 | 0.9292 | 0.9595 | 0.9431 | 0.9381 |

Best mean: R6 (0.9383). R7-R12 plateau -- diminishing returns from pure self-training.

### PVT-v2-B0

| Round | f0 | f1 | f2 | f3 | f4 | Mean |
|-------|------|------|------|------|------|------|
| R1 | 0.8991 | 0.9243 | 0.8519 | 0.9101 | 0.9170 | 0.9005 |
| R2 | 0.9108 | 0.9182 | 0.9064 | 0.9455 | 0.9316 | 0.9225 |
| R3 | 0.9225 | 0.9380 | 0.9218 | 0.9516 | 0.9410 | 0.9350 |
| R4 | 0.9313 | 0.9493 | 0.9202 | 0.9567 | 0.9473 | **0.9410** |

PVT improves faster than B0 (+0.0540 over R1-R8 vs +0.0263 for B0).

## Limitation

Each NS chain only learns from its own predictions, leading to confirmation bias and knowledge silos. This motivated the [[noisy-classmate]] framework.

## Related Pages

- [[noisy-classmate]] -- Cross-architecture evolution that addresses NS limitations
- [[residual-corrector]] -- BiSSM correction applied between rounds
- [[training-configs]] -- Full hyperparameter reference
- [[pipeline-automation]] -- Auto scripts and monitoring
