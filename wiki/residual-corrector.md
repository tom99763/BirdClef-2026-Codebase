---
tags: [architecture, residual-corrector, bissm]
last-updated: 2026-04-08
---

# Residual Corrector (BiSSM)

A lightweight Temporal Residual Corrector that learns to correct SED predictions by modeling the residual between SED outputs and Perch teacher outputs.

Implemented in `scripts/train_sed_residual_corrector.py`.

## Motivation

SED predictions are noisy, especially in early rounds. The Perch teacher, while imperfect on real soundscapes, provides a useful correction signal. Rather than blending SED and Perch at prediction time (which requires Perch at inference), the corrector learns the residual pattern and applies it to SED predictions offline before pseudo label generation.

## Architecture

```
SED probs (234-d per 5s frame)
  -> Linear(234 -> 128) + LayerNorm + GELU + Dropout(0.1)
  -> BiSSM(d_model=128, d_state=16)
       Forward SSM: SelectiveSSM(128, 16)
       Backward SSM: SelectiveSSM(128, 16) on flipped sequence
       Merge: Linear(256 -> 128)
       Residual connection + LayerNorm
  -> Linear(128 -> 234) -> residual delta
  -> corrected = SED_probs + alpha * delta
```

The `TemporalResidualCorrector` class uses bidirectional Selective SSM (Mamba-style) consistent with the ProtoSSM module in the codebase.

## Key Design Choices

### Bidirectional SSM

Uses both forward and backward SSM passes merged via linear projection. This allows the corrector to use both past and future context within a soundscape file (12 windows of 5s each = 60s total).

### Zero Initialization

The output head is initialized to zeros:
```python
nn.init.zeros_(self.output_head.weight)
nn.init.constant_(self.output_head.bias, 0.0)
```

This ensures corrections start near-zero and grow only where the data supports them, preventing catastrophic initial corrections.

### Alpha Parameter

The correction strength is controlled by `alpha`:
```
corrected = clip(SED_probs + alpha * correction, 0, 1)
```

Default `alpha=0.40`. This is a hyperparameter set empirically; it balances correction magnitude against stability.

## Training

### Data Preparation

1. Load SED predictions: `all_ss_probs.npz` (N rows x 234 species)
2. Load Perch teacher predictions: `perch_teacher_aug_all_ss.csv`
3. Align by row_id
4. Group into file-level tensors: (n_files, 12, 234) -- 12 windows per 60s file
5. Target = teacher_probs - SED_probs (residual in probability space)

### Training Loop

| Parameter | Value |
|-----------|-------|
| d_model | 128 |
| d_state | 16 |
| dropout | 0.10 |
| learning_rate | 3e-4 |
| optimizer | AdamW (weight_decay=1e-3) |
| scheduler | CosineAnnealing |
| epochs | 80 |
| patience | 15 |
| batch_size | 64 |
| loss | MSE on residuals |
| val split | 12% of files |

### Effectiveness

PVT R8 corrected vs uncorrected: macro-AUC **+0.0113** (0.9565 -> 0.9678), with 48/70 species classes improving. The corrector is consistently beneficial across rounds.

## Usage in Pipeline

After each round's `infer_all_ss`:

```bash
python3 scripts/train_sed_residual_corrector.py \
    --sed_dir   outputs/sed-ns-b0-20s-r1 \
    --teacher   outputs/perch_teacher_aug_all_ss.csv \
    --round     1 \
    --alpha     0.40 \
    --out_ckpt  checkpoints/sed_corrector_r1.pt
```

Outputs:
- `checkpoints/sed_corrector_r{R}.pt` -- model checkpoint
- `{sed_dir}/all_ss_probs_corrected.npz` -- corrected predictions

The auto scripts (`auto_sed_ns_20s_full.sh`, `auto_nc_dual_gpu.sh`) call the corrector automatically. The NC pseudo label generator (`gen_noisy_classmate_pseudo.py`) prefers `all_ss_probs_corrected.npz` over `all_ss_probs.npz` when available.

## Parameter Count

Approximately ~200K trainable parameters -- lightweight enough to train in under a minute.

## Related Pages

- [[noisy-student]] -- Where the corrector fits in the NS pipeline
- [[architecture-sed-model]] -- The SED model whose outputs are corrected
- [[pipeline-automation]] -- Automated corrector integration
