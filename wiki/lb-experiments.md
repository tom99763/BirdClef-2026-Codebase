---
tags: [experiments, lb, submissions]
last-updated: 2026-04-08
---

# LB Experiment History

Complete record of Kaggle leaderboard submissions with analysis.

## Submission History

| Date | Config | LB | Key Change |
|------|--------|----|-----------|
| 2026-04-08 | B0 R12 f0 + PVT R5 f2 + B0 R6 f3 | **0.943** | PVT fold2 > fold4 (diversity) |
| 2026-04-07 | B0 R12 f0 + PVT R5 f4 + B0 R6 f3 | 0.942 | Max round diversity (R6 vs R12 = 6 rounds) |
| 2026-04-07 | B0 R12 f0 + PVT R5 f4 + B0 R4 f3 | 0.941 | R4 fold3 round diversity more but no gain |
| 2026-04-07 | B0 R12 f0 + PVT R5 f4 + B0 R8 f3 | 0.941 | fold0 + late round |
| 2026-04-07 | B0 R12 f0 + PVT R5 f4 + B0 R12 f2 | 0.940 | Same-round penalty |
| 2026-04-06 | B0 R10 f2 + PVT R5 f4 + B0 R8 f3 | 0.938 | Same fold upgrade |
| 2026-04-06 | B0 R8 f2 + PVT R5 f4 + B0 R8 f3 | 0.938 | Original baseline |
| 2026-04-06 | B0 R8 f2 + PVT R5 f4(w=2.0) + B0 R8 f3 | 0.938 | Weight adjustment no effect |
| 2026-04-06 | B0 R9 f2 + PVT R5 f4 + B0 R8 f3 | 0.937 | R9 fold2 worse |
| 2026-04-06 | B0 R8 f2 + PVT R7 f4 + B0 R8 f3 | 0.937 | PVT R5->R7 upgrade fails |
| 2026-04-06 | B0 R8 f2 + PVT R8 f0 + B0 R8 f3 | 0.934 | PVT fold change fails |
| 2026-04-06 | B0 R5 f2 + PVT R5 f4 + B0 R5 f3 | 0.933 | All R5 worse |
| 2026-03-30 | v17 baseline | 0.933 | Original baseline |

## Key Lessons

### 1. Round Diversity is King

The jump from 0.938 to 0.941 came from replacing B0 R8 f2 with B0 R12 f0. The further jump to 0.942 came from using B0 R6 f3 instead of R8 f3 (maximizing the round gap between the two B0 models).

### 2. Same-Round Penalty

B0 R12 f0 + B0 R12 f2 = 0.940. Both models trained on identical R11 pseudo labels, so their predictions are correlated (0.987). Using R6 f3 instead (corr ~0.983) gives 0.942.

### 3. PVT Fold Matters for Diversity, Not AUC

PVT R5 fold4 (val 0.9514) was thought to be the only effective fold. But fold2 (val 0.9235) actually produces higher LB (0.943 vs 0.942). The reason: fold combination 0/2/3 has higher prediction diversity than 0/4/3. Different fold val sets are not comparable.

### 4. PVT Round Upgrades Can Hurt

PVT R7 f4 (0.937) and PVT R8 f0 (0.934) both scored worse than PVT R5 f4 (0.938). Later PVT rounds may overfit to pseudo labels, reducing their complementarity with B0.

### 5. Weight Adjustments Have Minimal Effect

Giving PVT 2x weight (0.938) produced the same LB as equal weights (0.938). The diversity of model selection matters far more than weight tuning.

### 6. The GT Paradox

Models with higher GT soundscape AUC do not necessarily score higher on LB. This is because labeled soundscapes are not representative of test conditions. See [[ensemble-diversity]].

## Notebook Configuration (Current Best)

```python
SED_CHECKPOINTS = [
    {'name': 'b0_0',  'onnx_path': '.../sed_ns_b0_r12_fold0.onnx',
     'backbone': 'tf_efficientnet_b0.ns_jft_in1k', 'weight': 1.0, 'clip_sec': 20},
    {'name': 'pvt_2', 'onnx_path': '.../sed_ns_pvt_r5_fold2.onnx',
     'backbone': 'pvt_v2_b0', 'weight': 1.0, 'clip_sec': 20},
    {'name': 'b0_3',  'onnx_path': '.../sed_ns_r6_fold3.onnx',
     'backbone': 'tf_efficientnet_b0.ns_jft_in1k', 'weight': 1.0, 'clip_sec': 20},
]
```

## Related Pages

- [[ensemble-diversity]] -- Theory behind diversity > AUC
- [[onnx-export]] -- How models are exported for submission
- [[overview]] -- Score progression timeline
