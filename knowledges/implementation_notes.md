# Implementation Notes

## Current System (0.918 LB)
```
Perch 1536-dim embeddings
→ StandardScaler (zero-mean, unit-var per dim)
→ PCA(64) — drops to 64 dims
→ LogisticRegression per class (C=0.5, liblinear)
   - Only for classes with ≥8 positives in labeled soundscapes
   - Feature: [Z_64, raw_score, prior_score, base_score, prev/next/mean/max_base]
→ alpha=0.40 blend: (1-alpha)*base_score + alpha*probe_score
```

## Planned Upgrades (prioritized by expected impact)

### Tier 1: Low effort, high gain (implement first)
1. **L2-normalize before PCA** — SimpleShot insight, free improvement
2. **Prototype fallback for min_pos<8** — handles rare species currently skipped
3. **Increase PCA dims**: 64→128 or 256, or use full-dim ZCA whitening

### Tier 2: Medium effort, high gain
4. **Tip-Adapter**: support cache retrieval + blend with Perch native logit
5. **Embedding-graph smoothing per file** (LaplacianShot-lite): replace current 1D temporal smooth with 2D embedding-based smooth

### Tier 3: Higher effort, potentially high gain
6. **Iterative prototype refinement**: use query clips to update prototypes
7. **LP++ init**: initialize LogReg coef_ from class prototype

## Results Log
| Method | Holdout AUC | LB | Notes |
|--------|------------|-----|-------|
| Perch only (no probe) | — | 0.899 | Baseline |
| + PCA(64) + LogReg | — | 0.918 | v1 probe (+0.019) |
| v2: L2+PCA(128,whiten)+LogReg+Proto+TipAdapter+GraphSmooth | LOO +0.15 | **0.915** | Full-data fit |
| v2: GroupKFold-5 (per-fold PCA/LogReg/TIP, no leakage) | — | **TBD** | Expected ≥ 0.915 |

---

## Why B3 > B0 on Holdout Despite Soundscape Instability (2026-03-20)

**Observed**: `sed-b3-v1-asl` — soundscape val only 0.7805 (oscillating), but holdout = **0.9553**
This is the largest soundscape↔holdout gap we've seen. Two papers explain why:

### Paper 1: Generalization in birdsong classification (Scientific Reports 2025)
*arxiv.org/abs/2409.15383*
- Larger domain-pretrained backbones consistently outperform smaller ones on holdout soundscapes
- **Shallow fine-tuning** (freeze early layers, train only head) generalizes better than deep fine-tuning
- Early layers from large-scale pretraining encode rich low-level features that survive fine-tuning noise
- **Implication**: B3's extra capacity in frozen early layers is the source of the holdout advantage, not soundscape adaptation quality

### Paper 2: NoisyStudent (Xie et al., CVPR 2020)
*arxiv.org/abs/1911.04252*
- NoisyStudent trains with aggressive augmentation (RandAugment, dropout, stochastic depth) → smoother decision boundaries
- ImageNet-C corruption error drops 45.7→28.3; ImageNet-A accuracy 61%→83.7%
- **Implication for us**: EfficientNet-B3-NS weights encode corruption-robust features from the start. When fine-tuned on noisy/variable-quality bird audio, backbone is less likely to overfit to recording artifacts → better holdout generalization

### Practical Rule for Future B3 Experiments
If soundscape fine-tuning is unstable on B3:
1. **Lower backbone LR**: differential LR (head: 5e-4, backbone: 1e-4) rather than global lower LR
2. **Freeze early layers** for first N epochs, then unfreeze gradually
3. The instability is in the soundscape head — the backbone features are still excellent
4. `sed-b3-v2-lower-lr` (lr=2e-4, warmup=5ep) should help, but differential LR is better long-term
