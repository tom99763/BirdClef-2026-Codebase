---
tags: [ensemble, diversity, lb-strategy]
last-updated: 2026-04-08
---

# Ensemble Diversity

The single most important insight for LB improvement: **prediction decorrelation (diversity) matters more than individual model AUC**.

## The GT Paradox

On the 1,478 labeled soundscape windows (ground truth):
- Perch macro-AUC: **0.9915**
- SED B0 R8 macro-AUC: **0.9431**

Perch dominates on GT. But on the LB (test soundscapes):
- SED-dominant ensemble (SED_W=0.7): **0.938**

Why? The labeled soundscapes are "easy" -- carefully selected, clean recordings. Test soundscapes have more noise, overlapping species, and different recording conditions. SED's 20s window training provides better real-domain adaptation.

**Lesson**: Do not use GT soundscape AUC as a LB proxy.

## Round Diversity

Models from different NS rounds have lower prediction correlation because they were trained on different pseudo labels.

| Pair | Correlation | LB Impact |
|------|-------------|-----------|
| R12 f0 + R12 f2 | 0.987 | 0.940 (same-round penalty) |
| R12 f0 + R8 f3 | 0.983 | 0.941 (round diversity) |
| R12 f0 + R6 f3 | lower | **0.942** (more round diversity) |

Same-round models share identical pseudo labels, producing highly correlated predictions. Different-round models learned from different pseudo label distributions.

## Architecture Diversity

B0 (CNN) and PVT (Transformer) have fundamentally different inductive biases:
- B0: strong local feature extraction, translation equivariance
- PVT: global attention, better at long-range temporal patterns

Combining them reduces correlated errors. B0 + PVT > B0 + B0 in ensemble.

## Fold Diversity

The current best ensemble uses folds 0, 2, 3 -- maximally spread across the 5-fold space. Each fold has a different validation split, leading to different optimization targets.

**Critical insight (2026-04-08)**: PVT R5 fold2 (val AUC 0.9235) outperforms fold4 (val AUC 0.9514) on LB (0.943 vs 0.942) despite much lower val AUC. This is because fold combination 0/2/3 has higher prediction diversity than 0/4/3.

## Ensemble Selection Rules (Empirically Validated)

1. **Maximize round diversity**: Pick models from different NS rounds (e.g., R6 + R12 = 6-round gap)
2. **Never use same-round models**: Same pseudo labels -> high correlation -> worse LB
3. **Architecture diversity**: Always include at least one B0 and one PVT model
4. **Fold diversity**: Use non-overlapping folds (e.g., 0/2/3 rather than 0/1/2)
5. **Val AUC is not LB**: Higher individual AUC can mean lower LB (GT Paradox)
6. **Use prediction correlation as metric**: target corr < 0.98 between ensemble members

## Current Best Ensemble (LB 0.943)

| Model | Round | Fold | Val AUC | Role |
|-------|-------|------|---------|------|
| B0 | R12 | 0 | 0.9651 | Late-round CNN |
| PVT | R5 | 2 | 0.9235 | Early-round Transformer (diversity) |
| B0 | R6 | 3 | 0.9664 | Mid-round CNN (6-round gap from R12) |

All three models use different folds (0, 2, 3), different rounds (R5, R6, R12), and two different architectures (B0, PVT).

## Non-Aves Species

For non-Aves taxa (Amphibia, Insecta, Mammalia, Reptilia), Perch has overwhelming advantage:

| Taxon | SED B0 R8 | Perch | Gap |
|-------|-----------|-------|-----|
| Mammalia | 0.8632 | 0.9955 | -0.1323 |
| Insecta | 0.9586 | 0.9944 | -0.0358 |
| Amphibia | 0.9559 | 0.9865 | -0.0307 |
| Reptilia | 0.9390 | 0.9697 | -0.0307 |

The Kaggle notebook gives higher ensemble weight to Perch/CLAP for non-Aves classes.

## Related Pages

- [[lb-experiments]] -- Full submission history demonstrating diversity effects
- [[backbones]] -- Why B0 and PVT have different error patterns
- [[noisy-classmate]] -- NC framework designed to maximize cross-architecture diversity
