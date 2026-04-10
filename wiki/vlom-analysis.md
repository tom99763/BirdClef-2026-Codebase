---
tags: [analysis, vlom, ensemble, statistics]
last_updated: 2026-04-09
---

# VLOM Blend Weight Analysis

## Summary

The VLOM (Variance-weighted Log-Odds Mean) blend combines SED and Perch predictions. Optimal weight: **w_SED=0.70, w_Perch=0.30** (LB 0.943).

## LB Results

| w_SED | w_Perch | LB |
|-------|---------|-----|
| 0.50 | 0.50 | 0.937 |
| **0.70** | **0.30** | **0.943** |
| 0.75 | 0.25 | 0.942 |
| 0.90 | 0.10 | 0.928 |

## CV-LB Inversion

The most important finding: **CV and LB give opposite recommendations**.

- CV optimal: w_SED ≈ 0.50 (Perch solo AUC 0.992 >> SED 0.959)
- LB optimal: w_SED = 0.70

### Root Cause: Logit Magnitude Ratio

On our 739 labeled windows, SED and Perch logit magnitudes are equal (ratio = 0.998). But on the hidden test, SED's effective logit is ~2.33× Perch's, implying distribution shift:
- More rare species (SED adapted via pseudo labels)
- Harder acoustics (SED augmentation training)
- Temporal complexity (SED 20s attention > Perch 5s)

## Statistical Tests

| Test | Result | Conclusion |
|------|--------|-----------|
| Bootstrap paired (CV) | w=0.65 > w=0.70, P=100% | CV unreliable (inversion) |
| Wilcoxon signed-rank | w=0.65 wins 17/0 classes, p<0.0001 | CV-biased |
| Bayesian posterior (4 LB pts) | Peak mode = 0.695 | Too few data points |
| Cohen's d | All < 0.2 | Negligible effect size |
| GP regression (4 LB pts) | Peak ~0.54, σ=0.022 | High uncertainty |

**Verdict**: No statistical evidence to change from 0.70. CV tests are invalid due to inversion.

## Score Distribution (21 Submissions)

- Mean ± Std: 0.9389 ± 0.0036
- Non-normal (Shapiro-Wilk p=0.006), left-skewed
- 3 clusters: [0.928-0.934], [0.937-0.938], [0.940-0.943]
- Bootstrap P(max > 0.943) = 0%
- Bayesian ceiling: 0.953 [0.946, 0.959]
- Power: ~13 submissions needed for 0.944

## Diminishing Returns

```
+0.005 (NS baseline)
  → +0.003 (fold diversity)
    → +0.001 (round diversity)
      → +0.001 (PVT fold)
        → -0.002 (NC)
          → -0.001 (VLOM tuning)
```

## Recommended Strategy

1. **4-model INT8 ensemble** (+0.001-0.003, low effort)
2. **NC v2 confidence fix** (+0.001-0.002, medium effort)
3. **Per-class VLOM weight** (+0.001-0.003, medium effort)
4. **TTA 2.5s shift** (+0.001-0.002, low effort)
5. **Platt calibration** (uncertain, low effort)

## Related Pages

- [[ensemble-diversity]] — Why diversity > AUC
- [[noisy-classmate]] — NC framework and confidence issue
- [[lb-experiments]] — Full submission history
- [[training-configs]] — Model configurations

## References

Full LaTeX report: `reports/vlom_analysis.tex` with 13 references and 19 sub-figures.
