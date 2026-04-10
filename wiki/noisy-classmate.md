---
tags: [training, noisy-classmate, innovation]
last-updated: 2026-04-08
---

# Noisy Classmate (NC) Framework

Noisy Classmate is our novel contribution: multiple heterogeneous-architecture students stop self-studying (NS) and start peer-teaching. Five progressive phases introduce increasingly sophisticated cross-architecture knowledge transfer.

## Motivation

NS has a fundamental limitation: each chain only learns from its own predictions, creating confirmation bias and knowledge silos. B0 and PVT have different inductive biases (CNN vs Transformer), so they make different errors. NC exploits this complementarity.

## The Five Phases

| Phase | Name | What It Does | NS Equivalent |
|-------|------|-------------|---------------|
| 1 | Ensemble Pseudo | Blend B0 + PVT predictions for shared pseudo labels | Self-only pseudo |
| 2 | Confidence Weighting | Per-sample entropy-weighted fusion (trust more confident model) | Equal weight |
| 3 | Disagreement Mining | 3x training weight on samples where models disagree | Equal weight |
| 4 | Soft Distillation | FocalBCE + KLD loss preserving full probability structure | Hard labels only |
| 5 | Bidirectional Loop | B0->PVT->B0->PVT alternating co-evolution | Unidirectional |

## Phase Details

### Phase 1: Ensemble Pseudo Labels

Simple weighted average of B0 and PVT predictions (default: 0.5/0.5). Even this basic blending reduces individual model blind spots.

```python
blended = 0.5 * b0_probs + 0.5 * pvt_probs
```

### Phase 2: Confidence-Aware Blending

Instead of fixed weights, each sample uses the model with lower entropy (higher confidence) more heavily. Per-sample adaptive weighting.

```python
# Per-sample entropy for each chain
entropies = [binary_entropy(probs) for probs in chain_probs]
# Inverse entropy * base weight -> normalize per sample
sample_weights = inv_entropy * base_weight / sum(...)
blended = sum(w_i * probs_i for w_i, probs_i in zip(sample_weights, chain_probs))
```

Implemented in `confidence_aware_blend()` in `scripts/gen_noisy_classmate_pseudo.py`.

### Phase 3: Disagreement Mining

Computes per-sample disagreement (mean variance across chains per species). High disagreement = classmates disagree = high learning potential. These samples get up to 3x training weight.

```python
disagreement = var(chain_probs, axis=0).mean(axis=1)  # per sample
train_weights = 1.0 + alpha * normalize(disagreement)  # alpha=2.0 -> max 3x
```

The `_nc_weight` column in the pseudo CSV carries these weights into training.

### Phase 4: Soft Distillation

NCDistillLoss in `train_sed_ns.py` combines hard and soft targets:

```
L = (1 - beta) * FocalBCE(logits, hard_targets)
  + beta * KLD(sigmoid(logits/T), soft_probs) * T^2
```

- `beta=0.3`: 30% soft distillation, 70% hard classification
- `T=2.0`: Temperature smoothing for KLD
- Soft probs are saved as `*_soft.npz` alongside the hard pseudo CSV

### Phase 5: Bidirectional Co-Evolution

The key innovation. Instead of each architecture only learning from itself:

```
NC Generation 1:  B0 R11 + PVT R8 -> blend -> PVT R9
NC Generation 2:  B0 R11 + PVT R9 -> blend -> PVT R10
                  PVT R10 + B0 R11 -> blend -> B0 R12  <- knowledge flows back!
NC Generation 3:  B0 R12 + PVT R10 -> blend -> PVT R11
                  ... (continues co-evolution)
```

Implemented in `scripts/auto_nc_dual_gpu.sh` with alternating rounds.

## NC vs NS Comparison

| Aspect | NS | NC |
|--------|----|----|
| Pseudo source | Self (previous round) | Cross-architecture blend |
| Blending | None (single model) | Confidence-weighted |
| Sample importance | Equal | Disagreement-weighted (up to 3x) |
| Loss | FocalBCE (hard labels) | FocalBCE + KLD soft distillation |
| Knowledge flow | Unidirectional (self -> self) | Bidirectional (B0 <-> PVT) |
| Training cost | 1 chain | 2 chains alternating |

## Implementation Files

| File | Purpose |
|------|---------|
| `scripts/gen_noisy_classmate_pseudo.py` | Phase 1-4: blend + confidence + disagreement + soft labels |
| `train_sed_ns.py` | `NCDistillLoss`, `_nc_weight` in `PseudoSoundscapeDataset` |
| `scripts/auto_nc_dual_gpu.sh` | Phase 5: bidirectional dual-GPU pipeline |
| `scripts/auto_nc_full.sh` | Phase 5: single-GPU bidirectional pipeline |
| `scripts/monitor_nc.sh` | Pipeline monitoring |
| `reports/noisy_classmate_plan.html` | Research plan with theoretical justification |

## NC Pseudo Label Generation

```bash
python3 scripts/gen_noisy_classmate_pseudo.py \
    --chains "b0:outputs/sed-ns-b0-20s-r11" "pvt:outputs/sed-ns-pvt-20s-r8" \
    --weights 0.5 0.5 \
    --confidence_weighting \
    --disagreement_mining \
    --soft_labels \
    --nonaves_perch_only \
    --percentile 95 --gamma 2.0 \
    --out pseudo_labels/noisy_classmate_pvt_r9.csv
```

## NC Config (add to training YAML for R10+)

```yaml
training:
  nc_distill_beta: 0.3    # Phase 4: KLD soft distillation weight
  nc_temperature: 2.0     # Phase 4: temperature for soft targets
```

## Early Results

NC PVT R9 fold0 reached 0.9700 (NS PVT R8 fold0 was 0.9575), confirming significant improvement from cross-architecture learning.

## Known Issues

- `nc_distill_beta=0.3` may be too aggressive; R10 showed regression in most folds. Consider reducing to 0.1-0.15 for stability.
- NC outputs go to separate directories (`outputs/sed-ns-{arch}-20s-r{R}-nc/`) to avoid mixing with NS checkpoints.

## Related Pages

- [[noisy-student]] -- Foundation framework that NC extends
- [[ensemble-diversity]] -- Why cross-architecture diversity matters
- [[backbones]] -- B0 vs PVT architectural differences
- [[pipeline-automation]] -- Dual-GPU automation details
