---
tags: [nc3, architecture, ensemble, diversity]
last_updated: 2026-04-10
---

# NC v3 — 3-Architecture Co-evolution

## Motivation

NC v1/v2 (B0+PVT) failed to beat NS ensemble (LB 0.941 vs 0.943) due to confidence collapse. NC v3 takes a different approach: instead of fixing confidence, use **3 completely new architectures** for maximum ensemble diversity.

## Architecture Selection

| Architecture | timm name | Params | ONNX FP32 | CPU (ms) | Type |
|-------------|-----------|--------|-----------|----------|------|
| **ConvNeXt-Femto** | `convnext_femto.d1_in1k` | 4.83M | ~18.7MB | 11.2 | Pure CNN (large kernel depthwise + LayerNorm) |
| **FastViT-T8** | `fastvit_t8.apple_dist_in1k` | 3.26M | ~12.9MB | 21.9 | Hybrid (RepMixer + MHSA, Apple distilled) |
| **RegNetY-008** | `regnety_008.pycls_in1k` | 5.49M | ~22MB | 16.4 | Pure CNN (structured design, BirdCLEF 2025 1st place) |

### Why these 3?

1. **Maximum architectural diversity**: ConvNeXt (modern CNN), FastViT (hybrid reparameterizable), RegNetY (structured regular CNN) — all different from each other AND from B0/PVT
2. **Similar speed budget**: All within 11-22ms on CPU, compatible with Kaggle 90-min budget
3. **Proven foundations**: RegNetY used by BirdCLEF 2025 winner, ConvNeXt/FastViT are SOTA lightweight architectures

## Pipeline Design

### Phase 0 — Bootstrap (B0+PVT as Perch)
- Source: NS B0 R11 + NS PVT R8 corrected predictions (50/50 blend)
- Flags: nonaves_perch_only, confidence_weighting, disagreement_mining, soft_labels
- Train: ConvNeXt R1, FastViT R1, RegNetY R1 (all from scratch)

### Phase 1 — First NC (teacher reduction)
- Source: B0+PVT (weight 0.3) + ConvNeXt+FastViT+RegNetY (weight 0.7)
- Train: All 3 architectures R2 (with EMA from R1)

### Phase 2+ — Full 3-way NC
- Source: ConvNeXt + FastViT + RegNetY only (no more B0/PVT)
- Full NC features: disagreement mining, soft distillation, confidence weighting
- Continue co-evolution R3, R4, ...

## Ensemble Usage

Final Kaggle notebook: pick best fold from each architecture → 3-model INT8 ensemble + per-class VLOM.

## Related Pages

- [[noisy-classmate]] — NC v1 framework
- [[ensemble-diversity]] — Why diversity > AUC
- [[backbones]] — EfficientNet-B0 vs PVT-v2-B0 comparison
