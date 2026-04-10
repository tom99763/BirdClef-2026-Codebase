---
tags: [architecture, backbone, efficientnet, pvt]
last-updated: 2026-04-08
---

# Backbone Comparison: EfficientNet-B0 vs PVT-v2-B0

The project uses two backbone architectures with fundamentally different inductive biases. Their diversity is critical for ensemble performance (see [[ensemble-diversity]]).

## EfficientNet-B0

| Property | Value |
|----------|-------|
| Type | CNN (convolutional) |
| timm name | `tf_efficientnet_b0.ns_jft_in1k` |
| Parameters | ~5.3M |
| Feature dim | 1280 |
| Pretrained on | ImageNet (Noisy Student + JFT) |
| Output shape | (B, 1280, F, T) before GEMFreqPool |

**Inductive biases**:
- Strong local feature extraction via depthwise separable convolutions
- Translation equivariance: a bird call at time=2s looks the same as at time=15s
- Efficient compound scaling (depth, width, resolution)
- Fast training and inference

**Strengths in this task**:
- Excellent at detecting sharp spectral features (bird calls with clear harmonic structure)
- Efficient memory usage allows larger batch sizes
- Stable training across rounds (B0 R1-R8 fold std = 0.0223)

## PVT-v2-B0

| Property | Value |
|----------|-------|
| Type | Transformer (Pyramid Vision Transformer v2) |
| timm name | `pvt_v2_b0` |
| Parameters | ~3.4M |
| Feature dim | 256 |
| Pretrained on | ImageNet |
| Output shape | (B, 256, F, T) before GEMFreqPool |

**Inductive biases**:
- Global self-attention: can model relationships across entire spectrogram
- Pyramid structure: multi-scale feature extraction
- Spatial reduction attention: efficient O(n) attention via key/value downsampling
- No hard-coded locality assumption

**Strengths in this task**:
- Better at long-range temporal patterns (species with irregular call timing)
- Improves faster through NS rounds (+0.0540 R1->R8 vs +0.0263 for B0)
- Lower fold variance (R8 fold std = 0.0099 vs 0.0223 for B0)
- More robust generalization across different soundscape recording conditions
- Attention mechanism particularly effective on "hard" samples (Fold 0 that stagnated for B0 was resolved by PVT)

## Head-to-Head Comparison

### NS Training Progress

| Round | B0 Mean AUC | PVT Mean AUC | Winner |
|-------|------------|-------------|--------|
| R1 | 0.9120 | 0.9005 | B0 |
| R2 | 0.9256 | 0.9225 | B0 |
| R3 | 0.9305 | 0.9350 | PVT |
| R4 | 0.9341 | **0.9410** | PVT |
| R6 | **0.9383** | -- | B0 best |
| R8 | 0.9381 | 0.9545 | PVT |

PVT overtakes B0 at R3 and continues to improve more rapidly. By R8, PVT leads by +0.016.

### Why Both Are Needed

Despite PVT's higher individual AUC, the ensemble benefits from both:

1. **Different error patterns**: CNN and Transformer fail on different species/samples
2. **Prediction decorrelation**: Cross-architecture pairs have lower correlation than same-architecture pairs
3. **Confirmed on LB**: B0 + PVT ensembles consistently outperform B0-only or PVT-only

## Integration with SED Model

Both backbones plug into the same `SEDModel` class:

```python
model = SEDModel(
    backbone='tf_efficientnet_b0.ns_jft_in1k',  # or 'pvt_v2_b0'
    num_classes=234,
    dropout=0.1,
    drop_path_rate=0.1,
    gem_p_init=3.0,
)
```

The GEMFreqPool and AttentionSEDHead adapt automatically to the backbone's feature dimension (1280 for B0, 256 for PVT).

## ConvNeXt (Experimental)

A ConvNeXt backbone was also tested (`sed_ns_cnxt_20s_r{1-8}.yaml`). R4 mean AUC = 0.9411, comparable to PVT R4. The pipeline was paused to focus on B0+PVT co-evolution via [[noisy-classmate]].

## Related Pages

- [[architecture-sed-model]] -- Full SED model architecture
- [[ensemble-diversity]] -- Why architecture diversity helps ensemble
- [[noisy-classmate]] -- Cross-architecture co-evolution
