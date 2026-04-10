---
tags: [architecture, model, sed]
last-updated: 2026-04-08
---

# SED Model Architecture

The core model is a Sound Event Detection (SED) network that processes raw waveforms and outputs per-species probabilities. Defined in `train_sed_ns.py`.

## Pipeline Overview

```
20s waveform (640,000 samples @ 32kHz)
  -> MelTransform (GPU-accelerated)
  -> 3-channel mel spectrogram (B, 3, 224, T)
  -> SpecAugment (freq + time masking)
  -> [SumixFreq] (optional, mel-level)
  -> [MixUp] (wave-level, before mel)
  -> Backbone (EfficientNet-B0 or PVT-v2-B0)
  -> GEMFreqPool (collapse frequency axis)
  -> AttentionSEDHead (attention-weighted classification)
  -> sigmoid -> (B, 234) probabilities
```

## MelTransform

Converts raw waveforms to normalized mel spectrograms on GPU. Key parameters:

| Parameter | Value | Notes |
|-----------|-------|-------|
| sample_rate | 32,000 Hz | Standard for bioacoustic tasks |
| n_mels | 224 | Matches backbone input height |
| n_fft | 2048 | ~64ms window |
| hop_length | 512 | ~16ms hop |
| fmin | 0 Hz | Full frequency range |
| fmax | 16,000 Hz | Nyquist for 32kHz |
| power | 2.0 | Power spectrogram |
| norm | slaney | Slaney-style mel normalization |
| mel_scale | htk | HTK frequency scale |
| top_db | 80.0 | Dynamic range |

Processing steps:
1. NaN/Inf cleanup on input waveform
2. Optional peak normalization (absmax)
3. MelSpectrogram -> AmplitudeToDB
4. Per-sample min-max normalization to [0, 1]
5. Expand to 3 channels via `unsqueeze(1).repeat(1, 3, 1, 1)`

Source: `MelTransform` class in `train_sed_ns.py` (lines 122-148).

## GEMFreqPool

Generalized Mean (GeM) pooling along the frequency axis. Collapses the frequency dimension while preserving the time dimension.

```python
class GEMFreqPool(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        self.p = nn.Parameter(torch.tensor(p_init))  # learnable

    def forward(self, x):  # x: (B, C, F, T)
        p = self.p.clamp(min=1.0)
        return x.clamp(min=eps).pow(p).mean(dim=2).pow(1.0 / p)  # -> (B, C, T)
```

- `p=1.0` -> average pooling
- `p=inf` -> max pooling
- `p_init=3.0` -> between average and max; learned during training

Source: `GEMFreqPool` class in `train_sed_ns.py` (lines 74-82).

## AttentionSEDHead

Attention-based Sound Event Detection head. Uses soft attention over time frames to weight classification scores.

```python
class AttentionSEDHead(nn.Module):
    def __init__(self, feat_dim, num_classes, dropout=0.1):
        self.fc = Linear(feat_dim, feat_dim) + ReLU + Dropout
        self.att_conv = Conv1d(feat_dim, num_classes, 1)  # attention weights
        self.cls_conv = Conv1d(feat_dim, num_classes, 1)  # class logits

    def forward(self, x):  # x: (B, feat_dim, T_frames)
        x = self.fc(x.permute(0,2,1)).permute(0,2,1)
        att = softmax(tanh(att_conv(x)), dim=-1)  # (B, 234, T)
        cls = cls_conv(x)                          # (B, 234, T)
        logit = (att * cls).sum(-1)                # (B, 234)
        return {'clipwise_logit': logit, 'clipwise_prob': sigmoid(logit)}
```

The `tanh` before `softmax` constrains attention scores to [-1, 1], preventing any single frame from dominating. The softmax across time frames creates a soft attention distribution.

Source: `AttentionSEDHead` class in `train_sed_ns.py` (lines 85-100).

## SEDModel (full model)

```python
class SEDModel(nn.Module):
    def __init__(self, backbone, num_classes=234, in_channels=3,
                 dropout=0.1, drop_path_rate=0.0, gem_p_init=3.0):
        self.backbone = timm.create_model(backbone, pretrained=True,
                          in_chans=3, features_only=False,
                          global_pool='', num_classes=0,
                          drop_path_rate=drop_path_rate)
        self.gem_pool = GEMFreqPool(p_init=gem_p_init)
        self.head = AttentionSEDHead(backbone.num_features, num_classes, dropout)

    def forward(self, x):  # x: (B, 3, 224, T)
        return self.head(self.gem_pool(self.backbone(x)))
```

Source: `SEDModel` class in `train_sed_ns.py` (lines 103-118).

## Loss Functions

### FocalBCE

Binary cross-entropy with focal modulation to down-weight easy negatives:

```
FocalBCE = (1 - pt)^gamma * BCE(logits, targets)
```

Default `gamma=2.0`. Used for all NS training and NC Phase 1-3.

### NCDistillLoss

Noisy Classmate Phase 4 loss combining hard and soft targets:

```
L = (1 - beta) * FocalBCE(logits, hard_targets)
  + beta * KLD(sigmoid(logits/T), soft_targets) * T^2
```

Default `beta=0.3`, `T=2.0`. Supports per-sample disagreement weights.

Source: `FocalBCE` and `NCDistillLoss` classes in `train_sed_ns.py` (lines 422-484).

## EMA (Exponential Moving Average)

Model weights are tracked with EMA (decay=0.999). The EMA checkpoint is inherited across NS/NC rounds for warm-starting.

```python
class ModelEMA:
    def update(self, model):
        shadow[k] = decay * shadow[k] + (1 - decay) * model[k]
```

Source: `ModelEMA` class in `train_sed_ns.py` (lines 54-69).

## Related Pages

- [[backbones]] -- EfficientNet-B0 vs PVT-v2-B0 comparison
- [[augmentation]] -- Data augmentation pipeline
- [[training-configs]] -- Full config reference
- [[onnx-export]] -- How the model is exported for inference
