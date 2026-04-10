---
tags: [augmentation, training, mixup]
last-updated: 2026-04-08
---

# Augmentation Pipeline

Data augmentation is applied at two levels: waveform (before mel) and spectrogram (after mel). The design follows BirdCLEF 2025 1st-place insights.

## Augmentation Flow

```
Raw waveform (20s, 640K samples)
  -> absmax normalization
  -> [Wave-level MixUp] (labeled x pseudo, lambda=0.5)
  -> MelTransform (GPU)
  -> [SpecAugment] (freq + time masking)
  -> [SumixFreq] (per-frequency-bin mixing)
  -> Backbone
```

## Wave-Level Augmentations

### Absmax Normalization

Every audio clip is normalized by its absolute maximum before any mixing:

```python
def absmax_normalize(audio):
    m = np.abs(audio).max()
    return audio / (m + 1e-8) if m > 1e-8 else audio
```

This ensures consistent amplitude scale across clips from different sources (train_audio vs soundscapes). From BirdCLEF 2025 1st place.

### Audio MixUp

Fixed lambda=0.5 mixing between two randomly paired samples:

```python
mixed_x = 0.5 * x + 0.5 * x[idx]
mixed_y = max(y, y[idx])  # union of species labels
```

Key insight from 1st place: variable lambda near 0 or 1 suppresses meaningful signal. Constant 0.5 ensures both clips always contribute equally. Labels take the union (element-wise max) because all species from both clips are present in the mix.

In practice, labeled train_audio clips and pseudo-labeled soundscape clips are concatenated into one batch, then MixUp is applied across the combined batch. This creates cross-domain mixing (labeled x pseudo), which is critical for bridging the domain gap.

Source: `audio_mixup()` in `train_sed_ns.py` (lines 202-213).

## Mel-Level Augmentations

### SpecAugment

Standard SpecAugment with frequency and time masking applied to the mel spectrogram:

| Parameter | Value | Notes |
|-----------|-------|-------|
| freq_mask_param | 24 | Max frequency bins to mask |
| time_mask_param | 64 | Wider than standard (32) for 20s clips |
| n_freq_masks | 2 | Number of frequency masks |
| n_time_masks | 2 | Number of time masks |

SpecAugment prevents the model from relying on specific frequency bands or time regions, improving generalization.

Source: `SpecAug` class in `train_sed_ns.py` (lines 153-170).

### SumixFreq

Per-frequency-bin random selection between two samples. From BirdCLEF 2025 1st place.

```python
def sumix_freq(mel, labels):
    idx = torch.randperm(B)
    mask = (torch.rand(n_mels) > 0.5).view(1, 1, -1, 1)  # per-freq binary mask
    mixed = torch.where(mask, mel[idx], mel)
    return mixed, torch.max(labels, labels[idx])
```

**Rationale**: Different species occupy different frequency bands. Mixing at the frequency bin level (each bin comes entirely from one recording) creates more realistic multi-species spectrograms than waveform mixing, because:
- Low-frequency amphibian calls mixed with high-frequency bird calls
- Each frequency bin has coherent signal (no destructive interference)
- Labels = union of both clips' species

Enabled via `use_sumix_freq: true` in config. Applied AFTER mel computation, AFTER SpecAugment.

Source: `sumix_freq()` in `train_sed_ns.py` (lines 181-199).

## Augmentation Interaction

The augmentations are applied in sequence, and their interaction matters:

1. **MixUp first (wave-level)**: Mixes raw audio, creating blended waveforms. Both species' acoustic signatures are present.
2. **Mel transform**: Converts mixed waveform to spectrogram.
3. **SpecAugment**: Masks random regions, preventing overfitting to specific patterns.
4. **SumixFreq**: Further mixes at the frequency level, creating even more diverse multi-species spectrograms.

This layered approach provides strong regularization while maintaining realistic acoustic characteristics.

## Per-Domain Strategy

| Domain | Augmentation | Purpose |
|--------|-------------|---------|
| train_audio (labeled) | MixUp with pseudo, SpecAugment, SumixFreq | Domain bridging, regularization |
| Soundscapes (pseudo) | Same + WeightedRandomSampler by confidence | Focus on high-quality pseudo labels |
| Soundscapes (val) | None | Clean evaluation |

## WeightedRandomSampler

Pseudo soundscape samples are weighted by total confidence (sum of per-window max class probabilities). Soundscapes with higher confidence pseudo labels are sampled more frequently, giving preference to high-quality pseudo labels. From BirdCLEF 2025 1st place.

## Related Pages

- [[architecture-sed-model]] -- Where augmentation fits in the model pipeline
- [[training-configs]] -- Augmentation parameter settings
- [[data]] -- Training data sources and domain gap
