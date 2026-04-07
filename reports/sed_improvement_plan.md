# SED Improvement Plan: From LB 0.862 to LB 0.88+

**Date**: 2026-03-16
**Current baseline**: `sed-b0-v5` (EfficientNet-B0, dual clip+frame BCE, holdout TBD)
**Reference**: Competitor SED holdout=0.9883, LB=0.862
**Goal**: Exceed competitor on holdout AUC, target LB ≥ 0.875

---

## Summary of Key Findings from BirdCLEF 2024–2025 Top Solutions

The research across 11 top-5 solutions reveals a clear hierarchy of impact:

| Rank | Technique | Expected Gain | Confirmed By |
|------|-----------|--------------|--------------|
| 1 | Multi-round noisy student pseudo-labeling (soundscapes) | +0.04–0.06 LB | 2025 1st (×4 rounds), 2nd (×2 rounds) |
| 2 | Backbone upgrade (EfficientNetV2-S-in21k, ECA-NFNet-L0) | +0.01–0.03 | 2025 1st/2nd/5th |
| 3 | CrossEntropyLoss (softmax train, sigmoid infer) | +0.044 over BCE | 2024 1st |
| 4 | 10-second context windows at inference | +0.015 | 2024 1st |
| 5 | TTA: 2.5s temporal shifts (left + right) | +0.012 | 2025 2nd |
| 6 | Temporal smoothing across adjacent predictions | +0.010 | 2024 3rd |
| 7 | SpecAugment / XY masking + CutMix | +0.010–0.013 | 2024 1st, 2025 1st |
| 8 | Xeno Archive pretraining | +0.03 baseline | 2025 1st |
| 9 | Square-root class frequency balancing | marginal | 2025 1st |
| 10 | min() ensemble instead of mean() | meaningful | 2024 1st |

---

## Phase 1: Quick Wins — Inference Improvements (No Retraining)

These apply to the **already-trained** `sed-b0-v5` checkpoint and take < 1 hour to implement.

### 1.1 TTA: 2.5s Temporal Shifts
At inference, run each 5-second clip 3 times (no shift, −2.5s, +2.5s) and average:
```python
shifts = [0, -CLIP_SAMPLES//2, +CLIP_SAMPLES//2]
preds = [model(get_clip(audio, start + shift)) for shift in shifts]
final = np.mean(preds, axis=0)
```
**Expected**: +0.012 LB (confirmed by 2025 2nd place)

### 1.2 Temporal Smoothing Across Clips
For a 60-second soundscape (12 clips), apply kernel `[0.1, 0.2, 0.4, 0.2, 0.1]` across adjacent predictions:
```python
kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
smoothed = np.convolve(clip_preds, kernel, mode='same')
```
**Expected**: +0.010 LB (confirmed by 2024 3rd place)

### 1.3 10-Second Context Windows at Inference
Instead of predicting each 5s clip independently, concatenate with adjacent 5s clip (10s total), then pool:
```python
chunk_10s = audio[start - CLIP_SAMPLES//2 : start + CLIP_SAMPLES + CLIP_SAMPLES//2]
# Run model on 10s, take clip-wise attention output for target 5s
```
**Expected**: +0.015 LB (confirmed by 2024 1st place)

---

## Phase 2: Training Improvements — sed-b0-v6

### 2.1 CrossEntropyLoss Instead of BCE

The 2024 1st place winner found CE **+0.044 over BCE** on a similar EfficientNet-B0 SED baseline. The key difference:
- **Training**: softmax over all 234 classes (single-label assumption per clip)
- **Inference**: sigmoid independently per class
- Rationale: forces model to commit to one species per time frame, sharper gradients

```python
# Training: CE with label smoothing
loss = F.cross_entropy(logits, hard_labels, label_smoothing=0.05)

# Inference: still sigmoid for multi-label output
probs = torch.sigmoid(logits)
```

**Note**: requires treating each clip as approximately single-label during training. Works because most 5s clips in BirdCLEF are dominated by one species even if multi-labeled.

### 2.2 SpecAugment / XY Masking

Add frequency masking + time masking directly on the mel spectrogram:
```python
# After mel transform, before model forward
mel = torchaudio.transforms.FrequencyMasking(freq_mask_param=30)(mel)
mel = torchaudio.transforms.TimeMasking(time_mask_param=40)(mel)
```
Combined with CutMix: **+0.013 LB** in 2024. Our current config has `time_masking: true` but no frequency masking — add freq masking.

### 2.3 Square-Root Class Frequency Balancing

For sampling weights during training, weight rare species more:
```python
class_counts = train_df['primary_label'].value_counts()
weights = (class_counts / class_counts.sum()) ** (-0.5)
sample_weights = train_df['primary_label'].map(weights)
```
Especially important for 234-species setup where some species have < 10 clips.

### 2.4 Longer Training + Cosine Restart

Current: 20 epochs, cosine LR.
Recommended: 30–40 epochs with 1-cycle cosine or cosine restarts.
2025 1st place used extended training with StochasticDepth regularization.

**Config changes for sed-b0-v6**:
```yaml
training:
  epochs: 30
  loss: ce              # CrossEntropy instead of BCE
  label_smoothing: 0.05
  mixup_alpha: 0.5
augmentation:
  freq_masking: true
  freq_mask_param: 30
  time_masking: true
  time_mask_n: 2
  time_mask_ratio: 0.15
```

---

## Phase 3: Backbone Upgrade — sed-b2-v1

### 3.1 EfficientNet-B2 or EfficientNetV2-S (in21k)

| Backbone | Params | Top-5 usage | Notes |
|----------|--------|-------------|-------|
| `tf_efficientnet_b0.ns_jft_in1k` | 5.3M | Our current | Good baseline |
| `tf_efficientnet_b2.ns_jft_in1k` | 9.1M | 2025 1st/3rd | Better capacity, manageable size |
| `tf_efficientnetv2_s.in21k` | 24M | 2025 2nd | Best accuracy, slower |
| `eca_nfnet_l0` | 24M | 2025 2nd | Top performer, harder to tune |

**Recommended upgrade path**: `tf_efficientnet_b2.ns_jft_in1k` — same family as B0, 2× better capacity, noisy student pretraining, CPU-feasible for submission.

**Key requirement**: Must fit in Kaggle 2×CPU inference budget (≤ 9 hours for ~800 soundscapes).

### 3.2 In21k Pretraining

The `_in21k` suffix (ImageNet-21k pretrained) consistently outperforms standard IN1k in all 2024–2025 analyses. Use `tf_efficientnetv2_s_in21k` or `tf_efficientnet_b2.ns_jft_in1k`.

---

## Phase 4: Multi-Round Pseudo-Labeling (Highest Total Impact)

This is the **#1 technique** across all top solutions, delivering +0.04–0.06 LB gain. Our current pipeline has round-1 pseudo labels for Perch only. We need SED-based pseudo labels.

### 4.1 Round 2 SED Pseudo-Labels

After sed-b0-v5 or sed-b0-v6 converges:

1. **Run SED on all 66 training soundscapes** (full files, not just val split)
2. **Filter labels**:
   - max probability per clip ≥ 0.4
   - secondary class must be < 0.1 to avoid confusion
   - consensus required from ≥ 2 models (Perch ensemble + SED)
3. **Add pseudo-labeled soundscape clips to training set** with weight coefficient 0.5–1.0
4. **Retrain** from scratch or fine-tune

```python
# Pseudo-label selection criteria (from 2025 2nd place)
mask = (preds.max(axis=1) >= 0.4) & \
       (np.sort(preds, axis=1)[:, -2] < 0.1)  # margin condition
```

### 4.2 Multi-Model Agreement Filtering

Use our Perch ensemble (holdout=0.9780) + SED to cross-validate pseudo labels:
- Only accept labels where **both Perch ensemble AND SED agree** (IoU > 0.5 on top-3 predicted species)
- Disagreement = ambiguous clip, discard from pseudo training

### 4.3 Iterative Rounds

```
Round 1 (done): Perch pseudo labels on individual recordings
Round 2 (next): SED pseudo labels on 66 training soundscapes
Round 3:        Retrain SED on original + round-2 pseudo → pseudo-label again
Round 4:        Final model with all pseudo-labeled data
```
2025 1st place: 0.872 → 0.930 over 4 rounds.

---

## Phase 5: Ensemble Strategy Upgrade

### 5.1 min() Instead of mean() for CE-Trained Models

2024 1st place found `min()` over 6 folds outperformed `mean()`:
- `min()` penalizes uncertain predictions more aggressively
- Particularly effective when CE loss trains models to be "committed"

```python
# Instead of ensemble mean:
final_pred = np.stack([pred1, pred2, pred3]).min(axis=0)
```

### 5.2 Optimal Ensemble Weights

Based on current holdout AUCs (to be optimized after SED completes):
```python
# Starting point weights (tune on holdout)
W_PSEUDO     = 1.0  # holdout=0.9453
W_SOUNDSCAPE = 1.0  # holdout=0.9550
W_EMBEDDING  = 1.0  # holdout=0.9537
W_SED        = 2.0  # SED is primary — weight it higher
```

---

## Implementation Priority & Timeline

| Priority | Task | Estimated LB gain | GPU | Time |
|----------|------|------------------|-----|------|
| 🔴 NOW | Wait for sed-b0-v5 holdout AUC | — | — | ~4h |
| 🔴 HIGH | Add TTA + temporal smoothing to inference_sed.py | +0.012–0.025 | CPU | 2h |
| 🟡 NEXT | Train sed-b0-v6: CE loss + freq masking + 30ep | +0.02–0.05 | GPU1 | 8h |
| 🟡 NEXT | Generate round-2 SED pseudo labels on soundscapes | +0.01–0.02 | GPU0 | 2h |
| 🟠 LATER | Upgrade to EfficientNet-B2 (sed-b2-v1) | +0.01–0.02 | GPU1 | 12h |
| 🟠 LATER | Multi-round noisy student (rounds 2–4) | +0.03–0.05 | Both | days |

---

## Critical Caveat: CE Loss vs BCE

The 2024 finding (CE >> BCE by +0.044) is counter-intuitive for a multi-label task. However:

1. Most 5s clips are **effectively single-label** (one dominant species)
2. CE forces sharper, more confident predictions — better calibration
3. The "+0.044" was for the specific case of "softmax training + sigmoid inference" with hard pseudo-labels

**Risk**: if our 234-species task has more genuine multi-label clips than 2024's task, CE may hurt. **Test on holdout first** before committing to sed-b0-v6 with CE.

---

## Recommended Next Run Config: sed-b0-v6

```yaml
experiment:
  name: sed-b0-v6

model:
  backbone: tf_efficientnet_b0.ns_jft_in1k  # same backbone, different training
  dropout: 0.1
  in_chans: 3
  use_gem: true
  gem_p_init: 3.0

training:
  epochs: 30
  batch_size: 32
  optimizer: adamw
  learning_rate: 0.0005
  weight_decay: 0.0001
  scheduler: cosine
  warmup_epochs: 3
  loss: ce                  # CrossEntropy (key change from v5)
  label_smoothing: 0.05
  clip_loss_weight: 0.5
  frame_loss_weight: 0.5
  mixup_alpha: 0.5
  use_soundscapes_in_train: true
  soundscape_val_frac: 0.2
  soundscape_oversample: 5

augmentation:
  enabled: true
  time_masking: true
  time_mask_ratio: 0.15
  time_mask_n: 2
  freq_masking: true        # NEW: frequency masking
  freq_mask_param: 30       # NEW
  noise_level: 0.005
  gain_range: [0.7, 1.3]
```

---

## References

- BirdCLEF 2025 1st Place: Nikita Babych — Multi-Iterative Noisy Student
- BirdCLEF 2025 2nd Place: VSydorskyy — NFNet + EfficientNetV2 + 2-round pseudo-labeling
- BirdCLEF 2024 1st Place: CE loss + 10s context + min() ensemble
- BirdCLEF 2024 3rd Place: 2-level cascade pseudo-labeling + temporal smoothing
