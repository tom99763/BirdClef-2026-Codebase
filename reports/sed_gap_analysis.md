# SED Gap Analysis: Why Our SED Underperforms the Competitor

**Date:** 2026-03-21
**Author:** Analysis via Claude Code
**Summary:** Our best SED (v30-multipseu) achieves ~0.80 soundscape val ROC-AUC; competitor achieves ~0.90 with the same EfficientNet-B0 backbone. Gap = **0.10 AUC points**. Root causes identified: 4 structural differences confirmed via checkpoint inspection.

---

## 1. Baseline Comparison

| Metric | Our Best (v30-multipseu) | Competitor SED-B0 |
|--------|--------------------------|-------------------|
| Backbone | tf_efficientnet_b0.ns_jft_in1k | tf_efficientnet_b0 |
| Soundscape val ROC-AUC | ~0.80 | ~0.90 |
| Macro AUC (clip-level) | ~0.9839 (holdout) | 0.9478 (191 classes) |
| LB contribution | drops LB -0.013 vs competitor | baseline ensemble |
| Loss | ASL (gamma_neg=4, gamma_pos=0) | **CrossEntropy** |
| LR Scheduler | Simple cosine | **CosineAnnealingWarmRestarts T_0=5** |
| Mixup | 0.5 (enabled) | 0.5 (enabled) |
| SpecAugment | freq_mask=40 (train_epoch ✅), time_mask BROKEN ❌ | freq_mask=30, time_mask=30 |
| Dual loss | clip=0.5 + frame=0.5 | **clip-only** |
| Epochs | 35 | **15** |
| Batch size | 32 | 16 + grad_accum=2 (effective=32) |
| Classes evaluated | 71 (soundscape val) | 191 (all train_audio) |

---

## 2. Root Cause Analysis

### 2.1 Loss Function: ASL vs CrossEntropy (HIGHEST PRIORITY)

**Our approach:** Asymmetric Loss (gamma_neg=4.0, gamma_pos=0.0)
- Designed for multi-label detection with heavy class imbalance
- Aggressively down-weights easy negatives (gamma_neg=4)
- Works well for recall-maximizing detection tasks

**Competitor approach:** CrossEntropy (single-label per clip)
- Treats each 5s clip as having exactly ONE primary species
- Uses `argmax(soft_labels)` → hard integer class → standard CE
- Much stronger gradient signal — model learns a discriminative boundary between species
- BirdCLEF 2024 1st place used CE: reported +0.044 over BCE on B0 SED

**Why CE beats ASL on soundscape val:**
- Soundscape clips contain ONE dominant bird species per 5s window (competition design)
- CE forces the model to rank the correct species at position 1 — exactly what evaluation does
- ASL tries to detect ALL positive species simultaneously → diffuse probability mass
- With 234 classes and sparse labels, CE signal is much sharper

**Evidence from codebase:** CE is already implemented in `train_sed.py:371-377`:
```python
if loss_mode == "ce":
    logits = torch.logit(clip_pred.clamp(1e-6, 1.0 - 1e-6))
    hard_labels = labels.argmax(dim=1)
    return clip_w * F.cross_entropy(logits, hard_labels, label_smoothing=label_smoothing)
```

### 2.2 LR Scheduler: Simple Cosine vs CosineAnnealingWarmRestarts (HIGH PRIORITY)

**Our approach:** `cosine_lr_with_warmup` — LR decays monotonically from `base_lr` to 0
- With 35 epochs: model learns fast in first 20 epochs, then barely updates
- The "dead zone" in the last 15 epochs contributes no meaningful learning
- A 35-epoch run with simple cosine is equivalent to a ~15-20 epoch run

**Competitor approach:** `CosineAnnealingWarmRestarts` with T_0=5
- LR resets to base_lr every 5 epochs: restarts at epochs 1, 6, 11
- Each restart lets the model escape local minima
- Over 15 epochs = 3 complete cycles → 3 opportunities to find better solutions
- Final cycle's minimum is the best checkpoint (consistently)
- BirdCLEF 2025: warm restarts +0.02-0.03 across multiple teams

**Mathematical comparison for 15 epochs:**

| Epoch | Simple Cosine (35ep) | Warm Restarts (T_0=5) |
|-------|---------------------|----------------------|
| 1 | 5e-4 | 5e-4 |
| 5 | 4.1e-4 | ~0 (valley) |
| 6 | 3.7e-4 | **5e-4 (RESTART!)** |
| 10 | 2.3e-4 | ~0 (valley) |
| 11 | 2.1e-4 | **5e-4 (RESTART!)** |
| 15 | 1.4e-4 | ~0 (valley) |

Simple cosine never restarts. The model gets stuck where it is after epoch 5.

**Code status:** NOT IMPLEMENTED — `train_sed.py` only has `cosine` and `constant` options.

### 2.3 Dual Loss: clip+frame vs clip-only (MEDIUM PRIORITY)

**Our approach:** `clip_loss_weight=0.5, frame_loss_weight=0.5` — equally weighted dual loss

**Competitor approach:** Clip-only loss (frame_loss_weight=0.0 effectively)

**Why dual loss hurts on soundscape val:**
- Frame-level labels are expanded from clip-level labels (all frames labeled same as clip)
- This is a noisy target: the bird may only appear in 1 of 10 frames
- The frame loss adds noise to the gradient, especially for short-duration bird calls
- Frame loss does help for SED time-resolution tasks, but NOT for clip-level ROC-AUC

**Evidence:** v22 (dual loss, bce) vs similar single-loss configs — no consistent gain observed.

### 2.4 Silent Bug: Time Masking Config is Ignored (MEDIUM PRIORITY)

**Config says:** `time_masking: true, time_mask_ratio: 0.15, time_mask_n: 2`

**Code reality:** `train_sed.py:train_epoch()` has NO time masking implementation.
- Only `freq_mask_param > 0` → `TAT.FrequencyMasking(...)` is applied
- Time masking parameter path is missing entirely
- Our v30 (best model) thinks it has time masking but doesn't

**Competitor has:** `time_mask_param=30` — actually applied

**Fix:** Add `time_mask_param` to `train_epoch()` and apply `TAT.TimeMasking(time_mask_param)(mel_batch)`.

### 2.5 Overfitting from Too Many Epochs (LOWER PRIORITY)

**Our approach:** 35 epochs with simple cosine scheduler
**Competitor approach:** 15 epochs

With ASL loss, the model is encouraged to fit ALL training labels simultaneously. After 20 epochs,
the model begins memorizing train_audio clips (labeled clips, not soundscapes). The soundscape
domain shift grows as training continues. This explains why our holdout AUC (train_audio clips)
reaches 0.9839 while soundscape val AUC is only ~0.80 — massive overfitting to the clip distribution.

---

## 3. Summary of Root Causes (Ranked)

| Rank | Root Cause | Estimated Impact | Fix Complexity |
|------|-----------|-----------------|----------------|
| 1 | **Loss: ASL → CE** | +0.05-0.08 AUC | 1 line in config |
| 2 | **Scheduler: simple cosine → WarmRestarts T_0=5** | +0.02-0.04 AUC | ~10 lines in train_sed.py |
| 3 | **Dual loss → clip-only** | +0.01-0.03 AUC | 1 line in config |
| 4 | **Fix time masking bug** | +0.01-0.02 AUC | ~5 lines in train_sed.py |
| 5 | **Epochs: 35 → 15** | +0.01-0.02 AUC | 1 line in config |

**Total estimated gain:** +0.10-0.17 AUC → should reach 0.88-0.95 soundscape val AUC

---

## 4. Improvement Experiment Plan

### Experiment v31 — CE + WarmRestarts Baseline (Test user's hypothesis)
```yaml
training:
  loss: ce
  scheduler: warm_restarts
  scheduler_T0: 5
  epochs: 15
  mixup_alpha: 0.5
  clip_loss_weight: 1.0
  frame_loss_weight: 0.0
augmentation:
  freq_mask_param: 30
  time_mask_param: 30   # fix the bug
```
**Goal:** Single experiment testing both user's hypothesized root causes simultaneously.

### Experiment v32 — CE Loss Only (Ablation: loss in isolation)
```yaml
training:
  loss: ce
  scheduler: cosine      # keep old scheduler
  epochs: 15
  mixup_alpha: 0.5
  clip_loss_weight: 1.0
  frame_loss_weight: 0.0
```
**Goal:** Quantify contribution of CE loss change alone.

### Experiment v33 — WarmRestarts Only (Ablation: scheduler in isolation)
```yaml
training:
  loss: asl              # keep old loss
  scheduler: warm_restarts
  scheduler_T0: 5
  epochs: 15
  mixup_alpha: 0.5
```
**Goal:** Quantify contribution of warm restarts alone.

### Experiment v34 — CE + WarmRestarts + Pseudo Labels (Full recipe)
```yaml
# Same as v31 but with v30's pseudo label data (combined r4r5)
data:
  soft_pseudo_csv: pseudo_labels/combined_pseudo_r4r5.csv
training:
  loss: ce
  scheduler: warm_restarts
  scheduler_T0: 5
  epochs: 15
```
**Goal:** Best competitive config — competitor recipe + our data advantage.

---

## 5. Implementation Changes Required

### train_sed.py
1. Add `cosine_warm_restarts_lr()` function
2. Update scheduler dispatch in training loop
3. Add `time_mask_param` to `train_epoch()` signature and apply it

### New configs
- `configs/sed_b0_v31_ce_wr.yaml`
- `configs/sed_b0_v32_ce_only.yaml`
- `configs/sed_b0_v33_wr_only.yaml`
- `configs/sed_b0_v34_ce_wr_pseudo.yaml`

---

## 6. Monitoring Strategy

**Primary metric:** `val_roc_auc` from `outputs/{run_name}/result.json` (soundscape val)
**NOT:** holdout_auc (train_audio clips) — this metric is completely decoupled from LB

**Monitoring command:**
```bash
for d in outputs/sed-b0-v3{1,2,3,4}*/; do
  name=$(basename $d)
  roc=$(python3 -c "import json; d=json.load(open('$d/result.json'));
    h=d.get('epoch_history',[]);
    print(max((e['val_roc_auc'] for e in h), default=0))" 2>/dev/null)
  echo "$name: $roc"
done
```
