# SED Training Knowledge for BirdCLEF

Compiled from BirdCLEF 2022–2025 top solutions (Kaggle, GitHub, arXiv).
Last updated: 2026-03-21

---

## Sources

- BirdCLEF 2025 1st place (Nikita Babich) — multi-iterative pseudo-labeling
- BirdCLEF 2025 2nd place (VSydorskyy) — GitHub: VSydorskyy/BirdCLEF_2025_2nd_place
- BirdCLEF 2025 5th place (myso1987) — GitHub: myso1987/BirdCLEF-2025-5th-place-solution
- BirdCLEF 2024 1st place — Zenn writeup (Japanese)
- BirdCLEF 2024 2nd place — Zenn writeup
- BirdCLEF 2024 3rd place (Theo Viel / jfpuget) — GitHub: TheoViel/kaggle_birdclef2024
- BirdCLEF 2024 4th place — Zenn writeup
- BirdCLEF 2024 pseudo-label paper — arXiv:2407.06291
- BirdCLEF 2022 8th place — GitHub: ffs333/BirdCLEF_2022_8th_place
- Top 2% BirdCLEF 2025 Medium writeup

---

## 1. Loss Functions

| Solution | Loss | Notes |
|---|---|---|
| 2024 1st | **CrossEntropy** | "BCE shows significantly worse results than CE." |
| 2024 3rd | BCEWithLogitsLoss | Secondary labels masked entirely (+0.01 LB) |
| 2025 1st | CrossEntropy + AdamW | — |
| 2025 2nd | **FocalBCELoss** | Combination of Focal + BCE |
| 2025 4th | **SoftAUCLoss** | Pairwise differentiable AUC, resistant to overfitting, supports soft labels |
| 2025 5th | FocalLoss | Three training stages |
| Academic (DS@GT 2024) | ASL (γ+=1, γ-=4) | Asymmetric Loss for multi-label imbalance |

**Key findings:**
- **CE is strongest for single-primary-label setups** (one dominant species per clip)
- **BCE / Focal BCE** works when secondary labels are masked from loss
- **ASL** is theoretically best for noisy multi-label with false-positive bias (high γ- penalizes easy negatives)
- **SoftAUCLoss** directly optimizes the evaluation metric — an emerging approach
- The 2024 1st place used CE for training but sigmoid at inference — deliberate train/test mismatch that helped
- Raw BCE without Focal weight consistently underperforms CE in clean single-label setups

**For BirdCLEF 2026:**
- If primary label dominates: use CE
- If multi-label and secondary labels are reliable: use ASL or FocalBCE
- Our current ASL (γ-=4, γ+=0) is reasonable for multi-label but may be suboptimal vs CE for soundscape domain

---

## 2. Learning Rate Schedulers

**Dominant choice: CosineAnnealingLR (single cycle)**

| Solution | Scheduler | LR | Epochs |
|---|---|---|---|
| 2024 1st | CosineAnnealingLR | 1e-3 to 3e-3 | 7–12 |
| 2024 2nd | CosineAnnealingLR + 3ep warmup | 1e-3 | 50 |
| 2025 2nd (EfficientNetV2-S) | CosBatchLR (per batch) | 1e-3 | 50 |
| 2025 2nd (eca_nfnet_l0) | CosBatchLR (per batch) | 1e-4 | 50 |
| 2025 5th | Cosine + warmup | — | — |
| 2022 8th (multi-phase) | CosineAnnealingLR per phase | 1e-3 → 1.3e-3 → 8e-4 | 30 → 60 → 85 |

**Key findings:**
- **Single-cycle cosine is dominant** across all years. CosineAnnealingWarmRestarts (CAWR) is NOT prominent in top BirdCLEF solutions.
- "CosBatchLR" (2025 2nd): decay scheduled per gradient step rather than per epoch — smoother decay with large datasets.
- **3–5 epoch linear warmup** before cosine decay is standard in 2024–2025 solutions.
- **AdamW** is the most common optimizer (weight_decay=1e-4). RAdam used for very large models.
- CAWR can help for very long training (>50 ep) with pseudo-label stages, but single-cycle suffices for 15–50 ep.

**For BirdCLEF 2026:**
- Our warmup-then-cosine setup is correct
- v31's WarmRestarts (CAWR T₀=5) is NOT the competitor recipe — competitor likely uses simple cosine
- Consider switching to single-cycle CosineAnnealingLR with 3ep warmup as v32 baseline

---

## 3. Epoch Counts & Batch Sizes

| Solution | Epochs | Batch | Notes |
|---|---|---|---|
| 2024 1st | 7–12 | 96 | Hard sampling, large LR |
| 2024 2nd | 50 | 64 | Standard, model soup over epochs 13–50 |
| 2025 2nd | 50 | 64 | Standard across all folds |
| 2025 5th | Multi-stage | — | 3 pseudo-label stages |
| 2022 8th | 85 cumulative | — | 3 phases with increasing epochs |

**Key findings:**
- **30–50 epochs is the mainstream range** for single-stage training on B0/B2-scale models
- Very short (7–12 ep) works with aggressive LR and large datasets
- Our current 15–35 epoch configs may be too short for soundscape generalization
- With pseudo-label stages, cumulative effective epochs = 85+
- **Model soup (averaging checkpoints ep13–50)** is more stable than early stopping

**For BirdCLEF 2026:**
- Consider 30–50 epoch training for B0 (currently using 15–35)
- Soundscape val AUC often peaks later than train_audio val AUC

---

## 4. Augmentation Strategies

### What works:

**Waveform-level:**
- Gaussian noise (amplitude 0.001–0.011, p=0.4)
- Time shift (±50%, p=0.1) — one of few augmentations that consistently helps
- Background noise injection from unlabeled soundscapes
- Gain adjustment (±12dB, p=0.2)

**Spectrogram-level:**
- **Horizontal CutMix** — called "most impactful" by 2024 1st place
- FrequencyMasking(24 bins) — mask up to 24 of 128 mel bins (~19%)
- TimeMasking(96 steps) — mask up to 96 of ~313 steps (~31%)

**Mixing:**
- **Additive mixup** (2024 3rd): sum audio waveforms rather than interpolate, labels = max(label_A, label_B). Better simulates real soundscapes with overlapping species.
- Standard mixup alpha=0.4 early → taper to 0.07 later (2022 8th). Sustained high mixup hurts convergence.
- CutMix with horizontal cuts in spectrogram domain.

### What does NOT work (negative results):
- **Heavy noise injection** on focal recordings (2024 1st)
- **1D audio augmentations** (waveform blur, pixdrop) — neutral to negative
- **Heavy SpecAugment** often hurts soundscape AUC because test clips already have natural masking
- PitchShift and time warp — marginal at best

**For BirdCLEF 2026:**
- Our current: noise_level=0.005, gain[0.7,1.3], time_mask=30, freq_mask=30
- Freq mask 30/224 ≈ 13% — lighter than top solutions (24/128 ≈ 19%)
- Missing: background soundscape noise injection, horizontal CutMix
- Consider additive mixup instead of interpolative mixup

---

## 5. Secondary Labels Handling

| Solution | Approach | Impact |
|---|---|---|
| 2024 1st | primary=0.5, each secondary=0.5/N | — |
| 2024 3rd | **Mask entirely from loss** | **+0.01 LB** |
| 2025 2nd | "AddRareBirdsNoLeak" rare-class strategy | — |
| 2022 8th | weight threshold tested: 0.3, 0.4, 0.5 | — |

**Key finding:** Masking secondary labels entirely is the strongest single finding for label quality — +0.01 LB from dropping unreliable annotations. Secondary labels in Xeno-Canto are incidental background species, often uncertain.

**For BirdCLEF 2026:**
- Our current `secondary_label_weight=1.0` is likely suboptimal
- Consider testing `secondary_label_weight=0.0` (mask) or 0.3

---

## 6. Rating Filtering

- General: train on **rating ≥ 3** to remove noisiest recordings
- 2024 1st: Used BirdNET confidence filter (signal activity: top 0.8 quantile of std+var+rms+pwr)
- 2024 3rd: Cap per species at **500 records** (keep most recent). Floor at **10 samples** (upsample rare classes)
- 2025 5th: For species <30 samples, manually curated recordings

**For BirdCLEF 2026:**
- Our `min_rating=0.0` (no filtering) may include noisy low-rating recordings
- Try `min_rating=3.0` or `min_rating=2.0` as an experiment

---

## 7. Validation Set Construction

**Critical insight: Internal CV from train_audio does NOT reliably predict soundscape LB.**

| Solution | Val Approach |
|---|---|
| 2024 1st | Standard k-fold on train_audio |
| 2024 3rd | No reliable val set — relied on LB + trend analysis |
| 2025 1st | **No local val — validated directly on public LB** |
| 2025 2nd | 5-fold stratified CV |
| 2025 3rd | **Reserved 20% of 2023 soundscape data as holdout** |

**Best practice for soundscape val:**
- Use prior-year labeled soundscapes (BirdCLEF 2023/2024) as held-out validation
- Supplement with synthetic soundscapes (mix focal recordings with unlabeled soundscape backgrounds)
- 2024 4th place built test-like synthetic val by mixing Xeno-Canto bird calls with soundscape noise — correlated better with LB than standard CV

**For BirdCLEF 2026:**
- Our current val = last 20% of train_soundscapes sorted alphabetically (13 files, 312 clips)
- This is the right approach for local monitoring
- Soundscape val AUC = 0.6397 (v31 ep3) vs competitor = 0.9918 on same set → 0.35 gap

---

## 8. Pseudo-Label Strategies

**Near-universal technique — largest single source of LB improvement.**

### Core procedure:
1. Train model on labeled train_audio only
2. Inference on unlabeled soundscape clips (5-second chunks)
3. Filter by confidence threshold
4. Add as pseudo-labeled training data (soft or hard labels)
5. Retrain. Repeat 2–4 rounds.

### Detailed configurations:

**2025 1st place (best documented):**
- 4 rounds with **power scaling** on pseudo-labels (raise probs to power 0.5–0.8 for softer labels)
- Round progression: 0.872 → 0.898 → 0.930 → 0.933 LB
- Multi-iterative noisy student approach

**2025 2nd place:**
- Filter: F2-score threshold 0.5, min threshold 0.1, positive fraction 0.4, 3 iterations
- Square-root class balancing applied to pseudo-labels

**2024 3rd place:**
- Batch: 128 real + 128–192 pseudo per batch (50% real / 50% pseudo)
- Labels: `target = max(original_label, pseudo_label)`
- Found: pseudo-labeling on **soundscapes** helps; pseudo-labeling on train_audio hurts (acts as label smoothing)

**2025 5th place:**
- 3 stages: full labeled → 50% labeled + 50% pseudo × 2

### Key parameters:
- **Include probability**: add pseudo-labeled samples at 25–50% probability per batch (not always)
- **Power transform**: raise pseudo probabilities to power < 1 (softer) for round 1, sharpen in later rounds
- **Middle-5s selection**: use center 5 seconds of soundscape clips (avoids silent edges) — outperforms random selection

**For BirdCLEF 2026:**
- We have pseudo labels (rounds 1–5) already generated
- Current SED training uses them but competitor's SED does NOT use pseudo labels
- The 0.9918 competitor soundscape val is WITHOUT pseudo labels → more impressive
- Our SED + pseudo = lower val AUC → pseudo labels may hurt SED's soundscape generalization

---

## 9. Multi-Clip Inference Tricks

| Solution | Technique | Impact |
|---|---|---|
| 2024 1st | 10s input with 5s center + 2.5s context neighbors | +LB |
| 2024 1st | **Min() ensemble reduction** (vs mean) | Reduces false positives |
| 2024 2nd | Sum adjacent windows × 0.5 + TTA (2.5s shifts) | — |
| 2024 3rd | Convolution smoothing [0.1, 0.2, 0.4, 0.2, 0.1] on adjacent clips | — |
| 2025 2nd | TTA: ±2.5s shifts | **+0.012 LB** |
| 2025 3rd | Power adjust: boost top-N confident predictions | — |

**Key TTA insight:** Shifting inference window by 2.5 seconds (half a clip) and averaging with original gives +0.012 LB improvement (2025 2nd place). This is high-ROI.

**Min() ensemble reduction (2024 1st):** More conservative than mean — takes lowest confidence prediction across ensemble members. Reduces false positives in soundscapes.

**For BirdCLEF 2026:**
- Our TRICK2 (confidence-sharpened smoothing [0.05, 0.15, 0.60, 0.15, 0.05]) is similar to 2024 3rd's kernel
- Add ±2.5s TTA at inference (+0.012 LB worth attempting)

---

## 10. Model Soup / Checkpoint Averaging

**2024 2nd place (best documented):**
- Averaged weights from epochs 13–50 if each epoch improved any of (LRAP, cMAP, F1, AUC)
- "Yielded more stable and sometimes better LB scores" than early stopping

**2025 3rd place:**
- Model Soup explicitly cited: "averaging checkpoint weights for stability without heavy ensembles"

**For BirdCLEF 2026:**
- We already do model soup (averaging top-k checkpoints)
- Ensure soup includes epochs AFTER convergence plateau, not just best single epoch

---

## 11. Architecture Choices

**Dominant: EfficientNet-B0 (tf_efficientnet_b0_ns)**
- Fastest, lowest memory, strong baseline
- Used by most solutions as at least one ensemble member

**2024 1st place surprising finding:**
> "SED and complex architectures from previous years work slower and provide **worse** results than pure backbones."
> Used plain CNN classifier WITHOUT attention SED head.

**Attention SED head:** Primarily useful for generating frame-level pseudo-labels for soundscape pseudo-labeling pipeline. For direct clip-level classification, may be unnecessary overhead.

**Mel spectrogram:**
- 128 mel bins, n_fft=2048, hop_length=512–627
- 224×224 for ImageNet-pretrained model compatibility (our setup)
- Some teams use 64/128 n_mels with smaller hop for temporal resolution

---

## 12. SED Architecture — Critical Finding

**2024 1st place explicitly found SED WORSE than plain backbone:**
- SED (attention head + frame-level pooling) is more complex but not always better for clip-level prediction
- Plain CNN backbone with global pooling can outperform SED for soundscape classification

**Why this matters for BirdCLEF 2026:**
- Our 0.35 gap vs competitor may not be about loss/scheduler at all
- Competitor's 0.9918 soundscape val AUC is 13 epochs into B0 training
- Our SED-based training (with attention head) may be over-engineering the problem
- **Action item:** Test plain backbone (no SED head) with CE loss as ablation

---

## 13. Domain Adaptation: train_audio → Soundscapes

**The fundamental problem:** Model trained on focal recordings (one bird, close mic) must generalize to ambient soundscapes (multiple birds, far mic, background noise).

### Most effective approaches:
1. **Pseudo-labeling unlabeled soundscapes** (universal — used by all top solutions)
2. **Background soundscape noise injection** during focal recording training
3. **Including labeled soundscapes in training** (our approach: `use_soundscapes_in_train=True`)
4. **Matching clip lengths:** Train on 5s = test 5s windows
5. **Signal activity filtering:** Remove silent/background-only clips from pseudo-labeled soundscapes
6. **Human voice removal (Silero VAD):** Stop model from associating human speech with bird activity

### What does NOT help:
- Training on soundscape data alone (without focal recordings) — worse results
- High-coefficient pseudo-labeling on train_audio (acts as label smoothing)

---

## 14. Cross-Solution Impact Summary

| Technique | Observed LB Impact |
|---|---|
| Pseudo-labeling unlabeled soundscapes (1 round) | +0.02 to +0.04 |
| Multi-round pseudo-labeling (4 rounds) | +0.06 cumulative |
| Masking secondary labels from loss | +0.01 |
| TTA with ±2.5s shifts | +0.012 |
| Model soup (checkpoint averaging) | +0.005 to +0.01 |
| Horizontal CutMix augmentation | "Most impactful" augmentation |
| SED head vs plain backbone | Negative or neutral |
| Heavy SpecAugment | Neutral to negative for soundscapes |
| Additive (not interpolative) mixup | Preferred for multi-label |

---

## 15. Best-Practice Recipe (Consensus from 2024–2025 Top Solutions)

1. **Loss:** CE if single-primary-label dominant; FocalBCE or ASL if true multi-label
2. **Scheduler:** Single-cycle CosineAnnealingLR, 30–50 epochs, peak LR 1e-3, 3–5ep warmup, AdamW wd=1e-4
3. **Augmentation:** Horizontal CutMix (most impactful), FreqMask(24), TimeMask(96), background noise injection, mixup alpha=0.4 → 0.07
4. **Secondary labels:** Mask entirely OR weight ≤ 0.3
5. **Rating filter:** min_rating ≥ 3
6. **Pseudo labels:** 2–4 rounds on unlabeled soundscapes. 25–50% probability per batch. max() merge with original labels. Power scaling (0.5–0.8) on pseudo probs.
7. **Val set:** Prior-year labeled soundscapes (most reliable). Supplement with synthetic soundscape mixing.
8. **Inference:** ±2.5s shift TTA. Temporal smoothing [0.1,0.2,0.4,0.2,0.1]. Sigmoid even if trained with softmax.
9. **Model soup:** Average checkpoints from epoch 13 onward if each improves any metric.
10. **Ensemble:** ≥3 diverse architectures (B0 + B2/B4 + EfficientNetV2-S or RegNetY). Architecture diversity > fold diversity.

---

## 16. Specific Issues in Our Current Setup (BirdCLEF 2026)

| Our Setting | Top Solution Consensus | Recommendation |
|---|---|---|
| Loss: ASL (γ-=4, γ+=0) | CE (single label) or FocalBCE | Test CE with secondary=0 |
| Scheduler: CAWR T₀=5 | Single-cycle cosine | Switch to CosineAnnealingLR |
| secondary_label_weight=1.0 | Mask or ≤ 0.3 | Test 0.0 or 0.3 |
| epochs=15 | 30–50 for B0 | Increase to 30–50 |
| min_rating=0.0 | ≥ 3 | Test min_rating=3 |
| Mixup alpha=0.5 constant | 0.4 → 0.07 taper | Add LR-based taper |
| SED attention head | Plain backbone often better | Test no-SED-head ablation |
| Pseudo labels in SED | Hurts Perch diversity | Correctly excluded already |
| Soundscape val 13 files | Prior-year soundscapes better | Keep, but note limitation |
| No CutMix | "Most impactful" aug | Add horizontal CutMix |
| No bg soundscape noise | Used by multiple solutions | Add bg noise injection |
