# BirdCLEF 2025 Top Solutions Summary

## Competition Context
- Task: identify species from 5-sec audio clips (birds/amphibians/mammals/insects)
- Data: 12GB "dirty" crowdsourced xeno-canto + iNaturalist
- Constraint: 90-min CPU inference, single CPU
- Domain: Middle Magdalena Valley, Colombia (similar to 2026: Pantanal, Brazil)

---

## Top-5 Solutions

### 1st Place — Nikita Babych: "Multi-Iterative Noisy Student Is All You Need"
**Final score: 0.933**

Key techniques:
1. **Multi-round pseudo labeling (4 rounds)**: 0.872 → 0.930 (+0.058 total)
   - Each round: train model → generate pseudo labels → retrain with pseudo
   - Power scaling: predictions^gamma where gamma < 1 (sharpen soft labels)
2. **SED models for frame-level boundaries** (not just clip-level)
3. **Added external data**: 5,489 Xeno Archive entries + 17,197 insect/amphibian samples
4. **Separate pipeline for insects/amphibians** (different acoustic properties → texture taxa)
5. **EfficientNet ensembles** + SED combination

**Key lesson**: Each pseudo-labeling round is worth ~+0.014 LB. 4 rounds compound.

### 2nd Place
Key techniques:
1. **Two-round pseudo labeling**: +0.04 LB total
2. **TTA with 2.5-second shifts**: +0.012 LB
   - For each 5s clip: evaluate at original position AND +2.5s offset
   - Average 2 predictions → more robust, especially for clips at silence/call transitions
3. **Historical data exploitation** (prior competition data)
4. Pretraining: +0.03 LB

### 3rd Place
Key techniques:
1. **Model Soup** (averaging checkpoint weights)
2. Multiple spectrogram types + tf_efficientnet + mnasnet models
3. **Power adjustment post-processing**: boost confident predictions
   - `scores = scores^gamma`, gamma tuned on holdout (gamma < 1 sharpens)
4. Supplemented 2025 data with 80% of prior year data

### 4th Place
Key techniques:
1. **SoftAUCLoss**: directly optimize AUC instead of BCE/focal
   - Pairwise ranking loss: maximize area under ROC curve directly
   - Better for imbalanced multi-label problems
2. Semi-supervised two training rounds
3. Deliberately avoided self-distillation (underperformed in their experiments)

### 5th Place
Key techniques:
1. **Manual data curation** for rare classes (<30 samples): "secret technique everyone is too lazy to do"
2. **Silero voice removal**: remove clips with human speech contamination
3. Three-stage training with self-distillation + pseudolabels
4. EfficientNet ensembles (v2_s, v2_b3, b3_ns variants)

---

## Detailed Training Trajectories (LB confirmed)

### 1st place progression
| Stage | Score | Delta |
|-------|-------|-------|
| Baseline (EfficientNet-L0 + RegNetY-8, CE+AdamW+Cosine) | 0.872 | — |
| + Round 1 pseudo-labels + MixUp + StochasticDepth | 0.898 | +0.026 |
| + Power-scaled pseudo labels × 4 more rounds | 0.930 | +0.032 |
| + Separate insect/amphibian pipeline | 0.933 | +0.003 |

### 2nd place progression
| Stage | Score | Delta |
|-------|-------|-------|
| Baseline | 0.84 | — |
| + Pretraining on full Xeno Archive (500-sample cap bug) | 0.87 | +0.03 |
| + Two-round pseudo-labeling | 0.91 | +0.04 |
| + TTA 2.5s shifts | 0.922 | +0.012 |

### 2nd place class weighting formula
```python
sample_weights = (class_freq / total) ** (-0.5)  # sqrt-inverse frequency
```

### 2nd place pseudo-label filtering
- Confidence thresholds: score ≥ 0.5, multi-threshold ≥ 0.1, probability ≥ 0.4
- 3 filtering iterations per round

---

## Techniques Applicable to BirdCLEF 2026

### Already implemented
- SED + EfficientNet ensemble ✅
- Model Soup ✅
- Pseudo labeling (rounds 1-5) ✅
- ASL loss (similar to SoftAUC direction) ✅
- Texture taxa separation ✅

### NOT yet implemented (high priority)
1. **TTA with time shifts**: +0.012 LB, trivial to implement
   - Run Perch/SED on each clip AND on +2.5s shifted version
   - Average predictions

2. **Power scaling post-processing**: tune gamma on OOF
   - `final_scores = final_scores^gamma` (element-wise)
   - gamma < 1 sharpens peaks, gamma > 1 suppresses noise
   - BirdCLEF 2025: used between 0.5-0.8

3. **SoftAUC loss for LogReg probe**: replace BCE with pairwise AUC loss
   - Especially impactful for rare species where positive/negative ratio is extreme

4. **SoftAUCLoss for direct AUC optimization** (4th place):
   - Pairwise log-loss that directly maximizes ROC-AUC
   - Implemented as: for each pos/neg pair, maximize log P(score_pos > score_neg)
   - More principled than BCE for our macro-AUC evaluation metric

5. **sqrt-inverse class frequency weighting** (2nd place):
   ```python
   sample_weights = (class_freq / total) ** (-0.5)
   ```
   - More conservative than inverse-frequency; prevents extreme overweighting of rare classes

---

## Perch-Specific Insights from 2025

From Kaggle dataset "BirdCLEF 2025 Perch Embeddings" (Carlo Lepelaars):
- Competitors pre-computed Perch embeddings and used them as features
- Student-teacher KL divergence against Perch predictions (embed distillation)
- Token embeddings averaged → linear/shallow neural classifier (same as our probe)
- Key open questions: rare class detection, out-of-distribution generalization

From Perch 2.0 paper (arXiv 2508.04665):
- Perch 2.0 improves on 1.0 for few-shot classification with k=16 examples/class
- Uses mean embeddings for few-shot classification baseline
- Better few-shot performance with more soundscape-domain fine-tuning

---

## Critical Insight: Zero-Coverage Classes

For BirdCLEF 2026 specifically:
- 163/234 classes have 0 labeled soundscape positives
- These classes contribute equally to the evaluation metric
- Current system: falls back entirely to Perch native logit for these classes
- **This is the single biggest gap to fill**

The 2025 winner's approach (noisy student + external data) directly addresses this:
→ Generate pseudo labels for unlabeled soundscapes → expand coverage

Our approach: use `perch_teacher_all_ss.csv` (127k clips, 234 Perch scores):
- thresh=0.3: 62/163 zero-positive classes get pseudo coverage
- thresh=0.2: 84/163 zero-positive classes get pseudo coverage
- Need to extract 1536-dim Perch embeddings for these clips to expand probe support set
