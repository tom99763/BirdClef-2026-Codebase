# BirdCLEF 2026 — Codebase

Kaggle competition: multi-label bird/amphibian/insect species classification from 5-second audio segments.
**Metric**: Macro-averaged ROC-AUC over 234 species. **Test data**: Soundscape recordings (Pantanal, Brazil).

---

## Current Best Results (2026-03-21)

| Model / Ensemble | Holdout AUC | LB | Notes |
|-----------------|-------------|-----|-------|
| **v9-asl-soup ensemble** | — | **0.892** ⭐ | Best LB, Perch + SED VLOM blend |
| Perch 3-model ensemble | 0.9780 | — | label×2 + embedding |
| nohuman-label-soundscape-train | 0.9550 | 0.839 | Perch label-head, soundscape domain |
| *Competitor SED (reference)* | *0.9918* | *0.862* | *soundscape val AUC target* |
| **sed-ss-fold0~3** | — | — | 🔄 Training (GPU1, pair-parallel) |

> **Soundscape val gap**: Our best SED = 0.8153 vs competitor = 0.9918 (0.18 gap, root cause analysed).
> Only nohuman models are evaluated from 2026-03-15 onwards.

---

## Architecture Overview

### 1. Perch Embedding Probe (submissions_v3, best LB 0.892)

```
Audio (60s) → 12×5s clips → Perch v2 TFLite →
  ├── 14795-dim logits → gather 234 → Bayesian prior fusion → texture/event smooth
  └── 1536-dim embedding → PCA(64) → LGBM probe (74-dim features)
  → final = (1-α)×base + α×probe → Gaussian logit smooth → sigmoid(/T) → submission
```

Key techniques (LB 0.892 → 0.910 reference):
- **Bayesian prior fusion**: site/hour joint priors fused into logits
- **Texture smooth** (avg-neighbor, α=0.35): for Amphibia/Insecta classes
- **Event smooth** (local-max, α=0.15): for Aves classes — preserves transient peaks (ref: 0-908)
- **LGBM probe**: 74-dim features (PCA32 + raw/prior/base + seq + 3 interactions)
- **Gaussian logit smooth**: `convolve1d([0.1,0.2,0.4,0.2,0.1])` on logits before sigmoid (ref: 0-910)
- **Temperature scaling**: T=1.15 on logits before sigmoid (ref: 0-908)

### 2. SED EfficientNet-B0 (current: sed-ss-fold0~3)

```
Audio (5s) → MelSpec(224-mel, n_fft=2048, hop=512, fmin=0, fmax=16k) →
  EfficientNet-B0 (tf_efficientnet_b0.ns_jft_in1k) → GEMFreqPool(p=3.0) → FC(234) → sigmoid
```

**Training data** (soundscape-domain focus):
- `train_audio`: 35,549 focal recordings (3 clips/file)
- `train_soundscapes`: 1,194 labeled clips × 20× oversample
- **Validation**: soundscape k-fold (4 folds, 16~18 files/fold)

**Root cause of SED gap (0.82 vs 0.99)**:
1. Dual frame loss (`frame_loss=0.5`) → synthetic label gradient conflict
2. ASL + `secondary_weight=1.0` → noisy XC secondary label amplification
3. CAWR scheduler → AUC collapse at restart epochs
4. No soundscape-domain training signal in earlier experiments

**Current fixes** (sed-ss-fold configs):
- CE loss + `label_smoothing=0.1`, `frame_loss_weight=0.0`
- Soundscape oversample 20× + explicit k-fold validation
- Cosine scheduler + `warmup_epochs=5`, `early_stopping_patience=15`
- Mixup α=0.5

---

## LB Score History

| Date | Submission | LB | Notes |
|------|-----------|-----|-------|
| 2026-03-17 | v9-asl-soup ensemble | **0.892** | Best — Perch+SED VLOM blend |
| 2026-03-16 | nohuman-label-soundscape-train | 0.839 | Perch only, no SED |
| 2026-03-15 | nohuman-label-pseudo + PP | 0.849 | Perch + pseudo labels |

---

## Project Structure

```
BirdClef-2026-Codebase/
│
├── configs/
│   ├── default.yaml                          # Base Perch config
│   ├── sed_default.yaml                      # Base SED config
│   ├── sed_ss_fold0~3.yaml                   # Current: soundscape 4-fold SED
│   ├── ss_folds/ss_fold{0-3}_val.txt         # Soundscape k-fold val file lists
│   ├── perch_probe_*.yaml                    # Perch embedding probe experiments
│   └── embed_distill_*.yaml                  # Embedding distillation experiments
│
├── submissions_v3/                           # Current submission notebooks
│   ├── birdclef-2026-v3-lgbm-infer.ipynb         # LGBM probe + SED VLOM blend
│   ├── birdclef-2026-v3-lgbm-event-smooth.ipynb  # + Event smooth (0-908) + Gaussian logit (0-910)
│   ├── birdclef-2026-v3-lgbm-4fold.ipynb         # 4-fold LGBM probe ensemble
│   └── weights/                                  # Model weights for Kaggle upload
│
├── src/
│   ├── data/
│   │   ├── dataset.py                       # CachedEmbeddingDataset, SoundscapeDataset
│   │   ├── augment.py                       # Mixup, time masking, gain
│   │   └── mel_dataset.py                   # MelClipDataset, MelSoundscapeDataset
│   │                                        # SoftPseudoSoundscapeDataset
│   │                                        # Bug fix: max_files=0 now correctly filters
│   ├── model/
│   │   ├── classifier.py                    # PerchClassifier (label/embedding head)
│   │   ├── sed_model.py                     # SEDModel, GEMFreqPool, AttentionSEDHead
│   │   ├── pcen.py                          # PCEN learnable frontend
│   │   └── losses.py                        # FocalBCELoss, ASLoss
│   └── utils/config.py                      # YAML config loader
│
├── train_sed.py                             # SED end-to-end training (raw audio → mel)
│                                            # Supports: k-fold val via soundscape_val_files_txt,
│                                            # soundscape oversample, CE/ASL loss, GPU mel,
│                                            # soft pseudo labels, soundscape_only mode
├── train_distill.py                         # Perch→SED knowledge distillation
├── train_embed_distill.py                   # Embedding distillation training
├── train_sedp.py                            # SED_P (PCEN+MaskedBCE+LLRD+AMP)
│
├── scripts/
│   ├── run_ss_4folds.sh                     # Soundscape 4-fold launcher (2 folds parallel)
│   ├── log_ss_folds_to_excel.py             # Auto-log fold status → reports/exp_results.xlsx
│   ├── eval_sed_holdout.py                  # SED holdout AUC eval
│   ├── eval_sed_holdout_tta.py              # Holdout eval with TTA
│   └── pseudo_label_sed.py                  # Generate pseudo labels from SED
│
├── reports/
│   └── exp_results.xlsx                     # Experiment log (SS-Folds sheet auto-updated)
│
└── pseudo_labels/
    ├── round2_pseudo.csv ~ round5_pseudo.csv
    └── combined_pseudo_r1.csv
```

---

## Key Technical Findings

### What Works

| Technique | Effect | Evidence |
|-----------|--------|----------|
| Human voice removal (Silero VAD) | +0.039 LB | Ablation confirmed |
| Pseudo labels | +0.003 LB | Small but consistent |
| Soundscape domain adaptation | +0.010 holdout | label-soundscape vs label-pseudo |
| 3-model Perch ensemble | +0.0327 holdout | 0.9453 → 0.9780 |
| VLOM blend (Perch+SED) | Best LB 0.892 | Geometric-RMS blend beats linear |
| Bayesian prior fusion | ~+0.02 LB | Site/hour priors on Perch logits |
| LGBM probe (74-dim) | Best probe | Better than LogReg; 3 interaction features |
| Gaussian logit smooth | 0-910: 0.910 LB | Applied to logits (not probs) before sigmoid |
| Event smooth (local-max) | 0-908: 0.908 LB | Aves classes — preserves transient peaks |
| Temperature scaling T=1.15 | 0-908: 0.908 LB | Softens overconfident logits |
| CE loss + frame_loss=0 | Root-cause fix | Eliminates synthetic label gradient conflict |
| Soundscape oversample 20× | SED training | Amplifies domain-relevant training signal |

### What Didn't Work

| Technique | Result |
|-----------|--------|
| Dual clip+frame loss (frame_loss=0.5) | Gradient conflict → SED gap vs competitor |
| ASL + secondary_weight=1.0 | Amplifies noisy XC secondary labels |
| CAWR scheduler | AUC collapse at restart epochs |
| Gaussian smooth on probs (post-sigmoid) | Weaker than logit-space smoothing |

---

## Running Experiments

```bash
# Soundscape 4-fold SED (2 folds parallel on GPU1)
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_ss_4folds.sh > outputs/ss_4fold_chain.log 2>&1 &

# Monitor + log to Excel
python3 scripts/log_ss_folds_to_excel.py

# SED holdout eval
CUDA_VISIBLE_DEVICES=1 python scripts/eval_sed_holdout.py --checkpoint outputs/sed-ss-fold0/best.pt
```

---

## Submission Rules

- Only submit when **individual SED soundscape val AUC > 0.9193** (v5 benchmark)
- Only evaluate/submit **nohuman models** (non-nohuman results discarded from 2026-03-15)
- All training must run on **GPU1** (`CUDA_VISIBLE_DEVICES=1`)
