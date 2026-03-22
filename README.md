# BirdCLEF 2026 — Codebase

Kaggle competition: multi-label bird/amphibian/insect species classification from 5-second audio segments.
**Metric**: Macro-averaged ROC-AUC over 234 species. **Test data**: Soundscape recordings (Pantanal, Brazil).

---

## Current Best Results (2026-03-22)

| Model / Ensemble | Holdout AUC | LB | Notes |
|-----------------|-------------|-----|-------|
| **LGBM + R46.08 event smooth** | 0.8140 OOF | **0.926** ⭐ | Current best — R46.08 post-proc on LGBM ensemble |
| LGBM probe (ptmap-lgbm / lgbm-infer) | — | **0.925** | LGBM per-class probe, no post-proc |
| v3-ensemble (Perch 70/30 + SED VLOM) | — | **0.921** | Bayesian probe PCA64+LogReg α=0.40 + SED 50/50 |
| v9-asl-soup ensemble | — | 0.892 | First VLOM blend submission |
| *Competitor SED (reference)* | *~0.90 soundscape* | — | *Our SED soundscape val AUC target* |
| **sed-b0-v33 (WarmRestart)** | 0.7625 ep3 | — | 🔄 Training (GPU1), v33→v36 chain |

> **Soundscape val gap**: Our best SED ~0.80 vs competitor ~0.90. v33-v36 experiments target this gap.
> Only nohuman models evaluated from 2026-03-15 onwards.

---

## LB Score History

| Date | Submission | LB | Notes |
|------|-----------|-----|-------|
| 2026-03-22 | LGBM + R46.08 event smooth | **0.926** ⭐ | lmax_pre_aves→SoftRich→cSEBBs OOF=0.8140 |
| 2026-03-21 | lgbm-infer / ptmap-lgbm | **0.925** | LGBM probe breakthrough; PT-MAP ineffective |
| 2026-03-21 | lgbm-4fold (our SED only) | 0.908 | Replaced competitor SED → -0.013. Our SED weak. |
| 2026-03-21 | kNN+LGBM (competitor SED) | 0.923 | kNN probe change -0.002 vs 0.925 baseline |
| 2026-03-20 | v3-ensemble | 0.921 | Perch 70% Bayes + SED 50/50 VLOM + 3 tricks |
| 2026-03-17 | v9-asl-soup ensemble | 0.892 | First competitive submission |

**Key insight**: Replacing competitor SED with our own SED in ensemble → -0.013 LB drop (0.921→0.908).
Our SED soundscape val AUC ~0.80 vs competitor ~0.90. Improving SED is top priority.

---

## Architecture Overview

### 1. Perch Embedding Probe (current best: LB 0.926)

```
Audio (60s) → 12×5s clips → Perch v2 TFLite →
  ├── 14795-dim logits → gather 234 → Bayesian prior fusion
  └── 1536-dim embedding → PCA(64) → LGBM probe (74-dim features)
  → final = (1-α)×base + α×lgbm_pred   [α=0.40 blend weight]
  → lmax_pre_aves(α=0.1) → SoftRich(α=0.40) → cSEBBs → submission
```

**LGBM probe**: `alpha=0.40` is the blend weight `final = (1-0.40)*perch_base + 0.40*lgbm_pred`.
It is NOT a LogReg regularization param. The probe itself is LightGBM, not LogReg.

Key techniques:
- **Bayesian prior fusion**: site/hour joint priors fused into logits
- **Texture smooth** (avg-neighbor, α=0.35): for Amphibia/Insecta classes
- **Event smooth** (local-max, α=0.15): for Aves classes — preserves transient peaks
- **Gaussian logit smooth**: `convolve1d([0.1,0.2,0.4,0.2,0.1])` on logits before sigmoid
- **Temperature scaling**: T=1.15 on logits before sigmoid
- **LGBM probe**: 74-dim features (PCA32 + raw/prior/base + seq + 3 interactions)

### 2. Post-Processing Pipeline (R51, OOF AUC 0.8164)

Applied to SED branch in VLOM blend. Best pipeline found by `scripts/eval_smooth_experiments.py`:

```
SED logits → lmax_pre_aves(α=0.1, radius=1, Aves-only idx 72-233)
           → SoftRich(alpha=0.40)  [cross-file richness normalization]
           → cSEBBs(cp_blend=0.60, cp_thr=0.05)
           → OOF AUC = 0.8164
```

History of post-processing OOF AUC:
- R46.08: `SoftRich+cp_blend0.60+cp_thr0.05+cSEBBs` = 0.8140 → **LB 0.926**
- R50: `lmax_pre_aves(α=0.1)→SoftRich(α=0.38)→cSEBBs` = 0.8163
- R51: `lmax_pre_aves(α=0.1)→SoftRich(α=0.40)→cSEBBs` = **0.8164** (current best OOF)

**R52 (running)**: bidirectional lmax, logit-scale sweep (lscale=0.9 → 0.8165), power scaling, adaptive nSEBBs.
**R53 (queued)**: P_max soundscape lifting (BirdCLEF 2024 3rd place +0.01-0.02), per-class PCR nSEBBs, onset peak-finding.

### 3. SED EfficientNet-B0 (current: v33-v36 improvement chain)

```
Audio (5s) → MelSpec(224-mel, n_fft=2048, hop=512, fmin=0, fmax=16k) →
  EfficientNet-B0 (tf_efficientnet_b0.ns_jft_in1k) → GEMFreqPool(p=3.0) → FC(234) → sigmoid
```

**Dual loss** (clip + frame):
```
total_loss = clip_w * clip_loss(clip_pred, labels)
           + frame_w * BCE(frame_logit, frame_labels)
```
All custom clip losses (focal, bce_pos_weight, asl) now correctly include frame BCE.
Bug: prior to 2026-03-22, `_compute_loss` else-branch silently dropped frame supervision.

**v33-v36 improvement chain** (running on GPU1):
| Version | Key change | Status |
|---------|-----------|--------|
| v33 | WarmRestart (T0=25, T_mult=2), bce dual loss | ep3/50, best=0.7625 |
| v34 | Focal loss γ=2.0 + frame BCE (bug fixed) | Queued |
| v35 | Focal loss γ=3.0 + frame BCE (bug fixed) | Queued |
| v36 | BCE pos_weight=2.0 + frame BCE (bug fixed) | Queued |

---

## Project Structure

```
BirdClef-2026-Codebase/
│
├── configs/
│   ├── sed_b0_v*.yaml                        # SED experiment configs
│   ├── sed_b0_v33_warmrestart.yaml           # v33: warm restart
│   ├── sed_b0_v34_focal_g2.yaml              # v34: focal γ=2
│   ├── sed_b0_v35_focal_g3.yaml              # v35: focal γ=3
│   ├── sed_b0_v36_pos_weight.yaml            # v36: BCE pos_weight=2
│   └── embed_distill_*.yaml                  # Embedding distillation experiments
│
├── submissions_v3/                           # Current submission notebooks
│   ├── birdclef-2026-v3-lgbm-infer.ipynb              # LGBM probe (LB 0.925)
│   ├── birdclef-2026-v3-lgbm-event-smooth.ipynb       # + event smooth (LB 0.925)
│   ├── birdclef-2026-v3-lgbm-event-r50-softrich-postproc.ipynb  # + R50/R51 post-proc
│   ├── birdclef-2026-v3-lgbm-4fold.ipynb              # 4-fold LGBM probe ensemble
│   └── weights/                                        # Model weights for Kaggle upload
│
├── event_smooth/                             # Post-processing experiment notebooks
│   ├── postproc_R50_lmax_pre_aves_softrich.ipynb     # R50: OOF 0.8163
│   ├── postproc_R51_lmax_pre_softrich_a040.ipynb     # R51: OOF 0.8164 (best)
│   └── best_postproc_R51_*.ipynb                     # Save-best checkpoint
│
├── src/
│   ├── data/
│   │   ├── dataset.py                       # CachedEmbeddingDataset, SoundscapeDataset
│   │   ├── augment.py                       # Mixup, time masking, gain
│   │   └── mel_dataset.py                   # MelClipDataset, MelSoundscapeDataset
│   │                                        # SoftPseudoSoundscapeDataset
│   ├── model/
│   │   ├── classifier.py                    # PerchClassifier (label/embedding head)
│   │   ├── sed_model.py                     # SEDModel, GEMFreqPool, AttentionSEDHead
│   │   ├── pcen.py                          # PCEN learnable frontend
│   │   └── losses.py                        # FocalBCELoss, ASLoss
│   └── utils/config.py                      # YAML config loader
│
├── train_sed.py                             # SED end-to-end training (raw audio → mel)
│                                            # _compute_loss: dual loss fixed 2026-03-22
│                                            # Supports: focal/bce_pos_weight/asl clip loss
│                                            # + frame BCE always applied when frame_w>0
├── train_distill.py                         # Perch→SED knowledge distillation
├── train_embed_distill.py                   # Embedding distillation training
├── train_sedp.py                            # SED_P (PCEN+MaskedBCE+LLRD+AMP)
│
├── scripts/
│   ├── eval_smooth_experiments.py           # Post-processing sweep (R46→R53)
│   │                                        # Finds optimal lmax/SoftRich/cSEBBs params
│   ├── sweep_vlom_blend.py                  # Sweep PERCH_W/SED_W blend on soundscape val
│   ├── eval_sed_holdout.py                  # SED holdout AUC eval
│   ├── eval_sed_holdout_tta.py              # Holdout eval with TTA
│   ├── eval_geo_holdout.py                  # Geo-holdout evaluation
│   ├── pseudo_label_sed.py                  # Generate pseudo labels from SED
│   └── update_exp_results.py               # Auto-update reports/exp_results.xlsx
│
├── pseudo_labels/
│   ├── round2_pseudo.csv ~ round5_pseudo.csv
│   └── combined_pseudo_r1.csv
│
└── reports/
    └── exp_results.xlsx                     # Experiment log (auto-updated)
```

---

## Key Technical Findings

### What Works

| Technique | Effect | Evidence |
|-----------|--------|----------|
| Human voice removal (Silero VAD) | +0.039 LB | Ablation confirmed |
| LGBM probe (vs LogReg) | LB 0.925 | Better than LogReg; 74-dim with interaction features |
| Bayesian prior fusion | ~+0.02 LB | Site/hour priors on Perch logits |
| 3-model Perch ensemble | +0.0327 holdout | 0.9453 → 0.9780 |
| VLOM blend (Perch+SED) | LB 0.892→0.921 | Geometric-RMS blend beats linear |
| Event smooth (local-max α=0.15) | LB 0.908 | Aves classes — preserves transient peaks |
| Gaussian logit smooth | LB 0.910 | Applied to logits (not probs) before sigmoid |
| Temperature scaling T=1.15 | LB 0.908 | Softens overconfident logits |
| lmax_pre_aves (α=0.1, Aves-only) | OOF +0.0001 | Local-max propagation in logit space |
| SoftRich (α=0.40) | OOF 0.8164 | Cross-file richness normalization |
| cSEBBs (cp_blend=0.60, cp_thr=0.05) | OOF 0.8140→0.8164 | Change-point rich-segment boosting |
| Pseudo labels | +0.003 LB | Small but consistent |
| Soundscape domain adaptation | +0.010 holdout | label-soundscape vs label-pseudo |

### What Didn't Work

| Technique | Result | Notes |
|-----------|--------|-------|
| Focal clip loss only (no frame BCE) | Silent bug | Frame supervision ignored — now fixed |
| ASL + secondary_weight=1.0 | Noisy gradients | Amplifies unreliable XC secondary labels |
| CAWR scheduler | AUC collapse | Collapses at restart epochs |
| PT-MAP (few-shot meta-learning) | No gain | lgbm-infer = ptmap-lgbm = 0.925. PT-MAP ineffective |
| kNN probe (vs LGBM) | -0.002 LB | 0.925→0.923 when replacing LGBM with kNN |
| Quantile-Mix / RankBlend (R31) | Catastrophic | Destroys class-relative ordering |
| Gaussian smooth on probs (post-sigmoid) | Weaker | Logit-space smoothing consistently better |
| Replacing competitor SED with our SED | -0.013 LB | 0.921→0.908 — our SED still weak on soundscapes |

---

## Running Experiments

```bash
# SED v33-v36 improvement chain (GPU1)
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/after_sed_v5.sh > outputs/sed_v33_v36_chain.log 2>&1 &

# Post-processing sweep (R46→R53)
CUDA_VISIBLE_DEVICES=1 python3 scripts/eval_smooth_experiments.py

# VLOM blend sweep (find optimal PERCH_W/SED_W)
CUDA_VISIBLE_DEVICES=1 python3 scripts/sweep_vlom_blend.py

# SED holdout eval
CUDA_VISIBLE_DEVICES=1 python scripts/eval_sed_holdout.py --checkpoint outputs/sed-b0-v33-warmrestart/best.pt
```

---

## Submission Rules

- Only submit when **individual SED soundscape val AUC > 0.9193** (v5 benchmark), OR as part of ensemble with competitor SED
- Only evaluate/submit **nohuman models** (non-nohuman results discarded from 2026-03-15)
- All training must run on **GPU1** (`CUDA_VISIBLE_DEVICES=1`)
- Current LB anchor: **0.926** — only submit if expected to beat this
