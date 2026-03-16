# BirdCLEF 2026 — Codebase

Kaggle competition: multi-label bird/amphibian/insect species classification from 5-second audio segments.
**Metric**: Macro-averaged ROC-AUC over 234 species. **Test data**: Soundscape recordings (Pantanal, Brazil).

---

## Current Best Results (2026-03-16)

| Model / Ensemble | Holdout AUC | LB | Notes |
|-----------------|-------------|-----|-------|
| **ensemble(label×2 + emb)** | **0.9780** ⭐ | TBD | 3-model Perch ensemble, today's best |
| ensemble(label×2) | 0.9595 | — | label-pseudo + label-soundscape |
| nohuman-label-soundscape-train | 0.9550 | **0.839** | Latest submitted |
| nohuman-embedding-soundscape | 0.9537 | — | 1536-dim Perch embedding head |
| nohuman-label-pseudo | 0.9453 | 0.849+PP | Pseudo labels, best individual |
| *Competitor SED (reference)* | *0.9883* | *0.862* | *EfficientNet-B0, target to beat* |
| sed-b0-v5 | TBD | — | 🔄 Training (ep3/20) |

> **Holdout set**: 7,037 individual recordings, 206/234 species, NO data leak.
> Holdout AUC is ~0.04 higher than soundscape val AUC due to domain difference.

---

## Architecture Overview

### 1. Perch Label-Head (3 variants, ensemble = 0.9780 holdout)

```
Audio (5s) → Silero VAD (human removal) → Perch v2 TFLite →
  ├── 14795-dim label logits → gather 234 → FC(256)→ReLU→FC(234) → sigmoid  [label_head]
  └── 1536-dim embedding               → FC(1024)→ReLU→FC(234) → sigmoid  [embedding_head]
```

**Key insight**: Single Perch forward pass extracts both label logits AND embedding simultaneously — no redundant computation for 3-model ensemble.

Trained models:
- `nohuman-label-pseudo`: label_head, trained with round-1 pseudo labels
- `nohuman-label-soundscape-train`: label_head, + 53 soundscape files for domain adaptation
- `nohuman-embedding-soundscape`: embedding_head (1536-dim), richer features, best SS val (0.9810)

### 2. SED EfficientNet-B0 (sed-b0-v5, training)

```
Audio (5s) → peak-norm → MelSpec(224-mel, n_fft=2048, hop=512, norm=slaney, htk) →
  EfficientNet-B0 → GEMFreqPool(p=3.0) → AttentionSEDHead → clip_prob (234,)
                                        → frame_logit (T, 234)
```

Loss: dual clip+frame BCE (`clip_loss_weight=0.5, frame_loss_weight=0.5`)
Training: plain BCE on sigmoid outputs, soundscape 80/20 file-level split, mixup=0.5

---

## Confirmed LB History

| Date | Model | LB | Notes |
|------|-------|----|-------|
| 2026-03-15 | nohuman-label-pseudo + PP | **0.849** | Previous best |
| 2026-03-16 | nohuman-label-soundscape-train (TFLite) | 0.839 | No PP |

---

## Project Structure

```
BirdClef-2026-Codebase/
│
├── configs/
│   ├── default.yaml                             # Base Perch config
│   ├── exp_nohuman_label_pseudo.yaml            # label-head + pseudo labels
│   ├── exp_nohuman_label_soundscape_train.yaml  # label-head + soundscape train
│   ├── nohuman-embedding-soundscape.yaml        # embedding-head (1536-dim)
│   ├── sed_b0_v5.yaml                           # SED EfficientNet-B0 v5 (current)
│   ├── sed_b0_v4.yaml                           # SED v4 (killed — val data leak)
│   ├── sed_b0_v3.yaml                           # SED v3 (reference)
│   └── holdout_val_files.csv                    # 7,037 holdout files (never in training)
│
├── submissions/
│   └── ensemble_tflite.ipynb                    # ⭐ Production: 4-model ensemble notebook
│                                                #   Perch×3 TFLite + SED PyTorch CPU
│
├── submissions/weights/                         # Kaggle dataset: birdclef2026-ensemble-weights
│   ├── perch_v2_cpu.tflite                      # Perch backbone (391 MB)
│   ├── label_head_pseudo.tflite                 # label-head, pseudo labels
│   ├── label_head_soundscape_train.tflite       # label-head, soundscape domain
│   ├── embedding_head_nohuman_embedding_soundscape.tflite  # embedding-head (1536-dim)
│   └── best_sed_b0_v5.pt                        # [pending] SED PyTorch weights
│
├── src/
│   ├── audio/human_filter.py                    # Silero VAD human-voice removal
│   ├── data/
│   │   ├── dataset.py                           # CachedEmbeddingDataset, SoundscapeDataset
│   │   ├── augment.py                           # Mixup, time masking, gain
│   │   └── mel_dataset.py                       # MelClipDataset, MelSoundscapeDataset [SED]
│   ├── model/
│   │   ├── classifier.py                        # PerchClassifier (label_head / embedding_head)
│   │   ├── losses.py                            # FocalBCELoss [TF]
│   │   └── sed_model.py                         # SEDModel, GEMFreqPool, AttentionSEDHead
│   ├── metrics/kaggle_metric.py                 # Macro ROC-AUC scorer
│   └── utils/config.py                          # YAML config loader
│
├── train.py                                     # Perch head training (cached embeddings)
├── train_sed.py                                 # SED end-to-end training (raw audio → mel)
├── extract_embeddings.py                        # Cache Perch embeddings (label + embedding)
├── pseudo_label.py                              # Generate pseudo labels from trained model
├── inference.py                                 # Perch inference on test soundscapes
├── inference_sed.py                             # SED inference on test soundscapes
│
├── evaluate_holdout.py                          # Single-model holdout AUC eval
├── evaluate_ensemble_v2_holdout.py              # 3-model Perch ensemble eval
├── evaluate_ensemble_v3_holdout.py              # 4-model Perch+SED ensemble eval
├── evaluate_soundscape_val.py                   # Soundscape val AUC (SS domain)
├── evaluate_competitor_sed.py                   # Competitor model (best_fold0.pt) eval
├── convert_embedding_head_tflite.py             # Convert TF head → TFLite
│
├── scripts/
│   ├── after_embedding_head.sh                  # Watcher: TFLite + eval after emb-head done
│   └── after_sed_v5.sh                          # Watcher: copy .pt + 4-model eval after SED
│
└── pseudo_labels/
    ├── round1_pseudo.csv                        # Round 1 pseudo labels (used in training)
    └── combined_pseudo_r1.csv                   # Combined pseudo labels
```

---

## Key Technical Findings

### What Works

| Technique | Effect | Evidence |
|-----------|--------|----------|
| Human voice removal (Silero VAD) | +0.039 LB | Ablation confirmed |
| Pseudo labels (round 1) | +0.003 LB | Small but consistent |
| Soundscape domain adaptation | +0.010 holdout | label-soundscape vs label-pseudo |
| 1536-dim embedding head | +0.0185 ensemble | Complementary to label-head features |
| 3-model Perch ensemble | +0.0327 holdout | 0.9453 → 0.9780 over baseline |
| File-level soundscape split | Prevents data leak | Clip-level split → val=0.9999 (sed-b0-v4 bug) |
| Dual clip+frame SED loss | Matches competitor | clip_w=0.5, frame_w=0.5 |
| Post-processing threshold=0.02 | +0.012 LB | Only effective on soundscape domain |

### What Didn't Work

| Technique | Result |
|-----------|--------|
| Embedding-head alone vs label-head | −0.07 LB (worse solo, but valuable in ensemble) |
| FocalBCE loss on SED sigmoid outputs | Trivial minimum — loss expects logits, not probs |
| soundscape_val_frac=1.0 + soundscape training | Data leak: val=0.9999 (sed-b0-v4, killed) |
| Post-processing on individual recording holdout | No benefit (0.9780 raw vs 0.9778 +PP) |

### Architecture Notes

- **SED val metric is unreliable** when val files come from stations not in training soundscapes.
  52% of val species absent from training soundscapes → use holdout AUC as primary metric.
- **Perch ensemble ceiling** is ~0.978–0.982. SED is the primary path to exceed competitor (0.9883).
- **Single Perch forward pass** extracts both 14795-dim label logits and 1536-dim embedding
  simultaneously, enabling efficient 3-head inference with no redundant computation.

---

## Embeddings Cache

| Cache | Dim | Splits | Purpose |
|-------|-----|--------|---------|
| `embeddings_cache_nohuman` | 1536 | train=85536, holdout=21111, soundscape=739 | embedding_head |
| `embeddings_cache_nohuman_label` | 234 | train=85536, holdout=21111, soundscape=739, pseudo=2421 | label_head |

---

## Running Experiments

```bash
# Train Perch label-head (cached embeddings, fast)
python train.py --config configs/exp_nohuman_label_soundscape_train.yaml --gpu 0

# Train SED end-to-end
python train_sed.py --config configs/sed_b0_v5.yaml --gpu 1

# Evaluate single model holdout AUC
python evaluate_holdout.py --runs nohuman-embedding-soundscape

# 3-model Perch ensemble holdout eval
python evaluate_ensemble_v2_holdout.py

# 4-model Perch+SED ensemble holdout eval
python evaluate_ensemble_v3_holdout.py

# Soundscape val AUC table (SS domain comparison)
python evaluate_soundscape_val.py

# Convert Perch head to TFLite for submission
python convert_embedding_head_tflite.py --run nohuman-embedding-soundscape
```

---

## Next Steps

1. **sed-b0-v5** (ep3/20, GPU1): Complete → holdout eval → 4-model ensemble
2. **SED improvement**: Larger backbone (EfficientNet-B2/B4, EfficientNetV2-S), SpecAugment, more epochs
3. **Submit v2**: 3-model Perch TFLite ensemble (weights ready, `submissions/ensemble_tflite.ipynb`)
4. **Submit v3**: 4-model Perch+SED after `best_sed_b0_v5.pt` generated by SED watcher
