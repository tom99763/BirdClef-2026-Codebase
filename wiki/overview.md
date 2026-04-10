---
tags: [overview, project]
last-updated: 2026-04-08
---

# Project Overview

## Competition

**BirdCLEF 2026** on Kaggle: classify bird, amphibian, insect, mammal, and reptile species from 5-second soundscape segments recorded in the Pantanal region.

- **Task**: Multi-label classification (234 species)
- **Metric**: Macro-averaged ROC-AUC
- **Input**: 5-second audio segments from 60-second soundscape recordings
- **Current Best LB**: 0.943

## Method Summary

The project combines two training frameworks on top of a shared SED (Sound Event Detection) architecture:

### 1. Noisy Student (NS) -- Foundation

Multi-round self-training. A Perch foundation model generates initial pseudo labels on unlabeled soundscapes, then SED students train on labeled data + pseudo-labeled soundscapes. Each round, the student becomes the teacher for the next round.

```
Perch Teacher -> R0 pseudo -> B0 R1 -> R1 pseudo -> B0 R2 -> ... -> B0 R12
                                                      |
                                              (seed at R4)
                                         PVT R1 -> PVT R2 -> ... -> PVT R8
```

See [[noisy-student]] for full details.

### 2. Noisy Classmate (NC) -- Innovation

Multiple heterogeneous-architecture students stop self-studying and start peer-teaching. Five progressive phases introduce ensemble blending, confidence weighting, disagreement mining, soft distillation, and bidirectional co-evolution.

See [[noisy-classmate]] for full details.

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| 20-second clip duration | Covers 4 Perch 5s windows; matches BirdCLEF 2025 1st-place insight |
| Non-Aves use Perch only | SED unreliable for Amphibia/Insecta/Mammalia/Reptilia (see [[data]]) |
| EMA inheritance across rounds | Warm-start each round from previous EMA checkpoint |
| Round diversity in ensemble | Models from different rounds have lower prediction correlation (see [[ensemble-diversity]]) |
| All training on GPU1 | `CUDA_VISIBLE_DEVICES=1` for all NS training; NC uses dual-GPU |
| No competitor model weights | Only self-trained models used |

## Project Structure

```
BirdClef-2026-Codebase/
  configs/                          YAML configs per backbone/round
  train_sed_ns.py                   Main training script (NS + NC)
  scripts/
    gen_pseudo_ns.py                NS pseudo label generation
    gen_noisy_classmate_pseudo.py   NC pseudo (Phase 1-4)
    train_sed_residual_corrector.py BiSSM residual corrector
    export_sed_to_onnx.py           ONNX + INT8 export
    auto_sed_ns_20s_full.sh         B0 NS R1-R4 pipeline
    auto_nc_dual_gpu.sh             NC dual-GPU pipeline
    monitor_nc.sh                   Pipeline monitor
  outputs/                          Checkpoints, npz, logs
  pseudo_labels/                    Per-round pseudo label CSVs
  reports/                          HTML reports and paper draft
  wiki/                             This wiki
```

## Score Progression

| Date | LB | Config |
|------|----|--------|
| 2026-04-08 | **0.943** | B0 R12 f0 + PVT R5 f2 + B0 R6 f3 |
| 2026-04-07 | 0.942 | B0 R12 f0 + PVT R5 f4 + B0 R6 f3 |
| 2026-04-07 | 0.941 | B0 R12 f0 + PVT R5 f4 + B0 R8 f3 |
| 2026-04-06 | 0.938 | B0 R8 f2 + PVT R5 f4 + B0 R8 f3 |
| 2026-04-03 | 0.934 | B0 R8 f2 + PVT R5 f4 + B0 R8 f3 |
| 2026-03-30 | 0.933 | v17 baseline |

See [[lb-experiments]] for the complete submission history.

## Constraints

- Only **nohuman** models evaluated and submitted
- No competitor model weights -- only models we trained ourselves
- Domain Generalization framing (test_soundscapes has no audio for adaptation)
