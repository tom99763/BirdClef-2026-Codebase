---
tags: [index, navigation]
last-updated: 2026-04-08
---

# BirdCLEF 2026 Wiki

Project wiki for the BirdCLEF 2026 Kaggle competition: multi-label bioacoustic species recognition (234 species) from 5-second soundscape segments. Metric: macro-averaged ROC-AUC.

**Current Best LB: 0.943**

---

## Project Overview

| Page | Summary |
|------|---------|
| [[overview]] | High-level project overview, current status, and best scores |
| [[data]] | Dataset description: train_audio, train_soundscapes, taxonomy, domain gap |
| [[glossary]] | Key terms and abbreviations used throughout the project |

## Model Architecture

| Page | Summary |
|------|---------|
| [[architecture-sed-model]] | SED model: MelTransform, GEMFreqPool, AttentionSEDHead, forward pass |
| [[backbones]] | EfficientNet-B0 vs PVT-v2-B0: architecture differences and inductive biases |
| [[residual-corrector]] | BiSSM Temporal Residual Corrector: architecture, training, alpha parameter |

## Training Frameworks

| Page | Summary |
|------|---------|
| [[noisy-student]] | Noisy Student (NS): multi-round self-training with Perch teacher bootstrap |
| [[noisy-classmate]] | Noisy Classmate (NC): 5-phase cross-architecture co-evolutionary training |
| [[training-configs]] | Key hyperparameters, per-round pseudo label schedule, config reference |
| [[augmentation]] | SumixFreq, MixUp, SpecAugment, wave-level and mel-level augmentation |

## Ensemble & Submission

| Page | Summary |
|------|---------|
| [[ensemble-diversity]] | Why diversity beats AUC, GT paradox, fold/round selection rules |
| [[vlom-analysis]] | VLOM blend weight analysis: CV-LB inversion, logit ratio, Bayesian ceiling, strategy |
| [[lb-experiments]] | Complete LB submission history with analysis and lessons learned |
| [[onnx-export]] | ONNX export pipeline, INT8 quantization, Kaggle notebook integration |

## Infrastructure

| Page | Summary |
|------|---------|
| [[pipeline-automation]] | Auto scripts, dual-GPU scheduling, watchdog monitoring |
| [[log]] | Chronological experiment log (template) |
