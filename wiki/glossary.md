---
tags: [glossary, reference]
last-updated: 2026-04-08
---

# Glossary

Key terms, abbreviations, and concepts used throughout this project.

## Models & Architecture

| Term | Definition |
|------|-----------|
| **SED** | Sound Event Detection -- the core model architecture that classifies audio into species |
| **B0** | EfficientNet-B0 backbone (`tf_efficientnet_b0.ns_jft_in1k`) |
| **PVT** | Pyramid Vision Transformer v2 B0 backbone (`pvt_v2_b0`) |
| **ConvNeXt / CNeXt** | ConvNeXt backbone (experimental, paused) |
| **GEMFreqPool** | Generalized Mean pooling along frequency axis; learnable parameter p between avg and max pooling |
| **AttentionSEDHead** | Attention-weighted classification head: soft attention over time frames multiplied by per-class logits |
| **Perch** | Google's bioacoustic foundation model; used as initial teacher for pseudo labels |
| **CLAP** | Contrastive Language-Audio Pretraining; helps non-Aves species classification |

## Training Frameworks

| Term | Definition |
|------|-----------|
| **NS** | Noisy Student -- multi-round self-training with pseudo labels |
| **NC** | Noisy Classmate -- cross-architecture co-evolutionary peer-training (our innovation) |
| **Round (R)** | One iteration of the NS/NC pipeline: train -> infer -> correct -> pseudo -> repeat |
| **Fold (f)** | One of 5 cross-validation splits (GroupKFold by soundscape file ID) |
| **Generation** | One bidirectional cycle in NC Phase 5 (PVT round + B0 round) |

## Loss & Optimization

| Term | Definition |
|------|-----------|
| **FocalBCE** | Focal Binary Cross-Entropy: `(1-pt)^gamma * BCE`. gamma=2.0 down-weights easy negatives |
| **NCDistillLoss** | NC Phase 4 loss: `(1-beta)*FocalBCE + beta*KLD*T^2`. Combines hard and soft targets |
| **KLD** | Kullback-Leibler Divergence -- measures distribution distance for soft distillation |
| **EMA** | Exponential Moving Average of model weights (decay=0.999); inherited across rounds |
| **AdamW** | Adam optimizer with decoupled weight decay |
| **CosineAnnealing** | Learning rate scheduler: cosine decay from lr to eta_min |

## Pseudo Labels

| Term | Definition |
|------|-----------|
| **Pseudo label** | Model-generated soft label for unlabeled soundscape windows |
| **Power transform (gamma)** | `probs^gamma` -- sharpens predictions. gamma=2.0 compresses low-confidence scores |
| **Dynamic threshold** | Per-class threshold at Nth percentile of predictions (default: 95th) |
| **Perch weight** | Blend weight for Perch teacher in ensemble pseudo labels; decreases across rounds |
| **Residual Corrector** | BiSSM model that corrects SED predictions using Perch-SED residual patterns |
| **BiSSM** | Bidirectional Selective State Space Model (Mamba-style); used in the Residual Corrector |

## Augmentation

| Term | Definition |
|------|-----------|
| **MixUp** | Waveform mixing: `0.5*x_a + 0.5*x_b`, labels take union (max) |
| **SumixFreq** | Per-frequency-bin random selection between two spectrograms; creates multi-species mels |
| **SpecAugment** | Frequency and time masking on mel spectrograms |
| **absmax** | Peak normalization: divide waveform by its absolute maximum |

## Data & Evaluation

| Term | Definition |
|------|-----------|
| **Macro AUC** | Macro-averaged ROC-AUC: compute AUC per species, then average. Competition metric |
| **LB** | Leaderboard score on Kaggle's hidden test set |
| **CV** | Cross-validation score on labeled soundscapes (5-fold) |
| **GT** | Ground Truth -- the 1,478 labeled soundscape windows |
| **GT Paradox** | Phenomenon where higher GT AUC does not predict higher LB score |
| **Domain gap** | Acoustic distribution difference between train_audio (point recordings) and soundscapes |
| **nohuman** | Model configuration excluding human-labeled soundscape data from training (required) |

## Ensemble

| Term | Definition |
|------|-----------|
| **Round diversity** | Using models from different NS rounds in ensemble (lower prediction correlation) |
| **Architecture diversity** | Using both B0 and PVT in ensemble (different inductive biases) |
| **Fold diversity** | Using models from different CV folds (different validation targets) |
| **Prediction correlation** | Pearson correlation between two models' output probabilities; lower = more diverse |
| **Model soup** | Averaging state_dicts across folds into single model; less effective than prediction ensemble |

## Export & Deployment

| Term | Definition |
|------|-----------|
| **ONNX** | Open Neural Network Exchange -- portable model format for inference |
| **INT8** | 8-bit integer quantization (dynamic, via ONNX Runtime); ~4x size reduction |
| **FP32** | Full precision (32-bit float) ONNX export |

## Infrastructure

| Term | Definition |
|------|-----------|
| **GPU0 / GPU1** | Two available GPUs; NS uses GPU1, NC dual-GPU uses both |
| **Watchdog** | Background process monitoring pipeline health |
| **npz** | NumPy compressed archive format; used for `all_ss_probs.npz` predictions |

## Related Pages

- [[overview]] -- Project overview
- [[architecture-sed-model]] -- Model architecture details
- [[training-configs]] -- Configuration reference
