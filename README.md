# BirdCLEF 2026 — Noisy Classmate Training Framework

Kaggle competition: multi-label bird/amphibian/insect species classification from 5-second soundscape segments.
**Metric**: Macro-averaged ROC-AUC over 234 species. **Current Best LB: 0.943**

---

## Method Overview

### Noisy Student (NS) — Foundation

Multi-round self-training: Perch foundation model generates initial pseudo labels on unlabeled soundscapes → SED student trains on labeled train_audio + pseudo-labeled soundscapes → student becomes teacher for next round → repeat.

```
Perch Teacher → R0 pseudo → B0 R1 → R1 pseudo → B0 R2 → ... → B0 R12
                                         ↓ (one-time seed at R4)
                                    PVT R1 → PVT R2 → ... → PVT R8
```

**Limitation**: Each chain only learns from its own predictions → confirmation bias, knowledge silos.

### Noisy Classmate (NC) — Innovation

Multiple heterogeneous-architecture students (B0, PVT) stop self-studying and start peer-teaching. Five progressive phases:

| Phase | Name | What It Does | NS Equivalent |
|-------|------|-------------|---------------|
| 1 | Ensemble Pseudo | Blend B0 + PVT predictions for shared pseudo labels | Self-only pseudo |
| 2 | Confidence Weighting | Per-sample entropy-weighted fusion (trust the more confident model) | Equal weight |
| 3 | Disagreement Mining | 3x training weight on samples where models disagree | Equal weight |
| 4 | Soft Distillation | FocalBCE + KLD loss preserving full probability structure | Hard labels only |
| 5 | Bidirectional Loop | B0→PVT→B0→PVT alternating co-evolution | Unidirectional |

```
NC Generation 1:  B0 R11 + PVT R8 → blend → PVT R9
NC Generation 2:  B0 R11 + PVT R9 → blend → PVT R10
                  PVT R10 + B0 R11 → blend → B0 R12  ← knowledge flows back!
NC Generation 3:  B0 R12 + PVT R10 → blend → PVT R11
                  ... (continues co-evolution)
```

---

## Data Preparation

### Prerequisites

```bash
export CUDA_VISIBLE_DEVICES=1   # All training on GPU1

# Required data structure:
# birdclef-2026/
#   ├── train_audio/          # Labeled per-species recordings
#   ├── train_soundscapes/    # Unlabeled 60s soundscape recordings
#   ├── taxonomy.csv          # 234 species list
#   └── train_soundscapes_labels.csv  # Sparse GT labels (66 soundscapes)
```

### Step 0 — Perch Teacher

Fine-tune a linear head on frozen Perch v2 embeddings, then extract predictions on all soundscapes:

```bash
python3 train.py --config configs/exp_nohuman_label_soundscape_train.yaml
python3 scripts/extract_perch_teacher_all_ss.py \
    --output outputs/perch_teacher_aug_all_ss.csv
```

### Step 1 — Round 0 Pseudo Labels

```bash
python3 scripts/gen_pseudo_ns.py \
    --round 0 --clip_sec 20 \
    --perch_csv outputs/perch_teacher_aug_all_ss.csv \
    --perch_w 1.0 --out pseudo_labels/ns_r0.csv
```

---

## NS Training (Rounds 1–12)

### Automated Pipeline

```bash
# B0 chain (R1→R8)
nohup bash scripts/auto_sed_ns_20s_full.sh > outputs/logs/auto_sed_ns_20s_full.log 2>&1 &

# B0 extended (R9→R15)
nohup bash scripts/auto_sed_ns_20s_r9r15.sh > outputs/logs/auto_sed_ns_20s_r9r15.log 2>&1 &

# PVT chain (seeded from B0 R4 pseudo, independent R1→R8)
nohup bash scripts/auto_sed_ns_pvt_20s_r1r4.sh > outputs/logs/auto_sed_ns_pvt_20s_r1r4.log 2>&1 &
nohup bash scripts/auto_sed_ns_pvt_20s_r5r8.sh > outputs/logs/auto_sed_ns_pvt_20s_r5r8.log 2>&1 &
```

### What Each Round Does

```
For each round R:
  1. Train 5 folds:  train_sed_ns.py --config sed_ns_{arch}_20s_r{R}.yaml --fold {0..4}
  2. Infer all soundscapes: train_sed_ns.py --infer_all_ss → all_ss_probs.npz
  3. Residual Corrector: train_sed_residual_corrector.py → all_ss_probs_corrected.npz
  4. Generate pseudo labels: gen_pseudo_ns.py → pseudo_labels/sed_20s_r{R}.csv
  5. Update next round config → repeat
```

### SED Model Architecture

```
20s audio clip (absmax normalized)
  → MelSpec(n_mels=224, n_fft=2048, hop=512, fmin=0, fmax=16000, slaney/htk)
  → 3-channel expand → SpecAugment(freq=24, time=32)
  → SumixFreq: per-freq-bin binary mask mixing between two clips
  → Wave-level MixUp (λ=0.5): labeled × pseudo cross-domain mixing
  → Backbone (EfficientNet-B0 or PVT-v2-B0) → GEMFreqPool(p=3) → AttentionSEDHead
  → FocalBCE (γ=2.0), AdamW lr=1e-3, CosineAnnealing, 25 epochs, early_stop=4
  → EMA decay=0.999, inherited across rounds
```

### Residual Corrector

```
SED probs (234-d per 5s frame)
  → Linear(234→128) + LayerNorm + GELU
  → BiSSM(d_model=128, d_state=16)
  → Linear(128→234) → residual delta
  → corrected = SED_probs + α × delta  (α=0.40)
```

### NS Pseudo Label Config

| Round | Perch Weight | Threshold Pct | Gamma | Notes |
|-------|-------------|---------------|-------|-------|
| R0 | 1.00 | 95 | 2.0 | Perch-only teacher |
| R1 | 0.50 | 92 | 1.00 | Bootstrap, raw soft labels |
| R2 | 0.30 | 93 | 1.54 | Reduce teacher weight |
| R3 | 0.10 | 94 | 1.82 | Student dominant |
| R4+ | 0.00 | 95 | 2.00 | Pure self-training |

---

## NC Training (Noisy Classmate, R9+)

### Automated Pipeline (All 5 Phases)

```bash
nohup bash scripts/auto_nc_full.sh > outputs/logs/auto_nc_full.log 2>&1 &

# Monitor
bash scripts/monitor_nc.sh
```

### NC Pseudo Label Generation

```bash
python3 scripts/gen_noisy_classmate_pseudo.py \
    --chains "b0:outputs/sed-ns-b0-20s-r11" "pvt:outputs/sed-ns-pvt-20s-r8" \
    --weights 0.5 0.5 \
    --confidence_weighting \     # Phase 2
    --disagreement_mining \      # Phase 3
    --soft_labels \              # Phase 4
    --percentile 95 --gamma 2.0 \
    --out pseudo_labels/noisy_classmate_pvt_r9.csv
```

### NC Config (add to training YAML for R10+)

```yaml
training:
  nc_distill_beta: 0.3    # Phase 4: KLD soft distillation weight
  nc_temperature: 2.0     # Phase 4: temperature for soft targets
```

### Key NC Implementation Files

| File | Purpose |
|------|---------|
| `scripts/gen_noisy_classmate_pseudo.py` | Phase 1–4: ensemble blend + confidence + disagreement + soft labels |
| `train_sed_ns.py` | `NCDistillLoss`, `_nc_weight` in `PseudoSoundscapeDataset` |
| `scripts/auto_nc_dual_gpu.sh` | Phase 5: bidirectional B0↔PVT co-evolution, dual GPU |
| `scripts/auto_nc2_dual_gpu.sh` | NC v2: confidence-preserved (no disagreement/KLD) |
| `scripts/batch_export_onnx.sh` | Batch ONNX FP32 + INT8 export for all rounds |
| `scripts/monitor_nc.sh` | Pipeline monitoring |
| `scripts/watchdog_nc.sh` | Auto-recovery monitoring |
| `reports/noisy_classmate_plan.html` | Research plan with theoretical justification |
| `reports/vlom_analysis.tex` | VLOM blend weight statistical analysis |

---

## ONNX Export & Submission

### Export

```bash
python3 scripts/export_sed_to_onnx.py \
    --pt outputs/sed-ns-pvt-20s-r9/fold4_best.pt \
    --out "birdclef-2026/notebook resource/new direction/weights/sed/sed_ns_pvt_r9_fold4.onnx" \
    --backbone pvt_v2_b0
# Produces FP32 ONNX + INT8 quantized
```

### Notebook Config (Kaggle)

```python
SED_CHECKPOINTS = [
    {'name': 'b0_0',  'onnx_path': '.../sed_ns_b0_r12_fold0.onnx',
     'backbone': 'tf_efficientnet_b0.ns_jft_in1k', 'weight': 1.0, 'clip_sec': 20},
    {'name': 'pvt_4', 'onnx_path': '.../sed_ns_pvt_r5_fold4.onnx',
     'backbone': 'pvt_v2_b0', 'weight': 1.0, 'clip_sec': 20},
    {'name': 'b0_3',  'onnx_path': '.../sed_ns_r6_fold3.onnx',
     'backbone': 'tf_efficientnet_b0.ns_jft_in1k', 'weight': 1.0, 'clip_sec': 20},
]
```

### Ensemble Selection Rules (Empirically Validated)

1. **Maximize round diversity**: Models from different NS rounds (e.g., R6 + R12) have lower prediction correlation → better LB
2. **Never use same-round models**: R12f0 + R12f2 (corr=0.987) → worse than R12f0 + R8f3 (corr=0.983)
3. **PVT R5 fold4 is locked**: Only effective PVT configuration; changing fold/round always degrades LB
4. **Architecture diversity**: B0 + PVT > B0 + B0 in ensemble
5. **Val AUC ≠ LB**: Higher individual AUC can mean lower LB (GT Paradox)

---

## LB Score History

| Date | Config | LB | Key Change |
|------|--------|----|-----------|
| Date | Config | LB | Key Change |
|------|--------|----|-----------|
| 2026-04-08 | B0 R12 f0 + PVT R5 f2 + B0 R6 f3, VLOM 0.70/0.30 | **0.943** | Best: fold2 PVT + VLOM weight tuning |
| 2026-04-07 | B0 R12 f0 + PVT R5 f4 + B0 R6 f3 | 0.942 | Max round diversity |
| 2026-04-08 | VLOM 0.75/0.25 | 0.942 | Slightly worse than 0.70 |
| 2026-04-08 | NC PVT R9 f0 replaces PVT slot | 0.941 | NC confidence problem |
| 2026-04-08 | NC PVT R9 f2 replaces PVT slot | 0.941 | NC = 0.941 ceiling |
| 2026-04-08 | NC B0 R12 f0 replaces B0 slot | 0.941 | NC regardless of position |
| 2026-04-07 | B0 R12 f0 + PVT R5 f4 + B0 R8 f3 | 0.941 | fold0 + late round |
| 2026-04-07 | B0 R12 f0 + PVT R5 f4 + B0 R12 f2 | 0.940 | Same-round penalty |
| 2026-04-06 | B0 R10 f2 + PVT R5 f4 + B0 R8 f3 | 0.938 | Same fold upgrade |
| 2026-04-06 | B0 R8 f2 + PVT R5 f4 + B0 R8 f3 | 0.938 | Original baseline |
| 2026-04-06 | B0 R9 f2 + PVT R5 f4 + B0 R8 f3 | 0.937 | R9 fold2 worse |
| 2026-04-06 | B0 R8 f2 + PVT R7 f4 + B0 R8 f3 | 0.937 | PVT upgrade fails |
| 2026-04-06 | B0 R8 f2 + PVT R8 f0 + B0 R8 f3 | 0.934 | PVT fold change fails |
| 2026-04-06 | B0 R5 f2 + PVT R5 f4 + B0 R5 f3 | 0.933 | All R5 worse |

---

## Project Structure

```
BirdClef-2026-Codebase/
├── configs/
│   ├── sed_ns_b0_20s_r{1-15}.yaml      # B0 NS configs
│   ├── sed_ns_pvt_20s_r{1-15}.yaml     # PVT NS configs
│   └── sed_ns_cnxt_20s_r{1-8}.yaml     # ConvNeXt NS configs
│
├── train_sed_ns.py                      # SED training (NS + NC support)
├── scripts/
│   ├── auto_sed_ns_20s_full.sh          # B0 NS R1→R8
│   ├── auto_sed_ns_pvt_20s_r{1-4,5-8}.sh  # PVT NS chains
│   ├── auto_nc_full.sh                  # NC bidirectional pipeline
│   ├── gen_pseudo_ns.py                 # NS pseudo label generation
│   ├── gen_noisy_classmate_pseudo.py    # NC pseudo (Phase 1-4)
│   ├── train_sed_residual_corrector.py  # Residual Corrector
│   ├── export_sed_to_onnx.py           # ONNX + INT8 export
│   ├── optimize_sed_weights.py          # Ensemble weight optimization
│   └── monitor_nc.sh                    # NC pipeline monitor
│
├── outputs/
│   ├── sed-ns-b0-20s-r{1-12}/          # B0 checkpoints + npz
│   ├── sed-ns-pvt-20s-r{1-9+}/         # PVT checkpoints + npz
│   └── logs/                            # Training logs
│
├── pseudo_labels/
│   ├── sed_20s_r{1-11}.csv             # NS pseudo labels
│   ├── sed_20s_pvt_r{1-8}.csv          # PVT NS pseudo labels
│   └── noisy_classmate_*.csv           # NC blended pseudo labels
│
└── reports/
    ├── noisy_classmate_plan.html        # NC research plan
    └── method_paper.tex                 # IEEE paper draft
```

---

## VLOM Ensemble Blend

VLOM (Variance-weighted Log-Odds Mean) blends SED and Perch predictions in logit space.

**Per-class VLOM** (P3 notebook):
- Aves (162 species): SED=0.70, Perch=0.30
- Non-Aves (72 species): SED=0.30, Perch=0.70

**Key finding**: CV-LB inversion — CV favors 0.50/0.50, LB favors 0.70/0.30. Root cause: SED logit magnitude is 2.33× Perch on hidden test but 1.0× on CV.

## NC Confidence Problem

NC v1 models have mean prediction probability = 35% of NS (0.019 vs 0.054). In 3-model ensemble, NC contributes only ~12% signal. Root cause: disagreement mining + KLD loss smooths predictions.

**NC v2** removes disagreement mining and KLD (beta=0.0), uses gamma=3.0 and student-weighted blend (0.3/0.7) to preserve confidence.

## Constraints

- All training: **GPU0 + GPU1** (dual GPU pipeline)
- Only **nohuman models** evaluated/submitted
- **No competitor model weights** — only self-trained models
- Submit threshold: individual SED soundscape val AUC > 0.9193
