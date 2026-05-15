# BirdCLEF 2026 — Noisy Classmate Training Framework

Kaggle competition: multi-label bird/amphibian/insect species classification from 5-second soundscape segments.
**Metric**: Macro-averaged ROC-AUC over 234 species. **Current Best LB: 0.953**

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

## Tucker SED — Distilled Student with Unlabeled NS

Tucker SED is a standalone EfficientNet-B0 model trained via Perch-v2 MSE distillation, optimized to match the inference protocol of the public `bc2026-distilled-sed` notebook (which achieves 0.946 LB on its own).

### Architecture

```
5s audio clip (32kHz, int16 cache)
  → MelSpec(n_mels=128, n_fft=1024, hop=320, fmin=50, fmax=14000, slaney norm)
  → per-sample z-score normalization
  → tf_efficientnet_b0.ns_jft_in1k → GEMFreqPool(p=3) → AttentionSEDHead
  → clip_logit (234-d) + framewise_logit (T×234)
  → inference blend: 0.5 × sigmoid(clip) + 0.5 × sigmoid(fmax(framewise))
  → 5-fold ensemble average
```

Training uses two data streams with MSE distillation from Perch-v2:
```
focal clips (GT labels, train_audio/)     ← 85% of batch
labeled soundscapes (66 files, GT labels) ← 15% of batch
+ MSE distillation loss on Perch teacher probabilities
```

### Tucker NS — Noisy Student on Unlabeled Soundscapes

Tucker NS extends the BASE model with correct NS design: pseudo labels are applied **only to unlabeled soundscapes** (10,592 files). The 66 GT-labeled soundscapes and focal clip GT labels are never modified.

```
Tucker BASE
  │
  ├─ infer unlabeled SS cache (2000 files × 12 windows = 24,000 rows)
  │   └─ tucker_ns_b0_unlabeled_r0.csv   ← BASE predictions (fixed teacher)
  │
  ├─ R1: pseudo = BASE predictions (50/50 blend base/student at first)
  │       train 3-stream:
  │         focal clips 85% (GT)
  │         labeled SC  7.5% (66 files, GT unchanged)
  │         unlabeled SC 7.5% (2000 cached files, pseudo labels)
  │       EMA decay=0.99, early stop patience=3
  │
  ├─ R2: blend = 0.30×BASE + 0.70×R1_student → new pseudo CSV
  │       train (same 3-stream setup)
  │
  └─ R3+: blend = 0.05×BASE + 0.95×R(n-1)_student → pseudo CSV
          train (same 3-stream setup)
```

Pseudo label blend schedule (BASE is always the fixed teacher):

| Round | Base Weight | Student Weight | Notes |
|-------|-------------|----------------|-------|
| R1    | 0.50        | 0.50           | bootstrap from BASE predictions |
| R2    | 0.30        | 0.70           | student gains majority |
| R3+   | 0.05        | 0.95           | near-pure self-training |

Unlabeled soundscape cache preparation (one-time):

```bash
python scripts/cache_unlabeled_ss.py --n 2000
# Outputs: birdclef-2026/unlabeled_ss_cache/*.pt + unlabeled_ss_cache_meta.csv
# 2000 files × 12 windows (5s each) = 24,000 rows; stored as int16 .pt tensors
```

Launch Tucker NS chain:

```bash
GPU=0 START_ROUND=1 END_ROUND=8 \
  nohup bash scripts/auto_tucker_ns.sh > outputs/logs/tucker_ns_b0.log 2>&1 &
```

---

## 0.953 LB — Rank-Blend with Tucker SED

LB 0.953 (2026-05-09) is achieved by rank-normalizing and blending our VLOM pipeline with Tucker SED predictions. Direct linear blending fails because the two pipelines have a 7× probability scale mismatch (Tucker mean=0.010 vs our mean=0.070); rank-percentile converts both to a uniform [0,1] scale before combining.

```
╔══════════════════════════════════════════════════════════════════════╗
║               0.953 LB — Rank-Blend Inference Pipeline               ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ┌─────────────────────────────────────┐                             ║
║  │         OUR VLOM PIPELINE           │                             ║
║  │                                     │                             ║
║  │  B0 R11 fold0 (ONNX)               │                             ║
║  │  PVT R7  fold4 (ONNX)              │                             ║
║  │          ↓                          │                             ║
║  │  VLOM blend (logit space):          │                             ║
║  │    Aves    → 0.70×SED + 0.30×Perch │                             ║
║  │    non-Aves→ 0.30×SED + 0.70×Perch │                             ║
║  │          ↓                          │                             ║
║  │  V17 sharpening  ← applied HERE    │                             ║
║  │  (before rank conversion)           │                             ║
║  └───────────────┬─────────────────────┘                             ║
║                  │                                                    ║
║                  │  percentile-rank                                   ║
║                  ↓                                                    ║
║  our_rank[0,1] ──────────────┐                                        ║
║                              │  weighted sum                          ║
║                              │  OUR_W = 0.60                          ║
║  ┌─────────────────────────┐ │  TUCKER_W = 0.40                       ║
║  │    TUCKER SED (public)  │ │                                        ║
║  │                         │ │                                        ║
║  │  5-fold B0 checkpoints  │ │                                        ║
║  │  (sed_fold[0-4].onnx)   │ │                                        ║
║  │  5s sliding window      │ │                                        ║
║  │  0.5×clip+0.5×fmax      │ │                                        ║
║  └──────────┬──────────────┘ │                                        ║
║             │                │                                        ║
║             │  percentile-rank                                        ║
║             ↓                │                                        ║
║  tucker_rank[0,1] ───────────┘                                        ║
║                              │                                        ║
║                              ↓                                        ║
║                     blended_rank[0,1]                                 ║
║                              │                                        ║
║              post-processing │                                        ║
║              ┌───────────────┼───────────────┐                        ║
║              ↓               ↓               ↓                        ║
║       PROTO_CONT        SED_ONLY        Sonotype mirror               ║
║       (prototype       (rank>0.95      + rare suppress                ║
║        continuity)      boost)                                        ║
║              └───────────────┼───────────────┘                        ║
║                              ↓                                        ║
║                   Final predictions → LB 0.953                       ║
╚══════════════════════════════════════════════════════════════════════╝
```

**Why rank-blend works**: Tucker SED (mean prob=0.010, p99=0.19) vs our VLOM (mean=0.070, p99=0.44) have a 7× scale difference — linear blending would let our pipeline dominate 87% of the signal. Percentile-rank converts both distributions to uniform [0,1], so OUR_W=0.60 actually means 60/40 influence. Per-species rank correlation median=0.35, confirming sufficient diversity between the two pipelines.

**Key parameters (battle-tested, do not change)**:
- `OUR_W=0.60`, Tucker weight=0.40
- `FAKE_ONLY_THR=0.50, SED_LOW_THR=0.05, FAKE_ONLY_BLEND=0.08`
- PROTO_CONT: `RADIUS=3, DF=2.0, SCALE=1.20, RANK_THR=0.88, LOCAL=0.75, LOW=0.12, BLEND=0.15`
- SED_ONLY: `RANK=0.95, LOW=0.80, BLEND=0.12`
- Sharpening is applied **before** rank conversion (v1 ordering — confirmed better)

**Notebook**: `birdclef-2026/notebook resource/new direction/notebooks/test-model-family-onnx-perch-tucker-rankblend.ipynb`

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
| 2026-05-09 | VLOM + Tucker SED rank-blend (OUR_W=0.60) | **0.953** | Rank-blend normalizes 7× scale gap; Tucker SED diversity |
| 2026-04-16 | B0 R11 f0 + PVT R7 f4, Perch-style post-proc, top_k=2 | 0.949 | top_k=2, Perch-style SED smoothing |
| 2026-04-10 | B0 R12 f0 + PVT R5 f2 + B0 R6 f3, per-class VLOM | **0.944** | Per-class VLOM: Aves 0.7/0.3, non-Aves 0.3/0.7 |
| 2026-04-08 | B0 R12 f0 + PVT R5 f2 + B0 R6 f3, VLOM 0.70/0.30 | 0.943 | Uniform VLOM |
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

NC v1 models have mean prediction probability = 35% of NS (0.019 vs 0.054). In 3-model ensemble, NC contributes only ~12% signal. Root cause: ensemble pseudo label blending inherently produces smoother targets.

**NC v2** (removed disagreement/KLD) showed same confidence (82.9% vs 83.1%), confirming the problem is in Phase 1/2 blending, not Phase 3/4.

## NC v3 — 3-Architecture Co-evolution

New approach: 3 completely different architectures for maximum ensemble diversity.

| Architecture | timm name | Params | CPU Speed |
|-------------|-----------|--------|-----------|
| ConvNeXt-Femto | `convnext_femto.d1_in1k` | 4.83M | 11.2ms |
| FastViT-T8 | `fastvit_t8.apple_dist_in1k` | 3.26M | 21.9ms |
| RegNetY-008 | `regnety_008.pycls_in1k` | 5.49M | 16.4ms |

Pipeline: B0+PVT as teachers (Phase 0) → 5-model blend (Phase 1) → 3-way NC co-evolution (Phase 2+)

## Constraints

- All training: **GPU0 + GPU1** (dual GPU pipeline)
- Only **nohuman models** evaluated/submitted
- Competitor weights allowed for submission ensemble, KD teacher, and as pretraining starting points (policy from 2026-03-30)
- Submit threshold: individual SED soundscape val AUC > 0.9193
