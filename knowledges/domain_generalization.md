# Domain Generalization for BirdCLEF 2026

**Written 2026-03-20. Replaces earlier SFDA framing (which was wrong).**

---

## Why Domain Generalization, NOT SFDA

**SFDA (Source-Free Domain Adaptation)** requires unlabeled target data to adapt to at
training or test time. We do not have this:
- `test_soundscapes/` contains only a `readme.txt` — no audio provided
- The actual test soundscapes will only be seen at Kaggle inference time, inside a notebook
  with no opportunity for iterative adaptation

**Domain Generalization (DG)** trains a model to generalize to *unseen* target domains
using only source domain data and whatever labeled target-domain proxy we have
(train_soundscapes_labels: 1,478 clips). No test-domain access required.

**Correct framing for BirdCLEF 2026:**
- Source domains: train_audio (global XC/iNat focal recordings, many recording conditions)
- Proxy target: train_soundscapes (labeled Pantanal ARU clips) — use to evaluate & tune
- True target: test_soundscapes (unseen, Pantanal ARU, same distribution as train_soundscapes)
- Goal: train a model robust to the focal→soundscape distribution shift, generalizing at test time
  without any adaptation

---

## Domain Gap Axes (BirdCLEF 2026)

| Axis | Source | Target | Implication for DG |
|------|--------|--------|-------------------|
| Geography | Global | Pantanal, Brazil | Species prior shift → need calibrated priors |
| Device | Directional mic (XC recorder) | Omnidirectional ARU | Frequency response shift → Freq-MixStyle |
| Recording type | Focal (1 species intended) | Passive (multi-species) | Label structure mismatch → soundscape simulation |
| SNR | High (deliberate recording) | Low (rain, insects, frogs) | Noise robustness → background augmentation |
| Label granularity | Single primary label | Multi-label weak annotations | Multi-label training crucial |

---

## Method 1: Freq-MixStyle (DG-SED, DCASE 2024)

**Paper**: DG-SED: Domain Generalization for Sound Event Detection with Heterogeneous Training Data
(arXiv 2407.03654, DCASE 2024)
**Based on**: MixStyle (Zhou et al., ICLR 2021)

**Core idea**: During training, randomly mix the frequency-wise statistics (mean μ and std σ)
of mel-spectrograms between two training samples. This synthesizes new virtual domains in
the frequency dimension — simulating the response differences between different microphones/devices.

**Freq-MixStyle on spectrogram X:**
```
μ_x = mean(X, dim=time)        # shape: [freq_bins]
σ_x = std(X, dim=time)         # shape: [freq_bins]

# Sample a random partner from the batch
μ_y, σ_y = stats of partner sample

# Mix with random lambda ~ Beta(0.1, 0.1)
λ ~ Beta(α, α), α=0.1
μ_mix = λ * μ_x + (1-λ) * μ_y
σ_mix = λ * σ_x + (1-λ) * σ_y

# Re-normalize X with mixed stats
X_mix = (X - μ_x) / σ_x * σ_mix + μ_mix
```

**Reported gain**: Joint score 1.270 → 1.343 on DCASE 2024 Task 4 (Freq-MixStyle + FDY conv)

**BirdCLEF application (SED model)**:
- Insert Freq-MixStyle as augmentation in `train_sed.py` mel-spectrogram pipeline
- Apply stochastically (p=0.5) during training only
- Simulates different ARU frequency responses — directly addresses device shift

**BirdCLEF application (Perch probe)**:
- On PCA-projected embeddings, mix the channel-wise statistics between training samples
- Simpler than spectrogram-level mixing, no code change to Perch needed

---

## Method 2: SWAD — Dense Checkpoint Averaging (NeurIPS 2021)

**Paper**: "SWAD: Domain Generalization by Seeking Flat Minima" (Cha et al., NeurIPS 2021)
arXiv: 2102.08604

**Core idea**: Standard SWA averages checkpoints at fixed intervals (sparse). SWAD averages
*densely* — every epoch after a warmup period — using an overfit-aware stopping criterion.
This finds flat minima in the loss landscape, which correlate with better OOD generalization.

**Algorithm**:
1. Train normally for `start_step` iterations (warmup)
2. Start densely sampling checkpoints: average weights every K steps
3. Stop averaging when OOD val loss starts rising (overfit detection)
4. Use the averaged model as final model

**Reported gain**: +1.6% avg accuracy on 5 DG benchmarks (PACS, VLCS, OfficeHome, etc.)

**BirdCLEF application**:
- We already do model soup (sparse checkpoint averaging) → this IS essentially SWAD
- The key SWAD improvement: average *every epoch* starting from the first good epoch,
  not just at fixed intervals
- Our SED model soup (checkpoint every epoch) is already close to SWAD behavior
- Key insight: the reason model soup helps is because it finds flatter minima → DG benefit

---

## Method 3: WiSE-FT — Weight-Space Ensemble Fine-Tuning (CVPR 2022)

**Paper**: "Robust Fine-Tuning of Zero-Shot Models" (Wortsman et al., CVPR 2022)
arXiv: 2109.01903

**Core idea**: After fine-tuning a pre-trained (zero-shot) model on target task:
1. Keep the original zero-shot weights θ₀ (Perch base model)
2. Keep the fine-tuned weights θ_ft (our trained MLP probe)
3. Linearly interpolate: θ_WiSE = (1-α)·θ₀ + α·θ_ft

**Why this works**: Fine-tuning on task-specific data improves task accuracy but can
"forget" general representations that help OOD. Interpolating back toward the original
weights recovers robustness without losing most of the task-specific gains.

**Reported gain**: +4–6pp on ImageNet distribution shifts vs. pure fine-tuning

**BirdCLEF application (Perch probe)**:
- θ₀: Perch TFLite/PyTorch zero-shot linear probe (baseline)
- θ_ft: Our trained MLP head (v3-addtrain-only, OOF 0.8119)
- θ_WiSE: Interpolation → should improve soundscape generalization
- In practice: blend predictions (output-space WiSE) or blend weights (parameter-space WiSE)
- Output-space WiSE already done in ensemble (TFLite 30% + MLP 70%) — this IS WiSE-FT!
- The 70%/30% blend is the α=0.70 WiSE-FT operating point

---

## Method 4: ProtoCLR — Prototype Contrastive Learning (arXiv 2409.08589)

**Paper**: "Domain-Invariant Representation Learning of Bird Sounds" (2024)

**Core idea**: Use supervised contrastive learning where each class is represented by a
prototype (mean embedding). Same-class examples from *different domains* are pulled together,
while different-class examples are pushed apart. This enforces domain-invariant class representations.

**ProtoCLR loss** (computationally cheaper than SupCon):
```
L_ProtoCLR = -log[ exp(z_i · p_c / τ) / Σ_k exp(z_i · p_k / τ) ]
where p_c = prototype of class c, computed on the fly
```

**BirdCLEF application (SED model)**:
- Add a ProtoCLR auxiliary loss on frame-level features during SED training
- Use train_audio clips (source) and train_soundscapes clips (target proxy) as separate domains
- Pull same-species embeddings from both domains toward shared prototypes

**BirdCLEF application (Perch probe)**:
- Replace ASL loss with ASL + ProtoCLR combined
- Forces probe to learn representations where focal and soundscape embeddings cluster together

---

## Method 5: Soundscape Simulation Augmentation

**Core idea**: At training time, artificially create multi-species soundscape-like mixtures
from focal source recordings to close the recording-type gap.

**Steps**:
1. Randomly select 2–4 focal clips from different species
2. Mix them with random gains (simulate simultaneous calling)
3. Optionally add background noise from `train_soundscapes` (unlabeled audio)
4. Label: multi-hot union of all mixed species

**Relationship to existing techniques**:
- ClipMix (already implemented) is a simpler 2-clip version
- This extends to 3–4 clips with more realistic gain distributions
- v28/v30 formula already has ClipMix — this would be an upgrade

**Implementation note**: Requires changing the data collation in `mel_dataset.py`.
The `clip_mix` flag already controls this; extending to multi-clip is a small change.

---

## Method 6: GroupDRO — Worst-Case Group Optimization

**Paper**: "Distributionally Robust Neural Networks" (Sagawa et al., ICLR 2020)

**Core idea**: Treat each recording site (S01, S05, S09...) as a separate "group".
Instead of minimizing average loss, minimize the *worst-group* loss:

```
L_DRO = max_{g ∈ groups} 𝔼[L | group=g]
```

This forces the model to perform well even on the hardest recording site — which is the true
test-time generalization requirement (test soundscapes include multiple sites).

**BirdCLEF application (Perch probe)**:
- 9 soundscape sites available in train_soundscapes_labels.csv
- Use site ID as group label
- Train probe to minimize worst-site AUC instead of average OOF AUC

---

## Method 7: Geographic Logit Masking (BirdSet 2024)

**Paper**: "BirdSet: A Dataset and Benchmark for Classification in Avian Bioacoustics"
(arXiv 2403.10380, INTERSPEECH/ICASSP 2024)

**Core idea**: At inference time, zero out logits for species that are geographically impossible
at the test location. For BirdCLEF 2026, the test soundscapes are all Pantanal, Brazil.
A species known to never occur in the Pantanal ecoregion can have its logit set to -∞.

**BirdCLEF application**:
- `scripts/build_geo_mask.py` already exists in this repo — generates a binary mask over 234 species
- Apply mask to ensemble predictions before padded cmap scoring
- Zero-cost at training time; tiny inference overhead
- Expected: +0.010–0.030 on Pantanal-specific evaluation (shifts probability mass to real species)

**Status**: Infrastructure already built. Just needs integration into submission notebook.

---

## Method 8: Multi-Label Mixup / Extended ClipMix (BirdSet 2024)

**Core idea**: Extend existing 2-clip ClipMix to 3–4 clips, simulating real Pantanal soundscape
density (2–4 co-calling species per 5s window). Labels are multi-hot union with soft weights
proportional to mixing coefficient.

**BirdCLEF application (SED model)**:
- Modify `mel_dataset.py` `clip_mix` logic: draw 3–4 source clips instead of 2
- Sample mixing gains from Dirichlet(1,1,1,1) for realistic amplitude variation
- Assign label weight proportional to each clip's gain
- Directly simulates the Pantanal "multi-species chorus" recording condition

**Expected gain**: +0.005–0.010 soundscape val AUC (more realistic soundscape simulation)

---

## Method 9: ProtoCLR — Prototype Contrastive Learning (arXiv 2409.08589)

**Paper**: "Domain-Invariant Representation Learning of Bird Sounds" (2024–2026, v8 Jan 2026)

**Core Technique**:
Replace O(N²) SupCon loss with O(N×C) prototype comparisons:
```
L_ProtoCLR = -log[ exp(z_i · p_c / τ) / Σ_k exp(z_i · p_k / τ) ]
p_c = running mean embedding for class c (updated each batch)
```

Since prototype = average over focal + soundscape clips, the model must discard
device-specific features to predict the shared prototype → enforces domain invariance.

**Key result**: BIRB few-shot benchmark: ProtoCLR 42.4% vs SupCon 39.5% (5-shot mean).
Directly benchmarked on focal→soundscape generalization.

**BirdCLEF application**:
- Add as auxiliary loss to SED backbone embedding head: `L = L_ASL + λ·L_ProtoCLR`
- Or use to fine-tune Perch probe head (replace ASL with ASL+ProtoCLR)
- Requires only species labels (already available), no domain labels needed

**Implementation**: ~150 lines PyTorch. Difficulty: Medium.
**Expected gain**: +0.010–0.025 holdout AUC

---

## Updated Freq-MixStyle Parameters (from Paper)

**Correction**: DCASE 2024 paper uses Beta(**0.6, 0.6**), NOT Beta(0.1, 0.1).
- Beta(0.6, 0.6): U-shaped with modes near 0 and 1 → mostly one domain, occasionally blended
- Beta(0.1, 0.1): Very sharp U-shape → almost always pure one domain
- The 0.6 setting provides more mixing diversity while still allowing pure-domain samples

Update `sed_b0_v31_freqmixstyle.yaml`: `freq_mixstyle_alpha: 0.6`

---

## Priority Ranking for BirdCLEF 2026

| Priority | Method | Where to apply | Expected gain | Effort |
|----------|--------|---------------|---------------|--------|
| **1** | WiSE-FT blend (output-space) | Perch probe + TFLite ensemble | Already doing! Tune α | Done |
| **2** | **Freq-MixStyle** (β=0.6) | SED model mel spectrogram | +1–3pp soundscape val | Easy (~20 lines) |
| **3** | **Geographic logit masking** | Inference post-processing | +1–3pp on Pantanal eval | Easy (script exists) |
| **4** | SWAD / dense soup | SED (already doing) | Already baked in | Done |
| **5** | **Multi-label mixup** (3–4 clips) | SED mel_dataset.py | +0.5–1pp soundscape val | Easy |
| **6** | MixStyle on embeddings | Perch probe PCA space | +1–2pp OOF | Easy (20 lines) |
| **7** | **ProtoCLR** auxiliary loss | SED head or Perch probe | +1–2.5pp holdout | Medium (150 lines) |
| **8** | GroupDRO by site | Perch probe | +0–2pp OOF | Medium |

---

## Designed Experiments

### DG-Probe-1: MixStyle on PCA Embeddings
**Config**: `perch_probe_v4_mixstyle.yaml`
- Apply embedding-space MixStyle during MLP training
- Mix mean/std of PCA-projected embeddings between samples in each batch (p=0.5)
- No code change to Perch backbone needed
- Expected: +1–2pp OOF AUC

### DG-Probe-2: GroupDRO by Soundscape Site
**Config**: `perch_probe_v4_groupdro.yaml`
- Parse site ID from source_file column (S01, S05, etc.)
- Minimize worst-site loss instead of average OOF loss
- Expected: more consistent per-site performance

### DG-Probe-3: WiSE-FT Alpha Sweep
- Already implemented as TFLite 30% / MLP 70% blend
- Sweep α ∈ {0.5, 0.6, 0.7, 0.8, 0.9} on soundscape val
- Pick best α for ensemble notebook

### DG-SED-1: Freq-MixStyle in SED Training
**Config**: `configs/sed_b0_v31_freqmixstyle.yaml`
- Add Freq-MixStyle augmentation to mel spectrogram pipeline in `mel_dataset.py`
- p=0.5, Beta(0.1, 0.1) mixing coefficient
- Expected: +1–3pp soundscape val AUC

### DG-SED-2: Multi-clip Soundscape Simulation
**Config**: `configs/sed_b0_v32_ss_sim.yaml`
- Extend current ClipMix (2-clip) to 3–4 clips per sample
- Multi-hot label union
- Simulate real soundscape density (2–4 species per 5s window)

---

## Key Insight

The current pipeline already implicitly applies several DG techniques:
1. **Joint PCA on train+soundscape**: Feature alignment → similar to CORAL
2. **Model soup / SWAD**: Dense checkpoint averaging → flat minima → OOD robustness
3. **WiSE-FT**: TFLite zero-shot 30% + MLP 70% blend = output-space WiSE-FT at α=0.70
4. **ClipMix**: 2-species mixing → primitive soundscape simulation
5. **Pseudo labels from soundscapes**: Train on proxy target distribution

The most impactful unexplored DG technique remaining: **Freq-MixStyle on the SED spectrogram pipeline** — addresses the device/microphone frequency response gap directly.
