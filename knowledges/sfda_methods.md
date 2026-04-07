# ⚠️ DEPRECATED — SFDA does NOT apply to BirdCLEF 2026

**Deprecated 2026-03-20. See `domain_generalization.md` instead.**

SFDA requires unlabeled target data at training/test time.
`test_soundscapes/` contains only a readme.txt — no audio available.
We have no target test data to adapt to. Domain Generalization is the correct framing.

---

# Source-Free Domain Adaptation for BirdCLEF 2026 (ARCHIVED)

**Written 2026-03-20. Based on literature survey + BirdCLEF 2026 dataset analysis.**

---

## 1. Domain Gap Analysis

### Source Domain: train_audio
- 35,549 clips, 206 species
- **Global** lat/lon (-54.9°N to 69.6°N, -159.7°W to 175.3°E)
- Collections: XenoCanto (23,043) + iNat (12,506)
- Mostly Aves (34,799/35,549)
- **Single-species per clip**, curated focal recordings, directional microphone

### Target Domain: train_soundscapes + test_soundscapes
- **Fixed location**: Pantanal, Mato Grosso do Sul, Brazil (lat -16.5 to -21.6, lon -55.9 to -57.6)
- 9 ARU deployment sites (S01–S23), 2021–2025
- 10,658 soundscape files total, 5-second windows
- **Multi-species per window**, passive acoustic monitoring, omnidirectional microphone
- 1,478 labeled clips (train_soundscapes_labels.csv)
- Test soundscapes: same Pantanal location (confirmed by BC2026_Test_*_S05_* filenames)

### Nature of Domain Gap
| Axis | Source | Target |
|------|--------|--------|
| Geography | Global | Pantanal, Brazil only |
| Recording type | Focal (directional) | Soundscape (omnidirectional) |
| Species density | 1 per clip | Multiple per 5s window |
| Noise level | Low | High (rain, insects, amphibians) |
| Label type | Primary species certain | Weak, multi-label |
| Dataset size | 35K clips | 10.6K clips (1.4K labeled) |

**Key insight**: This is **semi-supervised domain adaptation**, not pure SFDA, because we have
1,478 labeled target clips (train_soundscapes). We have access to labeled + unlabeled target data.

Perch is already partially adapted (trained globally), but the MLP probe was fit on source domain.

---

## 2. SFDA & TTA Methods Applicable Here

### 2.1 TENT — Fully Test-Time Adaptation by Entropy Minimization
- **Paper**: Wang et al., ICLR 2021 (spotlight)
- **Method**: During inference, update BN affine params (γ, β) by minimizing prediction entropy H(p) on each batch
- **Mechanism**: Entropy is a surrogate for error; confident = correct. Batch-level update only.
- **Pros**: Trivial to implement, no source data needed, online
- **Cons**: Can collapse in multi-label settings; batch statistics unstable for small batches
- **BirdCLEF adaptation**: Apply to MLP probe — update LayerNorm affine params on soundscape batches before eval

### 2.2 SHOT — Source Hypothesis Transfer
- **Paper**: Liang et al., ICML 2020; TPAMI 2021 (SHOT+)
- **Method**: Freeze classifier head (hypothesis). Adapt feature extractor via:
  1. **Information Maximization (IM)**: maximize mutual information I(X; Y) = H(p̄) - E[H(p)] where p̄ is marginal prediction
  2. **Pseudo-label self-training**: cluster target features, assign pseudo labels, fine-tune
- **Key insight**: IM = maximize diversity (H(p̄) high) + maximize confidence (E[H(p)] low)
- **BirdCLEF adaptation**: Freeze Perch backbone, apply IM loss to MLP on unlabeled soundscapes
- **Multi-label caution**: IM designed for single-label. For multi-label, use per-class IM or just entropy regularization.

### 2.3 NRC — Neighborhood Reciprocal Clustering
- **Paper**: Yang et al., NeurIPS 2021; Trust Your Good Friends, TPAMI 2023
- **Method**: Build kNN graph on target embeddings. Assign higher weight to **reciprocal neighbors** (k-NN of k-NN). Self-regularization loss penalizes disagreement with neighborhood.
- **Intuition**: If A is in B's k-NN and B is in A's k-NN → mutual agreement → reliable pseudo label
- **BirdCLEF adaptation**: Build kNN on Perch embeddings of all 10,658 soundscape clips. Use neighborhood consensus to refine probe predictions.

### 2.4 AdaContrast — Contrastive Test-Time Adaptation
- **Paper**: Chen et al., CVPR 2022
- **Method**: Contrastive learning on augmented target views during test time. Maintains memory bank of target representations. Online pseudo-label refinement.
- **BirdCLEF adaptation**: Augment soundscape clips (time-shift, gain jitter, background swap), contrastive alignment

### 2.5 LaplacianShot — Transductive Inference via Graph
- **Paper**: Ziko et al., ICML 2020
- **Method**: Given all test embeddings, solve joint energy minimization:
  E = Σ_i H(p_i) + λ Σ_{i,j∈kNN} ||p_i - p_j||²
  First term → confidence. Second term → local smoothness.
- **Pros**: Batch-transductive; test clips in same soundscape are naturally co-occurring
- **BirdCLEF adaptation**: Already implemented in perch-probe-v3-laplacian config! Run on full soundscape set.

### 2.6 FeCAM — Mahalanobis NCM
- **Paper**: Goswami et al., NeurIPS 2023
- **Method**: Replace Euclidean prototype distance with per-class Mahalanobis using Ledoit-Wolf + Tukey regularization
- **BirdCLEF adaptation**: Fit class covariance on train+soundscape embeddings. Already implemented in perch-probe-v3-fecam config.

### 2.7 Domain-Aware Prototype Shift (New for BirdCLEF)
- **Method**: Blend source prototypes with target prototypes using labeled soundscape data (1,478 clips)
  - μ_c_adapted = α·μ_c_source + (1-α)·μ_c_target
- **BirdCLEF advantage**: We have labeled target data → this is supervised prototype adaptation, not zero-shot
- **Expected gain**: Shifts decision boundary toward Pantanal distribution

---

## 3. BirdCLEF-Specific SFDA Strategies

### Strategy A: Target-Domain PCA Re-fit (Already Done Partially)
- Current: PCA fit on [train, soundscape] embeddings in v3-addtrain-only (OOF 0.8119 vs 0.7515 baseline)
- The +0.060 OOF gain from add_train_clips is essentially **domain-informed feature space adaptation**
- This is the simplest and most validated SFDA technique we have

### Strategy B: Unlabeled Soundscape Entropy Minimization
- Take the trained v3-addtrain-only probe (pca_params.npz + probe_head.pt)
- Fine-tune MLP head only on all 10,658 unlabeled soundscape clips using entropy minimization (EM) loss
- Multi-label EM: use per-class binary entropy H(p) = -p·log(p) - (1-p)·log(1-p), minimize average
- Validate on 1,478 labeled soundscape clips (OOF AUC)
- Risk: can increase false positives; use small LR (1e-5) + few steps (5-10 epochs)

### Strategy C: Labeled Soundscape Fine-Tuning
- NOT SFDA, but highly applicable: fine-tune probe on labeled soundscape data
- Split 1,478 clips by site for site-stratified validation
- Multi-label supervised training on real Pantanal recordings
- This is the most direct domain adaptation and likely highest impact

### Strategy D: Pseudo-Label Loop on Unlabeled Soundscapes
- Step 1: Get predictions from probe on all soundscape clips
- Step 2: High-confidence predictions (p > 0.8) → pseudo labels
- Step 3: Re-train probe on train_audio + labeled soundscapes + pseudo-labeled soundscapes
- Step 4: Repeat (3 rounds expected to stabilize)
- Similar to what we do for SED but for the Perch probe

### Strategy E: SHOT-style Information Maximization
- Objective: L_IM = -H(p̄) + E[H(p)] (maximize diversity, minimize per-sample entropy)
- Apply to MLP probe with frozen PCA transform
- For 206-class multi-label: treat each class independently, or use mean-field approximation
- Expected: +1-3pp OOF AUC over entropy-only minimization

### Strategy F: Site-Conditioned Prototype Adaptation
- Pantanal has 9 sites with different acoustic ecologies
- Compute site-specific embedding centroids from labeled soundscape data
- Shift prototypes based on site metadata available in train_soundscapes_labels.csv
- During test inference, use site ID (from filename) to select site-adapted prototype

---

## 4. Practical Priority Ranking for BirdCLEF 2026

| Priority | Method | Expected Gain | Implementation Effort | Risk |
|----------|--------|---------------|----------------------|------|
| **1** | Strategy C: Labeled soundscape fine-tune | +5-10pp OOF | Medium | Low |
| **2** | Strategy D: Pseudo-label loop | +2-5pp OOF | Medium | Medium |
| **3** | Strategy A + LaplacianShot (already coded) | +1-3pp OOF | Low (already coded) | Low |
| **4** | Strategy B: Entropy minimization | +1-3pp OOF | Low | Medium (collapse) |
| **5** | Strategy F: Site adaptation | +1-2pp OOF | Medium | Low |
| **6** | Strategy E: SHOT IM | +1-2pp OOF | Medium | High (multi-label) |

---

## 5. Recommended Experiment Sequence

### Exp SFDA-1: Soundscape Probe Fine-Tuning (Supervised DA)
```yaml
# perch_probe_v3_soundscape_ft.yaml
data:
  add_train_clips: true
  add_soundscape_clips: true   # NEW: include 1478 labeled soundscape clips in training
  soundscape_weight: 3.0       # oversample soundscape (target domain)
model:
  pca_dim: 128
  pca_fit_splits: [train, soundscape]
  use_hidden: true
  hidden_dim: 512
training:
  epochs: 150
```

### Exp SFDA-2: Entropy Minimization Fine-Tune
```python
# After training SFDA-1:
# Load probe_head.pt, freeze PCA
# For each batch of UNLABELED soundscapes:
#   loss = mean(per_class_binary_entropy(sigmoid(logits)))
#   optimizer.step()  # LR=1e-5, max 10 epochs
```

### Exp SFDA-3: Site-Stratified OOF Eval
- Split 1,478 labeled clips by site ID (S01, S05, S09, S17, S21, S23)
- Train-test split: leave one site out
- Measures true generalization to new deployment sites (matches test scenario)

---

## 6. Key Papers & Links
- TENT: https://arxiv.org/abs/2006.10726
- SHOT: https://arxiv.org/abs/2002.08546
- NRC: https://arxiv.org/abs/2110.04202
- LaplacianShot: ICML 2020 (Ziko et al.)
- FeCAM: NeurIPS 2023 (Goswami et al.)
- Mind the Domain Gap (bioacoustics): https://arxiv.org/html/2403.18638v1
- Domain-Invariant Bird Sounds: https://arxiv.org/html/2409.08589
- BirdCLEF 2024 domain adaptation overview: https://hal.science/hal-05183115

---

## 7. BirdCLEF 2026 Specific Realization

**The most important realization**: We are NOT doing pure source-free adaptation.
We have:
- 1,478 labeled target clips → use for supervised fine-tuning (Strategy C, highest priority)
- 10,658 unlabeled target clips → use for entropy min / pseudo-label (Strategies B, D)
- Same Pantanal location for train+test soundscapes → site adaptation is valid (Strategy F)

The add_train_clips=True trick in v3-addtrain-only (+0.060 OOF AUC) is already a form of
covariate shift correction: fitting PCA on both source and target jointly removes source-domain
PCA artifacts and captures target geometry.

**Next highest-leverage move**: Include 1,478 labeled soundscape clips directly in probe training.
This is effectively domain adaptation with labeled target data — maximally supervised.
