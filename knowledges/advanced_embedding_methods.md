# Advanced Embedding Methods for Few-Shot Bioacoustic Classification

## System Context
- Perch 1536-dim embeddings, 708 labeled soundscape clips, 234 classes
- Current: L2→center→PCA(128,whiten)→LogReg+Proto+TipAdapter = **0.915 LB**
- Each soundscape file = 12 clips; ~186 fully-labeled files
- Positives per class: 0–200 (most rare species: 2–20)

---

## Method 1: Distribution Calibration (Yang et al., ICLR 2021)
**Paper**: "Free Lunch for Few-Shot Learning: Distribution Calibration"

### Core Idea
Rare classes have few positive samples → biased, high-variance prototypes.
Fix: **borrow variance statistics from similar base classes** to calibrate the rare class distribution, then sample synthetic embeddings for augmented LogReg training.

### Algorithm
```python
def calibrate_rare_class(emb_rare_pos, all_prototypes, all_variances, K=2, alpha=0.5):
    """
    emb_rare_pos: (n_pos, D) — rare class positive embeddings  (n_pos < 8)
    all_prototypes: (C, D) — all class mean embeddings
    all_variances: (C, D) — per-class per-dim variance
    K: number of nearest base classes to borrow from
    alpha: how much variance to borrow (0=rare class only, 1=base class only)
    Returns: (N_aug, D) — augmented embeddings sampled from calibrated Gaussian
    """
    rare_proto = emb_rare_pos.mean(axis=0)  # (D,)

    # Find K nearest base classes by prototype similarity
    dists = np.linalg.norm(all_prototypes - rare_proto, axis=1)
    nn_idx = np.argsort(dists)[1:K+1]  # top-K neighbors (exclude self)

    # Calibrated variance: blend rare-class variance with neighbor variance
    if len(emb_rare_pos) > 1:
        rare_var = emb_rare_pos.var(axis=0)
    else:
        rare_var = np.zeros(emb_rare_pos.shape[1])

    neighbor_var = all_variances[nn_idx].mean(axis=0)  # (D,)
    calibrated_var = (1 - alpha) * rare_var + alpha * neighbor_var

    # Sample N_aug synthetic embeddings from N(rare_proto, calibrated_var)
    N_aug = 50
    synthetic = np.random.randn(N_aug, len(rare_proto)) * np.sqrt(calibrated_var) + rare_proto
    return synthetic.astype(np.float32)

# Usage in LogReg training for rare species (n_pos < 8):
for cls in rare_classes:
    emb_pos = emb_train[y_train[:, cls] == 1]
    emb_neg = emb_train[y_train[:, cls] == 0][:200]  # sample negatives
    aug_pos = calibrate_rare_class(emb_pos, all_prototypes, all_variances)
    X = np.vstack([emb_pos, aug_pos, emb_neg])
    y = np.hstack([np.ones(len(emb_pos)+len(aug_pos)), np.zeros(len(emb_neg))])
    clf.fit(X, y)
```

### Expected Gain
- Targets the ~80 rare classes with <8 positives (currently no LogReg, only proto)
- Even if only +0.002 AUC on rare classes → significant LB gain (rare classes count equally)
- **Zero extra compute at inference** — only changes probe_cache precomputation

### Integration Notes
- Apply in `precompute_probe_cache.py`: for classes with 2 ≤ n_pos < 8, add calibration then fit LogReg
- Need to first compute all class prototypes & variances from train fold
- Borrow only from **in-distribution** base classes (use cosine similarity not Euclidean for 1536-dim)

---

## Method 2: TIM (Transductive Information Maximization)
**Paper**: Malik et al., NeurIPS 2020 — "Towards Realistic Few-Shot Classification"

### Core Idea
At inference time, use the **query set structure** (12 clips from one file) to refine predictions.
Maximize mutual information between queries and labels:
```
maximize:  H(Ȳ) - H(Y|X)     [high class diversity, low per-clip uncertainty]
```
This naturally: 1) encourages all species to be used (marginal entropy high), 2) makes each clip's prediction sharp (conditional entropy low).

### Algorithm (soft-EM variant)
```python
def tim_inference(base_logits, emb_query, prototype, lam=0.1, n_iter=10, tau=0.1):
    """
    base_logits: (12, C) — initial Perch/LogReg scores for file
    emb_query: (12, D) — normalized embeddings of query clips
    prototype: (C, D) — class prototypes from support
    Returns: (12, C) — refined scores
    """
    # Start from base predictions
    q = softmax(base_logits / tau, axis=1)  # (12, C)

    for _ in range(n_iter):
        # --- E-step: compute soft marginals ---
        q_mean = q.mean(axis=0)  # (C,) — marginal distribution over classes

        # --- M-step: maximize MI objective ---
        # New logits = base + lambda * log(q_mean) [prior from marginal]
        # Encourages uniform class usage across 12 clips
        adjusted_logits = base_logits + lam * np.log(q_mean + 1e-9)[None, :]
        q = softmax(adjusted_logits / tau, axis=1)

    return q  # (12, C)
```

### BirdCLEF Adaptation
- Apply per soundscape file (12 clips)
- `base_logits` = combined LogReg+Proto+TipAdapter scores before temporal smoothing
- `lam=0.1` (conservative — we have strong prior from Perch)
- Especially helps files with mixed species composition (most soundscapes)

### Key Benefit
- **Transductive**: uses 12 clips jointly, not clip-by-clip
- Complementary to temporal smoothing (different mechanism)
- No stored cache needed — pure inference-time optimization

### Expected Gain
+2-4% on soundscape files; especially strong for species with 3-6 clips presence
(TIM prevents overconfident predictions on one species)

---

## Method 3: Embedding Subspace Ensemble (multi-scale PCA)

### Core Idea
Different PCA subspace sizes capture different granularity of embedding structure:
- PCA(32): captures only the dominant variance modes → strong for common species
- PCA(128): balanced (current)
- PCA(512): captures fine-grained distinctions → better for rare species

Average predictions across subspaces = implicit regularization.

### Algorithm
```python
# At precompute time: fit 3 PCA models
pca_32  = PCA(n_components=32,  whiten=True).fit(emb_centered)
pca_128 = PCA(n_components=128, whiten=True).fit(emb_centered)  # current
pca_512 = PCA(n_components=512, whiten=True).fit(emb_centered)

# At inference time:
Z_32  = pca_32.transform(emb_query)
Z_128 = pca_128.transform(emb_query)
Z_512 = pca_512.transform(emb_query)

scores_32  = run_probe(Z_32,  probe_models_32)
scores_128 = run_probe(Z_128, probe_models_128)  # current
scores_512 = run_probe(Z_512, probe_models_512)

final_scores = (scores_32 + scores_128 + scores_512) / 3
```

### Practical Notes
- Probe_cache size 3×: ~70 MB (manageable in Kaggle)
- Each PCA+LogReg is independent → can train in parallel
- Alternative: use LogReg on concatenated [Z_32; Z_128; Z_512] (520 dims) — but larger model
- **Best approach**: average predictions (avoids overfitting concatenation)

### Expected Gain
+0.003-0.008 LB if subspaces are complementary. Low risk experiment.

---

## Method 4: Mahalanobis Distance Prototype (Class-Conditional Gaussian)

### Core Idea
Standard prototype uses Euclidean distance in PCA space.
Mahalanobis accounts for different covariance structures per class:
```
d(x, cls) = (x - μ_cls)^T Σ_cls^{-1} (x - μ_cls)
```

### Why Better Than Cosine
- Different bird species occupy differently shaped regions in embedding space
- Cosine assumes spherical clusters; Mahalanobis captures ellipsoidal clusters
- Especially useful when a class has multiple acoustic variants (male/female calls)

### Algorithm
```python
def mahalanobis_prototype_score(emb_query, emb_pos, shared_cov=True):
    """
    emb_query: (12, D) — query embeddings
    emb_pos: (n, D) — positive support embeddings for this class
    shared_cov: use shared covariance (more stable for rare classes)
    """
    mu = emb_pos.mean(axis=0)  # (D,)

    if shared_cov:
        # Use global covariance (estimated from all classes)
        # This is pre-computed from all 708 clips in train fold
        Sigma_inv = global_sigma_inv  # (D, D)
    else:
        Sigma = np.cov(emb_pos.T) + 1e-4 * np.eye(D)
        Sigma_inv = np.linalg.inv(Sigma)

    diff = emb_query - mu[None, :]  # (12, D)
    dist = np.einsum('nd,dd,nd->n', diff, Sigma_inv, diff)  # (12,)
    return -dist  # negative distance as score

# Practical: shared covariance (1 global Σ^{-1} pre-computed) is efficient
# Per-class covariance needs n_pos >> D (impossible for rare species)
```

### Integration
- Replaces cosine similarity in `proto_prototypes` branch
- Pre-compute `global_sigma_inv` in probe_cache (1536×1536 matrix, 9.4 MB)
- Apply only in PCA space (128×128 matrix, much smaller)

---

## Method 5: APE-style Per-Class Feature Weighting

### Core Idea (already in tip_adapter_family.md)
For each class, weight the 128 PCA dimensions by their discriminative power:
```
w_dim[cls] = (μ_pos[dim] - μ_neg[dim])^2 / (σ_pos[dim]^2 + σ_neg[dim]^2 + ε)
```
This is Fisher's criterion per dimension. High weight = that dimension separates class from others.

### Integration
```python
# Pre-compute per-class dimension weights (128-dim, one per class)
for cls in classes:
    pos_emb = Z_train[y_train[:, cls] == 1]  # (n_pos, 128)
    neg_emb = Z_train[y_train[:, cls] == 0]  # (n_neg, 128)
    mu_pos, mu_neg = pos_emb.mean(0), neg_emb.mean(0)
    var_pos = pos_emb.var(0) + 1e-6
    var_neg = neg_emb.var(0) + 1e-6
    w = (mu_pos - mu_neg)**2 / (var_pos + var_neg)
    w /= w.sum()  # normalize
    ape_weights[cls] = w.astype(np.float32)

# At inference:
def ape_proto_score(Z_query, prototype, ape_weights, cls):
    w = ape_weights[cls]  # (128,)
    sim = (Z_query * w[None, :]) @ prototype  # weighted cosine
    return sim
```

### Expected Gain
+0.002-0.005; acts as soft feature selection per class. Low-cost addition to proto.

---

## Experiment Priority Queue

| # | Method | Expected Gain | Effort | Status |
|---|--------|--------------|--------|--------|
| 1 | Distribution Calibration (rare LogReg) | **+0.005-0.015** | Medium | Not implemented |
| 2 | TIM per-file inference | **+0.003-0.008** | Low | Not implemented |
| 3 | Subspace Ensemble (32+128+512) | +0.003-0.008 | Medium | Not implemented |
| 4 | APE feature weighting | +0.002-0.005 | Low | Not implemented |
| 5 | Mahalanobis prototype | +0.002-0.005 | Medium | Not implemented |
| 6 | LP++ init | +0.002-0.004 | Low | In knowledge base |

### Recommended Next Steps

**Step 1 (today)**: Add TIM to inference notebook cell 11 — pure inference change, no cache update needed.

**Step 2 (next cache version)**: Implement Distribution Calibration in `precompute_probe_cache.py` for rare classes (2 ≤ n_pos < 8).

**Step 3**: Add subspace ensemble (fit 3 PCAs in cache, average at inference).

---

## Key Insight: Why PCA Works So Well

PCA on Perch embeddings achieves impressive results because:
1. **Perch is already a specialized audio model** — its 1536 dims are not random; they encode meaningful acoustic structure
2. **Whitening** removes correlation between dimensions → more uniform feature importance → better for cosine/Euclidean similarity
3. **L2-norm before PCA** puts all embeddings on the unit sphere → centering then projects onto the most variable directions ON THE SPHERE (not just globally)
4. **128 dims captures ~85-90% of variance** in Perch embeddings for bioacoustic data — not much is lost

The surprise is that **PCA is nearly as powerful as ZCA** for this task, because the 1536 dims already have low effective rank due to Perch's inductive bias toward bird calls.

---

## References
- Distribution Calibration: Yang et al., ICLR 2021 (arXiv 2101.06395)
- TIM: Malik et al., NeurIPS 2020 (arXiv 2003.11533)
- APE: Zhu et al., ICCV 2023 (arXiv 2304.01195)
- Mahalanobis few-shot: Fort (2017), Ren et al. ICLR 2018
- Embedding ensembles: Ho & Salimans 2020 (diverse subspaces)
