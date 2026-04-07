# Transductive Methods: LaplacianShot & Graph Label Propagation

## Why Transductive Matters for BirdCLEF

Each soundscape file = 12 clips (60s / 5s).
These 12 clips share the **same acoustic environment** (recording site, weather, background noise).
Neighboring clips in time likely share similar species presence.

**Key insight**: We can use the embedding structure WITHIN a single soundscape file
to refine predictions transductively — no extra labels needed.

---

## 1. LaplacianShot (Ziko et al., ICML 2020 — arXiv 2006.15486)

### Core Idea
Add **Laplacian regularization** to few-shot inference:
- Objective: find soft label assignments q(y|x) that are:
  1. Close to prototype-based predictions (fidelity)
  2. Smooth over the embedding-space neighbor graph (regularization)

```
minimize Σ_i KL(q_i || prototype_softmax_i) + λ Σ_{i,j} w_{ij} ||q_i - q_j||²
```
where `w_{ij} = exp(-||z_i - z_j||² / σ²)` (RBF kernel over embeddings)

### Efficient Solution
Alternating optimization:
1. Update `q_i` given graph structure (graph smoothing step)
2. Update prototypes given soft labels (M-step)

Converges in ~10 iterations.

### Application: Per-File Transductive Inference
```python
def laplacian_shot_per_file(emb_12, prototype, lam=0.1, n_iter=10):
    """
    emb_12: (12, 1536) embeddings for one soundscape file
    prototype: (C, 1536) class prototypes
    Returns: (12, C) refined probability scores
    """
    # Initial prediction
    sim = emb_12 @ prototype.T  # (12, C)
    q = softmax(sim, axis=1)    # (12, C)

    # Build affinity graph over 12 clips
    D = pairwise_distances(emb_12)  # (12, 12)
    W = np.exp(-D / D.mean())
    W -= np.diag(np.diag(W))        # no self-loops
    D_deg = np.diag(W.sum(axis=1))
    L = D_deg - W                    # Laplacian

    for _ in range(n_iter):
        # Graph smoothing: q ← (I + λL)^{-1} q_0
        q = np.linalg.solve(np.eye(12) + lam * L, q)
        q = np.clip(q, 1e-9, 1)
        q /= q.sum(axis=1, keepdims=True)

    return q
```

### Expected Gain
+3-6% over static prototype when file has multiple clips with shared acoustics.
Especially useful for files where presence/absence is temporally structured.

---

## 2. Iterative Prototype Refinement (Zhu & Koniusz, CVPR 2023)

### Core Idea
Prototypes computed only from labeled support clips may be biased
(small support set = unrepresentative mean).

Fix: iteratively include query clips into prototype computation:
1. Compute prototype from support
2. Soft-label query clips using prototype
3. Update prototype = weighted mean of support + soft-labeled queries
4. Repeat until convergence

### Application
For each soundscape file (12 query clips) + labeled soundscape clips (support):
```python
def iterative_prototype(emb_support, y_support, emb_query, n_iter=5):
    for cls in range(C):
        proto = emb_support[y_support[:, cls] == 1].mean(axis=0)
        for _ in range(n_iter):
            # Soft-label query clips
            sim = emb_query @ proto  # (12,)
            soft_labels = sigmoid(sim)
            # Update prototype with soft-weighted queries
            proto = (
                emb_support[y_support[:, cls] == 1].mean(axis=0) +
                (soft_labels[:, None] * emb_query).sum(axis=0) / (soft_labels.sum() + 1e-8)
            ) / 2
    return proto
```

---

## 3. Practical Implementation Strategy for BirdCLEF

### Simple Transductive: Temporal Smoothing via Embedding Graph
The current `smooth_cols_fixed12` does 1D temporal smoothing on Amphibia/Insecta only.

**Upgrade**: Build a 2D graph over all 12 clips using embedding similarity,
smooth ALL classes (not just texture taxa):
```python
def embedding_graph_smooth(probs_12, emb_12, alpha=0.3):
    """probs_12: (12, C), emb_12: (12, 1536)"""
    # Cosine similarity graph
    emb_norm = emb_12 / (np.linalg.norm(emb_12, axis=1, keepdims=True) + 1e-8)
    W = emb_norm @ emb_norm.T  # (12, 12) cosine similarities
    W = np.maximum(W, 0)       # only positive correlations
    W /= W.sum(axis=1, keepdims=True) + 1e-8  # row-normalize

    smoothed = (1 - alpha) * probs_12 + alpha * (W @ probs_12)
    return smoothed
```

### Expected Total Pipeline
```
Perch embeddings (1536-dim)
    → L2-normalize + center             [SimpleShot preprocessing]
    → Tip-Adapter retrieval score       [support cache lookup]
    → Blend with native Perch logit     [zero-shot + retrieved]
    → Per-file embedding graph smoothing [transductive]
    → Final predictions
```

---

## References
- LaplacianShot: https://arxiv.org/abs/2006.15486
- Zhu & Koniusz CVPR 2023: https://arxiv.org/abs/2304.11598
