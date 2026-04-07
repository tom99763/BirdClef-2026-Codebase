# Few-Shot Foundations: SimpleShot & Prototypical Networks

## Current System Context
- Perch embeddings: 1536-dim, frozen
- Current probe: PCA(64) + LogisticRegression → 0.918 LB
- Key insight: the embedding quality dominates; classifier head matters less

---

## 1. SimpleShot (Wang et al., 2019 — arXiv 1911.04623)

### Core Idea
Strip away all meta-learning. Apply:
1. **L2-normalize** each embedding vector
2. **Subtract global dataset mean** (centering)
3. Classify by **nearest centroid** (cosine distance to class prototype)

Outperforms episodically-trained meta-learners with zero training overhead.

### Key Finding
> "The proper normalization of features matters more than the choice of meta-learning algorithm."

### Application to BirdCLEF
```python
# L2-normalize Perch embeddings
emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

# Subtract global mean (computed from all labeled soundscape clips)
global_mean = emb_norm[labeled_mask].mean(axis=0)
emb_centered = emb_norm - global_mean

# Per-class prototype = mean of positive clips in embedding space
prototype[cls] = emb_centered[y[:, cls] == 1].mean(axis=0)

# Score = cosine similarity to prototype
score[cls] = emb_centered @ prototype[cls]  # already L2-normalized
```

### Expected Gain
- Replaces/supplements PCA+LogReg for classes with few positives
- L2-norm + centering is a free improvement over current StandardScaler → PCA

---

## 2. Prototypical Networks (Snell et al., 2017 — NeurIPS)

### Core Idea
Class representation = mean embedding of support (labeled) examples.
Distance metric = Euclidean or cosine in embedding space.
Softmax over negative distances → class probabilities.

### Key Difference from SimpleShot
Prototypical Networks use Euclidean distance; SimpleShot uses L2-normalized cosine.
SimpleShot shows cosine + centering is usually better.

### Application
For rare species (1–5 positives), prototype is more stable than LogReg:
- LogReg with 2 positives is unreliable (hyperplane undefined)
- Prototype always works: distance to mean positive embedding

### Recommended Hybrid Strategy
```
if n_positives >= 8:
    use LogisticRegression (current approach)
elif n_positives >= 2:
    use prototypical nearest centroid
else:
    use Perch native logit only
```

---

## 3. Key Normalization Insight (SimpleShot)

The paper tests multiple normalizations:
- None: weak
- L2-norm: +3-5%
- L2-norm + center: +5-8% (best)
- PCA whitening: similar to L2-norm + center

**Our current StandardScaler → PCA(64) misses L2 normalization.**
Adding L2-norm before PCA should be a free improvement.

---

## References
- SimpleShot: https://arxiv.org/abs/1911.04623
- Prototypical Networks: https://arxiv.org/abs/1703.05175
- Ghani et al. (2023) bioacoustics benchmark: https://arxiv.org/abs/2307.06292
