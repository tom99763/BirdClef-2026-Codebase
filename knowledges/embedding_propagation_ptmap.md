# Embedding Propagation & Power Transform — Graph-Based Few-Shot Methods

## Method 1: Embedding Propagation (ECCV 2020)
**Paper**: Rodríguez et al., "Embedding Propagation: Smoother Manifold for Few-Shot Classification"
arXiv:2003.04151

### Core Idea
Build a k-NN similarity graph over ALL available embeddings (labeled support + query + unlabeled).
Propagate embeddings through graph edges to smooth the feature manifold.
The smoothed embeddings replace raw embeddings for prototype computation.

**Key insight**: In Perch embedding space, acoustically similar species (same genus, similar call
structure) cluster near each other. Propagation lets labeled species "bleed" toward nearby unlabeled
clusters, providing soft labels even for species with no direct labels.

### Algorithm
```python
def embedding_propagation(all_feats, alpha=0.5, k=5):
    """
    all_feats: (N, 1536) — support + query + unlabeled
    Returns: (N, 1536) — smoothed embeddings
    """
    N = len(all_feats)

    # Build k-NN affinity matrix using cosine similarity
    feats_norm = all_feats / (np.linalg.norm(all_feats, axis=1, keepdims=True) + 1e-8)
    sim = feats_norm @ feats_norm.T  # (N, N) cosine similarities

    # k-NN sparsification: keep top-k neighbors per node
    for i in range(N):
        topk = np.argsort(sim[i])[:-(k+1):-1]  # top-k
        mask = np.zeros(N, dtype=bool)
        mask[topk] = True
        sim[i][~mask] = 0

    # Symmetrize
    W = (sim + sim.T) / 2

    # Degree normalization: D^{-1/2} W D^{-1/2}
    D = np.diag(W.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / (np.sqrt(W.sum(axis=1)) + 1e-8))
    L = D_inv_sqrt @ W @ D_inv_sqrt  # normalized adjacency

    # One-step propagation: Z_smooth = (1-alpha)*Z + alpha*L@Z
    Z_smooth = (1 - alpha) * all_feats + alpha * (L @ all_feats)

    return Z_smooth
```

### BirdCLEF Application
```python
# Build joint embedding set: 708 labeled + 127k unlabeled soundscape clips
# (need 1536-dim emb for unlabeled — requires extraction run)
all_emb = np.vstack([labeled_emb_708, unlabeled_emb_127k])  # (127708, 1536)

# Too large for dense k-NN — use FAISS approximate k-NN
import faiss
index = faiss.IndexFlatIP(1536)
index.add(all_emb_norm)
D, I = index.search(all_emb_norm, k=10)  # top-10 neighbors per clip

# Apply propagation
Z_smooth = embedding_propagation_sparse(all_emb, I, D, alpha=0.3)

# Use smoothed embeddings for probe training on labeled 708 clips
Z_smooth_labeled = Z_smooth[:708]
# Run standard PCA + LogReg on Z_smooth_labeled
```

### Expected Gain
- +3–16% in semi-supervised settings (ECCV 2020 benchmarks)
- Most gain when unlabeled data is plentiful (we have 127k clips — ideal)
- Especially helpful for the 163 zero-positive classes where the manifold smoothing
  can transfer label information from acoustically nearby labeled classes

### Practical Notes
- With 127k clips, the k-NN computation requires FAISS (not brute-force)
- One-step propagation is fast; multi-step gives diminishing returns
- Apply on L2-normalized embeddings only
- Alpha=0.3 is conservative; tune on OOF

---

## Method 2: PT-MAP / Power Transform (ECCV 2020 follow-up)
**Paper**: Hu et al., "Leveraging the Feature Distribution in Transfer-based Few-Shot Learning"
**Also referenced as**: β-normalization, ReLU-Norm

### Core Idea
Perch embeddings from EfficientNet-B3 have **heavy-tailed dimension distributions**.
PCA whitening helps but doesn't fix the tail problem.
A power transform makes each dimension more Gaussian before normalization:
```
feats_pt[i, d] = sign(feats[i,d]) * |feats[i,d]|^beta   (beta ∈ [0.3, 0.7])
```

### Why It Helps
- Heavy tails: a few extreme values dominate cosine similarity → wrong prototypes
- Power transform compresses extremes, making each dimension contribute equally
- Combined with centering + L2-norm: significantly improves prototype quality

### Algorithm
```python
def power_transform_normalize(feats, beta=0.5):
    """
    beta: exponent — 0.5 is square root (most common)
    Pipeline: power-transform → L2-norm → center → L2-norm
    """
    # Step 1: Power transform
    feats_pt = np.sign(feats) * np.abs(feats) ** beta

    # Step 2: L2-normalize
    feats_pt = feats_pt / (np.linalg.norm(feats_pt, axis=1, keepdims=True) + 1e-8)

    # Step 3: Center (subtract global mean)
    feats_pt = feats_pt - feats_pt.mean(axis=0)

    # Step 4: L2-normalize again
    feats_pt = feats_pt / (np.linalg.norm(feats_pt, axis=1, keepdims=True) + 1e-8)

    return feats_pt

# Usage: apply before PCA in precompute_probe_cache.py
# Current: L2-norm → center → PCA
# Upgraded: power_transform(beta=0.5) → L2-norm → center → PCA
```

### Expected Gain
+1–3% additional accuracy over SimpleShot-style normalization alone.
Low cost: 3 extra lines of code in preprocessing.

---

## Method 3: Perch 2.0 Prototypical Probing
**Paper**: Burns et al., "Perch 2.0: The Bittern Lesson for Bioacoustics" (arXiv:2508.04665, 2025)

### Core Idea
Instead of a single centroid prototype per class, learn **K prototypes per class** (K=4 default).
Optimize inter-class orthogonality to maximize separation.

```python
class PrototypicalProbe(nn.Module):
    def __init__(self, embed_dim=128, n_classes=234, n_protos=4):
        super().__init__()
        # K learned prototypes per class
        self.protos = nn.Parameter(torch.randn(n_classes, n_protos, embed_dim))

    def forward(self, x):
        # x: (N, embed_dim) — L2-normalized query embeddings
        # Compute similarity to each prototype
        p = F.normalize(self.protos, dim=-1)  # (C, K, D)
        x_e = x.unsqueeze(1).unsqueeze(1)    # (N, 1, 1, D)
        sim = (x_e * p.unsqueeze(0)).sum(-1)  # (N, C, K)
        return sim.max(dim=-1).values          # (N, C) — max-pool over prototypes

    def orthogonality_loss(self):
        # Penalize similarity between prototypes of DIFFERENT classes
        p = F.normalize(self.protos.view(-1, self.protos.shape[-1]), dim=-1)
        sim = p @ p.T  # (C*K, C*K)
        # Exclude same-class pairs
        ...
        return off_diag_sim.mean()

# Training: BCE multi-hot loss + lambda * orthogonality_loss
# 708 clips, GroupKFold-5, 200 epochs, AdamW lr=1e-3
```

### Results from Perch 2.0 paper
- BirdSet ROC-AUC: 0.839 (Perch 1.0) → 0.907 (Perch 2.0) with this probing method
- 4 prototypes per class significantly outperforms 1 prototype (single centroid)
- Multi-prototype captures acoustic variants (male/female calls, geographic variation)

### BirdCLEF Application
Most valuable for classes where:
- Multiple call types exist (e.g., alarm calls vs contact calls)
- Male and female have different vocalizations
- Geographic variants of the same species

---

## Combined Pipeline: Best of All Methods

```
Perch 1536-dim embeddings
    ↓
Power transform (beta=0.5)    ← NEW (PT-MAP)
    ↓
L2-normalize
    ↓
Center (global mean subtraction)
    ↓
L2-normalize again
    ↓
PCA(128, whiten=True)          ← current
    ↓
[Labeled 708 clips path]       ← Embedding Propagation enriches these
    ↓
LogReg (≥8 pos) / DistCal+LogReg (2-7 pos) / K-Prototype (all classes)  ← NEW K-Proto
    ↓
Blend with TipAdapter + Perch native
    ↓
TIM per-file refinement        ← current (not yet added)
    ↓
Power scaling (gamma∈[0.5,0.9])← NEW (BirdCLEF 2025 technique)
```

---

## Implementation Priority

| Method | Effort | Expected Gain | Dependency |
|--------|--------|--------------|------------|
| PT-MAP (power transform beta=0.5) | 3 lines | +0.002-0.005 | None |
| K-Prototype probe (K=4) | ~50 lines | +0.005-0.010 | Need GPU train |
| Embedding Propagation | ~100 lines | +0.005-0.015 | Need 127k embeddings |

## References
- Embedding Propagation: https://arxiv.org/abs/2003.04151
- PT-MAP: Hu et al. 2021 (follows up on ECCV 2020 work)
- Perch 2.0: https://arxiv.org/abs/2508.04665
