# Advanced Perch Probe Methods: Beyond PCA

**Written 2026-03-20. Based on literature survey of 2023-2025 papers.**

## Current Pipeline Baseline
- Perch 1536-dim → L2-norm → center → PCA(128, whiten) → MLP (2-layer) → 234 classes
- OOF AUC: ~0.8041 (with add_train_clips=True, v2-full)
- Holdout AUC: ~0.97+
- LB: 0.915 (v9-asl-soup ensemble)

The bottleneck is the **inference probe quality** and **embedding geometry**, not the SED model.
PCA(128) is a strong baseline but has known limitations:
1. Linear projection only — ignores nonlinear cluster structure
2. Whiten-per-dimension ≠ full covariance whitening (CL2N does full L2-norm)
3. Prototype distances are Euclidean in PCA space — assumes spherical clusters

---

## Method A: CL2N — Center + L2 Normalization (SimpleShot, Wang et al. 2019)

**Priority: HIGH | Effort: 15 min | Expected: +3-7pp over unnormalized baseline**

### Core Idea
Subtract the global training mean, then L2-normalize to unit sphere. This is the
**single most consistently effective post-processing** for frozen embeddings across benchmarks.
Unlike PCA whiten (per-dimension variance normalization), CL2N normalizes the FULL L2 norm,
placing all embeddings on the unit hypersphere.

Key difference from current pipeline:
- Current: L2-norm → center → PCA(128, whiten) = center + per-dim std normalization
- CL2N: center → L2-normalize (no PCA, no per-dim whitening)

CL2N + kNN outperforms many more complex methods on iNat (bird-relevant).

### Implementation
```python
import numpy as np

# Fit (once, on training embeddings):
global_mean = all_train_embs.mean(axis=0)  # (1536,)
np.save("outputs/cl2n_mean_1536.npy", global_mean)

# Transform:
def cl2n(emb, global_mean):
    centered = emb - global_mean
    norm = np.linalg.norm(centered, axis=-1, keepdims=True)
    return centered / (norm + 1e-8)

# Experiment: replace PCA(128) with CL2N(1536) → bigger space, sphere geometry
# Then use kNN or MLP head on 1536-dim CL2N features (not 128-dim PCA)
```

### Config Suggestion
`perch_probe_v3_cl2n_knn.yaml`: use_pca=False, cl2n=True, classifier=knn, k=15

---

## Method B: All-But-The-Top (Mu & Viswanath, ICLR 2018)

**Priority: MEDIUM | Effort: 30 min | Expected: +1-3pp over PCA baseline**

### Core Idea
Foundation model embeddings have "rogue dimensions" — a few dominant PCA directions
with enormous variance that carry NO class-discriminative signal (they encode audio length,
recording device, etc.). Removing these top-D directions BEFORE applying the main PCA
reduces noise and improves prototype quality.

This is different from standard PCA: standard PCA KEEPS top-D directions; ABT REMOVES them.

### Algorithm
1. Fit PCA on training embeddings, take top-D=10 components
2. Project embeddings onto those components and SUBTRACT the projection
3. Apply standard CL2N on the cleaned embeddings
4. Then apply the probe (PCA(128) + MLP or kNN)

```python
from sklearn.decomposition import PCA
import numpy as np

def all_but_top(embs, n_top=10):
    """Remove top n_top PCA directions (rogue dimensions)."""
    mu = embs.mean(axis=0)
    X = embs - mu
    pca_top = PCA(n_components=n_top)
    pca_top.fit(X)
    # Remove projection onto top components
    proj = X @ pca_top.components_.T @ pca_top.components_
    return (X - proj) + mu

# Apply BEFORE fitting the main PCA(128):
embs_abt = all_but_top(all_train_embs, n_top=10)
# Then fit PCA(128) on embs_abt as usual
```

### Why It Works for Perch
Perch was trained on diverse audio from many ecosystems. Its top variance directions
likely capture recording environment, not species identity. ABT removes this nuisance.

---

## Method C: UMAP Reduction (vs PCA)

**Priority: MEDIUM | Effort: 2h | Expected: +1-4pp on prototype quality**

### Core Idea
PCA is linear — it cannot preserve the nonlinear manifold structure of 234-class
bird species distributions. UMAP preserves local topology (nearby pairs stay nearby)
while compressing to a target dimension. For classification:
- PCA(128): best linear subspace by variance
- UMAP(128): best manifold-preserving projection for THIS data

Bioacoustic benchmarking (arXiv:2504.06710, 2025): "UMAP(300) embeddings perform
as well as 1536-dim for kNN classification but produce much cleaner cluster structure."

### Implementation
```python
import umap

reducer = umap.UMAP(
    n_components=128,
    n_neighbors=15,
    min_dist=0.1,          # tight clusters → better for classification
    metric='cosine',       # Perch operates in cosine space
    random_state=42,
    n_jobs=-1,
)
train_embs_umap = reducer.fit_transform(all_train_embs)  # (N, 128)

# Apply CL2N on top of UMAP embeddings:
global_mean = train_embs_umap.mean(axis=0)
train_cl2n = cl2n(train_embs_umap, global_mean)
```

### Caveats
- Stochastic: results vary between runs; use random_state=42
- Fit on 107k clips takes ~5-10 min on CPU
- Out-of-distribution transform can degrade; validate on holdout soundscape

---

## Method D: FeCAM — Mahalanobis NCM (Goswami et al., NeurIPS 2023)

**Priority: HIGH | Effort: 2h | Expected: +2-4pp vs Euclidean NCM prototype**

### Core Idea
Standard prototype networks use Euclidean or cosine distance, assuming SPHERICAL
class clusters. FeCAM uses Mahalanobis distance with PER-CLASS covariance matrices,
accounting for the fact that different species occupy differently-shaped regions.

Classification rule:
  score(x, c) = -[(x - μ_c)^T Σ_c^{-1} (x - μ_c)]  ← Mahalanobis

Three stabilization tricks that make this practical:
1. **Tukey transform**: x → sign(x) * |x|^0.5  (Gaussianizes skewed embedding dims)
2. **Ledoit-Wolf shrinkage**: regularize Σ_c to avoid singular matrix (crucial for rare classes)
3. **Correlation normalization**: normalize diags to 1 before inversion

### Why It Matters Here
- Our 85k train_audio clips → per-class covariance is now well-conditioned
- Bird species have acoustically diverse calls (breeding vs. non-breeding) → ellipsoidal clusters
- FeCAM naturally handles this multi-modal within-class structure

### Implementation
```python
import numpy as np
from sklearn.covariance import LedoitWolf

def tukey(X, lam=0.5):
    return np.sign(X) * np.abs(X) ** lam

class FeCAMClassifier:
    def fit(self, embeddings_per_class: dict):
        """embeddings_per_class: {c: array (N_c, D)}"""
        self.prototypes = {}
        self.inv_covs = {}
        for c, X in embeddings_per_class.items():
            X_t = tukey(X)
            mu = X_t.mean(axis=0)
            lw = LedoitWolf().fit(X_t)
            cov = lw.covariance_
            # Correlation normalization
            std = np.sqrt(np.diag(cov))
            corr = cov / (np.outer(std, std) + 1e-8)
            self.prototypes[c] = mu
            self.inv_covs[c] = np.linalg.pinv(corr)

    def score(self, x):
        """x: (D,) → {c: mahalanobis score}"""
        x_t = tukey(x)
        return {c: -(x_t - mu) @ self.inv_covs[c] @ (x_t - mu)
                for c, mu in self.prototypes.items()}

    def predict_proba(self, X):
        """X: (N, D) → (N, C) soft scores"""
        from scipy.special import softmax
        scores = np.array([[self.score(x)[c] for c in sorted(self.prototypes)]
                           for x in X])
        return softmax(scores, axis=1)
```

### Integration
Works on raw 1536-dim OR PCA(128) embeddings. PCA(128) is preferred (cheaper inversion):
128×128 matrix inversion per class = trivial. 1536×1536 would be 144× slower.

---

## Method E: LaplacianShot — Transductive Batch Inference

**Priority: HIGH | Effort: 3h | Expected: +7-10pp on iNat (234-class analogous task)**

### Core Idea
At soundscape inference time, ALL 739 clips are available simultaneously.
LaplacianShot (Ziko et al., ICML 2020) exploits this: clips from the SAME soundscape
file should share species. Nearby clips in embedding space should get similar predictions.

Key idea: build a kNN graph over all 739 clips, then jointly optimize labels to:
1. Match prototype distances (unary term)
2. Be smooth over the graph (Laplacian regularization)

This makes "consistent" predictions across the batch — no more isolated single-clip
mis-classifications that contradict their neighborhood.

iNat benchmark (most BirdCLEF-relevant): **+9.11 pp** over SimpleShot.
Your 234-class scenario will show smaller absolute gain but direction is reliable.

### Implementation
```python
import numpy as np
from scipy.special import softmax

def laplacianshot(query_embs, prototypes, k=5, lam=0.5, n_iter=20):
    """
    query_embs: (Q, D) — L2-normalized
    prototypes: (C, D) — L2-normalized class centroids
    Returns: (Q, C) soft label predictions
    """
    Q = len(query_embs)

    # Unary potentials: cosine distance to each prototype
    unary = 1 - query_embs @ prototypes.T      # (Q, C) in [0, 2]

    # Build kNN affinity graph over queries
    sim = query_embs @ query_embs.T            # (Q, Q) cosine
    np.fill_diagonal(sim, -np.inf)
    W = np.zeros_like(sim)
    for i in range(Q):
        top_k = np.argsort(sim[i])[-k:]
        W[i, top_k] = np.clip(sim[i, top_k], 0, None)
    W = np.maximum(W, W.T)                     # symmetrize

    # Normalized Laplacian
    D_inv_sqrt = np.diag(1 / (np.sqrt(W.sum(1)) + 1e-8))
    L = np.eye(Q) - D_inv_sqrt @ W @ D_inv_sqrt

    # Iterative bound optimizer
    Y = softmax(-unary, axis=1)
    for _ in range(n_iter):
        grad = unary + lam * (L @ Y)
        Y = softmax(-grad, axis=1)

    return Y  # (Q, C)

# Usage: replace per-clip MLP/kNN inference with LaplacianShot batch inference
# Applied AFTER MLP logits are computed → can use MLP scores as unary
```

### Hyperparameter Guidance
- k=5 to 10: start with k=7 for 739 clips
- lam=0.3 to 0.7: tune on holdout soundscape val set
- n_iter=20: converges in <10 usually

---

## Method F: LP++ — Few-Shot Linear Probe with Prototype Initialization (Huang et al., CVPR 2024)

**Priority: MEDIUM | Effort: 1h | Expected: +1-3pp vs standard linear probe**

### Core Idea
Standard linear probe initializes weights randomly → poor in few-shot regime.
LP++ initializes from class prototypes (mean embeddings) then learns PER-CLASS
blending scalars β_c to interpolate between prototype and an optimized weight vector.

Key insight: prototypes from 85k train_audio are reliable priors. The 739 soundscape
strong labels fine-tune β_c per class. Result: a linear head that generalizes like
zero-shot but discriminates like few-shot.

### Implementation
```python
import torch
import torch.nn.functional as F

def lp_plus_plus(
    train_embs, train_labels_1hot,   # 85k clips, weak labels
    ss_embs, ss_labels_multi,        # 739 clips, strong multi-label
    n_classes=234, tau=0.07, n_iters=50
):
    # Build prototype priors from large train set
    W_prior = torch.zeros(n_classes, train_embs.shape[1])
    for c in range(n_classes):
        mask = train_labels_1hot[:, c] > 0
        if mask.sum() > 0:
            W_prior[c] = train_embs[mask].mean(0)
    W_prior = F.normalize(W_prior, dim=1)

    # Fine-tune blending via LBFGS on soundscape strong labels
    beta = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.LBFGS([beta], lr=1.0, max_iter=n_iters)

    def closure():
        opt.zero_grad()
        b = torch.sigmoid(beta)  # per-class blend in [0,1]
        W = b[:, None] * F.normalize(ss_embs.T @ ss_labels_multi / ss_labels_multi.sum(0).clamp(1)[:, None], dim=1) \
          + (1 - b[:, None]) * W_prior
        logits = ss_embs @ F.normalize(W, dim=1).T / tau
        loss = F.binary_cross_entropy_with_logits(logits, ss_labels_multi.float())
        loss.backward()
        return loss

    opt.step(closure)
    return W_prior  # or return final blended W

# Benefit: blending prevents overfitting to 739 clips while leveraging strong labels
```

---

## Method G: Label Propagation for Better Train-Audio Pseudo-Labels

**Priority: HIGH | Effort: 4h | Expected: better SED training (indirect +AUC)**

### Core Idea
Current pseudo-labels for 85k train_audio clips come from Perch's raw predictions.
Label propagation on the Perch embedding kNN graph can REFINE these pseudo-labels:
nearby clips share labels (structural consistency), correcting isolated errors.

This produces better soft-labels for the SED distillation pipeline:
`perch_teacher_all_ss.csv` → label-propagated version → better soft KD targets

### Implementation (high level)
```python
import faiss
import numpy as np
import scipy.sparse as sp

# 1. Build mutual kNN graph over all 107k Perch embeddings (FAISS, ~2min)
# 2. Seed: soundscape strong labels → Y_labeled (739 clips)
# 3. Propagate: F = (I - α D^{-1/2} W D^{-1/2})^{-1} (1-α) Y
# 4. Save propagated labels as new soft pseudo-labels CSV
# 5. Use in SED training: extra_pseudo_csv = propagated_labels.csv

# Alpha=0.99: strong propagation (trust graph structure over model)
# k=15: mutual kNN ensures clean graph edges
```

---

## Experiment Design: Ranked by Expected Impact

| # | Experiment Config Name | Method | Key Change vs Baseline | Expected OOF AUC | GPU Needed |
|---|------------------------|--------|------------------------|-----------------|------------|
| 1 | `perch_probe_v3_fecam` | FeCAM | Mahalanobis NCM on PCA(128) | ~0.82-0.84 | No |
| 2 | `perch_probe_v3_cl2n_knn` | CL2N + kNN | No PCA, raw 1536 CL2N-normalized + kNN-15 | ~0.81-0.83 | No |
| 3 | `perch_probe_v3_abt` | All-But-Top(10) + PCA | Remove 10 rogue dims first, then PCA(128) | ~0.81-0.82 | No |
| 4 | `perch_probe_v3_laplacian` | LaplacianShot | Batch transductive inference over all soundscape clips | ~0.83-0.87 | No |
| 5 | `perch_probe_v3_umap` | UMAP(128) + CL2N | Nonlinear dim reduction instead of PCA | ~0.80-0.83 | No |
| 6 | `perch_probe_v3_sed_concat` | Perch + SED concat | Concatenate 1536 Perch + 1280 SED-B0 embeddings | ~0.83-0.87 | Yes (extract SED) |
| 7 | `perch_probe_v3_lp++` | LP++ | Prototype-initialized linear probe | ~0.81-0.83 | No |

### Immediate Next Steps (All CPU-only, can run now)

**Step 1** — `perch_probe_v3_abt`: Easiest change, add `all_but_top(n=10)` before PCA.
  Run in ~same time as v2. Quick confirmation of whether rogue dims hurt.

**Step 2** — `perch_probe_v3_fecam`: Replace MLP head with FeCAM classifier on PCA(128).
  No training loop needed. Should see +2-4pp from better distance metric.

**Step 3** — `perch_probe_v3_laplacian`: Add LaplacianShot post-processing.
  The 739 clips fit in memory easily. Can even be added to the NOTEBOOK (not just probe training).

**Step 4** — `perch_probe_v3_sed_concat`: After fold0 finishes, extract backbone embeddings
  from `sed-b0-v30-multipseu` and concatenate with Perch. Best expected gain overall.

---

## Key Insight: PCA Limitations and When to Replace It

| Scenario | Best Approach |
|----------|---------------|
| Prototype / kNN inference | CL2N (no PCA) — sphere geometry is better for cosine kNN |
| MLP classification head | PCA(128, whiten) — whitening helps MLP by equalizing dim scales |
| Rare class (<5 samples) | Mahalanobis with Ledoit-Wolf + Tukey → more stable than Euclidean proto |
| Batch inference (full soundscape) | LaplacianShot → joint optimization over 739 clips |
| Strong label propagation | kNN graph on raw CL2N embeddings → better propagation than PCA space |

**DO NOT use ZCA whitening before MLP head** — confirmed harmful for MLP classifiers
(NAACL 2024 "Whitening Not Recommended for Classification Tasks").
PCA whiten is fine because it only normalizes MARGINAL variances, not the full covariance.

---

## References
- SimpleShot (CL2N): Wang et al., arXiv 1911.04623 (2019)
- All-But-The-Top: Mu & Viswanath, ICLR 2018
- FeCAM: Goswami et al., NeurIPS 2023, arXiv 2309.14062
- LaplacianShot: Ziko et al., ICML 2020, arXiv 2006.15486
- LP++: Huang et al., CVPR 2024, arXiv 2404.02285
- UMAP bioacoustics: arXiv 2504.06710 (2025)
- Label Propagation audio: arXiv 1904.04717
- ZCA warning: ACL 2024 "Whitening Not Recommended for Classification"
- Diverse audio embeddings: arXiv 2309.08751
- PUTM (Conditional Transport): ICCV 2023, arXiv 2308.03047
