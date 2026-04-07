# Bioacoustic-Specific Transfer Learning Notes

## Key Paper: Ghani et al. 2023 (Scientific Reports / arXiv 2307.06292)
**"Global birdsong embeddings enable superior transfer learning for bioacoustic classification"**

### Main Findings Relevant to Us
1. **Perch dominates** all other audio embedding models (BirdNET, PANNs, VGGish) for bioacoustics
2. Linear SVM on Perch embeddings = best average across all tasks
3. **5-shot Perch** ≈ fully supervised with 100+ examples (other models need much more data)
4. Embedding quality >> classifier head complexity

### What This Means
- We're already using the right embedding (Perch)
- Our PCA+LogReg is near-optimal for the classifier head
- Main gains will come from:
  a. Better use of the full 1536 dims (L2-norm + center vs PCA projection)
  b. Transductive inference leveraging soundscape structure
  c. Better blending of Perch native logits with embedding-derived scores

---

## Self-Supervised for OOD Species (Moummad et al., arXiv 2312.15824)

### Key Point
Self-supervised bird audio embeddings can complement Perch for species
outside Perch's 10k training distribution.

### Relevance
BirdCLEF 2026 may include species not covered by Perch's 14795 classes.
For those, the embedding may still place them near acoustically-similar relatives.
The proxy genus approach we use (→ genus BC indices) handles this at the logit level.

---

## ZCA Whitening (from Zheng et al. 2023)

### Core Idea
Standard PCA projects to top-K components. ZCA whitening decorrelates
all dimensions while preserving input space geometry.

```python
from sklearn.decomposition import PCA

# ZCA whitening
pca = PCA(n_components=1536, whiten=True)
emb_zca = pca.fit_transform(emb)  # (N, 1536) decorrelated + unit variance

# vs current approach: PCA(64) — loses 1472 dims!
```

### Why ZCA Over PCA(64)
- PCA(64) keeps only 64 dims, discards the rest
- ZCA keeps all 1536 dims but decorrelates them
- For nearest-centroid classification, ZCA often outperforms PCA
- Especially important for rare species with few positives (every dim matters)

### Practical Note
Full ZCA on 1536 dims is expensive (1536×1536 SVD).
Alternative: PCA with whitening + keep more components (e.g., 256 or 512).

---

## Key Practical Insight: Support Set = Labeled Soundscape Clips

We have ~186 fully-labeled soundscape files × 12 clips = ~2232 labeled clips.
Each clip has multi-hot species labels.

This IS the few-shot support set. The embedding structure over these 2232 clips
defines the per-class prototypes and the support cache for Tip-Adapter.

### Size per Class (approx)
- Common species: 50–200 positive clips in support
- Rare species: 2–20 positive clips
- Very rare: 0–1 clips (need to fall back to Perch native logit)

The methods above are most impactful for the **rare species** (2–20 positives)
where LogReg is unreliable but prototypical works.
