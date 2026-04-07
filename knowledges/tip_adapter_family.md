# Tip-Adapter Family: Cache-Based Few-Shot Adapters

## Papers Covered
1. **Tip-Adapter** (Zhang et al., ECCV 2022 — arXiv 2207.09519)
2. **APE** (Zhu et al., ICCV 2023 — arXiv 2304.01195)
3. **LP++** (Huang et al., CVPR 2024 — arXiv 2404.02285)

---

## 1. Tip-Adapter

### Core Idea
Build a **key-value cache** from labeled support examples:
- Keys = support embeddings (normalized)
- Values = one-hot labels

At inference:
```
retrieved_logit = softmax(query @ keys.T / τ) @ values   # (C,)
final_logit = alpha * retrieved_logit + (1-alpha) * zero_shot_logit
```

Parameters: `alpha` (blend weight), `tau` (temperature/sharpness)

### Why This Helps
- Captures **local manifold structure** in embedding space
- The "zero-shot prior" = Perch's own native logits (14795 → 234 mapped)
- Retrieved neighbors vote for a class even if the class prototype is imprecise

### Application to BirdCLEF
```python
# Build cache from all labeled positive soundscape clips
# Key = L2-normalized Perch embedding (1536-dim)
# Value = one-hot label (234-dim)

keys   = emb_norm[labeled_mask]   # (N_labeled, 1536)
values = Y_labeled                 # (N_labeled, 234)

def tip_adapter_score(query_emb, keys, values, alpha=0.5, tau=1.0):
    # query_emb: (12, 1536) normalized
    sim = query_emb @ keys.T / tau          # (12, N_labeled)
    weights = np.exp(sim)                    # softmax numerator
    weights = weights / weights.sum(axis=1, keepdims=True)
    retrieved = weights @ values             # (12, 234)
    return retrieved

# Blend with Perch native logits
final = alpha * tip_adapter_score(q) + (1 - alpha) * perch_native_prob
```

### Hyperparameters to Tune (OOF)
- `alpha`: blend weight (0.3–0.7)
- `tau`: temperature (0.1–1.0, lower = sharper = more like 1-NN)

### Expected Gain
+2-5% over pure nearest-prototype. Particularly useful when positive clips
cluster in embedding space (which Perch guarantees for bird species).

---

## 2. APE (Adaptive Prior Refinement)

### Core Idea
Not all 1536 embedding dimensions are equally useful for the **downstream task**.
APE selects per-class the most discriminative dimensions by measuring
inter-class disparity in the support set.

### Training-Free Variant
```python
# For each class, compute feature relevance
for cls in classes:
    pos_mean = emb[y[:, cls]==1].mean(axis=0)  # (1536,)
    neg_mean = emb[y[:, cls]==0].mean(axis=0)  # (1536,)
    relevance = (pos_mean - neg_mean) ** 2      # feature importance
    top_dims[cls] = np.argsort(relevance)[-K:]  # top-K dimensions
    score[cls] = (emb[:, top_dims[cls]] @ pos_mean[top_dims[cls]]) / K
```

### Application
Per-class feature selection within 1536-dim space; works as a preprocessing
step before prototype matching or LogReg.

---

## 3. LP++ (Prior-Initialized Linear Probe)

### Core Idea
Standard linear probe trains classifier weights randomly initialized.
LP++ initializes weights from the **class prototype** (support set mean):
```
W_init[cls] = mean_emb_positive[cls]  # shape (1536,)
```
Then fine-tunes via standard cross-entropy. This initialization is close to the
optimal solution (nearest-centroid), so it converges much faster with fewer samples.

### Blending Coefficients
LP++ adds per-class blending coefficients that mix prototype logit with probe logit:
```
final[cls] = beta[cls] * probe_logit[cls] + (1 - beta[cls]) * prototype_logit[cls]
```
`beta[cls]` is learned from the support set validation.

### Application to BirdCLEF
Directly replace current LogReg initialization with prototype-based init:
```python
# Current: LogisticRegression(C=0.5) with random init
# LP++ style: initialize coef_ from class prototype
clf = LogisticRegression(C=C, solver='liblinear')
clf.fit(X, y)
# Override: warm start from prototype
clf.coef_ = prototype[cls].reshape(1, -1)
clf.fit(X, y)  # continues from prototype init
```

### Expected Gain
+2-4% over standard LogReg, especially in low-data regime (5–20 positives).

---

## Key Takeaway
The Tip-Adapter approach (support cache + zero-shot blend) is the most
directly applicable because:
1. We have labeled soundscape clips as a support set
2. We have Perch native logits as the "zero-shot prior"
3. No training needed — just tune alpha and tau via OOF

## References
- Tip-Adapter: https://arxiv.org/abs/2207.09519
- APE: https://arxiv.org/abs/2304.01195
- LP++: https://arxiv.org/abs/2404.02285
