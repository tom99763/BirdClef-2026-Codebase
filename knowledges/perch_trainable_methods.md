# Trainable Methods on Perch Embeddings

## System Context
- Current: Perch 1536-dim → L2 → center → PCA(128, whiten) → LogReg + Proto + TipAdapter = **0.915 LB**
- Problem: 163/234 classes have 0 labeled positives → no probe, pure Perch fallback
- Available unlabeled data: 127k soundscape clips with 234-class Perch scores (perch_teacher_all_ss.csv)

---

## Method T1: Unsupervised Domain PCA (Priority: HIGH, Effort: LOW)

### Core Idea
Currently PCA is fitted on **708 labeled soundscape clips** only.
If we can get 1536-dim Perch embeddings for ALL soundscape clips, fitting PCA on
**127k+ clips** would give much more representative principal axes for the soundscape domain.

### Why It Helps
- 708 clips PCA: principal components reflect the 52 common species distribution
- 127k clips PCA: principal components reflect the FULL Pantanal soundscape distribution
- Better PCA axes → better separation of rare species in the low-dimensional space

### Implementation
```python
# Step 1: Extract 1536-dim embeddings for all 127k soundscape clips
# (Modify extract_perch_teacher_all_ss.py to also save embeddings)
# Runtime: ~15h on CPU for 127k clips; or select smart subset

# Step 2: Fit PCA on ALL embeddings (L2-normalized)
all_emb = np.vstack([
    full_perch_arrays['emb'],     # 708 labeled
    pseudo_emb_127k,              # 127k unlabeled
])
all_emb_norm = l2_normalize(all_emb)
all_emb_centered = all_emb_norm - all_emb_norm.mean(axis=0)
pca = PCA(n_components=128, whiten=True).fit(all_emb_centered)
# pca is now fit on 127k+ clips, not just 708

# Step 3: Same probe training as before, but using better PCA transform
```

### Practical Constraint
Requires extracting 1536-dim embeddings for all soundscape clips.
Alternative: use a random sample of 10k clips (still 14× more data, ~15 min extraction).

---

## Method T2: MLP Adapter (Priority: MEDIUM, Effort: MEDIUM)

### Core Idea
Replace frozen PCA(128) with a **trainable 2-layer MLP** that maps 1536 → 128.
The MLP is trained on the 708 labeled soundscape clips with multi-hot BCE loss.
Unlike PCA (unsupervised), the MLP learns soundscape-discriminative features.

### Architecture
```python
class PerchAdapter(nn.Module):
    def __init__(self, in_dim=1536, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)  # L2-normalize output

# Training: 708 clips, multi-hot BCE, AdamW lr=1e-3, 200 epochs
# Heavy L2 reg (weight_decay=0.1) to avoid overfitting
```

### GroupKFold Integration
Use same GroupKFold-5 as current probe:
- For each fold: train MLP on 4/5 clips → transform val fold → run LogReg/Proto on transformed emb
- Fully leak-free

### Expected Gain
The MLP learns:
1. Which Perch dimensions are most useful for soundscape discrimination
2. Non-linear interactions between Perch dimensions
3. Domain-specific class boundaries

Risk: 708 clips × 1536 dims = small training set for MLP → heavy regularization needed.

---

## Method T3: SED Backbone Embeddings as Probe Input (Priority: HIGH, Effort: LOW)

### Core Idea (Novel!)
Our best SED model (v30-multipseu, holdout 0.9839) is **domain-adapted** to Pantanal soundscapes.
Instead of using Perch embeddings for the probe, use the SED model's backbone embeddings.

```
SED model (v30-multipseu) architecture:
  Input: Mel spectrogram
  → EfficientNet-B0 backbone → 1280-dim features (after GEMFreqPool)
  → Attention head → 234-dim output

What we want:
  Extract the 1280-dim backbone features for each labeled soundscape clip
  → Use as probe input instead of Perch 1536-dim embeddings
  → Run PCA(128) + LogReg + Proto on these 1280-dim domain-adapted features
```

### Why This Could Be Better Than Perch
- Perch was trained on diverse bird audio, not Pantanal soundscapes
- v30-multipseu was fine-tuned on Pantanal + pseudo labels → domain adapted
- The backbone features may better separate Pantanal species in embedding space
- Can also combine: [Perch_emb; SED_emb] → joint 2816-dim probe

### Implementation
```python
# Extract SED backbone embeddings for 708 labeled soundscape clips
model = load_sed_model('checkpoints/sed-b0-v30-multipseu/best_sed.pt')
model.eval()

sed_embeddings = []
for clip in labeled_soundscape_clips:
    mel = compute_mel(clip)
    with torch.no_grad():
        emb = model.extract_embedding(mel)  # 1280-dim GEMFreqPool output
    sed_embeddings.append(emb)

# Replace Perch emb with SED emb in precompute_probe_cache.py
# Run same PCA + LogReg + Proto pipeline
```

### Hybrid: Perch + SED Fusion
```python
# Stack both embeddings and run probe on joint representation
joint_emb = np.hstack([perch_emb_norm, sed_emb_norm])  # (708, 2816)
# Apply PCA to joint embedding
pca = PCA(n_components=256, whiten=True).fit(joint_emb_centered)
# Then standard LogReg + Proto
```

---

## Method T4: Semi-Supervised Prototype Expansion via Perch Scores

### Core Idea
For the 163 zero-positive classes, use `perch_teacher_all_ss.csv` to find pseudo-positive clips.
Then extract their 1536-dim Perch embeddings to create pseudo support points.

This is the most direct fix for the zero-coverage problem.

### Two-Step Process
```
Step 1 — Identify pseudo-positives:
  For class c with 0 labeled positives:
    Find top-K clips from perch_teacher_all_ss.csv where score[c] > 0.3
    K = min(10, n_clips_above_threshold)

Step 2 — Extract embeddings for those clips:
  Run Perch TFLite on each selected clip
  Save 1536-dim embedding + confidence weight

Step 3 — Weighted prototype:
  proto[c] = weighted_mean(pseudo_emb_c, weights=scores_c)
```

### Expected Coverage
- thresh=0.3: 62/163 zero-positive classes covered (mean 13.6 clips/class)
- thresh=0.2: 84/163 covered

### Data Statistics
- ~1500 pseudo clips to extract (62 classes × ~24 clips average)
- Runtime: ~12 min on CPU
- Result: proto[c] for 62 previously uncovered classes

---

## Method T5: Confidence-Weighted TipAdapter Keys

### Core Idea
Current TipAdapter: all 708 support clips contribute equally as keys.
Extension: also add pseudo-positive clips from 127k soundscape pool as additional keys,
weighted by Perch confidence (score).

```python
# Current keys: (708, 1536) with equal weight
# New keys: (708 + N_pseudo, 1536) with soft weights

tip_key_weights = np.ones(708)  # labeled clips get full weight
tip_key_weights_pseudo = pseudo_scores  # 0.3-0.8 for pseudo clips

def weighted_tip_adapter(query_emb, all_keys, all_values, key_weights, tau=0.5):
    sim = query_emb @ all_keys.T / tau  # (12, N_keys)
    # Weight retrieval by confidence
    weighted_sim = sim * key_weights[None, :]
    weights = np.exp(weighted_sim)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ all_values
```

---

## Method T6: Power Scaling Post-Processing (BirdCLEF 2025 Technique)

### Core Idea (from 3rd + Winner of BirdCLEF 2025)
Apply element-wise power transformation to final scores:
```python
final_scores = final_scores ** gamma
```
- gamma < 1: sharpens peaks, suppresses low-confidence noise
- gamma = 0.7: typical value from 2025 solutions
- Tune gamma on OOF validation (try 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

### Why It Works
The combined Perch+SED scores are "soft" probabilities. Many low-confidence species
contribute noise. Power sharpening focuses on high-confidence predictions while
reducing the contribution of near-zero scores.

### Quick Experiment
```python
for gamma in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    scores_gamma = np.power(np.clip(base_scores, 1e-9, 1), gamma)
    auc = compute_auc(scores_gamma, Y_true)
    print(f'gamma={gamma}: OOF AUC={auc:.4f}')
```

---

## Method T7: TTA with Time Shifts (+0.012 from BirdCLEF 2025 2nd Place)

### Core Idea
For each 5-second soundscape window, evaluate Perch/SED at BOTH:
1. Original clip boundaries
2. +2.5 second shifted boundaries

Average both predictions → more robust to calls at clip edges.

### Implementation
```python
# At inference time:
for file in soundscape_files:
    audio = load_audio(file, 60s)  # full 60-second file

    # Standard 5s windows: 0-5, 5-10, ..., 55-60
    scores_standard = evaluate_windows(audio, step=5)   # (12, 234)

    # Shifted 5s windows: 2.5-7.5, 7.5-12.5, ..., 52.5-57.5
    scores_shifted = evaluate_windows(audio, step=5, offset=2.5)  # (11, 234)

    # Align and average
    final_scores = combine_tta(scores_standard, scores_shifted)
```

### Expected Gain
+0.010-0.015 LB based on BirdCLEF 2025 results.
The gain is largest for species whose calls happen at clip boundaries.

---

## Experiment Priority Table (Updated)

| # | Method | Expected Gain | GPU Hours | Category |
|---|--------|--------------|-----------|----------|
| 1 | T3: SED backbone → probe (v30-multipseu emb) | **+0.010-0.020** | 0 (CPU) | Novel |
| 2 | T4: Pseudo support expansion (62 zero classes) | **+0.010-0.020** | 1h extraction | Data |
| 3 | T7: TTA time shifts | **+0.010-0.015** | 0 (inference) | Quick |
| 4 | T6: Power scaling gamma tuning | +0.005-0.015 | 0 (OOF eval) | Quick |
| 5 | TIM inference (notebook) | +0.003-0.008 | 0 | Quick |
| 6 | T1: Domain PCA (127k clips) | +0.005-0.010 | 15h extraction | Data |
| 7 | Multi-scale PCA ensemble | +0.003-0.008 | 0 (recompute) | Probe |
| 8 | T2: MLP adapter (trainable) | +0.005-0.010 | 2h GPU | Novel |
| 9 | T5: Confidence-weighted TipAdapter | +0.003-0.008 | 0 | Probe |

## Recommended Next Actions

**TODAY (no GPU, quick)**:
1. Add power scaling gamma search to OOF evaluation — can test in 30 min
2. Add TIM to notebook — can test in 1 hour

**THIS WEEK (1-2 days)**:
3. Extract SED backbone embeddings for 708 labeled clips → run probe on SED emb vs Perch emb
4. Extract 1536-dim Perch emb for ~1500 pseudo-positive clips → expand support for 62 classes
5. Combine: probe on [Perch_emb + SED_emb] fusion

**NEXT WEEK (requires training)**:
6. MLP adapter training on 708 soundscape clips (with leave-one-file-out CV)
7. TTA time shifts (requires modifying inference pipeline)
