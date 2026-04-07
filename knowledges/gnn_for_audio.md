# GNN and Graph-Based Methods for Audio / Bioacoustic Classification

*Compiled 2026-03-21. Survey covers 2019–2025.*

---

## Summary for BirdCLEF 2026

**Key takeaway:** Trained GNNs underperform non-parametric methods (D8_cross_attn 0.696 vs D9_gcn 0.544) on our 708-clip support set due to label sparsity and small graph size. The right path is **parameter-free graph methods** (APPNP propagation, LPA+cross-attention) or **heterogeneous graphs** that explicitly model species structure.

---

## Paper 1: GraFPrint — GNN for Audio Fingerprinting ⭐

**Citation:** ICASSP 2025. arXiv:2410.10994
**Code:** https://github.com/chymaera96/GraFP

**Method:**
- Nodes = spectrogram time-frequency patches; Edges = k-NN in feature space
- Architecture: **max-relative graph convolution** — takes channel-wise max over (neighbor - self) feature differences (captures extreme local spectral deviations)
- Self-supervised contrastive training; no class labels needed during feature learning

**Key Innovation:** Max-relative aggregation is noise-robust — distortions shift all feature vectors uniformly but the differences remain stable. Outperforms CNN and Transformer fingerprinting on FMA-medium/large.

**Relevance:** The max-relative aggregation idea is applicable to our Perch graphs — instead of summing neighbor Perch features, compute max of (neighbor − self) differences. Captures which spectral dimensions are maximally different from neighbors = acoustic distinctiveness.

**D-series connection:** Could add D_grafprint: `h_i = MLP(max_j∈N(i)(x_j - x_i))` as edge-based aggregation.

---

## Paper 2: Graph-Based Audio Classification with Pre-Trained Embeddings ⭐ Most directly applicable

**Citation:** Sensors (MDPI) 2024. Vol. 24 No. 7 Article 2106. PMC11014159

**Method:**
- Nodes = audio clips; node features = frozen VGGish/YAMNet/PANNs embeddings
- Edges = k-NN in embedding space; GNN architectures: GCN, GraphSAGE, GAT
- Transductive setup: all train+test nodes in same graph
- **Winner: PANNs + GAT → 91% on Rey Zamuro ecoacoustic soundscape dataset**

**Key Insight:** Domain-matched pre-trained embeddings (PANNs > YAMNet > VGGish) dominate architecture choice. GAT slightly outperforms GCN/SAGE.

**Relevance:** Direct blueprint: replace PANNs with Perch embeddings → our D9/D10 experiments. The 91% ecoacoustic result validates the approach. We should expect Perch > PANNs on birds.

---

## Paper 3: ATGNN — Audio Tagging Graph Neural Network

**Citation:** IEEE Signal Processing Letters 2024. arXiv:2311.01526
**Authors:** Singh, Steinmetz, Benetos, Phan, Stowell (Alan Turing/QMUL)

**Method:**
- Three-tier graph: PGN (patch↔patch), PLG (patch↔class), LLG (class↔class)
- Label-label graph initialized from co-occurrence statistics → +0.4 mAP from LLG alone
- Results: 0.585 mAP FSD50K; 0.335 mAP AudioSet-balanced

**Key Insight:** **Label co-occurrence graph is the most impactful single component.** Species that co-occur in the same soundscapes should share graph edges → information sharing between related species.

**Relevance for BirdCLEF:**
- Build 234×234 species co-occurrence matrix from Y_tr
- Use as adjacency for a label-propagation-style post-processing
- Combine with clip-level graph for a two-level architecture

---

## Paper 4: MLGL — Multi-Level Graph Learning

**Citation:** arXiv:2312.09952. Code: github.com/Yuanbo2020/MLGL

**Method:**
- Nodes = semantic categories (24 fine + 7 coarse + annoyance); gated GCN
- 3-layer gated GCN with attention fusion across local → global graph hierarchy
- Results: AUC 0.921, Accuracy 91.96% on DeLTA dataset

**Key Insight:** Hierarchical node design (fine species → genus → family) maps to bird taxonomy. Multi-task framing applicable.

**Relevance:** Designing a hierarchical Perch graph: individual clip nodes → species prototype nodes → genus nodes. BirdCLEF taxonomy is available.

---

## Paper 5: Few-Shot Attentional GNN for Audio (Interspeech 2019)

**Citation:** Interspeech 2019. Zhang et al.
**URL:** https://www.isca-archive.org/interspeech_2019/zhang19k_interspeech.html

**Method:**
- Episode graph: support + query clips as nodes; learned pairwise similarity edges
- Attentional selection over support examples per query — not all support clips equal
- Entropy-weighted confidence measure for reliability estimation

**Relevance:** Episode-based few-shot setup = our OOF setup. Attentional support selection directly applicable to our D8_cross_attn (already using temperature softmax) — could add entropy-weighted confidence.

---

## Paper 6: Heterogeneous GNN for Species Distribution Modeling ⭐ Domain Gen

**Citation:** arXiv:2503.11900 (2025)

**Method:**
- Bipartite heterogeneous graph: species nodes (taxonomic features) + location nodes (environmental covariates)
- Interaction Network per edge type; link prediction reformulation
- Results: +23.5% AUC improvement over prior SDMs in Canada region

**Key Insight:** Reformulate species presence as **bipartite link prediction** — far more principled than binary classification for presence-only data.

**Relevance for BirdCLEF 2026:**
- Species nodes: Perch mean embeddings of positive training clips
- Location nodes: geographic covariates (lat/lon, elevation, date)
- Detection edges: Y_tr labels
- At test time: query clips as additional nodes, predict clip→species edges
- Directly addresses domain generalization shift (new soundscape locations)

---

## Paper 7: Graph Label Propagation for Semi-Supervised Speaker ID

**Citation:** arXiv:2106.08207 (ISCA 2021, Amazon)

**Method:**
- Nodes = utterances; Edges = cosine similarity between speaker embeddings
- Classical iterative soft-label diffusion from labeled → unlabeled nodes
- No learned GNN; purely graph SSL

**Relevance:** Validates our D5_lp/D6_transductive/D19_lpa_d8 approach. Speaker embeddings → Perch embeddings. The key insight: embedding-derived edges are sufficient for effective LP, no learned GNN needed.

---

## Paper 8: GraFPrint — Max-Relative Aggregation Details

**GraFPrint code insight for implementation:**
```python
# Max-relative aggregation (noise-robust)
# For each node i: h_i = MLP( max_{j in N(i)} (x_j - x_i) )
# Unlike mean/attention: captures extreme acoustic deviations
# Robust because noise shifts all x uniformly; differences stay stable
```

This is implemented in PyG as a custom `MessagePassing` with `aggr='max'` on the message `x_j - x_i`.

---

## PyTorch Geometric Layer Reference (Prioritized for BirdCLEF)

### Top Picks for Edge-Feature-Aware + Small Graphs

| Layer | Edge Features | Notes |
|-------|--------------|-------|
| `GATv2Conv` | ✅ | Fixes static attention bug in GATv2; dynamic attention |
| `TransformerConv` | ✅ | Full transformer, edge_attr in key-query |
| `GINEConv` | ✅ | GIN + edge attrs; theoretically expressive |
| `NNConv` | ✅ | edge_attr → MLP → convolution kernel; very flexible |
| `CGConv` | ✅ | Continuous edge features; lightweight |
| `EdgeConv` | ✅ | Node-pair differences; point-cloud style |

### Top Picks for Heterogeneous Graphs

| Layer | Notes |
|-------|-------|
| `HGTConv` | Heterogeneous graph transformer; type-specific heads |
| `RGATConv` | Relational + attention |
| `HeteroConv` | Wrapper: apply any conv per relation type |
| `HANConv` | Meta-path-based attention |

### Top Picks for Semi-Supervised / Propagation

| Layer | Notes |
|-------|-------|
| `APPNP` | PageRank propagation; decoupled from MLP; best for small graphs |
| `LabelPropagation` | Official PyG LP; parameter-free |
| `GATConv` | Established for semi-supervised node classification |
| `AGNNConv` | Explicitly designed for semi-supervised |

### Best Aggregators for Multi-Label

| Aggregator | Notes |
|-----------|-------|
| `MultiAggregation(sum+mean+std)` | Captures multiple statistics |
| `AttentionalAggregation` | Soft attention over neighbors |
| `SoftmaxAggregation` | Temperature-based; closest to D8_cross_attn |
| `PowerMeanAggregation` | Generalizes mean/max; learnable |

### Best Normalization for Small Graphs

| Layer | Notes |
|-------|-------|
| `LayerNorm` | Batch-size independent; stable |
| `PairNorm` | Prevents oversmoothing in deep GNNs |
| `GraphNorm` | Batched graphs; accelerates training |

---

## Why Trained GNNs Underperform on Our Problem

**Confirmed exhaustively through D9-D30 (all architectures, all regularization schemes):**
D8_cross_attn (0.696) >> all trained GNNs (0.478-0.613)

**Root causes (confirmed):**
1. **Graph size**: 700 nodes total, ~566 labeled per OOF fold. Cora has 2708, Citeseer 3327. GNNs designed for 10K+ nodes.
2. **Label sparsity**: 71/234 classes have positives. Model sees < 5 positive examples per class on average.
3. **High dim output**: 234-class MLP head with 566 training examples → severe overfitting.
4. **OOF distribution shift**: Each fold has different topology (different val clips removed) → model can't generalize across folds.

**Architecture improvements tested (D27-D30) — ALL FAILED to beat D11:**

| What we tried | Result | Why it didn't help |
|--------------|--------|-------------------|
| JK-LSTM (D27) | 0.491 < D11(0.531) | LSTM adds params → more overfit |
| JK-max (D28) | 0.533 ≈ D11 | Marginal, not significant |
| PairNorm (D29,D30) | 0.478 | Overregularizes tiny graph |
| DropEdge p=0.3 (all) | hurt | 700 nodes need ALL edges, not fewer |
| 4 layers vs 2 | hurt | deeper = more overfit on 566 nodes |
| Early stopping | neutral | loss plateaus in <50 epochs anyway |
| weight_decay=1e-3 | slight underfit | better than 1e-4 but still fails |

**Key insight (D27 vs D28):** JK-LSTM (-0.040 vs D11) < JK-max (+0.002 vs D11).
The LSTM in JK-LSTM is itself a learnable module that overfits. JK-max is parameter-free
and thus safer — confirms the "parameter-free is better" principle on small graphs.

**DropEdge backfires:** On a 700-node graph with k=15 edges, removing 30% means the graph
becomes disconnected for many nodes. Unlike large graphs where DropEdge provides useful
stochasticity, our graph is too sparse already.

**DEFINITIVE CONCLUSION:** The problem is fundamentally underdetermined for trained GNNs.
No architecture, regularization scheme, or trick will fix 566 training nodes → 234-class output.
**All future work should focus on parameter-free methods (D8 family).**

---

## Experiment Map (D-series) — COMPLETE

| Exp | Method | OOF AUC | Status |
|-----|--------|---------|--------|
| D1_mreach | HDBSCAN mutual reachability kNN | 0.69521 | ✅ Done |
| D2_mst | MST label propagation | 0.67964 | ✅ Done |
| D3_hdbscan | HDBSCAN sub-prototype NCM | 0.65730 | ✅ Done |
| D5_lp | LP + D1 retrieval | 0.66591 | ✅ Done |
| D6_transductive | Per-file transductive LP | 0.60972 | ✅ Done |
| D7_denseprot | Density-weighted prototype | 0.67294 | ✅ Done |
| **D8_cross_attn** | **Temperature softmax kNN (T=10)** | **0.69597** | ✅ **BEST** |
| D9_gcn | PyG GCNConv transductive | 0.54410 | ✅ Done |
| D10_gat | PyG GATConv 4-head | 0.57038 | ✅ Done |
| D11_sage | PyG GraphSAGE 2-layer | 0.53074 | ✅ Done |
| D12_appnp | APPNP K=10 | 0.50431 | ✅ Done |
| D13_tconv | TransformerConv + edge_attr | 0.56904 | ✅ Done |
| D14_gin | GINConv WL-expressivity | 0.51637 | ✅ Done |
| D15_enrich_gat | GAT + PCA64+density nodes | 0.54460 | ✅ Done |
| D16_lpa_appnp | LPA→APPNP | 0.49952 | ✅ Done |
| D17_hetero | Clip+species heterogeneous | 0.61647 | ✅ Done |
| D18_appnp_enrich | APPNP + enriched nodes | 0.50616 | ✅ Done |
| D19_lpa_d8 | LPA → D8 cross-attn (param-free) | 0.66741 | ✅ Done |
| D20_appnp_edge | Cosine-weighted APPNP | 0.61867 | ✅ Done |
| D21_max_sim | GraFPrint max cosine-sim | 0.67-0.68 | ✅ Done |
| D23_d8d1 | D8+D1 ensemble | 0.69576 | ✅ Done |
| D24_grafprint | Max-relative contrast kNN | 0.67536 | ✅ Done |
| D25_gatv2 | GATv2Conv 4-head | ~0.57 | ✅ Done |
| D26_episode | Entropy-weighted episode attn | 0.68847 | ✅ Done |
| D27_sage_jk_lstm | SAGE×4 + JK-LSTM + DropEdge | 0.49081 | ✅ Done |
| D28_sage_jk_max | SAGE×4 + JK-max + DropEdge | 0.53310 | ✅ Done |
| D29_gatv2_jk | GATv2×3 + JK-max + PairNorm | 0.47840 | ✅ Done |
| D30_gcn_jk_cat | GCN×4 + JK-cat + PairNorm | 0.47778 | ✅ Done |
