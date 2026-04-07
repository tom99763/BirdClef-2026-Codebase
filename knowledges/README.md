# BirdCLEF 2026 — Knowledge Base

## Context

Our Perch branch uses 1536-dim frozen embeddings from Google's Perch v2 model.
Current probe: StandardScaler → PCA(64) → LogisticRegression per class = **0.918 LB**

This knowledge base collects papers and techniques for improving few-shot classification
on top of fixed pre-trained embeddings.

## Files

| File | Topic |
|------|-------|
| `fewshot_foundations.md` | SimpleShot, LaplacianShot, prototypical networks |
| `tip_adapter_family.md` | Tip-Adapter, APE, LP++ — CLIP-era few-shot adapters |
| `transductive_methods.md` | Transductive inference, graph label propagation |
| `bioacoustics_transfer.md` | Domain-specific papers on bird/audio embedding transfer |
| `advanced_embedding_methods.md` | Distribution calibration, TIM, subspace ensemble, Mahalanobis |
| `implementation_notes.md` | Code-level notes on what we tried and results |
| `birdclef2025_solutions.md` | BirdCLEF 2025 top-5 solutions: noisy student, TTA, power scaling, SoftAUC |
| `perch_trainable_methods.md` | Trainable extensions: MLP adapter, SED-emb probe, pseudo expansion |
| `embedding_propagation_ptmap.md` | Graph smoothing (EP), power transform (PT-MAP), Perch 2.0 K-prototype |
