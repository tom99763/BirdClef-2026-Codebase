---
tags: [data, dataset, domain-gap]
last-updated: 2026-04-08
---

# Dataset Description

## Competition Data

BirdCLEF 2026 provides bioacoustic recordings for 234 species across 5 taxonomic classes.

### Data Sources

| Source | Count | Format | Labels |
|--------|-------|--------|--------|
| `train_audio/` | ~24,000 clips | Per-species folders, variable length | Weak per-clip labels (primary + secondary) |
| `train_soundscapes/` | ~1,479 labeled windows (66 files) | 60s OGG recordings, 5s annotation windows | Sparse ground-truth labels |
| `train_soundscapes/` (unlabeled) | ~119,000 windows | Same 60s recordings | No labels (pseudo-labeled by us) |
| `test_soundscapes/` | Unknown | 60s recordings | Hidden (Kaggle evaluation) |
| `taxonomy.csv` | 234 species | CSV | Species list with taxonomic metadata |

### Taxonomic Distribution

| Class | Count | Notes |
|-------|-------|-------|
| Aves (birds) | ~200 | Majority class; SED performs well |
| Amphibia | ~15 | SED struggles; Perch much better |
| Insecta | ~10 | SED struggles; Perch much better |
| Mammalia | ~5 | Largest SED-Perch gap (-0.13 AUC) |
| Reptilia | ~4 | SED struggles; Perch much better |

### Label Format

**train_audio**: `train.csv` with `primary_label` and `secondary_labels` columns. Secondary labels encoded at 0.5 weight (soft secondary).

**train_soundscapes**: `train_soundscapes_labels.csv` with per-5s-window annotations. Sparse -- most windows have no annotation.

**Pseudo labels**: `pseudo_labels/*.csv` with `row_id`, 234 species columns (soft probs), `primary_label`, and `secondary_labels`.

## Domain Gap

This is the central challenge. The correct framing is **Domain Generalization** (not SFDA), because `test_soundscapes/` contains no audio for adaptation.

### Source Domain (train_audio)
- Isolated point recordings
- Single species per clip (mostly)
- Clean, close-range recordings
- Controlled recording conditions

### Target Domain (soundscapes)
- Multi-species overlapping calls
- Background noise (wind, rain, traffic, insects)
- Variable recording distance
- 60-second continuous recordings
- Different microphone characteristics

### How We Bridge the Gap

| Technique | Implementation |
|-----------|---------------|
| Pseudo-labeled soundscapes | NS/NC framework trains on ~119K pseudo-labeled soundscape windows |
| Cross-domain MixUp | Each labeled clip mixed 1:1 with pseudo soundscape clip (lambda=0.5) |
| 20s clip duration | Longer context captures soundscape characteristics |
| Residual Corrector | Corrects SED predictions using Perch teacher signal |
| Non-Aves Perch-only | For non-Aves taxa, pseudo labels use pure Perch (SED unreliable) |

## Validation Strategy

5-fold GroupKFold on the 66 labeled soundscape files, grouped by `file_id` (extracted from filename). Same split as used in the SSM baseline for fair comparison.

- **Training**: ALL train_audio + pseudo soundscapes (excluding val files) + labeled soundscapes (training split)
- **Validation**: Labeled soundscape windows from held-out fold
- **Metric**: Macro-averaged ROC-AUC on validation fold

Important: Validation AUC on labeled soundscapes does not reliably predict LB performance (see [[ensemble-diversity]] GT Paradox).

## File ID Format

- **Row ID**: `{soundscape_stem}_{end_sec}` (e.g., `BC2026_Train_0001_S08_20260115_060000_25`)
- **Filename**: `{soundscape_stem}.ogg` (e.g., `BC2026_Train_0001_S08_20260115_060000.ogg`)
- **File ID group**: Third field after splitting by `_` (e.g., `0001`)

## Pseudo Label Health

Tracked across NS rounds:
- Coverage: R1 103,628 windows -> R7 ~120,000 (+16%)
- Unique species labeled: 119 -> 177 (+49%)
- Mean confidence: 0.614 -> 0.586 (slight decrease = "conservative expansion", healthy)

## Related Pages

- [[noisy-student]] -- How pseudo labels are generated and refined
- [[augmentation]] -- Data augmentation for domain bridging
- [[ensemble-diversity]] -- GT Paradox and why val AUC is unreliable
