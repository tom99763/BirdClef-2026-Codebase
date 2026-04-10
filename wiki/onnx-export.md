---
tags: [onnx, export, quantization, inference]
last-updated: 2026-04-08
---

# ONNX Export Pipeline

Models are exported from PyTorch to ONNX for use in the Kaggle submission notebook. The export script is `scripts/export_sed_to_onnx.py`.

## Export Architecture

The ONNX model takes mel spectrograms as input (not raw waveforms), because `torch.stft` cannot be exported to ONNX. MelTransform is reimplemented in the Kaggle notebook.

```
ONNX Model:
  Input:  mel spectrogram  (B, 3, 224, T)  float32
  Output: probabilities    (B, 234)         float32

Kaggle Notebook:
  waveform -> MelTransform (Python) -> ONNX model -> probs
```

## Export Commands

### Single Model

```bash
python3 scripts/export_sed_to_onnx.py \
    --pt  outputs/sed-ns-pvt-20s-r9/fold4_best.pt \
    --out "weights/sed/sed_ns_pvt_r9_fold4.onnx" \
    --backbone pvt_v2_b0
```

### Verify Export

```bash
python3 scripts/export_sed_to_onnx.py \
    --pt outputs/sed-ns-b0-20s-r12/fold0_best.pt \
    --out weights/sed/sed_ns_b0_r12_fold0.onnx \
    --verify
```

Verification compares PyTorch and ONNX outputs on random input; max abs diff should be < 1e-4.

## FP32 Export Details

- **Opset version**: 17
- **Dynamic axes**: batch dimension and time dimension are dynamic
- **Constant folding**: enabled for optimization
- Input names: `['mel']`; output names: `['probs']`

```python
torch.onnx.export(
    model, dummy,
    onnx_path,
    input_names=['mel'],
    output_names=['probs'],
    dynamic_axes={'mel': {0: 'batch', 3: 'time'}, 'probs': {0: 'batch'}},
    opset_version=17,
    do_constant_folding=True,
)
```

## INT8 Quantization

After FP32 export, dynamic INT8 quantization is applied using ONNX Runtime:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
```

This reduces model size by ~4x while maintaining acceptable accuracy. The Kaggle notebook can use either FP32 or INT8 models.

### Size Comparison

| Model | FP32 | INT8 |
|-------|------|------|
| EfficientNet-B0 | ~20 MB | ~5 MB |
| PVT-v2-B0 | ~14.6 MB | ~4 MB |

## Checkpoint Loading

The export script handles various checkpoint formats:

```python
ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
# Handle legacy key naming
if any('freq_pool' in k for k in state):
    state = {k.replace('freq_pool', 'gem_pool'): v for k, v in state.items()}
```

## Kaggle Notebook Configuration

The submission notebook loads multiple ONNX models and ensembles their predictions:

```python
SED_CHECKPOINTS = [
    {'name': 'b0_0',  'onnx_path': '.../sed_ns_b0_r12_fold0.onnx',
     'backbone': 'tf_efficientnet_b0.ns_jft_in1k', 'weight': 1.0, 'clip_sec': 20},
    {'name': 'pvt_2', 'onnx_path': '.../sed_ns_pvt_r5_fold2.onnx',
     'backbone': 'pvt_v2_b0', 'weight': 1.0, 'clip_sec': 20},
    {'name': 'b0_3',  'onnx_path': '.../sed_ns_r6_fold3.onnx',
     'backbone': 'tf_efficientnet_b0.ns_jft_in1k', 'weight': 1.0, 'clip_sec': 20},
]
```

### Inference Pipeline in Notebook

1. Load 60s soundscape audio
2. Slice into overlapping 20s windows (stride=5s, producing 9 windows)
3. For each window: MelTransform -> ONNX inference -> (1, 234) probs
4. Aggregate: average predictions across windows for each 5s output slot
5. Ensemble: average across all models (equal weight)
6. Non-Aves override: use Perch/CLAP predictions for non-Aves taxa

## Model Soup (Alternative)

Model soup (averaging state_dicts across folds) can also be exported:

```bash
# Produces a single soup model from all 5 folds
# Location: weights/sed/sed_ns_b0_r{R}_soup.onnx
```

However, prediction-level ensemble of individual fold models outperforms soup in practice, because it preserves model diversity. Soup is dragged down by weaker folds (e.g., B0 fold0 at ~0.90 vs fold3 at ~0.96).

## Related Pages

- [[architecture-sed-model]] -- Model architecture being exported
- [[lb-experiments]] -- Which ONNX models are used in submissions
- [[ensemble-diversity]] -- Why individual fold models > soup
