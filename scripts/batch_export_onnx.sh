#!/bin/bash
# Batch export all missing ONNX (FP32 + INT8) for NS and NC models
# Output dir: birdclef-2026/notebook resource/new direction/weights/sed/

set -e
cd /home/lab/BirdClef-2026-Codebase
PYTHON=/home/lab/miniconda3/envs/tom/bin/python

OUT_DIR="birdclef-2026/notebook resource/new direction/weights/sed"
SCRIPT="scripts/export_sed_to_onnx.py"
CONDA_ENV="tom"

export_model() {
    local pt_path="$1"
    local onnx_name="$2"
    local backbone="$3"

    local fp32_path="${OUT_DIR}/${onnx_name}.onnx"
    local int8_path="${OUT_DIR}/${onnx_name}_int8.onnx"

    if [ ! -f "$pt_path" ]; then
        echo "  SKIP (no checkpoint): $pt_path"
        return
    fi

    # Export FP32 if missing
    if [ ! -f "$fp32_path" ]; then
        echo "  Exporting FP32: $onnx_name"
        if [ -n "$backbone" ]; then
            $PYTHON "$SCRIPT" --pt "$pt_path" --out "$fp32_path" --fp32 --backbone "$backbone"
        else
            $PYTHON "$SCRIPT" --pt "$pt_path" --out "$fp32_path" --fp32
        fi
    fi

    # Export INT8 if missing
    if [ ! -f "$int8_path" ]; then
        if [ -f "$fp32_path" ]; then
            echo "  Quantizing INT8: $onnx_name"
            $PYTHON -c "
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic('${fp32_path}', '${int8_path}', weight_type=QuantType.QInt8)
print(f'  Done: ${int8_path}')
"
        else
            echo "  SKIP INT8 (no FP32): $onnx_name"
        fi
    fi
}

echo "============================================"
echo "  Batch ONNX Export — $(date)"
echo "============================================"

# ── B0 NS rounds (old naming: sed_ns_r{R}) ──
echo ""
echo "=== B0 NS (old naming) ==="
for R in 1 2; do
    for F in 0 1 2 3 4; do
        export_model "outputs/sed-ns-b0-20s-r${R}/fold${F}_best.pt" "sed_ns_b0_r${R}_fold${F}" ""
    done
done

# R7 old naming (sed_ns_r7)
echo ""
echo "=== B0 R7 NS (old naming, INT8 only) ==="
for F in 0 2 3 4; do
    fp32="$OUT_DIR/sed_ns_r7_fold${F}.onnx"
    int8="$OUT_DIR/sed_ns_r7_fold${F}_int8.onnx"
    if [ -f "$fp32" ] && [ ! -f "$int8" ]; then
        echo "  Quantizing INT8: sed_ns_r7_fold${F}"
        $PYTHON -c "
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic('${fp32}', '${int8}', weight_type=QuantType.QInt8)
print(f'  Done: ${int8}')
"
    fi
done

# B0 R8 NS
echo ""
echo "=== B0 R8 NS (INT8 only) ==="
for F in 0 1 2 3 4; do
    fp32="$OUT_DIR/sed_ns_b0_r8_fold${F}.onnx"
    int8="$OUT_DIR/sed_ns_b0_r8_fold${F}_int8.onnx"
    if [ -f "$fp32" ] && [ ! -f "$int8" ]; then
        echo "  Quantizing INT8: sed_ns_b0_r8_fold${F}"
        $PYTHON -c "
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic('${fp32}', '${int8}', weight_type=QuantType.QInt8)
print(f'  Done: ${int8}')
"
    fi
done

# ── B0 NC rounds ──
echo ""
echo "=== B0 NC ==="
for R in 12 13 14 15; do
    for F in 0 1 2 3 4; do
        export_model "outputs/sed-ns-b0-20s-r${R}-nc/fold${F}_best.pt" "sed_ns_b0_r${R}_nc_fold${F}" ""
    done
done

# ── PVT NS rounds (INT8 only for existing FP32) ──
echo ""
echo "=== PVT NS (INT8 only) ==="
for R in 1 2 3 4 5 7 8; do
    for F in 0 1 2 3 4; do
        fp32="$OUT_DIR/sed_ns_pvt_r${R}_fold${F}.onnx"
        int8="$OUT_DIR/sed_ns_pvt_r${R}_fold${F}_int8.onnx"
        if [ -f "$fp32" ] && [ ! -f "$int8" ]; then
            echo "  Quantizing INT8: sed_ns_pvt_r${R}_fold${F}"
            $PYTHON -c "
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic('${fp32}', '${int8}', weight_type=QuantType.QInt8)
print(f'  Done: ${int8}')
"
        fi
    done
done

# ── PVT NC rounds ──
echo ""
echo "=== PVT NC ==="
for R in 10 11 12 13; do
    for F in 0 1 2 3 4; do
        export_model "outputs/sed-ns-pvt-20s-r${R}-nc/fold${F}_best.pt" "sed_ns_pvt_r${R}_nc_fold${F}" "pvt_v2_b0"
    done
done

echo ""
echo "============================================"
echo "  Batch export complete — $(date)"
echo "============================================"
