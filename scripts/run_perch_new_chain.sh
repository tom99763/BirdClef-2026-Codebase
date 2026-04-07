#!/bin/bash
# run_perch_new_chain.sh — Perch New 三階段實驗鏈
#
# 順序:
#   A) proto_head  — 替換 MLP → 原型頭 (從頭訓練)
#   B) protocl     — 原型頭 + ProtoCLR 跨域對比  (熱啟動 A)
#   C) fixmatch    — 原型頭 + FixMatch 半監督     (熱啟動 A)
#
# 使用方式:
#   bash scripts/run_perch_new_chain.sh            # 完整三個實驗
#   bash scripts/run_perch_new_chain.sh proto_head # 只跑 A
#   bash scripts/run_perch_new_chain.sh protocl    # 只跑 B (需要 A 已完成)
#   bash scripts/run_perch_new_chain.sh fixmatch   # 只跑 C (需要 A 已完成)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

GPU="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES="$GPU"
echo "[perch_new_chain] 使用 GPU: $GPU"

LOG_DIR="outputs/perch_new/logs"
mkdir -p "$LOG_DIR" weights/perch_new

run_exp() {
    local NAME="$1"
    local CFG="configs/perch_new/${NAME}.yaml"
    local LOG="$LOG_DIR/${NAME}_$(date +%Y%m%d_%H%M%S).log"
    echo ""
    echo "========================================"
    echo "[perch_new_chain] 開始實驗: $NAME"
    echo "  Config: $CFG"
    echo "  Log:    $LOG"
    echo "========================================"
    CUDA_VISIBLE_DEVICES="$GPU" python scripts/train_perch_new.py \
        --config "$CFG" 2>&1 | tee "$LOG"
    echo "[perch_new_chain] ✅ 完成: $NAME"
}

TARGET="${1:-all}"

case "$TARGET" in
    proto_head)
        run_exp proto_head
        ;;
    protocl)
        run_exp protocl
        ;;
    fixmatch)
        run_exp fixmatch
        ;;
    all)
        run_exp proto_head
        run_exp protocl
        run_exp fixmatch
        ;;
    *)
        echo "未知實驗: $TARGET"
        echo "可選: proto_head | protocl | fixmatch | all"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "[perch_new_chain] 全部完成！"
echo "  結果存於: weights/perch_new/"
ls -lh weights/perch_new/ 2>/dev/null
echo "========================================"
