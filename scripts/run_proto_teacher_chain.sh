#!/usr/bin/env bash
# Master script: ProtoSSM teacher → pseudo labels → SED student chain (20s, R1-R4)
#
# Steps:
#   1. Extract Perch 1536-dim embeddings for ALL soundscapes
#   2. Train ProtoSSM on labeled soundscapes, generate pseudo_labels/ns_r0_protossm.csv
#   3. Launch SED-20s noisy student chain (fold0+1 / fold2+3 / fold4 in parallel)
#
# Usage:
#   nohup bash scripts/run_proto_teacher_chain.sh > outputs/logs/proto_teacher_chain.log 2>&1 &

set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
LOG="outputs/logs"
mkdir -p "$LOG" pseudo_labels outputs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [PROTO-CHAIN] $*"; }

# ── Step 1: Extract Perch embeddings for ALL soundscapes ──────────────────────
EMB_NPZ="outputs/perch_emb_all_ss.npz"
if [ -f "$EMB_NPZ" ]; then
    log "Perch embeddings already exist: ${EMB_NPZ} — skipping extraction"
else
    log "Extracting Perch embeddings for all soundscapes (~45-90 min on GPU) ..."
    python3 scripts/extract_perch_emb_all_ss.py \
        --output      "$EMB_NPZ" \
        --batch_files 8 \
        --save_every  500 \
        > "${LOG}/extract_perch_emb_all_ss.log" 2>&1
    log "Extraction done → ${EMB_NPZ}"
fi

# ── Step 2: Train ProtoSSM teacher + generate pseudo labels ───────────────────
PSEUDO_R0="pseudo_labels/ns_r0_protossm.csv"
if [ -f "$PSEUDO_R0" ]; then
    log "Pseudo labels already exist: ${PSEUDO_R0} — skipping teacher"
else
    log "Training ProtoSSM teacher + generating pseudo labels ..."
    python3 scripts/gen_proto_teacher_pseudo.py \
        --labeled_npz  outputs/perch_labeled_ss.npz \
        --all_ss_npz   "$EMB_NPZ" \
        --clip_sec     20 \
        --perch_w      0.55 \
        --ssm_w        0.45 \
        --out          "$PSEUDO_R0" \
        --save_model   outputs/proto_ssm_teacher.pt \
        > "${LOG}/gen_proto_teacher_pseudo.log" 2>&1
    log "Pseudo labels ready: ${PSEUDO_R0}"
fi

ROWS=$(wc -l < "$PSEUDO_R0" 2>/dev/null || echo 0)
log "ns_r0_protossm.csv: ${ROWS} rows"

# Ensure R1 config points to the correct pseudo labels
sed -i "s|pseudo_labels_csv:.*|pseudo_labels_csv:      ${PSEUDO_R0}|" \
    configs/sed_ns_b0_20s_r1.yaml
log "configs/sed_ns_b0_20s_r1.yaml updated → pseudo_labels_csv: ${PSEUDO_R0}"

# ── Step 3: SED-only noisy student chain ──────────────────────────────────────
log "Launching SED-20s noisy student chain (R1-R4, fold pairs in parallel) ..."
bash scripts/auto_sed_ns_20s_full.sh \
    >> "${LOG}/auto_sed_ns_20s_full.log" 2>&1

log "════════════════════════════════════════"
log "  PROTO-TEACHER CHAIN COMPLETE"
log "════════════════════════════════════════"
