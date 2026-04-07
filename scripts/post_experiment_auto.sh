#!/bin/bash
# ============================================================
# BirdCLEF 2026 — Post-Experiment Automation
#
# Called when a training experiment finishes.
# Args: $1 = experiment name, $2 = best_auc, $3 = best_epoch, $4 = gpu_id
#
# Actions:
# 1. Write conclusion to knowledge file
# 2. Trigger Claude to: review knowledge → search papers → design next exp
# 3. Launch next experiment if GPU is free
# ============================================================

EXP_NAME="${1:-unknown}"
BEST_AUC="${2:-0.0}"
BEST_EPOCH="${3:-0}"
GPU_ID="${4:-0}"

cd /home/lab/BirdClef-2026-Codebase
LOG="outputs/post_experiment.log"
mkdir -p outputs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')][POST-EXP] $*" | tee -a "$LOG"; }

log "=== Post-experiment triggered: $EXP_NAME (best_auc=$BEST_AUC @ ep$BEST_EPOCH, GPU$GPU_ID) ==="

# ── 1. Append conclusion to knowledge ────────────────────────────────────────
TIMESTAMP=$(date '+%Y-%m-%d %H:%M')
cat >> knowledges/experiment_conclusions.md << EOF

---

## $EXP_NAME (Auto-recorded $TIMESTAMP)

**Best val ROC-AUC**: $BEST_AUC @ epoch $BEST_EPOCH
**GPU**: $GPU_ID

*[Detailed analysis to be added by Claude post-experiment review]*

EOF

log "Conclusion appended to knowledges/experiment_conclusions.md"

# ── 2. Check if GPU is free ───────────────────────────────────────────────────
sleep 10  # give process time to fully exit

gpu_free() {
    local gid=$1
    local procs
    procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader --id=$gid 2>/dev/null | wc -l)
    [ "$procs" -eq 0 ]
}

if gpu_free $GPU_ID; then
    log "GPU$GPU_ID is free — marking available for next experiment"
    echo "$GPU_ID" >> outputs/free_gpus.txt
else
    log "GPU$GPU_ID still busy — next experiment must wait"
fi

log "Post-experiment script complete for $EXP_NAME"
log "Next step: run Claude with /loop to review knowledge and design next experiment"
