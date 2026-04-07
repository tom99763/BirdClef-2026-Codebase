#!/bin/bash
# Run after v15 finishes: soup + holdout eval on GPU0
cd /home/lab/BirdClef-2026-Codebase
GPU=0
LOG="outputs/chain3.log"
log() { echo "[$(date '+%H:%M:%S')][POST-V15] $*" | tee -a "$LOG"; }

log "Waiting for v15 to finish..."
while true; do
    f="outputs/sed-b0-v15-no-sec/result.json"
    [ -f "$f" ] && finished=$(python3 -c "import json; print(json.load(open('$f')).get('finished',False))" 2>/dev/null) || finished=False
    [ "$finished" = "True" ] && break
    sleep 120
done

log "v15 finished! Running soup + holdout..."
python3 scripts/model_soup.py --run sed-b0-v15-no-sec --config configs/sed_b0_v15_no_sec.yaml 2>&1 | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval_sed_holdout.py \
    --checkpoint checkpoints/sed-b0-v15-no-sec/best_sed.pt \
    --config configs/sed_b0_v15_no_sec.yaml \
    --run_name sed-b0-v15-no-sec 2>&1 | tee -a "$LOG"
if [ -f "checkpoints/sed-b0-v15-no-sec/soup_sed.pt" ]; then
    CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval_sed_holdout.py \
        --checkpoint checkpoints/sed-b0-v15-no-sec/soup_sed.pt \
        --config configs/sed_b0_v15_no_sec.yaml \
        --run_name sed-b0-v15-no-sec-soup 2>&1 | tee -a "$LOG"
fi
log "v15 post-processing complete."
