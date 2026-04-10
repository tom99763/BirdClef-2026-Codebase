#!/usr/bin/env bash
# NC monitoring job — runs every 10 min for 3 days then self-terminates
# Usage: nohup bash scripts/monitor_nc_job.sh > outputs/logs/nc_monitor_job.log 2>&1 &

EXPIRE_AT=$(($(date +%s) + 3*24*3600))  # 3 days from now
INTERVAL=600  # 10 minutes

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Monitor started. Expires: $(date -d @$EXPIRE_AT '+%Y-%m-%d %H:%M:%S')"

while [ $(date +%s) -lt $EXPIRE_AT ]; do
    echo ""
    echo "================================================================"
    echo "  NC Monitor — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"

    # Pipeline alive?
    if pgrep -f "auto_nc_dual_gpu" > /dev/null 2>&1; then
        echo "  Pipeline: RUNNING"
    else
        echo "  Pipeline: *** DEAD ***"
    fi

    # GPU
    echo "  GPU0: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | tr ',' '/' )% / MB"
    echo "  GPU1: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i 1 2>/dev/null | tr ',' '/' )% / MB"

    # Latest pipeline action
    echo "  Last action: $(tail -1 outputs/logs/auto_nc_dual_gpu.log 2>/dev/null)"

    # Active training
    ACTIVE_LOG=$(ls -t outputs/logs/sed_ns_*_nc_fold*.log outputs/logs/sed_ns_*_nc_infer.log outputs/logs/sed_corrector_*_nc.log 2>/dev/null | head -1)
    if [ -n "$ACTIVE_LOG" ]; then
        LAST=$(grep "ss_auc=\|New best\|Inference\|corrector" "$ACTIVE_LOG" 2>/dev/null | tail -1 | head -c 120)
        echo "  Active: $(basename $ACTIVE_LOG) → $LAST"
    fi

    # Summary of completed NC rounds
    echo "  --- NC Rounds ---"
    for arch in pvt b0; do
        for r in 9 10 11 12 13 14 15; do
            dir="outputs/sed-ns-${arch}-20s-r${r}-nc"
            [ ! -d "$dir" ] && continue
            done=$(ls ${dir}/fold*_best.pt 2>/dev/null | wc -l)
            npz=""; [ -f "${dir}/all_ss_probs_corrected.npz" ] && npz="+corr"
            [ -f "${dir}/all_ss_probs.npz" ] && [ -z "$npz" ] && npz="+npz"
            aucs=""
            for f in 0 1 2 3 4; do
                log="outputs/logs/sed_ns_${arch}_r${r}_nc_fold${f}.log"
                a=$(grep "best_val_auc" "$log" 2>/dev/null | tail -1 | grep -o '[0-9]\.[0-9]*')
                [ -z "$a" ] && a=$(grep "New best" "$log" 2>/dev/null | tail -1 | grep -o '[0-9]\.[0-9]*')
                aucs="$aucs ${a:-.}"
            done
            echo "  ${arch} R${r} NC: ${done}/5${npz} |$aucs"
        done
    done

    sleep $INTERVAL
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Monitor expired after 3 days."
