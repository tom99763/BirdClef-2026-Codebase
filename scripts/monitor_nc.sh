#!/usr/bin/env bash
# Monitor Noisy Classmate pipeline progress
# Usage: bash scripts/monitor_nc.sh

echo "============================================================"
echo "  Noisy Classmate Pipeline Monitor — $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Pipeline process
echo ""
echo "=== Pipeline Process ==="
if pgrep -f "auto_nc_full" > /dev/null 2>&1; then
    echo "  STATUS: RUNNING (PID $(pgrep -f auto_nc_full | head -1))"
else
    echo "  STATUS: NOT RUNNING"
fi

# Latest pipeline log
echo ""
echo "=== Latest Pipeline Activity ==="
tail -5 outputs/logs/auto_nc_full.log 2>/dev/null || echo "  No log found"

# GPU
echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null

# PVT rounds
echo ""
echo "=== PVT Rounds (NC) ==="
for r in 9 10 11 12 13 14 15; do
    dir="outputs/sed-ns-pvt-20s-r${r}"
    folds_done=0
    best_aucs=""
    for f in 0 1 2 3 4; do
        log="outputs/logs/sed_ns_pvt_r${r}_fold${f}.log"
        if [ -f "${dir}/fold${f}_best.pt" ]; then
            folds_done=$((folds_done + 1))
            auc=$(grep "best_val_auc" "$log" 2>/dev/null | tail -1 | grep -o '[0-9]\.[0-9]*')
            best_aucs="${best_aucs} ${auc:-?}"
        else
            cur=$(grep "ss_auc=" "$log" 2>/dev/null | tail -1 | grep -o 'ss_auc=[0-9.]*')
            if [ -n "$cur" ]; then
                best_aucs="${best_aucs} (${cur})"
                folds_done=-1  # mark as in-progress
            fi
        fi
    done
    npz=""
    [ -f "${dir}/all_ss_probs.npz" ] && npz=" [npz]"
    [ -f "${dir}/all_ss_probs_corrected.npz" ] && npz=" [npz+corr]"
    if [ $folds_done -eq 5 ]; then
        echo "  R${r}: DONE (5/5)${npz} |${best_aucs}"
    elif [ $folds_done -gt 0 ] || [ $folds_done -eq -1 ]; then
        echo "  R${r}: IN PROGRESS |${best_aucs}"
    else
        echo "  R${r}: --"
    fi
done

# B0 rounds (bidirectional)
echo ""
echo "=== B0 Rounds (NC Bidirectional) ==="
for r in 12 13 14 15; do
    dir="outputs/sed-ns-b0-20s-r${r}"
    folds_done=0
    best_aucs=""
    for f in 0 1 2 3 4; do
        log="outputs/logs/sed_ns_b0_r${r}_fold${f}.log"
        if [ -f "${dir}/fold${f}_best.pt" ]; then
            folds_done=$((folds_done + 1))
            auc=$(grep "best_val_auc" "$log" 2>/dev/null | tail -1 | grep -o '[0-9]\.[0-9]*')
            best_aucs="${best_aucs} ${auc:-?}"
        else
            cur=$(grep "ss_auc=" "$log" 2>/dev/null | tail -1 | grep -o 'ss_auc=[0-9.]*')
            if [ -n "$cur" ]; then
                best_aucs="${best_aucs} (${cur})"
            fi
        fi
    done
    npz=""
    [ -f "${dir}/all_ss_probs.npz" ] && npz=" [npz]"
    [ -f "${dir}/all_ss_probs_corrected.npz" ] && npz=" [npz+corr]"
    pseudo=""
    [ -f "pseudo_labels/noisy_classmate_b0_r${r}.csv" ] && pseudo=" [nc_pseudo]"
    if [ $folds_done -eq 5 ]; then
        echo "  R${r}: DONE (5/5)${npz}${pseudo} |${best_aucs}"
    elif [ $folds_done -gt 0 ]; then
        echo "  R${r}: IN PROGRESS (${folds_done}/5) |${best_aucs}"
    else
        echo "  R${r}: --"
    fi
done

# NC Pseudo labels generated
echo ""
echo "=== NC Pseudo Labels ==="
ls -lh pseudo_labels/noisy_classmate_*.csv 2>/dev/null | awk '{print "  "$NF, $5, $6, $7}' || echo "  None"

echo ""
echo "============================================================"
