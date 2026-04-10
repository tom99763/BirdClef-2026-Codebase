#!/usr/bin/env bash
# Monitor Noisy Classmate pipeline progress (supports _nc directories)
# Usage: bash scripts/monitor_nc.sh

echo "============================================================"
echo "  Noisy Classmate Pipeline Monitor — $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Pipeline process
echo ""
echo "=== Pipeline Process ==="
if pgrep -f "auto_nc_dual_gpu" > /dev/null 2>&1; then
    echo "  STATUS: RUNNING (PID $(pgrep -f auto_nc_dual_gpu | head -1))"
elif pgrep -f "auto_nc_full" > /dev/null 2>&1; then
    echo "  STATUS: RUNNING (PID $(pgrep -f auto_nc_full | head -1))"
else
    echo "  STATUS: NOT RUNNING"
fi

# Watchdog
if pgrep -f "watchdog_nc" > /dev/null 2>&1; then
    echo "  WATCHDOG: RUNNING"
else
    echo "  WATCHDOG: NOT RUNNING"
fi

# Latest pipeline log
echo ""
echo "=== Latest Pipeline Activity ==="
tail -5 outputs/logs/auto_nc_dual_gpu.log 2>/dev/null || \
    tail -5 outputs/logs/auto_nc_full.log 2>/dev/null || echo "  No log found"

# GPU
echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null

# PVT rounds — check both NS dir and NC dir
echo ""
echo "=== PVT Rounds ==="
for r in 5 6 7 8 9 10 11 12 13 14 15; do
    # Try NC dir first, then NS dir
    nc_dir="outputs/sed-ns-pvt-20s-r${r}-nc"
    ns_dir="outputs/sed-ns-pvt-20s-r${r}"
    if [ -d "$nc_dir" ]; then
        dir="$nc_dir"; tag="NC"
    elif [ -d "$ns_dir" ]; then
        dir="$ns_dir"; tag="NS"
    else
        continue
    fi

    folds_done=0; best_aucs=""
    for f in 0 1 2 3 4; do
        if [ -f "${dir}/fold${f}_best.pt" ]; then
            folds_done=$((folds_done + 1))
            # Try NC log, then NS log
            log_nc="outputs/logs/sed_ns_pvt_r${r}_nc_fold${f}.log"
            log_ns="outputs/logs/sed_ns_pvt_r${r}_fold${f}.log"
            log_f="$log_nc"; [ ! -f "$log_f" ] && log_f="$log_ns"
            auc=$(grep "best_val_auc" "$log_f" 2>/dev/null | tail -1 | grep -o '[0-9]\.[0-9]*')
            [ -z "$auc" ] && auc=$(grep "New best" "$log_f" 2>/dev/null | tail -1 | grep -o '[0-9]\.[0-9]*')
            best_aucs="${best_aucs} ${auc:-?}"
        else
            log_nc="outputs/logs/sed_ns_pvt_r${r}_nc_fold${f}.log"
            log_ns="outputs/logs/sed_ns_pvt_r${r}_fold${f}.log"
            log_f="$log_nc"; [ ! -f "$log_f" ] && log_f="$log_ns"
            cur=$(grep "ss_auc=" "$log_f" 2>/dev/null | tail -1 | grep -o 'ss_auc=[0-9.]*')
            if [ -n "$cur" ]; then
                best_aucs="${best_aucs} (${cur})"
            fi
        fi
    done

    npz=""
    [ -f "${dir}/all_ss_probs.npz" ] && npz=" [npz]"
    [ -f "${dir}/all_ss_probs_corrected.npz" ] && npz=" [npz+corr]"

    if [ $folds_done -eq 5 ]; then
        echo "  R${r} [${tag}]: DONE (5/5)${npz} |${best_aucs}"
    elif [ $folds_done -gt 0 ]; then
        echo "  R${r} [${tag}]: ${folds_done}/5${npz} |${best_aucs}"
    elif [ -n "$best_aucs" ]; then
        echo "  R${r} [${tag}]: training |${best_aucs}"
    fi
done

# B0 rounds — check both NS dir and NC dir
echo ""
echo "=== B0 Rounds ==="
for r in 8 9 10 11 12 13 14 15; do
    nc_dir="outputs/sed-ns-b0-20s-r${r}-nc"
    ns_dir="outputs/sed-ns-b0-20s-r${r}"
    if [ -d "$nc_dir" ]; then
        dir="$nc_dir"; tag="NC"
    elif [ -d "$ns_dir" ]; then
        dir="$ns_dir"; tag="NS"
    else
        continue
    fi

    folds_done=0; best_aucs=""
    for f in 0 1 2 3 4; do
        if [ -f "${dir}/fold${f}_best.pt" ]; then
            folds_done=$((folds_done + 1))
            log_nc="outputs/logs/sed_ns_b0_r${r}_nc_fold${f}.log"
            log_ns="outputs/logs/sed_ns_20s_r${r}_fold${f}.log"
            log_f="$log_nc"; [ ! -f "$log_f" ] && log_f="$log_ns"
            auc=$(grep "best_val_auc" "$log_f" 2>/dev/null | tail -1 | grep -o '[0-9]\.[0-9]*')
            [ -z "$auc" ] && auc=$(grep "New best" "$log_f" 2>/dev/null | tail -1 | grep -o '[0-9]\.[0-9]*')
            best_aucs="${best_aucs} ${auc:-?}"
        else
            log_nc="outputs/logs/sed_ns_b0_r${r}_nc_fold${f}.log"
            log_ns="outputs/logs/sed_ns_20s_r${r}_fold${f}.log"
            log_f="$log_nc"; [ ! -f "$log_f" ] && log_f="$log_ns"
            cur=$(grep "ss_auc=" "$log_f" 2>/dev/null | tail -1 | grep -o 'ss_auc=[0-9.]*')
            if [ -n "$cur" ]; then
                best_aucs="${best_aucs} (${cur})"
            fi
        fi
    done

    npz=""
    [ -f "${dir}/all_ss_probs.npz" ] && npz=" [npz]"
    [ -f "${dir}/all_ss_probs_corrected.npz" ] && npz=" [npz+corr]"

    if [ $folds_done -eq 5 ]; then
        echo "  R${r} [${tag}]: DONE (5/5)${npz} |${best_aucs}"
    elif [ $folds_done -gt 0 ]; then
        echo "  R${r} [${tag}]: ${folds_done}/5${npz} |${best_aucs}"
    elif [ -n "$best_aucs" ]; then
        echo "  R${r} [${tag}]: training |${best_aucs}"
    fi
done

# NC Pseudo labels
echo ""
echo "=== NC Pseudo Labels ==="
ls -lh pseudo_labels/noisy_classmate_*.csv 2>/dev/null | awk '{print "  "$NF, $5, $6, $7}' || echo "  None"

echo ""
echo "============================================================"
