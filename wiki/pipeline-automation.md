---
tags: [automation, scripts, pipeline, gpu]
last-updated: 2026-04-08
---

# Pipeline Automation

The NS and NC training pipelines are fully automated with bash scripts that handle multi-round training, inference, correction, and pseudo label generation.

## NS Automation Scripts

### B0 Chain

| Script | Rounds | GPU | Notes |
|--------|--------|-----|-------|
| `scripts/auto_sed_ns_20s_full.sh` | R1-R4 | GPU1 | Sequential folds, sequential rounds |
| `scripts/auto_sed_ns_20s_r5r8.sh` | R5-R8 | GPU1 | Extension |
| `scripts/auto_sed_ns_20s_r9r15.sh` | R9-R15 | GPU1 | Extended chain |

### PVT Chain

| Script | Rounds | GPU | Notes |
|--------|--------|-----|-------|
| `scripts/auto_sed_ns_pvt_20s_r1r4.sh` | R1-R4 | GPU1 | Seeded from B0 R4 pseudo |
| `scripts/auto_sed_ns_pvt_20s_r5r8.sh` | R5-R8 | GPU1 | Extension |

### Per-Round Pipeline

Each round in the NS scripts executes:

```bash
# 1. Train 5 folds sequentially
for F in 0 1 2 3 4; do
    train_fold $R $F
done

# 2. Infer all soundscapes (5-fold ensemble)
python3 train_sed_ns.py --config ... --infer_all_ss

# 3. Train Residual Corrector -> all_ss_probs_corrected.npz
python3 scripts/train_sed_residual_corrector.py --sed_dir ... --alpha 0.40

# 4. Generate pseudo labels for next round
python3 scripts/gen_pseudo_ns.py --round $R ...

# 5. Update next round config (sed -i on YAML)
```

### Checkpoint Skip Logic

Every step checks for existing outputs before running:

```bash
if [ -f "$CKPT" ]; then
    log "R${R} fold${F}: checkpoint exists, skipping"
    return 0
fi
```

This makes the pipeline **idempotent**: it can be safely restarted after crashes.

## NC Dual-GPU Automation

### `scripts/auto_nc_dual_gpu.sh`

The NC pipeline uses both GPUs for parallel fold training:

```
Wave 1: fold0 (GPU0) + fold1 (GPU1)  -- parallel
Wave 2: fold2 (GPU0) + fold3 (GPU1)  -- parallel
Wave 3: fold4 (GPU0)                  -- sequential
```

This achieves ~2.5x speedup over sequential training. Each fold gets a temporary copy of the config to avoid race conditions.

### Bidirectional Loop

The dual-GPU script implements Phase 5 co-evolution:

```bash
# Initial NC rounds
gen_nc_pseudo b0 $B0_LATEST pvt $PVT_LATEST -> PVT R9 pseudo
full_round pvt R9 ...
gen_nc_pseudo b0 $B0_LATEST pvt $PVT_R9 -> PVT R10 pseudo
full_round pvt R10 ...
gen_nc_pseudo pvt $PVT_R10 b0 $B0_LATEST -> B0 R12 pseudo  # first backflow!
full_round b0 R12 ...

# Bidirectional loop (5 generations)
for GEN in 1 2 3 4 5; do
    # PVT next round (using latest B0)
    gen_nc_pseudo + full_round pvt ...
    # B0 next round (using latest PVT) -- bidirectional!
    gen_nc_pseudo + full_round b0 ...
done
```

### NC Directory Convention

NC outputs use `_nc` suffix to avoid mixing with NS checkpoints:
- NS: `outputs/sed-ns-b0-20s-r12/`
- NC: `outputs/sed-ns-b0-20s-r12-nc/`

## Monitoring

### `scripts/monitor_nc.sh`

Real-time pipeline status dashboard:

```bash
bash scripts/monitor_nc.sh
```

Shows:
- Pipeline process status (running/not running PID)
- Watchdog status
- GPU utilization
- Per-round completion status (folds done, npz, corrected)
- Best AUCs per fold from training logs
- NC pseudo label file listing

Checks both NS and NC directories, displaying `[NS]` or `[NC]` tags.

### Log Locations

| Log | Path |
|-----|------|
| B0 NS pipeline | `outputs/logs/auto_sed_ns_20s_full.log` |
| PVT NS pipeline | `outputs/logs/auto_sed_ns_pvt_20s_r{range}.log` |
| NC dual-GPU pipeline | `outputs/logs/auto_nc_dual_gpu.log` |
| Per-fold training | `outputs/logs/sed_ns_{arch}_r{R}_fold{F}.log` |
| NC per-fold training | `outputs/logs/sed_ns_{arch}_r{R}_nc_fold{F}.log` |
| Inference | `outputs/logs/sed_ns_{arch}_r{R}_infer.log` |
| Residual Corrector | `outputs/logs/sed_corrector_{arch}_r{R}.log` |
| Pseudo generation | `outputs/logs/gen_pseudo_sed_20s_r{R}.log` |
| NC pseudo generation | `outputs/logs/gen_nc_pseudo_*.log` |

## Launching

### NS Pipeline

```bash
# B0 chain
nohup bash scripts/auto_sed_ns_20s_full.sh > outputs/logs/auto_sed_ns_20s_full.log 2>&1 &

# PVT chain (independent)
nohup bash scripts/auto_sed_ns_pvt_20s_r1r4.sh > outputs/logs/auto_sed_ns_pvt_20s_r1r4.log 2>&1 &
```

### NC Pipeline

```bash
# Dual-GPU NC
nohup bash scripts/auto_nc_dual_gpu.sh > outputs/logs/auto_nc_dual_gpu.log 2>&1 &

# Monitor
bash scripts/monitor_nc.sh
```

## Error Handling

- All scripts use `set -euo pipefail` for strict error handling
- Each round verifies all 5 fold checkpoints exist before proceeding
- Temporary configs are created per-fold to prevent race conditions in parallel execution
- Corrected npz files are not generated if the teacher CSV is missing (graceful skip)

## Related Pages

- [[noisy-student]] -- NS training pipeline details
- [[noisy-classmate]] -- NC framework and phases
- [[residual-corrector]] -- Corrector step in the pipeline
