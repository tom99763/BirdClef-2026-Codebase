#!/usr/bin/env bash
# =============================================================================
# Baseline run — default.yaml, no modifications.
# This establishes the reference cMAP score every other experiment is compared to.
# =============================================================================
set -e
cd "$(dirname "$0")/.."   # always run from project root

python train.py \
    --config configs/default.yaml \
    experiment.name="baseline"
