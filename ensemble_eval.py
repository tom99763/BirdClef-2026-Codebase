"""
Ensemble Evaluation — BirdCLEF 2026
====================================
Blend cached submission_train_soundscapes.csv predictions from multiple models,
apply PP, evaluate locally, output ranked results.

Usage:
    python ensemble_eval.py
    python ensemble_eval.py --models nohuman-label-pseudo nohuman-label-head-r3
    python ensemble_eval.py --out outputs/ensemble_results.json
"""

import argparse, json, os, re
import numpy as np
import pandas as pd

from src.data.dataset import build_species_mapping
from src.metrics.kaggle_metric import score as kaggle_score
from postproc_eval_v2 import (
    build_ground_truth,
    build_sc_groups,
    pp_sliding_max,
    pp_quantile_mix,
    pp_threshold_zero,
    pp_sigmoid_temperature,
    pp_blend_vlom,
    score_preds,
)


DEFAULT_MODELS = [
    "nohuman-label-pseudo",
    "nohuman-label-head",
    "perch-label-head",
    "label-head-pseudo",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--outputs_dir", default="outputs")
    p.add_argument("--out", default="outputs/ensemble_results.json")
    return p.parse_args()


def load_cached(run_name, outputs_dir, target_species):
    """Load submission_train_soundscapes.csv → (row_ids, preds array)."""
    path = os.path.join(outputs_dir, run_name, "submission_train_soundscapes.csv")
    if not os.path.isfile(path):
        print(f"  MISSING: {path}")
        return None, None
    df = pd.read_csv(path)
    row_ids = df["row_id"].tolist()
    cols = [c for c in df.columns if c != "row_id"]
    # Reorder to target_species order
    preds = df[target_species].values.astype(np.float32)
    return row_ids, preds


def main():
    args = parse_args()

    from src.utils.config import load_config
    config = load_config(args.config)
    target_species, _ = build_species_mapping(config.data.sample_submission_csv)

    labels_csv = config.data.soundscapes_labels_csv
    solution = build_ground_truth(labels_csv, target_species)
    print(f"Ground truth: {len(solution)} rows, {len(target_species)} species\n")

    # Load all cached predictions
    preds_dict = {}
    row_ids_ref = None
    for model in args.models:
        row_ids, preds = load_cached(model, args.outputs_dir, target_species)
        if preds is not None:
            preds_dict[model] = preds
            if row_ids_ref is None:
                row_ids_ref = row_ids
            print(f"  Loaded {model}: {preds.shape}")

    if len(preds_dict) < 2:
        print("Need at least 2 models.")
        return

    sc_groups = build_sc_groups(row_ids_ref)

    # ── Baseline individual scores ──────────────────────────────────────────
    print("\n=== Individual Model Baselines ===")
    baselines = {}
    for name, preds in preds_dict.items():
        s = score_preds(preds, row_ids_ref, solution, target_species)
        baselines[name] = s
        print(f"  {name}: {s:.4f}")

    # ── Ensemble grid ────────────────────────────────────────────────────────
    # Focus on best 2 models: pseudo + head
    models_list = list(preds_dict.keys())
    pseudo_key = next((k for k in models_list if "pseudo" in k and "r2" not in k and "r3" not in k), models_list[0])
    head_key   = next((k for k in models_list if "head" in k and "pseudo" not in k and "r2" not in k and "r3" not in k), models_list[1] if len(models_list) > 1 else models_list[0])

    print(f"\n=== Ensemble Grid: {pseudo_key} × {head_key} ===")
    results = []

    # Weight ratios to try
    ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for w_pseudo in ratios:
        w_head = 1.0 - w_pseudo
        # Linear blend
        blend = w_pseudo * preds_dict[pseudo_key] + w_head * preds_dict[head_key]

        # PP variants
        pp_configs = [
            ("raw",         blend),
            ("q+slide",     pp_sliding_max(pp_quantile_mix(blend, sc_groups, 0.5), sc_groups, 0.5, 9)),
            ("thr0.02+slide", pp_sliding_max(pp_threshold_zero(blend, 0.02), sc_groups, 0.5, 9)),
            ("thr0.1+slide",  pp_sliding_max(pp_threshold_zero(blend, 0.1), sc_groups, 0.5, 9)),
            ("slide_only",  pp_sliding_max(blend, sc_groups, 0.5, 9)),
        ]

        for pp_name, pp_preds in pp_configs:
            s = score_preds(pp_preds, row_ids_ref, solution, target_species)
            tag = f"pseudo={w_pseudo:.1f} head={w_head:.1f} | {pp_name}"
            results.append({"pseudo_w": w_pseudo, "head_w": w_head, "pp": pp_name, "score": s, "tag": tag})

    # ── VLOM blend ────────────────────────────────────────────────────────────
    print("\n=== VLOM Blend ===")
    for alpha in [0.3, 0.5, 0.7]:
        vlom = pp_blend_vlom(preds_dict[pseudo_key], preds_dict[head_key], alpha=alpha)
        s_raw = score_preds(vlom, row_ids_ref, solution, target_species)
        s_qslide = score_preds(
            pp_sliding_max(pp_quantile_mix(vlom, sc_groups, 0.5), sc_groups, 0.5, 9),
            row_ids_ref, solution, target_species
        )
        results.append({"pseudo_w": -1, "head_w": -1, "pp": f"vlom(α={alpha})+raw", "score": s_raw, "tag": f"VLOM α={alpha} raw"})
        results.append({"pseudo_w": -1, "head_w": -1, "pp": f"vlom(α={alpha})+q+slide", "score": s_qslide, "tag": f"VLOM α={alpha} q+slide"})

    # ── 3-model ensembles ─────────────────────────────────────────────────────
    if len(preds_dict) >= 3:
        print("\n=== 3-Model Ensemble ===")
        keys3 = [pseudo_key, head_key]
        third_key = next((k for k in models_list if k not in [pseudo_key, head_key]), None)
        if third_key:
            for w3 in [0.1, 0.2]:
                w12 = 1.0 - w3
                best_w_pseudo = 0.6  # from 2-model grid result
                blend3 = (best_w_pseudo * w12 * preds_dict[pseudo_key]
                         + (1 - best_w_pseudo) * w12 * preds_dict[head_key]
                         + w3 * preds_dict[third_key])
                s = score_preds(
                    pp_sliding_max(pp_quantile_mix(blend3, sc_groups, 0.5), sc_groups, 0.5, 9),
                    row_ids_ref, solution, target_species
                )
                results.append({"pseudo_w": -1, "head_w": -1, "pp": f"3model+q+slide", "score": s,
                                 "tag": f"pseudo{best_w_pseudo*w12:.2f}+head{(1-best_w_pseudo)*w12:.2f}+{third_key}{w3:.1f} q+slide"})

    # ── Print ranked results ──────────────────────────────────────────────────
    sorted_results = sorted(results, key=lambda x: -(x["score"] or 0))
    print("\n=== TOP 20 ENSEMBLE RESULTS ===")
    print(f"{'Rank':>4}  {'Score':>7}  {'Delta vs pseudo-base':>20}  Config")
    print("-" * 80)
    base_pseudo = baselines.get(pseudo_key, 0)
    for i, r in enumerate(sorted_results[:20], 1):
        delta = (r["score"] or 0) - base_pseudo
        print(f"  {i:>2}  {r['score']:.4f}  {delta:+.4f}                    {r['tag']}")

    # Save
    output = {
        "baselines": baselines,
        "pseudo_key": pseudo_key,
        "head_key": head_key,
        "top20": sorted_results[:20],
        "all_results": sorted_results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {args.out}")

    # Summary
    best = sorted_results[0]
    print(f"\n🏆 BEST: {best['score']:.4f} (+{best['score']-base_pseudo:+.4f} vs pseudo baseline)")
    print(f"   Config: {best['tag']}")


if __name__ == "__main__":
    main()
