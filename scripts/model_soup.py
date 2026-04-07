"""Model Soup: average weights of top-k SED checkpoints.

BirdCLEF 2025 1st place technique — checkpoint weight averaging improves
generalization by ~0.002-0.005 AUC with zero extra training cost.

Finds the top-k epoch checkpoints saved by train_sed.py (soup_ep*.pt),
averages their weights, and saves as checkpoints/<run>/soup_sed.pt.

Usage:
    python scripts/model_soup.py --run sed-b0-v6 --config configs/sed_b0_v6.yaml
    python scripts/model_soup.py --run sed-b2-v1 --config configs/sed_b2_v1.yaml --topk 5
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import load_config
from src.data.dataset import build_species_mapping
from src.model.sed_model import SEDModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run",       required=True, help="Experiment name (e.g. sed-b0-v6)")
    p.add_argument("--config",    required=True, help="Config YAML for this run")
    p.add_argument("--topk",      type=int, default=None,
                   help="Use top-k checkpoints (default: all soup_ep*.pt found)")
    p.add_argument("--checkpoints_dir", default="checkpoints")
    p.add_argument("--outputs_dir",     default="outputs")
    p.add_argument("--gpu",       default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(args.config)
    _, species_to_idx = build_species_mapping(config.data.sample_submission_csv)
    num_classes = len(species_to_idx)

    ckpt_dir = os.path.join(args.checkpoints_dir, args.run)

    # ── Find soup checkpoints ─────────────────────────────────────────────────
    soup_paths = sorted(glob.glob(os.path.join(ckpt_dir, "soup_ep*.pt")))
    if not soup_paths:
        print(f"No soup_ep*.pt found in {ckpt_dir}")
        # Fall back to best checkpoint only
        best_path = os.path.join(ckpt_dir, "best_sed.pt")
        if os.path.isfile(best_path):
            print(f"  Only best_sed.pt available — copying as soup (no averaging)")
            import shutil
            shutil.copy(best_path, os.path.join(ckpt_dir, "soup_sed.pt"))
        return

    # Sort by val AUC stored in checkpoint metadata
    scored = []
    for path in soup_paths:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            auc  = ckpt.get("metrics", {}).get("macro_auc", 0.0)
            ep   = ckpt.get("epoch", 0)
            scored.append((auc, ep, path))
        except Exception as e:
            print(f"  WARN: cannot load {path}: {e}")

    scored.sort(key=lambda x: -x[0])   # best AUC first

    topk = args.topk or len(scored)
    selected = scored[:topk]

    print(f"\nModel Soup — {args.run}")
    print(f"  Found {len(soup_paths)} soup checkpoints, using top-{topk}:")
    for auc, ep, path in selected:
        print(f"    ep{ep:3d}  val_auc={auc:.4f}  {os.path.basename(path)}")

    # ── Load and average weights ──────────────────────────────────────────────
    state_dicts = []
    for _, _, path in selected:
        ckpt  = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        if any("gem_pool" in k for k in state):
            state = {k.replace("gem_pool", "freq_pool"): v for k, v in state.items()}
        state_dicts.append(state)

    avg_state = {}
    for key in state_dicts[0]:
        tensors = [sd[key].float() for sd in state_dicts]
        avg_state[key] = torch.stack(tensors).mean(dim=0)

    # ── Build model and load soup weights ─────────────────────────────────────
    model = SEDModel(
        backbone   = config.model.backbone,
        num_classes = num_classes,
        in_chans   = config.model.get("in_chans", 3),
        pretrained = False,
        drop_rate  = config.model.get("dropout", 0.1),
        use_gem    = config.model.get("use_gem", True),
        gem_p_init = config.model.get("gem_p_init", 3.0),
        n_mels     = config.mel.n_mels,
    ).to(device)
    model.load_state_dict(avg_state)

    # ── Save soup checkpoint ──────────────────────────────────────────────────
    soup_out = os.path.join(ckpt_dir, "soup_sed.pt")
    best_auc  = selected[0][0]
    best_ep   = selected[0][1]
    torch.save({
        "model_state_dict": avg_state,
        "epoch":   f"soup(top{topk})",
        "metrics": {"macro_auc": best_auc},
        "soup_sources": [{"epoch": ep, "auc": auc, "path": path}
                         for auc, ep, path in selected],
    }, soup_out)
    print(f"\n  Saved soup → {soup_out}")
    print(f"  Best individual AUC: {best_auc:.4f} @ep{best_ep}")

    # ── Record in result.json ─────────────────────────────────────────────────
    result_path = os.path.join(args.outputs_dir, args.run, "result.json")
    if os.path.isfile(result_path):
        with open(result_path) as f:
            result = json.load(f)
        result["soup"] = {
            "topk": topk,
            "best_individual_auc": round(best_auc, 6),
            "sources": [{"epoch": ep, "auc": round(auc, 6)} for auc, ep, _ in selected],
        }
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Updated result.json with soup metadata")


if __name__ == "__main__":
    main()
