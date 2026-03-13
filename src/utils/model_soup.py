"""Model Soup: checkpoint weight averaging (BirdCLEF 2025 3rd place).

Instead of keeping separate ensemble models, average the weights of the
top-K checkpoints saved during training. This:
  - reduces variance without increasing inference cost
  - is equivalent to an ensemble over the weight trajectory
  - often outperforms the best single checkpoint by 0.3–1% cMAP

Reference: "Model soups: averaging weights of multiple fine-tuned models
improves accuracy without increasing inference time" (Wortsman et al. 2022)

Usage:
    python -m src.utils.model_soup \
        --checkpoints checkpoints/run_a/best_head checkpoints/run_b/best_head \
        --output checkpoints/soup/souped_head \
        --config configs/default.yaml
"""

import argparse
import numpy as np
import tensorflow as tf
from typing import List


def average_checkpoints(
    model,
    checkpoint_paths: List[str],
) -> None:
    """
    Load multiple checkpoints and set model weights to their element-wise mean.

    Args:
        model            : PerchClassifier instance (head must already be built).
        checkpoint_paths : List of paths passed to model.load_head().
    """
    if len(checkpoint_paths) == 1:
        model.load_head(checkpoint_paths[0])
        print("Only one checkpoint — loaded directly (no averaging needed).")
        return

    # Accumulate weights
    accumulated: List[List[np.ndarray]] = []

    for ckpt_path in checkpoint_paths:
        model.load_head(ckpt_path)
        snapshot = [v.numpy().copy() for v in model.head.trainable_variables]
        accumulated.append(snapshot)
        print(f"  Loaded: {ckpt_path}")

    # Element-wise mean across checkpoints
    n = len(accumulated)
    souped = [
        sum(accumulated[k][i] for k in range(n)) / n
        for i in range(len(accumulated[0]))
    ]

    # Assign averaged weights back to model
    for var, val in zip(model.head.trainable_variables, souped):
        var.assign(val)

    print(f"\nModel Soup: averaged {n} checkpoints.")


def main():
    parser = argparse.ArgumentParser(description="Model Soup: average checkpoints")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Paths to head checkpoints to average")
    parser.add_argument("--output", required=True,
                        help="Output path for the souped checkpoint")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Config used to build the model")
    args = parser.parse_args()

    from src.utils.config import load_config
    from src.data.dataset import build_species_mapping
    from src.model.classifier import PerchClassifier

    config = load_config(args.config)
    _, species_to_idx = build_species_mapping(config.data.sample_submission_csv)
    num_classes = len(species_to_idx)

    model = PerchClassifier(
        perch_dir=config.model.perch_dir,
        num_classes=num_classes,
        mode=config.model.mode,
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
    )

    average_checkpoints(model, args.checkpoints)
    model.save_head(args.output)
    print(f"Souped checkpoint saved → {args.output}")


if __name__ == "__main__":
    main()
