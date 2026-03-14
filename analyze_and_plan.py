"""
Deep post-experiment analysis and automatic Phase 4 experiment planning.

For the best-performing Perch checkpoint this script:
  1. Computes per-class ROC-AUC on the training soundscapes
  2. Breaks down performance by taxonomy (Aves / Amphibia / Reptilia / Insecta)
  3. Correlates performance with training frequency (rare vs common species)
  4. Identifies the worst-performing species
  5. Diagnoses the likely bottleneck (rare species / domain gap / capacity / etc.)
  6. Writes new experiment configs for Phase 4
  7. Returns a machine-readable plan: outputs/phase4_plan.json

Usage:
    python analyze_and_plan.py                          # uses best checkpoint automatically
    python analyze_and_plan.py --run best-derived-v1    # specific run
    python analyze_and_plan.py --gpu 0
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from src.utils.config import load_config
from src.utils.audio import load_audio
from src.data.dataset import build_species_mapping


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--run", default=None,
                   help="Run name to analyse (default: best by kaggle_roc_auc or best_val_roc_auc)")
    p.add_argument("--outputs_dir", default="outputs")
    p.add_argument("--checkpoints_dir", default="checkpoints")
    p.add_argument("--soundscapes_dir", default=None)
    p.add_argument("--labels_csv", default=None)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--gpu", default=None)
    p.add_argument("--plan_key", default="phase4_plan.json",
                   help="Filename for the plan JSON (e.g. phase4_plan.json, phase5_plan.json)")
    p.add_argument("--phase", type=int, default=4,
                   help="Phase number for experiment naming (4 or 5)")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pick_best_run(outputs_dir: str) -> str:
    best_name, best_score = None, -1.0
    for path in glob.glob(os.path.join(outputs_dir, "*/result.json")):
        try:
            with open(path) as f:
                d = json.load(f)
            s = d.get("kaggle_roc_auc") or d.get("best_val_roc_auc") or 0
            if s > best_score:
                best_score, best_name = s, d.get("run_name", os.path.basename(os.path.dirname(path)))
        except Exception:
            pass
    return best_name


def _end_to_seconds(end_str: str) -> int:
    h, m, s = str(end_str).strip().split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def build_ground_truth(labels_csv: str, target_species: list) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    rows = []
    for _, row in df.iterrows():
        fname = re.sub(r"\.ogg$", "", row["filename"], flags=re.IGNORECASE)
        end_sec = _end_to_seconds(row["end"])
        row_id = f"{fname}_{end_sec}"
        label_vec = np.zeros(len(target_species), dtype=np.float32)
        for code in str(row["primary_label"]).split(";"):
            code = code.strip()
            if code in target_species:
                label_vec[target_species.index(code)] = 1.0
        rows.append([row_id] + label_vec.tolist())
    return pd.DataFrame(rows, columns=["row_id"] + target_species)


def run_inference(model, ogg_files, sample_rate, clip_duration, batch_size):
    all_row_ids, all_preds = [], []
    for filepath in ogg_files:
        audio = load_audio(filepath, sample_rate)
        if audio is None:
            continue
        clip_length = clip_duration * sample_rate
        n_segments = len(audio) // clip_length
        if n_segments == 0:
            continue
        clips = np.stack([audio[i*clip_length:(i+1)*clip_length] for i in range(n_segments)])
        ss_id = re.sub(r"\.ogg$", "", os.path.basename(filepath), flags=re.IGNORECASE)
        row_ids = [f"{ss_id}_{(i+1)*clip_duration}" for i in range(n_segments)]
        preds = []
        for start in range(0, len(clips), batch_size):
            batch = tf.constant(clips[start:start+batch_size], dtype=tf.float32)
            logits = model(batch, training=False)
            preds.append(tf.sigmoid(logits).numpy())
        all_row_ids.extend(row_ids)
        all_preds.append(np.concatenate(preds))
    if not all_preds:
        return None, None
    return np.concatenate(all_preds, axis=0), all_row_ids


# ── Per-class analysis ────────────────────────────────────────────────────────

def compute_per_class_roc(y_true: np.ndarray, y_pred: np.ndarray,
                           species: list) -> pd.DataFrame:
    rows = []
    for i, sp in enumerate(species):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        n_pos = int(yt.sum())
        if n_pos == 0:
            auc = None
        else:
            try:
                auc = float(roc_auc_score(yt, yp))
            except Exception:
                auc = None
        rows.append({"species": sp, "n_positives": n_pos, "roc_auc": auc})
    return pd.DataFrame(rows)


def enrich_with_taxonomy(df: pd.DataFrame, taxonomy_csv: str,
                          train_csv: str) -> pd.DataFrame:
    tax = pd.read_csv(taxonomy_csv)[["primary_label", "class_name", "common_name"]]
    tax = tax.rename(columns={"primary_label": "species"})
    tax["species"] = tax["species"].astype(str)
    df["species"] = df["species"].astype(str)
    df = df.merge(tax, on="species", how="left")

    # training frequency
    tr = pd.read_csv(train_csv)
    freq = tr["primary_label"].astype(str).value_counts().rename("train_count")
    df = df.merge(freq.reset_index().rename(columns={"index": "species"}),
                  on="species", how="left")
    df["train_count"] = df["train_count"].fillna(0).astype(int)
    return df


# ── Diagnosis ─────────────────────────────────────────────────────────────────

RARE_THRESHOLD   = 10   # ≤ N recordings in train → "rare"
MEDIUM_THRESHOLD = 50
AUC_BAD          = 0.60  # below this is "problematic"
AUC_WARN         = 0.75  # below this is "needs attention"


def diagnose(df: pd.DataFrame, all_results: dict) -> dict:
    scored = df[df["roc_auc"].notna()].copy()
    macro_auc  = scored["roc_auc"].mean()

    # Taxonomy breakdown
    tax_auc = (
        scored.groupby("class_name")["roc_auc"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "mean_roc_auc", "count": "n_species"})
        .sort_values("mean_roc_auc")
        .to_dict("index")
    )

    # Frequency buckets
    rare   = scored[scored["train_count"] <= RARE_THRESHOLD]
    medium = scored[(scored["train_count"] > RARE_THRESHOLD) &
                    (scored["train_count"] <= MEDIUM_THRESHOLD)]
    common = scored[scored["train_count"] > MEDIUM_THRESHOLD]

    rare_auc   = rare["roc_auc"].mean()   if len(rare)   > 0 else None
    medium_auc = medium["roc_auc"].mean() if len(medium) > 0 else None
    common_auc = common["roc_auc"].mean() if len(common) > 0 else None

    # Worst 20
    worst = (
        scored[scored["n_positives"] > 0]
        .sort_values("roc_auc")
        .head(20)[["species", "common_name", "class_name", "train_count", "roc_auc"]]
        .to_dict("records")
    )

    # Determine dominant bottleneck
    rare_gap   = (common_auc or 0) - (rare_auc or 0)
    tax_gap    = max((v["mean_roc_auc"] for v in tax_auc.values()), default=0) - \
                 min((v["mean_roc_auc"] for v in tax_auc.values()), default=0)

    soundscape_score = (
        all_results.get("soundscape-in-train", {}).get("best_val_roc_auc") or
        all_results.get("soundscape-in-train", {}).get("kaggle_roc_auc") or 0
    )
    baseline_score = (
        all_results.get("baseline", {}).get("best_val_roc_auc") or
        all_results.get("baseline", {}).get("kaggle_roc_auc") or 0
    )
    soundscape_gain = soundscape_score - baseline_score

    bottleneck = "unknown"
    if rare_gap > 0.10:
        bottleneck = "rare_species"
    elif tax_gap > 0.15:
        bottleneck = "taxonomy_imbalance"
    elif soundscape_gain > 0.02:
        bottleneck = "domain_gap"
    elif macro_auc < 0.80:
        bottleneck = "model_capacity"
    else:
        bottleneck = "fine_tuning"

    return {
        "macro_roc_auc":   round(macro_auc, 4),
        "taxonomy":        tax_auc,
        "frequency": {
            "rare":   {"n": len(rare),   "mean_auc": round(rare_auc, 4)   if rare_auc   else None},
            "medium": {"n": len(medium), "mean_auc": round(medium_auc, 4) if medium_auc else None},
            "common": {"n": len(common), "mean_auc": round(common_auc, 4) if common_auc else None},
        },
        "rare_vs_common_gap": round(rare_gap, 4),
        "taxonomy_gap":       round(tax_gap, 4),
        "soundscape_gain":    round(soundscape_gain, 4),
        "bottleneck":         bottleneck,
        "worst_20_species":   worst,
    }


# ── Phase 4 experiment planner ────────────────────────────────────────────────

def plan_phase4(diagnosis: dict, best_config_path: str, all_results: dict = None,
                phase: int = 4) -> list:
    """
    Returns a list of experiment dicts:
      { name, config, rationale, type }
    based on the diagnosed bottleneck.  Always returns an even number ≥ 8 so
    that both GPUs stay busy throughout Phase 4.
    """
    bottleneck  = diagnosis["bottleneck"]
    macro_auc   = diagnosis["macro_roc_auc"]
    rare_gap    = diagnosis["rare_vs_common_gap"]
    tax_gap     = diagnosis["taxonomy_gap"]
    sc_gain     = diagnosis["soundscape_gain"]
    all_results = all_results or {}
    experiments = []

    # Helper: skip experiments that are already finished OR currently running
    def already_good(name: str, threshold: float = 0.0) -> bool:
        r = all_results.get(name, {})
        s = r.get("kaggle_roc_auc") or r.get("best_val_roc_auc") or 0
        if s > threshold:
            return True
        # Also skip if result.json exists at all (experiment is in progress)
        if os.path.isfile(os.path.join("outputs", name, "result.json")):
            return True
        return False

    # ── Always: pseudo-labeling round 1 ──────────────────────────────────────
    experiments.append({
        "name": "pseudo-r1",
        "config": "configs/pseudo_label_round1.yaml",
        "rationale": "Pseudo-labeling (BirdCLEF 2025 1st place) is the highest-gain "
                     "technique. PowerTransform(power=2) sharpens confident predictions "
                     "before retraining.",
        "type": "pseudo_label",
    })

    # ── Always: soundscape-heavy (soundscapes showed massive gain) ───────────
    if not already_good("soundscape-heavy"):
        experiments.append({
            "name": "soundscape-heavy",
            "config": "configs/exp_soundscape_heavy.yaml",
            "rationale": f"Soundscape gain was {sc_gain:.3f}. This variant uses 3× "
                         "soundscape oversampling to more aggressively reduce the "
                         "train/test domain gap.",
            "type": "ablation",
        })

    # ── Always: taxon-balanced ────────────────────────────────────────────────
    if not already_good("taxon-balanced"):
        experiments.append({
            "name": "taxon-balanced",
            "config": "configs/exp_taxon_balanced.yaml",
            "rationale": "Non-bird taxa severely underrepresented (Reptilia: 1 recording, "
                         "Insecta: 199, Amphibia: 451 vs Aves: 34799). 3× non-bird boost "
                         "ensures adequate gradient updates.",
            "type": "ablation",
        })

    # ── Always: taxon-multitask ───────────────────────────────────────────────
    if not already_good("taxon-multitask"):
        experiments.append({
            "name": "taxon-multitask",
            "config": "configs/exp_taxon_multitask.yaml",
            "rationale": "Multi-task auxiliary taxonomy loss (weight=0.1) forces the model "
                         "to learn taxon-discriminative features, improving non-bird performance.",
            "type": "ablation",
        })

    # ── Rare-species bottleneck ───────────────────────────────────────────────
    if bottleneck == "rare_species" or rare_gap > 0.08:
        if not already_good("linear-classweights"):
            experiments.append({
                "name": "linear-classweights",
                "config": "configs/exp_linear_classweights.yaml",
                "rationale": f"Rare species ROC-AUC is {rare_gap:.3f} below common. "
                             "Linear inverse-frequency weights (stronger than sqrt) increase "
                             "rare-class sampling pressure.",
                "type": "ablation",
            })
        if not already_good("more-clips"):
            experiments.append({
                "name": "more-clips",
                "config": "configs/exp_more_clips.yaml",
                "rationale": "n_clips_per_file=5 (from 3) gives rare species more gradient "
                             "updates per epoch without adding new data.",
                "type": "ablation",
            })

    # ── Taxonomy imbalance ────────────────────────────────────────────────────
    if bottleneck == "taxonomy_imbalance" or tax_gap > 0.12:
        if not already_good("focal-stronger"):
            experiments.append({
                "name": "focal-stronger",
                "config": "configs/exp_focal_gamma3.yaml",
                "rationale": f"Taxonomy ROC-AUC gap is {tax_gap:.3f}. FocalLoss(γ=3) "
                             "stronger down-weighting of easy negatives helps under-represented taxa.",
                "type": "ablation",
            })

    # ── Model capacity ────────────────────────────────────────────────────────
    if bottleneck == "model_capacity" or macro_auc < 0.85:
        if not already_good("larger-head"):
            experiments.append({
                "name": "larger-head",
                "config": "configs/exp_larger_head.yaml",
                "rationale": f"Macro ROC-AUC {macro_auc:.4f}. Larger MLP head "
                             "(Dense(1024)→ReLU→Dense(512)→ReLU→Dense(num_classes)) "
                             "adds capacity for the multi-class problem.",
                "type": "ablation",
            })

    # ── Human voice removal (BirdCLEF top solutions technique) ───────────────
    nohuman_cache = os.path.join("outputs", "embeddings_cache_nohuman", "manifest.csv")
    if os.path.isfile(nohuman_cache) and not already_good("no-human-voice"):
        experiments.append({
            "name": "no-human-voice",
            "config": "configs/exp_no_human_voice.yaml",
            "rationale": "Silero VAD removes human speech frames before Perch embedding. "
                         "BirdCLEF 2024/2025 top solutions used this to reduce field-observer "
                         "voice contamination in recordings.",
            "type": "ablation",
        })

    # ── Always: longer training ───────────────────────────────────────────────
    if not already_good("longer-training"):
        experiments.append({
            "name": "longer-training",
            "config": "configs/exp_longer_training.yaml",
            "rationale": "100 epochs (vs 50) to check whether model has saturated or still gains.",
            "type": "ablation",
        })

    # ── Soundscape + min_rating combined (if not already run) ────────────────
    if not already_good("birdclef25-soundscape"):
        experiments.append({
            "name": "birdclef25-soundscape",
            "config": "configs/exp_birdclef25_soundscape.yaml",
            "rationale": "Combines min_rating=3.0 (cleaner data) and soundscapes "
                         "(domain adaptation). Both techniques individually showed gains.",
            "type": "ablation",
        })

    # ── Soundscape + taxon multitask (combined — best of both) ───────────────
    if not already_good("soundscape-taxon-multitask"):
        experiments.append({
            "name": "soundscape-taxon-multitask",
            "config": "configs/exp_soundscape_taxon_multitask.yaml",
            "rationale": "Soundscapes (domain adaptation) + taxon auxiliary loss + "
                         "non-bird 3× upweight. Combines the two strongest signals.",
            "type": "ablation",
        })

    # ── Soundscape + AdamW + sqrt class weights ───────────────────────────────
    if not already_good("soundscape-adamw"):
        experiments.append({
            "name": "soundscape-adamw",
            "config": "configs/exp_soundscape_adamw.yaml",
            "rationale": "Soundscapes + AdamW optimizer + sqrt inverse-frequency class "
                         "weights. AdamW showed +0.007 gain in isolation; testing with soundscapes.",
            "type": "ablation",
        })

    # ── Soundscape + focal loss ───────────────────────────────────────────────
    if not already_good("soundscape-focal"):
        experiments.append({
            "name": "soundscape-focal",
            "config": "configs/exp_soundscape_focal.yaml",
            "rationale": "Soundscape training + FocalLoss(γ=2) to address class imbalance "
                         "in soundscape segments which have fewer positive labels per row.",
            "type": "ablation",
        })

    # ── Soundscape + lower LR for stability ──────────────────────────────────
    if not already_good("soundscape-lowlr") and phase >= 5:
        experiments.append({
            "name": "soundscape-lowlr",
            "config": "configs/exp_soundscape_lowlr.yaml",
            "rationale": "Soundscape training with LR=1e-4 (conservative). "
                         "Soundscape data may need gentler optimization to avoid overfitting "
                         "to soundscape-specific artefacts.",
            "type": "ablation",
        })

    # ── Mixup stronger ───────────────────────────────────────────────────────
    if not already_good("mixup-heavy") and phase >= 5:
        experiments.append({
            "name": "mixup-heavy",
            "config": "configs/exp_mixup_heavy.yaml",
            "rationale": "mixup_alpha=0.6 (from 0.3) creates harder mixed examples, "
                         "which can improve generalisation when training data is limited.",
            "type": "ablation",
        })

    # Ensure even count (pad with a duplicate-prevention check)
    if len(experiments) % 2 == 1:
        # Add pseudo-r2 as the pairing experiment if we need an even number
        if phase >= 5 and not already_good("pseudo-r2"):
            experiments.append({
                "name": "pseudo-r2",
                "config": "configs/pseudo_label_round2.yaml",
                "rationale": "Second round of pseudo-labeling on top of pseudo-r1 checkpoint "
                             "to further leverage unlabelled soundscape data.",
                "type": "pseudo_label",
            })

    return experiments


# ── Config writers ────────────────────────────────────────────────────────────

def _load_best_config_dict(best_config_path: str) -> dict:
    import yaml
    with open(best_config_path) as f:
        return yaml.safe_load(f)


def write_phase4_configs(experiments: list, base_config_path: str):
    import yaml
    with open(base_config_path) as f:
        base = yaml.safe_load(f)

    os.makedirs("configs", exist_ok=True)

    configs_written = []
    for exp in experiments:
        if exp["type"] == "pseudo_label":
            # pseudo-label config already exists
            if os.path.isfile(exp["config"]):
                configs_written.append(exp["config"])
            continue

        cfg = {k: (v.copy() if isinstance(v, dict) else v) for k, v in base.items()}
        name = exp["name"]
        cfg["experiment"] = {"name": name, "seed": 42}

        if name == "linear-classweights":
            cfg["training"]["class_weight_mode"] = "linear"

        elif name == "more-clips":
            cfg["audio"]["n_clips_per_file"] = 5

        elif name == "focal-stronger":
            cfg["training"]["loss"] = "focal"
            cfg["training"]["focal_gamma"] = 3.0
            cfg["training"]["focal_alpha"] = 0.25

        elif name == "soundscape-heavy":
            cfg["training"]["use_soundscapes_in_train"] = True
            cfg["training"]["soundscape_oversample"] = 3

        elif name == "larger-head":
            cfg["model"]["hidden_dim"] = 1024
            cfg["model"]["hidden_layers"] = 2

        elif name == "longer-training":
            cfg["training"]["epochs"] = 100
            cfg["training"]["warmup_epochs"] = 5

        elif name == "taxon-balanced":
            cfg["training"]["class_weight_mode"] = "taxon_upweight"
            cfg["training"]["taxon_nonbird_boost"] = 3.0

        elif name == "taxon-multitask":
            cfg["training"]["taxon_aux_weight"] = 0.1
            cfg["training"]["class_weight_mode"] = "taxon_upweight"
            cfg["training"]["taxon_nonbird_boost"] = 3.0

        elif name == "birdclef25-soundscape":
            cfg["data"]["min_rating"] = 3.0
            cfg["training"]["use_soundscapes_in_train"] = True

        elif name == "soundscape-focal":
            cfg["training"]["use_soundscapes_in_train"] = True
            cfg["training"]["loss"] = "focal"
            cfg["training"]["focal_gamma"] = 2.0
            cfg["training"]["focal_alpha"] = 0.25

        elif name == "soundscape-lowlr":
            cfg["training"]["use_soundscapes_in_train"] = True
            cfg["training"]["learning_rate"] = 1.0e-4
            cfg["training"]["warmup_epochs"] = 5

        elif name == "mixup-heavy":
            cfg["training"]["mixup_alpha"] = 0.6

        elif name == "pseudo-r2":
            # pseudo_label_round2 config: reuse round1 as base, update checkpoint ref
            cfg["experiment"] = {"name": "pseudo-r2", "seed": 42}
            cfg["training"]["pseudo_label_csv"] = "pseudo_labels/round1_pseudo.csv"
            cfg["training"]["pseudo_label_power"] = 2.0

        path = exp["config"]
        with open(path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        configs_written.append(path)
        print(f"  Config written → {path}")

    return configs_written


# ── Report section ────────────────────────────────────────────────────────────

def print_analysis(diagnosis: dict, experiments: list):
    print("\n" + "═" * 65)
    print("  DEEP ANALYSIS RESULTS")
    print("═" * 65)
    print(f"\n  Macro ROC-AUC (scored species): {diagnosis['macro_roc_auc']:.4f}")
    print(f"  Diagnosed bottleneck:           {diagnosis['bottleneck'].upper()}")

    print("\n  ── Taxonomy breakdown ──────────────────────────────────────")
    for cls, v in sorted(diagnosis["taxonomy"].items(), key=lambda x: x[1]["mean_roc_auc"]):
        bar = "█" * int(v["mean_roc_auc"] * 20)
        print(f"  {cls:<12}  {v['mean_roc_auc']:.4f}  {bar}  (n={v['n_species']})")

    print("\n  ── Frequency bucket breakdown ──────────────────────────────")
    fb = diagnosis["frequency"]
    for bucket, vals in [("rare (≤10)", fb["rare"]),
                          ("medium (11–50)", fb["medium"]),
                          ("common (>50)", fb["common"])]:
        auc_str = f"{vals['mean_auc']:.4f}" if vals["mean_auc"] else "  n/a"
        print(f"  {bucket:<17}  ROC-AUC={auc_str}  n={vals['n']}")

    print(f"\n  Rare vs Common gap:  {diagnosis['rare_vs_common_gap']:+.4f}")
    print(f"  Taxonomy gap:        {diagnosis['taxonomy_gap']:+.4f}")
    print(f"  Soundscape gain:     {diagnosis['soundscape_gain']:+.4f}")

    print("\n  ── Worst 10 species ────────────────────────────────────────")
    for r in diagnosis["worst_20_species"][:10]:
        print(f"  {r['species']:<15} {r.get('common_name','')[:22]:<22} "
              f"{r['class_name']:<10} train={r['train_count']:3d}  "
              f"AUC={r['roc_auc']:.3f}")

    print("\n  ── Phase 4 experiments planned ─────────────────────────────")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp['name']}")
        print(f"     {exp['rationale'][:80]}")
    print("═" * 65 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = load_config(args.config)
    soundscapes_dir = args.soundscapes_dir or config.data.train_soundscapes_dir
    labels_csv      = args.labels_csv      or config.data.soundscapes_labels_csv

    target_species, _ = build_species_mapping(config.data.sample_submission_csv)
    num_classes = len(target_species)

    # Load all existing result.json for diagnosis context
    all_results = {}
    for path in glob.glob(os.path.join(args.outputs_dir, "*/result.json")):
        try:
            with open(path) as f:
                d = json.load(f)
            all_results[d.get("run_name", "")] = d
        except Exception:
            pass

    # Pick best run
    run_name = args.run or _pick_best_run(args.outputs_dir)
    if not run_name:
        print("ERROR: No finished runs found in outputs/.")
        return
    print(f"\nAnalysing run: {run_name}")

    ckpt_path = os.path.join(args.checkpoints_dir, run_name, "best_head")
    if not os.path.isfile(ckpt_path + ".weights.h5") and not os.path.isfile(ckpt_path):
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        return

    # Load model
    from src.model.classifier import PerchClassifier
    model = PerchClassifier(
        perch_dir=config.model.perch_dir,
        num_classes=num_classes,
        mode=config.model.mode,
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
    )
    model.load_head(ckpt_path)

    # Inference on soundscapes
    ogg_files = sorted(glob.glob(os.path.join(soundscapes_dir, "*.ogg")))
    print(f"Running inference on {len(ogg_files)} soundscape files …")
    y_pred, row_ids = run_inference(
        model, ogg_files,
        config.audio.sample_rate,
        config.audio.clip_duration,
        args.batch_size,
    )
    if y_pred is None:
        print("ERROR: No predictions generated.")
        return

    # Ground truth
    solution = build_ground_truth(labels_csv, target_species)
    sol_idx = solution.set_index("row_id")
    common_rows = [r for r in row_ids if r in sol_idx.index]
    if not common_rows:
        print("ERROR: No common row_ids between predictions and ground truth.")
        return

    row_mask = [i for i, r in enumerate(row_ids) if r in sol_idx.index]
    y_pred_aligned = y_pred[row_mask]
    y_true_aligned = sol_idx.loc[common_rows][target_species].values

    # Per-class analysis
    print("Computing per-class ROC-AUC …")
    per_class_df = compute_per_class_roc(y_true_aligned, y_pred_aligned, target_species)
    per_class_df = enrich_with_taxonomy(
        per_class_df,
        config.data.taxonomy_csv,
        config.data.train_csv,
    )

    # Diagnose
    diagnosis = diagnose(per_class_df, all_results)

    # Plan Phase 4
    best_config_path = (
        "configs/best_derived_v1.yaml"
        if os.path.isfile("configs/best_derived_v1.yaml")
        else "configs/birdclef25_improvements.yaml"
    )
    phase = getattr(args, "phase", 4)
    experiments = plan_phase4(diagnosis, best_config_path,
                              all_results=all_results, phase=phase)

    # Print
    print_analysis(diagnosis, experiments)

    # Write configs
    print("Writing experiment configs …")
    write_phase4_configs(experiments, best_config_path)

    # Save plan
    plan_key = getattr(args, "plan_key", "phase4_plan.json")
    plan = {
        "analysed_run": run_name,
        "diagnosis": diagnosis,
        "phase4_experiments": experiments,
    }
    os.makedirs(args.outputs_dir, exist_ok=True)
    plan_path = os.path.join(args.outputs_dir, plan_key)
    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2, default=str)
    print(f"Plan saved → {plan_path}")

    # Save per-class CSV
    csv_path = os.path.join(args.outputs_dir, run_name, "per_class_roc_auc.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    per_class_df.to_csv(csv_path, index=False)
    print(f"Per-class AUC → {csv_path}")

    return plan


if __name__ == "__main__":
    main()
