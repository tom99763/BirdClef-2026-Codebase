"""
Auto-orchestration pipeline:

  Phase 1  (running): baseline (GPU 0) + birdclef25-base (GPU 1)
  Phase 2a (auto):    focal-isolated (GPU 0) + adamw-classweights (GPU 1)  [parallel]
  Phase 2b (auto):    soundscape-in-train (GPU 0) + birdclef25-soundscape (GPU 1)  [parallel]
  Phase 3  (auto):    derive best config  →  best-derived-v1 (GPU 0)
  Phase 3b (auto):    SED training (GPU 1, parallel with Phase 3)
  Phase 4  (auto):    deep analysis → ablations 2-at-a-time on GPU 0+1
  Phase 4b (auto):    pseudo-label round 1 (GPU 0) + longer-training (GPU 1) [parallel]
  Phase 5  (auto):    second analysis → derive best-v2 → more experiments

Usage:
    python orchestrate.py
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime


LOG_PREFIX = "[orchestrate]"


def log(msg: str):
    print(f"{LOG_PREFIX} {datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)


# ── Result helpers ─────────────────────────────────────────────────────────

def is_finished(run_name: str, outputs_dir: str = "outputs") -> bool:
    path = os.path.join(outputs_dir, run_name, "result.json")
    if not os.path.isfile(path):
        return False
    try:
        with open(path) as f:
            d = json.load(f)
        return bool(d.get("finished", False))
    except Exception:
        return False


def best_cmap(run_name: str, outputs_dir: str = "outputs") -> float:
    path = os.path.join(outputs_dir, run_name, "result.json")
    try:
        with open(path) as f:
            return float(json.load(f).get("best_val_roc_auc", 0.0))
    except Exception:
        return 0.0


def wait_for(run_names: list, poll_secs: int = 30):
    log(f"Waiting for: {run_names}")
    while True:
        done = [r for r in run_names if is_finished(r)]
        pending = [r for r in run_names if not is_finished(r)]
        if not pending:
            log(f"All done: {run_names}")
            return
        log(f"  Done: {done} | Still running: {pending}")
        time.sleep(poll_secs)


def run_bg(cmd: list, gpu: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log(f"[GPU {gpu}] Launching: {' '.join(cmd)}")
    return subprocess.Popen(cmd, env=env)


# ── Phase 2 ───────────────────────────────────────────────────────────────

def run_phase2a():
    """focal-isolated (GPU 0) + adamw-classweights (GPU 1) in parallel."""
    log("═══ Phase 2a: focal-isolated (GPU 0) + adamw-classweights (GPU 1) ═══")
    p0 = run_bg(["python", "train.py",
                 "--config", "configs/exp_focal_isolated.yaml",
                 "experiment.name=focal-isolated"], gpu=0)
    p1 = run_bg(["python", "train.py",
                 "--config", "configs/exp_adamw_classweights.yaml",
                 "experiment.name=adamw-classweights"], gpu=1)
    p0.wait()
    p1.wait()
    log("Phase 2a complete.")


def run_phase2b():
    """soundscape-in-train (GPU 0) + birdclef25-soundscape (GPU 1) in parallel.
    Both must finish before derive_best_config."""
    log("═══ Phase 2b: soundscape-in-train (GPU 0) + birdclef25-soundscape (GPU 1) ═══")
    # Skip if already finished (e.g. launched manually before orchestrator reached here)
    procs = []
    if not is_finished("soundscape-in-train"):
        p0 = run_bg(["python", "train.py",
                     "--config", "configs/exp_soundscape_train.yaml",
                     "experiment.name=soundscape-in-train"], gpu=0)
        procs.append(p0)
    else:
        log("  soundscape-in-train already finished, skipping.")

    if not is_finished("birdclef25-soundscape"):
        p1 = run_bg(["python", "train.py",
                     "--config", "configs/exp_birdclef25_soundscape.yaml",
                     "experiment.name=birdclef25-soundscape"], gpu=1)
        procs.append(p1)
    else:
        log("  birdclef25-soundscape already finished, skipping.")

    for p in procs:
        p.wait()
    log("Phase 2b complete.")


# ── Phase 3: derive best config + run ────────────────────────────────────

def derive_best_config(version: int = 1) -> str:
    """
    Compare all ablation experiments and build a new config combining
    the best-performing techniques.  Returns the path of the new config.
    """
    scores = {
        "baseline":              best_cmap("baseline"),
        "focal-isolated":        best_cmap("focal-isolated"),
        "adamw-classweights":    best_cmap("adamw-classweights"),
        "birdclef25-base":       best_cmap("birdclef25-base"),
        "soundscape-in-train":   best_cmap("soundscape-in-train"),
        "birdclef25-soundscape": best_cmap("birdclef25-soundscape"),
    }
    log(f"Scores: {scores}")

    baseline_score  = scores["baseline"]
    focal_gain      = scores["focal-isolated"]        - baseline_score
    adamw_gain      = scores["adamw-classweights"]    - baseline_score
    full_gain       = scores["birdclef25-base"]       - baseline_score
    sc_gain         = scores["soundscape-in-train"]   - baseline_score
    sc25_gain       = scores["birdclef25-soundscape"] - baseline_score

    log(f"  FocalLoss gain:                +{focal_gain:.4f}")
    log(f"  AdamW+ClassWeights gain:       +{adamw_gain:.4f}")
    log(f"  BirdCLEF25 full gain:          +{full_gain:.4f}")
    log(f"  Soundscape-in-train gain:      +{sc_gain:.4f}")
    log(f"  BirdCLEF25+Soundscape gain:    +{sc25_gain:.4f}")

    use_focal            = focal_gain  > 0.002
    use_adamw_cw         = adamw_gain  > 0.002
    use_time_masking     = full_gain > (focal_gain + adamw_gain + 0.002)
    use_min_rating       = scores["birdclef25-base"] > max(
        scores["focal-isolated"], scores["adamw-classweights"]
    )
    use_soundscape_train = sc_gain > 0.002 or sc25_gain > 0.002

    log(f"  → use_focal={use_focal}, use_adamw_cw={use_adamw_cw}, "
        f"use_time_masking={use_time_masking}, use_min_rating={use_min_rating}, "
        f"use_soundscape_train={use_soundscape_train}")

    loss_cfg             = '"focal"' if use_focal else '"bce"'
    optimizer_cfg        = '"adamw"' if use_adamw_cw else '"adam"'
    cw_mode              = '"sqrt"'  if use_adamw_cw else '"none"'
    time_mask_cfg        = "true"    if use_time_masking else "false"
    min_rating_cfg       = "3.0"     if use_min_rating else "0.0"
    soundscape_train_cfg = "true"    if use_soundscape_train else "false"

    run_name    = f"best-derived-v{version}"
    config_path = f"configs/best_derived_v{version}.yaml"

    config_content = f"""experiment:
  name: "{run_name}"
  seed: 42

data:
  train_audio_dir: "birdclef-2026/train_audio"
  train_soundscapes_dir: "birdclef-2026/train_soundscapes"
  train_csv: "birdclef-2026/train.csv"
  taxonomy_csv: "birdclef-2026/taxonomy.csv"
  soundscapes_labels_csv: "birdclef-2026/train_soundscapes_labels.csv"
  sample_submission_csv: "birdclef-2026/sample_submission.csv"
  min_rating: {min_rating_cfg}
  use_secondary_labels: true
  max_files: null
  noise_dir: null

audio:
  sample_rate: 32000
  clip_duration: 5
  n_clips_per_file: 3

model:
  perch_dir: "models/bird-vocalization-classifier-tensorflow2-perch_v2-v2"
  mode: "embedding_head"
  hidden_dim: 512
  dropout: 0.3

training:
  epochs: 50
  batch_size: 256
  optimizer: {optimizer_cfg}
  learning_rate: 1.0e-3
  weight_decay: 1.0e-4
  scheduler: "cosine"
  warmup_epochs: 3
  label_smoothing: 0.05
  mixup_alpha: 0.3
  use_soundscapes_in_train: {soundscape_train_cfg}
  loss: {loss_cfg}
  focal_gamma: 2.0
  focal_alpha: 0.25
  class_weight_mode: {cw_mode}

augmentation:
  enabled: true
  noise_level: 0.005
  gain_range: [0.7, 1.3]
  time_masking: {time_mask_cfg}
  time_mask_ratio: 0.1
  time_mask_n: 2
  background_noise: false
  snr_db_range: [5.0, 30.0]

cache:
  enabled: true
  cache_dir: "outputs/embeddings_cache"

wandb:
  enabled: true
  project: "birdclef-2026"
  entity: null
  tags: ["best-derived", "auto-generated", "v{version}"]
  log_model: false

output:
  dir: "outputs"
  checkpoint_dir: "checkpoints"
"""
    with open(config_path, "w") as f:
        f.write(config_content)
    log(f"New config written → {config_path}")
    return config_path


def run_phase3(config_path: str):
    """Phase 3 (GPU 0): best-derived Perch config."""
    log("═══ Phase 3: best-derived-v1 (GPU 0) ═══")
    p0 = run_bg(["python", "train.py", "--config", config_path], gpu=0)
    p0.wait()
    log("Phase 3 complete.")


# ── Report ────────────────────────────────────────────────────────────────

def generate_report():
    log("Generating HTML report …")
    os.makedirs("reports", exist_ok=True)
    result = subprocess.run(
        ["python", "generate_report.py",
         "--output", "reports/experiment_report.html"],
        capture_output=True, text=True
    )
    log(result.stdout.strip())
    if result.returncode != 0:
        log(f"Report error: {result.stderr.strip()}")
    else:
        log("Report saved → reports/experiment_report.html")


def run_final_eval(gpu: int = 0):
    """Score all checkpoints with the official BirdCLEF 2026 metric."""
    log("═══ Final evaluation: official BirdCLEF ROC-AUC ═══")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    result = subprocess.run(
        ["python", "evaluate_final.py",
         "--config", "configs/default.yaml"],
        capture_output=True, text=True, env=env
    )
    log(result.stdout.strip())
    if result.returncode != 0:
        log(f"Eval error: {result.stderr.strip()}")
    else:
        log("Official scores written to outputs/*/result.json (kaggle_roc_auc)")


def run_analysis(gpu: int = 0, phase: int = 4) -> list:
    """Deep per-class analysis → experiment plan."""
    log(f"═══ Phase {phase}: Deep analysis + experiment planning ═══")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    plan_key = f"phase{phase}_plan.json"
    result = subprocess.run(
        ["python", "analyze_and_plan.py",
         "--config", "configs/default.yaml",
         "--gpu", str(gpu),
         "--plan_key", plan_key],
        capture_output=True, text=True, env=env
    )
    log(result.stdout.strip())
    if result.returncode != 0:
        log(f"Analysis error: {result.stderr.strip()}")
        return []
    log(f"Analysis complete. Plan saved → outputs/{plan_key}")

    plan_path = os.path.join("outputs", plan_key)
    try:
        with open(plan_path) as f:
            plan = json.load(f)
        return plan.get("phase4_experiments", [])
    except Exception:
        return []


def run_phase4(experiments: list):
    """
    Run Phase 4 experiments two at a time (GPU 0 + GPU 1).
    Pseudo-label + longer-training run in parallel as the final batch.
    """
    if not experiments:
        log("No Phase 4 experiments to run.")
        return

    # Split by type
    ablations  = [e for e in experiments if e.get("type") not in ("pseudo_label",)]
    pseudo_exp = [e for e in experiments if e.get("type") == "pseudo_label"]

    # Run ablations two at a time
    log(f"═══ Phase 4a: {len(ablations)} ablation experiments (2 per GPU batch) ═══")
    for i in range(0, len(ablations), 2):
        batch = ablations[i: i + 2]
        procs = []
        for gpu_id, exp in enumerate(batch):
            if not os.path.isfile(exp["config"]):
                log(f"  Config not found, skipping: {exp['config']}")
                continue
            log(f"  [{exp['name']} on GPU {gpu_id}]: {exp['rationale'][:72]}")
            p = run_bg(["python", "train.py",
                        "--config", exp["config"],
                        f"experiment.name={exp['name']}"], gpu=gpu_id)
            procs.append(p)
        for p in procs:
            p.wait()

    generate_report()

    # Pseudo-labeling + longer-training in parallel (both GPUs used)
    if pseudo_exp:
        log("═══ Phase 4b: pseudo-r1 (GPU 0) + longer-training (GPU 1) [parallel] ═══")
        procs = []
        p0 = run_bg(["python", "train.py",
                     "--config", "configs/pseudo_label_round1.yaml",
                     "experiment.name=pseudo-r1"], gpu=0)
        procs.append(p0)
        if os.path.isfile("configs/exp_longer_training.yaml") and \
                not is_finished("longer-training"):
            p1 = run_bg(["python", "train.py",
                         "--config", "configs/exp_longer_training.yaml",
                         "experiment.name=longer-training"], gpu=1)
            procs.append(p1)
        for p in procs:
            p.wait()
        log("Phase 4b complete.")


def derive_best_config_v2(phase4_experiments: list) -> str:
    """
    After Phase 4, pick the best-performing Phase 4 run and build best-derived-v2
    that layers the winning Phase 4 technique on top of best-derived-v1.
    Returns the new config path.
    """
    import glob
    import yaml

    # Find best Phase 4 run by kaggle_roc_auc or best_val_roc_auc
    best_name, best_score = None, -1.0
    for exp in phase4_experiments:
        name = exp.get("name", "")
        path = os.path.join("outputs", name, "result.json")
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                d = json.load(f)
            s = d.get("kaggle_roc_auc") or d.get("best_val_roc_auc") or 0
            if s > best_score:
                best_score, best_name = s, name
        except Exception:
            pass

    # Also check best-derived-v1
    v1_score = best_cmap("best-derived-v1")
    if v1_score > best_score:
        best_score = v1_score
        best_name = "best-derived-v1"

    log(f"Best Phase 4 run: {best_name} ({best_score:.4f}) → base for v2")

    # Load best run's config as base
    base_path = f"configs/best_derived_v1.yaml"
    if best_name and os.path.isfile(f"configs/{best_name.replace('-', '_')}.yaml"):
        base_path = f"configs/{best_name.replace('-', '_')}.yaml"

    try:
        with open(base_path) as f:
            cfg = yaml.safe_load(f)
    except Exception:
        return derive_best_config(version=2)

    # Upgrade config
    cfg["experiment"] = {"name": "best-derived-v2", "seed": 42}
    cfg["training"]["epochs"] = 80
    cfg["training"]["warmup_epochs"] = 5

    # Incorporate taxon upweighting if it helped
    taxon_score = best_cmap("taxon-balanced")
    if taxon_score > v1_score + 0.002:
        cfg["training"]["class_weight_mode"] = "taxon_upweight"
        cfg["training"]["taxon_nonbird_boost"] = 3.0
        log("  → Including taxon upweighting in v2")

    # Incorporate focal loss if it helped
    focal_score = best_cmap("focal-stronger")
    if focal_score > v1_score + 0.002:
        cfg["training"]["loss"] = "focal"
        cfg["training"]["focal_gamma"] = 3.0
        log("  → Including stronger focal loss in v2")

    config_path = "configs/best_derived_v2.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    log(f"best-derived-v2 config written → {config_path}")
    return config_path


def extract_nohuman_embeddings(gpu: int = 0):
    """Pre-extract embeddings with Silero VAD human-voice removal.
    Saves to outputs/embeddings_cache_nohuman/ (separate from the default cache).
    Only runs if the filtered cache doesn't already have a manifest.
    """
    import os
    nohuman_manifest = os.path.join("outputs", "embeddings_cache_nohuman", "manifest.csv")
    if os.path.isfile(nohuman_manifest):
        log("Human-voice-filtered cache already exists, skipping extraction.")
        return
    log("═══ Extracting human-voice-filtered embeddings (Silero VAD) ═══")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    result = subprocess.run(
        ["python", "extract_embeddings.py",
         "--config", "configs/default.yaml",
         "--filter_human_voice",
         "--vad_threshold", "0.4",
         "--gpu", str(gpu)],
        env=env
    )
    if result.returncode != 0:
        log("Warning: human-voice extraction failed — skipping no-human-voice experiment.")
    else:
        log("Human-voice-filtered cache ready → outputs/embeddings_cache_nohuman/")


def run_phase5(phase4_experiments: list):
    """
    Phase 5: derive best-derived-v2 from Phase 4 winners, then run:
      - best-derived-v2 (GPU 0) + pseudo-r2 (GPU 1) in parallel
    Then a second round of analysis-driven experiments.
    """
    log("═══ Phase 5: Deriving best-v2 from Phase 4 results ═══")

    # Pre-extract human-voice-filtered embeddings on GPU 1
    # while best-derived-v2 runs on GPU 0
    # (or extract first if GPUs are free)
    extract_nohuman_embeddings(gpu=0)

    # Derive improved config
    config_v2 = derive_best_config_v2(phase4_experiments)

    # Run best-derived-v2 (GPU 0) + no-human-voice (GPU 1)
    log("═══ Phase 5a: best-derived-v2 (GPU 0) + no-human-voice (GPU 1) ═══")
    procs = []
    p0 = run_bg(["python", "train.py", "--config", config_v2], gpu=0)
    procs.append(p0)

    # GPU 1: no-human-voice experiment (requires filtered cache)
    if os.path.isfile("configs/exp_no_human_voice.yaml") and \
            not is_finished("no-human-voice"):
        nohuman_manifest = os.path.join(
            "outputs", "embeddings_cache_nohuman", "manifest.csv")
        if os.path.isfile(nohuman_manifest):
            p1 = run_bg(["python", "train.py",
                         "--config", "configs/exp_no_human_voice.yaml",
                         "experiment.name=no-human-voice"], gpu=1)
            procs.append(p1)
        else:
            log("  no-human-voice cache not ready, skipping GPU 1.")
    elif os.path.isfile("configs/pseudo_label_round2.yaml") and \
            not is_finished("pseudo-r2"):
        p1 = run_bg(["python", "train.py",
                     "--config", "configs/pseudo_label_round2.yaml",
                     "experiment.name=pseudo-r2"], gpu=1)
        procs.append(p1)

    for p in procs:
        p.wait()
    log("Phase 5a complete.")

    generate_report()

    # Second analysis round → Phase 5b experiments
    phase5_experiments = run_analysis(gpu=0, phase=5)
    if phase5_experiments:
        log(f"═══ Phase 5b: {len(phase5_experiments)} more experiments ═══")
        ablations = [e for e in phase5_experiments if e.get("type") != "pseudo_label"]
        for i in range(0, len(ablations), 2):
            batch = ablations[i: i + 2]
            procs = []
            for gpu_id, exp in enumerate(batch):
                if not os.path.isfile(exp["config"]):
                    log(f"  Config not found, skipping: {exp['config']}")
                    continue
                log(f"  [{exp['name']} on GPU {gpu_id}]: {exp['rationale'][:72]}")
                p = run_bg(["python", "train.py",
                            "--config", exp["config"],
                            f"experiment.name={exp['name']}"], gpu=gpu_id)
                procs.append(p)
            for p in procs:
                p.wait()


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    log("Orchestration started.")

    # Phase 1: wait for the two experiments already running
    wait_for(["baseline", "birdclef25-base"])

    # Intermediate report (after phase 1)
    generate_report()

    # Phase 2a: focal-isolated + adamw-classweights (parallel)
    run_phase2a()

    # Phase 2b: soundscape-in-train (GPU 0) + birdclef25-soundscape (GPU 1) [parallel]
    run_phase2b()

    # Report after all ablations
    generate_report()

    # Phase 3: derive best config (now has all 6 ablation scores) + run
    config_path = derive_best_config(version=1)
    run_phase3(config_path)

    # Final report
    generate_report()

    # Official metric evaluation — score all checkpoints
    run_final_eval(gpu=0)

    # Regenerate report with official scores included
    generate_report()
    log("Phases 1–3 complete. Starting deep analysis …")

    # Phase 4: analyse results, plan and run new experiments
    phase4_experiments = run_analysis(gpu=0, phase=4)
    run_phase4(phase4_experiments)

    # Official eval + report after Phase 4
    run_final_eval(gpu=0)
    generate_report()

    # Phase 5: derive v2, run more experiments, second analysis loop
    run_phase5(phase4_experiments)

    # Final eval + report
    run_final_eval(gpu=0)
    generate_report()
    log("All phases complete. Final report → reports/experiment_report.html")


if __name__ == "__main__":
    main()
