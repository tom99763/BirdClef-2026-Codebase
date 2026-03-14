"""Generate an HTML technical report from all experiment result.json files.

Usage:
    python generate_report.py
    python generate_report.py --output reports/report.html
"""

import argparse
import glob
import json
import os
from datetime import datetime


# ── Load results ─────────────────────────────────────────────────────────────

def load_results(outputs_dir: str = "outputs") -> list:
    results = []
    for path in sorted(glob.glob(os.path.join(outputs_dir, "*/result.json"))):
        try:
            with open(path) as f:
                d = json.load(f)
            d["_path"] = path
            results.append(d)
        except Exception:
            pass
    return results


# ── Technique attribution ─────────────────────────────────────────────────────

EXPERIMENT_META = {
    "baseline": {
        "label": "Baseline",
        "color": "#6c757d",
        "desc": "Adam · BCE · no class weights · all data",
        "changes": [],
    },
    "pseudo-r1": {
        "label": "Pseudo-label R1",
        "color": "#6f42c1",
        "desc": "Best-derived + pseudo-labels (PowerTransform power=2)",
        "changes": ["pseudo-labeling", "PowerTransform"],
    },
    "linear-classweights": {
        "label": "Linear Class Weights",
        "color": "#20c997",
        "desc": "Linear inverse-frequency weighting (stronger than sqrt)",
        "changes": ["linear class weights"],
    },
    "more-clips": {
        "label": "More Clips (5/file)",
        "color": "#0dcaf0",
        "desc": "n_clips_per_file=5 for rare species coverage",
        "changes": ["n_clips=5"],
    },
    "focal-stronger": {
        "label": "FocalLoss γ=3",
        "color": "#fd7e14",
        "desc": "Stronger FocalLoss gamma=3 for taxonomy imbalance",
        "changes": ["FocalLoss γ=3"],
    },
    "soundscape-heavy": {
        "label": "Soundscape 3×",
        "color": "#198754",
        "desc": "3× soundscape oversampling to reduce domain gap",
        "changes": ["soundscape 3×"],
    },
    "larger-head": {
        "label": "Larger Head",
        "color": "#dc3545",
        "desc": "Dense(1024)→Dense(512)→Dense(234) deeper head",
        "changes": ["hidden_dim=1024", "2 hidden layers"],
    },
    "longer-training": {
        "label": "Longer Training (100ep)",
        "color": "#6610f2",
        "desc": "100 epochs to check if model saturated at 50",
        "changes": ["epochs=100"],
    },
    "focal-isolated": {
        "label": "FocalLoss (isolated)",
        "color": "#fd7e14",
        "desc": "Adam · FocalLoss(γ=2) · no class weights · all data",
        "changes": ["FocalLoss γ=2"],
    },
    "adamw-classweights": {
        "label": "AdamW + Class Weights (isolated)",
        "color": "#0d6efd",
        "desc": "AdamW · BCE · sqrt class weights · all data",
        "changes": ["AdamW", "sqrt class weights"],
    },
    "birdclef25-base": {
        "label": "BirdCLEF25 Full",
        "color": "#198754",
        "desc": "AdamW · FocalLoss · sqrt weights · time masking · min_rating=3",
        "changes": ["AdamW", "FocalLoss γ=2", "sqrt class weights", "time masking", "min_rating=3"],
    },
    "taxon-balanced": {
        "label": "Taxon-Balanced Sampling",
        "color": "#f0a500",
        "desc": "3× boost for non-Aves species (Amphibia/Reptilia/Insecta/Mammalia)",
        "changes": ["taxon_upweight 3×"],
    },
    "taxon-multitask": {
        "label": "Multi-task Taxonomy Loss",
        "color": "#e83e8c",
        "desc": "Auxiliary taxonomy head (weight=0.1) + taxon-balanced sampling",
        "changes": ["taxonomy aux loss", "taxon_upweight 3×"],
    },
    "sed-extended-freq": {
        "label": "SED Extended Freq (20kHz)",
        "color": "#17a2b8",
        "desc": "fmax=20kHz, 256 mel bins — optimised for insects/frogs",
        "changes": ["fmax=20kHz", "n_mels=256"],
    },
    "birdclef25-soundscape": {
        "label": "BirdCLEF25 + Soundscape",
        "color": "#2ecc71",
        "desc": "min_rating=3.0 (quality filter) + soundscapes (domain adaptation)",
        "changes": ["min_rating=3", "soundscapes"],
    },
    "soundscape-focal": {
        "label": "Soundscape + FocalLoss",
        "color": "#e67e22",
        "desc": "Soundscape training + FocalLoss(γ=2) for class imbalance",
        "changes": ["soundscapes", "FocalLoss γ=2"],
    },
    "soundscape-lowlr": {
        "label": "Soundscape Low LR",
        "color": "#1abc9c",
        "desc": "Soundscape training with LR=1e-4 for stable convergence",
        "changes": ["soundscapes", "LR=1e-4"],
    },
    "mixup-heavy": {
        "label": "Heavy Mixup (α=0.6)",
        "color": "#9b59b6",
        "desc": "mixup_alpha=0.6 (stronger) for better generalisation",
        "changes": ["mixup_alpha=0.6"],
    },
    "pseudo-r2": {
        "label": "Pseudo-label R2",
        "color": "#8e44ad",
        "desc": "Second pseudo-label round on top of pseudo-r1 checkpoint",
        "changes": ["pseudo-labeling R2"],
    },
    "best-derived-v2": {
        "label": "Best Derived v2",
        "color": "#c0392b",
        "desc": "Auto-derived config combining best Phase 4 techniques",
        "changes": ["auto-derived", "Phase 4 winners"],
    },
    "soundscape-heavy": {
        "label": "Soundscape 3× Oversample",
        "color": "#27ae60",
        "desc": "min_rating=3.0 + soundscapes 3× oversampled — aggressive domain adaptation",
        "changes": ["soundscapes 3×", "min_rating=3"],
    },
    "soundscape-taxon-multitask": {
        "label": "Soundscape + Taxon Multi-task",
        "color": "#8e44ad",
        "desc": "Soundscapes + auxiliary taxonomy loss + non-bird 3× upweight",
        "changes": ["soundscapes", "taxon aux loss", "taxon_upweight 3×"],
    },
    "soundscape-adamw": {
        "label": "Soundscape + AdamW",
        "color": "#2980b9",
        "desc": "Soundscapes + AdamW optimizer + sqrt class weights",
        "changes": ["soundscapes", "AdamW", "sqrt class weights"],
    },
    "no-human-voice": {
        "label": "No Human Voice (Silero VAD)",
        "color": "#e74c3c",
        "desc": "Silero VAD removes human speech before Perch embedding extraction",
        "changes": ["Silero VAD", "human voice removed", "min_rating=3", "soundscapes"],
    },
}


def get_meta(run_name: str) -> dict:
    for key, meta in EXPERIMENT_META.items():
        if key in run_name:
            return meta
    return {"label": run_name, "color": "#adb5bd", "desc": "", "changes": []}


# ── HTML template ─────────────────────────────────────────────────────────────

def build_html(results: list) -> str:
    if not results:
        return "<html><body><h1>No results found.</h1></body></html>"

    # Sort by official kaggle_roc_auc if available, else best_val_roc_auc
    has_official = any(r.get("kaggle_roc_auc") is not None for r in results)
    sort_key = "kaggle_roc_auc" if has_official else "best_val_roc_auc"
    results_sorted = sorted(
        results,
        key=lambda r: r.get(sort_key) or r.get("best_val_roc_auc") or r.get("best_val_cmap") or 0,
        reverse=True,
    )

    baseline_cmap = next(
        (r["best_val_roc_auc"] for r in results if "baseline" in r.get("run_name", "")), None
    )

    # ── Rankings table rows ───────────────────────────────────────────────────
    ranking_rows = ""
    for rank, r in enumerate(results_sorted, 1):
        name = r.get("run_name", "?")
        meta = get_meta(name)
        cmap = r.get("best_val_roc_auc") or r.get("best_val_cmap") or 0
        kaggle_cmap = r.get("kaggle_roc_auc")
        best_ep = r.get("best_epoch", "?")
        total_eps = r.get("total_epochs_run", "?")
        total_t = r.get("total_time_s")
        finished = r.get("finished", False)
        delta = f"+{cmap - baseline_cmap:.4f}" if baseline_cmap and name != "baseline" else "—"
        delta_color = "text-success" if baseline_cmap and cmap > baseline_cmap else "text-danger"
        t_str = f"{total_t/60:.1f} min" if total_t else "running…"
        status_badge = (
            '<span class="badge bg-success">done</span>'
            if finished else
            '<span class="badge bg-warning text-dark">running</span>'
        )
        changes_html = " ".join(
            f'<span class="badge bg-secondary">{c}</span>' for c in meta["changes"]
        ) or '<span class="text-muted">—</span>'
        kaggle_str = (
            f'<span class="fw-bold text-primary">{kaggle_cmap:.4f}</span>'
            if kaggle_cmap is not None else
            '<span class="text-muted small">pending</span>'
        )
        ranking_rows += f"""
        <tr>
          <td class="fw-bold">#{rank}</td>
          <td>
            <span class="fw-semibold" style="color:{meta['color']}">{meta['label']}</span><br>
            <small class="text-muted">{meta['desc']}</small>
          </td>
          <td class="text-center fw-bold">{cmap:.4f}</td>
          <td class="text-center">{kaggle_str}</td>
          <td class="text-center {delta_color}">{delta}</td>
          <td class="text-center">{best_ep} / {total_eps}</td>
          <td class="text-center">{t_str}</td>
          <td>{changes_html}</td>
          <td class="text-center">{status_badge}</td>
        </tr>"""

    # ── Epoch curves (Chart.js) ───────────────────────────────────────────────
    datasets_cmap = []
    datasets_loss = []
    for r in results_sorted:
        name = r.get("run_name", "?")
        meta = get_meta(name)
        history = r.get("epoch_history", [])
        cmap_data = [{"x": e["epoch"], "y": round(e.get("val_roc_auc", e.get("val_cmap", 0)), 5)} for e in history]
        loss_data = [{"x": e["epoch"], "y": round(e["train_loss"], 6)} for e in history]
        datasets_cmap.append(
            f'{{"label":"{meta["label"]}","data":{json.dumps(cmap_data)},'
            f'"borderColor":"{meta["color"]}","backgroundColor":"{meta["color"]}22",'
            f'"tension":0.3,"pointRadius":2,"fill":false}}'
        )
        datasets_loss.append(
            f'{{"label":"{meta["label"]}","data":{json.dumps(loss_data)},'
            f'"borderColor":"{meta["color"]}","backgroundColor":"{meta["color"]}22",'
            f'"tension":0.3,"pointRadius":2,"fill":false}}'
        )

    datasets_cmap_js = "[" + ",".join(datasets_cmap) + "]"
    datasets_loss_js = "[" + ",".join(datasets_loss) + "]"

    # ── Technique attribution bar chart ──────────────────────────────────────
    attr_labels, attr_values, attr_colors = [], [], []
    techniques = {
        "FocalLoss (isolated)": "focal-isolated",
        "AdamW+ClassWeights (isolated)": "adamw-classweights",
        "BirdCLEF25 Full": "birdclef25-base",
    }
    if baseline_cmap:
        for label, key in techniques.items():
            match = next((r for r in results if key in r.get("run_name", "")), None)
            if match:
                gain = match["best_val_roc_auc"] - baseline_cmap
                attr_labels.append(label)
                attr_values.append(round(gain, 5))
                attr_colors.append(get_meta(key)["color"])

    attr_js = (
        f'labels:{json.dumps(attr_labels)},'
        f'datasets:[{{label:"ΔcMAP vs Baseline",'
        f'data:{json.dumps(attr_values)},'
        f'backgroundColor:{json.dumps(attr_colors)}}}]'
    )

    # ── Best hyperparameters summary ──────────────────────────────────────────
    best = results_sorted[0]
    best_hparams = best.get("hparams", {})
    hparam_rows = "".join(
        f"<tr><td class='text-muted'>{k}</td><td class='fw-semibold'>{v}</td></tr>"
        for k, v in best_hparams.items()
    )

    # ── Epoch timing table ────────────────────────────────────────────────────
    timing_rows = ""
    for r in results_sorted:
        name = r.get("run_name", "?")
        meta = get_meta(name)
        history = r.get("epoch_history", [])
        if history:
            avg_t = sum(e.get("epoch_time_s", 0) for e in history) / len(history)
            timing_rows += (
                f"<tr><td style='color:{meta['color']}'>{meta['label']}</td>"
                f"<td class='text-center'>{avg_t:.1f}s</td>"
                f"<td class='text-center'>{len(history)}</td></tr>"
            )

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BirdCLEF 2026 — Experiment Report</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #f8f9fa; }}
  .card {{ border: none; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,.07); margin-bottom: 1.5rem; }}
  .card-header {{ border-radius: 12px 12px 0 0 !important; font-weight: 600; }}
  h1 {{ font-weight: 700; }}
  .metric-card {{ text-align: center; padding: 1.5rem; }}
  .metric-value {{ font-size: 2.2rem; font-weight: 700; }}
  .metric-label {{ color: #6c757d; font-size: .85rem; text-transform: uppercase; letter-spacing: 1px; }}
  table td, table th {{ vertical-align: middle; }}
  .chart-container {{ position: relative; height: 320px; }}
</style>
</head>
<body>
<div class="container py-4">

  <!-- Header -->
  <div class="d-flex justify-content-between align-items-start mb-4">
    <div>
      <h1 class="mb-1">🐦 BirdCLEF 2026 — Technical Report</h1>
      <p class="text-muted mb-0">Perch v2 Embedding Head · Ablation Study · Generated {now}</p>
    </div>
    <span class="badge bg-primary fs-6">Padded cMAP</span>
  </div>

  <!-- Summary metrics -->
  <div class="row g-3 mb-4">
    <div class="col-md-3">
      <div class="card metric-card">
        <div class="metric-value text-success">{results_sorted[0].get('best_val_roc_auc', 0):.4f}</div>
        <div class="metric-label">Best Val ROC-AUC</div>
        <div class="text-muted small mt-1">{get_meta(results_sorted[0]['run_name'])['label']}</div>
      </div>
    </div>
    <div class="col-md-3">
      <div class="card metric-card">
        {"<div class='metric-value text-primary'>" + f"{results_sorted[0]['kaggle_roc_auc']:.4f}</div><div class='metric-label'>Official ROC-AUC</div><div class='text-muted small mt-1'>macro · scored species only</div>" if results_sorted[0].get('kaggle_roc_auc') is not None else "<div class='metric-value text-muted'>—</div><div class='metric-label'>Official ROC-AUC</div><div class='text-muted small mt-1'>run evaluate_final.py</div>"}
      </div>
    </div>
    <div class="col-md-2">
      <div class="card metric-card">
        <div class="metric-value text-primary">{len(results)}</div>
        <div class="metric-label">Experiments</div>
        <div class="text-muted small mt-1">{sum(1 for r in results if r.get('finished'))} completed</div>
      </div>
    </div>
    <div class="col-md-2">
      <div class="card metric-card">
        <div class="metric-value text-warning">
          {f"+{results_sorted[0]['best_val_roc_auc'] - baseline_cmap:.4f}" if baseline_cmap else "—"}
        </div>
        <div class="metric-label">Best Gain vs Baseline</div>
        <div class="text-muted small mt-1">padded cMAP</div>
      </div>
    </div>
    <div class="col-md-2">
      <div class="card metric-card">
        <div class="metric-value text-info">
          {sum(r.get('total_epochs_run', 0) for r in results)}
        </div>
        <div class="metric-label">Total Epochs Run</div>
        <div class="text-muted small mt-1">across all experiments</div>
      </div>
    </div>
  </div>

  <!-- Rankings table -->
  <div class="card">
    <div class="card-header bg-dark text-white">📊 Experiment Rankings</div>
    <div class="card-body p-0">
      <div class="table-responsive">
        <table class="table table-hover mb-0">
          <thead class="table-light">
            <tr>
              <th>Rank</th><th>Experiment</th><th class="text-center">Val cMAP</th>
              <th class="text-center">Official ROC-AUC</th>
              <th class="text-center">Δ Baseline</th><th class="text-center">Best Epoch</th>
              <th class="text-center">Time</th><th>Techniques</th><th class="text-center">Status</th>
            </tr>
          </thead>
          <tbody>{ranking_rows}</tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Charts row -->
  <div class="row g-3">
    <div class="col-md-8">
      <div class="card">
        <div class="card-header bg-primary text-white">📈 Validation ROC-AUC over Epochs</div>
        <div class="card-body">
          <div class="chart-container">
            <canvas id="cmapChart"></canvas>
          </div>
        </div>
      </div>
    </div>
    <div class="col-md-4">
      <div class="card">
        <div class="card-header bg-warning text-dark">🎯 Technique Attribution (ΔcMAP)</div>
        <div class="card-body">
          <div class="chart-container">
            <canvas id="attrChart"></canvas>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Loss curve -->
  <div class="card">
    <div class="card-header bg-danger text-white">📉 Training Loss over Epochs</div>
    <div class="card-body">
      <div class="chart-container" style="height:240px">
        <canvas id="lossChart"></canvas>
      </div>
    </div>
  </div>

  <!-- Analysis row -->
  <div class="row g-3">
    <div class="col-md-6">
      <div class="card h-100">
        <div class="card-header bg-success text-white">🏆 Best Config Hyperparameters</div>
        <div class="card-body">
          <h6 class="text-muted">{get_meta(best['run_name'])['label']}</h6>
          <table class="table table-sm">
            <tbody>{hparam_rows}</tbody>
          </table>
        </div>
      </div>
    </div>
    <div class="col-md-6">
      <div class="card h-100">
        <div class="card-header bg-info text-white">⏱ Epoch Timing (with embedding cache)</div>
        <div class="card-body">
          <table class="table table-sm">
            <thead><tr><th>Experiment</th><th class="text-center">Avg sec/epoch</th><th class="text-center">Epochs</th></tr></thead>
            <tbody>{timing_rows}</tbody>
          </table>
          <div class="alert alert-success py-2 mt-2 mb-0 small">
            <strong>Cache speedup:</strong> Pre-computed Perch embeddings reduce epoch time
            from ~20 min → ~40–55s (batch_size=256, @tf.function).
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Findings -->
  <div class="card">
    <div class="card-header bg-secondary text-white">🔬 Findings &amp; Recommendations</div>
    <div class="card-body">
      <div class="row">
        <div class="col-md-6">
          <h6>Ablation Design</h6>
          <ul>
            <li><strong>Baseline</strong>: Adam · BCE · no class weights · all recordings</li>
            <li><strong>FocalLoss isolated</strong>: +FocalLoss(γ=2) only → measures loss function effect</li>
            <li><strong>AdamW + ClassWeights isolated</strong>: +AdamW + sqrt weights → measures optimizer &amp; sampling effect</li>
            <li><strong>BirdCLEF25 Full</strong>: all improvements combined + min_rating=3</li>
          </ul>
        </div>
        <div class="col-md-6">
          <h6>Architecture</h6>
          <ul>
            <li>Backbone: Google Perch v2 (frozen, 1536-dim embedding)</li>
            <li>Head: Dense(512) → ReLU → Dropout(0.3) → Dense(234)</li>
            <li>Training: cosine LR with 3-epoch warmup, mixup α=0.3</li>
            <li>Validation: labeled soundscape segments (matches test format)</li>
          </ul>
        </div>
      </div>
    </div>
  </div>

  <p class="text-muted text-center small mt-3">
    Generated by <code>generate_report.py</code> · BirdCLEF 2026 · {now}
  </p>
</div>

<script>
const cmapCtx = document.getElementById('cmapChart').getContext('2d');
new Chart(cmapCtx, {{
  type: 'line',
  data: {{ datasets: {datasets_cmap_js} }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    scales: {{
      x: {{ type: 'linear', title: {{ display: true, text: 'Epoch' }} }},
      y: {{ title: {{ display: true, text: 'val ROC-AUC' }}, beginAtZero: false }}
    }},
    plugins: {{ legend: {{ position: 'top' }} }}
  }}
}});

const lossCtx = document.getElementById('lossChart').getContext('2d');
new Chart(lossCtx, {{
  type: 'line',
  data: {{ datasets: {datasets_loss_js} }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    scales: {{
      x: {{ type: 'linear', title: {{ display: true, text: 'Epoch' }} }},
      y: {{ type: 'logarithmic', title: {{ display: true, text: 'Train Loss (log)' }} }}
    }},
    plugins: {{ legend: {{ position: 'top' }} }}
  }}
}});

const attrCtx = document.getElementById('attrChart').getContext('2d');
new Chart(attrCtx, {{
  type: 'bar',
  data: {{ {attr_js} }},
  options: {{
    indexAxis: 'y',
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{ x: {{ title: {{ display: true, text: 'ΔcMAP vs Baseline' }} }} }}
  }}
}});
</script>
</body>
</html>"""


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--output", default="reports/experiment_report.html")
    args = parser.parse_args()

    results = load_results(args.outputs_dir)
    print(f"Loaded {len(results)} experiment(s)")
    for r in sorted(results, key=lambda x: x.get("best_val_roc_auc", 0), reverse=True):
        print(f"  {r.get('run_name','?'):30s}  cMAP={r.get('best_val_roc_auc',0):.4f}  "
              f"finished={r.get('finished', False)}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    html = build_html(results)
    with open(args.output, "w") as f:
        f.write(html)
    print(f"\nReport saved → {args.output}")


if __name__ == "__main__":
    main()
