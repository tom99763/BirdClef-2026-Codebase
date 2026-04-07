"""Update sed_improvement_plan.html with live experiment results.

Injects a "Live Results" section into the HTML report based on:
  - outputs/*/result.json          (all SED experiments)
  - outputs/phase1_inference_eval.json
  - outputs/ensemble_v3_holdout_eval.log
  - outputs/ensemble_v2_holdout_eval.log

Usage:
    python scripts/update_plan_report.py
"""

import json
import os
import re
import glob
from datetime import datetime


HTML_PATH = "reports/sed_improvement_plan.html"


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def parse_ensemble_log(path):
    """Extract holdout AUC values from ensemble eval log."""
    results = {}
    if not os.path.isfile(path):
        return results
    with open(path) as f:
        for line in f:
            m = re.search(r"([\w\(\)×+\-]+)\s+([\d.]+)\s+raw", line)
            if m:
                results[m.group(1)] = float(m.group(2))
    return results


def build_results_block():
    """Build complete HTML block for live results."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    rows_html = ""
    summary_cards = ""
    phases_done = []

    # ── SED experiments ────────────────────────────────────────────────────────
    sed_results = []
    for path in sorted(glob.glob("outputs/*/result.json")):
        d = load_json(path)
        if not d:
            continue
        name = d["run_name"]
        if name in ("sed-b0-v3", "sed-b0-v4"):  # skip killed runs
            continue
        finished = d.get("finished", False)
        best = d.get("best_val_roc_auc", 0)
        best_ep = d.get("best_epoch", 0)
        total_ep = d.get("total_epochs_run", 0)
        epochs = d.get("hparams", {}).get("epochs", "?")
        hist = d.get("epoch_history", [])
        last5 = hist[-5:] if len(hist) >= 5 else hist
        status = "✅ Done" if finished else f"🔄 ep{total_ep}/{epochs}"
        badge_col = "#3fb950" if finished else "#d29922"

        spark = " → ".join(f"{h['val_roc_auc']:.4f}" for h in last5)
        trend = ""
        if len(last5) >= 2:
            delta = last5[-1]["val_roc_auc"] - last5[-2]["val_roc_auc"]
            trend = f'<span style="color:{"#3fb950" if delta>0 else "#f85149"}">{"↑" if delta>0 else "↓"}{abs(delta):.4f}</span>'

        rows_html += f"""
        <tr>
          <td><span style="color:#e6edf3;font-weight:600">{name}</span></td>
          <td><span style="background:{badge_col};color:#0d1117;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700">{status}</span></td>
          <td style="color:#79c0ff">{best:.4f} @ep{best_ep}</td>
          <td style="color:#8b949e;font-size:12px">{spark}</td>
          <td>{trend}</td>
        </tr>"""
        sed_results.append(d)

    # ── Phase 1 inference eval ─────────────────────────────────────────────────
    p1 = load_json("outputs/phase1_inference_eval.json")
    p1_html = ""
    if p1:
        phases_done.append("Phase 1")
        variants = p1.get("variants", {})
        baseline = p1.get("baseline_auc", 0)
        best_var = p1.get("best_variant", "")
        best_auc = p1.get("best_auc", 0)
        best_gain = p1.get("best_gain", 0)

        var_rows = ""
        for var, auc in sorted(variants.items(), key=lambda x: -(x[1] or 0)):
            delta = (auc or 0) - baseline
            is_best = (var == best_var)
            bg = ' style="background:#0d2a0d"' if is_best else ""
            var_rows += f"""
            <tr{bg}>
              <td style="color:{"#39d353" if is_best else "#e6edf3"};font-weight:{"700" if is_best else "400"}">{var}{"  ⭐" if is_best else ""}</td>
              <td style="color:#79c0ff">{auc:.4f if auc else "N/A"}</td>
              <td style="color:{"#3fb950" if delta>0 else "#f85149"}">{delta:+.4f if auc else "—"}</td>
            </tr>"""

        p1_html = f"""
        <div style="margin-top:20px">
          <div style="font-size:13px;font-weight:600;color:#3fb950;margin-bottom:10px">
            ✅ Phase 1 Complete — Inference Variants (checkpoint ep{p1.get("checkpoint_epoch","?")})</div>
          <div style="color:#8b949e;font-size:12px;margin-bottom:8px">
            Best: <span style="color:#39d353;font-weight:700">{best_var}</span>
            SS Val AUC = <span style="color:#39d353;font-weight:700">{best_auc:.4f}</span>
            (gain <span style="color:#39d353">+{best_gain:.4f}</span> vs baseline {baseline:.4f})
          </div>
          <table style="width:100%;border-collapse:collapse;font-size:12px">
            <thead><tr>
              <th style="background:#1c2128;color:#8b949e;padding:6px 10px;text-align:left;border-bottom:1px solid #30363d">Variant</th>
              <th style="background:#1c2128;color:#8b949e;padding:6px 10px;text-align:left;border-bottom:1px solid #30363d">SS Val AUC</th>
              <th style="background:#1c2128;color:#8b949e;padding:6px 10px;text-align:left;border-bottom:1px solid #30363d">vs Baseline</th>
            </tr></thead>
            <tbody>{var_rows}</tbody>
          </table>
        </div>"""
        summary_cards += f"""
        <div style="background:#0d2a0d;border:1px solid #2d6a2d;border-radius:8px;padding:14px;text-align:center">
          <div style="font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px">Phase 1 Best Inference</div>
          <div style="font-size:22px;font-weight:700;color:#39d353">{best_auc:.4f}</div>
          <div style="font-size:11px;color:#8b949e">{best_var} (+{best_gain:.4f})</div>
        </div>"""

    # ── Ensemble v3 holdout eval ───────────────────────────────────────────────
    v3_results = parse_ensemble_log("outputs/ensemble_v3_holdout_eval.log")
    v3_html = ""
    if v3_results:
        phases_done.append("Ensemble v3")
        v3_rows = ""
        for model_name, auc in sorted(v3_results.items(), key=lambda x: -x[1]):
            is_best = (auc == max(v3_results.values()))
            bg = ' style="background:#0d2137"' if is_best else ""
            v3_rows += f"""
            <tr{bg}>
              <td style="color:{"#58a6ff" if is_best else "#e6edf3"}">{model_name}{"  ⭐" if is_best else ""}</td>
              <td style="color:#79c0ff;font-weight:{"700" if is_best else "400"}">{auc:.4f}</td>
            </tr>"""
        best_v3 = max(v3_results.values())
        v3_html = f"""
        <div style="margin-top:20px">
          <div style="font-size:13px;font-weight:600;color:#58a6ff;margin-bottom:10px">
            ✅ 4-Model Ensemble v3 Holdout Results</div>
          <table style="width:100%;border-collapse:collapse;font-size:12px">
            <thead><tr>
              <th style="background:#1c2128;color:#8b949e;padding:6px 10px;text-align:left;border-bottom:1px solid #30363d">Model</th>
              <th style="background:#1c2128;color:#8b949e;padding:6px 10px;text-align:left;border-bottom:1px solid #30363d">Holdout AUC</th>
            </tr></thead>
            <tbody>{v3_rows}</tbody>
          </table>
        </div>"""
        summary_cards += f"""
        <div style="background:#0d2137;border:1px solid #1a4a7a;border-radius:8px;padding:14px;text-align:center">
          <div style="font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px">4-Model Ensemble Holdout</div>
          <div style="font-size:22px;font-weight:700;color:#58a6ff">{best_v3:.4f}</div>
          <div style="font-size:11px;color:#8b949e">Perch×3 + SED</div>
        </div>"""

    # ── v2 ensemble for reference ──────────────────────────────────────────────
    v2_results = parse_ensemble_log("outputs/ensemble_v2_holdout_eval.log")
    v2_best = max(v2_results.values()) if v2_results else None
    if v2_best:
        summary_cards = f"""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;text-align:center">
          <div style="font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px">3-Model Perch Ensemble</div>
          <div style="font-size:22px;font-weight:700;color:#f0883e">0.9780</div>
          <div style="font-size:11px;color:#8b949e">Holdout AUC (baseline)</div>
        </div>""" + summary_cards

    # ── Active experiments status bar ──────────────────────────────────────────
    active = [d for d in sed_results if not d.get("finished")]
    active_html = ""
    if active:
        for d in active:
            name = d["run_name"]
            total = d.get("total_epochs_run", 0)
            epochs = d.get("hparams", {}).get("epochs", 20)
            pct = int(100 * total / max(epochs, 1))
            best = d.get("best_val_roc_auc", 0)
            active_html += f"""
            <div style="margin-bottom:10px">
              <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px">
                <span style="color:#e6edf3;font-weight:600">{name}</span>
                <span style="color:#8b949e">ep{total}/{epochs}  best={best:.4f}</span>
              </div>
              <div style="background:#30363d;border-radius:4px;height:6px">
                <div style="background:#58a6ff;width:{pct}%;height:6px;border-radius:4px"></div>
              </div>
            </div>"""

    # ── Assemble full block ────────────────────────────────────────────────────
    block = f"""
<!-- LIVE_RESULTS_START -->
<hr>
<div id="live-results" style="scroll-margin-top:20px">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px">
    <span style="background:#161b22;border:1px solid #3fb950;color:#3fb950;border-radius:20px;padding:3px 12px;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.5px">Live Results</span>
    <h2 style="font-size:18px;font-weight:600;color:#e6edf3">Experiment Progress &amp; Results</h2>
    <span style="font-size:11px;color:#8b949e;margin-left:auto">Updated: {now}</span>
  </div>

  <!-- KPI Summary Cards -->
  <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px">
    {summary_cards}
  </div>

  <!-- Active training progress bars -->
  {f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;margin-bottom:16px"><div style="font-size:11px;color:#d29922;font-weight:600;text-transform:uppercase;letter-spacing:.5px;margin-bottom:10px">Active Training</div>{active_html}</div>' if active_html else ""}

  <!-- SED experiment table -->
  <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;overflow:hidden;margin-bottom:16px">
    <div style="padding:10px 14px;border-bottom:1px solid #30363d;font-size:11px;color:#8b949e;font-weight:600;text-transform:uppercase;letter-spacing:.5px">
      SED Experiments (SS Val AUC)
    </div>
    <table style="width:100%;border-collapse:collapse;font-size:13px">
      <thead><tr>
        <th style="background:#1c2128;color:#8b949e;padding:8px 12px;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.5px;border-bottom:1px solid #30363d">Experiment</th>
        <th style="background:#1c2128;color:#8b949e;padding:8px 12px;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.5px;border-bottom:1px solid #30363d">Status</th>
        <th style="background:#1c2128;color:#8b949e;padding:8px 12px;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.5px;border-bottom:1px solid #30363d">Best Val AUC</th>
        <th style="background:#1c2128;color:#8b949e;padding:8px 12px;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.5px;border-bottom:1px solid #30363d">Last 5 epochs</th>
        <th style="background:#1c2128;color:#8b949e;padding:8px 12px;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.5px;border-bottom:1px solid #30363d">Trend</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>

  {p1_html}
  {v3_html}
</div>
<!-- LIVE_RESULTS_END -->
"""
    return block


def inject_into_html(live_block):
    """Replace or append the live results block in the HTML file."""
    with open(HTML_PATH) as f:
        html = f.read()

    # Remove old live block if present
    html = re.sub(
        r"<!-- LIVE_RESULTS_START -->.*?<!-- LIVE_RESULTS_END -->",
        "", html, flags=re.DOTALL
    )

    # Inject before </body>
    html = html.replace("</body>", live_block + "\n</body>")

    with open(HTML_PATH, "w") as f:
        f.write(html)


if __name__ == "__main__":
    print("Building live results block …")
    block = build_results_block()
    inject_into_html(block)
    print(f"Updated → {HTML_PATH}")
