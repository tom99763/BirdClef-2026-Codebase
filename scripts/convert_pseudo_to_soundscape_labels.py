"""
Convert Perch pseudo_labels/round5_pseudo.csv to soundscape labels format
compatible with MelSoundscapeDataset (train_sed.py extra_soundscape_csv).

Input  (round5_pseudo.csv):
  row_id, <234 species soft probs>, primary_label, secondary_labels
  row_id format: BC2026_Train_0009_S09_20250828_000000_45
                  └─ base filename + offset in seconds

Output (pseudo_soundscape_labels.csv):
  filename, start, end, primary_label, secondary_labels
  where primary_label = highest-prob species
        secondary_labels = semicolon-joined species above threshold (excl. primary)

Usage:
  python3 scripts/convert_pseudo_to_soundscape_labels.py \\
      --input pseudo_labels/round5_pseudo.csv \\
      --output outputs/pseudo_soundscape_labels_r5.csv \\
      --threshold 0.2
"""

import argparse
import re
import pandas as pd

# Columns that are NOT species labels
NON_SPECIES = {"row_id", "primary_label", "secondary_labels"}


def seconds_to_hhmmss(sec: int) -> str:
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def convert(input_csv: str, output_csv: str, threshold: float = 0.2,
            clip_seconds: int = 5):
    df = pd.read_csv(input_csv)
    species_cols = [c for c in df.columns if c not in NON_SPECIES]

    rows = []
    skipped = 0
    for _, row in df.iterrows():
        row_id = str(row["row_id"])
        # Parse offset from last underscore-separated token
        parts = row_id.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            skipped += 1
            continue
        base_name = parts[0]
        offset_sec = int(parts[1])
        filename = base_name + ".ogg"
        start = seconds_to_hhmmss(offset_sec)
        end   = seconds_to_hhmmss(offset_sec + clip_seconds)

        # Sort species by soft prob
        probs = {c: float(row[c]) for c in species_cols}
        sorted_sp = sorted(probs, key=probs.__getitem__, reverse=True)

        # Primary = highest prob species
        primary = sorted_sp[0]
        # Secondary = all above threshold except primary
        secondary = [s for s in sorted_sp[1:] if probs[s] >= threshold]

        rows.append({
            "filename":         filename,
            "start":            start,
            "end":              end,
            "primary_label":    primary,
            "secondary_labels": ";".join(secondary) if secondary else "",
            "soft_primary_prob": round(probs[primary], 4),
        })

    out = pd.DataFrame(rows)
    out.to_csv(output_csv, index=False)
    print(f"Converted {len(out)} clips  (skipped {skipped})  → {output_csv}")
    print(f"  threshold={threshold}  avg secondary labels per clip: "
          f"{out['secondary_labels'].apply(lambda x: len(x.split(';')) if x else 0).mean():.1f}")
    print(f"  primary prob range: {out['soft_primary_prob'].min():.4f} – "
          f"{out['soft_primary_prob'].max():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     default="pseudo_labels/round5_pseudo.csv")
    parser.add_argument("--output",    default="outputs/pseudo_soundscape_labels_r5.csv")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Min soft prob to include as secondary label")
    parser.add_argument("--clip_seconds", type=int, default=5)
    args = parser.parse_args()
    convert(args.input, args.output, args.threshold, args.clip_seconds)
