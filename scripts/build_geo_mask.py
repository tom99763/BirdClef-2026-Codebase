"""Build geographic species mask for BirdCLEF 2026.

Two signal sources:
  1. train_soundscapes_labels.csv — confirmed Pantanal presence (strongest signal)
  2. train.csv lat/lon — South American recordings as broader proxy

Output: outputs/geo_mask.csv with columns:
  primary_label, in_soundscape, soundscape_count,
  sa_count, total_count, sa_fraction, geo_score

geo_score = max(in_soundscape, sa_fraction)
  - 1.0 if confirmed in Pantanal soundscapes
  - sa_fraction otherwise (fraction of training recordings from South America)

Usage:
    python scripts/build_geo_mask.py
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = "birdclef-2026"
OUT_PATH = "outputs/geo_mask.csv"

# South America bounding box
SA_LAT = (-56, 15)
SA_LON = (-82, -34)


def main():
    os.makedirs("outputs", exist_ok=True)

    tax      = pd.read_csv(f"{DATA_DIR}/taxonomy.csv")
    train    = pd.read_csv(f"{DATA_DIR}/train.csv")
    ss_lbl   = pd.read_csv(f"{DATA_DIR}/train_soundscapes_labels.csv")

    # ── 1. Soundscape presence (confirmed Pantanal) ───────────────────────────
    species_in_ss = set()
    for row in ss_lbl["primary_label"]:
        for s in str(row).split(";"):
            s = s.strip()
            if s and s != "nan":
                species_in_ss.add(s)

    # ── 2. South America lat/lon filter ───────────────────────────────────────
    sa_mask    = (train["latitude"].between(*SA_LAT)) & \
                 (train["longitude"].between(*SA_LON))
    sa_counts  = train[sa_mask].groupby("primary_label").size()
    tot_counts = train.groupby("primary_label").size()

    # ── 3. Build per-species row ───────────────────────────────────────────────
    all_species = tax["primary_label"].astype(str).tolist()
    rows = []
    for sp in all_species:
        sa_c    = int(sa_counts.get(sp, 0))
        tot_c   = int(tot_counts.get(sp, 0))
        ss_c    = 0
        for row in ss_lbl["primary_label"]:
            if sp in str(row).split(";"):
                ss_c += 1
        in_ss   = sp in species_in_ss
        sa_frac = round(sa_c / tot_c, 4) if tot_c > 0 else 0.0
        geo     = round(max(float(in_ss), sa_frac), 4)
        rows.append({
            "primary_label":    sp,
            "in_soundscape":    in_ss,
            "soundscape_count": ss_c,
            "sa_count":         sa_c,
            "total_count":      tot_c,
            "sa_fraction":      sa_frac,
            "geo_score":        geo,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)

    print(f"Geo mask saved → {OUT_PATH}")
    print(f"  Species total          : {len(df)}")
    print(f"  In Pantanal soundscape : {df.in_soundscape.sum()}")
    print(f"  Has SA recordings      : {(df.sa_count>0).sum()}")
    print(f"  Zero Pantanal signal   : {((~df.in_soundscape) & (df.sa_count==0)).sum()}")
    print(f"  geo_score > 0.9        : {(df.geo_score>0.9).sum()}")
    print(f"  geo_score > 0.5        : {(df.geo_score>0.5).sum()}")
    print(f"\n  Mask effect (mean geo_score) : {df.geo_score.mean():.4f}")

    # Print species with zero signal
    zero = df[(~df.in_soundscape) & (df.sa_count == 0)]
    if len(zero) > 0:
        print(f"\n  Species with ZERO Pantanal signal ({len(zero)}):")
        for _, r in zero.iterrows():
            print(f"    {r.primary_label}")


if __name__ == "__main__":
    main()
