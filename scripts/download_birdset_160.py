"""Download BirdSet XCL + PER + NES data for our 160 overlapping Aves species.
Saves audio as .ogg files with labels.

Usage:
    python scripts/download_birdset_160.py --max_per_species 50
"""
import argparse
import os
import json
import numpy as np
import soundfile as sf
from collections import Counter, defaultdict
from pathlib import Path

def get_our_160():
    """Return set of 160 Aves ebird codes that overlap with BirdSet XCL."""
    import pandas as pd
    tax = pd.read_csv('birdclef-2026/taxonomy.csv')
    aves = set(tax[tax['class_name'] == 'Aves']['primary_label'].values)
    # These 2 are not in XCL
    missing = {'palhor3', 'strher2'}
    return aves - missing

def download_subset(subset_name, our_species, out_dir, max_per_species=50, split='test'):
    """Download a BirdSet subset, filtering to our species."""
    from datasets import load_dataset

    ds = load_dataset('DBD-research-group/BirdSet', subset_name, split=split,
                      trust_remote_code=True, streaming=True)
    names = ds.features['ebird_code'].names

    # Map our species to indices
    our_indices = {}
    for i, n in enumerate(names):
        if n in our_species:
            our_indices[i] = n

    if not our_indices:
        print(f"  {subset_name}: no overlap, skipping")
        return {}

    print(f"  {subset_name}: {len(our_indices)} overlapping species")

    counts = Counter()
    labels = []
    subset_dir = out_dir / subset_name
    subset_dir.mkdir(parents=True, exist_ok=True)

    for ex in ds:
        code_idx = ex['ebird_code']
        if code_idx not in our_indices:
            continue

        code_name = our_indices[code_idx]
        if counts[code_name] >= max_per_species:
            continue

        # Get multilabel
        multilabel_indices = ex.get('ebird_code_multilabel') or [code_idx]
        multilabel_names = []
        for idx in multilabel_indices:
            if idx < len(names):
                n = names[idx]
                if n in our_species:
                    multilabel_names.append(n)
        if not multilabel_names:
            multilabel_names = [code_name]

        # Save audio
        audio = ex['audio']
        if isinstance(audio, dict) and 'array' in audio:
            arr = np.array(audio['array'], dtype=np.float32)
            sr = audio.get('sampling_rate', 32000)
        elif isinstance(audio, dict) and 'bytes' in audio:
            # Save raw bytes
            fname = f"{code_name}_{counts[code_name]:04d}.ogg"
            fpath = subset_dir / fname
            with open(fpath, 'wb') as f:
                f.write(audio['bytes'])
            labels.append({
                'filename': fname,
                'primary_label': code_name,
                'multilabel': multilabel_names,
                'subset': subset_name,
            })
            counts[code_name] += 1
            continue
        else:
            continue

        fname = f"{code_name}_{counts[code_name]:04d}.ogg"
        fpath = subset_dir / fname
        sf.write(str(fpath), arr, sr)

        labels.append({
            'filename': fname,
            'primary_label': code_name,
            'multilabel': multilabel_names,
            'subset': subset_name,
        })
        counts[code_name] += 1

        total = sum(counts.values())
        if total % 100 == 0:
            print(f"    Downloaded {total} clips ({len(counts)} species)")

        # Check if all species have enough
        if all(counts[s] >= max_per_species for s in our_indices.values()):
            break

    total = sum(counts.values())
    print(f"  {subset_name}: {total} clips, {len(counts)} species")
    return {'counts': dict(counts), 'labels': labels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_per_species', type=int, default=30,
                        help='Max clips per species from XCL')
    parser.add_argument('--max_per_species_scape', type=int, default=100,
                        help='Max clips per species from soundscape subsets')
    parser.add_argument('--out', type=str, default='data/birdset')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    our_species = get_our_160()
    print(f"Target: {len(our_species)} Aves species")

    all_labels = []
    all_counts = Counter()

    # 1. Soundscape subsets (PER, NES) — higher priority
    print("\n=== Downloading soundscape subsets ===")
    for subset in ['PER', 'NES']:
        result = download_subset(subset, our_species, out_dir,
                                max_per_species=args.max_per_species_scape)
        if result:
            all_labels.extend(result.get('labels', []))
            all_counts.update(result.get('counts', {}))

    # 2. XCL (focal) — fill remaining species
    print("\n=== Downloading XCL (focal recordings) ===")
    result = download_subset('XCL', our_species, out_dir,
                            max_per_species=args.max_per_species, split='train')
    if result:
        all_labels.extend(result.get('labels', []))
        all_counts.update(result.get('counts', {}))

    # Save labels
    labels_path = out_dir / 'labels.json'
    with open(labels_path, 'w') as f:
        json.dump(all_labels, f, indent=2)

    # Summary
    print(f"\n=== Summary ===")
    print(f"Total clips: {len(all_labels)}")
    print(f"Total species: {len(all_counts)}")
    print(f"Species with >0 clips: {sum(1 for c in all_counts.values() if c > 0)}")
    print(f"Labels saved: {labels_path}")

    # Save species coverage
    coverage = {
        'total_clips': len(all_labels),
        'total_species': len(all_counts),
        'per_species': dict(all_counts.most_common()),
    }
    with open(out_dir / 'coverage.json', 'w') as f:
        json.dump(coverage, f, indent=2)
    print(f"Coverage saved: {out_dir / 'coverage.json'}")


if __name__ == '__main__':
    main()
