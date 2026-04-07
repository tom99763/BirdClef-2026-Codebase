"""Extract CLAP embeddings for labeled soundscapes (train_soundscapes_labels.csv).

These soundscapes are NOT in the pseudo labels CSV so they were skipped during
the main embedding extraction. Required for CLAP Stage 2 validation.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/extract_labeled_ss_embeddings.py \
        --config configs/clap_v2_supcon.yaml \
        --device cuda:0
"""

import argparse, csv, sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from train_clap import ClapExtractor   # reuse existing extractor (correct API)


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  required=True)
    parser.add_argument("--device",  default="cuda:0")
    args = parser.parse_args()

    cfg    = load_cfg(args.config)
    device = torch.device(args.device)

    ss_dir       = Path(cfg["data"]["soundscape_dir"])
    labels_csv   = cfg["data"].get("soundscape_labels_csv",
                                   "birdclef-2026/train_soundscapes_labels.csv")
    out_dir      = Path(cfg["data"]["wild_emb_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_dur = 5
    sr_out   = 48000

    # Load CLAP extractor (reuse train_clap.py's ClapExtractor — correct API)
    print("Loading CLAP...")
    clap_name = cfg.get("clap_pretrained", "laion/clap-htsat-unfused")
    extractor = ClapExtractor(clap_name, device=device)

    # Read labeled soundscapes
    rows = list(csv.DictReader(open(labels_csv)))
    print(f"Labeled soundscape rows: {len(rows)}")

    skipped = extracted = errors = 0
    for row in rows:
        fname     = str(row["filename"])
        ss_id     = fname.replace(".ogg", "").replace(".wav", "")
        start_str = str(row["start"])
        parts     = start_str.split(":")
        if len(parts) == 3:
            offset_s = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(float(parts[2]))
        else:
            try:
                offset_s = int(float(start_str))
            except Exception:
                continue
        end_offset = offset_s + clip_dur
        row_id     = f"{ss_id}_{end_offset}"

        out_path = out_dir / f"{row_id}.npy"
        if out_path.exists():
            skipped += 1
            continue

        ss_file = ss_dir / f"{ss_id}.ogg"
        if not ss_file.exists():
            ss_file = ss_dir / f"{ss_id}.wav"
        if not ss_file.exists():
            print(f"  WARNING: audio not found: {ss_id}")
            errors += 1
            continue

        try:
            wav, sr = torchaudio.load(str(ss_file))
            if sr != sr_out:
                wav = torchaudio.functional.resample(wav, sr, sr_out)
            start_s = max(0, offset_s) * sr_out
            clip    = wav[:, start_s: start_s + clip_dur * sr_out]
            if clip.shape[1] < clip_dur * sr_out:
                clip = torch.nn.functional.pad(clip, (0, clip_dur * sr_out - clip.shape[1]))
            clip_np = clip.mean(0).numpy()

            emb = extractor.extract_audio(clip_np)   # (512,) L2-normed
            np.save(str(out_path), emb.astype(np.float32))
            extracted += 1
            if extracted % 50 == 0:
                print(f"  Extracted {extracted} so far...")
        except Exception as e:
            print(f"  ERROR {row_id}: {e}")
            errors += 1

    print(f"Done. Extracted={extracted}  Skipped(cached)={skipped}")


if __name__ == "__main__":
    main()
