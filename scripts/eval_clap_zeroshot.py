#!/usr/bin/env python3
"""
Zero-shot CLAP evaluation on train_soundscapes.
No training — pure cosine similarity between audio embedding and text prototypes.

Usage:
  python scripts/eval_clap_zeroshot.py \
      --device cuda:1 \
      --max_samples 500   # use subset for quick test; omit for full eval
"""

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import roc_auc_score
from transformers import ClapModel, ClapProcessor

warnings.filterwarnings("ignore")

PRETRAINED  = "laion/clap-htsat-unfused"
SR_IN       = 32000
SR_OUT      = 48000
CLIP_DUR    = 5
EMBED_DIM   = 512

# Time-slot definitions (same as train_clap.py)
TIME_SLOTS = [
    "dawn chorus (5am-8am)",
    "morning (8am-12pm)",
    "afternoon (12pm-5pm)",
    "night (8pm-5am)",
]

def map_hour_to_slot(hour: int) -> int:
    if 5 <= hour < 8:   return 0
    elif 8 <= hour < 12: return 1
    elif 12 <= hour < 20: return 2
    else:                return 3

def parse_hour_from_filename(fname: str) -> int:
    """Extract hour from BC2026_Train_XXXX_SYY_YYYYMMDD_HHMMSS.ogg"""
    m = re.search(r'_(\d{6})\.ogg$', fname)
    if m:
        return int(m.group(1)[:2])
    return 6  # default: dawn


def build_text_prompt(row: pd.Series, slot: str) -> str:
    class_name  = str(row.get("class_name", ""))
    common_name = str(row.get("common_name", ""))
    sci_name    = str(row.get("scientific_name", ""))

    # Insect sonotypes: replace meaningless code with generic insect sound description
    if "sonotype" in common_name.lower() or "Insect son" in sci_name:
        return (f"insect sound, stridulation or chirping or buzzing, "
                f"{slot}, Pantanal wetland, Brazil")

    # Amphibia: add explicit "frog call" cue
    if class_name == "Amphibia":
        return (f"frog call, {common_name} ({sci_name}), "
                f"advertisement call, {slot}, Pantanal wetland, Brazil")

    # Mammalia: add "animal vocalization"
    if class_name == "Mammalia":
        return (f"animal vocalization, {common_name} ({sci_name}), "
                f"{slot}, Pantanal wetland, Brazil")

    # Default: Aves + Reptilia
    return (f"sound of {common_name} ({sci_name}, {class_name}), "
            f"{slot}, Pantanal wetland, Brazil")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",      default="cuda:1")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Limit number of windows (0=all)")
    parser.add_argument("--proto_cache", default="outputs/clap_embeddings/text_prototypes.npy",
                        help="Path to cached text prototypes (created if missing)")
    parser.add_argument("--taxonomy_csv",  default="birdclef-2026/taxonomy.csv")
    parser.add_argument("--ss_labels_csv", default="birdclef-2026/train_soundscapes_labels.csv")
    parser.add_argument("--ss_dir",        default="birdclef-2026/train_soundscapes")
    parser.add_argument("--save_auc_npy",  default="",
                        help="If set, save per-class AUC (234,) to this path. "
                             "Classes not in eval set get -1.0 (sentinel).")
    args = parser.parse_args()

    device = args.device

    # ── Load CLAP (frozen) ───────────────────────────────────────────────────
    print(f"Loading CLAP from {PRETRAINED} ...")
    clap = ClapModel.from_pretrained(PRETRAINED)
    clap.eval()
    for p in clap.parameters():
        p.requires_grad = False
    audio_model      = clap.audio_model.to(device)
    audio_projection = clap.audio_projection.to(device)
    text_model       = clap.text_model.to(device)
    text_projection  = clap.text_projection.to(device)
    processor        = ClapProcessor.from_pretrained(PRETRAINED)
    print("CLAP loaded — all parameters frozen.")

    @torch.no_grad()
    def extract_audio(wav_np: np.ndarray) -> np.ndarray:
        inputs = processor(audio=wav_np, sampling_rate=SR_OUT, return_tensors="pt")
        feat   = inputs["input_features"].to(device)
        out    = audio_model(input_features=feat)
        emb    = audio_projection(out.pooler_output)
        return F.normalize(emb, dim=-1).squeeze(0).cpu().numpy()

    @torch.no_grad()
    def extract_text(text: str) -> np.ndarray:
        inputs = processor(text=text, return_tensors="pt",
                           padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()
                  if k in ("input_ids", "attention_mask")}
        out = text_model(**inputs)
        emb = text_projection(out.pooler_output)
        return F.normalize(emb, dim=-1).squeeze(0).cpu().numpy()

    # ── Build / load text prototypes ─────────────────────────────────────────
    tax_df = pd.read_csv(args.taxonomy_csv).set_index("primary_label")
    # primary_label is inat_taxon_id (int); convert index to str for consistency
    tax_df.index = tax_df.index.astype(str)
    species_list = sorted(tax_df.index.tolist())
    sp2idx = {sp: i for i, sp in enumerate(species_list)}
    n_cls  = len(species_list)

    proto_path = Path(args.proto_cache)
    if proto_path.exists():
        print(f"Loading cached text prototypes from {proto_path}")
        prototypes = np.load(proto_path)   # (4, C, 512)
    else:
        print("Building text prototypes (4 slots × 234 species) ...")
        proto_path.parent.mkdir(parents=True, exist_ok=True)
        prototypes = np.zeros((len(TIME_SLOTS), n_cls, EMBED_DIM), dtype=np.float32)
        for s_idx, slot in enumerate(TIME_SLOTS):
            print(f"  Slot {s_idx}: {slot}")
            for c_idx, sp in enumerate(species_list):
                row = tax_df.loc[sp]
                prompt = build_text_prompt(row, slot)
                prototypes[s_idx, c_idx] = extract_text(prompt)
        np.save(proto_path, prototypes)
        print(f"Text prototypes saved → {proto_path}  ({prototypes.nbytes/1e6:.1f} MB)")

    proto_tensor = torch.from_numpy(prototypes).to(device)  # (4, C, 512)

    # ── Load soundscape labels ────────────────────────────────────────────────
    ss_labels = pd.read_csv(args.ss_labels_csv)
    ss_dir     = Path(args.ss_dir)

    if args.max_samples > 0:
        ss_labels = ss_labels.sample(n=min(args.max_samples, len(ss_labels)),
                                     random_state=42).reset_index(drop=True)
    print(f"Evaluating {len(ss_labels)} labeled windows ...")

    # ── Inference ─────────────────────────────────────────────────────────────
    all_scores  = []
    all_targets = []
    skipped     = 0

    for i, row in ss_labels.iterrows():
        fname   = row["filename"]
        start_s = _parse_start(row["start"])   # seconds
        labels  = str(row["primary_label"]).split(";")
        labels  = [l.strip() for l in labels if l.strip() in sp2idx]
        if not labels:
            skipped += 1
            continue

        fpath = ss_dir / fname
        if not fpath.exists():
            skipped += 1
            continue

        try:
            wav, sr = torchaudio.load(str(fpath))
            if sr != SR_OUT:
                wav = torchaudio.functional.resample(wav, sr, SR_OUT)
            start_samp = int(start_s * SR_OUT)
            n_samp     = CLIP_DUR * SR_OUT
            clip = wav[0, start_samp:start_samp + n_samp].numpy()
            if len(clip) < n_samp:
                clip = np.pad(clip, (0, n_samp - len(clip)))
        except Exception as e:
            skipped += 1
            continue

        audio_emb = extract_audio(clip)           # (512,)
        hour      = parse_hour_from_filename(fname)
        slot_idx  = map_hour_to_slot(hour)
        proto     = proto_tensor[slot_idx]        # (C, 512)

        # Cosine similarity = dot product (both L2-normalised)
        a = torch.from_numpy(audio_emb).to(device)   # (512,)
        scores = (proto @ a).cpu().numpy()            # (C,)

        # Build multi-hot target
        target = np.zeros(n_cls, dtype=np.float32)
        for l in labels:
            target[sp2idx[l]] = 1.0

        all_scores.append(scores)
        all_targets.append(target)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(ss_labels)}] processed ...")

    if not all_scores:
        print("No valid samples found.")
        return

    scores_arr  = np.stack(all_scores)    # (N, C)
    targets_arr = np.stack(all_targets)   # (N, C)

    # ── Metrics ───────────────────────────────────────────────────────────────
    # Only evaluate classes that appear in the eval set
    active_cls = np.where(targets_arr.sum(0) > 0)[0]
    print(f"\nActive classes in eval set: {len(active_cls)}/{n_cls}")

    try:
        macro_auc = roc_auc_score(
            targets_arr[:, active_cls],
            scores_arr[:, active_cls],
            average="macro"
        )
    except ValueError as e:
        macro_auc = float("nan")
        print(f"AUC error: {e}")

    # Per-class breakdown (top / bottom)
    per_cls_auc = {}
    for c in active_cls:
        try:
            per_cls_auc[species_list[c]] = roc_auc_score(
                targets_arr[:, c], scores_arr[:, c])
        except Exception:
            pass

    sorted_auc = sorted(per_cls_auc.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'='*50}")
    print(f"Zero-shot CLAP  |  macro AUC = {macro_auc:.4f}")
    print(f"Skipped windows: {skipped}")
    print(f"{'='*50}")
    print(f"\nTop-10 species:")
    for sp, auc in sorted_auc[:10]:
        name = tax_df.loc[sp]["common_name"] if sp in tax_df.index else sp
        print(f"  {auc:.4f}  {name}")
    print(f"\nBottom-10 species:")
    for sp, auc in sorted_auc[-10:]:
        name = tax_df.loc[sp]["common_name"] if sp in tax_df.index else sp
        print(f"  {auc:.4f}  {name}")

    # ── Save per-class AUC npy ─────────────────────────────────────────────────
    if args.save_auc_npy:
        # Shape (234,): active classes → measured AUC; inactive → -1.0 (sentinel)
        auc_arr = np.full(n_cls, -1.0, dtype=np.float32)
        for sp, auc in per_cls_auc.items():
            if sp in sp2idx:
                auc_arr[sp2idx[sp]] = auc
        out_path = Path(args.save_auc_npy)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, auc_arr)
        print(f"\nPer-class AUC saved → {out_path}  "
              f"(active={len(per_cls_auc)}, inactive=-1.0, shape={auc_arr.shape})")


def _parse_start(time_str: str) -> float:
    """Convert HH:MM:SS or seconds string to float seconds."""
    try:
        return float(time_str)
    except (ValueError, TypeError):
        pass
    parts = str(time_str).split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return 0.0


if __name__ == "__main__":
    main()
