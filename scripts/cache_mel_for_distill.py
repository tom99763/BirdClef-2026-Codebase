"""Pre-compute and cache mel spectrograms for embedding distillation.

Reads outputs/embeddings_cache/manifest.csv, loads each audio clip,
computes log-mel spectrogram, and saves as float16 .npy.

Stored at: outputs/mel_cache/<split>/<basename>.npy
  Shape: (n_mels, T) = (224, 313) float16

One-time cost: ~30 min for 107k clips.
After caching, train_embed_distill.py loads mels directly (no librosa needed),
bypassing the DataLoader multiprocessing deadlock issue entirely.
"""

import argparse
import os
import numpy as np
import pandas as pd
import librosa
import torchaudio.transforms as AT
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


SAMPLE_RATE = 32000
CLIP_SECONDS = 5
CLIP_SAMPLES = SAMPLE_RATE * CLIP_SECONDS
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 224
FMIN = 0.0
FMAX = 16000.0

mel_transform = None

def init_mel():
    global mel_transform
    mel_transform = torch.nn.Sequential(
        AT.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0,
        ),
        AT.AmplitudeToDB(top_db=80.0),
    )


def process_row(args):
    npy_path, source_file, clip_idx, split, train_audio_dir, soundscapes_dir, out_dir = args

    # Output path
    stem = os.path.splitext(os.path.basename(npy_path))[0]  # e.g. "xyz_c0"
    out_path = os.path.join(out_dir, split, f"{stem}_mel.npy")
    if os.path.exists(out_path):
        return None  # already cached

    # Load audio
    if split == 'soundscape':
        audio_path = os.path.join(soundscapes_dir, source_file)
    else:
        audio_path = os.path.join(train_audio_dir, source_file)

    try:
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        return f"FAIL {audio_path}: {e}"

    start = int(clip_idx) * CLIP_SAMPLES
    clip = audio[start: start + CLIP_SAMPLES]
    if len(clip) < CLIP_SAMPLES:
        clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))

    # Compute mel
    if mel_transform is None:
        init_mel()
    with torch.no_grad():
        t = torch.from_numpy(clip).float().unsqueeze(0)
        mel = mel_transform(t).squeeze(0).numpy()  # (N_MELS, T)

    # Normalise to [0,1] float16
    mn, mx = mel.min(), mel.max()
    mel = ((mel - mn) / (mx - mn + 1e-7)).astype(np.float16)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, mel)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/embeddings_cache/manifest.csv")
    parser.add_argument("--train_audio_dir", default="birdclef-2026/train_audio")
    parser.add_argument("--soundscapes_dir", default="birdclef-2026/train_soundscapes")
    parser.add_argument("--out_dir", default="outputs/mel_cache")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--splits", default="train,soundscape",
                        help="Splits to cache (comma-separated)")
    args = parser.parse_args()

    splits = set(args.splits.split(","))
    df = pd.read_csv(args.manifest)
    df = df[df['split'].isin(splits)].reset_index(drop=True)
    print(f"Caching {len(df)} clips → {args.out_dir}")

    for split in splits:
        os.makedirs(os.path.join(args.out_dir, split), exist_ok=True)

    tasks = [
        (row['npy_path'], row['source_file'], row['clip_idx'], row['split'],
         args.train_audio_dir, args.soundscapes_dir, args.out_dir)
        for _, row in df.iterrows()
    ]

    errors = 0
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=init_mel) as pool:
        futs = {pool.submit(process_row, t): t for t in tasks}
        for fut in tqdm(as_completed(futs), total=len(tasks), desc="caching mels"):
            result = fut.result()
            if result:
                errors += 1
                if errors <= 10:
                    print(f"  {result}")

    print(f"Done. Errors: {errors}/{len(tasks)}")
    # Write a manifest for the mel cache
    out_manifest = os.path.join(args.out_dir, "manifest.csv")
    rows = []
    for _, row in df.iterrows():
        stem = os.path.splitext(os.path.basename(row['npy_path']))[0]
        mel_path = os.path.join(args.out_dir, row['split'], f"{stem}_mel.npy")
        rows.append({
            'mel_path': mel_path,
            'emb_path': row['npy_path'],
            'split': row['split'],
        })
    pd.DataFrame(rows).to_csv(out_manifest, index=False)
    print(f"Manifest → {out_manifest}")


if __name__ == "__main__":
    main()
