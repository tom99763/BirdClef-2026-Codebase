"""10-second context window inference evaluation.

2024 BirdCLEF 1st place technique: +0.015 LB.
For each 5-second target clip, use a 10-second window (2.5s before + 5s + 2.5s after)
to give the model temporal context. The AttentionSEDHead naturally attends to the full window.

The clipwise_prob is the attention-weighted sum over all time frames — with 10s input
the model sees twice as many frames and can attend to neighboring context.

Usage:
    python scripts/eval_10s_inference.py \
        --checkpoint checkpoints/sed-b0-v6/best_sed.pt \
        --config configs/sed_b0_v6.yaml \
        --run_name sed-b0-v6-10s

    python scripts/eval_10s_inference.py \
        --checkpoint checkpoints/sed-b0-v6/soup_sed.pt \
        --config configs/sed_b0_v6.yaml \
        --run_name sed-b0-v6-soup-10s
"""

import argparse
import os
import sys

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio.transforms as T
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import load_config
from src.data.dataset import build_species_mapping
from src.model.sed_model import SEDModel

HOLDOUT_CSV  = "configs/holdout_val_files.csv"
AUDIO_DIR    = "birdclef-2026/train_audio"
SR           = 32_000
CLIP_5S      = SR * 5
CLIP_10S     = SR * 10     # 10-second context window
HALF         = SR * 5 // 2  # 2.5s padding on each side


class MelTransform(nn.Module):
    def __init__(self, n_mels=224):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=SR, n_fft=2048, hop_length=512,
            n_mels=n_mels, f_min=0, f_max=16000,
            power=2.0, norm="slaney", mel_scale="htk",
        )
        self.db = T.AmplitudeToDB(stype="power", top_db=80.0)

    @torch.no_grad()
    def forward(self, waveforms):
        peak = waveforms.abs().amax(dim=1, keepdim=True).clamp(min=1e-7)
        waveforms = waveforms / peak
        mel = self.db(self.mel(waveforms))
        B = mel.shape[0]; flat = mel.reshape(B, -1)
        mn = flat.min(1, keepdim=True)[0].unsqueeze(-1)
        mx = flat.max(1, keepdim=True)[0].unsqueeze(-1)
        mel = (mel - mn) / (mx - mn + 1e-7)
        return mel.unsqueeze(1).repeat(1, 3, 1, 1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config",     required=True)
    p.add_argument("--run_name",   default=None)
    p.add_argument("--holdout_csv", default=HOLDOUT_CSV)
    p.add_argument("--audio_dir",   default=AUDIO_DIR)
    p.add_argument("--gpu",         default=None)
    p.add_argument("--batch_size",  type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = load_config(args.config)
    _, species_to_idx = build_species_mapping(config.data.sample_submission_csv)
    num_classes = len(species_to_idx)
    run_name    = args.run_name or (config.experiment.name + "-10s")

    # Load model
    print(f"\nLoading: {args.checkpoint}")
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    if any("gem_pool" in k for k in state):
        state = {k.replace("gem_pool", "freq_pool"): v for k, v in state.items()}

    model = SEDModel(
        backbone    = config.model.backbone,
        num_classes = num_classes,
        in_chans    = config.model.get("in_chans", 3),
        pretrained  = False,
        drop_rate   = config.model.get("dropout", 0.1),
        use_gem     = config.model.get("use_gem", True),
        gem_p_init  = config.model.get("gem_p_init", 3.0),
        n_mels      = config.mel.n_mels,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    mel_tf = MelTransform(n_mels=config.mel.n_mels).to(device)
    mel_tf.eval()
    print(f"  epoch={ckpt.get('epoch','?')}  ss_val_auc={ckpt.get('metrics',{}).get('macro_auc','?')}")

    # Load holdout
    holdout       = pd.read_csv(args.holdout_csv)
    files         = holdout["filename"].unique()
    file_to_lbl   = dict(zip(holdout["filename"], holdout["primary_label"].astype(str)))
    print(f"\nHoldout: {len(files)} files  ({holdout['primary_label'].nunique()} species)")

    y = np.zeros((len(files), num_classes), dtype=np.float32)
    for i, fname in enumerate(files):
        sp = file_to_lbl.get(fname, "")
        if sp in species_to_idx:
            y[i, species_to_idx[sp]] = 1.0
    swp = np.where(y.sum(0) > 0)[0]
    print(f"  Species with positives: {len(swp)}/234")

    print("\nRunning 10s-context holdout inference …")
    file_preds = []

    for fname in tqdm(files, desc="holdout-10s", ncols=80):
        try:
            audio, _ = librosa.load(os.path.join(args.audio_dir, fname), sr=SR, mono=True)
            audio = audio.astype(np.float32)
        except Exception as e:
            print(f"  WARN: {fname}: {e}")
            file_preds.append(np.zeros(num_classes, dtype=np.float32))
            continue

        n_clips = max(1, int(np.ceil(len(audio) / CLIP_5S)))
        # Pad audio with zeros on both sides for context
        padded  = np.pad(audio, (HALF, HALF))
        clip_preds = []

        for b_start in range(0, n_clips, args.batch_size):
            batch = []
            for ci in range(b_start, min(b_start + args.batch_size, n_clips)):
                # 10s window centered on 5s clip: [ci*5s-2.5s : ci*5s+7.5s] in padded
                start_padded = ci * CLIP_5S          # offset due to HALF padding
                window = padded[start_padded : start_padded + CLIP_10S]
                if len(window) < CLIP_10S:
                    window = np.pad(window, (0, CLIP_10S - len(window)))
                batch.append(window)
            t = torch.from_numpy(np.stack(batch)).to(device)
            with torch.no_grad():
                mel  = mel_tf(t)
                out  = model(mel)
                prob = (out[0] if isinstance(out, tuple) else out).cpu().numpy()
            clip_preds.append(prob)

        clip_preds = np.concatenate(clip_preds, axis=0)
        file_preds.append(clip_preds.max(axis=0))

    file_preds = np.stack(file_preds).astype(np.float32)

    # Score
    try:
        holdout_auc = float(roc_auc_score(y[:, swp], file_preds[:, swp], average="macro"))
    except Exception as e:
        print(f"Scoring error: {e}"); holdout_auc = None

    print(f"\n{'='*60}")
    print(f"  {run_name}")
    print(f"  Holdout ROC-AUC (10s) : {holdout_auc:.4f}" if holdout_auc else "  Holdout ROC-AUC: ERROR")
    print(f"  Files                  : {len(files)}")
    print(f"  Species w/ pos         : {len(swp)}/234")
    print(f"{'='*60}")

    # Save
    import json
    out_dir  = os.path.join("outputs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    result = {
        "run_name":      run_name,
        "checkpoint":    args.checkpoint,
        "epoch":         ckpt.get("epoch", "?"),
        "ss_val_auc":    ckpt.get("metrics", {}).get("macro_auc"),
        "holdout_auc_10s": round(holdout_auc, 6) if holdout_auc else None,
        "window_s":      10,
        "n_files":       int(len(files)),
        "species_w_pos": int(len(swp)),
    }
    out_path = os.path.join(out_dir, "holdout_eval_10s.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {out_path}")
    return holdout_auc


if __name__ == "__main__":
    main()
