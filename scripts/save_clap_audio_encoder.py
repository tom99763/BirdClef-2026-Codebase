#!/usr/bin/env python3
"""
Save only the CLAP audio encoder + projection for Kaggle upload.
Text encoder is NOT saved (text_prototypes.npy handles the text side).

Output:
  weights/clap/clap_audio_encoder.pt   (~200 MB)
  weights/clap/processor_config/       (tokenizer / feature extractor configs)

Also benchmarks inference speed on 2880 clips (240 files × 12 windows).
"""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import ClapModel, ClapProcessor

PRETRAINED = "laion/clap-htsat-unfused"
OUT_DIR    = Path("weights/clap")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading full CLAP model...")
clap      = ClapModel.from_pretrained(PRETRAINED)
processor = ClapProcessor.from_pretrained(PRETRAINED)
clap.eval()
for p in clap.parameters():
    p.requires_grad = False

# ── Save audio encoder + projection only ────────────────────────────────────
audio_state = {
    "audio_model":      clap.audio_model.state_dict(),
    "audio_projection": clap.audio_projection.state_dict(),
}
ckpt_path = OUT_DIR / "clap_audio_encoder.pt"
torch.save(audio_state, ckpt_path)
size_mb = ckpt_path.stat().st_size / 1e6
print(f"Saved audio encoder → {ckpt_path}  ({size_mb:.1f} MB)")

# ── Save processor config ────────────────────────────────────────────────────
proc_dir = OUT_DIR / "processor_config"
processor.save_pretrained(str(proc_dir))
print(f"Saved processor config → {proc_dir}")

# ── Benchmark: 2880 clips @ batch=24 on cuda:1 ──────────────────────────────
print("\nBenchmarking inference speed (2880 clips, batch=24)...")
device = "cuda:1"
audio_model      = clap.audio_model.to(device)
audio_projection = clap.audio_projection.to(device)

SR_OUT     = 48000
CLIP_DUR   = 5
N_CLIPS    = 2880
BATCH_SIZE = 24

dummy_clips = [np.random.randn(SR_OUT * CLIP_DUR).astype(np.float32)
               for _ in range(BATCH_SIZE)]

# Warm-up
inputs = processor(audio=dummy_clips, sampling_rate=SR_OUT,
                   return_tensors="pt", padding=True)
feat = inputs["input_features"].to(device)
with torch.no_grad():
    out = audio_model(input_features=feat)
    _   = audio_projection(out.pooler_output)
torch.cuda.synchronize()

# Timed run
t0 = time.time()
n_batches = N_CLIPS // BATCH_SIZE
for _ in range(n_batches):
    inputs = processor(audio=dummy_clips, sampling_rate=SR_OUT,
                       return_tensors="pt", padding=True)
    feat = inputs["input_features"].to(device)
    with torch.no_grad():
        out = audio_model(input_features=feat)
        emb = F.normalize(audio_projection(out.pooler_output), dim=-1)
torch.cuda.synchronize()
elapsed = time.time() - t0

per_clip = elapsed / N_CLIPS
total_est = elapsed  # already ran all 2880

print(f"\n{'='*50}")
print(f"Batch size      : {BATCH_SIZE}")
print(f"Total clips     : {N_CLIPS}")
print(f"Total time      : {elapsed:.1f}s  ({elapsed/60:.1f} min)")
print(f"Per clip        : {per_clip*1000:.1f}ms")
print(f"{'='*50}")
if elapsed < 300:
    print("✓ Well within 90-minute limit — no ONNX conversion needed")
else:
    print("✗ Too slow — ONNX conversion recommended")
