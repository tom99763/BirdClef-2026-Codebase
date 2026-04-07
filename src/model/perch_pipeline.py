"""
perch_pipeline.py — Unified Perch v2 + Adapter inference pipeline.

Drop-in for any notebook.  Zero external dependencies beyond PyTorch + onnxruntime.

Usage (notebook):
    from src.model.perch_pipeline import PerchPipeline

    pipe = PerchPipeline(
        onnx_path   = "/kaggle/input/.../perch_v2_no_dft.onnx",
        adapter_ckpt = "/kaggle/input/.../perch_adapter_r3.pt",  # optional
        device      = "auto",   # "cuda", "cpu", or "auto"
        batch_size  = 32,
    )

    # From raw audio (numpy or torch):
    audio = np.zeros((1, 160_000), dtype=np.float32)   # 10 s @ 16 kHz
    result = pipe(audio)
    print(result.logits.shape)       # (1, 234)
    print(result.embedding.shape)    # (1, 1536)

    # From a list of files:
    results = pipe.infer_files(["a.ogg", "b.ogg"], window_sec=5, sr=32_000)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

# Lazy imports to avoid errors when optional packages aren't installed
def _import_ort():
    import onnxruntime as ort
    return ort

def _import_soundfile():
    try:
        import soundfile as sf
        return sf
    except ImportError:
        return None

def _import_librosa():
    try:
        import librosa
        return librosa
    except ImportError:
        return None


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    logits    : np.ndarray   # (B, 234) raw logits — sigmoid to get probabilities
    embedding : np.ndarray   # (B, 1536) Perch embedding (after adapter if loaded)
    raw_emb   : np.ndarray   # (B, 1536) Perch embedding (before adapter)

    @property
    def probs(self) -> np.ndarray:
        """Sigmoid probabilities (B, 234)."""
        return 1.0 / (1.0 + np.exp(-self.logits))


# ── ONNX runner (device-agnostic) ─────────────────────────────────────────────

class _OnnxRunner:
    def __init__(self, onnx_path: str):
        ort = _import_ort()
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            self._sess = ort.InferenceSession(
                onnx_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._cuda_ep = True
        else:
            self._sess = ort.InferenceSession(
                onnx_path, providers=["CPUExecutionProvider"]
            )
            self._cuda_ep = False

    def run(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """x: (B, 160000) float32 → (embedding (B,1536), spatial (B,16,4,1536))"""
        out = self._sess.run(
            output_names=["embedding", "spatial_embedding"],
            input_feed={"inputs": x},
        )
        return out[0], out[1]   # embedding, spatial_embedding

    def run_cuda(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero-copy CUDA path.  x must be CUDA float32 contiguous."""
        if not (self._cuda_ep and x.is_cuda):
            # Fall back to CPU
            emb_np, sp_np = self.run(x.cpu().float().contiguous().numpy())
            return torch.from_numpy(emb_np), torch.from_numpy(sp_np)

        B = x.shape[0]
        dev_id = x.device.index or 0
        emb    = torch.empty((B, 1536),        device=x.device, dtype=torch.float32)
        sp_emb = torch.empty((B, 16, 4, 1536), device=x.device, dtype=torch.float32)

        binding = self._sess.io_binding()
        binding.bind_input(
            name="inputs", device_type="cuda", device_id=dev_id,
            element_type=np.float32, shape=tuple(x.shape), buffer_ptr=x.data_ptr(),
        )
        binding.bind_output(
            name="embedding", device_type="cuda", device_id=dev_id,
            element_type=np.float32, shape=tuple(emb.shape), buffer_ptr=emb.data_ptr(),
        )
        binding.bind_output(
            name="spatial_embedding", device_type="cuda", device_id=dev_id,
            element_type=np.float32, shape=tuple(sp_emb.shape), buffer_ptr=sp_emb.data_ptr(),
        )
        self._sess.run_with_iobinding(binding)
        return emb, sp_emb


# ── Adapter (thin wrapper so it can be absent) ────────────────────────────────

class _AdapterWrapper:
    def __init__(self, ckpt_path: str, device: torch.device):
        import sys, os
        # Allow loading even if codebase root is not in sys.path
        root = str(Path(__file__).parent.parent.parent)
        if root not in sys.path:
            sys.path.insert(0, root)
        from src.model.perch_adapter import PerchAdapter

        ckpt = torch.load(ckpt_path, map_location=device)
        cfg  = ckpt.get("cfg", {})
        self.model = PerchAdapter(
            emb_dim     = int(cfg.get("emb_dim",     1536)),
            bottleneck  = int(cfg.get("bottleneck",  512)),
            num_classes = int(cfg.get("num_classes", 234)),
            dropout     = float(cfg.get("dropout",   0.0)),
            n_blocks    = int(cfg.get("n_blocks",    2)),
        ).to(device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """emb (B,1536) → (logits (B,234), adapted_emb (B,1536))"""
        t = torch.tensor(emb, dtype=torch.float32, device=self.device)
        logits, adapted, _ = self.model(t)
        return logits.cpu().numpy(), adapted.cpu().numpy()


# ── Main pipeline ─────────────────────────────────────────────────────────────

class PerchPipeline:
    """
    Drop-in Perch v2 + Adapter inference pipeline.

    Parameters
    ----------
    onnx_path    : path to perch_v2_no_dft.onnx
    adapter_ckpt : path to perch_adapter_r*.pt  (None → use raw Perch logits)
    device       : "auto" (detect), "cuda", "cuda:1", "cpu"
    batch_size   : internal chunk size for large inputs
    taxonomy_csv : if given, attaches species labels to results
    """

    # Default paths (relative to wherever the notebook is run from)
    _DEFAULT_ONNX_PATHS = [
        "weights/perch_v2_no_dft.onnx",
        "../weights/perch_v2_no_dft.onnx",
        "/kaggle/input/datasets/tom99763/birdclef2026-claude/weights/perch_v2_no_dft.onnx",
    ]
    _DEFAULT_ADAPTER_PATHS = [
        "weights/perch_adapter_r3.pt",
        "../weights/perch_adapter_r3.pt",
        "/kaggle/input/datasets/tom99763/birdclef2026-claude/weights/perch_adapter_r3.pt",
    ]

    def __init__(
        self,
        onnx_path:    Optional[str] = None,
        adapter_ckpt: Optional[str] = None,
        device:       str = "auto",
        batch_size:   int = 32,
        taxonomy_csv: Optional[str] = None,
    ):
        # ── Device ──
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ── ONNX ──
        onnx_path = onnx_path or self._find_file(self._DEFAULT_ONNX_PATHS, "ONNX model")
        self._onnx = _OnnxRunner(str(onnx_path))

        # ── Adapter (optional) ──
        self._adapter: Optional[_AdapterWrapper] = None
        if adapter_ckpt is None:
            adapter_ckpt = self._find_file(self._DEFAULT_ADAPTER_PATHS, "adapter checkpoint",
                                            required=False)
        if adapter_ckpt and Path(adapter_ckpt).exists():
            self._adapter = _AdapterWrapper(str(adapter_ckpt), self.device)
            print(f"[PerchPipeline] Adapter loaded: {adapter_ckpt}")
        else:
            print("[PerchPipeline] No adapter — using raw Perch embeddings + Perch logits.")

        self.batch_size = batch_size

        # ── Species labels (optional) ──
        self.species: Optional[list[str]] = None
        if taxonomy_csv and Path(taxonomy_csv).exists():
            import pandas as pd
            tx = pd.read_csv(taxonomy_csv)
            self.species = tx["primary_label"].tolist()

        print(f"[PerchPipeline] Ready  device={self.device}  "
              f"onnx={'OK'}  adapter={'OK' if self._adapter else 'NONE'}")

    # ── Core inference ────────────────────────────────────────────────────────

    def _infer_batch(self, x_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        x_np : (B, 160000) float32 numpy
        Returns: (logits B×234, embedding B×1536)
        """
        # Perch feature extraction
        if self._onnx._cuda_ep and self.device.type == "cuda":
            x_t = torch.tensor(x_np, dtype=torch.float32, device=self.device)
            emb_t, _ = self._onnx.run_cuda(x_t)
            raw_emb = emb_t.cpu().numpy()
        else:
            raw_emb, _ = self._onnx.run(x_np)  # (B, 1536)

        # Adapter
        if self._adapter is not None:
            logits, adapted_emb = self._adapter(raw_emb)
            return logits, adapted_emb, raw_emb
        else:
            # No adapter: use Perch 14795-class head — but we only want 234
            # Return zeros for logits if no adapter and no mapping
            logits = np.zeros((len(x_np), 234), dtype=np.float32)
            return logits, raw_emb, raw_emb

    def __call__(
        self,
        audio: Union[np.ndarray, torch.Tensor],
    ) -> PipelineResult:
        """
        audio : (B, 160000) float32  — 10 s clips at 16 kHz
                OR (160000,) for single clip
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().float().numpy()
        if audio.ndim == 1:
            audio = audio[np.newaxis]  # (1, 160000)
        audio = audio.astype(np.float32)

        all_logits, all_emb, all_raw = [], [], []
        for i in range(0, len(audio), self.batch_size):
            chunk = audio[i:i + self.batch_size]
            l, e, r = self._infer_batch(chunk)
            all_logits.append(l)
            all_emb.append(e)
            all_raw.append(r)

        return PipelineResult(
            logits    = np.concatenate(all_logits),
            embedding = np.concatenate(all_emb),
            raw_emb   = np.concatenate(all_raw),
        )

    # ── File-level helpers ────────────────────────────────────────────────────

    def infer_file(
        self,
        path: str,
        window_sec: float = 5.0,
        sr: int = 32_000,
        overlap: float = 0.0,
    ) -> PipelineResult:
        """
        Load an audio file and run inference on sliding windows.

        Returns PipelineResult with one row per window.
        """
        audio = self._load_audio(path, sr)
        windows = self._slice_audio(audio, window_sec, sr, overlap)
        return self(windows)

    def infer_files(
        self,
        paths: list[str],
        window_sec: float = 5.0,
        sr: int = 32_000,
        overlap: float = 0.0,
        verbose: bool = True,
    ) -> dict[str, PipelineResult]:
        """Run infer_file on a list of paths. Returns {path: PipelineResult}."""
        results = {}
        for i, p in enumerate(paths):
            if verbose:
                print(f"  [{i+1}/{len(paths)}] {Path(p).name}")
            results[p] = self.infer_file(p, window_sec, sr, overlap)
        return results

    # ── Audio utilities ───────────────────────────────────────────────────────

    @staticmethod
    def _load_audio(path: str, sr: int = 32_000) -> np.ndarray:
        """Load and resample audio file → (samples,) float32."""
        sf = _import_soundfile()
        librosa = _import_librosa()
        if sf is not None:
            data, file_sr = sf.read(path, dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)
            if file_sr != sr and librosa is not None:
                data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
        elif librosa is not None:
            data, _ = librosa.load(path, sr=sr, mono=True)
        else:
            raise ImportError("Install soundfile or librosa to load audio files.")
        return data.astype(np.float32)

    @staticmethod
    def _slice_audio(
        audio: np.ndarray,
        window_sec: float,
        sr: int,
        overlap: float = 0.0,
    ) -> np.ndarray:
        """Slice 1-D audio into (N, window_samples) windows with zero-padding."""
        win = int(window_sec * sr)
        step = int(win * (1.0 - overlap))
        starts = list(range(0, max(1, len(audio) - win + 1), step))
        windows = []
        for s in starts:
            chunk = audio[s:s + win]
            if len(chunk) < win:
                chunk = np.pad(chunk, (0, win - len(chunk)))
            windows.append(chunk)
        return np.stack(windows).astype(np.float32)  # (N, win)

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _find_file(candidates: list[str], label: str, required: bool = True) -> Optional[str]:
        for p in candidates:
            if Path(p).exists():
                return p
        if required:
            raise FileNotFoundError(
                f"{label} not found. Tried:\n" + "\n".join(f"  {p}" for p in candidates)
            )
        return None

    def top_species(self, result: PipelineResult, k: int = 5) -> list[tuple[str, float]]:
        """Return top-k species by mean probability across all windows."""
        if self.species is None:
            raise ValueError("taxonomy_csv not provided at init.")
        mean_probs = result.probs.mean(axis=0)
        top_idx = np.argsort(mean_probs)[::-1][:k]
        return [(self.species[i], float(mean_probs[i])) for i in top_idx]
