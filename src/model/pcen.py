"""PCEN (Per-Channel Energy Normalisation) audio frontend for BirdCLEF 2026.

PCEN replaces AmplitudeToDB with a learnable adaptive gain normalisation,
improving robustness to varying recording distances and environmental noise.

Reference: "Per-Channel Energy Normalization: Why and How" (Wang et al., 2017)
Used in public BirdCLEF 2026 SED notebook (SED_P technique).

Formula:
    M[t] = (1 - s) * M[t-1] + s * E[t]          (IIR smoothing)
    PCEN[t] = (E[t] / (eps + M[t])^alpha + delta)^r - delta^r

All parameters (alpha, delta, r, s) are per-channel and learnable.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_func
import torchaudio.transforms as T


def _pcen_iir(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    First-order per-channel IIR smoothing via grouped convolution (GPU-parallel).

    M[t] = (1 - s) * M[t-1] + s * x[t],  M[0] = x[0]

    Replaces the O(T) sequential Python loop with a single grouped F.conv1d call,
    giving ~28x speedup (17ms → 0.6ms for B=32, F=224, T=313 on GPU).

    Derivation: M[t] = (h * x)[t] + d^t * (1-s) * x[0]
      where h[j] = s * d^j is an exponentially decaying causal kernel,
      and the correction term accounts for the M[0]=x[0] initial condition.
    F.conv1d computes cross-correlation, so the kernel is stored flipped: w[j] = s*d^{T-1-j}.

    Args:
        x: (B, F, T) power mel spectrogram on GPU
        s: (F,)      per-channel smoothing coefficient (learnable, in (0,1))

    Returns:
        M_seq: (B, F, T) smoothed energy envelope
    """
    B, F, T_len = x.shape
    d = 1.0 - s                                                 # (F,) decay per channel

    # Exponential decay powers: d_pow[f, k] = d[f]^k  for k = 0..T-1
    k_range = torch.arange(T_len, device=x.device, dtype=x.dtype)
    d_pow = d.unsqueeze(1).pow(k_range.unsqueeze(0))            # (F, T)

    # Build conv kernel (flipped for cross-correlation): w[f, 0, j] = s[f] * d[f]^{T-1-j}
    kernel = (s.unsqueeze(1) * d_pow.flip(-1)).unsqueeze(1)     # (F, 1, T)

    # Causal convolution via grouped F.conv1d (one group per (batch, freq) pair)
    x_flat   = x.reshape(1, B * F, T_len)
    x_padded = F_func.pad(x_flat, (T_len - 1, 0))              # causal zero-pad
    kernel_rep = kernel.repeat(B, 1, 1)                         # (B*F, 1, T)
    conv_out = F_func.conv1d(x_padded, kernel_rep,
                              groups=B * F).reshape(B, F, T_len)

    # Initial condition correction: M[0]=x[0] instead of 0
    init_corr = d_pow.unsqueeze(0) * (1.0 - s.view(1, F, 1)) * x[:, :, :1]

    return conv_out + init_corr


class PCENTransform(nn.Module):
    """
    Learnable PCEN normalisation applied to a power mel spectrogram.

    Input : (B, n_mels, T)  — power mel spectrogram (linear scale, power=1.0)
    Output: (B, n_mels, T)  — PCEN-normalised feature map

    Parameters are initialised to the values recommended in the original paper
    and are learnable during training (trainable=True by default).
    """

    def __init__(
        self,
        n_mels: int = 224,
        alpha: float = 0.98,    # AGC strength (how much smoothed energy suppresses)
        delta: float = 2.0,     # bias prevents division by zero, sets dynamic range floor
        r: float = 0.5,         # compression exponent
        s: float = 0.025,       # IIR smoothing coefficient (time constant)
        eps: float = 1e-6,
        trainable: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.n_mels = n_mels

        # Parameterise in log-space to keep values positive
        log_alpha = torch.full((n_mels,), np.log(alpha), dtype=torch.float32)
        log_delta = torch.full((n_mels,), np.log(delta), dtype=torch.float32)
        log_r     = torch.full((n_mels,), np.log(r),     dtype=torch.float32)
        log_s     = torch.full((n_mels,), np.log(s),     dtype=torch.float32)

        if trainable:
            self.log_alpha = nn.Parameter(log_alpha)
            self.log_delta = nn.Parameter(log_delta)
            self.log_r     = nn.Parameter(log_r)
            self.log_s     = nn.Parameter(log_s)
        else:
            self.register_buffer("log_alpha", log_alpha)
            self.register_buffer("log_delta", log_delta)
            self.register_buffer("log_r",     log_r)
            self.register_buffer("log_s",     log_s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_mels, T)
        B, F, T_len = x.shape

        alpha = self.log_alpha.exp().clamp(0.0, 1.0)   # (F,)
        delta = self.log_delta.exp()                    # (F,)
        r     = self.log_r.exp().clamp(0.0, 1.0)       # (F,)
        s     = self.log_s.exp().clamp(1e-4, 1.0)      # (F,)

        # Reshape for broadcasting: (1, F, 1)
        alpha_3d = alpha.view(1, F, 1)
        delta_3d = delta.view(1, F, 1)
        r_3d     = r.view(1, F, 1)
        s_2d     = s.view(1, F)         # used in IIR loop

        # IIR smoothing via grouped-conv (28x faster than Python loop)
        M_seq = _pcen_iir(x, s)                         # (B, F, T)

        # PCEN: (x / (eps + M)^alpha + delta)^r - delta^r
        pcen = (x / (self.eps + M_seq).pow(alpha_3d) + delta_3d).pow(r_3d) \
               - delta_3d.pow(r_3d)
        return pcen


class AudioToMelPCEN(nn.Module):
    """
    Full audio-to-PCEN frontend for SED_P pipeline.

    Pipeline:
        waveform (B, T) → MelSpectrogram (power=1.0) → PCENTransform → (B, 1, F, T)

    Compared to standard log-mel:
    - PCEN provides adaptive gain normalisation (robust to recording distance)
    - Learnable parameters allow task-specific adaptation
    - Better temporal contrast for onset detection (key for SED)

    Output is 3-channel (repeat) for compatibility with ImageNet-pretrained backbone.
    """

    def __init__(
        self,
        sample_rate: int = 32000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 224,
        fmin: float = 0.0,
        fmax: float = 16000.0,
        trainable_pcen: bool = True,
    ):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax,
            power=1.0,          # amplitude (not power) for PCEN
            norm="slaney",
            mel_scale="htk",
        )
        self.pcen = PCENTransform(
            n_mels=n_mels,
            trainable=trainable_pcen,
        )

    @torch.no_grad()
    def forward_inference(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Inference-mode forward (no gradient tracking)."""
        return self._forward_impl(waveforms)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        waveforms: (B, clip_samples) float32 on GPU
        Returns:   (B, 3, n_mels, T) PCEN features (3-channel for backbone compat)
        """
        # Peak-normalise waveform (same as standard pipeline)
        peak = waveforms.abs().amax(dim=1, keepdim=True).clamp(min=1e-7)
        waveforms = waveforms / peak

        mel = self.mel(waveforms)   # (B, n_mels, T) — amplitude spectrogram
        pcen = self.pcen(mel)       # (B, n_mels, T) — PCEN features

        # Min-max normalise to [0, 1] per sample
        B = pcen.shape[0]
        flat = pcen.reshape(B, -1)
        mn = flat.min(1, keepdim=True)[0].unsqueeze(-1)
        mx = flat.max(1, keepdim=True)[0].unsqueeze(-1)
        pcen = (pcen - mn) / (mx - mn + 1e-7)

        # Expand to 3 channels for ImageNet backbone
        return pcen.unsqueeze(1).expand(-1, 3, -1, -1)
