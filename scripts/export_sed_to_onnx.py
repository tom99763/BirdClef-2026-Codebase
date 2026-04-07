"""Export SED PyTorch models to ONNX (FP32) + INT8 quantized.

Wraps MelTransform + SEDModel into a single ONNX graph:
  Input : waveform  (B, 160000)  float32
  Output: probs     (B, 234)     float32

Usage:
    python scripts/export_sed_to_onnx.py \
        --pt  "birdclef-2026/notebook resource/current_subs/weights/best_sed_b0_v5.pt" \
        --out "birdclef-2026/notebook resource/current_subs/weights/best_sed_b0_v5_int8.onnx"

    # Export both models at once:
    python scripts/export_sed_to_onnx.py --all
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from dataclasses import dataclass

import timm

# ── Model definitions (copy from notebook) ────────────────────────────────────

@dataclass
class SEDConfig:
    sr: int = 32_000
    n_mels: int = 224
    n_fft: int = 2048
    hop_length: int = 512
    fmin: int = 0
    fmax: int = 16_000
    top_db: float = 80.0
    power: float = 2.0
    norm: str = 'slaney'
    mel_scale: str = 'htk'
    backbone: str = 'tf_efficientnet_b0.ns_jft_in1k'
    num_classes: int = 234
    in_channels: int = 3
    dropout: float = 0.1
    drop_path_rate: float = 0.0
    gem_p_init: float = 3.0


class GEMFreqPool(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p_init))
        self.eps = eps

    def forward(self, x):
        p = self.p.clamp(min=1.0)
        return x.clamp(min=self.eps).pow(p).mean(dim=2).pow(1.0 / p)


class AttentionSEDHead(nn.Module):
    def __init__(self, feat_dim, num_classes, dropout=0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.att_conv = nn.Conv1d(feat_dim, num_classes, kernel_size=1)
        self.cls_conv = nn.Conv1d(feat_dim, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.fc(x.permute(0, 2, 1)).permute(0, 2, 1)
        att = F.softmax(torch.tanh(self.att_conv(x)), dim=-1)
        cls = self.cls_conv(x)
        return torch.sigmoid((att * cls).sum(-1))


class SEDModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.backbone, pretrained=False, in_chans=cfg.in_channels,
            features_only=False, global_pool='', num_classes=0,
            drop_path_rate=cfg.drop_path_rate,
        )
        self.gem_pool = GEMFreqPool(p_init=cfg.gem_p_init)
        self.head = AttentionSEDHead(self.backbone.num_features, cfg.num_classes, cfg.dropout)

    def forward(self, x):
        return self.head(self.gem_pool(self.backbone(x)))


class MelTransform(nn.Module):
    """Waveform → 3-channel mel spectrogram (no peak-norm, per TRICK3)."""
    def __init__(self, cfg, peak_norm=False):
        super().__init__()
        self.peak_norm = peak_norm
        self.mel = T.MelSpectrogram(
            sample_rate=cfg.sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
            n_mels=cfg.n_mels, f_min=cfg.fmin, f_max=cfg.fmax,
            power=cfg.power, norm=cfg.norm, mel_scale=cfg.mel_scale,
        )
        self.db = T.AmplitudeToDB(stype='power', top_db=cfg.top_db)

    def forward(self, waveforms):  # (B, N) → (B, 3, n_mels, T)
        waveforms = torch.nan_to_num(waveforms.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if self.peak_norm:
            peak = waveforms.abs().amax(dim=1, keepdim=True).clamp(min=1e-7)
            waveforms = waveforms / peak
        mel = torch.nan_to_num(self.db(self.mel(waveforms)), nan=-80.0)
        B = mel.shape[0]
        flat = mel.reshape(B, -1)
        mn = flat.min(1, keepdim=True)[0].unsqueeze(-1)
        mx = flat.max(1, keepdim=True)[0].unsqueeze(-1)
        mel = torch.nan_to_num((mel - mn) / (mx - mn + 1e-7), nan=0.0)
        return mel.unsqueeze(1).expand(-1, 3, -1, -1)  # expand avoids data copy


# ── Export helpers ─────────────────────────────────────────────────────────────

CLIP_SAMPLES = 32_000 * 5  # 160,000
# Mel output shape for 5s clips: (B, 3, 224, T) where T = ceil(160000/512)+1 = 313
MEL_T = 313


def load_sed(pt_path: str, backbone: str = None) -> tuple:
    """Returns (SEDModel, MelTransform) both eval().
    Note: MelTransform is NOT exported to ONNX (torch.stft can't export).
    Only SEDModel (mel → probs) is ONNX-exported.
    """
    cfg   = SEDConfig()
    if backbone:
        cfg.backbone = backbone
    model = SEDModel(cfg)
    ckpt  = torch.load(pt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    if any('freq_pool' in k for k in state):
        state = {k.replace('freq_pool', 'gem_pool'): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def export_fp32(pt_path: str, onnx_path: str, backbone: str = None) -> None:
    """Export SEDModel (input: mel (B,3,224,T), output: probs (B,234)) to ONNX."""
    if os.path.isfile(onnx_path):
        print(f'  FP32 ONNX already exists: {onnx_path}')
        return

    print(f'  Loading: {pt_path}')
    model = load_sed(pt_path, backbone=backbone)

    # Dummy mel input: (1, 3, 224, 313)
    dummy = torch.zeros(1, 3, 224, MEL_T)
    with torch.no_grad():
        out = model(dummy)
    print(f'  Dry-run output shape: {out.shape}  (expect [1, 234])')

    os.makedirs(os.path.dirname(onnx_path) or '.', exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=['mel'],
            output_names=['probs'],
            dynamic_axes={'mel': {0: 'batch', 3: 'time'}, 'probs': {0: 'batch'}},
            opset_version=17,
            do_constant_folding=True,
        )
    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f'  Exported FP32 ONNX: {onnx_path}  ({size_mb:.1f} MB)')


def export_int8(onnx_path: str, int8_path: str) -> None:
    if os.path.isfile(int8_path):
        print(f'  INT8 ONNX already exists: {int8_path}')
        return

    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(onnx_path, int8_path, weight_type=QuantType.QInt8)
    size_mb = os.path.getsize(int8_path) / 1e6
    print(f'  Quantized INT8:  {int8_path}  ({size_mb:.1f} MB)')


def verify_onnx(pt_path: str, onnx_path: str) -> None:
    """Compare PyTorch vs ONNX output on random mel input."""
    import onnxruntime as ort
    model = load_sed(pt_path)
    sess  = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    dummy_mel = np.random.rand(4, 3, 224, MEL_T).astype(np.float32)
    with torch.no_grad():
        pt_out  = model(torch.from_numpy(dummy_mel)).numpy()
    ort_out = sess.run(['probs'], {'mel': dummy_mel})[0]

    max_diff = np.abs(pt_out - ort_out).max()
    print(f'  Max abs diff PyTorch vs ONNX: {max_diff:.2e}  (expect < 1e-4)')


# ── Main ──────────────────────────────────────────────────────────────────────

WEIGHTS_DIR = "birdclef-2026/notebook resource/current_subs/weights"

ALL_MODELS = [
    {'name': 'ours',       'pt': f'{WEIGHTS_DIR}/best_sed_b0_v5.pt'},
    {'name': 'competitor', 'pt': f'{WEIGHTS_DIR}/competitor_sed_fold0.pt'},
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--all',      action='store_true', help='Export all models')
    p.add_argument('--pt',       type=str, help='Path to .pt checkpoint')
    p.add_argument('--out',      type=str, help='Output path for ONNX')
    p.add_argument('--fp32',     action='store_true', help='Keep FP32 ONNX (skip INT8)')
    p.add_argument('--verify',   action='store_true', help='Verify ONNX vs PyTorch output')
    p.add_argument('--backbone', type=str, default=None, help='Override backbone (e.g. pvt_v2_b0)')
    args = p.parse_args()

    models_to_export = []
    if args.all:
        models_to_export = ALL_MODELS
    elif args.pt:
        out = args.out or args.pt.replace('.pt', '_int8.onnx')
        models_to_export = [{'name': os.path.basename(args.pt), 'pt': args.pt, 'out': out}]
    else:
        p.print_help()
        sys.exit(1)

    for m in models_to_export:
        pt_path   = m['pt']
        fp32_path = m.get('out', pt_path.replace('.pt', '.onnx'))
        int8_path = fp32_path.replace('.onnx', '_int8.onnx')
        if fp32_path.endswith('_int8.onnx'):
            fp32_path = fp32_path.replace('_int8.onnx', '.onnx')

        print(f'\n[{m["name"]}]')
        if not os.path.isfile(pt_path):
            print(f'  SKIP: {pt_path} not found'); continue

        export_fp32(pt_path, fp32_path, backbone=args.backbone)

        if not args.fp32:
            export_int8(fp32_path, int8_path)

        if args.verify:
            check_path = int8_path if (not args.fp32 and os.path.isfile(int8_path)) else fp32_path
            verify_onnx(pt_path, check_path)

    print('\nDone.')


if __name__ == '__main__':
    main()
