"""
Perch v2 ONNX feature extractor — works on any device (CUDA or CPU).

Model: weights/perch_v2_no_dft.onnx
  Input : (batch, 160000) float32  [10 s @ 16 kHz]
  Output: embedding         (batch, 1536)
          spatial_embedding (batch, 16, 4, 1536)

Device handling:
  - CUDA available + CUDAExecutionProvider installed → zero-copy IO binding
  - Otherwise → CPU numpy path (automatic fallback, no code changes needed)
"""
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

ONNX_FILE = 'weights/perch_v2_no_dft.onnx'


def _ort_session(onnx_path: str) -> tuple[ort.InferenceSession, bool]:
    """Create ORT session; return (session, cuda_available)."""
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        sess = ort.InferenceSession(
            onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        return sess, True
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return sess, False


class OnnxFeatureExtractor(nn.Module):
    """
    Perch v2 ONNX wrapper.  Accepts CPU or CUDA tensors / numpy arrays.

    Returns:
        spatial_embedding : (B, 16, 4, 1536) — on same device as input
        embedding         : (B, 1536)         — on same device as input
    """

    def __init__(self, onnx_path: str = ONNX_FILE):
        super().__init__()
        self.session, self._cuda_ep = _ort_session(onnx_path)

    @torch.no_grad()
    def forward(self, x):
        """
        Args:
            x : torch.Tensor (B, 160000) float32, CPU or CUDA
                OR np.ndarray (B, 160000) float32
        Returns:
            spatial_embedding, embedding  (same type/device as input)
        """
        # ── Accept numpy input ──────────────────────────────────────────────
        is_numpy = isinstance(x, np.ndarray)
        if is_numpy:
            x = torch.from_numpy(x)

        if x.dtype != torch.float32:
            x = x.float()
        x = x.contiguous()

        on_cuda = x.is_cuda and self._cuda_ep

        if on_cuda:
            # ── CUDA zero-copy path ─────────────────────────────────────────
            device_id = x.device.index if x.device.index is not None else 0
            B = x.shape[0]
            emb   = torch.empty((B, 1536),        device=x.device, dtype=torch.float32)
            sp_emb = torch.empty((B, 16, 4, 1536), device=x.device, dtype=torch.float32)

            binding = self.session.io_binding()
            binding.bind_input(
                name='inputs', device_type='cuda', device_id=device_id,
                element_type=np.float32, shape=tuple(x.shape),
                buffer_ptr=x.data_ptr(),
            )
            binding.bind_output(
                name='embedding', device_type='cuda', device_id=device_id,
                element_type=np.float32, shape=tuple(emb.shape),
                buffer_ptr=emb.data_ptr(),
            )
            binding.bind_output(
                name='spatial_embedding', device_type='cuda', device_id=device_id,
                element_type=np.float32, shape=tuple(sp_emb.shape),
                buffer_ptr=sp_emb.data_ptr(),
            )
            self.session.run_with_iobinding(binding)

        else:
            # ── CPU numpy path ──────────────────────────────────────────────
            x_np = x.cpu().numpy()
            out = self.session.run(
                output_names=['embedding', 'spatial_embedding'],
                input_feed={'inputs': x_np},
            )
            emb    = torch.from_numpy(out[0])   # (B, 1536)
            sp_emb = torch.from_numpy(out[1])   # (B, 16, 4, 1536)

            # Return on original device if CUDA tensor but no CUDAExecutionProvider
            if x.is_cuda:
                emb    = emb.to(x.device)
                sp_emb = sp_emb.to(x.device)

        if is_numpy:
            return sp_emb.cpu().numpy(), emb.cpu().numpy()

        return sp_emb, emb


if __name__ == '__main__':
    extractor = OnnxFeatureExtractor()
    extractor.eval()

    # CPU test
    x_cpu = torch.randn(4, 160_000)
    sp, e = extractor(x_cpu)
    print('CPU  spatial_embedding:', sp.shape, '  embedding:', e.shape)

    # numpy test
    x_np = np.random.randn(4, 160_000).astype(np.float32)
    sp_np, e_np = extractor(x_np)
    print('NumPy spatial_embedding:', sp_np.shape, '  embedding:', e_np.shape)

    # CUDA test (if available)
    if torch.cuda.is_available():
        x_gpu = x_cpu.cuda()
        sp_gpu, e_gpu = extractor(x_gpu)
        print('CUDA  spatial_embedding:', sp_gpu.shape, '  embedding:', e_gpu.shape)
