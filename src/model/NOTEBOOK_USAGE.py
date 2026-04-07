"""
=============================================================================
Perch v2 + Adapter — Notebook 快速上手
=============================================================================

① 安裝（Kaggle notebook 第一格）
─────────────────────────────────
!pip install -q onnxruntime-gpu soundfile

② Import
─────────
import sys
sys.path.insert(0, "/kaggle/input/datasets/tom99763/birdclef2026-claude/src")
from model.perch_pipeline import PerchPipeline

③ 載入 pipeline
───────────────
pipe = PerchPipeline(
    onnx_path    = "/kaggle/input/datasets/tom99763/birdclef2026-claude/weights/perch_v2_no_dft.onnx",
    adapter_ckpt = "/kaggle/input/datasets/tom99763/birdclef2026-claude/weights/perch_adapter_r3.pt",
    device       = "auto",          # 自動偵測 GPU/CPU
    taxonomy_csv = "/kaggle/input/birdclef-2026/taxonomy.csv",
)
# 輸出:
#   [PerchPipeline] Adapter loaded: .../perch_adapter_r3.pt
#   [PerchPipeline] Ready  device=cuda  onnx=OK  adapter=OK

④ 推理單一 numpy batch
──────────────────────
import numpy as np
audio = np.random.randn(4, 160_000).astype(np.float32)  # 4 clips × 10s
result = pipe(audio)

result.logits      # (4, 234) — raw logits
result.probs       # (4, 234) — sigmoid probabilities
result.embedding   # (4, 1536) — adapted Perch embedding
result.raw_emb     # (4, 1536) — original Perch embedding (before adapter)

⑤ 推理整個 soundscape 檔案
────────────────────────────
result = pipe.infer_file(
    path       = "/kaggle/input/birdclef-2026/test_soundscapes/soundscape_123.ogg",
    window_sec = 5.0,   # 每格 5 秒
    sr         = 32_000,
)
print(result.probs.shape)        # (12, 234) — 12 windows × 60s
print(pipe.top_species(result))  # 前 5 名物種

⑥ 批次推理多個檔案
──────────────────
import glob
files = glob.glob("/kaggle/input/birdclef-2026/test_soundscapes/*.ogg")
all_results = pipe.infer_files(files)  # dict: {path → PipelineResult}

⑦ 整合到提交格式
────────────────
import pandas as pd

rows = []
for path, res in all_results.items():
    fn = Path(path).stem
    for win_i, prob_row in enumerate(res.probs):
        t = (win_i + 1) * 5
        row_id = f"{fn}_{t}"
        rows.append({"row_id": row_id, **dict(zip(pipe.species, prob_row))})

submission = pd.DataFrame(rows)
=============================================================================
"""
