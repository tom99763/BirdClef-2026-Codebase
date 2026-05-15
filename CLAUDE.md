# Claude Code 專案指南

## 回覆語言
所有回覆請用**繁體中文**。

## 專案概覽
BirdCLEF 2026 Kaggle 競賽 — 234 species 多標籤音訊分類。核心方法：Noisy Student (NS) self-distillation pipeline。

## 關鍵架構
- **SED Model**: timm backbone + GEM Frequency Pool + Attention SED Head
- **支援的 backbone**: `tf_efficientnet_b0.ns_jft_in1k` (B0), `pvt_v2_b0` (PVT), `regnety_008.pycls_in1k` (RegNetY), `hgnetv2_b0` (HGNetV2)
- **HGNetV2 注意**: `num_features` 回報 1024 但實際 feat_dim=2048，必須用 dummy forward 偵測

## NS 訓練流程（以 B0 為例）

### 前置需求
1. `birdclef-2026/` — 競賽資料（train.csv, train_audio/, train_soundscapes/）
2. `outputs/perch_teacher_aug_all_ss.csv` — Perch teacher predictions
3. `pseudo_labels/noisy_classmate_b0_r15_nc_no_ncw.csv` — R0 初始 pseudo labels
4. `configs/sed_ns_b0_20s_r0.yaml` — R0 config

### 快速啟動
```bash
# 單架構單 GPU，R0-R11
ARCH=b0 BACKBONE=tf_efficientnet_b0.ns_jft_in1k GPU=0 START_ROUND=0 END_ROUND=11 \
  nohup bash scripts/auto_ns_single_gpu.sh > outputs/logs/auto_ns_b0_gpu0.log 2>&1 &
```

### Pipeline 自動流程（每 round）
1. **Train** 5 folds（sequential，early stop patience=4）
2. **Export ONNX**（FP32 + INT8 quantize）
3. **Infer** all soundscapes（5-fold ensemble → all_ss_probs.npz）
4. **Corrector**（residual corrector 校正 SED predictions）
5. **Gen Pseudo**（per-round 策略混合 Perch + SED）
6. → 下一個 round

### Pseudo Label 策略（開發規範，不可更改）
| Round | perch_w | sed_w | percentile | gamma |
|-------|---------|-------|-----------|-------|
| R0    | 0.50    | 0.50  | 92        | 1.00  |
| R1    | 0.30    | 0.70  | 93        | 1.54  |
| R2    | 0.10    | 0.90  | 94        | 1.82  |
| R3+   | 0.05    | 0.95  | 95        | 2.00  |

必須包含 `--nonaves_perch_only` flag。

### 訓練超參數（標準設定）
- epochs: 25, patience: 4
- batch_size: 16
- lr: 1e-3, weight_decay: 1e-4
- focal_gamma: 2.0（BCE 實驗用 0.0）
- mixup_alpha: 0.15
- ema_decay: 0.999
- drop_path_rate: 0.15（B0/PVT）, 0.1（其他）
- clip_duration: 20s, n_mels: 224

## 重要開發規範

### GPU 管理
- 啟動/停止 pipeline 時**必須確認 GPU 乾淨**
- `kill PID` 只殺 parent，必須 `pkill -P PID` 殺 child
- 啟動前檢查：`nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`

### 不要做的事
- 不要跨 fold 比較 val AUC（不同 fold val set 不同）
- 不要用固定 perch_w（必須 per-round 遞減）
- 不要同時跑超過 GPU 記憶體的進程（RegNetY ~7GB, HGNetV2 ~5GB, B0 ~5GB）
- 不要 sweep B0/PVT round（已到 0.948 天花板）

### 提交限制
- Kaggle 90 min timeout — 只用 2 個 SED checkpoint
- file_level_top_k=1 是當前最佳（LB 0.949）

## 檔案結構
```
train_sed_ns.py              — 主訓練腳本
scripts/auto_ns_single_gpu.sh — 自動化 NS pipeline（推薦使用）
scripts/gen_pseudo_ns.py      — Pseudo label 生成
scripts/train_sed_residual_corrector.py — Corrector 訓練
scripts/export_sed_to_onnx.py — ONNX export（含 INT8）
configs/sed_ns_*_r0.yaml      — 各架構 R0 config（R1+ 由 script 自動生成）
```

## Memory 系統
本專案有完整的 memory 記錄在 `.claude/projects/-home-lab-BirdClef-2026-Codebase/memory/`：
- `feedback_*.md` — 行為準則和過去教訓
- `project_*.md` — 專案狀態和決策記錄
- `MEMORY.md` — 索引檔

新對話開始時請參考 memory 了解專案脈絡。
