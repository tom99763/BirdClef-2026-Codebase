# Memory Index

## 核心流程
- [project_standard_pipeline.md](project_standard_pipeline.md) — 標準流程 NS→NC v3：Perch→B0/PVT NS→3-arch NC v3 co-evolution
- [project_sed_ns_design.md](project_sed_ns_design.md) — SED NS 完整設計：架構、流程、超參數、LB結果、PVT獨立支線
- [project_nc3_design.md](project_nc3_design.md) — NC v3：ConvNeXt-Femto + FastViT-T8 + RegNetY-008 三架構 co-evolution
- [project_ncv4_design.md](project_ncv4_design.md) — NC v4 設計：per-model self-weighted pseudo labels 保持 diversity
- [project_ns_inherit_design.md](project_ns_inherit_design.md) — NS-Inherit R0：EffNetV2-S (BC25 2nd) + EffB0 (BC25 5th)，BC25+BC26+XCL 擴充訓練集
- [project_ns_inherit_progress.md](project_ns_inherit_progress.md) — NS-Inherit B0 12-round (R0-R11) 準備快照：44,988 clips / 206 species，non-Aves 不再強制 Perch-only
- [project_ns_inherit_stopped_r2.md](project_ns_inherit_stopped_r2.md) — NS-Inherit 停在 R2 (04-14)：每 round val AUC 遞減 -0.007～-0.013，R0 LB 0.933 < 0.944 baseline
- [project_ncv4_progress_snapshot.md](project_ncv4_progress_snapshot.md) — NC v4 暫停快照（2026-04-13 19:19）：cnxtf R2 完整、fvit infer 中斷、regy 未開始
- [project_pseudo_label_sources.md](project_pseudo_label_sources.md) — 所有 NS/NC pseudo label 完整來源鏈與生成流程

## LB 結果與分析
- [project_perclass_vlom_lb.md](project_perclass_vlom_lb.md) — Per-class VLOM LB 0.944 新紀錄：Aves 0.7/0.3, non-Aves 0.3/0.7
- [project_lb_submissions_0411.md](project_lb_submissions_0411.md) — 04-11提交：PVT fold4替換=0.941, RegNetY R1替換=0.941, corrector不在notebook裡
- [project_lb_submissions_0412.md](project_lb_submissions_0412.md) — 04-12提交：p5 corrector=0.944（與p3相同），corrector推理時無額外收益
- [project_lb_submissions_0414.md](project_lb_submissions_0414.md) — 04-14提交：5-way per-class_name VLOM=0.94 勝過 2-way 原版 0.933，R0 fold0 weight=8.0 主導 ensemble
- [project_lb_submissions_0416.md](project_lb_submissions_0416.md) — 04-16 LB 0.947 新高：p8 Perch-style SED 後處理 + 2-model（移除 BranchEns），file_level_top_k=2 關鍵
- [project_vlom_analysis.md](project_vlom_analysis.md) — VLOM完整分析：0.70最佳、CV-LB反轉、logit ratio=1.0、Bayesian ceiling 0.953
- [project_cvlb_aves_split.md](project_cvlb_aves_split.md) — CV-LB不相關可能因為沒分開Aves/non-Aves；分開評估驗證中
- [project_eval_plan.md](project_eval_plan.md) — NC v3完成後：用val+BirdSet衡量所有組合，做CV-LB correlation報表
- [project_sed_prior_experiment.md](project_sed_prior_experiment.md) — SED Prior實驗：SED側加site/hour prior，兩版本已提交待BirdSet驗證
- [project_sed_temperature_plan.md](project_sed_temperature_plan.md) — 待做：SED temperature scaling，解決overconfidence搶Perch權重問題
- [project_nc_temp_synergy.md](project_nc_temp_synergy.md) — NC模型×temperature scaling協同：解鎖NC diversity收益的關鍵

## 關鍵 Insights
- [project_key_findings.md](project_key_findings.md) — Critical BirdCLEF 2026 insights: confirmed LB anchors, calibration formula
- [project_sed_eda_insights.md](project_sed_eda_insights.md) — SED超越Perch的EDA分析：GT悖論、非Aves差距、PVT穩健性
- [project_dg_insight.md](project_dg_insight.md) — Domain Generalization is correct framing; test_soundscapes has no audio
- [feedback_ensemble_diversity.md](feedback_ensemble_diversity.md) — Ensemble diversity > solo AUC；同round模型corr太高
- [feedback_generalization_goal.md](feedback_generalization_goal.md) — 核心目標：generalize到hidden test，不是追求CV分數
- [project_competition_structure.md](project_competition_structure.md) — 競賽30% public LB + 70% private LB，共600 files

## Feedback（行為準則）
- [feedback_language.md](feedback_language.md) — 所有回覆請用繁體中文
- [feedback_nohuman_only.md](feedback_nohuman_only.md) — Only evaluate nohuman models
- [feedback_use_own_models_only.md](feedback_use_own_models_only.md) — Competitor weights 允許使用（2026-03-30 起，含 submission ensemble / KD teacher / R13 預訓練起點）
- [feedback_fold_comparison.md](feedback_fold_comparison.md) — 不同 fold val set 不同，不可跨 fold 直接比較 AUC
- [feedback_birdset_eval.md](feedback_birdset_eval.md) — BirdSet eval 用 SED-only AUC；LB #1-9 VLOM 0.70/0.30，#10-15 VLOM 0.50/0.50
- [feedback_ns_training_insights.md](feedback_ns_training_insights.md) — NS 訓練 insights：無需 `_nc_weight`、pseudo 噪訊不污染 Aves、SED 單機可 0.947、non-Aves NS 迭代是實驗、28 類靠 Perch 救
- [feedback_2model_sed.md](feedback_2model_sed.md) — 只用 2 SED ckpt（Kaggle 90min timeout），替換不新增
- [feedback_perclass_val_report.md](feedback_perclass_val_report.md) — 每 round 結束必須產生 per-class 驗證報告（ss_auc/au_auc/domain_error），用於深度分析
- [project_lb_ceiling_0948.md](project_lb_ceiling_0948.md) — LB 0.948 天花板：B0-R9/R11+PVT-R7 等效，需新架構突破，不再 sweep B0/PVT round
- [project_postproc_sweep_plan.md](project_postproc_sweep_plan.md) — Post-proc sweep：top_k=1→0.949，待 sweep VLOM/rank_aware/temperature/delta_shift
- [feedback_clean_gpu_check.md](feedback_clean_gpu_check.md) — Pipeline 啟停時必須確認 GPU 乾淨，pkill -P 殺 child，檢查殭屍進程
- [feedback_pseudo_label_strategy.md](feedback_pseudo_label_strategy.md) — Pseudo label per-round 策略是開發規範：R0 perch_w=0.50→R1 0.30→R2 0.10→R3+ 0.05

## 使用者
- [user_profile.md](user_profile.md) — 使用者：Tom，任職 Realtek Semiconductor Corp.（瑞昱半導體），台灣新竹
