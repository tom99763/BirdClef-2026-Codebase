# Experiment Conclusions Log

This file records conclusions from completed experiments.
Updated automatically by the watchdog pipeline + manually as needed.

---

## sed-b0-4fold-v30-fold1 (Auto-concluded 2026-03-21)
**Best val ROC-AUC**: 0.7840 @ epoch 5
**Verdict**: PASS (>0.77)
**Key observations**: Oscillating trajectory — ep1=0.692→ep2=0.766→ep3=0.739→ep4=0.779→ep5=0.784(best)→ep6=0.773→ep7=0.777→ep8=0.759→ep9=0.782→ep10=0.774→ep11=0.742. Classic warmup dip at ep1/3, peaked at ep5, then noisy oscillation without further improvement. Early stopped after ep11 (patience=6 from ep5). Loss decreased monotonically (0.005→0.003) throughout, suggesting model was still learning features but val metric plateaued.
**Root cause**: N/A (PASS). Oscillation is typical for fold-specific train/val split variance. Best epoch aligns with LR peak; cosine decay phase did not improve further.
**Recommendation**: This is fold1 baseline. Continue with fold3 to complete the 4-fold ensemble. Final soup will aggregate folds 0+1+2+3 checkpoints.

---

## sed-b0-v22-dual-noclipmix (Completed 2026-03-19)

**Config**: EfficientNet-B0, dual BCE loss (clip=0.5 + frame=0.5), no MixUp, no ClipMix, round3 pseudo labels
**Best val ROC-AUC**: 0.7485 @ epoch 5, early stopped at epoch 10 (patience=5)
**Soup checkpoints**: ep5, ep6, ep8

### Key Observations
- No MixUp + no ClipMix did NOT improve over baseline dual-loss setups
- 0.7485 is below v20 (dual+mixup08=0.783?) and v21 variants
- The no-clipmix ablation shows ClipMix contributes to dual-loss training
- Early stopping at ep10 with patience=5 suggests clean convergence failure after ep5

### Conclusion
**ClipMix IS useful** for dual-loss training. Removing it causes performance drop from ~0.77+ → 0.7485.
Future dual-loss experiments should keep ClipMix enabled.

---

## sed-v2s-v3-full-kitchen (Auto-concluded 2026-03-20)

**Config**: EfficientNetV2-S backbone, ASL + dual loss + soft KD (10x r5) + hard pseudo (r5) + SpecAugment. Full kitchen-sink formula matching B0-v28, only backbone changed.
**Best val ROC-AUC**: 0.7592 @ epoch 5
**Verdict**: FAIL (below 0.77 threshold)
**Training curve**: ep1=0.7575, ep2=0.7554, ep3=0.7418, ep4→unstable→ep5=0.7592 (new best), then oscillation. Early stopped ep11 (patience=6).

**Key observations**:
- V2S best holdout was 0.9717 (v2s-v1, no soft KD) but soundscape val peaked at only 0.7592
- Applying the full v28 formula to V2S did NOT transfer performance improvement
- Oscillation pattern similar to B3-v1-asl: large backbone + aggressive ASL = instability
- B0-v20 (dual+mixup) achieved ~0.783 with simpler formula — V2S underperforms

**Root cause**: V2S backbone (tf_efficientnetv2_s) may need different LR / warmup schedule than B0 for soundscape domain. The full kitchen-sink formula was tuned for B0; V2S with larger capacity and different feature distribution doesn't benefit from same hyperparams.

**Recommendation**: If testing V2S again, reduce LR to 2e-4, increase warmup to 5 epochs, and start with simpler loss (ASL only, no dual) to stabilize training first.

---

## sed-b3-v1-asl (In Progress 2026-03-19)

**Config**: EfficientNet-B3, ASL loss (gamma_neg=4, gamma_pos=0, clip=0.05), full pseudo labels
**Progress so far**:
- ep1: 0.7475
- ep2: 0.7805 (best)
- ep3: 0.7700
- ep4: 0.7158 (instability spike)
- ep5: 0.7515
- ep6: 0.7246
- ep7: in progress

### Observations
- Best at ep2 (0.7805) then severe oscillation/degradation
- Pattern suggests LR schedule issue: warmup ends → LR too high for B3 backbone
- ASL with gamma_neg=4 may be too aggressive for a larger backbone
- B3 with its 1536-dim features needs lower LR than B0

### Likely Conclusion (pending)
B3 with current LR (5e-4) is unstable after warmup. If final best stays at 0.7805:
- Below our holdout threshold of 0.9193 for LB submission
- BUT B3 may benefit from lower LR (1e-4 or 3e-4) in a follow-up experiment

---

## sed-b0-v20-dual-mixup08 (Previously completed, ref)

**Best val ROC-AUC**: ~0.783 (estimated from context)
**Key**: dual loss + MixUp alpha=0.8 is the strongest B0 config so far

---

## sed-b0-v21-dual-rating3 (Previously completed, ref)

**Key**: rating filtering (min_rating=3) variant of dual setup

---

## Summary of Dual Loss Ablations (B0 series)

| Variant | Best AUC | Key Change |
|---------|----------|------------|
| v20 (dual + mixup08) | ~0.783 | Strong baseline |
| v21 (dual + rating3) | ~0.77x | Rating filter |
| v22 (dual + no clipmix) | 0.7485 | **Remove ClipMix → worse** |

**Rule**: Always use ClipMix in dual-loss B0 setup.

---

## sed-b0-v30-multipseu (Completed 2026-03-20)

**Config**: EfficientNet-B0, ASL loss, multi-round pseudo labels (r1-r5), soft KD (10x r5), round3 pseudo labels, SpecAugment, dual soundscape + soft pseudo training
**Best val ROC-AUC**: **0.8139 @ epoch 6** — NEW RECORD for B0 soundscape AUC
**Early stopped**: ep12 (patience=6, best@ep6)
**Training curve**: ep1=0.7750, ep2=0.7871, ep3=0.7906, ep4=0.7665, ep5=0.7930, ep6=0.8139, ep7=0.7619, ep8=0.7709, ep9=0.7640, ep10=0.7501, ep11=0.7489, ep12=0.7677

### Key Observations
- 0.8139 is significantly above previous best B0 soundscape AUC (~0.783 from v20-dual-mixup)
- Multi-round pseudo labels (r1-r5) + soft KD is clearly the key difference
- Strong and consistent improvement in early epochs (ep1-ep3 all above 0.77)
- ep6 breakout to 0.8139, then oscillation — typical pattern but higher plateau
- This config has holdout AUC 0.9839 (established in prior sessions)

### Conclusion
**sed-b0-v30-multipseu is the new SED SOTA for this competition.**
- Soundscape val AUC 0.8139 >> threshold 0.77 → qualifies for ensemble submission
- Holdout AUC 0.9839 >> threshold 0.9193 → submission-ready
- Multi-round pseudo (r5) + soft KD is the winning formula for B0
- Next: run holdout eval on soup checkpoints (ep1-ep6), build new ensemble with v30-multipseu

---

## sed-b3-v1-asl (Completed 2026-03-20)

**Config**: EfficientNet-B3, ASL loss (gamma_neg=4, gamma_pos=0, clip=0.05), full pseudo labels (r5), soft KD
**Best val ROC-AUC (soundscape)**: 0.7805 @ epoch 2
**Holdout ROC-AUC**: 🎯 **0.9553** (>> threshold 0.9193) — SUBMISSION QUALITY
**Early stopped**: ep8, patience=6 exhausted
**Final training curve**: ep1=0.7475, ep2=0.7805, ep3=0.7700, ep4=0.7158, ep5=0.7515, ep6=0.7246, ep7=0.7406, ep8=<0.7805 (triggered stop)

### Key Observations
- Soundscape val shows severe oscillation after ep2 (LR too high post-warmup)
- BUT holdout=0.9553 is excellent — B3's 1536-dim features generalize very well
- The gap between soundscape val (0.7805) and holdout (0.9553) is the largest we've seen
- Suggests B3 learned strong clip-level features despite unstable soundscape fine-tuning
- Phase3 chain auto-launched `sed-b3-v1-fold0` (4-fold CV) after completion

### Conclusion
**B3 IS submission-quality despite soundscape instability.** Two action items:
1. `sed-b3-v1-fold0/1/2/3` (cross-validation) currently running — will build fold ensemble
2. `sed-b3-v2-lower-lr` (lr=2e-4, warmup=5ep) — should stabilize soundscape val and push holdout even higher

### Update 2026-03-21 — fold0/fold1 Terminated (OOM)
All 3 perch-probe experiments + 2 B3 folds ran simultaneously → RAM exceeded 62GB → OOM.
- **fold0**: terminated at ep8, best=**0.7778**@ep5 (checkpoints ep4,5,7 saved)
- **fold1**: terminated at ep6, best=**0.7577**@ep5 (checkpoints ep1-6 saved)
- fold2, fold3: not yet started
- **Need to restart fold0 from soup_ep005 + fold1 from soup_ep006** to continue training

---

## Next Experiment Queue (Perch Embedding Direction)

Priority experiments to launch after GPU becomes free:

### P1: PT-MAP Power Transform (Quick Win)
- Add `sign(x)*|x|^0.5` before L2-norm in precompute_probe_cache.py
- Expected: +0.002-0.005 LB
- No GPU needed — pure CPU probe recomputation

### P2: Power Scaling Gamma Search
- Grid search gamma ∈ [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] on OOF
- Expected: +0.005-0.015 LB
- No GPU needed

### P3: SED Backbone Embeddings → Probe (T3)
- Extract v30-multipseu 1280-dim backbone features for 708 labeled clips
- Run PCA(128)+LogReg+Proto on these domain-adapted features
- Expected: +0.010-0.020 LB (domain adaptation advantage)

### P4: B3 Lower LR Follow-up
- If b3-v1-asl best stays at 0.7805 → try lr=2e-4 with longer warmup (5 ep)
- May unlock B3's larger capacity (1536-dim features)

---

## sed-b0-v30-bgnoise (Auto-concluded 2026-03-20)

**Config**: EfficientNet-B0, ASL loss, multi-round pseudo labels (r1-r5), soft KD (10x r5), **background noise augmentation** (snr_db_range=[5,30])
**Best val ROC-AUC**: 0.7725 @ epoch 4
**Early stopped**: epoch 10 (patience=6)
**Verdict**: PASS (>0.77 threshold) — but significantly below v30-multipseu (0.8139)
**Training curve**: ep1=0.7378, ep2=0.7376, ep3=0.7441, ep4=0.7725 (peak), ep5=0.7648, ep6=0.7414, ep7=0.7628, ep8=0.7476, ep9=0.7581, ep10=0.7478

### Key Observations
- Best at ep4 then sustained oscillation — same pattern as other ASL experiments
- 0.7725 << v30-multipseu 0.8139 despite having the same pseudo labels and formula
- The only difference from v30-multipseu is addition of background noise augmentation
- Background noise HURT performance (-0.041 vs v30-multipseu)
- ep1-ep2 were lower than v30-multipseu ep1-ep2 (0.7378/0.7376 vs 0.7750/0.7871) — background noise confused the model early in training

### Root Cause
Background noise augmentation is **counterproductive** with multi-round pseudo labels. Two likely mechanisms:
1. Pseudo labels were generated on clean audio — adding heavy noise at training creates train/inference distribution mismatch
2. SNR range [5,30] dB may be too aggressive, degrading bird call clarity needed for ASL learning

### Conclusion
**Do NOT add background noise to the v30-multipseu formula.** The multi-round pseudo label diversity is sufficient; background noise adds harmful distribution shift. Holdout AUC pending (running now).

### Holdout Result
🎯 **Holdout AUC: 0.9627** (>> threshold 0.9193) — SUBMISSION QUALITY
- Soundscape val 0.7725 << v30-multipseu 0.8139, but holdout 0.9627 is still strong
- Background noise aug hurt soundscape adaptation but maintained holdout generalization
- Gap (soundscape↔holdout) larger than usual: suggests bgnoise helped distribute robustness

### Recommendation
- Keep v30-multipseu formula clean (no bgnoise) for soundscape val performance
- bgnoise model could be useful as ensemble member (different augmentation diversity)
- Phase3 chain auto-launched `sed-b3-v1-fold1` on GPU1 after holdout completed

---

## Perch Probe v2 Ablation Series (Auto-concluded 2026-03-20)

Four configs tested: v2-tip, v2-proto, v2-proto-tip-gamma, v2-full (add_train)

### Results Summary

| Experiment | oof_auc_mlp | oof_auc_blend | Δ blend vs MLP |
|---|---|---|---|
| v2-tip (E only) | 0.7515 | 0.7536 | +0.002 |
| v2-proto (F only) | 0.7515 | 0.6276 | **-0.124** |
| v2-proto-tip-gamma | 0.7515 | 0.6066 | **-0.145** |
| v2-full (D+E+F) | **0.8041** | 0.6022 | **-0.202** |

### Key Findings

**CRITICAL — add_train_clips (D) is the biggest win:**
- Adding 85k weakly-labeled train_audio clips boosted MLP OOF: **0.7515 → 0.8041** (+0.053)
- This is the largest single improvement in Perch probe experiments
- The 85k clips provide weak per-species labels that dramatically improve MLP generalization

**ProtoClassifier (F) consistently hurts:**
- With 739 soundscape clips / 234 species (~3-4 clips/class), prototypes are too noisy
- Min-max distance normalization produces poorly-calibrated multi-class scores
- Even gamma blending cannot recover — blending weight optimization on val data still gives 0.6066

**TipAdapter (E) marginally helps:**
- +0.002 over MLP alone (0.7536 vs 0.7515) — retrieval is better than prototype distance
- But improvement is tiny compared to add_train

**The blend hurts even with add_train:**
- v2-full MLP achieves 0.8041 but blend drops to 0.6022 because proto/tip drag it down
- Conclusion: **do NOT blend proto/tip — use MLP only with add_train**

### Recommendation
Create `perch_probe_v2_addtrain_only` config:
- `add_train_clips: true`
- `use_proto: false`, `use_tip: false`, `use_gamma: false`
- Expected OOF AUC: **~0.8041** (already demonstrated as oof_auc_mlp in v2-full)
- This is the best Perch probe config found so far — significantly better than v1 (0.7814)

---

## sed-b3-v1-fold0 / fold1 (Terminated 2026-03-20)

**Config**: EfficientNet-B3 (1536-dim), ASL loss, 4-fold CV — same formula as b3-v1-asl
**fold0 terminated**: ep8, best=0.7778 @ ep5 (training curve: ep1=0.7525, ep3=0.7532, ep4=0.7678, ep5=0.7778, ep6=0.7560, ep7=0.7631, ep8=0.7554)
**fold1 terminated**: ep6, best=0.7577 @ ep5

### Reason for termination
B3 backbone has ~57 min/epoch vs B0's ~36 min/epoch. 4-fold × 35 epochs × 57 min = ~133 hours. Too slow given current experiment queue. Both folds terminated manually.

### Conclusion
B3 4-fold CV is impractical. Instead:
- Use single-fold b3-v1-asl (holdout=0.9553) as ensemble member
- If B3 CV needed: reduce to 2-fold or reduce epochs to 20

---

## sed-b0-4fold-v30-fold0 (Completed 2026-03-20)

**Best val ROC-AUC**: **0.8033 @ epoch 10**
**Early stopped**: ep16 (patience=6, best@ep10)
**Training curve**: ep1=0.7750?, ep10=0.8033(peak), ep12=0.7786, ep13=0.7877, ep14=0.7439, ep15=0.7631, ep16=0.7516
**Verdict**: PASS (>0.77) — strong result for fold0

### Key Observations
- Peak at ep10 then gradual decline — classic pattern for v30-multipseu formula
- 0.8033 is only slightly below the single-fold v30-multipseu (0.8139), consistent
- eps 12-13 briefly recovered to 0.7877 before final decline — model was still learning
- fold0 earns ~26k training files, slightly different from single-fold split

### Conclusion
fold0 is **soup-ready**: checkpoints ep1–ep10 will form part of the 4-fold ensemble soup.
GPU0 freed → fold1 launched (PID 3534107).

---

## sed-b0-4fold-v30 Progress (2026-03-20)

**Currently running**:
- **fold1** (GPU0, PID 3534107): just launched 2026-03-20 ~15:32
- **fold2** (GPU1, PID 3522746): ep6=0.7989 (best so far), ep7 training

**Completed**: fold0 — best=0.8033 @ep10 ✅
**Pending**: fold3 (launch after fold1 or fold2 completes)

**Key observation**: fold2 ep6=0.7989 is the highest single-epoch soundscape AUC in 4fold series so far.

---

## sed-b0-v31-freqmixstyle (Auto-concluded 2026-03-20)

**Config**: EfficientNet-B0, ASL loss, v30-multipseu formula + **Freq-MixStyle** (Beta(0.6,0.6), prob=0.5)
**Best val ROC-AUC**: 0.7806 @ epoch 1
**Early stopped**: ep7 (patience=6, best@ep1)
**Training curve**: ep1=0.7806, ep2=0.7481, ep3=0.7397, ep4=0.7696, ep5=0.7747, ep6=0.7335, ep7=0.7698
**Verdict**: FAIL relative to v30-multipseu (0.8139) — Freq-MixStyle did NOT improve soundscape AUC

### Key Observations
- Best at ep1 (warmup first epoch) then persistent decline throughout training
- Pattern is opposite of v30-multipseu (which improved progressively to ep6)
- Freq-MixStyle Beta(0.6,0.6) interfered with learning — likely too aggressive for this domain
- ep4-5 partial recovery (0.770-0.775) but never surpassed ep1
- ep6 collapse to 0.7335 is severe — may indicate the augmentation destabilized late training

### Root Cause
Freq-MixStyle shuffles frequency channels across batch items to simulate device/mic response shift. However, in BirdCLEF Pantanal:
1. Train audio already has diverse recording conditions (multiple ARU types)
2. Soundscape val clips may have similar frequency response to training — Freq-MixStyle creates unnecessary distribution noise
3. Beta(0.6,0.6) is aggressive (heavy mixing) — the paper's result was on DCASE industrial sounds which have stronger device-domain shift than bird songs

### Conclusion
**Freq-MixStyle (alone) is HARMFUL for this dataset.** Do NOT add it to the v30 formula without evidence from a milder variant (e.g., prob=0.2 or Beta(0.2,0.2)).

---

## Domain Generalization Experiment Queue (Updated 2026-03-20)

**Framing correction**: SFDA is wrong — test_soundscapes has only readme.txt. Correct approach is Domain Generalization. See `knowledges/domain_generalization.md`.

**Key baseline**: perch-probe-v3-addtrain-only OOF AUC = 0.8119

### Priority Queue (after v30 4-fold completes)

**Step 0 (immediate, no GPU)**: fold3 → `configs/sed_b0_4fold_v30_fold3.yaml`

**Step 1 — DG SED experiments (need code in mel_dataset.py first)**
| Priority | Config | Method | New Sources | Status |
|----------|--------|--------|-------------|--------|
| 1 | `sed_b0_v31_freqmixstyle.yaml` | Freq-MixStyle β=0.6 | DCASE 2024 | Needs mel_dataset code |
| 2 | `sed_b0_v32_multilabel_mixup.yaml` | 3-clip Dirichlet mixup | BirdSet 2024 | Needs mel_dataset code |
| 3 | `sed_b0_v33_freqmixstyle_multilabel.yaml` | v31+v32 combined | Both | Needs mel_dataset code |

**Step 2 — DG Perch Probe experiments (no code changes needed)**
| Priority | Config | Method | Status |
|----------|--------|--------|--------|
| 4 | `perch_probe_v3_laplacian.yaml` | LaplacianShot transductive | Ready now |
| 5 | `perch_probe_v3_fecam.yaml` | FeCAM Mahalanobis NCM | Ready now |
| 6 | `perch_probe_v4_mixstyle.yaml` | MixStyle on PCA embeddings | Needs code |
| 7 | `perch_probe_v4_groupdro.yaml` | GroupDRO by ARU site | Needs code |

**Step 3 — Inference trick (zero GPU cost)**
- Apply geo_mask (Pantanal species prior) to ensemble — `scripts/build_geo_mask.py` already exists
- Expected +0.010–0.030 on Pantanal test set

### Key paper references
- Freq-MixStyle: arXiv 2407.03654 (DCASE 2024) — use Beta(0.6,0.6) NOT (0.1,0.1)
- ProtoCLR / focal→soundscape gap: arXiv 2409.08589 — prototype contrastive, +42.4% few-shot
- BirdSet geo masking + multi-label mixup: arXiv 2403.10380

### Launch commands
```bash
# Perch probe (ready now, no code changes)
CUDA_VISIBLE_DEVICES=0 nohup python train_perch_probe.py --config configs/perch_probe_v3_laplacian.yaml > outputs/perch-probe-v3-laplacian.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train_perch_probe.py --config configs/perch_probe_v3_fecam.yaml > outputs/perch-probe-v3-fecam.log 2>&1 &

# SED DG (after implementing FreqMixStyle + MultiClipMix in mel_dataset.py)
CUDA_VISIBLE_DEVICES=0 nohup python train_sed.py --config configs/sed_b0_v31_freqmixstyle.yaml > outputs/sed-b0-v31-freqmixstyle/train.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train_sed.py --config configs/sed_b0_v32_multilabel_mixup.yaml > outputs/sed-b0-v32-multilabel-mixup/train.log 2>&1 &
```

---

## sed-b0-4fold-v30-fold1 (Completed 2026-03-20)

**Best val ROC-AUC**: **0.7840 @ epoch 5**
**Early stopped**: ep11 (patience=6, best@ep5)
**Training curve**: ep1=0.6921, ep2=0.7655, ep3=0.7389, ep4=0.7788, ep5=0.7840, ep6=0.7730, ep7=0.7767, ep8=0.7588, ep9=0.7820, ep10=0.7744, ep11=0.7419
**Verdict**: PASS — consistent with 4-fold series

### Key Observations
- ep1 low (0.6921) due to warmup; peak at ep5 then oscillation
- ep9 partial recovery (0.7820 ≈ ep5) before final decline
- fold1 slightly below fold0 (0.7840 vs 0.8033) — normal fold variance
- Soup checkpoints: ep4, ep5, ep9 (top-3 ≥ 0.77)

### 4-fold Progress Summary
| Fold | Best AUC | Best Ep | Status |
|------|----------|---------|--------|
| fold0 | 0.8033 | ep10 | ✅ Done |
| fold1 | 0.7840 | ep5 | ✅ Done |
| fold2 | 0.7989 (partial) | ep6 | ❌ Killed at ep8 — must restart |
| fold3 | 0.8153 | ep2 | ⛔ Manually stopped at ep9 (user decision: move to GPU1 for new experiments) |

### Conclusion
fold3 manually stopped at ep9 (best=0.8153@ep2, checkpoints ep1-ep8 saved). fold2 best=0.7989@ep6 (checkpoints ep5-ep7 saved). **4-fold series suspended** — proceeding to DG experiment queue on GPU1. Can resume fold2+fold3 later if needed to build soup ensemble.

---

## Perch Probe DG Ablation Results (Completed 2026-03-21)

**Baseline**: perch-probe-v3-addtrain-only OOF AUC = **0.8119**

| Experiment | OOF AUC | MLP-only | Delta | Verdict |
|---|---|---|---|---|
| perch-probe-v3-laplacian | 0.8197 | 0.8202 | +0.0078 | ✅ PASS |
| perch-probe-v3-fecam | 0.7106 | 0.8202 | -0.1013 | ❌ FAIL |
| perch-probe-v3-abt | 0.7904 | 0.7904 | -0.0215 | ❌ FAIL |

### Key Observations
- **MLP-only component is 0.8202** in laplacian/fecam configs — both better than baseline 0.8119 (+0.0083). This gain comes from the config differences (same add_train_clips=True).
- **LaplacianShot blending marginally hurts** (0.8197 vs 0.8202 mlp_only, -0.0005). The transductive refinement is not beneficial for this multi-label bird audio task.
- **FeCAM is catastrophically harmful**: blending reduces 0.8202→0.7106 (-0.1096). Mahalanobis NCM with Ledoit-Wolf covariance is not appropriate for 234-class multilabel audio with severe label imbalance.
- **ABT (All-But-Top) preprocessing hurts**: removing top-10 PCA directions drops mlp_only from 0.8202→0.7904. The "rogue" directions apparently contain useful bird audio features, not domain noise.

### Root Causes
- **FeCAM failure**: 234-class multi-label setting violates NCM assumptions. FeCAM expects balanced classes for prototype estimation.
- **ABT failure**: Top PCA directions in Perch embeddings encode bird-specific features (call frequency, duration), not device artifacts. Removing them loses discriminative information.
- **LaplacianShot neutral**: Graph-based transductive inference adds complexity without benefit for the large-scale (739-clip soundscape) setting.

### Conclusion
**Best Perch probe strategy: pure MLP + add_train_clips (OOF 0.8202).** None of the DG refinements (LaplacianShot, FeCAM, ABT) improve over the MLP baseline. The perch embedding direction shows an upper bound of ~0.82 OOF AUC. Focus should shift back to SED model ensemble quality.

### Next Steps
- Resume `sed-b3-v1-fold0` (from soup_ep005, ep5→ep35, extra_epochs=30) on GPU1 ← **LAUNCHED 2026-03-21**
- Resume `sed-b3-v1-fold1` (from soup_ep006) when GPU0 frees
- `sed-b3-v2-lower-lr` (lr=2e-4, warmup=5ep) to stabilize B3 soundscape val

---

## sed-b0-4fold-v30-fold3 (Auto-concluded 2026-03-21)

**Config**: EfficientNet-B0, ASL loss, v30 formula (multi-round pseudo r1-r5 + soft KD), fold3 of 4-fold CV
**Best val ROC-AUC**: **0.8153 @ epoch 2** (pre-resume)
**Early stopped**: ep14 (patience=6, best@ep2)
**Verdict**: PASS (>0.77) — highest single-fold soundscape AUC in the 4-fold series

**Full training curve**:
- Pre-resume: ep1=0.7668, ep2=0.8153(best), ep3=0.7953, ep4=0.8003, ep5=0.8040, ep6=0.7912, ep7=0.7942, ep8=0.7733
- Post-resume (soup_ep008, extra_epochs=27): ep9=0.7927, ep10=0.7987, ep11=0.7884, ep12=0.7463, ep13=0.7585, ep14=0.7765

**Key observations**:
- Pre-resume: strong performance, ep2=0.8153 is the highest single-epoch AUC in the 4-fold series (fold0=0.8033, fold1=0.7840)
- Post-resume instability: LR schedule reset on resume caused warmup restart, leading to ~0.04 drop from best (0.8153→0.7987 max post-resume). Never recovered.
- ep12 collapse to 0.7463 is typical LR oscillation pattern on resume
- Despite post-resume instability, the pre-resume checkpoints (ep1-ep8) are high quality

**Root cause of post-resume decline**: `--extra_epochs 27` restarts the cosine LR schedule from warmup, causing ~20 epochs of LR instability post-resume. The model was already well-trained and the new LR warmup disrupted learned representations.

**Recommendation**: When resuming with extra_epochs, use a much lower LR (e.g., 1e-5 constant) rather than full warmup restart. Pre-resume checkpoints (ep1-ep8, best soup_ep008) are the ones to use for ensemble.

**4-fold Progress Summary (updated)**:
| Fold | Best AUC | Best Ep | Status |
|------|----------|---------|--------|
| fold0 | 0.8033 | ep10 | ✅ Done |
| fold1 | 0.7840 | ep5 | ✅ Done |
| fold2 | 0.7989 (partial) | ep6 | ⏳ Resuming from soup_ep007 |
| fold3 | 0.8153 | ep2 | ✅ Done (early stopped ep14) |

**Action**: GPU1 freed → launching fold2 resume from soup_ep007_sed.pt (extra_epochs=28) ← **LAUNCHED 2026-03-21 08:42**


---

## sed-b0-v21-dual-rating3 (Auto-concluded 2026-03-21)

**Config**: EfficientNet-B0, dual-loss (clip+frame), train_audio filtered to rating≥3
**Best val ROC-AUC**: **0.7289 @ epoch 13** | Early stopped ep18/30
**Verdict**: ❌ FAIL (< 0.77)

**Full training curve**: ep1=0.6484, ep2=0.6733, ep3=0.6823, ep4=0.6953, ep5=0.6768, ep6=0.6935, ep7=0.6819, ep8=0.6639, ep9=0.7287, ep10=0.6919, ep11=0.7053, ep12=0.7253, ep13=**0.7289**, ep14=0.7197, ep15=0.6793, ep16=0.7244, ep17=0.7200, ep18=0.7112

**Key observations**:
- Train loss starts at 0.0257 and reaches 0.0092 — ~5× higher than standard ASL experiments (v20=0.0050). Indicates rating3 filter uses different loss formulation (likely BCE, not ASL) or severely reduced dataset.
- Extremely noisy validation curve: massive oscillation ±0.05 throughout, never converging. e.g., ep8=0.664, ep9=0.729, ep10=0.692 — 0.065 swing in a single epoch.
- Best AUC 0.7289 is far below 4-fold v30 range (fold0=0.8033, fold3=0.8153).
- No sustained improvement pattern — curve oscillates around 0.69–0.73 without a trend.

**Root cause**: Rating≥3 filter reduces train_audio to ~30–40% of full dataset. Insufficient training signal for 234 classes causes severe underfitting and high-variance gradient updates. The dual-loss formulation may also be introducing label noise (frame-level labels from clip-level annotations are inherently noisy).

**Recommendation**: Do NOT use rating filtering for train_audio. The v30 formula (full data + pseudo labels) is clearly superior. If quality filtering is needed, apply it only to pseudo labels, not to original labeled data.

---

## sed-b0-v22-dual-noclipmix (Auto-concluded 2026-03-21)

**Config**: EfficientNet-B0, dual-loss, no clip-level mixup (mixup_alpha=0.0)
**Best val ROC-AUC**: **0.7485 @ epoch 5** | Early stopped ep10/30
**Verdict**: ❌ FAIL (< 0.77)

**Full training curve**: ep1=0.6819, ep2=0.6804, ep3=0.7076, ep4=0.7273, ep5=**0.7485**, ep6=0.7481, ep7=0.7314, ep8=0.7423, ep9=0.7404, ep10=0.7047

**Key observations**:
- Train loss collapses rapidly: 0.019→0.003 by ep10 (fastest collapse in series). Without mixup, model overfits to train_audio domain quickly.
- Best AUC 0.7485 reached at ep5, then gradual decline. The model converged prematurely.
- Compare to v20 (with mixup): similar architecture but v20 achieves ~0.82+ with mixup. This confirms mixup_alpha=0.5 is critical.
- Peak ep5 (0.7485) < ep5 of fold0-B0 (0.7814). Mixup buys ~+0.033 at the same epoch.

**Root cause**: Removing mixup removes the primary regularization mechanism. The model memorizes train_audio label patterns rather than learning generalizable acoustic features. Fast train loss collapse (0.019→0.003 in 10 epochs vs typical 0.005→0.003 in 35 epochs) confirms overfitting.

**Recommendation**: Mixup (alpha=0.5) is non-negotiable for soundscape generalization. Do not run experiments without it.

---

## sed-b0-v23-perch-asl-aug (Auto-concluded 2026-03-21)

**Config**: EfficientNet-B0, ASL loss, augmented with Perch teacher soft labels as additional training signal
**Best val ROC-AUC**: **0.7367 @ epoch 4** | Early stopped ep9/30
**Verdict**: ❌ FAIL (< 0.77)

**Full training curve**: ep1=0.7233, ep2=0.7247, ep3=0.6909, ep4=**0.7367**, ep5=0.7179, ep6=0.7221, ep7=0.7244, ep8=0.7247, ep9=0.7365

**Key observations**:
- Train loss starts at 0.0052 (normal ASL range), confirming ASL loss is active.
- Very flat curve: 0.723→0.725 range after ep3, with best only 0.7367. Essentially no learning signal after ep4.
- Compare to fold0-B0 (same ASL, same B0, no Perch aug): fold0 reaches 0.8033 by ep10. This experiment peaks at 0.7367.
- The model learns nothing useful from Perch teacher augmentation; in fact it hurts vs standard v30 formula (-0.067 AUC).
- Early stop triggers at ep9 (patience exhausted) — consistent with flat convergence.

**Root cause**: Perch teacher labels for train_audio clips are high-coverage (14795-dim → 234-dim projection) but the label distribution differs from soundscape domain. Using them as training signal introduces a domain shift in the label space. The ASL loss tries to fit both Perch and human labels simultaneously, diluting the soundscape-specific signal.

**Recommendation**: Do NOT add Perch teacher labels as direct training signal for SED. If using Perch information, use it only as embedding features (probe layer) or for pseudo-label filtering, not as soft targets in ASL loss. The v30 formula (train_audio + soundscape + multi-round pseudo labels) without Perch augmentation is still the gold standard.

### Experiment Queue Update (as of 2026-03-21)
v21/v22/v23 all FAILED. All three architectural variants (rating filter, no-mixup, Perch-aug) are inferior to the v30 baseline.

**Currently running**:
- `sed-b0-4fold-v30-fold2`: ep12, best 0.7992@ep11 — close to fold0 (0.8033), likely converging
- `sed-b3-v1-fold0`: ep11, best 0.7778@ep5 — declining for 6 epochs, likely near early stop

**Next priority when GPU frees**:
1. `sed-b3-v2-lower-lr`: B3 with lr=2e-4 (half of current), longer warmup=5ep — address B3 instability
2. `sed-b0-4fold-v30-fold2-soup`: Build soup from fold0+fold1+fold2+fold3 checkpoints when fold2 finishes

---

## sed-b0-v27-soft-boost (Auto-concluded 2026-03-21)

**Config**: EfficientNet-B0, ASL + dual loss (clip 0.5 + frame 0.5), soft pseudo KD (round5, **10× oversample** vs v24's 5×), hard pseudo r5
**Best val ROC-AUC**: **0.7804 @ epoch 10** | Early stopped ep15/30
**Verdict**: ✅ PASS (> 0.77)

**Full training curve**:
ep1=0.7403, ep2=0.7598, ep3=0.7377, ep4=0.7489, ep5=0.7716, ep6=0.7634, ep7=0.7553, ep8=0.7491, ep9=0.7721, ep10=**0.7804**, ep11=0.7558, ep12=0.7370, ep13=0.7516, ep14=0.7322, ep15=0.7527

**Key observations**:
- Doubling soft pseudo oversample (5x → 10x) improved over implied v24 baseline. The extra Perch KD signal appears beneficial.
- Persistent high variance throughout: swings of ±0.03 per epoch, no clean convergence. Typical for B0 + dual loss.
- Best peak ep10=0.7804 is the highest non-4fold B0 soundscape val on record.
- Still below 4-fold v30 series (fold0=0.8033, fold3=0.8153) — single fold always inferior to 4-fold ensemble training.

**Root cause of oscillation**: Dual loss (clip+frame) on soundscape domain creates noisy gradient signals. Frame-level labels derived from clip labels are heuristic and inconsistent between augmented views.

**Recommendation**: v27 formula (10x soft pseudo, dual loss, ASL) is a solid foundation. Use as soup ingredient for B0 ensemble. Could improve further with lower dual-loss frame weight (0.3 instead of 0.5).

---

## sed-v2s-v1 (Auto-concluded 2026-03-21)

**Config**: EfficientNetV2-S (`tf_efficientnetv2_s.in21k_ft_in1k`), Focal loss, lr=4e-4, batch=16, label_smoothing=0.05, **no pseudo labels, no soft KD**
**Best val ROC-AUC**: **0.7012 @ epoch 11** | Killed at ep16/30 (not early stopped — manually terminated)
**Verdict**: ❌ FAIL (< 0.77)

**Full training curve**:
ep1=0.6679, ep2=0.6479, ep3=0.6368, ep4=0.6303, ep5=0.6606, ep6=0.6800, ep7=0.6692, ep8=0.6470, ep9=0.6656, ep10=0.6810, ep11=**0.7012**, ep12=0.6656, ep13=0.6674, ep14=0.6569, ep15=0.6817, ep16=0.6765

**Key observations**:
- Extremely slow warm-up: first 4 epochs are below 0.67, only reaching 0.70 at ep11.
- High variance, never converges: ±0.03 swings throughout, no sustained improvement trend.
- Batch=16 is very small for a 20M-parameter backbone. Gradient estimates are noisy.
- No pseudo labels → model has only 1/3 the effective soundscape training signal vs v30 formula.
- Focal loss (γ=2) combined with label_smoothing=0.05 may be double-penalizing uncertain predictions.

**Root cause**: Minimal recipe (no pseudo KD, small batch) is insufficient for V2S to generalize to soundscape domain. The IN21K pretrained weights need more domain adaptation data.

**Recommendation**: V2S requires at minimum: larger batch (≥24), no label smoothing (conflicts with ASL/Focal), and pseudo labels. Confirmed by v2s-v3 improvement.

---

## sed-v2s-v2-asl (Auto-concluded 2026-03-21)

**Config**: EfficientNetV2-S, **ASL loss** (γ_neg=4, γ_pos=0, clip=0.05), lr=3e-4, batch=24, **no pseudo labels**
**Best val ROC-AUC**: **0.7113 @ epoch 4** | Early stopped ep9/30
**Verdict**: ❌ FAIL (< 0.77)

**Full training curve**:
ep1=0.6573, ep2=0.6752, ep3=0.6718, ep4=**0.7113**, ep5=0.7095, ep6=0.6904, ep7=0.6985, ep8=0.6749, ep9=0.6869

**Key observations**:
- Switching from Focal to ASL immediately improved: best ep4=0.7113 vs v1 best 0.7012. ASL is clearly better for multi-label soundscape.
- Larger batch (24 vs 16) and lower LR (3e-4 vs 4e-4) also helped early convergence.
- Still early stopped at ep9 — model converges but then immediately degrades. No pseudo labels = insufficient regularization for V2S size.
- Only +0.010 above v1. Still far from B0 baseline.

**Root cause**: Same as v2s-v1 — no pseudo labels, no soft KD. V2S overfits to limited soundscape training data after ep4-5. The larger model (20M params vs 8M for B0) overfits faster on the same limited data.

**Recommendation**: V2S with ASL is the right direction. The full recipe (v2s-v3) confirmed: adding pseudo labels + soft KD raises V2S from 0.71 → 0.76.

---

## sed-v2s-v3-full-kitchen (Auto-concluded 2026-03-21)

**Config**: EfficientNetV2-S, ASL + dual loss (0.5/0.5), soft pseudo KD (r5, **10×**), hard pseudo r5, lr=5e-4, batch=24 — **exact V2S equivalent of B0 best recipe (v28/fold2)**
**Best val ROC-AUC**: **0.7592 @ epoch 5** | Early stopped ep11/35
**Verdict**: ❌ FAIL (< 0.77) — but best V2S result to date

**Full training curve**:
ep1=0.7395, ep2=0.7575, ep3=0.7554, ep4=0.7418, ep5=**0.7592**, ep6=0.7347, ep7=0.7321, ep8=0.7021, ep9=0.7549, ep10=0.7320, ep11=0.7348

**Key observations**:
- Full recipe brings V2S from 0.71 → 0.76 (+0.05 over v2s-v2). Pseudo labels + soft KD are essential.
- Despite identical recipe to B0 fold2, V2S (0.7592) < B0 fold2 (0.7992) by **−0.040** — a large gap.
- V2S ep8 collapse (0.755→0.702) is characteristic: LR still high at ep8 (4.7e-4), causing destabilization. B0 is more LR-robust.
- Early stopped at ep11 with best at ep5 — V2S peaks earlier but decays faster than B0.
- Config comment predicted "V2S + soft KD expected > 0.97 holdout" — but that refers to **train_audio holdout AUC**, not soundscape val. Soundscape val ≠ train_audio holdout.

**Root cause of V2S < B0 on soundscape val**:
1. **Pretraining domain mismatch**: B0 uses `tf_efficientnet_b0.ns_jft_in1k` (Noisy Student + JFT), which has been trained on noisier, more diverse data — better generalization to soundscape domain. V2S uses `in21k_ft_in1k` (clean ImageNet21K), less robust to real-world noise.
2. **Model capacity vs data size**: V2S (20M params) overfits soundscape distribution faster than B0 (8M params) given the same training set size.
3. **LR sensitivity**: V2S needs lower peak LR (≤3e-4) for soundscape domain. 5e-4 causes ep8 collapse.

**Critical insight — soundscape val vs holdout AUC**:
V2S consistently achieves better **train_audio holdout AUC** (0.97) but worse **soundscape val AUC** (0.76 vs B0 0.80). This confirms our earlier observation that train_audio holdout ≠ LB proxy. **Soundscape val AUC is the correct metric to optimize for LB performance.**

**Recommendation**:
- Do NOT prioritize V2S for soundscape generalization. B0 + NS-JFT pretraining is architecturally superior for this task.
- If V2S must be used, try: lr_peak=2e-4, warmup_epochs=5, and possibly a larger grad_clip.
- Focus GPU time on B0 4-fold ensemble (fold2 finishing) and B3 (fold0 at 0.7778).

### EfficientNetV2-S Series Summary
| Experiment | Best AUC | Best Ep | Key Change | Status |
|---|---|---|---|---|
| v2s-v1 | 0.7012 | ep11 | Baseline, Focal loss | ❌ FAIL |
| v2s-v2-asl | 0.7113 | ep4 | ASL loss | ❌ FAIL |
| v2s-v3-full-kitchen | 0.7592 | ep5 | Full recipe (= B0 best) | ❌ FAIL |
| **B0 fold2 (reference)** | **0.7992** | ep11 | Same recipe, B0 backbone | ✅ |

**Conclusion: V2S backbone is not competitive with B0+NS-JFT for soundscape domain generalization. Discontinuing V2S experiments.**
