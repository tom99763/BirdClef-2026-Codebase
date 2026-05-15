#!/usr/bin/env python3
"""Train Tucker-style Distilled SED — faithful port of bc2026-distilled-sed.ipynb.

Architecture  : EfficientNet-B0 + GeMFreqPool + SED attention head
                + Perch-v2 ONNX distillation branch (GAP+Linear→1536-d MSE)
Loss          : BCE(0.5*clip + 0.5*frame_max) + alpha * MSE(distill, perch)
Best ckpt     : non-S22 macro AUC  (S22 = known-noisy soundscape site)
Output        : fold{k}_best_ns22.pt  +  sed_fold{k}.onnx  (SED-only)

Usage (single fold):
    python train_tucker_sed.py --fold 0 --gpu 0

Usage (all folds, background):
    GPU=0 nohup bash scripts/auto_tucker_sed.sh > outputs/logs/tucker_train.log 2>&1 &
"""

import argparse, gc, math, os, random, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent
COMP_DIR     = ROOT / "birdclef-2026"
CACHE_DIR    = ROOT / "birdclef-2026/waveform_cache"
PERCH_ONNX   = ROOT / "weights/perch_v2_no_dft.onnx"
OUT_ROOT     = ROOT / "outputs/tucker-sed"

# ── Hyperparameters (mirror Tucker's notebook exactly) ────────────────────────
SEED          = 42
SR            = 32_000
NUM_CLASSES   = 234
TRAIN_DURATION = 5
VAL_DURATION   = 5
TRAIN_SAMPLES  = SR * TRAIN_DURATION
VAL_SAMPLES    = SR * VAL_DURATION

N_FFT         = 2048
HOP_LENGTH    = 512
N_MELS        = 256
FMIN          = 20
FMAX          = 16_000

BACKBONE_NAME     = "tf_efficientnet_b0.ns_jft_in1k"
DROP_PATH_RATE    = 0.1
USE_PERCH_DISTILL = True
PERCH_EMBED_DIM   = 1536
ALPHA_DISTILL     = 1.0

N_FOLDS       = 5
EPOCHS        = 25
BATCH         = 64
LR            = 5e-4
MIN_LR        = 1e-6
WD            = 1e-4
WARMUP_EPOCHS = 2
MIN_SAMPLE    = 20

AUG_PROB               = 0.5
AUG_GAIN_DB_RANGE      = (-6.0,  6.0)
AUG_NOISE_SNR_DB_RANGE = (10.0, 30.0)

# ── Ablation switches (all overridable via CLI) ───────────────────────────────
FOCAL_GAMMA     = 0.0    # 0.0 = plain BCE, >0 = focal weighting
LABEL_SMOOTH    = 0.0    # label smoothing ε (only for LOSS_TYPE="bce")
LOSS_TYPE       = "bce"  # "bce" | "focal" | "asl"
KD_TYPE         = "mse"  # "mse" | "cosine" | "infonce" | "mse_hard_infonce"
INFONCE_TEMP    = 0.07   # InfoNCE temperature
KD_LAMBDA       = 0.5    # weight for InfoNCE component in mse_hard_infonce combo
KD_HARD_RATIO   = 0.5    # fraction of hardest negatives to keep (0.3 = top-30%)
KD_BANK_SIZE    = 4096   # memory bank size (0 = disabled)
KD_BANK_TOPK    = 512    # top-K hardest bank negatives to sample per anchor
USE_TIME_SHIFT  = False  # waveform random circular shift ±1s
USE_SPECMIX     = False  # SpecMix: blend random freq band with shuffled peer
USE_CUTMIX_FREQ = False  # CutMix: hard freq band swap with shuffled peer
USE_SWA         = False  # Stochastic Weight Averaging (manual, last N epochs)
SWA_START_EP    = 30     # epoch index to start SWA state collection

USE_FOCAL_MIXUP    = True
MIXUP_PROB         = 0.5
MIXUP_ALPHA        = 0.4
MIXUP_HARD         = True

USE_FOCAL_SC_MIXUP    = True
FOCAL_SC_MIXUP_PROB   = 0.5
FOCAL_SC_MIXUP_ALPHA  = 0.4

FREQ_MASK_PARAM = 10
TIME_MASK_PARAM = 10
NUM_FREQ_MASKS  = 1
NUM_TIME_MASKS  = 2

USE_FOCAL           = True
USE_FOCAL_SECONDARY = True
USE_LABELED_SC      = True

INIT_CKPT       = None   # dir with fold{k}_best_ns22.pt to init from
FREEZE_BACKBONE = False  # freeze backbone, only train SED head
EMA_DECAY       = 0.0   # EMA decay for NS (0 = disabled, 0.99 = NS default)
PATIENCE        = 0      # early stopping patience on ns22 (0 = disabled)

# NS mode: pseudo labels for UNLABELED soundscapes (GT labels for labeled SC are never touched)
PSEUDO_UNLABELED_SC_CSV  = None  # CSV (filename, start_sec, species...) for unlabeled SS
UNLABELED_SC_CACHE_DIR   = None  # dir with unlabeled .pt waveform cache + unlabeled_ss_cache_meta.csv

SHARES         = {"focal": 0.9, "sc": 0.1}
SOURCE_WEIGHTS = {"focal": 1.0, "focal_missing": 0.0, "sc": 1.0, "unlabeled_sc": 1.0}

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


# ══════════════════════════════════════════════════════════════════════════════
# Mel + SpecAugment (GPU, torchaudio — matches Tucker's training pipeline)
# ══════════════════════════════════════════════════════════════════════════════
import torchaudio
import torchaudio.transforms as T

class MelSpecTransform(nn.Module):
    """GPU mel — torchaudio defaults (norm=None, mel_scale='htk').
    Matches Tucker's training mel pipeline exactly."""
    def __init__(self):
        super().__init__()
        self.mel_spec = T.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0,
        )
        self.to_db = T.AmplitudeToDB(top_db=80)

    @torch.no_grad()
    def forward(self, x):
        return self.to_db(self.mel_spec(x))


class SpecAugment(nn.Module):
    """1 freq mask + 2 time masks."""
    def __init__(self):
        super().__init__()
        self.freq_mask = T.FrequencyMasking(FREQ_MASK_PARAM)
        self.time_mask = T.TimeMasking(TIME_MASK_PARAM)

    def forward(self, mel):
        for _ in range(NUM_FREQ_MASKS):
            mel = self.freq_mask(mel)
        for _ in range(NUM_TIME_MASKS):
            mel = self.time_mask(mel)
        return mel


# ── Loss functions ────────────────────────────────────────────────────────────

def focal_bce_loss(logits, targets, gamma=2.0):
    """Per-element focal BCE for multi-label classification."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
    return ((1 - p_t) ** gamma) * bce


class AsymmetricLoss(nn.Module):
    """ICCV 2021: down-weights easy negatives in multi-label settings."""
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            prob = xs_pos * targets + xs_neg * (1 - targets)
            g = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            loss = loss * (1 - prob.clamp(min=0)) ** g
        return -loss  # [B, C], unreduced


def infonce_distill_loss(emb_s, emb_t, temperature=0.07):
    """InfoNCE contrastive distillation: (student_i, perch_i) = positive pair."""
    emb_s = F.normalize(emb_s.float(), dim=-1)
    emb_t = F.normalize(emb_t.float(), dim=-1)
    logits = torch.matmul(emb_s, emb_t.T) / temperature  # [B, B]
    labels = torch.arange(len(emb_s), device=emb_s.device)
    return F.cross_entropy(logits, labels)


def triplet_distill_loss(emb_s, emb_t, margin=0.5):
    """Triplet distillation: anchor=student_i, pos=perch_i, neg=hardest perch_j."""
    emb_s = F.normalize(emb_s.float(), dim=-1)
    emb_t = F.normalize(emb_t.float(), dim=-1)
    dist = 1.0 - torch.matmul(emb_s, emb_t.T)   # [B, B] cosine distance
    pos_dist = dist.diagonal()
    mask = torch.eye(len(emb_s), device=emb_s.device, dtype=torch.bool)
    neg_dist = dist.masked_fill(mask, float("inf")).min(dim=1).values
    return F.relu(pos_dist - neg_dist + margin).mean()


def hard_infonce_distill_loss(emb_s, emb_t, temperature=0.07, hard_ratio=0.5):
    """InfoNCE with hard negative mining: focuses only on top-50% hardest negatives.
    Hard negatives = in-batch Perch embeddings most similar to the student anchor."""
    emb_s = F.normalize(emb_s.float(), dim=-1)
    emb_t = F.normalize(emb_t.float(), dim=-1)
    sim = torch.matmul(emb_s, emb_t.T) / temperature   # [B, B]
    B = emb_s.size(0)
    labels = torch.arange(B, device=emb_s.device)
    eye = torch.eye(B, device=emb_s.device, dtype=torch.bool)
    # Find threshold: keep top hard_ratio hardest negatives per anchor
    sim_neg = sim.masked_fill(eye, float("-inf"))
    k = max(1, int(B * hard_ratio))
    thresh = sim_neg.topk(k, dim=1).values.min(dim=1, keepdim=True).values  # [B,1]
    # Mask out easy negatives (below threshold), keep positives
    easy_mask = (sim_neg < thresh) & (~eye)
    sim = sim.masked_fill(easy_mask, float("-inf"))
    return F.cross_entropy(sim, labels)


def supcon_distill_loss(emb_s, emb_t, labels, temperature=0.07):
    """Supervised Contrastive distillation (cross-modal, in-batch).

    For each student anchor i, positives = all teacher keys t_j where
    label_overlap(i, j) > 0 (multi-hot AND, including self j=i).
    Denominator = all B teacher keys.

    labels: [B, C] float (binarized at 0.5 for overlap test)
    """
    emb_s = F.normalize(emb_s.float(), dim=-1)   # [B, D]
    emb_t = F.normalize(emb_t.float(), dim=-1)   # [B, D]
    B = emb_s.size(0)

    sim = torch.matmul(emb_s, emb_t.T) / temperature  # [B, B]

    # Positive mask: any class overlap between clip i and clip j
    lb_bin = (labels > 0.5).float()              # [B, C]
    pos_mask = (torch.matmul(lb_bin, lb_bin.T) > 0).float()  # [B, B]

    # Numerically stable log-sum-exp
    sim_max = sim.detach().max(dim=1, keepdim=True).values
    exp_sim = torch.exp(sim - sim_max)           # [B, B]
    log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    log_prob = (sim - sim_max) - log_denom       # [B, B]

    # Average log-prob over positives per anchor
    n_pos = pos_mask.sum(dim=1).clamp(min=1)     # [B]
    loss = -(pos_mask * log_prob).sum(dim=1) / n_pos

    return loss.mean()


def supcon_membank_distill_loss(emb_s, emb_t, labels, memory_bank,
                                 temperature=0.07, bank_topk=512):
    """SupCon with memory bank: extends teacher key pool beyond current batch.

    Bank negatives: top-bank_topk hardest entries per anchor (no label info).
    Bank positives: cannot determine (no labels stored) → treated as negatives.
    Only in-batch pairs benefit from label supervision; bank adds hard negatives.
    """
    emb_s = F.normalize(emb_s.float(), dim=-1)   # [B, D]
    emb_t = F.normalize(emb_t.float(), dim=-1)   # [B, D]
    B = emb_s.size(0)

    # ── In-batch part (supervised) ──────────────────────────────────────────
    sim_inbatch = torch.matmul(emb_s, emb_t.T) / temperature  # [B, B]
    lb_bin = (labels > 0.5).float()
    pos_mask_inbatch = (torch.matmul(lb_bin, lb_bin.T) > 0).float()  # [B, B]

    # ── Bank part (unsupervised hard negatives) ──────────────────────────────
    bank = memory_bank.get()
    if bank is not None and bank.size(0) > 0:
        bank = bank.to(emb_s.device)
        k = min(bank_topk, bank.size(0))
        sim_bank_full = torch.matmul(emb_s, bank.T) / temperature  # [B, bank_size]
        sim_bank = sim_bank_full.topk(k, dim=1).values              # [B, k]
        # Bank entries have no labels → treat as negatives (pos_mask=0)
        pos_mask_bank = torch.zeros(B, k, device=emb_s.device)
        sim_all = torch.cat([sim_inbatch, sim_bank], dim=1)         # [B, B+k]
        pos_mask = torch.cat([pos_mask_inbatch, pos_mask_bank], dim=1)  # [B, B+k]
    else:
        sim_all = sim_inbatch
        pos_mask = pos_mask_inbatch

    # Numerically stable SupCon loss
    sim_max = sim_all.detach().max(dim=1, keepdim=True).values
    exp_sim = torch.exp(sim_all - sim_max)
    log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    log_prob = (sim_all - sim_max) - log_denom  # [B, B+k]

    n_pos = pos_mask.sum(dim=1).clamp(min=1)
    loss = -(pos_mask * log_prob).sum(dim=1) / n_pos

    memory_bank.update(emb_t)

    return loss.mean()


class MemoryBank:
    """FIFO queue of normalized Perch embeddings — richer negatives beyond the batch."""
    def __init__(self, size=4096, dim=PERCH_EMBED_DIM):
        self.size = size
        self.dim = dim
        self._bank = None   # lazy init (device unknown until first batch)
        self.ptr = 0
        self.n_filled = 0

    def _init(self, device):
        self._bank = F.normalize(torch.randn(self.size, self.dim, device=device), dim=-1)

    @torch.no_grad()
    def update(self, embs: torch.Tensor):
        if self._bank is None:
            self._init(embs.device)
        embs = F.normalize(embs.detach().float(), dim=-1)
        B = embs.size(0)
        if self.ptr + B <= self.size:
            self._bank[self.ptr:self.ptr + B] = embs
        else:
            split = self.size - self.ptr
            self._bank[self.ptr:] = embs[:split]
            self._bank[:B - split] = embs[split:]
        self.ptr = (self.ptr + B) % self.size
        self.n_filled = min(self.n_filled + B, self.size)

    def get(self):
        if self._bank is None or self.n_filled == 0:
            return None
        return self._bank[:self.n_filled].detach()


def membank_hard_infonce_loss(emb_s, emb_t, memory_bank, temperature=0.07,
                              hard_ratio=0.5, bank_topk=512):
    """InfoNCE with memory bank hard negative sampling.

    For each student anchor i:
      - Positive: (student_i, perch_i)
      - In-batch negatives: all perch_j (j≠i), 15 entries
      - Bank negatives: top-bank_topk hardest entries from memory bank,
        selected per anchor by student-bank cosine similarity.

    Two-stage hard selection:
      1. Bank: emb_s @ bank.T → topk per anchor (per-anchor, exact)
      2. Combined pool: hard_ratio filtering on in-batch + bank negatives
    """
    emb_s = F.normalize(emb_s.float(), dim=-1)   # [B, D]
    emb_t = F.normalize(emb_t.float(), dim=-1)   # [B, D]
    B = emb_s.size(0)

    # Positive: student_i · perch_i
    pos_sim = (emb_s * emb_t).sum(dim=1, keepdim=True) / temperature  # [B, 1]

    # In-batch negatives (mask diagonal)
    inbatch_sim = torch.matmul(emb_s, emb_t.T) / temperature  # [B, B]
    eye = torch.eye(B, device=emb_s.device, dtype=torch.bool)
    inbatch_sim = inbatch_sim.masked_fill(eye, float("-inf"))  # [B, B]

    # Bank negatives: top-bank_topk hardest per anchor
    bank = memory_bank.get()
    if bank is not None and bank.size(0) > 0:
        bank = bank.to(emb_s.device)
        k = min(bank_topk, bank.size(0))
        # [B, bank_size] → topk per anchor → [B, k] (reuse sim values, no gather needed)
        bank_sim_full = torch.matmul(emb_s, bank.T) / temperature  # [B, bank_size]
        bank_sim = bank_sim_full.topk(k, dim=1).values              # [B, k]
        neg_sim = torch.cat([inbatch_sim, bank_sim], dim=1)         # [B, B+k]
    else:
        neg_sim = inbatch_sim  # no bank yet, fall back to in-batch only

    # Hard ratio filtering on combined pool
    k2 = max(1, int(neg_sim.size(1) * hard_ratio))
    thresh = neg_sim.topk(k2, dim=1).values.min(dim=1, keepdim=True).values
    neg_sim = neg_sim.masked_fill(neg_sim < thresh, float("-inf"))

    # [pos | filtered_negs], label=0
    logits = torch.cat([pos_sim, neg_sim], dim=1)  # [B, 1+B+k]
    labels = torch.zeros(B, dtype=torch.long, device=emb_s.device)

    memory_bank.update(emb_t)   # update AFTER loss (current batch ≠ its own neg)

    return F.cross_entropy(logits, labels)


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════
import timm

class GeMFreqPool(nn.Module):
    """Generalized Mean pooling over frequency axis."""
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps

    def forward(self, x):
        p = self.p.clamp(min=1.0)
        return x.clamp(min=self.eps).pow(p).mean(dim=2).pow(1.0 / p)


class DistillHead(nn.Module):
    """GAP + Linear → Perch 1536-d embedding space."""
    def __init__(self, backbone_dim, embed_dim=PERCH_EMBED_DIM):
        super().__init__()
        self.proj = nn.Linear(backbone_dim, embed_dim)

    def forward(self, feat_map):
        return self.proj(feat_map.mean(dim=[2, 3]))


class BirdSEDModel(nn.Module):
    def __init__(self, backbone_name=BACKBONE_NAME, num_classes=NUM_CLASSES,
                 drop_path_rate=0.1, hidden_dim=512):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, in_chans=1,
            num_classes=0, global_pool="", drop_path_rate=drop_path_rate,
        )
        with torch.no_grad():
            n_tf = TRAIN_SAMPLES // HOP_LENGTH + 1
            dummy = torch.randn(1, 1, N_MELS, n_tf)
            feat = self.backbone(dummy)
            self.backbone_dim = feat.shape[1]
        print(f"  backbone_dim={self.backbone_dim}")

        self.gem_freq = GeMFreqPool()
        self.dense = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.backbone_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.att = nn.Conv1d(hidden_dim, num_classes, 1, bias=True)
        self.cla = nn.Conv1d(hidden_dim, num_classes, 1, bias=True)
        nn.init.xavier_uniform_(self.att.weight); self.att.bias.data.fill_(0.)
        nn.init.xavier_uniform_(self.cla.weight); self.cla.bias.data.fill_(0.)

        if USE_PERCH_DISTILL:
            self.distill_head = DistillHead(self.backbone_dim)

    def forward(self, x, return_framewise=False, return_distill=False):
        h = self.backbone(x)

        distill_emb = None
        if return_distill and hasattr(self, "distill_head"):
            distill_emb = self.distill_head(h)

        h_cls = h.detach() if USE_PERCH_DISTILL else h
        h_cls = self.gem_freq(h_cls)             # (B, C, T)
        h_cls = h_cls.permute(0, 2, 1)           # (B, T, C)
        h_cls = self.dense(h_cls)                # (B, T, 512)
        h_cls = h_cls.permute(0, 2, 1)           # (B, 512, T)

        norm_att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)
        framewise_logits = self.cla(h_cls)        # (B, C, T)
        clip_logits = (norm_att * framewise_logits).sum(dim=2)  # (B, C)

        fw = framewise_logits.permute(0, 2, 1) if return_framewise else None
        if return_framewise and return_distill: return clip_logits, fw, distill_emb
        elif return_framewise:                  return clip_logits, fw
        elif return_distill:                    return clip_logits, distill_emb
        return clip_logits


# ══════════════════════════════════════════════════════════════════════════════
# Perch teacher (PyTorch — GPU-native, cosine sim >0.999 vs ONNX reference)
# ══════════════════════════════════════════════════════════════════════════════

class PerchTeacher:
    """Frozen Perch v2 — PyTorch backbone, fully on GPU. No CPU roundtrip.
    Processes the SAME waveform fed to the model (same random crop), matching
    Tucker's notebook exactly. Verified cosine sim >0.999 vs ONNX."""

    _PARAMS = ROOT / "weights/perch_jax_backbone/perch_backbone_params.pkl"

    def __init__(self, device_str="cuda", fp16=True):
        from src.model.perch_pytorch import PerchNet, load_perch_weights
        from train_perch_ns import PerchMelTransform

        self.device = torch.device(device_str)
        self.fp16   = fp16

        net = PerchNet(num_classes=234, emb_dim=PERCH_EMBED_DIM)
        load_perch_weights(net, str(self._PARAMS), verbose=False)
        self.backbone = net.backbone.eval().to(self.device)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        if fp16:
            self.backbone = self.backbone.half()

        self.mel_tf = PerchMelTransform().to(self.device)

        print(f"  PerchTeacher PyTorch loaded (fp16={fp16}, device={device_str})")

    @torch.no_grad()
    def embed(self, waveforms_5s: torch.Tensor) -> torch.Tensor:
        """waveforms_5s: (B, 160000) float32. Returns (B, 1536) float32 on same device."""
        spec = self.mel_tf(waveforms_5s.to(self.device).float())  # (B, 1, 500, 128)
        if self.fp16:
            spec = spec.half()
        emb, _ = self.backbone(spec)
        return emb.float()


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation utilities
# ══════════════════════════════════════════════════════════════════════════════

def compute_macro_auc(y_true, y_pred, mask=None, class_mask=None):
    if mask is not None:
        y_true, y_pred = y_true[mask], y_pred[mask]
    if class_mask is not None:
        y_true, y_pred = y_true[:, class_mask], y_pred[:, class_mask]
    aucs = []
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        if col.sum() == 0 or col.sum() == len(col): continue
        try:   aucs.append(roc_auc_score(col, y_pred[:, c]))
        except ValueError: continue
    return (float(np.mean(aucs)) if aucs else float("nan")), len(aucs)


def full_eval(y_true, y_pred, ns22_mask, taxon_masks):
    r = {}
    a, n = compute_macro_auc(y_true, y_pred)
    r["macro_auc_all"], r["n_all"] = round(a, 4), n
    a, n = compute_macro_auc(y_true, y_pred, mask=ns22_mask)
    r["non_s22_macro"], r["n_ns22"] = round(a, 4), n
    for t, cm in taxon_masks.items():
        a, _ = compute_macro_auc(y_true, y_pred, mask=ns22_mask, class_mask=cm)
        r[f"non_s22_{t}"] = round(a, 4)
    return r


# ══════════════════════════════════════════════════════════════════════════════
# Waveform cache I/O
# ══════════════════════════════════════════════════════════════════════════════

_FC = {}
def load_focal(cache_file, cache_dir=CACHE_DIR):
    key = str(cache_dir / cache_file)
    if key in _FC: return _FC[key]
    pp = cache_dir / cache_file
    if not pp.exists(): return None
    a = torch.load(pp, map_location="cpu", weights_only=True).float().div(32767.0).numpy()
    if len(_FC) >= 2000: _FC.pop(next(iter(_FC)))
    _FC[key] = a
    return a


_SC = {}
def load_sc_waveform(cache_file, cache_dir=CACHE_DIR):
    key = str(cache_dir / cache_file)
    if key in _SC: return _SC[key]
    pp = cache_dir / cache_file
    if not pp.exists(): return None
    a = torch.load(pp, map_location="cpu", weights_only=True).float().div(32767.0).numpy()
    if len(_SC) >= 200: _SC.pop(next(iter(_SC)))
    _SC[key] = a
    return a


def extract_chunk(waveform, start_sample, n_samples):
    """Extract chunk with LEFT-padding if too short (matches Tucker's notebook)."""
    total = len(waveform)
    if total <= n_samples:
        return np.pad(waveform, (n_samples - total, 0))
    end = start_sample + n_samples
    if end > total: start_sample = max(0, total - n_samples)
    return waveform[start_sample: start_sample + n_samples]


def apply_aug(w):
    if USE_TIME_SHIFT and np.random.random() < AUG_PROB:
        shift = np.random.randint(-SR, SR + 1)   # ±1s circular shift
        w = np.roll(w, shift)
    if np.random.random() < AUG_PROB:
        w = w * (10 ** (np.random.uniform(*AUG_GAIN_DB_RANGE) / 20))
    if np.random.random() < AUG_PROB:
        sp = (w ** 2).mean()
        if sp > 1e-10:
            snr = np.random.uniform(*AUG_NOISE_SNR_DB_RANGE)
            w = w + np.random.randn(*w.shape).astype(w.dtype) * np.sqrt(sp / (10 ** (snr / 10)))
    return w


def specmix_batch(mel):
    """Blend a random freq band of each sample with a shuffled peer."""
    B, C, F, T = mel.shape
    perm = torch.randperm(B, device=mel.device)
    f_s = random.randint(0, F // 2)
    f_e = min(f_s + random.randint(F // 4, F // 2), F)
    lam = random.uniform(0.3, 0.7)
    mixed = mel.clone()
    mixed[:, :, f_s:f_e, :] = (lam * mel[:, :, f_s:f_e, :]
                                + (1 - lam) * mel[perm][:, :, f_s:f_e, :])
    return mixed, perm


def cutmix_freq_batch(mel):
    """Hard-replace a freq band in each sample with a shuffled peer's band."""
    B, C, F, T = mel.shape
    perm = torch.randperm(B, device=mel.device)
    f_s = random.randint(0, F // 2)
    f_e = min(f_s + random.randint(F // 4, F // 2), F)
    mixed = mel.clone()
    mixed[:, :, f_s:f_e, :] = mel[perm][:, :, f_s:f_e, :]
    return mixed, perm


# ══════════════════════════════════════════════════════════════════════════════
# Datasets
# ══════════════════════════════════════════════════════════════════════════════

class FocalDS(Dataset):
    """Focal recording dataset. Returns 5-tuple matching Tucker's notebook exactly."""
    def __init__(self, df, l2i, secondary_lookup=None,
                 sc_mixup_sources=None, fold_k=None, aug=False):
        self.df = df.reset_index(drop=True)
        self.l2i = l2i
        self.aug = aug
        self.secondary_lookup = secondary_lookup
        self.sc_mixup_sources = sc_mixup_sources
        self.fold_k = fold_k

    def __len__(self): return len(self.df)

    def _load_chunk(self, r):
        w = load_focal(r["cache_file"])
        if w is None: return None, None
        start = np.random.randint(0, max(1, len(w) - TRAIN_SAMPLES + 1)) \
                if self.aug and len(w) > TRAIN_SAMPLES else 0
        ch = extract_chunk(w, start, TRAIN_SAMPLES)
        lb = np.zeros(NUM_CLASSES, dtype=np.float32)
        if str(r["primary_label"]) in self.l2i:
            lb[self.l2i[str(r["primary_label"])]] = 1.0
        if self.secondary_lookup is not None and "original_idx" in self.df.columns:
            for s in self.secondary_lookup.get(int(r["original_idx"]), []):
                if s in self.l2i: lb[self.l2i[s]] = 1.0
        return ch, lb

    def __getitem__(self, i):
        r1 = self.df.iloc[i]
        ch1, lb1 = self._load_chunk(r1)
        if ch1 is None:
            return (torch.zeros(1, TRAIN_SAMPLES), torch.zeros(NUM_CLASSES),
                    torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES), "focal_missing")

        # Focal-Focal MixUp
        if USE_FOCAL_MIXUP and self.aug and np.random.random() < MIXUP_PROB:
            for _ in range(3):
                j = np.random.randint(len(self.df))
                ch2, lb2 = self._load_chunk(self.df.iloc[j])
                if ch2 is not None:
                    lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA) if MIXUP_ALPHA > 0 else 1.0
                    ch_mix = (lam * ch1 + (1 - lam) * ch2).astype(np.float32)
                    if self.aug: ch_mix = apply_aug(ch_mix)
                    lb = np.maximum(lb1, lb2) if MIXUP_HARD else lam*lb1 + (1-lam)*lb2
                    return (torch.from_numpy(ch_mix).unsqueeze(0), torch.from_numpy(lb),
                            torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES), "focal")

        # Focal-Soundscape MixUp
        if (USE_FOCAL_SC_MIXUP and self.aug and self.sc_mixup_sources
                and np.random.random() < FOCAL_SC_MIXUP_PROB):
            src = self.sc_mixup_sources[np.random.randint(len(self.sc_mixup_sources))]
            cache_dir, meta_df_sc, labels = src
            elig = meta_df_sc[meta_df_sc["fold"] != self.fold_k] \
                   if self.fold_k is not None else meta_df_sc
            if len(elig) > 0:
                sc_row = elig.iloc[np.random.randint(len(elig))]
                sc_wav = load_sc_waveform(sc_row["cache_file"], cache_dir)
                if sc_wav is not None and len(sc_wav) >= TRAIN_SAMPLES:
                    sc_chunk = extract_chunk(sc_wav, int(sc_row["start_sec"])*SR, TRAIN_SAMPLES)
                    lam = np.random.beta(FOCAL_SC_MIXUP_ALPHA, FOCAL_SC_MIXUP_ALPHA) if FOCAL_SC_MIXUP_ALPHA > 0 else 1.0
                    ch_mix = (lam * ch1 + (1 - lam) * sc_chunk).astype(np.float32)
                    if self.aug: ch_mix = apply_aug(ch_mix)
                    lb_sc = labels[int(sc_row["label_idx"])].astype(np.float32)
                    lb = np.maximum(lb1, lb_sc) if MIXUP_HARD else lam*lb1 + (1-lam)*lb_sc
                    return (torch.from_numpy(ch_mix).unsqueeze(0), torch.from_numpy(lb),
                            torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES), "focal")

        if self.aug: ch1 = apply_aug(ch1)
        return (torch.from_numpy(ch1.astype(np.float32)).unsqueeze(0),
                torch.from_numpy(lb1),
                torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES), "focal")


class ScDS(Dataset):
    def __init__(self, Y, sc_df, aug=False, cache_dir=None):
        self.Y = Y
        self.df = sc_df.reset_index(drop=True)
        self.aug = aug
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR

    def __len__(self): return len(self.Y)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        cf = row.get("cache_file")
        wav_full = load_sc_waveform(cf, self.cache_dir) if cf else None
        if wav_full is None:
            wav_t = torch.zeros(1, TRAIN_SAMPLES)
        else:
            chunk = extract_chunk(wav_full, int(row["start_sec"])*SR, TRAIN_SAMPLES)
            if self.aug: chunk = apply_aug(chunk)
            wav_t = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0)
        return (wav_t, torch.from_numpy(self.Y[i].astype(np.float32)),
                torch.ones(NUM_CLASSES), torch.ones(NUM_CLASSES), "sc")


class MixSamp(torch.utils.data.Sampler):
    """Controls batch composition: 90% focal / 10% SC by default."""
    def __init__(self, sizes, names, shares, bs, nst, seed=0):
        self.sizes = sizes; self.names = names
        self.bs = bs; self.nst = nst
        self.rng = np.random.default_rng(seed)
        per_src = [max(1, int(round(bs * shares.get(n, 0.0)))) for n in names]
        total = sum(per_src)
        if total != bs: per_src[int(np.argmax(per_src))] += (bs - total)
        self.per_src = per_src
        self.offsets = [0]
        for s in sizes[:-1]: self.offsets.append(self.offsets[-1] + s)

    def __len__(self): return self.nst

    def __iter__(self):
        for _ in range(self.nst):
            batch = []
            for off, size, n in zip(self.offsets, self.sizes, self.per_src):
                if n <= 0 or size <= 0: continue
                batch.extend([off + int(i) for i in self.rng.integers(0, size, size=n)])
            self.rng.shuffle(batch)
            yield batch


def collate_m(batch):
    return (torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]),
            torch.stack([b[3] for b in batch]),
            [b[4] for b in batch])


def mk_sw(sr):
    return torch.tensor([SOURCE_WEIGHTS.get(s, 0.0) for s in sr], dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# ONNX export wrapper (SED-only, no distill head)
# ══════════════════════════════════════════════════════════════════════════════

class SEDExportWrapper(nn.Module):
    """Inference-only SED model: Linear dense → Conv1d for stable ONNX tracing."""
    def __init__(self, backbone_name, num_classes, backbone_dim, hidden_dim=512):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, in_chans=1,
            num_classes=0, global_pool="", drop_path_rate=0.1,
        )
        self.gem_freq  = GeMFreqPool()
        self.dense_drop1 = nn.Dropout(0.25)
        self.dense_conv  = nn.Conv1d(backbone_dim, hidden_dim, 1)
        self.dense_relu  = nn.ReLU(inplace=True)
        self.dense_drop2 = nn.Dropout(0.5)
        self.att = nn.Conv1d(hidden_dim, num_classes, 1)
        self.cla = nn.Conv1d(hidden_dim, num_classes, 1)

    def forward(self, mel):
        h = self.backbone(mel)
        h = self.gem_freq(h)
        h = self.dense_drop1(h)
        h = self.dense_conv(h)
        h = self.dense_relu(h)
        h = self.dense_drop2(h)
        norm_att = torch.softmax(torch.tanh(self.att(h)), dim=-1)
        framewise = self.cla(h)
        clip = (norm_att * framewise).sum(dim=2)
        return clip, framewise.permute(0, 2, 1)


def _remap_state_to_export(export_model, trained_state):
    """Map BirdSEDModel state_dict → SEDExportWrapper keys (Linear → Conv1d)."""
    remap = {}
    for k, v in trained_state.items():
        if k.startswith("distill_head."): continue
        if k == "dense.1.weight":   remap["dense_conv.weight"] = v.unsqueeze(-1)
        elif k == "dense.1.bias":   remap["dense_conv.bias"]   = v
        else:                        remap[k] = v
    missing, unexpected = export_model.load_state_dict(remap, strict=False)
    if missing:     print(f"    WARN missing: {missing[:5]}")
    if unexpected:  print(f"    WARN unexpected: {unexpected[:5]}")


def export_onnx(trained_state, backbone_dim, out_path, device, backbone_name=None):
    exp = SEDExportWrapper(backbone_name or BACKBONE_NAME, NUM_CLASSES, backbone_dim).to(device)
    _remap_state_to_export(exp, trained_state)
    exp.eval()
    n_frames = VAL_SAMPLES // HOP_LENGTH + 1
    dummy = torch.randn(1, 1, N_MELS, n_frames, device=device)
    torch.onnx.export(
        exp, dummy, str(out_path),
        input_names=["mel"],
        output_names=["clip_logits", "framewise_logits"],
        dynamic_axes={"mel": {0: "batch"},
                      "clip_logits": {0: "batch"},
                      "framewise_logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"mel": dummy.cpu().numpy()})
    with torch.no_grad(): ref_clip, _ = exp(dummy)
    diff = float(np.abs(ref_clip.cpu().numpy() - ort_out[0]).max())
    print(f"    ONNX verify max|diff|={diff:.3e}  size={out_path.stat().st_size/1e6:.1f}MB")
    assert diff < 0.01, f"ONNX export diverged: {diff}"
    del exp, sess


# ══════════════════════════════════════════════════════════════════════════════
# Validation helper
# ══════════════════════════════════════════════════════════════════════════════

def predict_waveforms(model, mel_tf, wav_list, device, bs=64):
    model.eval()
    preds_clip, preds_fmax, preds_blend = [], [], []
    from torch.cuda.amp import autocast
    with torch.no_grad():
        for s in range(0, len(wav_list), bs):
            batch = torch.stack(wav_list[s:s+bs]).to(device)
            mel = mel_tf(batch)
            B = mel.size(0)
            for i in range(B):
                mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)
            with autocast():
                clip_logits, framewise = model(mel, return_framewise=True)
                fmax_logits = framewise.max(dim=1).values
                p_clip  = torch.sigmoid(clip_logits).float().cpu().numpy()
                p_fmax  = torch.sigmoid(fmax_logits).float().cpu().numpy()
                p_blend = 0.5 * p_clip + 0.5 * p_fmax
            preds_clip.append(p_clip)
            preds_fmax.append(p_fmax)
            preds_blend.append(p_blend)
    return {
        "clip":  np.concatenate(preds_clip),
        "fmax":  np.concatenate(preds_fmax),
        "blend": np.concatenate(preds_blend),
    }


def load_val_waveforms(val_sc_df, sc_file_dict):
    wavs = []
    for _, row in val_sc_df.iterrows():
        cf = sc_file_dict.get(row["filename"])
        if cf is not None:
            w = load_sc_waveform(cf)
            if w is not None:
                chunk = extract_chunk(w, int(row["start_sec"]) * SR, VAL_SAMPLES)
                wavs.append(torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0))
                continue
        wavs.append(torch.zeros(1, VAL_SAMPLES))
    return wavs


# ══════════════════════════════════════════════════════════════════════════════
# Training — single fold
# ══════════════════════════════════════════════════════════════════════════════

def train_fold(fold_k, audio_cache_meta, sc_cache_meta, Y_SC, non_s22_mask_sc,
               taxon_masks, sc_file_dict, focal_secondary_labels, sc_mixup_sources,
               device, out_dir, num_workers=4,
               unlabeled_sc_meta=None, Y_unlabeled_SC=None):
    from torch.cuda.amp import GradScaler, autocast

    out_dir.mkdir(parents=True, exist_ok=True)

    vm = sc_cache_meta["fold"].values == fold_k
    Y_val = Y_SC[vm]
    ns22_val = non_s22_mask_sc[vm]
    val_sc_df = sc_cache_meta[vm].reset_index(drop=True)
    val_wavs = load_val_waveforms(val_sc_df, sc_file_dict)

    # Datasets
    l2i = {l: i for i, l in enumerate(
        pd.read_csv(COMP_DIR / "sample_submission.csv").columns[1:].tolist())}
    items = []
    if USE_FOCAL:
        fds = FocalDS(audio_cache_meta[audio_cache_meta["fold"] != fold_k],
                      l2i, secondary_lookup=focal_secondary_labels,
                      sc_mixup_sources=sc_mixup_sources, fold_k=fold_k, aug=True)
        items.append(("focal", fds, len(fds)))
    if USE_LABELED_SC:
        sc_train_df = sc_cache_meta[~vm].reset_index(drop=True)
        Y_tr = Y_SC[~vm]
        sds = ScDS(Y_tr, sc_train_df, aug=True)
        items.append(("sc", sds, len(sds)))

    # Unlabeled soundscape pseudo-labeled data (NS mode — all folds use the full unlabeled pool)
    if unlabeled_sc_meta is not None and Y_unlabeled_SC is not None and UNLABELED_SC_CACHE_DIR is not None:
        ul_sds = ScDS(Y_unlabeled_SC, unlabeled_sc_meta, aug=True,
                      cache_dir=UNLABELED_SC_CACHE_DIR)
        items.append(("unlabeled_sc", ul_sds, len(ul_sds)))
        # Adjust batch shares: 85% focal, 7.5% labeled SC, 7.5% unlabeled SC
        SHARES.update({"focal": 0.85, "sc": 0.075, "unlabeled_sc": 0.075})
        print(f"  Unlabeled SC stream: {len(ul_sds)} windows  shares={SHARES}")

    names, datasets, sizes = zip(*items)
    nst = max(100, int(sum(sizes) / BATCH))
    print(f"  streams={dict(zip(names, sizes))}  steps/ep={nst}")

    m = BirdSEDModel(backbone_name=BACKBONE_NAME, drop_path_rate=DROP_PATH_RATE).to(device)

    # Load pre-trained checkpoint if specified
    if INIT_CKPT is not None:
        ckpt_file = Path(INIT_CKPT) / f"fold{fold_k}_best_ns22.pt"
        if not ckpt_file.exists():
            ckpt_file = Path(INIT_CKPT) / f"fold{fold_k}_best_macro.pt"
        raw = torch.load(ckpt_file, map_location=device)
        state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
        missing, unexpected = m.load_state_dict(state, strict=False)
        print(f"  Init ckpt: {ckpt_file}  missing={len(missing)}  unexpected={len(unexpected)}")

    # Freeze backbone if requested
    if FREEZE_BACKBONE:
        for p in m.backbone.parameters():
            p.requires_grad_(False)
        n_trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"  Backbone frozen. Trainable params: {n_trainable:,}")

    # EMA shadow model
    import copy as _copy
    ema_m = None
    if EMA_DECAY > 0:
        ema_m = _copy.deepcopy(m)
        for p in ema_m.parameters():
            p.requires_grad_(False)
        ema_m.eval()
        print(f"  EMA decay={EMA_DECAY}")

    mel_tf = MelSpecTransform().to(device)
    spec_aug = SpecAugment().to(device)
    perch = PerchTeacher(device_str=str(device)) if USE_PERCH_DISTILL else None
    asl_fn = AsymmetricLoss() if LOSS_TYPE == "asl" else None

    opt = torch.optim.AdamW(
        [p for p in m.parameters() if p.requires_grad], lr=LR, weight_decay=WD
    )
    scaler = GradScaler()
    warmup_steps = nst * WARMUP_EPOCHS
    total_steps  = nst * EPOCHS
    sch = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(opt, start_factor=1/25, end_factor=1.0,
                                               total_iters=warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps - warmup_steps,
                                                        eta_min=MIN_LR),
        ],
        milestones=[warmup_steps],
    )

    best_ns22, best_ns22_state = -1.0, None
    best_macro, best_macro_state = -1.0, None
    swa_states = []
    no_improve = 0

    # Memory bank for contrastive distillation (reset each fold)
    _bank_types = ("membank_hard_infonce", "mse_membank_hard_infonce",
                   "supcon_membank", "mse_supcon_membank")
    memory_bank = (MemoryBank(size=KD_BANK_SIZE)
                   if KD_TYPE in _bank_types and KD_BANK_SIZE > 0
                   else None)

    for ep in range(EPOCHS):
        m.train()
        smp = MixSamp(list(sizes), list(names), SHARES, BATCH, nst, seed=SEED + ep)
        tl  = DataLoader(ConcatDataset(list(datasets)),
                         batch_sampler=smp, collate_fn=collate_m,
                         num_workers=num_workers, pin_memory=True)
        el = el_cls = el_dist = nb = 0
        t0 = time.time()

        for wav, lb, wt, mk, sr in tl:
            wav, lb, wt, mk = wav.to(device), lb.to(device), wt.to(device), mk.to(device)
            sw = mk_sw(sr).to(device)

            with torch.no_grad():
                mel = mel_tf(wav)
                B = mel.size(0)
                for i in range(B):
                    mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)
                mel = spec_aug(mel)
                if USE_SPECMIX:
                    mel, _perm = specmix_batch(mel)
                    lb = torch.maximum(lb, lb[_perm])
                elif USE_CUTMIX_FREQ:
                    mel, _perm = cutmix_freq_batch(mel)
                    lb = torch.maximum(lb, lb[_perm])

            with autocast():
                if USE_PERCH_DISTILL:
                    clip_logits, framewise, distill_emb = m(mel, return_framewise=True,
                                                             return_distill=True)
                else:
                    clip_logits, framewise = m(mel, return_framewise=True)

                fmax_logits = framewise.max(dim=1).values
                if LOSS_TYPE == "asl":
                    raw = 0.5 * asl_fn(clip_logits, lb) + 0.5 * asl_fn(fmax_logits, lb)
                elif LOSS_TYPE == "focal":
                    raw = (0.5 * focal_bce_loss(clip_logits, lb, FOCAL_GAMMA)
                         + 0.5 * focal_bce_loss(fmax_logits, lb, FOCAL_GAMMA))
                else:  # bce (default)
                    lb_s = lb * (1 - LABEL_SMOOTH) + LABEL_SMOOTH * 0.5 if LABEL_SMOOTH > 0 else lb
                    raw = (0.5 * F.binary_cross_entropy_with_logits(clip_logits, lb_s, reduction="none")
                         + 0.5 * F.binary_cross_entropy_with_logits(fmax_logits, lb_s, reduction="none"))
                ps = (raw * wt * mk).sum(1) / (mk.sum(1) + 1e-8)
                cls_loss = (ps * sw).mean()

                # Distillation: same waveform fed to Perch as mel sees (Tucker-exact)
                if USE_PERCH_DISTILL and perch is not None:
                    with torch.no_grad():
                        perch_emb = perch.embed(wav.squeeze(1))  # (B, 1536) on device
                    if KD_TYPE == "cosine":
                        distill_loss = (1 - F.cosine_similarity(
                            F.normalize(distill_emb.float(), dim=-1),
                            F.normalize(perch_emb.float(), dim=-1)
                        )).mean()
                    elif KD_TYPE == "infonce":
                        distill_loss = infonce_distill_loss(distill_emb, perch_emb, INFONCE_TEMP)
                    elif KD_TYPE == "triplet":
                        distill_loss = triplet_distill_loss(distill_emb, perch_emb)
                    elif KD_TYPE == "hard_infonce":
                        distill_loss = hard_infonce_distill_loss(distill_emb, perch_emb, INFONCE_TEMP, KD_HARD_RATIO)
                    elif KD_TYPE == "mse_hard_infonce":
                        distill_loss = (F.mse_loss(distill_emb, perch_emb) +
                                        KD_LAMBDA * hard_infonce_distill_loss(distill_emb, perch_emb, INFONCE_TEMP, KD_HARD_RATIO))
                    elif KD_TYPE == "membank_hard_infonce":
                        distill_loss = membank_hard_infonce_loss(
                            distill_emb, perch_emb, memory_bank,
                            INFONCE_TEMP, KD_HARD_RATIO, KD_BANK_TOPK)
                    elif KD_TYPE == "mse_membank_hard_infonce":
                        distill_loss = (F.mse_loss(distill_emb, perch_emb) +
                                        KD_LAMBDA * membank_hard_infonce_loss(
                                            distill_emb, perch_emb, memory_bank,
                                            INFONCE_TEMP, KD_HARD_RATIO, KD_BANK_TOPK))
                    elif KD_TYPE == "supcon":
                        distill_loss = supcon_distill_loss(
                            distill_emb, perch_emb, lb, INFONCE_TEMP)
                    elif KD_TYPE == "mse_supcon":
                        distill_loss = (F.mse_loss(distill_emb, perch_emb) +
                                        KD_LAMBDA * supcon_distill_loss(
                                            distill_emb, perch_emb, lb, INFONCE_TEMP))
                    elif KD_TYPE == "supcon_membank":
                        distill_loss = supcon_membank_distill_loss(
                            distill_emb, perch_emb, lb, memory_bank,
                            INFONCE_TEMP, KD_BANK_TOPK)
                    elif KD_TYPE == "mse_supcon_membank":
                        distill_loss = (F.mse_loss(distill_emb, perch_emb) +
                                        KD_LAMBDA * supcon_membank_distill_loss(
                                            distill_emb, perch_emb, lb, memory_bank,
                                            INFONCE_TEMP, KD_BANK_TOPK))
                    else:  # mse (default)
                        distill_loss = F.mse_loss(distill_emb, perch_emb)
                    loss = cls_loss + ALPHA_DISTILL * distill_loss
                else:
                    distill_loss = torch.tensor(0.0)
                    loss = cls_loss

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            if ema_m is not None:
                with torch.no_grad():
                    for p_e, p_m in zip(ema_m.parameters(), m.parameters()):
                        p_e.data.mul_(EMA_DECAY).add_(p_m.data, alpha=1.0 - EMA_DECAY)
                    for b_e, b_m in zip(ema_m.buffers(), m.buffers()):
                        b_e.copy_(b_m)
            el += loss.item(); el_cls += cls_loss.item()
            el_dist += distill_loss.item(); nb += 1

        # Validation (use EMA model when enabled)
        val_model = ema_m if ema_m is not None else m
        val_preds = predict_waveforms(val_model, mel_tf, val_wavs, device)
        r = full_eval(Y_val, val_preds["blend"], ns22_val, taxon_masks)
        for mode in ["clip", "fmax", "blend"]:
            r[f"ns22_{mode}"] = full_eval(Y_val, val_preds[mode], ns22_val, taxon_masks)["non_s22_macro"]

        tag_ns22 = tag_macro = ""
        _state_src = val_model
        if r["non_s22_macro"] > best_ns22:
            best_ns22 = r["non_s22_macro"]
            best_ns22_state = {k: v.cpu().clone() for k, v in _state_src.state_dict().items()}
            tag_ns22 = " *ns22"
            no_improve = 0
        else:
            no_improve += 1
        if r["macro_auc_all"] > best_macro:
            best_macro = r["macro_auc_all"]
            best_macro_state = {k: v.cpu().clone() for k, v in _state_src.state_dict().items()}
            tag_macro = " *macro"

        dist_str = f" dist={el_dist/nb:.4f}" if USE_PERCH_DISTILL else ""
        print(f"  Ep{ep:02d}: loss={el/nb:.4f} cls={el_cls/nb:.4f}{dist_str} "
              f"lr={opt.param_groups[0]['lr']:.1e} | "
              f"ns22={r['ns22_blend']:.4f} "
              f"Av={r['non_s22_Aves']:.4f} Am={r['non_s22_Amphibia']:.4f} "
              f"In={r['non_s22_Insecta']:.4f} Ma={r['non_s22_Mammalia']:.4f} "
              f"[{time.time()-t0:.0f}s]{tag_ns22}{tag_macro}")
        if PATIENCE > 0 and no_improve >= PATIENCE:
            print(f"  Early stop at Ep{ep:02d} (no ns22 improve for {PATIENCE} epochs)")
            break
        if USE_SWA and ep >= SWA_START_EP:
            swa_states.append({k: v.cpu().clone() for k, v in m.state_dict().items()})

    # SWA: average collected states and compare to best_ns22
    if USE_SWA and swa_states:
        avg_state = {}
        for k in swa_states[0]:
            stacked = torch.stack([s[k].float() for s in swa_states])
            avg_state[k] = stacked.mean(0).to(swa_states[0][k].dtype)
        m_swa = BirdSEDModel(backbone_name=BACKBONE_NAME, drop_path_rate=DROP_PATH_RATE).to(device)
        m_swa.load_state_dict(avg_state)
        val_preds_swa = predict_waveforms(m_swa, mel_tf, val_wavs, device)
        r_swa = full_eval(Y_val, val_preds_swa["blend"], ns22_val, taxon_masks)
        swa_ns22 = r_swa["non_s22_macro"]
        print(f"  SWA(ep{SWA_START_EP}-{EPOCHS-1}): ns22={swa_ns22:.4f}  macro={r_swa['macro_auc_all']:.4f}", end="")
        if swa_ns22 > best_ns22:
            best_ns22 = swa_ns22
            best_ns22_state = avg_state
            print(" *SWA improves ns22!")
        else:
            print()
        del m_swa
        torch.cuda.empty_cache()

    # Save both ns22 and macro checkpoints + ONNX (Tucker: try both, may differ significantly on LB)
    if best_ns22_state is not None:
        torch.save({"state_dict": best_ns22_state, "fold": fold_k},
                   out_dir / f"fold{fold_k}_best_ns22.pt")
        print(f"  Saved fold{fold_k}_best_ns22.pt  (ns22={best_ns22:.4f})")
        export_onnx(best_ns22_state, m.backbone_dim,
                    out_dir / f"sed_fold{fold_k}_ns22.onnx", device, backbone_name=BACKBONE_NAME)

    if best_macro_state is not None:
        torch.save({"state_dict": best_macro_state, "fold": fold_k},
                   out_dir / f"fold{fold_k}_best_macro.pt")
        print(f"  Saved fold{fold_k}_best_macro.pt  (macro={best_macro:.4f})")
        export_onnx(best_macro_state, m.backbone_dim,
                    out_dir / f"sed_fold{fold_k}_macro.onnx", device, backbone_name=BACKBONE_NAME)

    # Default ONNX alias: ns22 preferred (Tucker's recommendation: test both on LB)
    ns22_onnx = out_dir / f"sed_fold{fold_k}_ns22.onnx"
    default_onnx = out_dir / f"sed_fold{fold_k}.onnx"
    if ns22_onnx.exists():
        import shutil; shutil.copy2(ns22_onnx, default_onnx)

    del m, mel_tf, spec_aug, perch
    torch.cuda.empty_cache(); gc.collect()
    return best_ns22, best_macro


# ══════════════════════════════════════════════════════════════════════════════
# Data loading + fold assignment (matches Tucker's notebook exactly)
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    sample_sub    = pd.read_csv(COMP_DIR / "sample_submission.csv")
    PRIMARY_LABELS = sample_sub.columns[1:].tolist()
    LABEL2IDX     = {l: i for i, l in enumerate(PRIMARY_LABELS)}
    taxonomy      = pd.read_csv(COMP_DIR / "taxonomy.csv")
    l2t = dict(zip(taxonomy["primary_label"].astype(str), taxonomy["class_name"].astype(str)))
    taxon_masks = {t: np.array([i for i, l in enumerate(PRIMARY_LABELS) if l2t.get(l,"") == t])
                   for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]}

    # Focal metadata
    audio_cache_meta = pd.read_csv(CACHE_DIR / "audio_cache_meta.csv")
    train_df = pd.read_csv(COMP_DIR / "train.csv")
    audio_cache_meta = audio_cache_meta.merge(
        train_df[["filename", "secondary_labels"]], on="filename", how="left")
    audio_cache_meta = audio_cache_meta[
        audio_cache_meta["primary_label"].isin(LABEL2IDX)].reset_index(drop=True)
    print(f"Focal cache: {len(audio_cache_meta)} entries")

    # Soundscape metadata
    sc_cache_meta = pd.read_csv(CACHE_DIR / "soundscape_cache_meta.csv")
    sc_cache_meta["label_list"] = sc_cache_meta["label_list"].apply(
        lambda x: x.split(";") if isinstance(x, str) else [])
    print(f"Soundscape cache: {len(sc_cache_meta)} windows")

    # Ground-truth label matrix
    sc_labels_raw = pd.read_csv(COMP_DIR / "train_soundscapes_labels.csv").drop_duplicates()
    sc_labels_raw["start_sec"] = pd.to_timedelta(sc_labels_raw["start"]).dt.total_seconds().astype(int)
    Y_SC = np.zeros((len(sc_cache_meta), len(PRIMARY_LABELS)), dtype=np.float32)
    for i, row in sc_cache_meta.iterrows():
        m = sc_labels_raw[(sc_labels_raw["filename"] == row["filename"]) &
                          (sc_labels_raw["start_sec"] == row["start_sec"])]
        for _, mr in m.iterrows():
            for lbl in str(mr["primary_label"]).split(";"):
                lbl = lbl.strip()
                if lbl in LABEL2IDX: Y_SC[i, LABEL2IDX[lbl]] = 1.0

    labeled_mask = Y_SC.sum(axis=1) > 0
    print(f"SC labels: {labeled_mask.sum()}/{len(Y_SC)} windows, {int(Y_SC.sum())} positives")

    # Unlabeled soundscape pseudo labels (NS mode — GT labels above are never touched)
    unlabeled_sc_meta = None
    Y_unlabeled_SC = None
    if PSEUDO_UNLABELED_SC_CSV is not None and UNLABELED_SC_CACHE_DIR is not None:
        ul_cache_dir = Path(UNLABELED_SC_CACHE_DIR)
        ul_meta = pd.read_csv(ul_cache_dir / "unlabeled_ss_cache_meta.csv")
        pseudo_df = pd.read_csv(PSEUDO_UNLABELED_SC_CSV)
        pseudo_df["start_sec"] = pseudo_df["start_sec"].astype(int)
        key_to_row = {(r["filename"], int(r["start_sec"])): r for _, r in pseudo_df.iterrows()}
        Y_ul = np.zeros((len(ul_meta), len(PRIMARY_LABELS)), dtype=np.float32)
        for i, row in ul_meta.iterrows():
            key = (row["filename"], int(row["start_sec"]))
            if key in key_to_row:
                pr = key_to_row[key]
                for j, lbl in enumerate(PRIMARY_LABELS):
                    if lbl in pr.index:
                        v = pr[lbl]
                        Y_ul[i, j] = float(v) if not pd.isna(v) else 0.0
        Y_ul = np.nan_to_num(Y_ul, nan=0.0)
        unlabeled_sc_meta = ul_meta
        Y_unlabeled_SC = Y_ul
        print(f"Unlabeled SC pseudo labels: {len(ul_meta)} windows  "
              f"positives(>0.5)={int((Y_ul > 0.5).any(axis=1).sum())}  "
              f"from {PSEUDO_UNLABELED_SC_CSV}")

    # Fold assignment — focal: StratifiedKFold by species
    af = audio_cache_meta.drop_duplicates("original_idx").reset_index(drop=True)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    af["fold"] = -1
    for fold, (_, vi) in enumerate(skf.split(af, af["primary_label"])):
        af.loc[vi, "fold"] = fold
    audio_cache_meta = audio_cache_meta.merge(af[["original_idx","fold"]], on="original_idx", how="left")

    # Fold assignment — soundscape: GroupKFold by file
    sc_files = sc_cache_meta[["filename","site"]].drop_duplicates().reset_index(drop=True)
    gkf = GroupKFold(n_splits=N_FOLDS)
    sc_files["fold"] = -1
    for fold, (_, vi) in enumerate(gkf.split(sc_files, groups=sc_files["filename"])):
        sc_files.loc[sc_files.index[vi], "fold"] = fold
    sc_cache_meta["fold"] = sc_cache_meta["filename"].map(
        dict(zip(sc_files["filename"], sc_files["fold"]))).fillna(-1).astype(int)

    # Upsample rare species
    counts = audio_cache_meta["primary_label"].value_counts()
    rare = counts[counts < MIN_SAMPLE].index
    extra = [audio_cache_meta[audio_cache_meta["primary_label"]==sp]
             for sp in rare
             for _ in range(int(np.ceil(MIN_SAMPLE / (audio_cache_meta["primary_label"]==sp).sum())) - 1)]
    if extra:
        audio_cache_meta = pd.concat([audio_cache_meta] + extra, ignore_index=True)
    print(f"After upsample: {len(audio_cache_meta)} focal clips")

    # Non-S22 mask
    non_s22 = sc_cache_meta["site"].values != "S22"
    print(f"S22={( ~non_s22).sum()}  non-S22={non_s22.sum()}")

    # Soundscape file → cache_file mapping
    sc_file_meta = pd.read_csv(CACHE_DIR / "soundscape_file_meta.csv")
    sc_file_dict = dict(zip(sc_file_meta["filename"], sc_file_meta["cache_file"]))

    # Focal secondary labels
    focal_secondary = None
    if USE_FOCAL_SECONDARY:
        focal_secondary = {}
        for idx, row in train_df.iterrows():
            sec = row.get("secondary_labels", "")
            if pd.isna(sec) or sec in ("", "[]"): continue
            try:   sec_list = eval(sec) if isinstance(sec, str) else []
            except: continue
            valid = [s for s in sec_list if s in LABEL2IDX]
            if valid: focal_secondary[idx] = valid
        print(f"Secondary labels: {len(focal_secondary)} files")

    # SC MixUp pool (labeled windows only)
    sc_mixup_sources = []
    labeled_rows = []
    for i in range(len(sc_cache_meta)):
        row = sc_cache_meta.iloc[i]
        if Y_SC[i].sum() > 0:
            cf = sc_file_dict.get(row["filename"])
            if cf: labeled_rows.append({
                "filename": row["filename"], "start_sec": int(row["start_sec"]),
                "cache_file": cf, "label_idx": i, "fold": int(row.get("fold",-1)),
            })
    if labeled_rows:
        lm = pd.DataFrame(labeled_rows)
        sc_mixup_sources.append((CACHE_DIR, lm, Y_SC))
        print(f"SC MixUp pool: {len(lm)} windows")

    return (audio_cache_meta, sc_cache_meta, Y_SC, non_s22, taxon_masks,
            sc_file_dict, focal_secondary, sc_mixup_sources, LABEL2IDX,
            unlabeled_sc_meta, Y_unlabeled_SC)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

_BACKBONE_SHORT = {
    "tf_efficientnet_b0.ns_jft_in1k": "b0",
    "pvt_v2_b0":                       "pvt",
    "regnety_008.pycls_in1k":          "regy",
    "tf_efficientnetv2_s.in21k":       "effv2s",
    "hgnetv2_b0":                      "hgnetv2",
}

def main():
    global EPOCHS, BATCH, BACKBONE_NAME, DROP_PATH_RATE, ALPHA_DISTILL, \
           MIXUP_ALPHA, FOCAL_SC_MIXUP_ALPHA, WARMUP_EPOCHS, \
           FOCAL_GAMMA, LABEL_SMOOTH, LOSS_TYPE, KD_TYPE, INFONCE_TEMP, \
           KD_LAMBDA, KD_HARD_RATIO, KD_BANK_SIZE, KD_BANK_TOPK, \
           USE_TIME_SHIFT, USE_SPECMIX, USE_CUTMIX_FREQ, USE_SWA, SWA_START_EP, \
           INIT_CKPT, FREEZE_BACKBONE, EMA_DECAY, PATIENCE, \
           PSEUDO_UNLABELED_SC_CSV, UNLABELED_SC_CACHE_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold",           type=int,   default=0)
    ap.add_argument("--gpu",            type=int,   default=0)
    ap.add_argument("--out",            type=str,   default=None,
                    help="output dir (default: outputs/tucker-sed-<backbone>)")
    ap.add_argument("--epochs",         type=int,   default=EPOCHS)
    ap.add_argument("--batch",          type=int,   default=BATCH)
    ap.add_argument("--folds",          type=str,   default=None,
                    help="comma-separated folds, e.g. 0,1,2,3,4")
    ap.add_argument("--backbone",       type=str,   default=BACKBONE_NAME,
                    help="timm backbone name")
    ap.add_argument("--drop_path_rate", type=float, default=None,
                    help="drop path rate (default: 0.1)")
    ap.add_argument("--num_workers",    type=int,   default=4,
                    help="DataLoader num_workers (use 2 when running parallel GPUs)")
    # ── Ablation args ──────────────────────────────────────────────────────────
    ap.add_argument("--alpha_distill",  type=float, default=None,   help="distill loss weight (default 1.0)")
    ap.add_argument("--mixup_alpha",    type=float, default=None,   help="mixup alpha (default 0.4)")
    ap.add_argument("--warmup_epochs",  type=int,   default=None,   help="LR warmup epochs (default 2)")
    ap.add_argument("--focal_gamma",    type=float, default=None,   help="focal loss gamma (requires --loss_type focal)")
    ap.add_argument("--label_smooth",   type=float, default=None,   help="label smoothing ε for BCE (default 0.0)")
    ap.add_argument("--loss_type",      type=str,   default=None,   choices=["bce", "focal", "asl"])
    ap.add_argument("--kd_type",        type=str,   default=None,   choices=["mse", "cosine", "infonce", "triplet", "hard_infonce", "mse_hard_infonce", "membank_hard_infonce", "mse_membank_hard_infonce", "supcon", "mse_supcon", "supcon_membank", "mse_supcon_membank"])
    ap.add_argument("--infonce_temp",   type=float, default=None,   help="InfoNCE temperature (default 0.07)")
    ap.add_argument("--kd_lambda",      type=float, default=None,   help="InfoNCE weight in mse_hard_infonce combo (default 0.5)")
    ap.add_argument("--kd_hard_ratio",  type=float, default=None,   help="fraction of hardest negatives to keep (default 0.5)")
    ap.add_argument("--kd_bank_size",   type=int,   default=None,   help="memory bank size for membank_hard_infonce (default 4096)")
    ap.add_argument("--kd_bank_topk",   type=int,   default=None,   help="top-K hardest bank entries to sample per anchor (default 512)")
    ap.add_argument("--time_shift",     action="store_true",        help="waveform circular shift ±1s")
    ap.add_argument("--specmix",        action="store_true",        help="SpecMix spectrogram augmentation")
    ap.add_argument("--cutmix_freq",    action="store_true",        help="CutMix frequency-band augmentation")
    ap.add_argument("--use_swa",        action="store_true",        help="Stochastic Weight Averaging")
    ap.add_argument("--swa_start",      type=int,   default=30,     help="SWA start epoch (default 30)")
    ap.add_argument("--init_ckpt",      type=str,   default=None,   help="dir with fold{k}_best_ns22.pt to init from")
    ap.add_argument("--freeze_backbone", action="store_true",       help="freeze backbone, only train SED head")
    ap.add_argument("--ema_decay",      type=float, default=None,   help="EMA decay for NS (e.g. 0.99); 0 = disabled")
    ap.add_argument("--patience",       type=int,   default=None,   help="early stopping patience on ns22 (0 = disabled)")
    ap.add_argument("--pseudo_unlabeled_sc_csv", type=str, default=None,
                    help="CSV with pseudo labels for UNLABELED soundscapes (NS mode)")
    ap.add_argument("--unlabeled_sc_cache_dir",  type=str, default=None,
                    help="dir with unlabeled soundscape .pt cache + unlabeled_ss_cache_meta.csv")
    args = ap.parse_args()

    EPOCHS        = args.epochs
    BATCH         = args.batch
    BACKBONE_NAME = args.backbone

    if args.drop_path_rate is not None:
        DROP_PATH_RATE = args.drop_path_rate
    else:
        DROP_PATH_RATE = 0.1

    # Override ablation globals
    if args.alpha_distill  is not None: ALPHA_DISTILL  = args.alpha_distill
    if args.mixup_alpha    is not None:
        MIXUP_ALPHA = args.mixup_alpha; FOCAL_SC_MIXUP_ALPHA = args.mixup_alpha
    if args.warmup_epochs  is not None: WARMUP_EPOCHS  = args.warmup_epochs
    if args.focal_gamma    is not None: FOCAL_GAMMA    = args.focal_gamma
    if args.label_smooth   is not None: LABEL_SMOOTH   = args.label_smooth
    if args.loss_type      is not None: LOSS_TYPE      = args.loss_type
    if args.kd_type        is not None: KD_TYPE        = args.kd_type
    if args.infonce_temp   is not None: INFONCE_TEMP   = args.infonce_temp
    if args.kd_lambda      is not None: KD_LAMBDA      = args.kd_lambda
    if args.kd_hard_ratio  is not None: KD_HARD_RATIO  = args.kd_hard_ratio
    if args.kd_bank_size   is not None: KD_BANK_SIZE   = args.kd_bank_size
    if args.kd_bank_topk   is not None: KD_BANK_TOPK   = args.kd_bank_topk
    USE_TIME_SHIFT  = args.time_shift
    USE_SPECMIX     = args.specmix
    USE_CUTMIX_FREQ = args.cutmix_freq
    USE_SWA         = args.use_swa
    SWA_START_EP    = args.swa_start
    INIT_CKPT       = args.init_ckpt
    FREEZE_BACKBONE = args.freeze_backbone
    if args.ema_decay  is not None: EMA_DECAY  = args.ema_decay
    if args.patience   is not None: PATIENCE   = args.patience
    if args.pseudo_unlabeled_sc_csv is not None:
        PSEUDO_UNLABELED_SC_CSV = args.pseudo_unlabeled_sc_csv
    if args.unlabeled_sc_cache_dir is not None:
        UNLABELED_SC_CACHE_DIR = args.unlabeled_sc_cache_dir

    short = _BACKBONE_SHORT.get(BACKBONE_NAME, BACKBONE_NAME.split(".")[0])
    out_default = ROOT / f"outputs/tucker-sed-{short}"
    device  = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out) if args.out else out_default
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device:   {device}")
    print(f"Backbone: {BACKBONE_NAME}  drop_path={DROP_PATH_RATE}")
    print(f"EPOCHS={EPOCHS}  BATCH={BATCH}  LR={LR}  warmup={WARMUP_EPOCHS}ep")
    print(f"Loss: {LOSS_TYPE}  focal_γ={FOCAL_GAMMA}  label_smooth={LABEL_SMOOTH}")
    print(f"KD: {KD_TYPE}  α={ALPHA_DISTILL}  infonce_T={INFONCE_TEMP}  λ={KD_LAMBDA}  hard_ratio={KD_HARD_RATIO}  bank={KD_BANK_SIZE}  topk={KD_BANK_TOPK}")
    print(f"Aug: time_shift={USE_TIME_SHIFT}  specmix={USE_SPECMIX}  cutmix_freq={USE_CUTMIX_FREQ}")
    print(f"SWA: {USE_SWA}  start_ep={SWA_START_EP}  mixup_α={MIXUP_ALPHA}")
    print(f"Init ckpt: {INIT_CKPT}  freeze_backbone={FREEZE_BACKBONE}")
    print(f"Output: {out_dir}")
    print(f"Cache:  {CACHE_DIR}")
    if PSEUDO_UNLABELED_SC_CSV:
        print(f"NS unlabeled SC pseudo: {PSEUDO_UNLABELED_SC_CSV}")
        print(f"NS unlabeled SC cache:  {UNLABELED_SC_CACHE_DIR}")

    assert CACHE_DIR.exists(), f"Waveform cache not found: {CACHE_DIR}\n" \
        "Download: kaggle datasets download tuckerarrants/birdclef-2026-waveform-cache"

    (audio_cache_meta, sc_cache_meta, Y_SC, non_s22, taxon_masks,
     sc_file_dict, focal_secondary, sc_mixup_sources, _,
     unlabeled_sc_meta, Y_unlabeled_SC) = load_data()

    folds_to_run = [int(f) for f in args.folds.split(",")] \
                   if args.folds else [args.fold]

    for fold_k in folds_to_run:
        ckpt_path = out_dir / f"fold{fold_k}_best_ns22.pt"
        if ckpt_path.exists():
            print(f"\nFold {fold_k}: checkpoint exists, skipping")
            continue
        print(f"\n{'='*60}")
        print(f"FOLD {fold_k}")
        print(f"{'='*60}")
        best_ns22, best_macro = train_fold(
            fold_k, audio_cache_meta, sc_cache_meta, Y_SC, non_s22,
            taxon_masks, sc_file_dict, focal_secondary, sc_mixup_sources,
            device, out_dir, num_workers=args.num_workers,
            unlabeled_sc_meta=unlabeled_sc_meta, Y_unlabeled_SC=Y_unlabeled_SC,
        )
        print(f"\nFold {fold_k} done: ns22_auc={best_ns22:.4f}  macro_auc={best_macro:.4f}")


if __name__ == "__main__":
    main()
