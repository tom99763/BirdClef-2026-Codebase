"""
train_perch_ft.py — Multi-round Perch adapter fine-tuning chain.

Rounds:
  R0 — Supervised warm-up on 66 labeled soundscape files (739 windows)
       Loss: Focal BCE + SupCon
       OOF: GroupKFold-5 (file-level groups)
       Output: weights/perch_adapter_r0.pt

  R1 — + Prototype-filtered train audio (106k clips → kept where
         cosine_sim_to_class_prototype >= proto_sim_thr)
       Loss: Focal BCE + SupCon  (soundscape clips upweighted ×3)
       Output: weights/perch_adapter_r1.pt

  R2 — Domain alignment (soundscape ↔ train_audio)
       Loss: Focal BCE + SupCon + MMD per class
       Mixed batch: 50% labeled SS + 50% filtered train audio
       Output: weights/perch_adapter_r2.pt

  R3 — Mean Teacher SSL on unlabeled soundscapes (127k windows)
       Loss: Focal BCE (labeled) + Consistency (unlabeled, teacher/student)
       Output: weights/perch_adapter_r3.pt

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/train_perch_ft.py --config configs/perch_ft_r0.yaml
    CUDA_VISIBLE_DEVICES=1 python scripts/train_perch_ft.py --config configs/perch_ft_r1.yaml
    CUDA_VISIBLE_DEVICES=1 python scripts/train_perch_ft.py --config configs/perch_ft_r2.yaml
    CUDA_VISIBLE_DEVICES=1 python scripts/train_perch_ft.py --config configs/perch_ft_r3.yaml
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.perch_adapter import FocalBCELoss, MMDLoss, MeanTeacher, PerchAdapter, SupConLoss


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULTS = {
    "round": 0,
    "emb_dim": 1536,
    "bottleneck": 512,
    "n_blocks": 2,
    "use_seq_attn": False,
    "dropout": 0.15,
    "num_classes": 234,
    "lr": 3e-4,
    "weight_decay": 1e-3,
    "n_epochs": 80,
    "batch_size": 256,
    "patience": 15,
    "focal_gamma": 2.0,
    "supcon_weight": 0.3,
    "supcon_temp": 0.07,
    "mmd_weight": 0.1,
    "consist_weight": 0.5,
    "consist_noise_std": 0.05,
    "consist_mask_ratio": 0.1,
    "ema_alpha": 0.999,
    "ss_upsample": 3,            # soundscape clip repeat factor in R1/R2
    "proto_sim_thr": 0.50,       # cosine sim threshold for pseudo filter
    "proto_top_k": 5,            # top-k clips per class from train audio
    "prev_ckpt": None,           # path to previous round checkpoint
    "mixup_alpha": 0.4,          # Beta(alpha,alpha) mixup coefficient; 0=disable
    "mixup_pseudo_ratio": 0.5,   # fraction of each batch replaced by pseudo mixup
    # ── R1+ unified pseudo-label settings ──
    "r0_ckpt": "weights/perch_adapter_r0.pt",  # R0 = baseline anchor for pseudo blend
    "pseudo_blend": 0.5,         # blend = pseudo_blend*R0_preds + (1-pseudo_blend)*prev_preds
    "val_per_missing_class": 3,  # train audio clips per class absent from SS val
    "labeled_ss_npz": "outputs/perch_labeled_ss.npz",
    "unlabeled_ss_npz": "outputs/perch_emb_all_ss.npz",
    "train_audio_manifest": "outputs/embeddings_cache_nohuman/manifest.csv",
    "train_audio_emb_dir": "outputs/embeddings_cache_nohuman/train",
    "taxonomy_csv": "birdclef-2026/taxonomy.csv",
    "output_dir": "weights",
    "output_name": "perch_adapter_r0.pt",
    "log_dir": "outputs/logs",
}


# ── Datasets ──────────────────────────────────────────────────────────────────

class EmbDataset(Dataset):
    """Simple embedding → label dataset."""
    def __init__(self, embs: np.ndarray, labels: np.ndarray, weight: float = 1.0,
                 noise_std: float = 0.0, mask_ratio: float = 0.0):
        self.embs    = torch.tensor(embs,   dtype=torch.float32)
        self.labels  = torch.tensor(labels, dtype=torch.float32)
        self.weight  = weight
        self.noise_std   = noise_std
        self.mask_ratio  = mask_ratio

    def __len__(self):
        return len(self.embs)

    def __getitem__(self, idx):
        e = self.embs[idx]
        if self.noise_std > 0:
            e = e + torch.randn_like(e) * self.noise_std
        if self.mask_ratio > 0:
            mask = torch.rand(e.shape[0]) < self.mask_ratio
            e = e.clone()
            e[mask] = 0.0
        return e, self.labels[idx], self.weight


class MixedDataset(Dataset):
    """Interleave labeled SS (high weight) and train-audio pseudo (low weight)."""
    def __init__(self, ss_embs, ss_labels, pseudo_embs, pseudo_labels,
                 ss_weight=3.0, pseudo_weight=1.0):
        self.ss_embs   = torch.tensor(ss_embs,     dtype=torch.float32)
        self.ss_labels = torch.tensor(ss_labels,   dtype=torch.float32)
        self.ps_embs   = torch.tensor(pseudo_embs, dtype=torch.float32)
        self.ps_labels = torch.tensor(pseudo_labels, dtype=torch.float32)
        self.ss_weight     = ss_weight
        self.pseudo_weight = pseudo_weight
        self.n_ss     = len(ss_embs)
        self.n_pseudo = len(pseudo_embs)

    def __len__(self):
        return self.n_ss + self.n_pseudo

    def __getitem__(self, idx):
        if idx < self.n_ss:
            return self.ss_embs[idx], self.ss_labels[idx], self.ss_weight
        else:
            i = idx - self.n_ss
            return self.ps_embs[i], self.ps_labels[i], self.pseudo_weight


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_labeled_ss(cfg: dict):
    d = np.load(cfg["labeled_ss_npz"])
    return d["emb"], d["labels"], d["filenames"]


def load_unlabeled_ss(cfg: dict):
    d = np.load(cfg["unlabeled_ss_npz"])
    return d["emb"], d["filenames"]


def load_train_audio_embeddings(cfg: dict, class_list: list[str]):
    """Load pre-extracted Perch 1536-dim embeddings for train audio."""
    manifest = pd.read_csv(cfg["train_audio_manifest"])
    # Keep only train split (exclude holdout)
    manifest = manifest[manifest["split"] == "train"].copy()
    emb_dir = Path(cfg["train_audio_emb_dir"])

    # Build class → index mapping
    cls2idx = {c: i for i, c in enumerate(class_list)}

    embs, labels, file_labels_list = [], [], []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Loading train audio emb"):
        npy_path = Path(row["npy_path"])
        if not npy_path.exists():
            continue
        e = np.load(npy_path)          # (1536,)
        cls = str(row["label"])
        if cls not in cls2idx:
            continue
        lbl = np.zeros(len(class_list), dtype=np.float32)
        lbl[cls2idx[cls]] = 1.0
        embs.append(e)
        labels.append(lbl)

    return np.stack(embs), np.stack(labels)


def build_prototypes(embs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-class mean embedding prototype (for pseudo filtering)."""
    C = labels.shape[1]
    D = embs.shape[1]
    protos = np.zeros((C, D), dtype=np.float32)
    has_proto = np.zeros(C, dtype=bool)
    for c in range(C):
        mask = labels[:, c] > 0.5
        if mask.sum() > 0:
            protos[c] = embs[mask].mean(0)
            has_proto[c] = True
    return protos, has_proto


def filter_pseudo_by_prototype(
    pseudo_embs: np.ndarray,
    pseudo_labels: np.ndarray,
    prototypes: np.ndarray,
    has_proto: np.ndarray,
    sim_thr: float = 0.5,
    top_k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep train-audio clips where cosine_sim_to_assigned_class_prototype >= sim_thr.
    For each clip, the "assigned class" is argmax of its single-hot label.
    Also caps at top_k per class (sorted by descending sim).
    """
    # L2 normalise
    emb_n  = pseudo_embs  / (np.linalg.norm(pseudo_embs,  axis=1, keepdims=True) + 1e-8)
    proto_n = prototypes   / (np.linalg.norm(prototypes,   axis=1, keepdims=True) + 1e-8)

    keep_embs, keep_labels = [], []
    per_class_sims = {}  # cls → list of (sim, idx)

    for i, lbl in enumerate(pseudo_labels):
        c = int(lbl.argmax())
        if not has_proto[c]:
            continue
        sim = float(emb_n[i] @ proto_n[c])
        if sim >= sim_thr:
            per_class_sims.setdefault(c, []).append((sim, i))

    for c, items in per_class_sims.items():
        items.sort(key=lambda x: -x[0])
        for _, i in items[:top_k]:
            keep_embs.append(pseudo_embs[i])
            keep_labels.append(pseudo_labels[i])

    if not keep_embs:
        return np.zeros((0, pseudo_embs.shape[1]), dtype=np.float32), \
               np.zeros((0, pseudo_labels.shape[1]), dtype=np.float32)

    return np.stack(keep_embs), np.stack(keep_labels)


@torch.no_grad()
def generate_pseudo_labels(model: PerchAdapter, embs: np.ndarray,
                            device: torch.device, batch: int = 512) -> np.ndarray:
    """
    Run prev-round adapter forward pass on embeddings → soft pseudo labels (sigmoid probs).
    Returns: np.ndarray (N, num_classes) float32 in [0, 1].
    """
    model.eval()
    all_probs = []
    for i in range(0, len(embs), batch):
        e = torch.tensor(embs[i:i + batch], dtype=torch.float32, device=device)
        logits, _, _ = model(e)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(all_probs, axis=0).astype(np.float32)


def build_comprehensive_valset(
    ss_embs: np.ndarray, ss_labels: np.ndarray,
    ta_embs: np.ndarray, ta_labels: np.ndarray,
    k: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a val set covering all num_classes.

    - All labeled SS windows (covers SS classes, hard labels)
    - For each class NOT present in SS: take up to k clips from train audio

    Returns
    -------
    val_embs   : (V, D)  all val embeddings
    val_labels : (V, C)  hard one-hot labels for AUC evaluation
    train_mask : (N,)    boolean mask over ta_embs rows → True = keep for training
    """
    ss_covered = ss_labels.sum(0) > 0          # (C,) classes already in SS val
    num_classes = ss_labels.shape[1]

    val_ta_idx: list[int] = []
    for c in range(num_classes):
        if ss_covered[c]:
            continue
        cls_idx = np.where(ta_labels[:, c] > 0.5)[0]
        if len(cls_idx) == 0:
            continue
        val_ta_idx.extend(cls_idx[:k].tolist())

    val_ta_idx = sorted(set(val_ta_idx))
    train_mask = np.ones(len(ta_embs), dtype=bool)
    train_mask[val_ta_idx] = False

    val_embs   = np.concatenate([ss_embs,   ta_embs[val_ta_idx]],   axis=0)
    val_labels = np.concatenate([ss_labels, ta_labels[val_ta_idx]], axis=0)

    return val_embs, val_labels, train_mask


def load_r0_model(cfg: dict, device: torch.device) -> PerchAdapter:
    """Load the R0 adapter (baseline anchor) for pseudo-label generation."""
    r0_path = cfg.get("r0_ckpt") or "weights/perch_adapter_r0.pt"
    m = PerchAdapter(
        emb_dim=cfg["emb_dim"], bottleneck=cfg["bottleneck"],
        num_classes=cfg["num_classes"], dropout=0.0,
        n_blocks=cfg["n_blocks"],
    ).to(device)
    m.load_state_dict(torch.load(r0_path, map_location=device)["model_state"])
    m.eval()
    return m


# ── Training ──────────────────────────────────────────────────────────────────

def make_optimizer(model: PerchAdapter, cfg: dict):
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )


def make_scheduler(optimizer, n_epochs: int):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)


def train_epoch(model, loader, optimizer, focal_loss, supcon_loss, cfg, device,
                mmd_loss=None, ss_batch=None, domain_round=False,
                pseudo_loader=None):
    """
    pseudo_loader : DataLoader over (pseudo_emb, pseudo_soft_label, weight)
                    When provided, each batch is mixed with a pseudo batch via
                    Beta(mixup_alpha, mixup_alpha) interpolation in embedding+label space.
    """
    model.train()
    total_loss = 0.0
    n = 0

    mixup_alpha = float(cfg.get("mixup_alpha", 0.4))
    # pseudo_loader is a DataLoader; manage iter cycle internally
    _pseudo_iter = [iter(pseudo_loader)] if pseudo_loader is not None else [None]

    def _next_pseudo():
        """Cycle through pseudo_loader, reinit on exhaustion."""
        if _pseudo_iter[0] is None:
            return None
        try:
            return next(_pseudo_iter[0])
        except StopIteration:
            _pseudo_iter[0] = iter(pseudo_loader)
            try:
                return next(_pseudo_iter[0])
            except StopIteration:
                return None

    for emb, labels, weights in loader:
        emb, labels, weights = emb.to(device), labels.to(device), weights.to(device)

        # ── Cross-round mixup with pseudo labels ──────────────────────────────
        if pseudo_loader is not None and mixup_alpha > 0:
            p_batch = _next_pseudo()
            if p_batch is not None:
                p_emb, p_lbl, _ = p_batch
                p_emb = p_emb.to(device)
                p_lbl = p_lbl.to(device)
                b = min(emb.shape[0], p_emb.shape[0])
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                lam = max(lam, 1.0 - lam)   # labeled side always majority
                emb     = lam * emb[:b]     + (1.0 - lam) * p_emb[:b]
                labels  = lam * labels[:b]  + (1.0 - lam) * p_lbl[:b]
                weights = weights[:b]

        logits, adapted_emb, proj_emb = model(emb)

        # Weighted focal BCE
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(-1)
        cls_loss = (bce * weights).mean()

        # SupCon (only on windows that have ≥1 positive label)
        has_label = labels.sum(1) > 0
        sc_loss = torch.tensor(0.0, device=device)
        if has_label.sum() >= 2:
            sc_loss = supcon_loss(proj_emb[has_label], labels[has_label])

        loss = (1 - cfg["supcon_weight"]) * cls_loss + cfg["supcon_weight"] * sc_loss

        # MMD domain alignment (R2): align SS vs train-audio in batch
        if domain_round and mmd_loss is not None and ss_batch is not None:
            try:
                ss_emb_b = next(ss_batch)[0].to(device)
            except StopIteration:
                ss_emb_b = None
            if ss_emb_b is None:
                continue
            if ss_emb_b.shape[0] >= 2 and adapted_emb.shape[0] >= 2:
                _, ss_adapted, _ = model(ss_emb_b)
                mmd = mmd_loss(adapted_emb, ss_adapted)
                loss = loss + cfg["mmd_weight"] * mmd

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * emb.shape[0]
        n += emb.shape[0]

    return total_loss / max(n, 1)


@torch.no_grad()
def eval_epoch(model, embs_np, labels_np, device, batch=512):
    model.eval()
    all_logits = []
    for i in range(0, len(embs_np), batch):
        e = torch.tensor(embs_np[i:i+batch], device=device)
        logits, _, _ = model(e)
        all_logits.append(logits.cpu().numpy())
    preds = torch.sigmoid(torch.tensor(np.concatenate(all_logits))).numpy()
    valid = labels_np.sum(0) > 0
    if valid.sum() == 0:
        return 0.0
    return roc_auc_score(labels_np[:, valid], preds[:, valid], average="macro")


def train_full(model, train_embs, train_labels, val_embs, val_labels, cfg, device,
               mmd_loss=None, ss_embs=None, domain_round=False,
               pseudo_embs=None, pseudo_labels=None):
    """
    Train for n_epochs with early stopping.

    pseudo_embs / pseudo_labels : soft pseudo labels from previous round.
        When provided, each training batch is mixed with a pseudo batch via
        Beta(mixup_alpha, mixup_alpha) mixup in embedding+label space.
    """
    focal = FocalBCELoss(gamma=cfg["focal_gamma"])
    supcon = SupConLoss(temperature=cfg["supcon_temp"])
    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg["n_epochs"])

    weights = np.ones(len(train_embs), dtype=np.float32)
    dataset = EmbDataset(train_embs, train_labels)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)

    # Pseudo-label DataLoader for cross-round mixup
    pseudo_dataloader = None
    if pseudo_embs is not None and pseudo_labels is not None and len(pseudo_embs) > 0:
        pseudo_ds = EmbDataset(pseudo_embs, pseudo_labels)
        pseudo_dataloader = DataLoader(pseudo_ds, batch_size=cfg["batch_size"],
                                       shuffle=True, num_workers=0)

    ss_loader = None
    ss_dataloader = None
    if domain_round and ss_embs is not None:
        ss_ds = EmbDataset(ss_embs, np.zeros((len(ss_embs), cfg["num_classes"]), np.float32))
        ss_dataloader = DataLoader(ss_ds, batch_size=cfg["batch_size"] // 2,
                                   shuffle=True, num_workers=0)
        ss_loader = iter(ss_dataloader)

    best_auc = 0.0
    best_state = None
    patience_cnt = 0
    history = []

    for epoch in range(cfg["n_epochs"]):
        # Refresh ss_loader at each epoch so it never gets exhausted mid-epoch
        if ss_dataloader is not None:
            ss_loader = iter(ss_dataloader)
        # Pass DataLoader directly — train_epoch manages its own cycling iterator
        tr_loss = train_epoch(
            model, loader, optimizer, focal, supcon, cfg, device,
            mmd_loss=mmd_loss, ss_batch=ss_loader, domain_round=domain_round,
            pseudo_loader=pseudo_dataloader,
        )
        val_auc = eval_epoch(model, val_embs, val_labels, device)
        scheduler.step()

        history.append({"epoch": epoch, "tr_loss": tr_loss, "val_auc": val_auc})
        print(f"  Epoch {epoch+1:3d}/{cfg['n_epochs']} | loss={tr_loss:.4f} | val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= cfg["patience"]:
                print(f"  Early stop at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_auc, history


# ── Mean Teacher training (R3) ────────────────────────────────────────────────

def train_r3_mean_teacher(model, labeled_embs, labeled_labels,
                           unlabeled_embs, val_embs, val_labels, cfg, device,
                           pseudo_embs=None, pseudo_labels=None):
    """
    Mean Teacher self-training on unlabeled soundscape windows.
    Student and teacher both output logits;
    consistency loss = MSE(student_logit, teacher_logit.detach()).

    pseudo_embs / pseudo_labels : R2 soft pseudo labels on unlabeled SS.
        When provided, each labeled batch is also mixed with pseudo via Beta mixup.
    """
    mt = MeanTeacher(model, alpha=cfg["ema_alpha"])
    focal = FocalBCELoss(gamma=cfg["focal_gamma"])
    supcon = SupConLoss(temperature=cfg["supcon_temp"])
    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg["n_epochs"])

    mixup_alpha = float(cfg.get("mixup_alpha", 0.4))
    # Build pseudo DataLoader for mixup within Mean Teacher
    pseudo_dl_r3 = None
    if pseudo_embs is not None and pseudo_labels is not None and mixup_alpha > 0:
        pseudo_ds_r3 = EmbDataset(pseudo_embs, pseudo_labels)
        pseudo_dl_r3 = DataLoader(pseudo_ds_r3, batch_size=cfg["batch_size"],
                                   shuffle=True, num_workers=0)

    labeled_ds   = EmbDataset(labeled_embs, labeled_labels)
    unlabeled_ds = EmbDataset(
        unlabeled_embs,
        np.zeros((len(unlabeled_embs), cfg["num_classes"]), dtype=np.float32),
        noise_std=cfg["consist_noise_std"],
        mask_ratio=cfg["consist_mask_ratio"],
    )

    labeled_loader   = DataLoader(labeled_ds,   batch_size=cfg["batch_size"],
                                   shuffle=True, num_workers=2, drop_last=True)
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=cfg["batch_size"],
                                   shuffle=True, num_workers=2, drop_last=True)

    best_auc, best_state = 0.0, None
    patience_cnt = 0

    for epoch in range(cfg["n_epochs"]):
        model.train()
        mt.teacher.train()
        total_loss = 0.0; n = 0

        unl_iter = iter(unlabeled_loader)
        _p3_iter = [iter(pseudo_dl_r3)] if pseudo_dl_r3 is not None else [None]

        def _next_p3():
            if _p3_iter[0] is None:
                return None
            try:
                return next(_p3_iter[0])
            except StopIteration:
                _p3_iter[0] = iter(pseudo_dl_r3)
                try:
                    return next(_p3_iter[0])
                except StopIteration:
                    return None

        for emb, labels, weights in labeled_loader:
            emb, labels = emb.to(device), labels.to(device)

            # ── Pseudo mixup within labeled batch ──────────────────────────
            if pseudo_dl_r3 is not None and mixup_alpha > 0:
                p3 = _next_p3()
                if p3 is not None:
                    p_emb_r3, p_lbl_r3, _ = p3
                    p_emb_r3 = p_emb_r3.to(device)
                    p_lbl_r3 = p_lbl_r3.to(device)
                    b3 = min(emb.shape[0], p_emb_r3.shape[0])
                    lam3 = float(np.random.beta(mixup_alpha, mixup_alpha))
                    lam3 = max(lam3, 1.0 - lam3)
                    emb    = lam3 * emb[:b3]    + (1.0 - lam3) * p_emb_r3[:b3]
                    labels = lam3 * labels[:b3] + (1.0 - lam3) * p_lbl_r3[:b3]

            # Labeled loss
            logits, _, proj = model(emb)
            lbl_loss = focal(logits, labels)
            has_lbl = labels.sum(1) > 0
            if has_lbl.sum() >= 2:
                lbl_loss = (1 - cfg["supcon_weight"]) * lbl_loss + \
                            cfg["supcon_weight"] * supcon(proj[has_lbl], labels[has_lbl])

            # Consistency loss on unlabeled
            consist_loss = torch.tensor(0.0, device=device)
            try:
                unl_emb, _, _ = next(unl_iter)
            except StopIteration:
                unl_iter = iter(unlabeled_loader)
                unl_emb, _, _ = next(unl_iter)

            unl_emb = unl_emb.to(device)
            student_logits, _, _ = model(unl_emb)
            with torch.no_grad():
                # Teacher uses clean embedding (no noise augmentation)
                clean_emb = unl_emb  # already noised in dataset, teacher gets same
                teacher_logits, _, _ = mt.teacher(clean_emb)
            consist_loss = F.mse_loss(student_logits, teacher_logits.detach())

            loss = lbl_loss + cfg["consist_weight"] * consist_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            mt.update(model)  # EMA update

            total_loss += loss.item() * emb.shape[0]
            n += emb.shape[0]

        val_auc = eval_epoch(model, val_embs, val_labels, device)
        scheduler.step()
        tr_loss = total_loss / max(n, 1)
        print(f"  R3 Epoch {epoch+1:3d}/{cfg['n_epochs']} | "
              f"loss={tr_loss:.4f} | val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= cfg["patience"]:
                print(f"  Early stop R3 at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)

    return best_auc


# ── OOF Evaluation ────────────────────────────────────────────────────────────

def oof_eval(embs, labels, filenames, cfg, device):
    """GroupKFold-5 OOF AUC (file-level groups)."""
    groups = filenames
    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros_like(labels)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(embs, labels, groups)):
        print(f"\n── OOF fold {fold+1}/5 ──")
        model_f = PerchAdapter(
            emb_dim=cfg["emb_dim"], bottleneck=cfg["bottleneck"],
            num_classes=cfg["num_classes"], dropout=cfg["dropout"],
            n_blocks=cfg["n_blocks"], use_seq_attn=cfg["use_seq_attn"],
        ).to(device)

        tr_e, tr_l = embs[tr_idx], labels[tr_idx]
        va_e, va_l = embs[va_idx], labels[va_idx]

        best_auc, _ = train_full(model_f, tr_e, tr_l, va_e, va_l, cfg, device)
        print(f"  Fold {fold+1} best val AUC: {best_auc:.4f}")

        # Predict on val
        with torch.no_grad():
            model_f.eval()
            preds = []
            for i in range(0, len(va_e), 512):
                e = torch.tensor(va_e[i:i+512], device=device)
                logits, _, _ = model_f(e)
                preds.append(torch.sigmoid(logits).cpu().numpy())
            oof_preds[va_idx] = np.concatenate(preds)

        del model_f; torch.cuda.empty_cache()

    valid = labels.sum(0) > 0
    oof_auc = roc_auc_score(labels[:, valid], oof_preds[:, valid], average="macro")
    return oof_auc, oof_preds


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        user_cfg = yaml.safe_load(f)

    cfg = {**DEFAULTS, **user_cfg}
    rnd = cfg["round"]

    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(cfg["log_dir"], exist_ok=True)
    log_path = Path(cfg["log_dir"]) / f"perch_ft_r{rnd}.log"
    out_path = Path(cfg["output_dir"]) / cfg["output_name"]

    def log(msg):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] [R{rnd}] {msg}"
        print(line)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    log(f"Round: {rnd}")

    # ── Load labeled soundscapes ──
    ss_embs, ss_labels, ss_filenames = load_labeled_ss(cfg)
    log(f"Labeled SS: {ss_embs.shape}, {ss_labels.sum():.0f} positives")

    # ── Load taxonomy for class list ──
    taxonomy = pd.read_csv(cfg["taxonomy_csv"])
    class_list = taxonomy["primary_label"].tolist()
    assert len(class_list) == cfg["num_classes"], \
        f"num_classes mismatch: {len(class_list)} vs {cfg['num_classes']}"

    # ── Round-specific data prep ──
    if rnd == 0:
        log("R0: Labeled SS only")
        log(f"  Running OOF evaluation on {len(ss_embs)} windows...")

        oof_auc, oof_preds = oof_eval(ss_embs, ss_labels, ss_filenames, cfg, device)
        log(f"  OOF AUC (R0): {oof_auc:.4f}")

        # Final model trained on all data (no val split; use full SS)
        log("  Training final model on all labeled SS...")
        model = PerchAdapter(
            emb_dim=cfg["emb_dim"], bottleneck=cfg["bottleneck"],
            num_classes=cfg["num_classes"], dropout=cfg["dropout"],
            n_blocks=cfg["n_blocks"], use_seq_attn=cfg["use_seq_attn"],
        ).to(device)

        # Hold out last 10% files as val for early stopping
        uniq_files = np.unique(ss_filenames)
        n_val = max(1, int(0.10 * len(uniq_files)))
        val_files = set(uniq_files[-n_val:])
        va_mask = np.array([fn in val_files for fn in ss_filenames])
        tr_embs, va_embs = ss_embs[~va_mask], ss_embs[va_mask]
        tr_lbls, va_lbls = ss_labels[~va_mask], ss_labels[va_mask]

        best_auc, history = train_full(model, tr_embs, tr_lbls, va_embs, va_lbls, cfg, device)
        log(f"  Final model best val AUC: {best_auc:.4f}")

    elif rnd >= 1:
        # ══════════════════════════════════════════════════════════════════════
        # R1, R2, R3 ... — Unified pseudo-label self-training
        #
        # Strategy:
        #   1. Load R0 (baseline anchor) + prev round model
        #   2. Generate soft pseudo labels on ALL train audio:
        #        pseudo = pseudo_blend * R0_preds + (1-pseudo_blend) * prev_preds
        #   3. Build comprehensive val set:
        #        - ALL labeled SS (75 classes covered, hard labels)
        #        - + k train audio clips per class absent from SS (cover all 234)
        #   4. Train on remaining train audio (minus val) with pseudo labels
        #   5. Val AUC on comprehensive set covers all 234 classes → honest metric
        # ══════════════════════════════════════════════════════════════════════
        log(f"R{rnd}: Pseudo-label self-training (blend of R0 + R{rnd-1})")

        # ── Load train audio embeddings ──
        log("  Loading train audio embeddings...")
        ta_embs, ta_labels = load_train_audio_embeddings(cfg, class_list)
        log(f"  Train audio: {ta_embs.shape[0]:,} clips × {ta_embs.shape[1]} dim")

        # ── Generate pseudo labels ──
        log("  Generating R0 baseline predictions on train audio...")
        r0_model = load_r0_model(cfg, device)
        r0_preds = generate_pseudo_labels(r0_model, ta_embs, device)
        del r0_model; torch.cuda.empty_cache()

        prev_ckpt_path = cfg["prev_ckpt"]
        log(f"  Generating R{rnd-1} predictions on train audio (prev: {prev_ckpt_path})...")
        prev_model = PerchAdapter(
            emb_dim=cfg["emb_dim"], bottleneck=cfg["bottleneck"],
            num_classes=cfg["num_classes"], dropout=0.0,
            n_blocks=cfg["n_blocks"],
        ).to(device)
        prev_model.load_state_dict(torch.load(prev_ckpt_path, map_location=device)["model_state"])
        prev_model.eval()
        prev_preds = generate_pseudo_labels(prev_model, ta_embs, device)
        del prev_model; torch.cuda.empty_cache()

        blend = float(cfg.get("pseudo_blend", 0.5))
        mixed_pseudo = blend * r0_preds + (1.0 - blend) * prev_preds
        log(f"  Mixed pseudo (blend={blend}): mean_max_prob={mixed_pseudo.max(1).mean():.3f}  "
            f"mean_sum={mixed_pseudo.sum(1).mean():.2f}")

        # ── Comprehensive val set ──
        k_val = int(cfg.get("val_per_missing_class", 3))
        val_embs, val_labels, train_mask = build_comprehensive_valset(
            ss_embs, ss_labels, ta_embs, ta_labels, k=k_val
        )
        val_classes = int((val_labels.sum(0) > 0).sum())
        log(f"  Val set: {len(val_embs):,} windows | classes covered: {val_classes}/{cfg['num_classes']}")
        log(f"    SS windows: {len(ss_embs)}  |  train audio val: {len(val_embs)-len(ss_embs)}")

        # ── Train set: remaining train audio with pseudo labels ──
        train_embs   = ta_embs[train_mask]
        train_labels = mixed_pseudo[train_mask]
        log(f"  Train set: {len(train_embs):,} clips (train audio minus val)")

        # ── Model init from prev checkpoint ──
        model = PerchAdapter(
            emb_dim=cfg["emb_dim"], bottleneck=cfg["bottleneck"],
            num_classes=cfg["num_classes"], dropout=cfg["dropout"],
            n_blocks=cfg["n_blocks"],
        ).to(device)
        ckpt = torch.load(prev_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        log(f"  Loaded prev ckpt: {prev_ckpt_path}")

        # Disable SupCon for soft pseudo labels (no hard class boundaries)
        cfg_r = {**cfg, "supcon_weight": 0.0}

        best_auc, history = train_full(model, train_embs, train_labels,
                                       val_embs, val_labels, cfg_r, device)
        log(f"  R{rnd} best val AUC (all {val_classes} classes): {best_auc:.4f}")

    else:
        raise ValueError(f"Unknown round: {rnd}")

    # ── Save checkpoint ──
    save_dict = {
        "model_state": model.state_dict(),
        "cfg": cfg,
        "round": rnd,
    }
    torch.save(save_dict, out_path)
    log(f"Saved → {out_path}")
    log(f"Parameters: {model.count_parameters():,}")


if __name__ == "__main__":
    main()
