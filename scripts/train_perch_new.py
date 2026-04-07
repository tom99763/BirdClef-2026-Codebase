"""
train_perch_new.py — Next-generation Perch adapter experiments.

Experiments (set via config `exp_type`):
  A) proto_head  — ProtoNet head replaces MLP; class prototype + cosine sim
                   (Bird-MAE paper: +37pp over linear probe on frozen features)
  B) protocl     — ProtoNet head + ProtoCLR domain-invariant contrastive loss
                   (arXiv:2409.08589; pulls SS and train_audio same-class together)
  C) fixmatch    — ProtoNet head + FixMatch consistency on unlabeled SS embeddings
                   (BirdCLEF 2025 2nd-place style semi-supervised)

All experiments:
  - Frozen Perch backbone (never touched)
  - Val set = ALL labeled SS + train audio covering all 234 classes
  - Weights saved to weights/perch_new/

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/train_perch_new.py --config configs/perch_new/proto_head.yaml
    CUDA_VISIBLE_DEVICES=1 python scripts/train_perch_new.py --config configs/perch_new/protocl.yaml
    CUDA_VISIBLE_DEVICES=1 python scripts/train_perch_new.py --config configs/perch_new/fixmatch.yaml
"""

from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.perch_proto import FixMatchEmbLoss, ProtoCLRLoss, ProtoHead

# Re-use data loaders from train_perch_ft
from scripts.train_perch_ft import (
    build_comprehensive_valset,
    load_labeled_ss,
    load_train_audio_embeddings,
    load_unlabeled_ss,
)

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULTS = {
    "exp_type": "proto_head",   # proto_head | protocl | fixmatch
    "emb_dim": 1536,
    "proj_dim": 512,
    "num_classes": 234,
    "n_blocks": 2,
    "dropout": 0.10,
    "temperature": 0.05,        # ProtoHead cosine temperature
    "lr": 3e-4,
    "weight_decay": 1e-3,
    "n_epochs": 80,
    "batch_size": 256,
    "patience": 15,
    # Losses
    "focal_gamma": 2.0,
    "protocl_weight": 0.3,      # B: ProtoCLR loss weight
    "fixmatch_weight": 0.5,     # C: FixMatch consistency weight
    "fixmatch_conf_thr": 0.70,  # C: confidence threshold for pseudo labels
    "fixmatch_noise_w": 0.02,
    "fixmatch_noise_s": 0.08,
    "fixmatch_mask_ratio": 0.15,
    # Val / data
    "val_per_missing_class": 3,
    "init_ckpt": None,          # optional warm-start from existing adapter
    # Paths
    "labeled_ss_npz": "outputs/perch_labeled_ss.npz",
    "unlabeled_ss_npz": "outputs/perch_emb_all_ss.npz",
    "train_audio_manifest": "outputs/embeddings_cache_nohuman/manifest.csv",
    "train_audio_emb_dir": "outputs/embeddings_cache_nohuman/train",
    "taxonomy_csv": "birdclef-2026/taxonomy.csv",
    "output_dir": "weights/perch_new",
    "output_name": "proto_head_r0.pt",
    "log_dir": "outputs/perch_new/logs",
}


# ── Dataset ───────────────────────────────────────────────────────────────────

class EmbDataset(Dataset):
    def __init__(self, embs: np.ndarray, labels: np.ndarray,
                 noise_std: float = 0.0, mask_ratio: float = 0.0):
        self.embs   = torch.tensor(embs,   dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.noise_std  = noise_std
        self.mask_ratio = mask_ratio

    def __len__(self): return len(self.embs)

    def __getitem__(self, idx):
        e = self.embs[idx].clone()
        if self.noise_std > 0:
            e += torch.randn_like(e) * self.noise_std
        if self.mask_ratio > 0:
            mask = torch.rand(e.shape[0]) < self.mask_ratio
            e[mask] = 0.0
        return e, self.labels[idx]


# ── Loss ──────────────────────────────────────────────────────────────────────

class FocalBCE(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt  = torch.exp(-bce)
        return ((1 - pt) ** self.gamma * bce).mean()


# ── Eval ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_auc(model: ProtoHead, embs_np: np.ndarray, labels_np: np.ndarray,
             device: torch.device, batch: int = 512) -> float:
    model.eval()
    all_logits = []
    for i in range(0, len(embs_np), batch):
        e = torch.tensor(embs_np[i:i+batch], device=device)
        logits, _ = model(e)
        all_logits.append(logits.cpu().numpy())
    preds = torch.sigmoid(torch.tensor(np.concatenate(all_logits))).numpy()
    valid = labels_np.sum(0) > 0
    if valid.sum() == 0:
        return 0.0
    return roc_auc_score(labels_np[:, valid], preds[:, valid], average="macro")


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model: ProtoHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    focal_loss: FocalBCE,
    device: torch.device,
    cfg: dict,
    # optional extras for B/C
    ta_loader: DataLoader | None = None,    # for ProtoCLR
    unl_loader: DataLoader | None = None,   # for FixMatch
    protocl_loss: ProtoCLRLoss | None = None,
    fixmatch_loss: FixMatchEmbLoss | None = None,
    ta_iter_state: list | None = None,      # mutable [iter] for cycling
    unl_iter_state: list | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0

    for emb, labels in loader:
        emb, labels = emb.to(device), labels.to(device)
        logits, proj = model(emb)

        loss = focal_loss(logits, labels)

        # ── B: ProtoCLR domain-invariant contrastive ──────────────────────
        if protocl_loss is not None and ta_loader is not None:
            if ta_iter_state[0] is None:
                ta_iter_state[0] = iter(ta_loader)
            try:
                ta_emb, ta_lbl = next(ta_iter_state[0])
            except StopIteration:
                ta_iter_state[0] = iter(ta_loader)
                ta_emb, ta_lbl = next(ta_iter_state[0])
            ta_emb = ta_emb.to(device)
            ta_lbl = ta_lbl.to(device)
            _, ta_proj = model(ta_emb)
            cl_loss = protocl_loss(proj, labels, ta_proj, ta_lbl)
            loss = loss + float(cfg["protocl_weight"]) * cl_loss

        # ── C: FixMatch consistency on unlabeled SS ───────────────────────
        if fixmatch_loss is not None and unl_loader is not None:
            if unl_iter_state[0] is None:
                unl_iter_state[0] = iter(unl_loader)
            try:
                unl_emb, _ = next(unl_iter_state[0])
            except StopIteration:
                unl_iter_state[0] = iter(unl_loader)
                unl_emb, _ = next(unl_iter_state[0])
            unl_emb = unl_emb.to(device)
            fm_loss, _ = fixmatch_loss(model, unl_emb)
            loss = loss + float(cfg["fixmatch_weight"]) * fm_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * emb.shape[0]
        n += emb.shape[0]

    return total_loss / max(n, 1)


def train(
    model: ProtoHead,
    train_embs: np.ndarray,
    train_labels: np.ndarray,
    val_embs: np.ndarray,
    val_labels: np.ndarray,
    cfg: dict,
    device: torch.device,
    ta_embs: np.ndarray | None = None,
    ta_labels: np.ndarray | None = None,
    unl_embs: np.ndarray | None = None,
    log_fn=print,
) -> float:
    focal = FocalBCE(gamma=cfg["focal_gamma"])
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(cfg["lr"]),
                                  weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["n_epochs"], eta_min=1e-6)

    ds = EmbDataset(train_embs, train_labels)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)

    # ── extra loaders for B / C ──
    exp = cfg["exp_type"]
    ta_loader = unl_loader = None
    protocl_loss_fn = fixmatch_loss_fn = None
    ta_iter_state = [None]
    unl_iter_state = [None]

    if exp in ("protocl",) and ta_embs is not None:
        ta_ds = EmbDataset(ta_embs, ta_labels)
        ta_loader = DataLoader(ta_ds, batch_size=cfg["batch_size"] // 2,
                               shuffle=True, num_workers=0)
        protocl_loss_fn = ProtoCLRLoss(
            num_classes=cfg["num_classes"],
            temperature=float(cfg["temperature"]),
        )

    if exp in ("fixmatch",) and unl_embs is not None:
        unl_ds = EmbDataset(unl_embs, np.zeros((len(unl_embs), cfg["num_classes"]), np.float32))
        unl_loader = DataLoader(unl_ds, batch_size=cfg["batch_size"] // 2,
                                shuffle=True, num_workers=0)
        fixmatch_loss_fn = FixMatchEmbLoss(
            conf_threshold=float(cfg["fixmatch_conf_thr"]),
            noise_w=float(cfg["fixmatch_noise_w"]),
            noise_s=float(cfg["fixmatch_noise_s"]),
            mask_ratio=float(cfg["fixmatch_mask_ratio"]),
        )

    best_auc = 0.0
    best_state = None
    patience_cnt = 0

    for epoch in range(cfg["n_epochs"]):
        # Refresh cycling iterators each epoch
        if ta_loader is not None:
            ta_iter_state[0] = iter(ta_loader)
        if unl_loader is not None:
            unl_iter_state[0] = iter(unl_loader)

        tr_loss = train_one_epoch(
            model, loader, optimizer, focal, device, cfg,
            ta_loader=ta_loader, unl_loader=unl_loader,
            protocl_loss=protocl_loss_fn, fixmatch_loss=fixmatch_loss_fn,
            ta_iter_state=ta_iter_state, unl_iter_state=unl_iter_state,
        )
        val_auc = eval_auc(model, val_embs, val_labels, device)
        scheduler.step()

        log_fn(f"  Epoch {epoch+1:3d}/{cfg['n_epochs']} | loss={tr_loss:.4f} | val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= cfg["patience"]:
                log_fn(f"  Early stop at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)

    return best_auc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        user_cfg = yaml.safe_load(f)
    cfg = {**DEFAULTS, **user_cfg}

    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(cfg["log_dir"], exist_ok=True)
    log_path = Path(cfg["log_dir"]) / f"{Path(args.config).stem}.log"
    out_path  = Path(cfg["output_dir"]) / cfg["output_name"]
    exp = cfg["exp_type"]

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] [{exp.upper()}] {msg}"
        print(line)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}  |  Experiment: {exp}")

    # ── Taxonomy ──
    taxonomy  = pd.read_csv(cfg["taxonomy_csv"])
    class_list = taxonomy["primary_label"].tolist()
    assert len(class_list) == cfg["num_classes"]

    # ── Labeled soundscape ──
    ss_embs, ss_labels, ss_filenames = load_labeled_ss(cfg)
    log(f"Labeled SS: {ss_embs.shape}  classes: {int((ss_labels.sum(0)>0).sum())}")

    # ── Train audio ──
    log("Loading train audio embeddings...")
    ta_embs, ta_labels = load_train_audio_embeddings(cfg, class_list)
    log(f"Train audio: {ta_embs.shape}")

    # ── Comprehensive val set ──
    k_val = int(cfg["val_per_missing_class"])
    val_embs, val_labels, train_mask = build_comprehensive_valset(
        ss_embs, ss_labels, ta_embs, ta_labels, k=k_val)
    val_classes = int((val_labels.sum(0) > 0).sum())
    log(f"Val set: {len(val_embs):,} windows | classes: {val_classes}/{cfg['num_classes']}")

    # Train = remaining train audio (hard one-hot labels)
    train_embs  = ta_embs[train_mask]
    train_labels = ta_labels[train_mask]
    log(f"Train set: {len(train_embs):,} clips (train audio, hard labels)")

    # ── Unlabeled SS for FixMatch ──
    unl_embs_fm = None
    if exp == "fixmatch":
        unl_embs_raw, unl_fns = load_unlabeled_ss(cfg)
        labeled_fn_set = set(ss_filenames)
        unl_mask = np.array([fn not in labeled_fn_set for fn in unl_fns])
        unl_embs_fm = unl_embs_raw[unl_mask]
        log(f"Unlabeled SS for FixMatch: {len(unl_embs_fm):,}")

    # ── Build model ──
    model = ProtoHead(
        emb_dim=cfg["emb_dim"],
        proj_dim=cfg["proj_dim"],
        num_classes=cfg["num_classes"],
        n_blocks=cfg["n_blocks"],
        dropout=cfg["dropout"],
        temperature=float(cfg["temperature"]),
    ).to(device)
    log(f"ProtoHead parameters: {model.count_parameters():,}")

    # Warm-start prototypes from labeled SS embeddings
    log("Initialising prototypes from labeled SS...")
    ss_t = torch.tensor(ss_embs, device=device)
    ss_lbl_t = torch.tensor(ss_labels, device=device)
    model.init_prototypes_from_data(ss_t, ss_lbl_t)
    del ss_t, ss_lbl_t; torch.cuda.empty_cache()

    # Optional: warm-start projection weights from existing adapter
    if cfg.get("init_ckpt") and Path(cfg["init_ckpt"]).exists():
        log(f"Loading init weights from {cfg['init_ckpt']}...")
        ckpt = torch.load(cfg["init_ckpt"], map_location=device)
        # Load only matching keys (projection layers)
        model_sd = model.state_dict()
        src_sd   = ckpt.get("model_state", ckpt)
        matched  = {k: v for k, v in src_sd.items() if k in model_sd and v.shape == model_sd[k].shape}
        model_sd.update(matched)
        model.load_state_dict(model_sd)
        log(f"  Matched {len(matched)}/{len(model_sd)} keys")

    # ── Train ──
    best_auc = train(
        model, train_embs, train_labels, val_embs, val_labels,
        cfg, device,
        ta_embs=ta_embs if exp == "protocl" else None,
        ta_labels=ta_labels if exp == "protocl" else None,
        unl_embs=unl_embs_fm,
        log_fn=log,
    )
    log(f"Best val AUC ({val_classes} classes): {best_auc:.4f}")

    # ── Save ──
    torch.save({
        "model_state": model.state_dict(),
        "cfg": cfg,
        "exp_type": exp,
        "best_val_auc": best_auc,
    }, out_path)
    log(f"Saved → {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
