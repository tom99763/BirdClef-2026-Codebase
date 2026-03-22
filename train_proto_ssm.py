"""ProtoSSM 5-fold cross-validation training.

Faithful reimplementation of pantanal-distill-birdclef2026.ipynb.
Supports optional mini-batch training + augmentation via config flags.

Usage:
    python train_proto_ssm.py --config configs/proto_ssm_v3.yaml
    python train_proto_ssm.py --config configs/proto_ssm_v3.yaml --fold 0
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model.proto_ssm import ProtoSSM, ProtoSSMLoss, build_proto_ssm
from src.utils.config import load_config


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset(cfg: dict) -> dict:
    data_cfg = cfg["data"]
    npz_path = data_cfg["labeled_npz"]
    print(f"Loading {npz_path} ...")
    npz = np.load(npz_path, allow_pickle=True)

    emb_flat    = npz["emb"]
    logits_flat = npz["logits"]
    labels_flat = npz["labels"]
    file_list   = npz["file_list"]
    n_windows   = npz["n_windows"]

    teacher_temp = cfg.get("teacher_temperature", 1.0)
    if teacher_temp != 1.0:
        logits_flat = logits_flat / teacher_temp
        print(f"  Teacher temperature: T={teacher_temp}")

    W         = data_cfg.get("windows_per_seq", 12)
    n_classes = cfg["model"]["n_classes"]

    taxonomy     = pd.read_csv(data_cfg["taxonomy_csv"])
    species_list = taxonomy["primary_label"].astype(str).tolist()
    tax_groups   = sorted(taxonomy["class_name"].unique().tolist())
    n_families   = len(tax_groups)
    sp2class     = dict(zip(taxonomy["primary_label"].astype(str), taxonomy["class_name"]))
    sp2tax       = {sp: tax_groups.index(sp2class[sp])
                    for sp in species_list if sp in sp2class}

    seqs_emb, seqs_logits, seqs_labels, seqs_fam, file_ids = [], [], [], [], []
    ptr = 0
    for fid, nw in enumerate(n_windows):
        e_f = emb_flat[ptr:ptr + nw]
        l_f = logits_flat[ptr:ptr + nw]
        y_f = labels_flat[ptr:ptr + nw]
        ptr += nw

        n_segs = max(1, nw // W)
        for s in range(n_segs):
            st, en = s * W, s * W + W
            e_s = e_f[st:en]; l_s = l_f[st:en]; y_s = y_f[st:en]

            if len(e_s) < W:
                pad = W - len(e_s)
                e_s = np.concatenate([e_s, np.zeros((pad, e_s.shape[1]), np.float32)])
                l_s = np.concatenate([l_s, np.zeros((pad, l_s.shape[1]), np.float32)])
                y_s = np.concatenate([y_s, np.zeros((pad, y_s.shape[1]), np.float32)])

            fam_label = np.zeros(n_families, np.float32)
            for sp_idx, sp in enumerate(species_list):
                if y_s[:, sp_idx].max() > 0.5 and sp in sp2tax:
                    fam_label[sp2tax[sp]] = 1.0

            seqs_emb.append(e_s)
            seqs_logits.append(l_s)
            seqs_labels.append(y_s)
            seqs_fam.append(fam_label)
            file_ids.append(fid)

    dataset = {
        "emb":        np.stack(seqs_emb),
        "logits":     np.stack(seqs_logits),
        "labels":     np.stack(seqs_labels),
        "fam":        np.stack(seqs_fam),
        "file_ids":   np.array(file_ids),
        "file_list":  file_list,
        "n_classes":  n_classes,
        "n_families": n_families,
        "W":          W,
    }
    print(f"  Sequences: {len(seqs_emb)}  (W={W}, n_files={len(file_list)}, n_families={n_families})")
    print(f"  Label density: {dataset['labels'].mean()*100:.2f}%")
    return dataset


# ── Augmentation helpers ──────────────────────────────────────────────────────

def augment_emb(emb, logits, labels, t_cfg):
    """Gaussian noise + random window masking."""
    noise_std   = t_cfg.get("aug_noise_std",   0.0)
    time_mask_p = t_cfg.get("aug_time_mask_p", 0.0)

    if noise_std > 0:
        emb = emb + torch.randn_like(emb) * noise_std

    if time_mask_p > 0:
        B, T, _ = emb.shape
        mask = torch.rand(B, T, device=emb.device) < time_mask_p
        emb = emb.clone()
        emb[mask] = 0.0

    return emb, logits, labels


def mixup_sequences(emb, logits, labels, alpha):
    """Beta(α,α) sequence-level mixup."""
    B = emb.shape[0]
    if B < 2 or alpha <= 0:
        return emb, logits, labels
    lam  = float(torch.distributions.Beta(torch.tensor(alpha),
                                          torch.tensor(alpha)).sample())
    lam  = max(lam, 1 - lam)
    perm = torch.randperm(B, device=emb.device)
    return (
        lam * emb    + (1 - lam) * emb[perm],
        lam * logits + (1 - lam) * logits[perm],
        lam * labels + (1 - lam) * labels[perm],
    )


# ── AUC helper ────────────────────────────────────────────────────────────────

def compute_auc(logits, labels):
    probs = 1 / (1 + np.exp(-logits))
    aucs  = []
    for c in range(labels.shape[-1]):
        y = labels[..., c].ravel()
        p = probs[..., c].ravel()
        if y.sum() > 0:
            try:
                aucs.append(roc_auc_score(y, p))
            except Exception:
                pass
    return float(np.mean(aucs)) if aucs else 0.0


# ── Training ──────────────────────────────────────────────────────────────────

def train_fold(fold, tr_idx, va_idx, dataset, cfg, device, out_dir):
    t_cfg = cfg["training"]
    W     = dataset["W"]

    def to_t(arr): return torch.from_numpy(arr).to(device)

    emb_tr   = to_t(dataset["emb"][tr_idx])
    logit_tr = to_t(dataset["logits"][tr_idx])
    label_tr = to_t(dataset["labels"][tr_idx])
    fam_tr   = to_t(dataset["fam"][tr_idx])

    emb_va   = to_t(dataset["emb"][va_idx])
    logit_va = to_t(dataset["logits"][va_idx])
    label_va = to_t(dataset["labels"][va_idx])
    fam_va   = to_t(dataset["fam"][va_idx])

    Ntr, Nva   = len(tr_idx), len(va_idx)
    n_families = dataset["n_families"]

    # Model
    model = build_proto_ssm(cfg).to(device)
    model.init_family_head(n_families)
    model = model.to(device)

    if t_cfg.get("init_prototypes", True):
        with torch.no_grad():
            model.init_prototypes_from_data(
                emb_tr.view(Ntr * W, -1),
                label_tr.view(Ntr * W, -1),
            )

    # Optimizer & scheduler
    epochs     = t_cfg["epochs"]
    batch_size = t_cfg.get("batch_size", None) or Ntr
    n_steps    = max(1, math.ceil(Ntr / batch_size))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = t_cfg["learning_rate"],
        weight_decay = t_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = t_cfg["learning_rate"],
        epochs          = epochs,
        steps_per_epoch = n_steps,
        pct_start       = 0.1,
        anneal_strategy = "cos",
    )

    pos_weight = ProtoSSMLoss.compute_pos_weight(label_tr).to(device)
    loss_fn = ProtoSSMLoss(
        pos_weight_cap = t_cfg.get("pos_weight_cap",      30.0),
        w_distill      = t_cfg.get("loss_distill_weight",  0.3),
        w_family       = t_cfg.get("loss_family_weight",   0.1),
    )

    # wandb
    exp_name = cfg["experiment"]["name"]
    run_name = f"{exp_name}_fold{fold}"
    _wandb = None
    try:
        import wandb
        _wandb = wandb.init(
            project = "birdclef-2026",
            name    = run_name,
            group   = exp_name,
            config  = {k: v for k, v in cfg.items() if not k.startswith("_")},
            reinit  = "finish_previous",
        )
    except Exception as e:
        print(f"  [wandb] disabled: {e}")

    progress_path = os.path.join(out_dir, f"fold{fold}_progress.jsonl")
    open(progress_path, "w").close()

    mixup_alpha  = t_cfg.get("aug_mixup_alpha", 0.0)
    label_smooth = t_cfg.get("label_smoothing", 0.0)
    patience     = t_cfg.get("early_stopping_patience", 0)
    grad_clip    = t_cfg.get("grad_clip", 1.0)

    print(f"  Fold {fold}: Ntr={Ntr} Nva={Nva}  params={model.count_parameters():,}  "
          f"batch={batch_size}  steps/ep={n_steps}  "
          f"noise={t_cfg.get('aug_noise_std',0)}  mask={t_cfg.get('aug_time_mask_p',0)}  "
          f"mixup={mixup_alpha}  smooth={label_smooth}")

    best_val_loss = float("inf")
    best_state    = None
    wait          = 0
    history       = []

    for ep in range(1, epochs + 1):
        # Train (mini-batch with aug)
        model.train()
        perm      = torch.randperm(Ntr, device=device)
        ep_losses = []

        for start in range(0, Ntr, batch_size):
            idx = perm[start:start + batch_size]
            Nb  = len(idx)

            emb_b   = emb_tr[idx].clone()
            logit_b = logit_tr[idx]
            label_b = label_tr[idx]
            fam_b   = fam_tr[idx]

            emb_b, logit_b, label_b = augment_emb(emb_b, logit_b, label_b, t_cfg)
            if mixup_alpha > 0:
                emb_b, logit_b, label_b = mixup_sequences(emb_b, logit_b, label_b, mixup_alpha)
            if label_smooth > 0:
                label_b = label_b * (1 - label_smooth) + 0.5 * label_smooth

            sp_out, fam_out, _ = model(emb_b, logit_b)
            total, comps = loss_fn(sp_out, fam_out, label_b, fam_b, logit_b,
                                   pos_weight=pos_weight)
            optimizer.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            ep_losses.append(comps["loss_total"])

        train_loss = float(np.mean(ep_losses))

        # Validate
        model.eval()
        with torch.no_grad():
            va_sp, va_fam, _ = model(emb_va, logit_va)
            val_loss = F.binary_cross_entropy_with_logits(
                va_sp, label_va,
                pos_weight=pos_weight[None, None, :],
            ).item()

        val_auc = compute_auc(va_sp.cpu().numpy(), label_va.cpu().numpy())
        lr_now  = optimizer.param_groups[0]["lr"]

        row = {"epoch": ep, "val_auc": val_auc, "val_loss": val_loss,
               "train_loss": train_loss, "lr": lr_now}
        history.append(row)

        if ep % 20 == 0 or ep <= 5:
            print(f"    ep{ep:3d}/{epochs}  train={train_loss:.4f}"
                  f"  val_loss={val_loss:.4f}  val_auc={val_auc:.4f}"
                  f"  lr={lr_now:.2e}")

        if _wandb:
            _wandb.log({"val_auc": val_auc, "val_loss": val_loss,
                        "train_loss": train_loss, "lr": lr_now, "epoch": ep})

        with open(progress_path, "a") as pf:
            json.dump({"experiment": run_name, **row}, pf)
            pf.write("\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if patience > 0 and wait >= patience:
                print(f"    Early stopping ep{ep}  best_val_loss={best_val_loss:.4f}")
                break

    if _wandb:
        best_auc = max(r["val_auc"] for r in history)
        _wandb.summary["best_val_loss"] = best_val_loss
        _wandb.summary["best_auc"]      = best_auc
        _wandb.finish()

    with torch.no_grad():
        alphas = torch.sigmoid(model.fusion_alpha).cpu().numpy()
        temp   = F.softplus(model.proto_temp).item()
        print(f"  fusion_alpha: mean={alphas.mean():.3f}  min={alphas.min():.3f}  max={alphas.max():.3f}")
        print(f"  proto_temp: {temp:.3f}")

    if cfg["output"].get("save_model", True) and best_state is not None:
        torch.save({"state_dict": best_state, "fold": fold,
                    "best_val_loss": best_val_loss},
                   os.path.join(out_dir, f"fold{fold}_best.pt"))

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        oof_sp, _, _ = model(emb_va, logit_va)
    oof_logits = oof_sp.cpu().numpy()

    best_auc = max(r["val_auc"] for r in history)
    return best_val_loss, best_auc, oof_logits, history


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/proto_ssm_v3.yaml")
    parser.add_argument("--fold",   type=int, default=None)
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg     = load_config(args.config)
    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")

    torch.manual_seed(cfg["experiment"]["seed"])
    np.random.seed(cfg["experiment"]["seed"])

    dataset = load_dataset(cfg)
    n_seqs  = len(dataset["file_ids"])
    n_folds = cfg["data"]["n_folds"]

    kf     = GroupKFold(n_splits=n_folds)
    splits = list(kf.split(np.arange(n_seqs), groups=dataset["file_ids"]))

    all_oof_logits = np.zeros((n_seqs, dataset["W"], dataset["n_classes"]), dtype=np.float32)
    all_oof_labels = dataset["labels"]
    fold_aucs      = []
    all_history    = {}

    for fold_idx, (tr_idx, va_idx) in enumerate(splits):
        if args.fold is not None and fold_idx != args.fold:
            continue

        print(f"\n{'='*60}")
        print(f" Fold {fold_idx + 1}/{n_folds}")
        print(f"{'='*60}")
        t0 = time.time()

        best_val_loss, best_auc, oof_logits, history = train_fold(
            fold    = fold_idx,
            tr_idx  = tr_idx,
            va_idx  = va_idx,
            dataset = dataset,
            cfg     = cfg,
            device  = device,
            out_dir = out_dir,
        )

        all_oof_logits[va_idx] = oof_logits
        fold_aucs.append(best_auc)
        all_history[f"fold{fold_idx}"] = history
        print(f"  Fold {fold_idx}: best_val_loss={best_val_loss:.4f}  best_auc={best_auc:.4f}  ({time.time()-t0:.0f}s)")

    if args.fold is None and len(fold_aucs) == n_folds:
        flat_logits = all_oof_logits.reshape(-1, dataset["n_classes"])
        flat_labels = all_oof_labels.reshape(-1, dataset["n_classes"])
        oof_auc     = compute_auc(flat_logits, flat_labels)

        print(f"\n{'='*60}")
        print(f" OOF Results")
        print(f"{'='*60}")
        for i, auc in enumerate(fold_aucs):
            print(f"  Fold {i}: best_auc={auc:.4f}")
        print(f"  OOF AUC: {oof_auc:.4f}  (mean per-fold: {np.mean(fold_aucs):.4f})")

        if cfg["output"].get("save_oof", True):
            np.savez_compressed(
                os.path.join(out_dir, "oof_predictions.npz"),
                oof_logits = all_oof_logits,
                oof_labels = all_oof_labels,
                file_ids   = dataset["file_ids"],
                file_list  = dataset["file_list"],
            )

        result = {
            "experiment":    cfg["experiment"]["name"],
            "fold_aucs":     fold_aucs,
            "oof_auc":       oof_auc,
            "mean_fold_auc": float(np.mean(fold_aucs)),
            "finished":      True,
        }
        with open(os.path.join(out_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=2)
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(all_history, f, indent=2)
        print(f"  Results saved → {out_dir}/")


if __name__ == "__main__":
    main()
