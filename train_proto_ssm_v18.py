"""Train ProtoSSM V18 — ProtoSSMv2 + ResidualSSM + Probe + Prior Tables.

Usage:
    CUDA_VISIBLE_DEVICES=1 python train_proto_ssm_v18.py --config configs/proto_ssm_v18.yaml
"""

import argparse
import csv
import json
import os
import pickle
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ── Add project root to path ───────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from src.model.proto_ssm import ProtoSSMv2, ResidualSSM

try:
    import yaml
    def load_cfg(path):
        with open(path) as f:
            return yaml.safe_load(f)
except ImportError:
    import json as _json
    def load_cfg(path):
        raise RuntimeError("PyYAML not installed; run: pip install pyyaml")


# ── Utilities ──────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def focal_bce(logits, targets, gamma=2.5, pos_weight=None, label_smoothing=0.03):
    """Focal binary cross-entropy with label smoothing."""
    targets_s = targets * (1 - label_smoothing) + 0.5 * label_smoothing
    bce = F.binary_cross_entropy_with_logits(
        logits, targets_s, pos_weight=pos_weight, reduction="none"
    )
    prob = torch.sigmoid(logits).detach()
    p_t  = targets * prob + (1 - targets) * (1 - prob)
    focal_w = (1 - p_t) ** gamma
    return (focal_w * bce).mean()


def mixup(emb, logits, labels, alpha=0.4):
    """Mixup augmentation in embedding space."""
    lam = np.random.beta(alpha, alpha)
    B   = emb.size(0)
    idx = torch.randperm(B, device=emb.device)
    emb_mix    = lam * emb    + (1 - lam) * emb[idx]
    logits_mix = lam * logits + (1 - lam) * logits[idx]
    labels_mix = lam * labels + (1 - lam) * labels[idx]
    return emb_mix, logits_mix, labels_mix


def parse_site(filename: str) -> str:
    """Extract site code from soundscape filename.
    E.g. BC2026_Train_0039_S22_20211231_201500.ogg -> S22
    """
    parts = filename.split("_")
    if len(parts) >= 4:
        return parts[3]
    return "UNK"


def parse_hour(filename: str) -> int:
    """Extract hour from soundscape filename.
    E.g. BC2026_Train_0039_S22_20211231_201500.ogg -> 20
    """
    parts = filename.split("_")
    if len(parts) >= 1:
        time_str = parts[-1].replace(".ogg", "").replace(".wav", "")
        if len(time_str) >= 2 and time_str[:2].isdigit():
            return int(time_str[:2])
    return 0


def load_soundscape_meta(labels_csv: str):
    """Returns dict: filename -> {site: str, hour: int}"""
    meta = {}
    with open(labels_csv) as f:
        for row in csv.DictReader(f):
            fn = row["filename"]
            if fn not in meta:
                meta[fn] = {
                    "site": parse_site(fn),
                    "hour": parse_hour(fn),
                }
    return meta


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_data(cfg: dict):
    """Load perch_labeled_ss.npz and build sequence-level tensors."""
    data_cfg   = cfg["data"]
    npz_path   = data_cfg["labeled_npz"]
    labels_csv = data_cfg["labels_csv"]
    T          = cfg["model"]["n_windows"]  # windows per sequence

    print(f"[Data] Loading {npz_path} ...")
    d = np.load(npz_path, allow_pickle=True)
    emb_all      = d["emb"].astype(np.float32)    # (N, 1536)
    logits_all   = d["logits"].astype(np.float32) # (N, 234)
    labels_all   = d["labels"].astype(np.float32) # (N, 234)
    filenames    = d["filenames"]                  # (N,) per-window
    file_list    = d["file_list"]                  # (F,) unique files
    n_windows    = d["n_windows"]                  # (F,)

    # Load soundscape metadata
    ss_meta = load_soundscape_meta(labels_csv)

    # Build site_to_idx from files in file_list
    sites_seen = []
    for fn in file_list:
        fn_base = os.path.basename(str(fn))
        s = parse_site(fn_base)
        if s not in sites_seen:
            sites_seen.append(s)
    site_to_idx = {s: i for i, s in enumerate(sites_seen)}
    # cap at n_sites
    n_sites = cfg["model"]["n_sites"]
    print(f"[Data] Sites found: {list(site_to_idx.keys())}")

    # Build per-sequence tensors
    seqs_emb     = []
    seqs_logits  = []
    seqs_labels  = []
    seqs_site    = []
    seqs_hour    = []
    seq_file_ids = []

    row_ptr = 0
    for file_idx, (fn, nw) in enumerate(zip(file_list, n_windows)):
        fn_base = os.path.basename(str(fn))
        nw = int(nw)
        end = row_ptr + nw

        e = emb_all[row_ptr:end]     # (nw, 1536)
        l = logits_all[row_ptr:end]  # (nw, 234)
        y = labels_all[row_ptr:end]  # (nw, 234)

        # Pad / truncate to T windows
        if nw < T:
            pad = T - nw
            e = np.concatenate([e, np.zeros((pad, e.shape[1]), dtype=np.float32)], 0)
            l = np.concatenate([l, np.zeros((pad, l.shape[1]), dtype=np.float32)], 0)
            y = np.concatenate([y, np.zeros((pad, y.shape[1]), dtype=np.float32)], 0)
        else:
            e = e[:T]; l = l[:T]; y = y[:T]

        site_str = parse_site(fn_base)
        hour_val = parse_hour(fn_base)
        # Try ss_meta for override
        if fn_base in ss_meta:
            hour_val = ss_meta[fn_base]["hour"]

        site_idx = site_to_idx.get(site_str, 0)
        site_idx = min(site_idx, n_sites - 1)

        seqs_emb.append(e)
        seqs_logits.append(l)
        seqs_labels.append(y)
        seqs_site.append(site_idx)
        seqs_hour.append(hour_val % 24)
        seq_file_ids.append(fn_base)

        row_ptr = end

    seqs_emb    = np.array(seqs_emb,    dtype=np.float32)   # (F, T, 1536)
    seqs_logits = np.array(seqs_logits, dtype=np.float32)   # (F, T, 234)
    seqs_labels = np.array(seqs_labels, dtype=np.float32)   # (F, T, 234)
    seqs_site   = np.array(seqs_site,   dtype=np.int64)     # (F,)
    seqs_hour   = np.array(seqs_hour,   dtype=np.int64)     # (F,)
    seq_file_ids = np.array(seq_file_ids)

    print(f"[Data] Sequences: {len(seqs_emb)}, shape emb={seqs_emb.shape}")
    return (
        seqs_emb, seqs_logits, seqs_labels,
        seqs_site, seqs_hour, seq_file_ids,
        site_to_idx
    )


def build_family_labels(taxonomy_csv: str, n_classes: int, primary_labels_list):
    """Build family label matrix and class_to_family mapping."""
    rows = list(csv.DictReader(open(taxonomy_csv)))
    class_names = sorted(set(r["class_name"] for r in rows))
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    n_families = len(class_names)

    # Map primary_label -> family_idx
    pl_to_family = {}
    for r in rows:
        pl_to_family[r["primary_label"]] = class_to_idx.get(r["class_name"], 0)

    class_to_family = []
    for pl in primary_labels_list:
        class_to_family.append(pl_to_family.get(str(pl), 0))

    return np.array(class_to_family, dtype=np.int64), n_families, class_names


# ── Training ───────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, cfg, device, epoch, pos_weight):
    model.train()
    t_cfg = cfg["training"]
    gamma     = t_cfg["focal_gamma"]
    w_distill = t_cfg["distill_weight"]
    w_family  = t_cfg["family_weight"]
    ls        = t_cfg["label_smoothing"]
    alpha_mix = t_cfg["mixup_alpha"]
    do_mixup  = epoch >= 5
    grad_clip = t_cfg["grad_clip"]
    pw = torch.tensor(pos_weight, device=device, dtype=torch.float32)

    total_loss = 0.0
    for emb, logits, labels, sites, hours, fam_lbl in loader:
        emb    = emb.to(device)
        logits = logits.to(device)
        labels = labels.to(device)
        sites  = sites.to(device)
        hours  = hours.to(device)
        fam_lbl = fam_lbl.to(device)

        if do_mixup and np.random.rand() < 0.5:
            # Mixup along batch dim on first window representation
            B = emb.size(0)
            lam = float(np.random.beta(alpha_mix, alpha_mix))
            idx = torch.randperm(B, device=device)
            emb    = lam * emb    + (1 - lam) * emb[idx]
            logits = lam * logits + (1 - lam) * logits[idx]
            labels = lam * labels + (1 - lam) * labels[idx]

        optimizer.zero_grad()
        sp_logits, fam_logits, _ = model(emb, perch_logits=logits,
                                         site_ids=sites, hours=hours)

        loss_focal = focal_bce(sp_logits, labels, gamma=gamma,
                               pos_weight=pw[None, None, :], label_smoothing=ls)
        loss_distill = F.mse_loss(sp_logits, logits)
        loss = loss_focal + w_distill * loss_distill

        if fam_logits is not None:
            loss_fam = F.binary_cross_entropy_with_logits(fam_logits, fam_lbl)
            loss = loss + w_family * loss_fam

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_preds  = []
    all_labels = []
    for emb, logits, labels, sites, hours, _ in loader:
        emb    = emb.to(device)
        logits = logits.to(device)
        sites  = sites.to(device)
        hours  = hours.to(device)
        sp_logits, _, _ = model(emb, perch_logits=logits,
                                site_ids=sites, hours=hours)
        probs = torch.sigmoid(sp_logits).cpu().numpy()  # (B, T, 234)
        all_preds.append(probs.max(axis=1))             # file-level max: (B, 234)
        all_labels.append(labels.numpy().max(axis=1))   # (B, 234)

    preds  = np.concatenate(all_preds,  axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Macro-average AUC (only classes with positives)
    try:
        from sklearn.metrics import roc_auc_score
        aucs = []
        for c in range(labels.shape[1]):
            if labels[:, c].sum() > 0:
                aucs.append(roc_auc_score(labels[:, c], preds[:, c]))
        return float(np.mean(aucs)) if aucs else 0.0, preds, labels
    except Exception:
        return 0.0, preds, labels


def compute_pos_weight(labels_np: np.ndarray, cap: float = 25.0):
    flat = labels_np.reshape(-1, labels_np.shape[-1])
    pos  = flat.sum(0).clip(min=1)
    neg  = flat.shape[0] - pos
    return (neg / pos).clip(max=cap).astype(np.float32)


# ── Prior Tables ───────────────────────────────────────────────────────────────

def build_prior_tables(labels_np, sites_np, hours_np, n_sites=20, n_hours=24,
                       n_classes=234, alpha=1.0):
    """Build site-hour-species prior table with Laplace smoothing."""
    sh_counts = np.zeros((n_sites, n_hours, n_classes), dtype=np.float64)
    sh_totals = np.zeros((n_sites, n_hours), dtype=np.float64)

    for i in range(len(labels_np)):
        s = int(sites_np[i]) % n_sites
        h = int(hours_np[i]) % n_hours
        # File-level max label
        y = labels_np[i].max(axis=0)  # (n_classes,)
        sh_counts[s, h] += y
        sh_totals[s, h] += 1.0

    # Laplace smoothing
    prior = (sh_counts + alpha) / (sh_totals[:, :, None] + alpha * n_classes)
    return prior.astype(np.float32)


# ── Probe Model ────────────────────────────────────────────────────────────────

def seq_features_1d(col, n_windows):
    """Temporal context features for a flat window-level score array.
    col: (N_files * n_windows,)
    Returns: prev, next, mean, max, std — all shape (N_files * n_windows,)
    """
    n_total = len(col)
    n_files = n_total // n_windows
    c = col[:n_files * n_windows].reshape(n_files, n_windows).astype(np.float32)
    prev = np.concatenate([c[:, :1], c[:, :-1]], axis=1).reshape(-1)
    nxt  = np.concatenate([c[:, 1:], c[:, -1:]], axis=1).reshape(-1)
    mean = np.repeat(c.mean(axis=1, keepdims=True), n_windows, axis=1).reshape(-1)
    mx   = np.repeat(c.max(axis=1,  keepdims=True), n_windows, axis=1).reshape(-1)
    std  = np.repeat(c.std(axis=1,  keepdims=True), n_windows, axis=1).reshape(-1)
    return prev, nxt, mean, mx, std


def build_class_features(emb_proj, raw_col, prior_col, base_col, n_windows):
    """Build per-class feature vector matching pantanal probe approach.
    emb_proj:  (N_win, pca_dim)
    raw/prior/base: (N_win,) per-window scalars
    Returns:   (N_win, pca_dim + 14)
    """
    prev_b, next_b, mean_b, max_b, std_b = seq_features_1d(base_col, n_windows)
    return np.concatenate([
        emb_proj,
        raw_col[:, None], prior_col[:, None], base_col[:, None],
        prev_b[:, None], next_b[:, None], mean_b[:, None], max_b[:, None], std_b[:, None],
        (base_col - mean_b)[:, None],
        (base_col - prev_b)[:, None],
        (base_col - next_b)[:, None],
        (raw_col * prior_col)[:, None],
        (raw_col * base_col)[:, None],
        (prior_col * base_col)[:, None],
    ], axis=1).astype(np.float32, copy=False)


def train_probe(oof_emb, oof_scores_raw, oof_preds_win, oof_labels,
                site_ids, hour_ids, prior_tables_arr, cfg):
    """Window-level probe training matching pantanal approach.

    oof_emb:          (N, T, 1536) OOF embeddings
    oof_scores_raw:   (N, T, n_cls) raw Perch logits
    oof_preds_win:    (N, T, n_cls) ProtoSSM window-level predictions (base scores)
    oof_labels:       (N, T, n_cls) ground-truth labels
    site_ids:         (N,) integer site indices
    hour_ids:         (N,) integer hour values
    prior_tables_arr: (n_sites, 24, n_cls) sh_prior
    """
    probe_cfg = cfg["probe"]
    pca_dim   = probe_cfg["pca_dim"]
    min_pos   = probe_cfg["min_pos"]
    N, T, _ = oof_emb.shape
    n_cls = oof_scores_raw.shape[2]
    n_sites_p   = prior_tables_arr.shape[0]

    # Per-window prior: broadcast per-file site-hour prior over T windows
    prior_win = np.zeros((N, T, n_cls), dtype=np.float32)
    for i in range(N):
        s = int(site_ids[i]) % n_sites_p
        h = int(hour_ids[i]) % 24
        prior_win[i] = prior_tables_arr[s, h]  # (n_cls,) broadcasts to (T, n_cls)

    # Flatten to window level
    emb_flat   = oof_emb.reshape(-1, 1536)
    raw_flat   = oof_scores_raw.reshape(-1, n_cls)
    prior_flat = prior_win.reshape(-1, n_cls)
    base_flat  = oof_preds_win.reshape(-1, n_cls)
    y_flat     = oof_labels.reshape(-1, n_cls)

    pca_dim = min(pca_dim, emb_flat.shape[0] - 1, emb_flat.shape[1])
    print(f"[Probe] Fitting StandardScaler + PCA({pca_dim}) on {emb_flat.shape} ...")
    scaler = StandardScaler()
    emb_s  = scaler.fit_transform(emb_flat)
    pca    = PCA(n_components=pca_dim, random_state=42)
    Z      = pca.fit_transform(emb_s).astype(np.float32)
    print(f"[Probe] PCA explained var ratio sum: {pca.explained_variance_ratio_.sum():.4f}")

    probe_models = {}
    trained = 0
    for c in range(n_cls):
        y_c = y_flat[:, c]
        if y_c.sum() < min_pos:
            continue
        X_c = build_class_features(Z, raw_flat[:, c], prior_flat[:, c], base_flat[:, c], T)
        clf = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            alpha=0.005,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            verbose=False,
        )
        clf.fit(X_c, y_c)
        probe_models[c] = clf
        trained += 1

    print(f"[Probe] Trained {trained}/{n_cls} class probes.")
    return scaler, pca, probe_models


def probe_predict(scaler, pca, probe_models, oof_emb, oof_scores_raw,
                  prior_tables_arr, oof_preds_win, site_ids, hour_ids,
                  n_classes, n_windows):
    """Window-level probe inference → file-level max output.

    oof_emb:       (N, T, 1536)
    oof_scores_raw:(N, T, n_cls)
    oof_preds_win: (N, T, n_cls) base scores
    Returns:       (N, n_cls) file-level max of probe predictions
    """
    N, T = oof_emb.shape[:2]
    n_sites_p = prior_tables_arr.shape[0]

    prior_win = np.zeros((N, T, n_classes), dtype=np.float32)
    for i in range(N):
        s = int(site_ids[i]) % n_sites_p
        h = int(hour_ids[i]) % 24
        prior_win[i] = prior_tables_arr[s, h]  # (n_cls,) broadcasts to (T, n_cls)

    emb_flat   = oof_emb.reshape(-1, 1536)
    raw_flat   = oof_scores_raw.reshape(-1, n_classes)
    prior_flat = prior_win.reshape(-1, n_classes)
    base_flat  = oof_preds_win.reshape(-1, n_classes)

    emb_s = scaler.transform(emb_flat)
    Z     = pca.transform(emb_s).astype(np.float32)

    preds_win = np.zeros((N * T, n_classes), dtype=np.float32)
    for c, clf in probe_models.items():
        X_c = build_class_features(Z, raw_flat[:, c], prior_flat[:, c], base_flat[:, c], T)
        if hasattr(clf, "predict_proba"):
            preds_win[:, c] = clf.predict_proba(X_c)[:, 1]
        else:
            preds_win[:, c] = clf.decision_function(X_c)

    return preds_win.reshape(N, T, n_classes).max(axis=1)  # (N, n_cls)


# ── Per-class Threshold Optimization ──────────────────────────────────────────

def optimize_thresholds(preds: np.ndarray, labels: np.ndarray,
                        grid=None, n_classes=234):
    """Find per-class threshold maximizing AUC (proxy: maximize F1-like score)."""
    if grid is None:
        grid = np.arange(0.25, 0.71, 0.05)
    from sklearn.metrics import roc_auc_score
    thresholds = np.full(n_classes, 0.5, dtype=np.float32)
    for c in range(n_classes):
        if labels[:, c].sum() < 2:
            continue
        try:
            auc = roc_auc_score(labels[:, c], preds[:, c])
            # threshold that best separates: use 0.5 as default but check AUC-guided
            best_thr = 0.5
            best_score = -1
            for thr in grid:
                pred_bin = (preds[:, c] >= thr).astype(float)
                tp = ((pred_bin == 1) & (labels[:, c] == 1)).sum()
                fp = ((pred_bin == 1) & (labels[:, c] == 0)).sum()
                fn = ((pred_bin == 0) & (labels[:, c] == 1)).sum()
                prec = tp / max(tp + fp, 1)
                rec  = tp / max(tp + fn, 1)
                f1   = 2 * prec * rec / max(prec + rec, 1e-8)
                if f1 > best_score:
                    best_score = f1
                    best_thr = thr
            thresholds[c] = best_thr
        except Exception:
            pass
    return thresholds


# ── ResidualSSM Training ───────────────────────────────────────────────────────

def train_residual_ssm(res_model, oof_emb, oof_first_pass, oof_labels,
                       oof_sites, oof_hours, cfg, device):
    """Train ResidualSSM on OOF first-pass scores."""
    r_cfg    = cfg["residual_ssm"]
    n_epochs = r_cfg["n_epochs"]
    lr       = r_cfg["lr"]
    patience = r_cfg["patience"]

    E = torch.tensor(oof_emb,        dtype=torch.float32)
    F_ = torch.tensor(oof_first_pass, dtype=torch.float32)
    Y  = torch.tensor(oof_labels,     dtype=torch.float32)
    S  = torch.tensor(oof_sites,      dtype=torch.long)
    H  = torch.tensor(oof_hours,      dtype=torch.long)

    ds = TensorDataset(E, F_, Y, S, H)
    split = max(1, int(0.15 * len(ds)))
    train_ds = torch.utils.data.Subset(ds, range(split, len(ds)))
    val_ds   = torch.utils.data.Subset(ds, range(split))
    t_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  drop_last=False)
    v_loader = DataLoader(val_ds,   batch_size=16, shuffle=False)

    optimizer = torch.optim.AdamW(res_model.parameters(), lr=lr, weight_decay=1e-3)

    best_val_loss = float("inf")
    best_state    = None
    patience_cnt  = 0

    for ep in range(n_epochs):
        res_model.train()
        for e_b, f_b, y_b, s_b, h_b in t_loader:
            e_b = e_b.to(device); f_b = f_b.to(device)
            y_b = y_b.to(device); s_b = s_b.to(device); h_b = h_b.to(device)
            optimizer.zero_grad()
            delta = res_model(e_b, f_b, site_ids=s_b, hours=h_b)
            loss  = F.mse_loss(f_b + delta, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(res_model.parameters(), 1.0)
            optimizer.step()

        # Validation
        res_model.eval()
        val_losses = []
        with torch.no_grad():
            for e_b, f_b, y_b, s_b, h_b in v_loader:
                e_b = e_b.to(device); f_b = f_b.to(device)
                y_b = y_b.to(device); s_b = s_b.to(device); h_b = h_b.to(device)
                delta = res_model(e_b, f_b, site_ids=s_b, hours=h_b)
                val_losses.append(F.mse_loss(f_b + delta, y_b).item())
        vl = np.mean(val_losses)
        print(f"  [ResSSM] Ep {ep+1:02d}/{n_epochs} val_loss={vl:.6f}")

        if vl < best_val_loss:
            best_val_loss = vl
            best_state    = {k: v.cpu().clone() for k, v in res_model.state_dict().items()}
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  [ResSSM] Early stop at epoch {ep+1}")
                break

    if best_state is not None:
        res_model.load_state_dict(best_state)
    print(f"[ResSSM] Best val_loss={best_val_loss:.6f}")
    return res_model


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/proto_ssm_v18.yaml")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    set_seed(cfg["experiment"]["seed"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Device: {device}")

    # Create output dirs
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    pretrained_dir = Path(cfg["output"]["pretrained_dir"])
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Main] Output dir: {out_dir}")
    print(f"[Main] Pretrained dir: {pretrained_dir}")

    # Load data
    (seqs_emb, seqs_logits, seqs_labels,
     seqs_site, seqs_hour, seq_file_ids,
     site_to_idx) = load_data(cfg)

    n_files   = len(seqs_emb)
    n_classes = cfg["model"]["n_classes"]

    # Load taxonomy / family labels
    taxonomy_csv    = cfg["data"]["taxonomy_csv"]
    rows_tax        = list(csv.DictReader(open(taxonomy_csv)))
    primary_labels  = [r["primary_label"] for r in rows_tax]  # (234,)
    class_to_family_arr, n_families, family_names = build_family_labels(
        taxonomy_csv, n_classes, primary_labels
    )
    print(f"[Main] Families: {family_names}  n_families={n_families}")

    # Build file-level family label matrix (B, n_families)
    # y_fam[i, f] = 1 if any window in file i has a species of family f
    seqs_fam = np.zeros((n_files, n_families), dtype=np.float32)
    y_file   = seqs_labels.max(axis=1)  # (F, 234)
    for c in range(n_classes):
        f_idx = class_to_family_arr[c]
        seqs_fam[:, f_idx] = np.maximum(seqs_fam[:, f_idx], y_file[:, c])

    # GroupKFold split
    n_folds = cfg["data"]["n_folds"]
    gkf     = GroupKFold(n_splits=n_folds)
    groups  = seq_file_ids  # one group per file

    # We collect 5-fold OOF predictions
    n_windows_cfg  = cfg["model"]["n_windows"]
    oof_preds      = np.zeros((n_files, n_classes), dtype=np.float32)  # file-level max
    oof_preds_win  = np.zeros((n_files, n_windows_cfg, n_classes), dtype=np.float32)  # window-level
    oof_emb_store  = np.zeros((n_files, n_windows_cfg, 1536), dtype=np.float32)
    fold_aucs      = []

    # Compute global pos_weight on training set (approximate with all data)
    pos_weight = compute_pos_weight(seqs_labels, cap=cfg["training"]["pos_weight_cap"])

    t_cfg     = cfg["training"]
    n_epochs  = t_cfg["n_epochs"]
    batch_sz  = t_cfg["batch_size"]
    patience  = t_cfg["patience"]
    swa_frac  = t_cfg["swa_start_frac"]

    best_fold0_model_state = None
    best_fold0_auc         = 0.0

    print(f"\n[Main] Starting {n_folds}-fold CV training ...")

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(seqs_emb, y=None, groups=groups)):
        print(f"\n{'='*60}")
        print(f"[Fold {fold}] Train={len(tr_idx)}, Val={len(va_idx)}")

        # Build datasets
        E_tr = torch.tensor(seqs_emb[tr_idx],    dtype=torch.float32)
        L_tr = torch.tensor(seqs_logits[tr_idx], dtype=torch.float32)
        Y_tr = torch.tensor(seqs_labels[tr_idx], dtype=torch.float32)
        S_tr = torch.tensor(seqs_site[tr_idx],   dtype=torch.long)
        H_tr = torch.tensor(seqs_hour[tr_idx],   dtype=torch.long)
        F_tr = torch.tensor(seqs_fam[tr_idx],    dtype=torch.float32)

        E_va = torch.tensor(seqs_emb[va_idx],    dtype=torch.float32)
        L_va = torch.tensor(seqs_logits[va_idx], dtype=torch.float32)
        Y_va = torch.tensor(seqs_labels[va_idx], dtype=torch.float32)
        S_va = torch.tensor(seqs_site[va_idx],   dtype=torch.long)
        H_va = torch.tensor(seqs_hour[va_idx],   dtype=torch.long)
        F_va = torch.tensor(seqs_fam[va_idx],    dtype=torch.float32)

        tr_ds   = TensorDataset(E_tr, L_tr, Y_tr, S_tr, H_tr, F_tr)
        va_ds   = TensorDataset(E_va, L_va, Y_va, S_va, H_va, F_va)
        tr_load = DataLoader(tr_ds, batch_size=batch_sz, shuffle=True,  drop_last=True)
        va_load = DataLoader(va_ds, batch_size=batch_sz, shuffle=False)

        # Build model
        m_cfg = cfg["model"]
        model = ProtoSSMv2(
            d_input          = m_cfg["d_input"],
            d_model          = m_cfg["d_model"],
            d_state          = m_cfg["d_state"],
            n_ssm_layers     = m_cfg["n_ssm_layers"],
            n_classes        = m_cfg["n_classes"],
            n_windows        = m_cfg["n_windows"],
            dropout          = m_cfg["dropout"],
            n_prototypes     = m_cfg["n_prototypes"],
            n_sites          = m_cfg["n_sites"],
            meta_dim         = m_cfg["meta_dim"],
            use_cross_attn   = m_cfg["use_cross_attn"],
            cross_attn_heads = m_cfg["cross_attn_heads"],
            n_families       = n_families,
        ).to(device)
        print(f"[Fold {fold}] Model params: {model.count_parameters():,}")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr           = t_cfg["lr"],
            weight_decay = t_cfg["weight_decay"],
        )
        steps_per_epoch = max(1, len(tr_load))
        total_steps     = n_epochs * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr     = t_cfg["lr"],
            total_steps= total_steps,
            pct_start  = 0.1,
        )

        # SWA setup
        swa_start = int(swa_frac * n_epochs)
        swa_model = AveragedModel(model)
        swa_sched = SWALR(optimizer, swa_lr=t_cfg["swa_lr"])

        best_auc      = 0.0
        best_state    = None
        patience_cnt  = 0

        for ep in range(n_epochs):
            t0 = time.time()
            tr_loss = train_epoch(model, tr_load, optimizer, scheduler,
                                  cfg, device, ep, pos_weight)
            val_auc, val_preds, val_lbls = eval_epoch(model, va_load, device)
            dt = time.time() - t0

            if ep >= swa_start:
                swa_model.update_parameters(model)
                swa_sched.step()

            print(f"  Ep {ep+1:03d}/{n_epochs}  loss={tr_loss:.4f}  "
                  f"val_auc={val_auc:.4f}  t={dt:.1f}s"
                  + (" [SWA]" if ep >= swa_start else ""))

            if val_auc > best_auc:
                best_auc   = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    print(f"  [Fold {fold}] Early stop at epoch {ep+1}")
                    break

        # Update BN stats for SWA model
        if ep >= swa_start:
            try:
                torch.optim.swa_utils.update_bn(tr_load, swa_model, device=device)
            except Exception:
                pass

        # Use best model for OOF
        if best_state is not None:
            model.load_state_dict(best_state)

        # Collect OOF predictions (file-level + window-level)
        model.eval()
        oof_probs_fold     = []
        oof_probs_win_fold = []
        with torch.no_grad():
            for e_b, l_b, _, s_b, h_b, _ in DataLoader(va_ds, batch_size=batch_sz):
                e_b = e_b.to(device); l_b = l_b.to(device)
                s_b = s_b.to(device); h_b = h_b.to(device)
                sp_logits, _, _ = model(e_b, perch_logits=l_b,
                                        site_ids=s_b, hours=h_b)
                probs = torch.sigmoid(sp_logits).cpu().numpy()  # (B, T, 234)
                oof_probs_fold.append(probs.max(axis=1))         # (B, 234) file-level
                oof_probs_win_fold.append(probs)                  # (B, T, 234) window-level

        oof_probs_fold     = np.concatenate(oof_probs_fold,     axis=0)  # (|va|, 234)
        oof_probs_win_fold = np.concatenate(oof_probs_win_fold, axis=0)  # (|va|, T, 234)
        oof_preds[va_idx]     = oof_probs_fold
        oof_preds_win[va_idx] = oof_probs_win_fold
        oof_emb_store[va_idx] = seqs_emb[va_idx]

        fold_aucs.append(best_auc)
        print(f"[Fold {fold}] Best val AUC = {best_auc:.4f}")

        # Save fold 0 model as the main model
        if fold == 0:
            best_fold0_auc         = best_auc
            best_fold0_model_state = {k: v.cpu().clone()
                                      for k, v in model.state_dict().items()}

    print(f"\n[Main] OOF fold AUCs: {[f'{a:.4f}' for a in fold_aucs]}")
    print(f"[Main] Mean OOF AUC: {np.mean(fold_aucs):.4f}")

    # Use fold 0 model as the final saved model
    # Re-build final model
    model = ProtoSSMv2(
        d_input          = m_cfg["d_input"],
        d_model          = m_cfg["d_model"],
        d_state          = m_cfg["d_state"],
        n_ssm_layers     = m_cfg["n_ssm_layers"],
        n_classes        = m_cfg["n_classes"],
        n_windows        = m_cfg["n_windows"],
        dropout          = m_cfg["dropout"],
        n_prototypes     = m_cfg["n_prototypes"],
        n_sites          = m_cfg["n_sites"],
        meta_dim         = m_cfg["meta_dim"],
        use_cross_attn   = m_cfg["use_cross_attn"],
        cross_attn_heads = m_cfg["cross_attn_heads"],
        n_families       = n_families,
    )
    model.load_state_dict(best_fold0_model_state)
    model = model.to(device)

    # ── OOF file-level labels (max over windows) ───────────────────────────────
    oof_labels_file = seqs_labels.max(axis=1)  # (N, 234)

    # ── Prior Tables ───────────────────────────────────────────────────────────
    print("\n[Main] Building prior tables ...")
    prior_tables = build_prior_tables(
        seqs_labels, seqs_site, seqs_hour,
        n_sites   = cfg["model"]["n_sites"],
        n_hours   = 24,
        n_classes = n_classes,
    )
    print(f"[Main] Prior tables shape: {prior_tables.shape}")

    # ── Probe Training ─────────────────────────────────────────────────────────
    print("\n[Main] Training probe models (window-level, pantanal-style) ...")
    emb_scaler, emb_pca, probe_models = train_probe(
        oof_emb_store, seqs_logits, oof_preds_win, seqs_labels,
        seqs_site, seqs_hour, prior_tables, cfg
    )

    # Probe OOF predictions (file-level max over windows)
    probe_oof_preds = probe_predict(
        emb_scaler, emb_pca, probe_models,
        oof_emb_store, seqs_logits, prior_tables, oof_preds_win,
        seqs_site, seqs_hour, n_classes, cfg["model"]["n_windows"]
    )

    # Blend base + probe OOF
    probe_alpha = cfg["probe"]["alpha"]
    blended_oof = (1 - probe_alpha) * oof_preds + probe_alpha * probe_oof_preds

    # ── Per-class Threshold Optimization ──────────────────────────────────────
    print("\n[Main] Optimizing per-class thresholds ...")
    thresholds = optimize_thresholds(blended_oof, oof_labels_file, n_classes=n_classes)
    print(f"[Main] Threshold stats: min={thresholds.min():.3f}  "
          f"mean={thresholds.mean():.3f}  max={thresholds.max():.3f}")

    # ── ResidualSSM Training ───────────────────────────────────────────────────
    print("\n[Main] Training ResidualSSM ...")
    r_cfg = cfg["residual_ssm"]
    res_model = ResidualSSM(
        d_input      = m_cfg["d_input"],
        d_scores     = n_classes,
        d_model      = r_cfg["d_model"],
        d_state      = r_cfg["d_state"],
        n_ssm_layers = r_cfg["n_ssm_layers"],
        n_windows    = m_cfg["n_windows"],
        dropout      = r_cfg["dropout"],
        n_sites      = m_cfg["n_sites"],
        meta_dim     = 16,
    ).to(device)
    print(f"[ResSSM] Params: {res_model.count_parameters():,}")

    # OOF first-pass scores as sequences: (N, T, 234)
    # Reconstruct from model
    model.eval()
    oof_first_pass_seqs = np.zeros((n_files, m_cfg["n_windows"], n_classes), dtype=np.float32)
    with torch.no_grad():
        all_e = torch.tensor(seqs_emb,    dtype=torch.float32)
        all_l = torch.tensor(seqs_logits, dtype=torch.float32)
        all_s = torch.tensor(seqs_site,   dtype=torch.long)
        all_h = torch.tensor(seqs_hour,   dtype=torch.long)
        for i in range(0, n_files, batch_sz):
            e_b = all_e[i:i+batch_sz].to(device)
            l_b = all_l[i:i+batch_sz].to(device)
            s_b = all_s[i:i+batch_sz].to(device)
            h_b = all_h[i:i+batch_sz].to(device)
            sp_logits, _, _ = model(e_b, perch_logits=l_b, site_ids=s_b, hours=h_b)
            oof_first_pass_seqs[i:i+batch_sz] = torch.sigmoid(sp_logits).cpu().numpy()

    res_model = train_residual_ssm(
        res_model,
        seqs_emb, oof_first_pass_seqs, seqs_labels,
        seqs_site, seqs_hour, cfg, device
    )

    # ── Save Artifacts ─────────────────────────────────────────────────────────
    print("\n[Main] Saving artifacts ...")

    # 1. ProtoSSMv2 state dict
    proto_path = pretrained_dir / "proto_ssm_v18.pt"
    torch.save(model.state_dict(), proto_path)
    print(f"  Saved: {proto_path}")

    # 2. ResidualSSM state dict
    res_path = pretrained_dir / "residual_ssm_v18.pt"
    torch.save(res_model.state_dict(), res_path)
    print(f"  Saved: {res_path}")

    # 3. Prior tables
    prior_path = pretrained_dir / "prior_tables_v18.pkl"
    with open(prior_path, "wb") as f:
        pickle.dump({"sh_prior": prior_tables, "site_to_idx": site_to_idx}, f)
    print(f"  Saved: {prior_path}")

    # 4. Sklearn artifacts
    sklearn_path = pretrained_dir / "sklearn_v18.pkl"
    with open(sklearn_path, "wb") as f:
        pickle.dump({
            "emb_scaler":   emb_scaler,
            "emb_pca":      emb_pca,
            "probe_models": probe_models,
        }, f)
    print(f"  Saved: {sklearn_path}")

    # 5. Thresholds
    thr_path = pretrained_dir / "thresholds_v18.npy"
    np.save(thr_path, thresholds)
    print(f"  Saved: {thr_path}")

    # 6. Manifest
    manifest = {
        "artifact_files": {
            "proto":        "proto_ssm_v18.pt",
            "residual":     "residual_ssm_v18.pt",
            "prior_tables": "prior_tables_v18.pkl",
            "sklearn":      "sklearn_v18.pkl",
            "thresholds":   "thresholds_v18.npy",
        },
        "proto_parameters": {
            "d_input":          m_cfg["d_input"],
            "d_model":          m_cfg["d_model"],
            "d_state":          m_cfg["d_state"],
            "n_ssm_layers":     m_cfg["n_ssm_layers"],
            "n_classes":        n_classes,
            "n_windows":        m_cfg["n_windows"],
            "dropout":          m_cfg["dropout"],
            "n_prototypes":     m_cfg["n_prototypes"],
            "n_sites":          m_cfg["n_sites"],
            "meta_dim":         m_cfg["meta_dim"],
            "use_cross_attn":   m_cfg["use_cross_attn"],
            "cross_attn_heads": m_cfg["cross_attn_heads"],
            "n_families":       n_families,
        },
        "best_probe":             "mlp",
        "ensemble_weight_proto":  0.65,
        "correction_weight":      r_cfg["correction_weight"],
        "proxy_reduce":           "max",
        "temperature":            {"aves": 1.10, "texture": 0.95},
        "file_level_top_k":       2,
        "tta_shifts":             [0],
        "rank_aware_scale":       True,
        "rank_aware_power":       0.4,
        "delta_shift_alpha":      0.20,
        "n_classes":              n_classes,
        "n_windows":              m_cfg["n_windows"],
        "primary_labels":         primary_labels,
        "site_to_idx":            site_to_idx,
        "class_to_family":        class_to_family_arr.tolist(),
        "family_names":           family_names,
        "has_residual":           True,
        "fold_aucs":              [float(a) for a in fold_aucs],
        "mean_oof_auc":           float(np.mean(fold_aucs)),
        "training_date":          "2026-03-31",
    }
    manifest_path = pretrained_dir / "artifacts_manifest_v18.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved: {manifest_path}")

    print(f"\n[Main] ✓ All V18 artifacts saved to {pretrained_dir}")
    print(f"[Main] Mean OOF AUC = {np.mean(fold_aucs):.4f}")
    print("[Main] Done.")


if __name__ == "__main__":
    main()
