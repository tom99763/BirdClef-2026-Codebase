"""Train ProtoSSM teacher on labeled soundscapes and generate pseudo labels.

Pipeline:
  1. Load labeled soundscape embeddings from perch_labeled_ss.npz
  2. Train ProtoSSM (Bidirectional SSM + Prototypical heads, ~60-120s on CPU)
  3. Load all-soundscape embeddings from perch_emb_all_ss.npz
  4. Run ProtoSSM inference on ALL soundscapes → ssm_probs (N_windows, 234)
  5. Ensemble: w_perch*perch_logits + w_ssm*ssm_logits  (sigmoid of both)
  6. Apply power transform + per-class dynamic threshold
  7. Save pseudo labels → pseudo_labels/ns_r0_protossm.csv

Usage:
    python scripts/gen_proto_teacher_pseudo.py
    python scripts/gen_proto_teacher_pseudo.py --clip_sec 20 --out pseudo_labels/ns_r0_protossm.csv
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Paths / constants ──────────────────────────────────────────────────────────
LABELED_NPZ   = "outputs/perch_labeled_ss.npz"
ALL_SS_NPZ    = "outputs/perch_emb_all_ss.npz"
LABELS_CSV    = "birdclef-2026/train_soundscapes_labels.csv"
TAXONOMY_CSV  = "birdclef-2026/taxonomy.csv"
OUTPUT_CSV    = "pseudo_labels/ns_r0_protossm.csv"

N_WINDOWS    = 12    # 5s windows per 60s file
NUM_CLASSES  = 234
N_EMBED      = 1536

# ProtoSSM architecture config
PROTO_SSM_CFG = dict(d_model=128, d_state=16, n_ssm_layers=2, dropout=0.15)
PROTO_SSM_TRAIN_CFG = dict(
    n_epochs=120, lr=2e-3, weight_decay=1e-3,
    val_ratio=0.15, patience=20,
    pos_weight_cap=30.0,
    distill_weight=0.3,
)

# Pseudo label generation config
PERCH_W    = 0.55
SSM_W      = 0.45
GAMMA      = 2.0
PERCENTILE = 95.0
MIN_THR    = 0.05
MAX_THR    = 0.50


# ── ProtoSSM Architecture ──────────────────────────────────────────────────────

class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj  = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d   = nn.Conv1d(d_model, d_model, d_conv, padding=d_conv - 1, groups=d_model)
        self.dt_proj  = nn.Linear(d_model, d_model, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.A_log   = nn.Parameter(torch.log(A))
        self.D       = nn.Parameter(torch.ones(d_model))
        self.B_proj  = nn.Linear(d_model, d_state, bias=False)
        self.C_proj  = nn.Linear(d_model, d_state, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B_size, T, D = x.shape
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)
        dt = F.softplus(self.dt_proj(x_conv))
        B_t = self.B_proj(x_conv)
        C_t = self.C_proj(x_conv)
        A   = -torch.exp(self.A_log)
        y   = self._selective_scan(x_conv, dt, A, B_t, C_t)
        y   = y * F.silu(z)
        return self.out_proj(y)

    def _selective_scan(self, x, dt, A, B, C):
        batch, T, D = x.shape
        N = self.d_state
        h = torch.zeros(batch, D, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            dt_t = dt[:, t, :, None]
            dA   = torch.exp(A[None] * dt_t)
            dB   = dt_t * B[:, t, None, :]
            h    = h * dA + x[:, t, :, None] * dB
            y_t  = (h * C[:, t, None, :]).sum(-1)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)
        return y + x * self.D[None, None, :]


class ProtoSSM(nn.Module):
    def __init__(self, d_input=1536, d_model=128, d_state=16,
                 n_ssm_layers=2, n_classes=234, n_windows=12, dropout=0.15):
        super().__init__()
        self.d_model   = d_model
        self.n_classes = n_classes
        self.n_windows = n_windows
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout),
        )
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        self.ssm_fwd   = nn.ModuleList()
        self.ssm_bwd   = nn.ModuleList()
        self.ssm_merge = nn.ModuleList()
        self.ssm_norm  = nn.ModuleList()
        for _ in range(n_ssm_layers):
            self.ssm_fwd.append(SelectiveSSM(d_model, d_state))
            self.ssm_bwd.append(SelectiveSSM(d_model, d_state))
            self.ssm_merge.append(nn.Linear(2 * d_model, d_model))
            self.ssm_norm.append(nn.LayerNorm(d_model))
        self.ssm_drop  = nn.Dropout(dropout)
        self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.proto_temp = nn.Parameter(torch.tensor(5.0))
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))

    def init_prototypes_from_data(self, embeddings, labels):
        with torch.no_grad():
            h = self.input_proj(embeddings)
            for c in range(self.n_classes):
                mask = labels[:, c] > 0.5
                if mask.sum() > 0:
                    self.prototypes.data[c] = F.normalize(h[mask].mean(0), dim=0)

    def forward(self, emb, perch_logits=None):
        B, T, _ = emb.shape
        h = self.input_proj(emb)
        h = h + self.pos_enc[:, :T, :]
        for fwd, bwd, merge, norm in zip(self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm):
            residual = h
            h_f = fwd(h)
            h_b = bwd(h.flip(1)).flip(1)
            h   = merge(torch.cat([h_f, h_b], dim=-1))
            h   = self.ssm_drop(h)
            h   = norm(h + residual)
        h_norm = F.normalize(h, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        temp   = F.softplus(self.proto_temp)
        sim    = torch.matmul(h_norm, p_norm.T) * temp
        if perch_logits is not None:
            alpha  = torch.sigmoid(self.fusion_alpha)[None, None, :]
            return alpha * sim + (1 - alpha) * perch_logits
        return sim

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── ProtoSSM Training ──────────────────────────────────────────────────────────

def macro_auc(y_true, y_pred):
    from sklearn.metrics import roc_auc_score
    mask = y_true.sum(0) > 0
    if mask.sum() < 2:
        return 0.0
    try:
        return roc_auc_score(y_true[:, mask], y_pred[:, mask], average="macro")
    except Exception:
        return 0.0


def train_proto_ssm(model, emb_files, logits_files, labels_files, cfg=None, verbose=True):
    if cfg is None:
        cfg = PROTO_SSM_TRAIN_CFG

    n_files = len(emb_files)
    n_val   = max(1, int(n_files * cfg["val_ratio"]))
    perm    = torch.randperm(n_files, generator=torch.Generator().manual_seed(42))
    val_idx   = perm[:n_val]
    train_idx = perm[n_val:]

    emb_train    = torch.tensor(emb_files[train_idx],    dtype=torch.float32)
    logits_train = torch.tensor(logits_files[train_idx], dtype=torch.float32)
    labels_train = torch.tensor(labels_files[train_idx], dtype=torch.float32)
    emb_val      = torch.tensor(emb_files[val_idx],      dtype=torch.float32)
    logits_val   = torch.tensor(logits_files[val_idx],   dtype=torch.float32)
    labels_val   = torch.tensor(labels_files[val_idx],   dtype=torch.float32)

    pos_counts = labels_train.sum(dim=(0, 1))
    total      = labels_train.shape[0] * labels_train.shape[1]
    pos_weight = ((total - pos_counts) / (pos_counts + 1)).clamp(max=cfg["pos_weight_cap"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg["lr"],
        epochs=cfg["n_epochs"], steps_per_epoch=1,
        pct_start=0.1, anneal_strategy="cos",
    )

    best_val_loss = float("inf")
    best_state    = None
    wait          = 0

    for epoch in range(cfg["n_epochs"]):
        model.train()
        species_out = model(emb_train, logits_train)
        loss_bce    = F.binary_cross_entropy_with_logits(
            species_out, labels_train, pos_weight=pos_weight[None, None, :]
        )
        loss_distill = F.mse_loss(species_out, logits_train)
        loss         = loss_bce + cfg["distill_weight"] * loss_distill
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_out  = model(emb_val, logits_val)
            val_loss = F.binary_cross_entropy_with_logits(
                val_out, labels_val, pos_weight=pos_weight[None, None, :]
            )
            val_pred = val_out.reshape(-1, NUM_CLASSES).numpy()
            val_true = labels_val.reshape(-1, NUM_CLASSES).numpy()
            val_auc  = macro_auc(val_true, val_pred)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            wait          = 0
        else:
            wait += 1

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}: train={loss.item():.4f} "
                  f"val={val_loss.item():.4f} auc={val_auc:.4f} wait={wait}")

        if wait >= cfg["patience"]:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1} (best val={best_val_loss:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if verbose:
        print(f"  Training done. Best val_loss={best_val_loss:.4f}")
        with torch.no_grad():
            alphas = torch.sigmoid(model.fusion_alpha).numpy()
            print(f"  Fusion alpha: mean={alphas.mean():.3f}  "
                  f"min={alphas.min():.3f}  max={alphas.max():.3f}")
    return model


# ── Pseudo Label Logic (mirrors gen_pseudo_ns.py) ─────────────────────────────

def merge_5s_to_Ns(row_ids, probs, clip_sec, species_cols):
    """Convert 5s-window predictions to N-second clip format (max pooling)."""
    df = pd.DataFrame(probs, columns=species_cols)
    df["row_id"]  = row_ids
    df["_fname"]  = df["row_id"].apply(lambda r: str(r).rsplit("_", 1)[0])
    df["_offset"] = df["row_id"].apply(lambda r: int(str(r).rsplit("_", 1)[1]))

    rows_out  = []
    stride    = 5
    for fname, grp in df.groupby("_fname"):
        grp     = grp.sort_values("_offset").reset_index(drop=True)
        offsets = grp["_offset"].tolist()
        p       = grp[species_cols].values
        off2row = {o: i for i, o in enumerate(offsets)}
        max_off = max(offsets)
        for end in range(clip_sec, max_off + stride + 1, stride):
            window_offs = range(end - clip_sec + 5, end + 1, 5)
            rows = [p[off2row[o]] for o in window_offs if o in off2row]
            if not rows:
                continue
            merged = np.max(rows, axis=0)
            rows_out.append([f"{fname}_{end}"] + merged.tolist())

    out = pd.DataFrame(rows_out, columns=["row_id"] + species_cols)
    print(f"  merge 5s→{clip_sec}s: {len(df)} rows → {len(out)} rows")
    return out


def run_pseudo_label_gen(row_ids, ensemble_probs, species_cols, labeled_files,
                         clip_sec, out_path, round_num):
    # Merge to N-second clips
    df = pd.DataFrame(ensemble_probs, columns=species_cols)
    df["row_id"] = row_ids

    if clip_sec > 5:
        df = merge_5s_to_Ns(row_ids, ensemble_probs, clip_sec, species_cols)
        row_ids_ns  = df["row_id"].tolist()
        probs_ns    = df[species_cols].values
    else:
        row_ids_ns = row_ids
        probs_ns   = ensemble_probs

    # Exclude labeled soundscapes
    def rid_to_fname(rid):
        parts = str(rid).rsplit("_", 1)
        return parts[0] + ".ogg" if len(parts) == 2 else rid + ".ogg"

    mask_unlab = np.array([rid_to_fname(r) not in labeled_files for r in row_ids_ns])
    probs_unlab = probs_ns[mask_unlab]
    rids_unlab  = [r for r, m in zip(row_ids_ns, mask_unlab) if m]
    print(f"  Unlabeled windows: {len(rids_unlab):,} / {len(row_ids_ns):,}")

    # Power transform
    probs_pt = np.power(np.clip(probs_unlab, 0, 1), GAMMA)

    # Per-class dynamic threshold
    thr = np.percentile(probs_pt, PERCENTILE, axis=0)
    thr = np.clip(thr, MIN_THR, MAX_THR)

    # Filter
    above      = (probs_pt >= thr[None, :]).any(axis=1)
    probs_keep = probs_unlab[above]
    rids_keep  = [r for r, a in zip(rids_unlab, above) if a]
    print(f"  Windows kept: {len(rids_keep):,} ({100*above.mean():.1f}%)")

    primary_labels    = [species_cols[i] for i in probs_keep.argmax(axis=1)]
    secondary_labels  = []
    for i, row_p in enumerate(probs_keep):
        sec = [species_cols[j] for j in np.where(row_p >= thr)[0]
               if species_cols[j] != primary_labels[i]]
        secondary_labels.append(";".join(sec))

    out_df = pd.DataFrame(probs_keep, columns=species_cols)
    out_df.insert(0, "row_id", rids_keep)
    out_df["primary_label"]    = primary_labels
    out_df["secondary_labels"] = secondary_labels

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(out_df):,} pseudo-labeled windows → {out_path}")

    print(f"\n=== Pseudo Label Stats (Round {round_num}) ===")
    print(f"  Total unlab windows : {len(rids_unlab):,}")
    print(f"  Kept                : {len(rids_keep):,} ({100*len(rids_keep)/max(1,len(rids_unlab)):.1f}%)")
    top5 = pd.Series(primary_labels).value_counts().head(5)
    print(f"  Top-5 primary labels:\n{top5.to_string()}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled_npz",  default=LABELED_NPZ)
    parser.add_argument("--all_ss_npz",   default=ALL_SS_NPZ)
    parser.add_argument("--labels_csv",   default=LABELS_CSV)
    parser.add_argument("--taxonomy_csv", default=TAXONOMY_CSV)
    parser.add_argument("--clip_sec",     type=int, default=20)
    parser.add_argument("--out",          default=OUTPUT_CSV)
    parser.add_argument("--perch_w",      type=float, default=PERCH_W)
    parser.add_argument("--ssm_w",        type=float, default=SSM_W)
    parser.add_argument("--round",        type=int,   default=0)
    parser.add_argument("--save_model",   default=None, help="Save trained ProtoSSM to path")
    args = parser.parse_args()

    # ── Load taxonomy ────────────────────────────────────────────────────────────
    taxonomy     = pd.read_csv(args.taxonomy_csv)
    species_cols = taxonomy["primary_label"].astype(str).tolist()
    assert len(species_cols) == NUM_CLASSES

    labeled_files_df = pd.read_csv(args.labels_csv)
    labeled_files    = set(labeled_files_df["filename"].astype(str).unique())
    print(f"Labeled soundscape files to exclude: {len(labeled_files)}")

    # ── Load labeled soundscape embeddings (training data for ProtoSSM) ──────────
    print(f"\nLoading labeled embeddings: {args.labeled_npz}")
    lab = np.load(args.labeled_npz, allow_pickle=True)
    emb_flat    = lab["emb"].astype(np.float32)      # (N, 1536)
    logits_flat = lab["logits"].astype(np.float32)   # (N, 234)
    labels_flat = lab["labels"].astype(np.float32)   # (N, 234)
    fnames_flat = [str(f) for f in lab["filenames"]] # (N,)
    print(f"  {emb_flat.shape[0]:,} windows from {len(set(fnames_flat))} labeled files")

    # Reshape to file-level arrays (only files with exactly N_WINDOWS windows)
    from collections import defaultdict
    file_windows = defaultdict(list)
    for i, fn in enumerate(fnames_flat):
        file_windows[fn].append(i)

    good_files = [fn for fn, idxs in file_windows.items() if len(idxs) == N_WINDOWS]
    print(f"  Files with exactly {N_WINDOWS} windows: {len(good_files)}")

    n_files = len(good_files)
    emb_files    = np.zeros((n_files, N_WINDOWS, N_EMBED),    dtype=np.float32)
    logits_files = np.zeros((n_files, N_WINDOWS, NUM_CLASSES), dtype=np.float32)
    labels_files = np.zeros((n_files, N_WINDOWS, NUM_CLASSES), dtype=np.float32)

    for fi, fn in enumerate(good_files):
        idxs = sorted(file_windows[fn])[:N_WINDOWS]
        emb_files[fi]    = emb_flat[idxs]
        logits_files[fi] = logits_flat[idxs]
        labels_files[fi] = labels_flat[idxs]

    print(f"  emb_files: {emb_files.shape}")

    # ── Train ProtoSSM ───────────────────────────────────────────────────────────
    print(f"\nTraining ProtoSSM ({n_files} files, {N_WINDOWS} windows each) ...")
    t0 = time.time()

    model = ProtoSSM(
        d_input     = N_EMBED,
        d_model     = PROTO_SSM_CFG["d_model"],
        d_state     = PROTO_SSM_CFG["d_state"],
        n_ssm_layers= PROTO_SSM_CFG["n_ssm_layers"],
        n_classes   = NUM_CLASSES,
        n_windows   = N_WINDOWS,
        dropout     = PROTO_SSM_CFG["dropout"],
    )
    print(f"  Parameters: {model.count_parameters():,}")

    # Initialize prototypes from flat labeled embeddings
    emb_t = torch.tensor(emb_flat, dtype=torch.float32)
    lab_t = torch.tensor(labels_flat, dtype=torch.float32)
    model.init_prototypes_from_data(emb_t, lab_t)
    del emb_t, lab_t; gc.collect()

    model = train_proto_ssm(
        model, emb_files, logits_files, labels_files,
        cfg=PROTO_SSM_TRAIN_CFG, verbose=True,
    )
    model.eval()
    print(f"  ProtoSSM training done in {time.time()-t0:.1f}s")

    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
        print(f"  Model saved → {args.save_model}")

    # ── Load all-soundscape embeddings ───────────────────────────────────────────
    print(f"\nLoading all-soundscape embeddings: {args.all_ss_npz}")
    all_ss = np.load(args.all_ss_npz, allow_pickle=True)
    all_emb     = all_ss["emb"].astype(np.float32)      # (N_all, 1536)
    all_logits  = all_ss["logits"].astype(np.float32)   # (N_all, 234) — Perch probs
    all_row_ids = [str(r) for r in all_ss["row_ids"]]
    all_fnames  = [str(f) for f in all_ss["filenames"]]
    print(f"  {len(all_row_ids):,} windows from {len(set(all_fnames))} files")

    # Reshape to file-level for SSM inference
    file_windows_all = defaultdict(list)
    for i, fn in enumerate(all_fnames):
        file_windows_all[fn].append(i)

    all_good_files = [fn for fn, idxs in file_windows_all.items()
                      if len(idxs) == N_WINDOWS]
    print(f"  Files with exactly {N_WINDOWS} windows: {len(all_good_files)}")

    # ── Run ProtoSSM inference on all soundscapes ────────────────────────────────
    print("\nRunning ProtoSSM inference on all soundscapes ...")
    ssm_probs_flat = np.zeros((len(all_row_ids), NUM_CLASSES), dtype=np.float32)
    # Use Perch probs as fallback for files with ≠12 windows
    ssm_probs_flat[:] = all_logits  # init with Perch probs

    batch_size_files = 64
    n_good = len(all_good_files)

    with torch.no_grad():
        for batch_start in range(0, n_good, batch_size_files):
            batch_files = all_good_files[batch_start:batch_start + batch_size_files]
            n_batch     = len(batch_files)

            batch_emb    = np.zeros((n_batch, N_WINDOWS, N_EMBED),     dtype=np.float32)
            batch_logits = np.zeros((n_batch, N_WINDOWS, NUM_CLASSES), dtype=np.float32)
            batch_idxs   = []  # flat indices for output

            for bi, fn in enumerate(batch_files):
                idxs = sorted(file_windows_all[fn])[:N_WINDOWS]
                batch_emb[bi]    = all_emb[idxs]
                batch_logits[bi] = all_logits[idxs]
                batch_idxs.extend(idxs)

            emb_t    = torch.tensor(batch_emb,    dtype=torch.float32)
            logits_t = torch.tensor(batch_logits, dtype=torch.float32)

            out = model(emb_t, logits_t)           # (n_batch, N_WINDOWS, NUM_CLASSES)
            out_probs = torch.sigmoid(out).numpy()  # convert logits → probs

            for bi in range(n_batch):
                for wi in range(N_WINDOWS):
                    flat_idx = batch_idxs[bi * N_WINDOWS + wi]
                    ssm_probs_flat[flat_idx] = out_probs[bi, wi]

            if (batch_start // batch_size_files) % 10 == 0:
                pct = 100 * (batch_start + n_batch) / n_good
                print(f"  Inference: {batch_start + n_batch}/{n_good} files ({pct:.0f}%)", flush=True)

    print(f"  ProtoSSM inference done. ssm_probs: {ssm_probs_flat.shape}")

    # ── Ensemble Perch + ProtoSSM ────────────────────────────────────────────────
    print(f"\nEnsembling: Perch×{args.perch_w} + SSM×{args.ssm_w}")
    w_total       = args.perch_w + args.ssm_w
    ensemble_probs = (args.perch_w * all_logits + args.ssm_w * ssm_probs_flat) / w_total
    print(f"  ensemble mean: {ensemble_probs.mean():.4f}  max: {ensemble_probs.max():.4f}")

    # ── Generate pseudo labels ───────────────────────────────────────────────────
    print(f"\nGenerating pseudo labels (clip_sec={args.clip_sec}) ...")
    run_pseudo_label_gen(
        row_ids       = all_row_ids,
        ensemble_probs= ensemble_probs,
        species_cols  = species_cols,
        labeled_files = labeled_files,
        clip_sec      = args.clip_sec,
        out_path      = args.out,
        round_num     = args.round,
    )


if __name__ == "__main__":
    main()
