"""Train a lightweight Temporal Residual Corrector on SED_R vs Perch-teacher residuals.

The corrector learns:  teacher_probs - SED_R_probs  (probability space)
After training it saves:
  1. checkpoints/sed_corrector_rR.pt  — model checkpoint
  2. <sed_dir>/all_ss_probs_corrected.npz  — corrected SED predictions
     (same format as all_ss_probs.npz, ready for gen_pseudo_ns.py)

Architecture: 1-layer Bidirectional Selective SSM (same as ProtoSSM/ResidualSSM in notebook).
Uses SelectiveSSM from src/model/proto_ssm.py — consistent with the rest of the codebase.
No Perch embeddings needed — operates purely on SED predictions.

Usage (after round R infer_all_ss):
    python3 scripts/train_sed_residual_corrector.py \\
        --sed_dir   outputs/sed-ns-b0-20s-r1 \\
        --teacher   outputs/perch_teacher_aug_all_ss.csv \\
        --round     1 \\
        --alpha     0.40 \\
        --out_ckpt  checkpoints/sed_corrector_r1.pt

Integration in auto_sed_ns_20s_full.sh:
    After each round's infer_all_ss, call this script.
    Then point gen_pseudo_ns.py at all_ss_probs_corrected.npz.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.proto_ssm import SelectiveSSM

NUM_CLASSES = 234
N_WINDOWS   = 12


# ── Model ─────────────────────────────────────────────────────────────────────

class TemporalResidualCorrector(nn.Module):
    """Bidirectional SSM corrector: (B, T, C) SED probs → (B, T, C) corrections.

    Uses SelectiveSSM (Mamba-style) consistent with ProtoSSM / ResidualSSM.
    Operates in probability space. Initialised to output near-zero corrections.
    """

    def __init__(self, n_classes: int = 234, d_model: int = 128, d_state: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_classes, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Bidirectional SSM: forward + backward passes
        self.ssm_fwd   = SelectiveSSM(d_model, d_state)
        self.ssm_bwd   = SelectiveSSM(d_model, d_state)
        self.ssm_merge = nn.Linear(2 * d_model, d_model)
        self.ssm_norm  = nn.LayerNorm(d_model)
        self.drop      = nn.Dropout(dropout)
        self.output_head = nn.Linear(d_model, n_classes)

        # Init output near zero so corrections start small
        nn.init.zeros_(self.output_head.weight)
        nn.init.constant_(self.output_head.bias, 0.0)

    def forward(self, x):
        # x: (B, T, C)
        h = self.input_proj(x)                              # (B, T, d_model)
        residual = h
        h_f = self.ssm_fwd(h)                              # forward
        h_b = self.ssm_bwd(h.flip(1)).flip(1)              # backward
        h   = self.ssm_merge(torch.cat([h_f, h_b], dim=-1))
        h   = self.drop(h)
        h   = self.ssm_norm(h + residual)                  # residual connection
        return self.output_head(h)                          # (B, T, C)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_sed_probs(sed_dir: str) -> tuple[np.ndarray, list]:
    """Load SED all-soundscape probs. Returns (N, 234) and row_ids list."""
    path = Path(sed_dir) / 'all_ss_probs.npz'
    if not path.exists():
        raise FileNotFoundError(f"all_ss_probs.npz not found in {sed_dir}")
    npz = np.load(str(path), allow_pickle=True)
    row_ids = list(npz['row_ids'])
    probs   = npz['probs'].astype(np.float32)
    print(f"SED probs: {probs.shape}  ({len(row_ids)} rows)")
    return probs, row_ids


def load_teacher_probs(teacher_csv: str, row_ids: list,
                       species_cols: list) -> np.ndarray:
    """Align teacher preds to SED row_ids. Returns (N, 234) float32."""
    df = pd.read_csv(teacher_csv)
    df = df.set_index('row_id')

    # Filter species cols that exist in teacher CSV
    valid_cols = [c for c in species_cols if c in df.columns]
    missing    = len(species_cols) - len(valid_cols)
    if missing:
        print(f"  WARNING: {missing} species not in teacher CSV — filling with 0")

    aligned = np.zeros((len(row_ids), len(species_cols)), dtype=np.float32)
    found = 0
    for i, rid in enumerate(row_ids):
        if rid in df.index:
            aligned[i, :len(valid_cols)] = df.loc[rid, valid_cols].values.astype(np.float32)
            found += 1
    print(f"Teacher probs aligned: {aligned.shape}  ({found}/{len(row_ids)} matched)")
    return aligned


def get_species_cols(sed_dir: str, taxonomy_csv: str = None) -> list:
    """Get 234 species column names from taxonomy or from npz."""
    if taxonomy_csv and Path(taxonomy_csv).exists():
        tax = pd.read_csv(taxonomy_csv)
        return tax['primary_label'].astype(str).tolist()
    # Fallback: infer from teacher CSV header
    return [str(i) for i in range(NUM_CLASSES)]


def group_by_file(row_ids: list, probs: np.ndarray) -> tuple[list, np.ndarray]:
    """Group (N, 234) flat rows into (n_files, N_WINDOWS, 234) file-level tensor.

    Returns (file_ids, probs_3d) where probs_3d is (n_files, N_WINDOWS, 234),
    padded with zeros if a file has fewer than N_WINDOWS rows.
    """
    # row_id format: <file_stem>_<offset>  e.g. BC2026_Train_0001_S08_..._5
    # stem = everything before the last underscore+number
    from collections import defaultdict, OrderedDict

    file_map = OrderedDict()
    for i, rid in enumerate(row_ids):
        # file stem = row_id minus trailing _<offset>
        parts = rid.rsplit('_', 1)
        stem  = parts[0] if len(parts) == 2 else rid
        if stem not in file_map:
            file_map[stem] = []
        file_map[stem].append(i)

    file_ids   = list(file_map.keys())
    n_files    = len(file_ids)
    probs_3d   = np.zeros((n_files, N_WINDOWS, probs.shape[1]), dtype=np.float32)

    for fi, (stem, indices) in enumerate(file_map.items()):
        t = min(len(indices), N_WINDOWS)
        probs_3d[fi, :t, :] = probs[indices[:t]]

    print(f"Grouped {len(row_ids)} rows → {n_files} files × {N_WINDOWS} windows")
    return file_ids, probs_3d


# ── Training ──────────────────────────────────────────────────────────────────

def train_corrector(sed_3d: np.ndarray, teacher_3d: np.ndarray,
                    cfg: dict, device: torch.device) -> TemporalResidualCorrector:
    """Train corrector on (n_files, T, 234) SED + teacher arrays."""
    residuals = teacher_3d - sed_3d   # target in prob space, range [-1, 1]
    print(f"Residuals: mean={residuals.mean():.4f}  std={residuals.std():.4f}  "
          f"abs_mean={np.abs(residuals).mean():.4f}")

    n_files = len(sed_3d)
    n_val   = max(1, int(n_files * 0.12))
    rng     = np.random.default_rng(42)
    perm    = rng.permutation(n_files)
    val_i, train_i = perm[:n_val], perm[n_val:]

    X_tr = torch.tensor(sed_3d[train_i],  dtype=torch.float32, device=device)
    Y_tr = torch.tensor(residuals[train_i], dtype=torch.float32, device=device)
    X_va = torch.tensor(sed_3d[val_i],   dtype=torch.float32, device=device)
    Y_va = torch.tensor(residuals[val_i],  dtype=torch.float32, device=device)

    model = TemporalResidualCorrector(
        n_classes=NUM_CLASSES,
        d_model=cfg['d_model'],
        d_state=cfg['d_state'],
        dropout=cfg['dropout'],
    ).to(device)
    print(f"TemporalResidualCorrector: {model.count_parameters():,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['n_epochs'], eta_min=cfg['lr'] * 0.01
    )

    # Mini-batch training
    batch_size = cfg.get('batch_size', 64)
    n_tr = len(train_i)
    best_val, best_state, wait = float('inf'), None, 0

    for epoch in range(cfg['n_epochs']):
        model.train()
        epoch_loss = 0.0; n_batches = 0
        idx_perm = torch.randperm(n_tr)
        for b_start in range(0, n_tr, batch_size):
            b_idx = idx_perm[b_start:b_start + batch_size]
            pred  = model(X_tr[b_idx])
            loss  = F.mse_loss(pred, Y_tr[b_idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item(); n_batches += 1
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = F.mse_loss(model(X_va), Y_va).item()

        if val_loss < best_val:
            best_val  = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  ep {epoch+1:3d}: train={epoch_loss/n_batches:.6f}  "
                  f"val={val_loss:.6f}  wait={wait}")

        if wait >= cfg['patience']:
            print(f"  Early stop at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    print(f"Best val MSE: {best_val:.6f}")
    return model


# ── Apply & save ──────────────────────────────────────────────────────────────

def apply_and_save(model: TemporalResidualCorrector,
                   sed_3d: np.ndarray,
                   sed_row_ids: list,
                   file_ids: list,
                   alpha: float,
                   sed_dir: str,
                   device: torch.device) -> None:
    """Apply corrector to full SED preds, save corrected all_ss_probs_corrected.npz."""
    model.eval()
    n_files  = len(file_ids)
    batch_sz = 128
    corr_3d  = np.zeros_like(sed_3d)

    sed_t = torch.tensor(sed_3d, dtype=torch.float32)
    with torch.no_grad():
        for b in range(0, n_files, batch_sz):
            corr_3d[b:b+batch_sz] = model(sed_t[b:b+batch_sz].to(device)).cpu().numpy()

    corrected_3d = np.clip(sed_3d + alpha * corr_3d, 0.0, 1.0).astype(np.float32)

    print(f"Correction (alpha={alpha}): "
          f"mean_abs={np.abs(corr_3d).mean():.4f}  max={np.abs(corr_3d).max():.4f}")
    print(f"SED range:       [{sed_3d.min():.3f}, {sed_3d.max():.3f}]")
    print(f"Corrected range: [{corrected_3d.min():.3f}, {corrected_3d.max():.3f}]")

    # Flatten back to (N, 234) preserving original row_id order
    # Build file→window offset mapping from sed_row_ids
    from collections import OrderedDict
    file_to_fidx = {stem: fi for fi, stem in enumerate(file_ids)}
    file_win_cnt = {stem: 0 for stem in file_ids}

    corrected_flat = np.zeros((len(sed_row_ids), NUM_CLASSES), dtype=np.float32)
    for i, rid in enumerate(sed_row_ids):
        parts = rid.rsplit('_', 1)
        stem  = parts[0] if len(parts) == 2 else rid
        fi    = file_to_fidx.get(stem)
        if fi is None:
            corrected_flat[i] = sed_3d.reshape(-1, NUM_CLASSES)[i]
            continue
        wi = file_win_cnt[stem]
        if wi < N_WINDOWS:
            corrected_flat[i] = corrected_3d[fi, wi]
            file_win_cnt[stem] += 1

    out_path = Path(sed_dir) / 'all_ss_probs_corrected.npz'
    np.savez_compressed(str(out_path),
                        row_ids=np.array(sed_row_ids),
                        probs=corrected_flat)
    print(f"Saved corrected probs → {out_path}  ({len(sed_row_ids)} rows)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train SED Residual Corrector')
    parser.add_argument('--sed_dir',    required=True,
                        help='SED experiment dir (has all_ss_probs.npz)')
    parser.add_argument('--teacher',    required=True,
                        help='Perch teacher CSV (perch_teacher_aug_all_ss.csv)')
    parser.add_argument('--round',      type=int, required=True,
                        help='NS round number (for checkpoint naming)')
    parser.add_argument('--taxonomy',   default='birdclef-2026/taxonomy.csv',
                        help='taxonomy.csv with primary_label column')
    parser.add_argument('--alpha',      type=float, default=0.40,
                        help='Correction weight: corrected = SED + alpha * correction')
    parser.add_argument('--out_ckpt',   default=None,
                        help='Output checkpoint path (default: checkpoints/sed_corrector_rR.pt)')
    parser.add_argument('--d_model',    type=int,   default=128)
    parser.add_argument('--d_state',    type=int,   default=16)
    parser.add_argument('--dropout',    type=float, default=0.10)
    parser.add_argument('--lr',         type=float, default=3e-4)
    parser.add_argument('--n_epochs',   type=int,   default=80)
    parser.add_argument('--patience',   type=int,   default=15)
    parser.add_argument('--batch_size', type=int,   default=64)
    parser.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    out_ckpt = args.out_ckpt or f'checkpoints/sed_corrector_r{args.round}.pt'
    Path(out_ckpt).parent.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    sed_probs, sed_row_ids = load_sed_probs(args.sed_dir)

    # Species column names
    tax_path = Path(args.taxonomy)
    if tax_path.exists():
        tax = pd.read_csv(str(tax_path))
        species_cols = tax['primary_label'].astype(str).tolist()
    else:
        # Try to infer from teacher CSV
        df_head = pd.read_csv(args.teacher, nrows=1)
        species_cols = [c for c in df_head.columns if c != 'row_id']
    species_cols = species_cols[:NUM_CLASSES]

    teacher_probs = load_teacher_probs(args.teacher, sed_row_ids, species_cols)

    # --- Group into file-level tensors ---
    file_ids, sed_3d      = group_by_file(sed_row_ids, sed_probs)
    _,         teacher_3d = group_by_file(sed_row_ids, teacher_probs)

    # --- Train corrector ---
    cfg = {
        'd_model':    args.d_model,
        'd_state':    args.d_state,
        'dropout':    args.dropout,
        'lr':         args.lr,
        'n_epochs':   args.n_epochs,
        'patience':   args.patience,
        'batch_size': args.batch_size,
    }
    model = train_corrector(sed_3d, teacher_3d, cfg, device)

    # --- Save checkpoint ---
    torch.save({
        'model_state_dict': model.state_dict(),
        'cfg': cfg,
        'round': args.round,
        'alpha': args.alpha,
        'n_classes': NUM_CLASSES,
        'd_model': args.d_model,
        'd_state': args.d_state,
    }, out_ckpt)
    print(f"Checkpoint saved → {out_ckpt}")

    # --- Apply corrector & save corrected npz ---
    apply_and_save(model, sed_3d, sed_row_ids, file_ids,
                   alpha=args.alpha, sed_dir=args.sed_dir, device=device)

    print('\nDone.')


if __name__ == '__main__':
    main()
