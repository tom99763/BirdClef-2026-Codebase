"""ProtoSSM — Prototypical State Space Model for BirdCLEF 2026.

Faithful reimplementation of pantanal-distill-birdclef2026.ipynb public notebook.

Architecture:
  Input: Perch v2 embeddings  (B, T, 1536)
  ├── Linear(1536 → d_model) + LayerNorm + GELU + Dropout
  ├── Learnable positional encoding  (1, T, d_model)
  ├── N × Bidirectional SelectiveSSM (Mamba-style gating)
  ├── Prototypical cosine head  → (B, T, n_classes) × temperature
  ├── Gated fusion with Perch logits  (per-class learnable α)
  └── Taxonomic auxiliary head  (B, n_families)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """Simplified Mamba-style selective state space model.

    Faithful to pantanal-distill-birdclef2026.ipynb:
      - in_proj: x → (x_ssm, z)  gating split
      - depthwise conv1d on x_ssm
      - input-dependent dt, B, C
      - sequential scan
      - z-gate:  y = scan_output * silu(z)
      - out_proj
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.in_proj  = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d   = nn.Conv1d(d_model, d_model, d_conv,
                                  padding=d_conv - 1, groups=d_model)
        self.dt_proj  = nn.Linear(d_model, d_model, bias=True)
        self.B_proj   = nn.Linear(d_model, d_state, bias=False)
        self.C_proj   = nn.Linear(d_model, d_state, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.D = nn.Parameter(torch.ones(d_model))

        # HiPPO-initialized A in log space
        A = torch.arange(1, d_state + 1, dtype=torch.float32
                         ).unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T, d_model)"""
        B_sz, T, D = x.shape

        xz          = self.in_proj(x)
        x_ssm, z    = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)

        dt    = F.softplus(self.dt_proj(x_conv))
        B_mat = self.B_proj(x_conv)
        C_mat = self.C_proj(x_conv)
        A     = -torch.exp(self.A_log)

        y = self._selective_scan(x_conv, dt, A, B_mat, C_mat)
        y = y * F.silu(z)
        return self.out_proj(y)

    def _selective_scan(self, x, dt, A, B, C):
        batch, T, D = x.shape
        N = self.d_state

        h  = torch.zeros(batch, D, N, device=x.device, dtype=x.dtype)
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
    """Prototypical State Space Model — faithful to public notebook."""

    def __init__(
        self,
        d_input:      int   = 1536,
        d_model:      int   = 128,
        d_state:      int   = 16,
        n_ssm_layers: int   = 2,
        n_classes:    int   = 234,
        n_windows:    int   = 12,
        dropout:      float = 0.15,
    ):
        super().__init__()
        self.d_model   = d_model
        self.n_classes = n_classes
        self.n_windows = n_windows

        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)

        self.ssm_fwd   = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(n_ssm_layers)])
        self.ssm_bwd   = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(n_ssm_layers)])
        self.ssm_merge = nn.ModuleList([nn.Linear(2 * d_model, d_model) for _ in range(n_ssm_layers)])
        self.ssm_norm  = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_ssm_layers)])
        self.ssm_drop  = nn.Dropout(dropout)

        self.prototypes   = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.proto_temp   = nn.Parameter(torch.tensor(5.0))
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))

        self.n_families  = 0
        self.family_head = None

    @torch.no_grad()
    def init_prototypes_from_data(self, embeddings, labels):
        h = self.input_proj(embeddings)
        h = F.normalize(h, dim=-1)
        for c in range(self.n_classes):
            mask = labels[:, c] > 0.5
            if mask.sum() > 0:
                self.prototypes.data[c] = F.normalize(h[mask].mean(0), dim=0)

    def init_family_head(self, n_families: int) -> None:
        self.n_families  = n_families
        self.family_head = nn.Linear(self.d_model, n_families)

    def forward(self, emb, perch_logits=None):
        B, T, _ = emb.shape

        h = self.input_proj(emb) + self.pos_enc[:, :T, :]

        for fwd, bwd, merge, norm in zip(
            self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm
        ):
            residual = h
            h_f = fwd(h)
            h_b = bwd(h.flip(1)).flip(1)
            h   = merge(torch.cat([h_f, h_b], dim=-1))
            h   = self.ssm_drop(h)
            h   = norm(h + residual)

        h_temporal = h

        h_norm = F.normalize(h,               dim=-1)
        p_norm = F.normalize(self.prototypes,  dim=-1)
        temp   = F.softplus(self.proto_temp)
        sim    = torch.matmul(h_norm, p_norm.T) * temp

        if perch_logits is not None:
            alpha          = torch.sigmoid(self.fusion_alpha)[None, None, :]
            species_logits = alpha * sim + (1 - alpha) * perch_logits
        else:
            species_logits = sim

        family_logits = None
        if self.family_head is not None:
            family_logits = self.family_head(h.mean(dim=1))

        return species_logits, family_logits, h_temporal

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ProtoSSMLoss(nn.Module):
    def __init__(self, pos_weight_cap=30.0, w_distill=0.3, w_family=0.1):
        super().__init__()
        self.pos_weight_cap = pos_weight_cap
        self.w_distill      = w_distill
        self.w_family       = w_family

    def forward(self, species_logits, family_logits, labels, family_labels,
                perch_logits, pos_weight=None):
        pw = pos_weight[None, None, :] if pos_weight is not None else None
        loss_bce     = F.binary_cross_entropy_with_logits(species_logits, labels, pos_weight=pw)
        loss_distill = F.mse_loss(species_logits, perch_logits)
        total        = loss_bce + self.w_distill * loss_distill

        loss_family = torch.tensor(0.0, device=species_logits.device)
        if family_logits is not None and family_labels is not None:
            loss_family = F.binary_cross_entropy_with_logits(family_logits, family_labels)
            total = total + self.w_family * loss_family

        return total, {
            "loss_bce":     loss_bce.item(),
            "loss_distill": loss_distill.item(),
            "loss_family":  loss_family.item(),
            "loss_total":   total.item(),
        }

    @staticmethod
    def compute_pos_weight(labels, cap=30.0):
        flat = labels.reshape(-1, labels.shape[-1])
        n    = flat.shape[0]
        if isinstance(flat, np.ndarray):
            pos  = flat.sum(0).clip(min=1)
            neg  = n - pos
            return (neg / pos).clip(max=cap).astype(np.float32)
        pos  = flat.sum(0).clamp(min=1)
        neg  = n - pos
        return (neg / pos).clamp(max=cap)


def build_proto_ssm(cfg: dict) -> ProtoSSM:
    m = cfg.get("model", {})
    return ProtoSSM(
        d_input      = m.get("d_input",       1536),
        d_model      = m.get("d_model",        128),
        d_state      = m.get("d_state",         16),
        n_ssm_layers = m.get("n_ssm_layers",     2),
        n_classes    = m.get("n_classes",       234),
        n_windows    = m.get("n_windows",        12),
        dropout      = m.get("dropout",        0.15),
    )


# ── V18 additions ──────────────────────────────────────────────────────────────

class TemporalCrossAttention(nn.Module):
    """Multi-head self-attention over the temporal dimension."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, d_model)
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + self.drop(attn_out))


class ProtoSSMv2(nn.Module):
    """V18 ProtoSSM: multi-prototype head + temporal cross-attention + metadata conditioning."""

    def __init__(
        self,
        d_input:          int   = 1536,
        d_model:          int   = 320,
        d_state:          int   = 32,
        n_ssm_layers:     int   = 4,
        n_classes:        int   = 234,
        n_windows:        int   = 12,
        dropout:          float = 0.12,
        n_prototypes:     int   = 2,
        n_sites:          int   = 20,
        meta_dim:         int   = 24,
        use_cross_attn:   bool  = True,
        cross_attn_heads: int   = 8,
        n_families:       int   = 0,
    ):
        super().__init__()
        self.d_model      = d_model
        self.n_classes    = n_classes
        self.n_prototypes = n_prototypes
        self.n_windows    = n_windows

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)

        # Metadata embeddings
        self.site_emb  = nn.Embedding(n_sites, meta_dim)
        self.hour_emb  = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)

        # SSM layers + cross-attention
        self.ssm_fwd   = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(n_ssm_layers)])
        self.ssm_bwd   = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(n_ssm_layers)])
        self.ssm_merge = nn.ModuleList([nn.Linear(2 * d_model, d_model) for _ in range(n_ssm_layers)])
        self.ssm_norm  = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_ssm_layers)])
        self.cross_attn = nn.ModuleList([
            TemporalCrossAttention(d_model, cross_attn_heads, dropout)
            if use_cross_attn else nn.Identity()
            for _ in range(n_ssm_layers)
        ])
        self.ssm_drop = nn.Dropout(dropout)

        # Multi-prototype head
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, n_classes, d_model) * 0.02)
        self.proto_temp = nn.Parameter(torch.tensor(5.0))
        self.proto_mix  = nn.Parameter(torch.ones(n_prototypes) / n_prototypes)

        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))

        # Family head
        self.family_head = nn.Linear(d_model, n_families) if n_families > 0 else None
        self.n_families  = n_families

    def forward(self, emb, perch_logits=None, site_ids=None, hours=None):
        B, T, _ = emb.shape

        h = self.input_proj(emb) + self.pos_enc[:, :T, :]

        # Add metadata conditioning
        if site_ids is not None and hours is not None:
            site_e = self.site_emb(site_ids)   # (B, meta_dim)
            hour_e = self.hour_emb(hours)       # (B, meta_dim)
            meta   = self.meta_proj(torch.cat([site_e, hour_e], dim=-1))  # (B, d_model)
            h = h + meta.unsqueeze(1)

        for fwd, bwd, merge, norm, ca in zip(
            self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm, self.cross_attn
        ):
            residual = h
            h_f = fwd(h)
            h_b = bwd(h.flip(1)).flip(1)
            h   = merge(torch.cat([h_f, h_b], dim=-1))
            h   = self.ssm_drop(h)
            h   = norm(h + residual)
            h   = ca(h)

        h_temporal = h

        # Multi-prototype scoring
        h_norm = F.normalize(h, dim=-1)          # (B, T, d_model)
        temp   = F.softplus(self.proto_temp)
        mix_w  = F.softmax(self.proto_mix, dim=0)  # (n_prototypes,)
        sim = torch.zeros(B, T, self.n_classes, device=h.device, dtype=h.dtype)
        for k in range(self.n_prototypes):
            p_norm = F.normalize(self.prototypes[k], dim=-1)  # (n_classes, d_model)
            sim = sim + mix_w[k] * torch.matmul(h_norm, p_norm.T) * temp

        if perch_logits is not None:
            alpha          = torch.sigmoid(self.fusion_alpha)[None, None, :]
            species_logits = alpha * sim + (1 - alpha) * perch_logits
        else:
            species_logits = sim

        family_logits = None
        if self.family_head is not None:
            family_logits = self.family_head(h.mean(dim=1))

        return species_logits, family_logits, h_temporal

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualSSM(nn.Module):
    """Residual correction model: input = [emb, first_pass_scores] → correction (B,T,n_classes)."""

    def __init__(
        self,
        d_input:      int   = 1536,
        d_scores:     int   = 234,
        d_model:      int   = 128,
        d_state:      int   = 16,
        n_ssm_layers: int   = 2,
        n_windows:    int   = 12,
        dropout:      float = 0.1,
        n_sites:      int   = 20,
        meta_dim:     int   = 16,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(d_input + d_scores, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_enc   = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        self.site_emb  = nn.Embedding(n_sites, meta_dim)
        self.hour_emb  = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)
        self.ssm_fwd   = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(n_ssm_layers)])
        self.ssm_bwd   = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(n_ssm_layers)])
        self.ssm_merge = nn.ModuleList([nn.Linear(2 * d_model, d_model) for _ in range(n_ssm_layers)])
        self.ssm_norm  = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_ssm_layers)])
        self.out_head  = nn.Linear(d_model, d_scores)

    def forward(self, emb, first_pass, site_ids=None, hours=None):
        x = torch.cat([emb, first_pass], dim=-1)
        h = self.input_proj(x) + self.pos_enc[:, :x.shape[1], :]
        if site_ids is not None and hours is not None:
            meta = self.meta_proj(
                torch.cat([self.site_emb(site_ids), self.hour_emb(hours)], dim=-1)
            )
            h = h + meta.unsqueeze(1)
        for fwd, bwd, merge, norm in zip(
            self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm
        ):
            residual = h
            h_f = fwd(h)
            h_b = bwd(h.flip(1)).flip(1)
            h   = merge(torch.cat([h_f, h_b], dim=-1))
            h   = norm(h + residual)
        return self.out_head(h)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
