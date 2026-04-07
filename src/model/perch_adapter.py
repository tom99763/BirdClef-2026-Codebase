"""
perch_adapter.py — Trainable Residual Adapter on frozen Perch v2 embeddings.

Architecture:
  emb (1536) → LayerNorm → Down(1536→bottleneck) → GELU → Dropout
              → Up(bottleneck→1536) → Residual → LayerNorm
              → ClassHead(1536→234)

Also produces adapted embedding for SupCon loss.

Supports:
  - Multi-label BCE / Focal classification
  - SupCon contrastive loss
  - Mean Teacher EMA (for R3 self-training)
  - Domain Discriminator (for R2 alignment, optional)
"""

import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Losses ───────────────────────────────────────────────────────────────────

class FocalBCELoss(nn.Module):
    """Multi-label focal BCE."""
    def __init__(self, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1 - prob) * (1 - targets)
        focal = ((1 - p_t) ** self.gamma) * bce
        return focal.mean()


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).
    Pulls same-class embeddings together, pushes different-class apart.
    Works with multi-label: a pair is "positive" if they share ≥1 class.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        emb   : (B, D) L2-normalised embeddings
        labels: (B, C) multi-hot label matrix
        """
        B = emb.shape[0]
        if B < 2:
            return emb.sum() * 0.0

        # Cosine similarity matrix
        sim = torch.matmul(emb, emb.T) / self.temperature   # (B, B)

        # Positive mask: same class shared (at least one class overlap)
        label_overlap = (labels @ labels.T) > 0             # (B, B)
        pos_mask = label_overlap & ~torch.eye(B, dtype=torch.bool, device=emb.device)

        if pos_mask.sum() == 0:
            return emb.sum() * 0.0

        # Numerical stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Exclude self from denominator
        exp_sim = torch.exp(sim)
        self_mask = torch.eye(B, dtype=torch.bool, device=emb.device)
        exp_sim = exp_sim.masked_fill(self_mask, 0.0)

        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Mean of positives per anchor
        n_pos = pos_mask.sum(dim=1).float().clamp(min=1)
        loss = -(pos_mask.float() * log_prob).sum(dim=1) / n_pos
        return loss.mean()


class MMDLoss(nn.Module):
    """Maximum Mean Discrepancy for domain alignment (R2)."""
    def __init__(self, kernel_mul: float = 2.0, kernel_num: int = 5):
        super().__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def _gaussian_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n = x.shape[0] + y.shape[0]
        total = torch.cat([x, y], dim=0)
        total_sq = (total ** 2).sum(1, keepdim=True)
        dist = total_sq + total_sq.T - 2 * total @ total.T
        dist = dist.clamp(min=0)
        bandwidth = dist.sum() / (n ** 2 - n + 1e-8)
        kernels = []
        for i in range(self.kernel_num):
            bw = bandwidth * (self.kernel_mul ** (i - self.kernel_num // 2))
            kernels.append(torch.exp(-dist / (2 * bw + 1e-8)))
        return sum(kernels)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        ns, nt = src.shape[0], tgt.shape[0]
        if ns < 2 or nt < 2:
            return src.sum() * 0.0
        K = self._gaussian_kernel(src, tgt)
        Kss = K[:ns, :ns].mean()
        Ktt = K[ns:, ns:].mean()
        Kst = K[:ns, ns:].mean()
        return (Kss + Ktt - 2 * Kst).clamp(min=0)


# ── Adapter Model ─────────────────────────────────────────────────────────────

class PerchAdapter(nn.Module):
    """
    Residual bottleneck adapter on frozen 1536-dim Perch embeddings.

    Can optionally add a second residual block and a cross-window attention
    layer when operating on sequences (T windows from one file).
    """

    def __init__(
        self,
        emb_dim: int = 1536,
        bottleneck: int = 512,
        num_classes: int = 234,
        dropout: float = 0.15,
        n_blocks: int = 2,
        use_seq_attn: bool = False,
        seq_heads: int = 8,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.use_seq_attn = use_seq_attn

        # ── Residual adapter blocks ──
        blocks = []
        for _ in range(n_blocks):
            blocks.append(nn.Sequential(
                nn.LayerNorm(emb_dim),
                nn.Linear(emb_dim, bottleneck),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck, emb_dim),
                nn.Dropout(dropout),
            ))
        self.blocks = nn.ModuleList(blocks)

        # ── Optional: cross-window self-attention (file-level context) ──
        if use_seq_attn:
            self.seq_attn = nn.MultiheadAttention(
                emb_dim, seq_heads, dropout=dropout, batch_first=True
            )
            self.seq_norm = nn.LayerNorm(emb_dim)

        # ── Final projection for contrastive ──
        self.proj_norm = nn.LayerNorm(emb_dim)
        self.proj_head = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )

        # ── Classification head ──
        self.cls_head = nn.Linear(emb_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        emb: torch.Tensor,
        seq_len: Optional[int] = None,
    ):
        """
        Args:
            emb     : (B, D) flat window embeddings  OR  (B, T, D) sequence
            seq_len : if given, reshape (B*T, D) → (B, T, D) for seq_attn

        Returns:
            logits       : (B, C)
            adapted_emb  : (B, D)   — used for contrastive loss
            proj_emb     : (B, 128) — L2-normalised projection for SupCon
        """
        x = emb  # (B, D)

        # Residual adapter blocks
        for blk in self.blocks:
            x = x + blk(x)

        # Optional sequence-level attention
        if self.use_seq_attn and seq_len is not None:
            B_full = x.shape[0]
            n_files = B_full // seq_len
            x_seq = x.view(n_files, seq_len, self.emb_dim)
            attn_out, _ = self.seq_attn(x_seq, x_seq, x_seq)
            x_seq = self.seq_norm(x_seq + attn_out)
            x = x_seq.view(B_full, self.emb_dim)

        adapted_emb = x
        logits = self.cls_head(self.proj_norm(x))

        # Contrastive projection
        proj = self.proj_head(self.proj_norm(x))
        proj_emb = F.normalize(proj, dim=-1)

        return logits, adapted_emb, proj_emb

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Mean Teacher EMA ──────────────────────────────────────────────────────────

class MeanTeacher:
    """
    EMA-based Mean Teacher for self-training (R3).
    Teacher weights = exponential moving average of student.
    """
    def __init__(self, student: PerchAdapter, alpha: float = 0.999):
        self.alpha = alpha
        self.teacher = copy.deepcopy(student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: PerchAdapter):
        for t, s in zip(self.teacher.parameters(), student.parameters()):
            t.data.mul_(self.alpha).add_(s.data, alpha=1.0 - self.alpha)

    def __call__(self, *args, **kwargs):
        return self.teacher(*args, **kwargs)
