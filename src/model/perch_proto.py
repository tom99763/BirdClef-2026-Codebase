"""
perch_proto.py — Advanced adapter heads for Perch frozen embeddings.

Three modules:
  1. ProtoHead       — Learnable class prototypes + cosine similarity classifier
                       (Bird-MAE paper: +37pp over linear probing on frozen features)
  2. ProtoCLRLoss    — Domain-invariant contrastive loss (ProtoCLR, arXiv:2409.08589)
                       Forces same-class SS+train_audio embeddings to cluster together
  3. FixMatchEmbLoss — Consistency regularisation in embedding space (FixMatch variant)
                       Uses two augmented views of same embedding (noise + masking)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 1. ProtoHead ───────────────────────────────────────────────────────────────

class ProtoHead(nn.Module):
    """
    Prototypical head: learns C learnable prototype vectors in a projected space.
    Classification score = cosine_similarity(projected_emb, prototype_c).

    Compared to a plain linear head this is more sample-efficient because
    the prototype vectors act as a structured inductive bias (one centre per class).

    Architecture:
        emb (D=1536) → projection MLP (D→proj_dim) → L2-norm
        prototype_c  (proj_dim,)                    → L2-norm
        score_c = dot(normed_emb, normed_proto_c) / temperature
    """

    def __init__(
        self,
        emb_dim: int = 1536,
        proj_dim: int = 512,
        num_classes: int = 234,
        n_blocks: int = 2,
        dropout: float = 0.10,
        temperature: float = 0.05,
    ):
        super().__init__()
        self.temperature = temperature

        # Projection MLP: emb_dim → proj_dim
        layers: list[nn.Module] = []
        in_d = emb_dim
        for i in range(n_blocks - 1):
            layers += [nn.Linear(in_d, proj_dim), nn.LayerNorm(proj_dim), nn.GELU(), nn.Dropout(dropout)]
            in_d = proj_dim
        layers += [nn.Linear(in_d, proj_dim)]
        self.projection = nn.Sequential(*layers)

        # Learnable prototypes (one per class), initialised from N(0, 1/sqrt(proj_dim))
        self.prototypes = nn.Parameter(
            torch.randn(num_classes, proj_dim) / (proj_dim ** 0.5)
        )
        self.num_classes = num_classes
        self.proj_dim = proj_dim

    def forward(self, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        emb : (B, emb_dim)
        Returns
        -------
        logits : (B, C)  — raw cosine scores / temperature (use BCEWithLogits)
        proj   : (B, proj_dim)  — L2-normed projected embedding (for ProtoCLR loss)
        """
        proj = self.projection(emb)                    # (B, proj_dim)
        proj_n = F.normalize(proj, dim=-1)             # unit sphere
        proto_n = F.normalize(self.prototypes, dim=-1) # (C, proj_dim)
        logits = proj_n @ proto_n.T / self.temperature # (B, C)
        return logits, proj_n

    def init_prototypes_from_data(
        self,
        embs: torch.Tensor,   # (N, emb_dim)
        labels: torch.Tensor, # (N, C) one-hot or multi-hot
    ):
        """Warm-start prototypes with mean of projected embeddings per class."""
        self.eval()
        with torch.no_grad():
            proj = F.normalize(self.projection(embs), dim=-1)  # (N, proj_dim)
            for c in range(self.num_classes):
                mask = labels[:, c] > 0.5
                if mask.sum() > 0:
                    self.prototypes.data[c] = proj[mask].mean(0)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── 2. ProtoCLRLoss ────────────────────────────────────────────────────────────

class ProtoCLRLoss(nn.Module):
    """
    ProtoCLR: Prototype-based Contrastive Learning for domain-invariant bird audio.
    Reference: arXiv:2409.08589 (Domain-Invariant Representation Learning of Bird Sounds)

    Standard SupCon loss is O(N²); ProtoCLR is O(N×C) — compare each sample
    to running class prototypes rather than all other samples.

    Usage:
        loss = ProtoCLRLoss(num_classes=234, proj_dim=512)
        # ss_proj  : (B_ss, proj_dim)  — soundscape projections (L2-normed)
        # ta_proj  : (B_ta, proj_dim)  — train-audio projections (L2-normed)
        # ss_labels: (B_ss, C)         — one-hot / soft labels
        # ta_labels: (B_ta, C)         — one-hot labels
        l = loss(ss_proj, ss_labels, ta_proj, ta_labels)
    """

    def __init__(self, num_classes: int = 234, temperature: float = 0.07):
        super().__init__()
        self.C = num_classes
        self.T = temperature

    def _class_prototypes(
        self,
        proj: torch.Tensor,   # (N, D) L2-normed
        labels: torch.Tensor, # (N, C)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (proto (C, D), has_proto (C,))."""
        protos = torch.zeros(self.C, proj.shape[1], device=proj.device, dtype=proj.dtype)
        has_proto = torch.zeros(self.C, device=proj.device, dtype=torch.bool)
        for c in range(self.C):
            mask = labels[:, c] > 0.5
            if mask.sum() > 0:
                protos[c] = F.normalize(proj[mask].mean(0, keepdim=True), dim=-1).squeeze(0)
                has_proto[c] = True
        return protos, has_proto

    def forward(
        self,
        ss_proj:   torch.Tensor,  # (B_ss, D)
        ss_labels: torch.Tensor,  # (B_ss, C)
        ta_proj:   torch.Tensor,  # (B_ta, D)
        ta_labels: torch.Tensor,  # (B_ta, C)
    ) -> torch.Tensor:
        # Build prototypes from BOTH domains combined
        all_proj   = torch.cat([ss_proj,   ta_proj],   dim=0)
        all_labels = torch.cat([ss_labels, ta_labels], dim=0)
        protos, has_proto = self._class_prototypes(all_proj, all_labels)

        total_loss = torch.tensor(0.0, device=ss_proj.device)
        n_terms = 0

        # For each sample, pull toward its class prototype, push away from others
        for proj_batch, lbl_batch in [(ss_proj, ss_labels), (ta_proj, ta_labels)]:
            # sim: (N, C)
            sim = proj_batch @ protos[has_proto].T / self.T  # (N, C_valid)
            valid_idx = has_proto.nonzero(as_tuple=True)[0]  # (C_valid,)
            pos_mask = lbl_batch[:, valid_idx] > 0.5          # (N, C_valid)

            if pos_mask.sum() == 0:
                continue

            log_softmax = F.log_softmax(sim, dim=-1)  # (N, C_valid)
            loss_per_sample = -(log_softmax * pos_mask).sum(-1) / pos_mask.sum(-1).clamp(min=1)
            total_loss = total_loss + loss_per_sample.mean()
            n_terms += 1

        return total_loss / max(n_terms, 1)


# ── 3. FixMatchEmbLoss ─────────────────────────────────────────────────────────

class FixMatchEmbLoss(nn.Module):
    """
    FixMatch in embedding space for unlabeled soundscape windows.

    Two augmented views of the same embedding (weak vs strong):
      - weak  : small Gaussian noise (sigma_w)
      - strong: larger noise + random feature masking

    If max(sigmoid(teacher(weak_view))) >= conf_threshold → use teacher's
    hard pseudo label for consistency loss on the strong view.

    This is a lightweight semi-supervised method that doesn't require
    unlabeled audio — it works purely in the frozen embedding space.
    """

    def __init__(
        self,
        conf_threshold: float = 0.70,
        noise_w: float = 0.02,
        noise_s: float = 0.08,
        mask_ratio: float = 0.15,
    ):
        super().__init__()
        self.conf_threshold = conf_threshold
        self.noise_w = noise_w
        self.noise_s = noise_s
        self.mask_ratio = mask_ratio

    @torch.no_grad()
    def _augment_weak(self, emb: torch.Tensor) -> torch.Tensor:
        return emb + torch.randn_like(emb) * self.noise_w

    @torch.no_grad()
    def _augment_strong(self, emb: torch.Tensor) -> torch.Tensor:
        out = emb + torch.randn_like(emb) * self.noise_s
        mask = torch.rand(emb.shape, device=emb.device) < self.mask_ratio
        out = out.clone()
        out[mask] = 0.0
        return out

    def forward(
        self,
        model: nn.Module,        # must output (logits, ...) from embedding
        unlabeled_emb: torch.Tensor,  # (B, D) frozen Perch embeddings
    ) -> tuple[torch.Tensor, float]:
        """
        Returns (loss, frac_above_threshold).
        loss = 0 if no sample passes confidence threshold.
        """
        B = unlabeled_emb.shape[0]

        weak  = self._augment_weak(unlabeled_emb)
        strong = self._augment_strong(unlabeled_emb)

        # Teacher prediction on weak view (no grad)
        with torch.no_grad():
            logits_w, *_ = model(weak)
            probs_w = torch.sigmoid(logits_w)          # (B, C)
            max_conf, pseudo_cls = probs_w.max(dim=-1)  # (B,)
            mask = max_conf >= self.conf_threshold      # (B,)

        frac = float(mask.float().mean().item())
        if mask.sum() == 0:
            return torch.tensor(0.0, device=unlabeled_emb.device, requires_grad=False), frac

        # Student prediction on strong view (with grad)
        logits_s, *_ = model(strong[mask])
        pseudo_lbl = (probs_w[mask] >= 0.5).float()   # hard multi-label from weak

        loss = F.binary_cross_entropy_with_logits(logits_s, pseudo_lbl)
        return loss, frac
