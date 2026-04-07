#!/usr/bin/env python3
"""
CLAP Cross-Domain Contrastive Bird Classifier
================================================
CLAP audio + text encoders are COMPLETELY FROZEN (eval + no_grad).
Only the lightweight projection head and classifier are trained.

Two-stage training:
  Stage 1: SupConLoss (cross-domain contrastive) — learns domain-invariant embeddings
  Stage 2: FocalBCE + SupConLoss×0.1 (classification fine-tuning)

Usage:
  # Step 0: Pre-extract CLAP embeddings (run once)
  python train_clap.py --config configs/clap_v1.yaml --extract_only

  # Step 1: Stage 1 — contrastive pre-training
  python train_clap.py --config configs/clap_v1.yaml --stage 1 --device cuda:1

  # Step 2: Stage 2 — classification fine-tuning
  python train_clap.py --config configs/clap_v1.yaml --stage 2 --device cuda:1
"""

import argparse
import random
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import yaml
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from transformers import ClapModel, ClapProcessor

warnings.filterwarnings("ignore")

NUM_CLASSES = 234
EMBED_DIM   = 512   # CLAP audio projection output

# Time-slot prototypes (Plan A): 4 slots × 234 species → (4, 234, 512)
TIME_SLOTS = [
    "dawn chorus (5am-8am)",        # slot 0: highest bird activity
    "morning (8am-12pm)",           # slot 1
    "afternoon (12pm-5pm)",         # slot 2
    "night (8pm-5am)",              # slot 3: nocturnal species
]

def map_hour_to_slot(hour: int) -> int:
    """Map hour-of-day (0-23) to TIME_SLOTS index."""
    if 5 <= hour < 8:
        return 0  # dawn
    elif 8 <= hour < 12:
        return 1  # morning
    elif 12 <= hour < 20:
        return 2  # afternoon
    else:
        return 3  # night


# ── Frozen CLAP Feature Extractor ─────────────────────────────────────────────

class ClapExtractor:
    """
    Frozen CLAP encoder — always eval() + no_grad().
    Never call .train() on this object.
    """
    def __init__(self, pretrained: str = "laion/clap-htsat-unfused",
                 device: str = "cuda"):
        print(f"Loading CLAP from {pretrained} ...")
        clap = ClapModel.from_pretrained(pretrained)
        clap.eval()
        for p in clap.parameters():
            p.requires_grad = False

        self.audio_model      = clap.audio_model.to(device)
        self.audio_projection = clap.audio_projection.to(device)
        self.text_model       = clap.text_model.to(device)
        self.text_projection  = clap.text_projection.to(device)
        self.processor        = ClapProcessor.from_pretrained(pretrained)
        self.device           = device
        print("CLAP loaded — all parameters frozen.")

    @torch.no_grad()
    def extract_audio(self, wav_48k_np: np.ndarray) -> np.ndarray:
        """wav_48k_np: (T,) float32 at 48 kHz → (512,) L2-normalised numpy."""
        inputs = self.processor(
            audio=wav_48k_np, sampling_rate=48000, return_tensors="pt"
        )
        feat = inputs["input_features"].to(self.device)
        out  = self.audio_model(input_features=feat)
        emb  = self.audio_projection(out.pooler_output)      # (1, 512)
        emb  = F.normalize(emb, dim=-1).squeeze(0)           # (512,)
        return emb.cpu().numpy()

    @torch.no_grad()
    def extract_text(self, text: str) -> np.ndarray:
        """text → (512,) L2-normalised numpy."""
        inputs = self.processor(
            text=text, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()
                  if k in ("input_ids", "attention_mask")}
        out = self.text_model(**inputs)
        emb = self.text_projection(out.pooler_output)         # (1, 512)
        emb = F.normalize(emb, dim=-1).squeeze(0)
        return emb.cpu().numpy()


# ── Trainable Head (only part that updates) ───────────────────────────────────

class ClapBirdHead(nn.Module):
    """Lightweight trainable head on top of frozen CLAP 512-dim embeddings."""
    def __init__(self, embed_dim: int = EMBED_DIM, proj_dim: int = 256,
                 num_classes: int = NUM_CLASSES, dropout: float = 0.1):
        super().__init__()
        # Projection head for SupConLoss
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
        # Classification head for FocalBCE
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, emb: torch.Tensor):
        """emb: (B, 512) — pre-computed frozen CLAP embedding."""
        logits = self.classifier(emb)
        proj   = F.normalize(self.proj(emb), dim=-1)
        return {"logits": logits, "proj": proj}


# ── Losses ────────────────────────────────────────────────────────────────────

class CrossDomainSupConLoss(nn.Module):
    """
    Supervised Contrastive Loss across clean/wild domains.
    Positive = same species, different domain (or same domain).
    In-batch negatives = different species.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temp = temperature

    def forward(self, z_clean: torch.Tensor, z_wild: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        z_clean, z_wild: (B, D) L2-normalised embeddings
        labels: (B,) species indices
        """
        B = z_clean.shape[0]
        z = torch.cat([z_clean, z_wild], dim=0)      # (2B, D)
        lbl = labels.repeat(2)                        # (2B,)

        sim = torch.mm(z, z.T) / self.temp            # (2B, 2B)
        sim.fill_diagonal_(-1e9)                       # exclude self

        pos_mask = (lbl.unsqueeze(0) == lbl.unsqueeze(1))  # (2B, 2B)
        pos_mask.fill_diagonal_(False)

        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        n_pos    = pos_mask.float().sum(1).clamp(min=1)
        loss     = -(log_prob * pos_mask).sum(1) / n_pos
        return loss.mean()


class FocalBCELoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p   = torch.sigmoid(logits)
        ce  = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt  = targets * p + (1 - targets) * (1 - p)
        return (ce * (1 - pt) ** self.gamma).mean()


# ── Dataset ───────────────────────────────────────────────────────────────────

class CrossDomainEmbeddingDataset(Dataset):
    """
    Loads pre-computed CLAP embeddings (.npy) from disk.
    Training is purely MLP forward — very fast.
    Each item: (clean_emb, wild_emb, multi_hot_label)
    """
    def __init__(self, clean_emb_dir: str, wild_emb_dir: str,
                 species_list: list,  # ordered list of all 234 species
                 clean_records: list, # [{"emb_file": str, "labels": [str]}]
                 wild_records:  list, # [{"emb_file": str, "label": str, "conf": float}]
                 min_conf: float = 0.8):
        self.clean_dir = Path(clean_emb_dir)
        self.wild_dir  = Path(wild_emb_dir)
        self.sp2idx    = {sp: i for i, sp in enumerate(species_list)}

        # Index clean embeddings per species
        self.clean = defaultdict(list)
        for rec in clean_records:
            sp = rec["primary_label"]
            p  = self.clean_dir / rec["emb_file"]
            if p.exists():
                self.clean[sp].append(p)

        # Index wild embeddings per species (high-confidence only)
        self.wild = defaultdict(list)
        for rec in wild_records:
            if rec["conf"] < min_conf:
                continue
            sp = rec["label"]
            p  = self.wild_dir / rec["emb_file"]
            if p.exists():
                self.wild[sp].append(p)

        # Only species with both domains
        self.species = sorted(set(self.clean) & set(self.wild))
        print(f"CrossDomainDataset: {len(self.species)} species | "
              f"clean={sum(len(v) for v in self.clean.values())} | "
              f"wild={sum(len(v) for v in self.wild.values())}")

    def __len__(self):
        return len(self.species) * 8

    def __getitem__(self, idx):
        sp = self.species[idx % len(self.species)]

        clean_emb = np.load(random.choice(self.clean[sp])).astype(np.float32)
        wild_emb  = np.load(random.choice(self.wild[sp])).astype(np.float32)
        label_idx = self.sp2idx[sp]

        return (torch.from_numpy(clean_emb),
                torch.from_numpy(wild_emb),
                label_idx)


class SingleEmbeddingDataset(Dataset):
    """
    For Stage 2 classification: loads single-domain embeddings with multi-hot labels.
    Includes both train_audio embeddings and pseudo-label soundscape embeddings.
    """
    def __init__(self, emb_dir: str, records: list, species_list: list,
                 pseudo_weight: float = 1.0):
        self.emb_dir      = Path(emb_dir)
        self.sp2idx       = {sp: i for i, sp in enumerate(species_list)}
        self.n_classes    = len(species_list)
        self.pseudo_weight = pseudo_weight
        self.items        = []  # [(emb_path, multi_hot, weight)]
        for rec in records:
            p = self.emb_dir / rec["emb_file"]
            if not p.exists():
                continue
            label_vec = np.zeros(self.n_classes, dtype=np.float32)
            for sp in rec.get("labels", [rec.get("primary_label", "")]):
                if sp in self.sp2idx:
                    label_vec[self.sp2idx[sp]] = 1.0
            w = rec.get("weight", 1.0)
            self.items.append((p, label_vec, w))
        print(f"SingleEmbeddingDataset: {len(self.items)} samples")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label_vec, weight = self.items[idx]
        emb = np.load(path).astype(np.float32)
        return (torch.from_numpy(emb),
                torch.from_numpy(label_vec),
                torch.tensor(weight, dtype=torch.float32))


# ── Text Prompt Builder ───────────────────────────────────────────────────────

def build_text_prompt(row, taxonomy_df) -> str:
    """
    Build an enriched text prompt for a train_audio recording.
    Includes: species name, call type (if available), secondary species,
    and location anchor.
    """
    try:
        sp = taxonomy_df.loc[row["primary_label"]]
        prompt = (f"sound of {sp['common_name']} "
                  f"({sp['scientific_name']}, {sp['class_name']})")
    except (KeyError, TypeError):
        prompt = f"sound of species {row.get('primary_label', 'unknown')}"

    # Call type (skip 'uncertain')
    t = str(row.get("type", "[]"))
    if t not in ("[]", "['uncertain']", "", "nan"):
        try:
            types = eval(t)
            if types and str(types[0]).lower() != "uncertain":
                prompt += f", {types[0]}"
        except Exception:
            pass

    # Secondary species (up to 2)
    sec = str(row.get("secondary_labels", "[]"))
    if sec not in ("[]", "", "nan"):
        try:
            secs = eval(sec)
            names = []
            for s in secs[:2]:
                if s in taxonomy_df.index:
                    names.append(taxonomy_df.loc[s]["common_name"])
            if names:
                prompt += f", with background sounds of {' and '.join(names)}"
        except Exception:
            pass

    prompt += ", recorded in Pantanal, Brazil"
    return prompt


def build_soundscape_prompt(species_scores: dict, taxonomy_df,
                             conf_thr: float = 0.8) -> str:
    """Build text prompt for a soundscape pseudo-label window."""
    sorted_sp = sorted(species_scores.items(), key=lambda x: x[1], reverse=True)
    if not sorted_sp:
        return "wild soundscape recording in Pantanal, Brazil"

    primary = sorted_sp[0][0]
    try:
        sp = taxonomy_df.loc[primary]
        prompt = (f"wild soundscape recording containing "
                  f"{sp['common_name']} ({sp['scientific_name']})")
    except (KeyError, TypeError):
        prompt = f"wild soundscape recording containing species {primary}"

    others = [taxonomy_df.loc[s]["common_name"]
              for s, c in sorted_sp[1:3]
              if c >= conf_thr and s in taxonomy_df.index]
    if others:
        prompt += f" and {', '.join(others)}"

    prompt += ", in Pantanal, Brazil wetland environment"
    return prompt


# ── Audio-Text Alignment Loss ─────────────────────────────────────────────────

class AudioTextAlignmentLoss(nn.Module):
    """
    Pull audio embeddings toward their species text prototype.
    Uses cosine distance: L = 1 - cosine_similarity(audio, text_proto).
    """
    def forward(self, audio_emb: torch.Tensor,
                text_proto: torch.Tensor) -> torch.Tensor:
        """
        audio_emb:  (B, 512) L2-normalised frozen CLAP audio embedding
        text_proto: (B, 512) L2-normalised text prototype for each sample's species
        """
        return (1.0 - F.cosine_similarity(audio_emb, text_proto, dim=-1)).mean()


# ── Pre-computation ───────────────────────────────────────────────────────────

def extract_text_prototypes(taxonomy_csv: str, out_path: str,
                            extractor: "ClapExtractor") -> np.ndarray:
    """
    Pre-compute time-slot text prototypes for all species.
    Output shape: (4, NUM_CLASSES, 512) — saved to out_path.
    Slots: 0=dawn, 1=morning, 2=afternoon, 3=night (see TIME_SLOTS).
    Upload out_path to Kaggle — only ~1.9 MB.
    """
    out_path = Path(out_path)
    if out_path.exists():
        print(f"Text prototypes already exist: {out_path}")
        return np.load(out_path)

    tax = pd.read_csv(taxonomy_csv).set_index("primary_label")
    species_list = sorted(tax.index.tolist())

    protos = np.zeros((len(TIME_SLOTS), len(species_list), EMBED_DIM),
                      dtype=np.float32)

    for s_idx, slot in enumerate(TIME_SLOTS):
        print(f"  Extracting text prototypes — slot {s_idx}: {slot}")
        for c_idx, sp in enumerate(species_list):
            try:
                row = tax.loc[sp]
                prompt = (f"sound of {row['common_name']} "
                          f"({row['scientific_name']}, {row['class_name']}), "
                          f"{slot}, Pantanal wetland, Brazil")
            except (KeyError, TypeError):
                prompt = f"sound of species {sp}, {slot}, Pantanal wetland, Brazil"
            protos[s_idx, c_idx] = extractor.extract_text(prompt)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, protos)
    print(f"Text prototypes saved → {out_path}  shape={protos.shape}  "
          f"size={protos.nbytes / 1e6:.1f} MB")
    return protos


def extract_train_audio_embeddings(train_csv: str, audio_dir: str,
                                   taxonomy_csv: str, out_dir: str,
                                   extractor: ClapExtractor,
                                   sr_in: int = 32000, sr_out: int = 48000,
                                   clip_dur: int = 5):
    """
    Pre-extract CLAP embeddings for all train_audio files.
    Saves: {out_dir}/{primary_label}/{filename_stem}.npy
    """
    out_dir = Path(out_dir)
    df      = pd.read_csv(train_csv)
    tax     = pd.read_csv(taxonomy_csv).set_index("primary_label")

    total = len(df)
    for i, row in df.iterrows():
        sp_dir = out_dir / str(row["primary_label"])
        sp_dir.mkdir(parents=True, exist_ok=True)
        stem   = Path(row["filename"]).stem
        cache  = sp_dir / f"{stem}.npy"
        if cache.exists():
            continue
        fpath  = Path(audio_dir) / row["filename"]
        if not fpath.exists():
            continue
        try:
            wav, sr = torchaudio.load(str(fpath))
            if sr != sr_out:
                wav = torchaudio.functional.resample(wav, sr, sr_out)
            # random crop
            n_samples = sr_out * clip_dur
            max_start = max(0, wav.shape[1] - n_samples)
            start     = random.randint(0, max_start)
            clip      = wav[0, start:start + n_samples].numpy()
            if len(clip) < n_samples:
                clip = np.pad(clip, (0, n_samples - len(clip)))
            emb = extractor.extract_audio(clip)
            np.save(cache, emb)
        except Exception as e:
            print(f"  [WARN] {fpath}: {e}")
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{total}] embeddings extracted")

    print(f"Done: train_audio embeddings → {out_dir}")


def extract_soundscape_embeddings(soundscape_dir: str, pseudo_csv: str,
                                  out_dir: str, extractor: ClapExtractor,
                                  sr_in: int = 32000, sr_out: int = 48000,
                                  clip_dur: int = 5):
    """
    Pre-extract CLAP embeddings for soundscape pseudo-label windows.
    Each row in pseudo_csv has row_id = {soundscape_id}_{offset_sec}.
    """
    out_dir    = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pseudo_df  = pd.read_csv(pseudo_csv)
    ss_dir     = Path(soundscape_dir)

    species_cols = [c for c in pseudo_df.columns
                    if c not in ("row_id", "soundscape_id", "offset_sec",
                                 "primary_label", "secondary_labels")]

    cached = 0
    for _, row in pseudo_df.iterrows():
        row_id    = row["row_id"]         # e.g. "XC123456_25"
        parts     = row_id.rsplit("_", 1)
        ss_id     = parts[0]
        offset    = int(parts[1])

        cache = out_dir / f"{row_id}.npy"
        if cache.exists():
            cached += 1
            continue

        # Find top species for filename organisation
        sp_scores  = {c: float(row[c]) for c in species_cols if float(row[c]) > 0}
        if not sp_scores:
            continue

        ss_file = ss_dir / f"{ss_id}.ogg"
        if not ss_file.exists():
            ss_file = ss_dir / f"{ss_id}.wav"
        if not ss_file.exists():
            continue

        try:
            wav, sr = torchaudio.load(str(ss_file))
            if sr != sr_out:
                wav = torchaudio.functional.resample(wav, sr, sr_out)
            start    = max(0, (offset - clip_dur // 2)) * sr_out
            n_samp   = clip_dur * sr_out
            clip     = wav[0, start:start + n_samp].numpy()
            if len(clip) < n_samp:
                clip = np.pad(clip, (0, n_samp - len(clip)))
            emb = extractor.extract_audio(clip)
            np.save(cache, emb)
        except Exception as e:
            print(f"  [WARN] {ss_file}: {e}")

    print(f"Soundscape embeddings done (cached={cached}) → {out_dir}")


# ── Stage 2 Dataset Builder ───────────────────────────────────────────────────

def build_stage2_datasets(cfg: dict, species_list: list):
    """
    Build train + val datasets for Stage 2.

    Train set:
      - train_audio embeddings (hard labels from train.csv)
      - soundscape embeddings (soft labels: blend hard pseudo + SED NS probs)

    Val set:
      - 20% of soundscape files with ground-truth labels from
        soundscape_labels_csv (GroupKFold by soundscape file).

    Returns (train_dataset, val_embs, val_labels):
      train_dataset : SingleEmbeddingDataset
      val_embs      : torch.Tensor (N_val, 512)
      val_labels    : np.ndarray  (N_val, 234)
    """
    sp2idx   = {sp: i for i, sp in enumerate(species_list)}
    n_cls    = len(species_list)
    train_emb_dir = Path(cfg["data"]["train_emb_dir"])
    wild_emb_dir  = Path(cfg["data"]["wild_emb_dir"])

    # ── Load SED NS probs (soft label teacher) ────────────────────────────────
    sed_ns_path = cfg["data"].get("sed_ns_probs_npz")
    sed_probs   = {}   # {row_id: (234,) float32}
    if sed_ns_path and Path(sed_ns_path).exists():
        print(f"Loading SED NS probs from {sed_ns_path} ...")
        d = np.load(sed_ns_path)
        for rid, prob in zip(d["row_ids"], d["probs"]):
            sed_probs[rid] = prob
        print(f"  Loaded {len(sed_probs)} SED NS rows")
    else:
        print("WARNING: sed_ns_probs_npz not found — using hard labels for soundscape train set")

    alpha = cfg["training"].get("soft_label_alpha", 0.3)  # blend weight for SED NS

    # ── Train records: train_audio (hard labels) ─────────────────────────────
    train_df = pd.read_csv(cfg["data"]["train_csv"])
    train_records = []
    for _, row in train_df.iterrows():
        sp   = str(row["primary_label"])
        stem = Path(row["filename"]).stem
        p    = train_emb_dir / sp / f"{stem}.npy"
        if not p.exists():
            continue
        label_vec = np.zeros(n_cls, dtype=np.float32)
        if sp in sp2idx:
            label_vec[sp2idx[sp]] = 1.0
        sec_labels = str(row.get("secondary_labels", "[]"))
        if sec_labels not in ("[]", "", "nan"):
            try:
                for s in eval(sec_labels):
                    s = str(s)
                    if s in sp2idx:
                        label_vec[sp2idx[s]] = 0.5   # secondary label gets 0.5
            except Exception:
                pass
        train_records.append({
            "emb_file": str(p.relative_to(train_emb_dir)),
            "label_vec": label_vec,
            "weight": 1.0,
        })

    # ── Soundscape records: soft labels ──────────────────────────────────────
    pseudo_df = pd.read_csv(cfg["data"]["pseudo_labels_csv"])
    species_cols = [c for c in pseudo_df.columns if c not in ("row_id", "primary_label", "secondary_labels")]

    # Group soundscape files for train/val split (80/20 by file)
    ss_files = sorted({row["row_id"].rsplit("_", 1)[0]
                       for _, row in pseudo_df.iterrows()})
    np.random.seed(42)
    n_val_files = max(1, int(len(ss_files) * 0.20))
    val_files   = set(np.random.choice(ss_files, n_val_files, replace=False))
    train_files = set(ss_files) - val_files

    sound_train_records = []
    for _, row in pseudo_df.iterrows():
        row_id = row["row_id"]
        ss_id  = row_id.rsplit("_", 1)[0]
        if ss_id not in train_files:
            continue
        p = wild_emb_dir / f"{row_id}.npy"
        if not p.exists():
            continue

        # Multi-label pseudo label: all species above threshold get their score
        sp_scores = {c: float(row[c]) for c in species_cols if float(row[c]) > 0}
        if not sp_scores:
            continue
        multi_label_thr = cfg["training"].get("multi_label_thr", 0.5)
        hard_label = np.zeros(n_cls, dtype=np.float32)
        for sp, score in sp_scores.items():
            if sp in sp2idx and score >= multi_label_thr:
                hard_label[sp2idx[sp]] = score  # keep raw probability (not clipped to 1.0)
        if hard_label.sum() == 0:
            # fallback: top-1 if nothing passes threshold
            top_sp = max(sp_scores, key=sp_scores.get)
            if top_sp in sp2idx:
                hard_label[sp2idx[top_sp]] = sp_scores[top_sp]

        # Blend with SED NS soft label
        if row_id in sed_probs:
            label_vec = (1 - alpha) * hard_label + alpha * sed_probs[row_id]
        else:
            label_vec = hard_label

        sound_train_records.append({
            "emb_file": f"{row_id}.npy",
            "label_vec": label_vec,
            "weight": cfg["training"].get("pseudo_weight", 1.0),
        })

    all_train_records = train_records + sound_train_records
    print(f"Stage2 train: {len(train_records)} clean + {len(sound_train_records)} soundscape = {len(all_train_records)} total")

    # ── Val records: ground truth from soundscape_labels_csv ─────────────────
    ss_labels_csv = cfg["data"].get("soundscape_labels_csv",
                                    "birdclef-2026/train_soundscapes_labels.csv")
    ss_labels_df  = pd.read_csv(ss_labels_csv)

    # NOTE: labeled soundscapes (train_soundscapes_labels.csv) are a completely
    # separate set from pseudo-labeled soundscapes — they do NOT appear in
    # pseudo_labels_csv, so the val_files check is irrelevant here.
    # All labeled soundscapes are used as val (ground-truth labels available).
    val_records = []
    for _, row in ss_labels_df.iterrows():
        fname  = str(row["filename"])
        ss_id  = fname.replace(".ogg", "").replace(".wav", "")
        # Parse start time → offset seconds
        start_str = str(row["start"])
        parts = start_str.split(":")
        if len(parts) == 3:
            offset = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(float(parts[2]))
        else:
            try:
                offset = int(float(start_str))
            except Exception:
                continue
        row_id = f"{ss_id}_{offset + 5}"   # ss probs use end offset
        p = wild_emb_dir / f"{row_id}.npy"
        if not p.exists():
            continue
        labels = str(row["primary_label"]).split(";")
        label_vec = np.zeros(n_cls, dtype=np.float32)
        for lbl in labels:
            lbl = lbl.strip()
            if lbl in sp2idx:
                label_vec[sp2idx[lbl]] = 1.0
        if label_vec.sum() == 0:
            continue
        val_records.append((p, label_vec))

    print(f"Stage2 val : {len(val_records)} labeled windows from {len(val_files)} soundscape files")

    # ── Build train dataset ───────────────────────────────────────────────────
    class _PreLabeledDataset(Dataset):
        """Dataset that uses pre-built label_vec (supports soft labels)."""
        def __init__(self, emb_dir: Path, wild_dir: Path, records: list):
            self.items = []
            for rec in records:
                ef = rec["emb_file"]
                # Try train_emb_dir first, then wild_emb_dir
                p = emb_dir / ef
                if not p.exists():
                    p = wild_dir / ef
                if not p.exists():
                    continue
                self.items.append((p, rec["label_vec"], rec.get("weight", 1.0)))
            print(f"_PreLabeledDataset: {len(self.items)} samples")

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            p, lv, w = self.items[idx]
            emb = np.load(str(p)).astype(np.float32)
            return (torch.from_numpy(emb),
                    torch.from_numpy(lv),
                    torch.tensor(w, dtype=torch.float32))

    train_dataset = _PreLabeledDataset(train_emb_dir, wild_emb_dir, all_train_records)

    # ── Pre-load val embeddings ───────────────────────────────────────────────
    val_embs_list, val_labels_list = [], []
    for p, lv in val_records:
        emb = np.load(str(p)).astype(np.float32)
        val_embs_list.append(emb)
        val_labels_list.append(lv)

    val_embs   = torch.from_numpy(np.stack(val_embs_list))    # (N, 512)
    val_labels = np.stack(val_labels_list)                     # (N, 234)
    return train_dataset, val_embs, val_labels


# ── Training ──────────────────────────────────────────────────────────────────

def train_stage1(model: ClapBirdHead, loader: DataLoader,
                 device: str, cfg: dict,
                 text_prototypes: np.ndarray,
                 species_list: list) -> ClapBirdHead:
    """
    Stage 1: cross-domain contrastive + audio-text alignment.
    text_prototypes: (4, NUM_CLASSES, 512) time-slot prototypes (numpy).
    Uses dawn slot (slot 0) as default anchor for clean train_audio recordings.
    """
    supcon    = CrossDomainSupConLoss(temperature=cfg.get("temperature", 0.07))
    at_align  = AudioTextAlignmentLoss()
    lam_at    = cfg.get("lambda_at_stage1", 0.3)   # audio-text alignment weight

    # Preload all 4 slots to GPU; pick slot per batch if hour info available
    # For train_audio (no timestamp), we average across slots as a general prototype
    proto_all = torch.from_numpy(text_prototypes).to(device)  # (4, C, 512)
    proto_mean = proto_all.mean(0)                             # (C, 512) averaged

    opt   = torch.optim.AdamW(model.parameters(),
                               lr=cfg["lr_stage1"],
                               weight_decay=cfg.get("weight_decay", 1e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["epochs_stage1"] * len(loader)
    )
    model = model.to(device)
    best_loss = float("inf")
    best_state = None

    for epoch in range(cfg["epochs_stage1"]):
        model.train()
        total_loss = 0.0
        for clean_emb, wild_emb, labels in loader:
            clean_emb = clean_emb.to(device)
            wild_emb  = wild_emb.to(device)
            labels    = labels.to(device)

            out_c = model(clean_emb)
            out_w = model(wild_emb)

            # Cross-domain SupCon
            l_supcon = supcon(out_c["proj"], out_w["proj"], labels)

            # Audio-text alignment: pull audio emb toward species text prototype
            # clean_emb is already L2-normalised from CLAP extractor
            text_anchor = proto_mean[labels]   # (B, 512)
            l_at = at_align(clean_emb, text_anchor)

            loss = l_supcon + lam_at * l_at

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"  Stage1 Ep {epoch+1:3d}/{cfg['epochs_stage1']}  "
              f"loss={avg:.4f}  (supcon+at_align)")
        if avg < best_loss:
            best_loss  = avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


def train_stage2(model: ClapBirdHead, train_loader: DataLoader,
                 val_embs: torch.Tensor, val_labels: torch.Tensor,
                 device: str, cfg: dict, out_dir: Path) -> ClapBirdHead:
    """Stage 2: classification fine-tuning with FocalBCE + SupCon auxiliary."""
    focal  = FocalBCELoss(gamma=cfg.get("focal_gamma", 2.0))
    supcon = CrossDomainSupConLoss(temperature=cfg.get("temperature", 0.07))
    lam    = cfg.get("lambda_supcon_stage2", 0.1)

    opt   = torch.optim.AdamW(model.parameters(),
                               lr=cfg["lr_stage2"],
                               weight_decay=cfg.get("weight_decay", 1e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["epochs_stage2"] * len(train_loader)
    )

    model    = model.to(device)
    best_auc = 0.0
    patience = cfg.get("patience", 5)
    no_impr  = 0

    val_embs   = val_embs.to(device)
    val_labels = val_labels.to(device)

    for epoch in range(cfg["epochs_stage2"]):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            # CrossDomain batch: (clean_emb, wild_emb, label_idx)
            # Single batch:      (emb, multi_hot, weight)
            # Distinguish by batch[1] shape: CrossDomain has batch[1] shape == batch[0] shape (both 512-d)
            # Single has batch[1] shape (B, n_cls) which differs from batch[0] (B, 512)
            if len(batch) == 3 and batch[1].shape == batch[0].shape:
                # CrossDomain batch: (clean, wild, label_idx)
                clean_emb, wild_emb, label_idx = batch
                clean_emb  = clean_emb.to(device)
                wild_emb   = wild_emb.to(device)
                label_idx  = label_idx.to(device)
                multi_hot  = F.one_hot(label_idx,
                                       num_classes=NUM_CLASSES).float()
                out_c = model(clean_emb)
                out_w = model(wild_emb)
                l_bce = 0.5 * (focal(out_c["logits"], multi_hot)
                               + focal(out_w["logits"], multi_hot))
                l_con = supcon(out_c["proj"], out_w["proj"], label_idx)
                loss  = l_bce + lam * l_con
            else:
                # Single batch: (emb, multi_hot, weight)
                emb, multi_hot, weight = batch
                emb       = emb.to(device)
                multi_hot = multi_hot.to(device)
                out       = model(emb)
                loss      = focal(out["logits"], multi_hot)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_out  = model(val_embs)
            val_prob = torch.sigmoid(val_out["logits"]).cpu().numpy()

        from sklearn.metrics import roc_auc_score
        try:
            vl_np = val_labels.cpu().numpy()
            # Only compute AUC for classes that have at least one positive example
            active_mask = vl_np.max(axis=0) > 0
            if active_mask.sum() >= 2:
                auc = roc_auc_score(vl_np[:, active_mask], val_prob[:, active_mask],
                                    average="macro")
            else:
                auc = 0.0
        except ValueError:
            auc = 0.0

        avg_loss = total_loss / len(train_loader)
        print(f"  Stage2 Ep {epoch+1:3d}/{cfg['epochs_stage2']}  "
              f"loss={avg_loss:.4f}  val_auc={auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            no_impr  = 0
            torch.save({"state_dict": model.state_dict(),
                        "val_auc": best_auc,
                        "epoch": epoch + 1},
                       out_dir / "clap_head_best.pt")
            print(f"    ✓ New best AUC={best_auc:.4f}")
        else:
            no_impr += 1
            if no_impr >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"Stage 2 complete — best val_auc={best_auc:.4f}")
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       required=True)
    parser.add_argument("--stage",        type=int, default=1,
                        choices=[1, 2],
                        help="Training stage (1=contrastive, 2=classification)")
    parser.add_argument("--extract_only", action="store_true",
                        help="Only extract CLAP embeddings, don't train")
    parser.add_argument("--device",       default="cuda:1")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 0: Extract embeddings ──────────────────────────────────────────
    extractor = ClapExtractor(
        pretrained=cfg.get("clap_pretrained", "laion/clap-htsat-unfused"),
        device=args.device
    )

    train_emb_dir = Path(cfg["data"]["train_emb_dir"])
    wild_emb_dir  = Path(cfg["data"]["wild_emb_dir"])
    wild_npy_count = len(list(wild_emb_dir.glob("*.npy"))) if wild_emb_dir.exists() else 0

    if not train_emb_dir.exists() or args.extract_only:
        print("Extracting train_audio CLAP embeddings ...")
        extract_train_audio_embeddings(
            train_csv=cfg["data"]["train_csv"],
            audio_dir=cfg["data"]["audio_dir"],
            taxonomy_csv=cfg["data"]["taxonomy_csv"],
            out_dir=cfg["data"]["train_emb_dir"],
            extractor=extractor,
        )

    if wild_npy_count == 0 or args.extract_only:
        print("Extracting soundscape CLAP embeddings ...")
        extract_soundscape_embeddings(
            soundscape_dir=cfg["data"]["soundscape_dir"],
            pseudo_csv=cfg["data"]["pseudo_labels_csv"],
            out_dir=cfg["data"]["wild_emb_dir"],
            extractor=extractor,
        )

    # Always extract text prototypes (cheap, ~1.9 MB, needed for Kaggle upload)
    proto_path = cfg["data"].get("text_prototypes_path",
                                 "outputs/clap_embeddings/text_prototypes.npy")
    text_prototypes_np = extract_text_prototypes(
        taxonomy_csv=cfg["data"]["taxonomy_csv"],
        out_path=proto_path,
        extractor=extractor,
    )

    if args.extract_only:
        print("Extraction complete.")
        return

    # ── Build datasets ──────────────────────────────────────────────────────
    tax_df    = pd.read_csv(cfg["data"]["taxonomy_csv"])
    tax_df    = tax_df.set_index("primary_label")
    train_df  = pd.read_csv(cfg["data"]["train_csv"])
    pseudo_df = pd.read_csv(cfg["data"]["pseudo_labels_csv"])

    species_list = sorted(tax_df.index.tolist())

    # Build records for CrossDomainDataset
    clean_records = []
    for _, row in train_df.iterrows():
        sp   = str(row["primary_label"])
        stem = Path(row["filename"]).stem
        clean_records.append({
            "primary_label": sp,
            "emb_file": f"{sp}/{stem}.npy",
        })

    species_cols = [c for c in pseudo_df.columns
                    if c not in ("row_id", "primary_label", "secondary_labels")]
    wild_records = []
    for _, row in pseudo_df.iterrows():
        sp_scores = {c: float(row[c]) for c in species_cols
                     if float(row[c]) > 0}
        if not sp_scores:
            continue
        top_sp  = max(sp_scores, key=sp_scores.get)
        top_conf = sp_scores[top_sp]
        wild_records.append({
            "label": top_sp,
            "conf":  top_conf,
            "emb_file": f"{row['row_id']}.npy",
        })

    dataset = CrossDomainEmbeddingDataset(
        clean_emb_dir=cfg["data"]["train_emb_dir"],
        wild_emb_dir=cfg["data"]["wild_emb_dir"],
        species_list=species_list,
        clean_records=clean_records,
        wild_records=wild_records,
        min_conf=cfg["training"].get("min_pseudo_conf", 0.8),
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # ── Model ───────────────────────────────────────────────────────────────
    model = ClapBirdHead(
        embed_dim=EMBED_DIM,
        proj_dim=cfg["model"].get("proj_dim", 256),
        num_classes=NUM_CLASSES,
        dropout=cfg["model"].get("dropout", 0.1),
    )

    # Load Stage 1 checkpoint for Stage 2
    if args.stage == 2:
        s1_ckpt = out_dir / "clap_head_stage1.pt"
        if s1_ckpt.exists():
            ckpt = torch.load(s1_ckpt, map_location="cpu")
            model.load_state_dict(ckpt["state_dict"])
            print(f"Loaded Stage 1 checkpoint (loss={ckpt.get('loss', 'N/A')})")

    # ── Train ────────────────────────────────────────────────────────────────
    if args.stage == 1:
        print("=" * 60)
        print("Stage 1: Cross-Domain Contrastive Pre-training")
        print("=" * 60)
        model = train_stage1(model, loader, args.device, cfg["training"],
                             text_prototypes=text_prototypes_np,
                             species_list=species_list)
        torch.save({"state_dict": model.state_dict()},
                   out_dir / "clap_head_stage1.pt")
        print(f"Stage 1 checkpoint saved → {out_dir}/clap_head_stage1.pt")

    elif args.stage == 2:
        print("=" * 60)
        print("Stage 2: Classification Fine-tuning")
        print("=" * 60)
        train_dataset, val_embs, val_labels = build_stage2_datasets(cfg, species_list)
        if len(train_dataset) == 0:
            raise RuntimeError("Stage 2 train set is empty — check emb_dirs and pseudo_labels_csv")
        if len(val_embs) == 0:
            raise RuntimeError("Stage 2 val set is empty — check soundscape_labels_csv and wild_emb_dir")
        train_loader2 = DataLoader(
            train_dataset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        val_labels_t = torch.from_numpy(val_labels)
        model = train_stage2(model, train_loader2, val_embs, val_labels_t,
                             args.device, cfg["training"], out_dir)
        print(f"Stage 2 best checkpoint → {out_dir}/clap_head_best.pt")


if __name__ == "__main__":
    main()
