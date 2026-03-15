"""Perch-based bird classifier.

Architecture:
  - Backbone : Google Perch v2 (TF SavedModel, frozen in embedding_head mode)
  - Head     : Small MLP trained for the 234 BirdClef 2026 species

Modes:
  embedding_head : Perch is a frozen feature extractor; only the head is trained
                   on its 1536-dim embedding output.
  label_head     : Uses Perch's `label` output (species logits) mapped to the
                   234 target species as input features. Preserves Perch's
                   pre-trained species knowledge. ~0.825 zero-shot LB.
  full_finetune  : Gradients flow through the whole model (experimental).
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Optional, Tuple


def _load_label_indices(
    perch_dir: str,
    taxonomy_csv: str,
    sample_submission_csv: str,
) -> Tuple[List[int], int]:
    """Map target species → Perch label indices via scientific names.

    Returns (label_indices, n_perch_classes) where label_indices[i] is the
    index in Perch's label output for target_species[i], or n_perch_classes
    for unmapped species (will be zero after tf.pad).
    """
    labels_csv = os.path.join(perch_dir, "assets", "labels.csv")
    bc_labels = pd.read_csv(labels_csv)
    bc_labels = (bc_labels.reset_index()
                 .rename({"inat2024_fsd50k": "scientific_name", "index": "bc_index"}, axis=1)
                 .set_index("scientific_name"))
    n_perch = len(bc_labels)

    taxonomy = pd.read_csv(taxonomy_csv)
    mapping = taxonomy.join(bc_labels, on="scientific_name", how="left")
    mapping["bc_index"] = mapping["bc_index"].fillna(n_perch).astype(int)
    mapping = mapping[["primary_label", "bc_index"]].set_index("primary_label")

    target_species = pd.read_csv(sample_submission_csv).columns[1:].tolist()
    indices = [int(mapping.loc[pl][0]) if pl in mapping.index else n_perch
               for pl in target_species]
    covered = sum(1 for i in indices if i < n_perch)
    print(f"  Label-head: {covered}/{len(target_species)} species covered by Perch taxonomy")
    return indices, n_perch


class ClassificationHead(tf.keras.Model):
    """Two-layer MLP that maps Perch embeddings to class logits."""

    def __init__(self, num_classes: int, hidden_dim: int = 512, dropout: float = 0.3,
                 num_taxon_classes: int = 0):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_dim)
        self.act = tf.keras.layers.Activation("relu")
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.fc2 = tf.keras.layers.Dense(num_classes)
        if num_taxon_classes > 0:
            self.taxon_head = tf.keras.layers.Dense(num_taxon_classes)
        else:
            self.taxon_head = None

    def call(self, x, training: bool = False):
        x = self.fc1(x)
        x = self.act(x)
        feat = self.dropout(x, training=training)
        species_logits = self.fc2(feat)
        if self.taxon_head is not None:
            taxon_logits = self.taxon_head(feat)
            return species_logits, taxon_logits
        return species_logits


class PerchClassifier:
    """
    Wraps a Perch SavedModel and adds a trainable classification head.

    Args:
        perch_dir   : Path to the Perch TF SavedModel directory.
        num_classes : Number of target species (234 for BirdClef 2026).
        mode        : "embedding_head" or "full_finetune".
        hidden_dim  : Hidden units in the classification head.
        dropout     : Dropout rate in the classification head.
    """

    def __init__(
        self,
        perch_dir: str,
        num_classes: int,
        mode: str = "embedding_head",
        hidden_dim: int = 512,
        dropout: float = 0.3,
        embedding_dim: int = None,
        num_taxon_classes: int = 0,
        taxonomy_csv: Optional[str] = None,
        sample_submission_csv: Optional[str] = None,
    ):
        self.mode = mode
        self.num_classes = num_classes
        self._label_indices = None
        self._n_perch_classes = None

        if embedding_dim is not None:
            # Cache mode: pre-computed features, skip loading Perch backbone.
            self._perch = None
            self._embedding_key = None
            self.embedding_dim = embedding_dim
            print(f"  Cache mode: Perch backbone skipped, embedding_dim={embedding_dim}")
        elif mode == "label_head":
            # Label-head mode: use Perch's species logits as features (not embedding).
            # Requires taxonomy mapping files to identify which Perch outputs to use.
            if taxonomy_csv is None or sample_submission_csv is None:
                raise ValueError("label_head mode requires taxonomy_csv and sample_submission_csv")
            print(f"Loading Perch model from: {perch_dir}")
            self._perch = tf.saved_model.load(perch_dir)
            self._embedding_key = "label"
            self._label_indices, self._n_perch_classes = _load_label_indices(
                perch_dir, taxonomy_csv, sample_submission_csv
            )
            self.embedding_dim = num_classes   # 234-dim feature space
        else:
            print(f"Loading Perch model from: {perch_dir}")
            self._perch = tf.saved_model.load(perch_dir)
            self._embedding_key, self.embedding_dim = self._probe_model()
            print(f"  Output key : '{self._embedding_key}'  dim={self.embedding_dim}")

        self.head = ClassificationHead(num_classes, hidden_dim, dropout, num_taxon_classes)
        dummy_emb = tf.zeros((1, self.embedding_dim))
        out = self.head(dummy_emb, training=False)
        # out may be tuple now; count params after
        n_params = sum(int(np.prod(v.shape)) for v in self.head.trainable_variables)
        print(f"  Head params: {n_params:,}")
        if num_taxon_classes > 0:
            print(f"  Taxonomy aux head: {num_taxon_classes} classes")

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _probe_model(self) -> Tuple[str, int]:
        """Run one dummy forward pass to find the best output key and its dim."""
        sig = self._perch.signatures["serving_default"]
        dummy = tf.zeros((1, 32000 * 5), dtype=tf.float32)
        outputs = sig(inputs=dummy)

        for key in ("embedding", "embeddings", "label", "logits"):
            if key in outputs:
                dim = int(outputs[key].shape[-1])
                return key, dim

        # Fallback: first available key
        key = next(iter(outputs.keys()))
        return key, int(outputs[key].shape[-1])

    # ── Public API ───────────────────────────────────────────────────────────

    def extract_embeddings(self, audio: tf.Tensor) -> tf.Tensor:
        """Run Perch and return features for the head.

        embedding_head : returns 1536-dim Perch embedding (stop_gradient applied)
        label_head     : returns 234-dim mapped Perch species logits (stop_gradient)
        """
        sig = self._perch.signatures["serving_default"]
        out = sig(inputs=audio)

        if self.mode == "label_head":
            # Pad label output so OOV index → 0, then gather target species
            label = tf.pad(out["label"], [[0, 0], [0, 1]])
            features = tf.gather(label, self._label_indices, axis=1)
            return tf.stop_gradient(features)   # (N, 234)

        embeddings = out[self._embedding_key]
        if self.mode == "embedding_head":
            embeddings = tf.stop_gradient(embeddings)
        return embeddings

    def __call__(self, audio: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass: raw audio waveform (batch, 160000) → class logits."""
        embeddings = self.extract_embeddings(audio)
        return self.head(embeddings, training=training)

    @property
    def trainable_variables(self) -> List[tf.Variable]:
        """Variables to optimise (head only in embedding_head mode)."""
        return self.head.trainable_variables

    # ── Checkpointing ────────────────────────────────────────────────────────

    def save_head(self, path: str) -> None:
        if not path.endswith(".weights.h5"):
            path = path + ".weights.h5"
        self.head.save_weights(path)
        print(f"  Checkpoint saved → {path}")

    def load_head(self, path: str) -> None:
        if not path.endswith(".weights.h5"):
            path = path + ".weights.h5"
        # Use h5py direct assignment to avoid Keras legacy-format mismatch.
        import h5py
        with h5py.File(path, "r") as wf:
            self.head.fc1.kernel.assign(wf["fc1"]["vars"]["0"][:])
            self.head.fc1.bias.assign(  wf["fc1"]["vars"]["1"][:])
            self.head.fc2.kernel.assign(wf["fc2"]["vars"]["0"][:])
            self.head.fc2.bias.assign(  wf["fc2"]["vars"]["1"][:])
            if self.head.taxon_head is not None and "taxon_head" in wf:
                self.head.taxon_head.kernel.assign(wf["taxon_head"]["vars"]["0"][:])
                self.head.taxon_head.bias.assign(  wf["taxon_head"]["vars"]["1"][:])
        print(f"  Checkpoint loaded ← {path}")
