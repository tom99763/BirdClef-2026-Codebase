"""Perch-based bird classifier.

Architecture:
  - Backbone : Google Perch v2 (TF SavedModel, frozen in embedding_head mode)
  - Head     : Small MLP trained for the 234 BirdClef 2026 species

Modes:
  embedding_head : Perch is a frozen feature extractor; only the head is trained.
                   tf.stop_gradient is applied so no gradient flows through Perch.
  full_finetune  : Gradients flow through the whole model (experimental; may be
                   slow and memory-intensive with a TF SavedModel).
"""

import numpy as np
import tensorflow as tf
from typing import List, Tuple


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
    ):
        self.mode = mode
        self.num_classes = num_classes

        if embedding_dim is not None and mode == "embedding_head":
            # Cache mode: skip loading the heavy Perch backbone entirely.
            self._perch = None
            self._embedding_key = None
            self.embedding_dim = embedding_dim
            print(f"  Cache mode: Perch backbone skipped, embedding_dim={embedding_dim}")
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
        """
        Run the Perch backbone and return embeddings.

        In 'embedding_head' mode gradients are stopped here so only the
        classification head is updated during training.
        """
        sig = self._perch.signatures["serving_default"]
        embeddings = sig(inputs=audio)[self._embedding_key]
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
        self.head.load_weights(path)
        print(f"  Checkpoint loaded ← {path}")
