#!/usr/bin/env python3
"""
Simple API for Pseq2Sites embedding model.

Provides easy functions to embed single or multiple proteins and return
per-residue matrices or mean-pooled vectors.
"""

import numpy as np
from modules.pocket_modules.pseq2sites_embeddings import Pseq2SitesEmbeddings


class Pseq2SitesClient:
    """
    Client for generating embeddings using Pseq2Sites model.
    """
    def __init__(self, config_path=None, checkpoint_path=None, device="auto"):
        self.model = Pseq2SitesEmbeddings(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device
        )

    def embed(
        self,
        protein_features: dict,
        protein_sequences: dict,
        batch_size: int = 32,
        return_attention: bool = False
    ) -> dict:
        """
        Embed multiple proteins at once.

        Args:
            protein_features: Mapping from protein_id to ProtBERT features (L×1024 numpy arrays).
            protein_sequences: Mapping from protein_id to amino acid sequences (str).
            batch_size: Inference batch size.
            return_attention: Whether to return attention weights.

        Returns:
            Dict of the same keys mapping to dicts with:
              - 'sequence_embeddings': (L×D) numpy array
              - 'protein_embeddings': (L×D) numpy array
              - ... additional keys
        """
        return self.model.extract_embeddings(
            protein_features=protein_features,
            protein_sequences=protein_sequences,
            batch_size=batch_size,
            return_predictions=False,
            return_attention=return_attention
        )

    def embed_single(
        self,
        protein_feature: np.ndarray,
        sequence: str,
        mean_pool: bool = False
    ) -> np.ndarray:
        """
        Embed a single protein and return either a matrix or vector.

        Args:
            protein_feature: ProtBERT features for one protein (L×1024 numpy array).
            sequence: Amino acid sequence (length L).
            mean_pool: If True, return a mean-pooled vector (D,) instead of the full matrix.

        Returns:
            Numpy array of shape (L×D) or (D,) if mean_pool is True.
        """
        pid = "P0"
        feats = {pid: protein_feature}
        seqs = {pid: sequence}
        result = self.embed(feats, seqs, batch_size=1)[pid]
        seq_emb = result["sequence_embeddings"]  # shape (L, D)
        if mean_pool:
            return np.mean(seq_emb, axis=0)
        return seq_emb


# Convenience functions
_default_client = Pseq2SitesClient()


def get_protein_matrix(
    protein_feature: np.ndarray,
    sequence: str,
    device: str = "auto"
) -> np.ndarray:
    """
    Quickly get per-residue embedding matrix for one protein.
    """
    return Pseq2SitesClient(device=device).embed_single(
        protein_feature, sequence, mean_pool=False
    )


def get_protein_vector(
    protein_feature: np.ndarray,
    sequence: str,
    device: str = "auto"
) -> np.ndarray:
    """
    Quickly get a single vector embedding for one protein.
    """
    return Pseq2SitesClient(device=device).embed_single(
        protein_feature, sequence, mean_pool=True
    )
