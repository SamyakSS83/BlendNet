#!/usr/bin/env python3
"""
Example usage of the Pseq2Sites interface client.

This script demonstrates how to call get_protein_matrix and
get_protein_vector on dummy data.
"""
import os
import sys
import numpy as np

# ensure module path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from pseq2sites_interface import get_protein_matrix, get_protein_vector


def create_dummy_input(length=100, dim=1024):
    """Generate a random ProtBERT-like feature matrix and dummy sequence."""
    # random feature matrix
    features = np.random.randn(length, dim).astype(np.float32)
    # random amino-acid sequence
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    sequence = ''.join(np.random.choice(amino_acids, length))
    return features, sequence


def main():
    # create dummy inputs
    feat, seq = create_dummy_input(length=50)

    print("Generating per-residue embedding matrix...")
    matrix = get_protein_matrix(feat, seq)
    print(f"Matrix shape: {matrix.shape} (should be 50 x D)")

    print("Generating mean-pooled vector embedding...")
    vector = get_protein_vector(feat, seq)
    print(f"Vector shape: {vector.shape} (should be D, e.g. 256)")
    print(vector)

    # -- Simple tests for shape correctness --
    assert matrix.ndim == 2 and matrix.shape[0] == len(seq), \
        f"Expected matrix shape ({{len(seq)}}, D), got {matrix.shape}"
    assert vector.ndim == 1 and vector.shape[0] == matrix.shape[1], \
        f"Expected vector length {matrix.shape[1]}, got {vector.shape[0]}"
    print("✓ Interface example tests passed")


if __name__ == '__main__':
    main()
