#!/usr/bin/env python3
"""
Example usage of the BindingDB interface client.

This script demonstrates how to call predict_Ki and predict_IC50 for a
single protein-compound pair.
"""
import os
import sys

# ensure module path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from bindingdb_interface import predict_Ki, predict_IC50


def main():
    # Example IDs (replace with real IDs available in your data)
    protein_id = 'P12345'
    compound_id = 0

    print(f"Predicting Ki for protein {protein_id}, compound {compound_id}...")
    ki_value = predict_Ki(protein_id, compound_id)
    print(f"Predicted Ki: {ki_value}")

    print(f"Predicting IC50 for protein {protein_id}, compound {compound_id}...")
    ic50_value = predict_IC50(protein_id, compound_id)
    print(f"Predicted IC50: {ic50_value}")

    # Simple assertion: outputs should be floats or convertible
    assert isinstance(ki_value, float), f"Expected float, got {type(ki_value)}"
    assert isinstance(ic50_value, float), f"Expected float, got {type(ic50_value)}"
    print("✓ BindingDB interface example passed")


if __name__ == '__main__':
    main()
