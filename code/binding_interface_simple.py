#!/usr/bin/env python3
"""
Simple binding affinity predictor using the new BindingPredictor class.
This serves as a lightweight wrapper for backward compatibility.
"""

from bindingdb_interface import BindingPredictor

# Global predictor instance (loaded once)
_predictor = None

def get_predictor():
    """Get or create the global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = BindingPredictor()
    return _predictor

def predict(seq: str, smiles: str):
    """
    Predict Ki and IC50 for a protein sequence and SMILES.
    
    Args:
        seq: Protein amino acid sequence
        smiles: Ligand SMILES string
        
    Returns:
        tuple: (ki_prediction, ic50_prediction)
    """
    predictor = get_predictor()
    return predictor.predict_binding_affinity(seq, smiles)

if __name__ == "__main__":
    seq = (
        "MLTFNHDAPWHTQKTLKTSEFGKSFGTLGHIGNISHQCWAGCAAGGRAVLSGEPEANMDQETVG"
        "NVVLLAIVTLISVVQNGFFAHKVEHESRTQNGRSFQRTGTLAFERVYTANQNCVDAYPTFLAVLWS"
        "AGLLCSQVPAAFAGLMYLFVRQKYFVGYLGERTQSTPGYIFGKRIILFLFLMSVAGIFNYYLIFF"
        "FGSDFENYIKTISTTISPLLLIP"
    )
    smiles = "Cc1ccc(COc2ccc3nc(C4C(C(=O)O)C4(C)C)n(Cc4ccc(Br)cc4)c3c2)nc1"

    ki, ic50 = predict(seq, smiles)
    print(f"Predicted Ki:  {ki:.4f}")
    print(f"Predicted IC50:{ic50:.4f}")
