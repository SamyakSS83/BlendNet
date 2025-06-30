#!/usr/bin/env python3
"""
Inspect smi-TED configuration to understand embedding dimensions
"""

import os
import sys
import torch

# Add parent directories to path
sys.path.append('/home/sarvesh/sura/plm_sura/BlendNet/materials.smi-ted/smi-ted/inference/smi_ted_light')

from load import load_smi_ted

def inspect_smi_ted():
    print("Loading smi-TED model...")
    
    # Load smi-TED
    smi_ted_path = '/home/sarvesh/sura/plm_sura/BlendNet/materials.smi-ted/smi-ted/inference/smi_ted_light'
    model = load_smi_ted(
        folder=smi_ted_path,
        ckpt_filename="smi-ted-Light_40.pt"
    )
    
    print("\n=== Model Configuration ===")
    if hasattr(model, 'config'):
        for key, value in model.config.items():
            print(f"{key}: {value}")
    
    print(f"\nModel dimensions:")
    print(f"n_embd: {model.n_embd}")
    print(f"max_len: {model.max_len}")
    print(f"n_vocab: {model.n_vocab}")
    
    # Check autoencoder dimensions
    autoencoder = model.decoder.autoencoder
    print(f"\nAutoencoder dimensions:")
    print(f"Encoder input size: {autoencoder.encoder.fc1.in_features}")
    print(f"Encoder output size (latent): {autoencoder.encoder.fc1.out_features}")
    print(f"Decoder input size (latent): {autoencoder.decoder.fc1.in_features}")
    print(f"Decoder output size: {autoencoder.decoder.rec.out_features}")
    
    # Test with a sample SMILES
    print(f"\n=== Testing embedding generation ===")
    test_smiles = ["CCO", "CC(=O)O", "c1ccccc1"]
    
    try:
        embeddings = model.encode(test_smiles)
        print(f"Embeddings type: {type(embeddings)}")
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Sample embedding (first row, first 10 values): {embeddings.iloc[0, :10].tolist()}")
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_smi_ted()
