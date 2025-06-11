#!/usr/bin/env python3
"""
Simple test script for ligand encoder/decoder imports
"""

print("Starting import test...")

try:
    import torch
    print("✓ PyTorch imported successfully, version:", torch.__version__)
except ImportError as e:
    print("✗ PyTorch import failed:", e)
    exit(1)

try:
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    print("✓ Path setup complete")
except Exception as e:
    print("✗ Path setup failed:", e)

try:
    from ligand_encoder_decoder import LigandEncoder, LigandDecoder
    print("✓ LigandEncoder and LigandDecoder imported successfully")
except ImportError as e:
    print("✗ LigandEncoder/LigandDecoder import failed:", e)
    import traceback
    traceback.print_exc()

try:
    from ligand_encoder_decoder import ligand_encoder, ligand_decoder
    print("✓ Convenience functions imported successfully")
except ImportError as e:
    print("✗ Convenience functions import failed:", e)

print("Test complete!")
