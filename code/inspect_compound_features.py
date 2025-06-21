#!/usr/bin/env python3
"""
Script to inspect the dtypes and format of pre-processed compound features
"""

import torch
import sys
import os

# Add module paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../modules/'))

from modules.common.utils import load_cfg

def main():
    # Load config
    config_path = "BindingDB.yml"
    config = load_cfg(config_path)
    
    print("Loading compound features...")
    compound_data_dict = torch.load(config['Path']['Ligand_graph'])
    
    print(f"Keys in compound_data_dict: {compound_data_dict.keys()}")
    
    atom_features = compound_data_dict['atom_features']
    edge_features = compound_data_dict['edge_features']
    edge_indices = compound_data_dict['edge_indices']
    
    print(f"\nAtom features:")
    print(f"  Shape: {atom_features.shape}")
    print(f"  Dtype: {atom_features.dtype}")
    print(f"  First 10 values: {atom_features[:10]}")
    
    print(f"\nEdge features:")
    print(f"  Shape: {edge_features.shape}")
    print(f"  Dtype: {edge_features.dtype}")
    print(f"  First 10 values: {edge_features[:10]}")
    
    print(f"\nEdge indices:")
    print(f"  Shape: {edge_indices.shape}")
    print(f"  Dtype: {edge_indices.dtype}")
    print(f"  First 10 values: {edge_indices[:, :10]}")

if __name__ == "__main__":
    main()
