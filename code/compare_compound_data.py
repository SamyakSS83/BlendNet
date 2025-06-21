#!/usr/bin/env python3
"""
Compare compound data creation between working BADataset and our interface
"""

import os
import sys
import torch
import pickle
import pandas as pd
from rdkit import Chem

# Add module paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../modules/'))

from modules.common.utils import load_cfg
from modules.interaction_modules.BDB_loaders import BADataset
from feature_generation.compound.Get_Mol_features import get_mol_features, remove_hydrogen

def main():
    # Load config
    config = load_cfg("BindingDB.yml")
    device = torch.device(f"cuda:{config['Train']['device']}")
    
    # Load Ki data to get a test compound
    Ki_data = pd.read_csv(f"{config['Path']['Ki_df']}", sep="\t")
    test_pubchem_id = Ki_data.iloc[0, 1]  # Get first compound ID
    print(f"Test PubChem ID: {test_pubchem_id}")
    
    # Method 1: Load using BADataset (working method)
    print("\n=== Method 1: BADataset (Working) ===")
    UniProt_IDs, PubChem_CIDs, labels = Ki_data.iloc[:, 0].values, Ki_data.iloc[:, 1].values, Ki_data.iloc[:, 2].values
    Interactions_IDs = [f"{u}_{c}" for u, c in zip(UniProt_IDs, PubChem_CIDs)]
    
    Ki_Dataset = BADataset(interaction_IDs=Interactions_IDs, labels=labels,
                          protein_feature_path=config['Path']['Ki_protein_feat'],
                          pocket_path=config['Path']['Ki_pockets'],
                          compound_feature_path=config['Path']['Ligand_graph'],
                          device=device)
    
    # Get the first compound data
    sample = Ki_Dataset[0]
    compound_graph_working = sample['compound_graph']
    
    print(f"Working method - x dtype: {compound_graph_working.x.dtype}")
    print(f"Working method - x shape: {compound_graph_working.x.shape}")
    print(f"Working method - x first values: {compound_graph_working.x[:3]}")
    print(f"Working method - edge_attr dtype: {compound_graph_working.edge_attr.dtype}")
    print(f"Working method - edge_attr first values: {compound_graph_working.edge_attr[:3]}")
    
    # Method 2: Our interface method (failing)
    print("\n=== Method 2: Our Interface (Failing) ===")
    
    # We need the SMILES for this compound - let me try to find it
    # For now, let's use a test SMILES
    test_smiles = "Cc1ccc(COc2ccc3nc(C4C(C(=O)O)C4(C)C)n(Cc4ccc(Br)cc4)c3c2)nc1"
    
    # Parse SMILES with RDKit
    mol = Chem.MolFromSmiles(test_smiles)
    
    # Extract molecular features
    atom_feats_list, edge_idx, edge_feats, _ = get_mol_features(mol)
    
    # Remove hydrogens
    n_atoms, atom_feats_list, edge_feats, edge_idx, _ = remove_hydrogen(
        atom_feats_list, edge_idx, edge_feats
    )
    
    # Convert to tensors with correct dtypes
    x = torch.tensor(atom_feats_list, dtype=torch.long, device=device)
    edge_index = edge_idx.clone().detach().to(device=device)
    edge_attr = edge_feats.clone().detach().to(device=device)
    
    print(f"Our method - x dtype: {x.dtype}")
    print(f"Our method - x shape: {x.shape}")
    print(f"Our method - x first values: {x[:3]}")
    print(f"Our method - edge_attr dtype: {edge_attr.dtype}")
    print(f"Our method - edge_attr first values: {edge_attr[:3]}")
    
    # Compare if they have similar ranges
    print(f"\n=== Comparison ===")
    print(f"Working x min/max: {compound_graph_working.x.min()}/{compound_graph_working.x.max()}")
    print(f"Our x min/max: {x.min()}/{x.max()}")
    print(f"Working edge_attr min/max: {compound_graph_working.edge_attr.min()}/{compound_graph_working.edge_attr.max()}")
    print(f"Our edge_attr min/max: {edge_attr.min()}/{edge_attr.max()}")

if __name__ == "__main__":
    main()
