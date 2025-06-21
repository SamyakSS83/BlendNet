#!/usr/bin/env python3
"""
Debug script to compare compound data creation methods and identify dtype issues
"""

import os
import sys
import torch
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data, Batch

# Add module paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../modules/'))

from modules.common.utils import load_cfg
from feature_generation.compound.Get_Mol_features import get_mol_features, remove_hydrogen

def create_compound_graph_my_way(smiles: str, device):
    """Create compound graph using my method"""
    print(f"\n=== My Method ===")
    
    # Parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    # Extract molecular features
    atom_feats_list, edge_idx, edge_feats, _ = get_mol_features(mol)
    
    print(f"After get_mol_features:")
    print(f"  atom_feats_list type: {type(atom_feats_list)}")
    print(f"  edge_idx type: {type(edge_idx)}, dtype: {edge_idx.dtype}")
    print(f"  edge_feats type: {type(edge_feats)}, dtype: {edge_feats.dtype}")
    
    # Remove hydrogens
    n_atoms, atom_feats_list, edge_feats, edge_idx, _ = remove_hydrogen(
        atom_feats_list, edge_idx, edge_feats
    )
    
    print(f"After remove_hydrogen:")
    print(f"  atom_feats_list type: {type(atom_feats_list)}")
    print(f"  edge_idx type: {type(edge_idx)}, dtype: {edge_idx.dtype}")
    print(f"  edge_feats type: {type(edge_feats)}, dtype: {edge_feats.dtype}")
    
    # Convert to tensors
    x = torch.tensor(atom_feats_list, dtype=torch.long, device=device)
    edge_index = edge_idx.clone().detach().to(device=device)
    edge_attr = edge_feats.clone().detach().to(device=device)
    
    print(f"Final tensors:")
    print(f"  x shape: {x.shape}, dtype: {x.dtype}")
    print(f"  edge_index shape: {edge_index.shape}, dtype: {edge_index.dtype}")
    print(f"  edge_attr shape: {edge_attr.shape}, dtype: {edge_attr.dtype}")
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n_atoms)
    
    return data

def load_compound_from_dataset(pubchem_id: str, device):
    """Load compound data using the working dataset method"""
    print(f"\n=== Dataset Method ===")
    
    # Load config
    config = load_cfg("BindingDB.yml")
    
    # Load compound data directly (without full BADataset)
    print("Loading compound data...")
    compound_data_dict = torch.load(config['Path']['Ligand_graph'])
    
    compound_ids = compound_data_dict['mol_ids']
    compound_feature_tensor = compound_data_dict['atom_features']
    compound_e_features_tensor = compound_data_dict['edge_features']
    edge_indices = compound_data_dict['edge_indices']
    n_atoms_list = compound_data_dict['n_atoms']
    atom_slices = compound_data_dict['atom_slices']
    edge_slices = compound_data_dict['edge_slices']
    
    print(f"Loaded {len(compound_ids)} compounds")
    print(f"compound_feature_tensor dtype: {compound_feature_tensor.dtype}")
    print(f"compound_e_features_tensor dtype: {compound_e_features_tensor.dtype}")
    print(f"edge_indices dtype: {edge_indices.dtype}")
    
    # Find the compound
    try:
        compound_idx = compound_ids.index(pubchem_id)
        print(f"Found compound at index: {compound_idx}")
    except ValueError:
        print(f"Compound {pubchem_id} not found in dataset")
        return None
    
    # Get compound data (mimic BADataset.get_graph method)
    start = atom_slices[compound_idx]
    n_atoms = n_atoms_list[compound_idx]
    e_start = edge_slices[compound_idx]
    e_end = edge_slices[compound_idx + 1] if compound_idx + 1 < len(edge_slices) else len(compound_e_features_tensor)
    
    print(f"Compound slice info:")
    print(f"  start: {start}, n_atoms: {n_atoms}")
    print(f"  e_start: {e_start}, e_end: {e_end}")
    
    # Extract data
    x = compound_feature_tensor[start: start + n_atoms].to(device)
    edge_index = edge_indices[:, e_start: e_end].to(device)
    edge_attr = compound_e_features_tensor[e_start: e_end].to(device)
    
    print(f"Dataset tensors:")
    print(f"  x shape: {x.shape}, dtype: {x.dtype}")
    print(f"  edge_index shape: {edge_index.shape}, dtype: {edge_index.dtype}")
    print(f"  edge_attr shape: {edge_attr.shape}, dtype: {edge_attr.dtype}")
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n_atoms)
    
    return data

def test_with_model(data, device):
    """Test the data with the actual model to see where the dtype conversion happens"""
    print(f"\n=== Model Test ===")
    
    # Load config and model
    config = load_cfg("BindingDB.yml")
    
    from modules.interaction_modules.BDB_models import BlendNetS
    
    # Load model (just the compound encoder part)
    model = BlendNetS(config["Path"]["Ki_interaction_site_predictor"], config, device).cuda()
    model.eval()
    
    # Create batch
    batch = Batch.from_data_list([data]).to(device)
    
    print(f"Before model call:")
    print(f"  batch.x dtype: {batch.x.dtype}")
    print(f"  batch.edge_attr dtype: {batch.edge_attr.dtype}")
    
    try:
        with torch.no_grad():
            # Call just the compound encoder
            node_representations, graph_representations = model.compound_encoder(batch)
        print("✓ Model call successful!")
        print(f"Node representations shape: {node_representations.shape}")
        print(f"Graph representations shape: {graph_representations.shape}")
        return True
    except Exception as e:
        print(f"✗ Model call failed: {e}")
        return False

def main():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    # Test compound - same as in the interface
    test_smiles = "Cc1ccc(COc2ccc3nc(C4C(C(=O)O)C4(C)C)n(Cc4ccc(Br)cc4)c3c2)nc1"
    test_pubchem_id = "77050673"  # This might not be exact, but let's try
    
    print(f"Testing with SMILES: {test_smiles}")
    print(f"Device: {device}")
    
    # Method 1: My method
    try:
        data_my_method = create_compound_graph_my_way(test_smiles, device)
        print("✓ My method successful")
        success_my_method = test_with_model(data_my_method, device)
    except Exception as e:
        print(f"✗ My method failed: {e}")
        success_my_method = False
    
    # Method 2: Dataset method (if compound exists)
    try:
        data_dataset_method = load_compound_from_dataset(test_pubchem_id, device)
        if data_dataset_method is not None:
            print("✓ Dataset method successful")
            success_dataset_method = test_with_model(data_dataset_method, device)
        else:
            print("✗ Dataset method failed: compound not found")
            success_dataset_method = False
    except Exception as e:
        print(f"✗ Dataset method failed: {e}")
        success_dataset_method = False
    
    print(f"\n=== Summary ===")
    print(f"My method success: {success_my_method}")
    print(f"Dataset method success: {success_dataset_method}")

if __name__ == "__main__":
    main()
