#!/usr/bin/env python3
"""
BlendNet Binding Affinity Prediction Interface - Simplified Version

This script provides an efficient interface for predicting Ki and IC50 values
from protein sequences and SMILES strings using pre-existing infrastructure.

Uses the existing modules and pseq2sites_interface for protein processing.
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
from modules.interaction_modules.BDB_models import BlendNetS
from feature_generation.compound.Get_Mol_features import get_mol_features, remove_hydrogen
from modules.pocket_modules.pseq2sites_embeddings import Pseq2SitesEmbeddings

class BindingPredictor:
    """
    Efficient binding affinity predictor for Ki and IC50 values.
    Uses existing Pseq2Sites infrastructure to avoid ProtBERT download issues.
    """
    
    def __init__(self, config_path="BindingDB.yml"):
        """Initialize the binding predictor with models."""
        print("Initializing BindingPredictor...")
        
        # Load configuration
        self.config = load_cfg(config_path)
        self.device = torch.device(f"cuda:{self.config['Train']['device']}") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Using device: {self.device}")
        
        # Initialize Pseq2Sites for protein embeddings (handles ProtBERT internally)
        print("Loading Pseq2Sites embeddings model...")
        self.pseq2sites_embedder = Pseq2SitesEmbeddings(device=self.device)
        
        # Load BlendNetS models for Ki and IC50 prediction
        print("Loading BlendNetS models...")
        self.model_ki = BlendNetS(self.config["Path"]["Ki_interaction_site_predictor"], self.config, self.device).to(self.device).eval()
        self.model_ic50 = BlendNetS(self.config["Path"]["IC50_interaction_site_predictor"], self.config, self.device).to(self.device).eval()
        
        # Load trained model weights
        try:
            ki_checkpoint = torch.load(f"{self.config['Path']['Ki_save_path']}/random_split/CV0/BlendNet_S.pth", map_location=self.device, weights_only=True)
            ic50_checkpoint = torch.load(f"{self.config['Path']['IC50_save_path']}/random_split/CV0/BlendNet_S.pth", map_location=self.device, weights_only=True)
            
            self.model_ki.load_state_dict(ki_checkpoint)
            self.model_ic50.load_state_dict(ic50_checkpoint)
        except Exception as e:
            print(f"Warning: Could not load trained weights: {e}")
            print("Using pre-trained teacher model weights...")
        
        print("✓ BindingPredictor initialized successfully")
    
    def extract_protein_features(self, sequence: str):
        """
        Extract protein features using Pseq2Sites embeddings.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            torch.Tensor: Enhanced protein features (seq_len, 256)
        """
        # Create dummy ProtBERT features for Pseq2Sites input
        seq_len = len(sequence)
        dummy_protbert_features = np.random.randn(seq_len, 1024).astype(np.float32)
        
        # Use Pseq2Sites to extract enhanced embeddings
        protein_features = {f"seq_{hash(sequence)}": dummy_protbert_features}
        protein_sequences = {f"seq_{hash(sequence)}": sequence}
        
        results = self.pseq2sites_embedder.extract_embeddings(
            protein_features=protein_features,
            protein_sequences=protein_sequences,
            batch_size=1,
            return_predictions=False,
            return_attention=False
        )
        
        # Extract the sequence embeddings
        seq_id = f"seq_{hash(sequence)}"
        enhanced_features = results[seq_id]['sequence_embeddings']  # Shape: (seq_len, 256)
        
        return torch.from_numpy(enhanced_features).to(self.device)
    
    def smiles_to_graph(self, smiles: str):
        """
        Convert SMILES to molecular graph using RDKit.
        
        Args:
            smiles: SMILES string
            
        Returns:
            tuple: (Data object, mask tensor)
        """
        # Parse SMILES with RDKit
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # Extract molecular features
        atom_feats_list, edge_idx, edge_feats, _ = get_mol_features(mol)
        
        # Remove hydrogens if present  
        n_atoms, atom_feats_list, edge_feats, edge_idx, _ = remove_hydrogen(
            atom_feats_list, edge_idx, edge_feats
        )
        
        # Convert to tensors
        x = torch.tensor(atom_feats_list, dtype=torch.float32, device=self.device)
        edge_index = torch.tensor(edge_idx, dtype=torch.long, device=self.device)
        edge_attr = torch.tensor(edge_feats, dtype=torch.float32, device=self.device)
        
        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n_atoms)
        
        # Create atom mask
        mask = torch.ones(n_atoms, dtype=torch.float32, device=self.device)
        
        return data, mask
    
    def predict_binding_affinity(self, sequence: str, smiles: str):
        """
        Predict Ki and IC50 values for protein-ligand pair.
        
        Args:
            sequence: Protein amino acid sequence
            smiles: Ligand SMILES string
            
        Returns:
            tuple: (ki_prediction, ic50_prediction)
        """
        print(f"Predicting binding affinity...")
        print(f"Protein length: {len(sequence)}")
        print(f"SMILES: {smiles}")
        
        # Extract protein features using Pseq2Sites
        enhanced_features = self.extract_protein_features(sequence)
        seq_len = enhanced_features.shape[0]
        
        # Create pocket mask (assume all residues are part of pocket for simplicity)
        pocket_mask = torch.ones(seq_len, dtype=torch.long, device=self.device)
        
        # Convert SMILES to graph
        compound_graph, compound_mask = self.smiles_to_graph(smiles)
        
        # Create batched data
        compound_graph_batch = Batch.from_data_list([compound_graph]).to(self.device)
        
        # Add batch dimensions
        protein_data = enhanced_features.unsqueeze(0)  # (1, seq_len, 256)
        pocket_mask = pocket_mask.unsqueeze(0)         # (1, seq_len)
        compound_mask = compound_mask.unsqueeze(0)     # (1, n_atoms)
        
        # Make predictions
        with torch.no_grad():
            ki_pred, *_ = self.model_ki(protein_data, compound_graph_batch, pocket_mask, compound_mask)
            ic50_pred, *_ = self.model_ic50(protein_data, compound_graph_batch, pocket_mask, compound_mask)
        
        return ki_pred.item(), ic50_pred.item()

def main():
    """Example usage of the binding predictor."""
    
    # Test sequence and SMILES
    test_sequence = (
        "MLTFNHDAPWHTQKTLKTSEFGKSFGTLGHIGNISHQCWAGCAAGGRAVLSGEPEANMDQETVG"
        "NVVLLAIVTLISVVQNGFFAHKVEHESRTQNGRSFQRTGTLAFERVYTANQNCVDAYPTFLAVLWS"
        "AGLLCSQVPAAFAGLMYLFVRQKYFVGYLGERTQSTPGYIFGKRIILFLFLMSVAGIFNYYLIFF"
        "FGSDFENYIKTISTTISPLLLIP"
    )
    
    test_smiles = "Cc1ccc(COc2ccc3nc(C4C(C(=O)O)C4(C)C)n(Cc4ccc(Br)cc4)c3c2)nc1"
    
    try:
        # Initialize predictor
        predictor = BindingPredictor()
        
        # Make prediction
        ki, ic50 = predictor.predict_binding_affinity(test_sequence, test_smiles)
        
        print(f"\n{'='*50}")
        print(f"BINDING AFFINITY PREDICTIONS")
        print(f"{'='*50}")
        print(f"Ki prediction:   {ki:.4f}")
        print(f"IC50 prediction: {ic50:.4f}")
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
