#!/usr/bin/env python3
"""
BlendNet Binding Affinity Prediction Interface

This script provides an efficient interface for predicting Ki and IC50 values
from protein sequences and SMILES strings without loading unnecessary data.

Based on analysis of BlendNet codebase architecture:
- Uses ProtBERT for protein embeddings  
- Uses Pseq2Sites for enhanced pocket representations
- Uses RDKit + molecular graphs for compound features
- Employs BlendNetS models for binding affinity prediction
"""

import os
import sys
import torch
import pickle
import numpy as np
from rdkit import Chem
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

# Add module paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../modules/'))

from modules.common.utils import load_cfg
from modules.interaction_modules.BDB_models import BlendNetS
from modules.pocket_modules.models import Pseq2Sites
from feature_generation.compound.Get_Mol_features import get_mol_features, remove_hydrogen

class BindingPredictor:
    """
    Efficient binding affinity predictor for Ki and IC50 values.
    """
    
    def __init__(self, config_path="BindingDB.yml"):
        """Initialize the binding predictor with models and tokenizers."""
        print("Initializing BindingPredictor...")
        
        # Load configuration
        self.config = load_cfg(config_path)
        self.device = torch.device(f"cuda:{self.config['Train']['device']}") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Using device: {self.device}")
        
        # Initialize ProtBERT for protein feature extraction
        print("Loading ProtBERT...")
        self.protein_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        self.protein_model = BertModel.from_pretrained("Rostlab/prot_bert").to(self.device).eval()
        
        # Initialize Pseq2Sites for enhanced protein embeddings
        print("Loading Pseq2Sites model...")
        self.pseq2sites = Pseq2Sites(self.config).to(self.device)
        pseq2sites_checkpoint = torch.load(self.config["Path"]["Ki_interaction_site_predictor"], map_location=self.device, weights_only=True)
        self.pseq2sites.load_state_dict(pseq2sites_checkpoint)
        self.pseq2sites.eval()
        
        # Load BlendNetS models for Ki and IC50 prediction
        print("Loading BlendNetS models...")
        self.model_ki = BlendNetS(self.config["Path"]["Ki_interaction_site_predictor"], self.config, self.device).to(self.device).eval()
        self.model_ic50 = BlendNetS(self.config["Path"]["IC50_interaction_site_predictor"], self.config, self.device).to(self.device).eval()
        
        # Load trained model weights
        ki_checkpoint = torch.load(f"{self.config['Path']['Ki_save_path']}/random_split/CV0/BlendNet_S.pth", map_location=self.device, weights_only=True)
        ic50_checkpoint = torch.load(f"{self.config['Path']['IC50_save_path']}/random_split/CV0/BlendNet_S.pth", map_location=self.device, weights_only=True)
        
        self.model_ki.load_state_dict(ki_checkpoint)
        self.model_ic50.load_state_dict(ic50_checkpoint)
        
        print("✓ BindingPredictor initialized successfully")
    
    def extract_protein_features(self, sequence: str):
        """
        Extract ProtBERT features from protein sequence.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            torch.Tensor: ProtBERT features (seq_len, 1024)
        """
        # Clean sequence and add spaces between amino acids for ProtBERT
        clean_seq = "".join([aa if aa in "ACDEFGHIKLMNPQRSTVWY" else "X" for aa in sequence.upper()])
        spaced_seq = " ".join(list(clean_seq))
        
        # Tokenize and encode
        inputs = self.protein_tokenizer.batch_encode_plus(
            [spaced_seq], 
            add_special_tokens=True, 
            padding=True,
            truncation=True,
            max_length=1024,  # ProtBERT max length
            return_tensors="pt"
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.protein_model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state[0]  # Remove batch dimension
            
            # Remove special tokens [CLS] and [SEP]
            seq_len = (attention_mask[0] == 1).sum()
            protein_features = embeddings[1:seq_len-1]  # Remove [CLS] and [SEP]
            
        return protein_features
    
    def extract_enhanced_protein_features(self, protein_features: torch.Tensor, sequence: str):
        """
        Extract enhanced protein features using Pseq2Sites.
        
        Args:
            protein_features: ProtBERT features (seq_len, 1024)
            sequence: Amino acid sequence
            
        Returns:
            tuple: (enhanced_features, pocket_mask)
        """
        seq_len = len(sequence)
        max_len = self.config['Architecture']['max_lengths']
        
        # Prepare inputs for Pseq2Sites
        padded_features = torch.zeros(max_len, 1024, device=self.device)
        padded_features[:seq_len] = protein_features
        
        attention_mask = torch.zeros(max_len, device=self.device, dtype=torch.long)
        attention_mask[:seq_len] = 1
        
        position_ids = torch.arange(max_len, device=self.device)
        
        # Add batch dimension
        padded_features = padded_features.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        position_ids = position_ids.unsqueeze(0)
        
        with torch.no_grad():
            feats, prot_feats, outputs, att_probs = self.pseq2sites(
                padded_features, padded_features, attention_mask, position_ids
            )
        
        # Extract relevant sequence portion
        enhanced_features = feats[0, :seq_len, :]  # (seq_len, 256)
        pocket_mask = torch.ones(seq_len, device=self.device, dtype=torch.long)
        
        return enhanced_features, pocket_mask
    
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
        
        # Extract protein features
        protein_features = self.extract_protein_features(sequence)
        enhanced_features, pocket_mask = self.extract_enhanced_protein_features(protein_features, sequence)
        
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

if __name__ == "__main__":
    main()
