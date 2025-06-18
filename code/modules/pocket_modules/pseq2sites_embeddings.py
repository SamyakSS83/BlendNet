"""
Pseq2Sites Embedding Model

This module provides an embedding model that generates binding site embeddings
using the pre-trained Pseq2Sites model. It can extract meaningful representations
that signify binding sites in protein sequences.

Author: BlendNet Team
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Optional, Union
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

# Add module paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../common/'))

from .models import Pseq2Sites
from .loaders import PocketTestDataset
from common.utils import load_cfg


class Pseq2SitesEmbeddings:
    """
    Pseq2Sites Embedding Model
    
    A wrapper around the Pseq2Sites model that extracts meaningful embeddings
    signifying binding sites from protein sequences. This model can be used
    for downstream tasks requiring binding site representations.
    """
    
    def __init__(
        self, 
        config_path: str = None,
        checkpoint_path: str = None,
        device: str = "auto"
    ):
        """
        Initialize the Pseq2Sites embedding model.
        
        Args:
            config_path: Path to the configuration YAML file
            checkpoint_path: Path to the model checkpoint (.pth file)
            device: Device to run the model on ('auto', 'cuda', 'cpu')
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = load_cfg(self.config_path)
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Set checkpoint path
        if checkpoint_path is None:
            self.checkpoint_path = self.config.get('Path', {}).get('check_point')
            if self.checkpoint_path is None:
                raise ValueError("No checkpoint path provided and none found in config")
        else:
            self.checkpoint_path = checkpoint_path
            
        # Initialize model
        self.model = None
        self._load_model()
        
        print(f"Pseq2Sites Embedding Model initialized on {self.device}")
        print(f"Model loaded from: {self.checkpoint_path}")
        
    def _get_default_config_path(self) -> str:
        """Get the default configuration path."""
        current_dir = os.path.dirname(__file__)
        return os.path.join(current_dir, '../../pocket_extractor_config.yml')
        
    def _load_model(self):
        """Load the pre-trained Pseq2Sites model."""
        try:
            # Initialize model
            self.model = Pseq2Sites(self.config).to(self.device)
            
            # Load checkpoint
            if os.path.exists(self.checkpoint_path):
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint)
                print(f"✓ Model checkpoint loaded successfully")
            else:
                raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
                
            # Set to evaluation mode
            self.model.eval()
            
            # Disable gradients for inference
            for param in self.model.parameters():
                param.requires_grad = False
                
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def prepare_protein_data(
        self, 
        protein_features: Dict[str, np.ndarray],
        protein_sequences: Dict[str, str],
        protein_ids: List[str] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Prepare protein data for inference.
        
        Args:
            protein_features: Dictionary mapping protein IDs to feature arrays
            protein_sequences: Dictionary mapping protein IDs to sequences
            protein_ids: List of protein IDs to process (if None, process all)
            
        Returns:
            Tuple of (protein_ids, protein_sequences)
        """
        if protein_ids is None:
            protein_ids = list(protein_features.keys())
            
        # Filter and align data
        valid_ids = []
        valid_seqs = []
        
        for pid in protein_ids:
            if pid in protein_features and pid in protein_sequences:
                valid_ids.append(pid)
                valid_seqs.append(protein_sequences[pid])
            else:
                print(f"Warning: Missing data for protein {pid}")
                
        return valid_ids, valid_seqs
    
    def extract_embeddings(
        self,
        protein_features: Dict[str, np.ndarray],
        protein_sequences: Dict[str, str],
        protein_ids: List[str] = None,
        batch_size: int = 32,
        return_predictions: bool = True,
        return_attention: bool = False
    ) -> Dict[str, Dict[str, Union[np.ndarray, torch.Tensor]]]:
        """
        Extract binding site embeddings from protein sequences.
        
        Args:
            protein_features: Dictionary mapping protein IDs to ProtBERT features
            protein_sequences: Dictionary mapping protein IDs to amino acid sequences
            protein_ids: List of specific protein IDs to process
            batch_size: Batch size for inference
            return_predictions: Whether to return binding site predictions
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary mapping protein IDs to their embeddings and predictions
        """
        # Prepare data
        valid_ids, valid_seqs = self.prepare_protein_data(
            protein_features, protein_sequences, protein_ids
        )
        
        if not valid_ids:
            raise ValueError("No valid protein data found")
            
        print(f"Processing {len(valid_ids)} proteins...")
        
        # Create dataset and dataloader
        dataset = PocketTestDataset(
            PID=valid_ids,
            Pseqs=valid_seqs,
            Pfeatures=protein_features,
            maxL=self.config['Architecture']['max_lengths'],
            inputD=self.config['Architecture']['prots_input_dim']
        )
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0  # Avoid multiprocessing issues
        )
        
        # Extract embeddings
        results = {}
        protein_idx = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                prots_data, total_prots_data, prots_mask, position_ids = batch
                
                # Move to device if not already there
                prots_data = prots_data.to(self.device)
                total_prots_data = total_prots_data.to(self.device)
                prots_mask = prots_mask.to(self.device)
                position_ids = position_ids.to(self.device)
                
                # Forward pass
                feats, prot_feats, outputs, att_probs = self.model(
                    prots_data, total_prots_data, prots_mask, position_ids
                )
                
                # Process each protein in the batch
                batch_size_actual = prots_data.shape[0]
                
                for i in range(batch_size_actual):
                    if protein_idx >= len(valid_ids):
                        break
                        
                    pid = valid_ids[protein_idx]
                    # Debug feature vs sequence length
                    pfeat_len = protein_features[pid].shape[0]
                    seq_len_seq = len(valid_seqs[protein_idx])
                    if pfeat_len != seq_len_seq:
                        print(f"⚠ Debug: Protein {pid} sequence length ({seq_len_seq}) != feature length ({pfeat_len}), using min for cropping")
                    # Use the minimum length to avoid broadcasting errors
                    max_len_cfg = self.config['Architecture']['max_lengths']
                    seq_len = min(pfeat_len, seq_len_seq, max_len_cfg)
                    
                    # Extract embeddings for this protein
                    protein_results = {}
                    
                    # Core embeddings (truncated to effective sequence length)
                    protein_results['sequence_embeddings'] = feats[i, :seq_len, :].cpu().numpy()
                    protein_results['protein_embeddings'] = prot_feats[i, :seq_len, :].cpu().numpy()
                    
                    # Binding site predictions
                    if return_predictions:
                        # Truncate predictions to effective sequence length
                        raw_predictions = outputs[i, :seq_len].cpu().numpy()
                        binding_probabilities = torch.sigmoid(outputs[i, :seq_len]).cpu().numpy()
                        
                        protein_results['binding_site_logits'] = raw_predictions
                        protein_results['binding_site_probabilities'] = binding_probabilities
                        protein_results['predicted_binding_sites'] = (binding_probabilities > 0.5).astype(int)
                    
                    # Attention weights
                    if return_attention and att_probs is not None:
                        # Truncate attention weights to effective sequence length
                        protein_results['attention_weights'] = att_probs[i, :, :seq_len, :seq_len].cpu().numpy()
                    
                    # Metadata
                    protein_results['sequence_length'] = seq_len
                    protein_results['sequence'] = valid_seqs[protein_idx][:seq_len]
                    protein_results['attention_mask'] = prots_mask[i, :seq_len].cpu().numpy()
                    
                    results[pid] = protein_results
                    protein_idx += 1
                    
                if protein_idx >= len(valid_ids):
                    break
        
        print(f"✓ Extracted embeddings for {len(results)} proteins")
        return results
    
    def get_binding_site_summary(
        self,
        embeddings_result: Dict[str, Dict],
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Generate a summary of binding site predictions.
        
        Args:
            embeddings_result: Result from extract_embeddings
            threshold: Threshold for binding site prediction
            
        Returns:
            DataFrame with binding site summary statistics
        """
        summary_data = []
        
        for pid, result in embeddings_result.items():
            if 'binding_site_probabilities' not in result:
                continue
                
            probs = result['binding_site_probabilities']
            seq_len = result['sequence_length']
            
            # Calculate statistics
            num_predicted_sites = np.sum(probs > threshold)
            max_probability = np.max(probs)
            mean_probability = np.mean(probs)
            
            # Get top binding sites
            top_indices = np.argsort(probs)[-10:][::-1]  # Top 10 sites
            top_sites = [(idx, probs[idx]) for idx in top_indices if probs[idx] > threshold]
            
            summary_data.append({
                'protein_id': pid,
                'sequence_length': seq_len,
                'num_predicted_binding_sites': num_predicted_sites,
                'binding_site_percentage': (num_predicted_sites / seq_len) * 100,
                'max_binding_probability': max_probability,
                'mean_binding_probability': mean_probability,
                'top_binding_sites': top_sites[:5]  # Top 5 sites
            })
        
        return pd.DataFrame(summary_data)
    
    def save_embeddings(
        self,
        embeddings_result: Dict[str, Dict],
        output_path: str,
        format: str = 'pickle'
    ):
        """
        Save embeddings to file.
        
        Args:
            embeddings_result: Result from extract_embeddings
            output_path: Path to save the embeddings
            format: Format to save in ('pickle', 'npz', 'pt')
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format == 'pickle':
            with open(output_path, 'wb') as f:
                pickle.dump(embeddings_result, f)
        elif format == 'npz':
            # Save as compressed numpy arrays
            np.savez_compressed(output_path, **embeddings_result)
        elif format == 'pt':
            # Save as PyTorch tensors
            torch.save(embeddings_result, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        print(f"✓ Embeddings saved to: {output_path}")
    
    def load_embeddings(
        self,
        input_path: str,
        format: str = 'pickle'
    ) -> Dict[str, Dict]:
        """
        Load embeddings from file.
        
        Args:
            input_path: Path to load embeddings from
            format: Format to load from ('pickle', 'npz', 'pt')
            
        Returns:
            Dictionary of embeddings
        """
        if format == 'pickle':
            with open(input_path, 'rb') as f:
                return pickle.load(f)
        elif format == 'npz':
            return dict(np.load(input_path, allow_pickle=True))
        elif format == 'pt':
            return torch.load(input_path, map_location=self.device)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Convenience function for quick usage
def extract_binding_site_embeddings(
    protein_features: Dict[str, np.ndarray],
    protein_sequences: Dict[str, str],
    config_path: str = None,
    checkpoint_path: str = None,
    device: str = "auto",
    batch_size: int = 32
) -> Dict[str, Dict]:
    """
    Convenience function to extract binding site embeddings.
    
    Args:
        protein_features: Dictionary mapping protein IDs to ProtBERT features
        protein_sequences: Dictionary mapping protein IDs to sequences
        config_path: Path to configuration file
        checkpoint_path: Path to model checkpoint
        device: Device to use
        batch_size: Batch size for inference
        
    Returns:
        Dictionary of embeddings and predictions
    """
    embedder = Pseq2SitesEmbeddings(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device
    )
    
    return embedder.extract_embeddings(
        protein_features=protein_features,
        protein_sequences=protein_sequences,
        batch_size=batch_size,
        return_predictions=True,
        return_attention=True
    )


if __name__ == "__main__":
    # Example usage
    print("Pseq2Sites Embedding Model")
    print("This module provides binding site embeddings from protein sequences.")
    print("\nExample usage:")
    print("""
    from pseq2sites_embeddings import Pseq2SitesEmbeddings
    
    # Initialize the model
    embedder = Pseq2SitesEmbeddings()
    
    # Prepare your data
    protein_features = {...}  # Dict of protein_id -> ProtBERT features
    protein_sequences = {...}  # Dict of protein_id -> amino acid sequence
    
    # Extract embeddings
    results = embedder.extract_embeddings(
        protein_features=protein_features,
        protein_sequences=protein_sequences,
        return_predictions=True,
        return_attention=True
    )
    
    # Get binding site summary
    summary = embedder.get_binding_site_summary(results)
    print(summary)
    
    # Save results
    embedder.save_embeddings(results, "binding_site_embeddings.pkl")
    """)
