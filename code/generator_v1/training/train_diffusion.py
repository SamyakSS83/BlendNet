"""
Training script for protein-conditioned ligand diffusion model.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from typing import Dict, List, Tuple
from tqdm import tqdm
import wandb
import argparse

# RDKit imports for SMILES validation
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    print("Warning: RDKit not available, SMILES validation disabled")
    RDKIT_AVAILABLE = False

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../materials.smi-ted/smi-ted/'))

from models.diffusion_model import ProteinLigandDiffusion
from database.vector_database import ProteinLigandVectorDB
from bindingdb_interface import BindingDBInterface
from inference.smi_ted_light.load import load_smi_ted


class ProteinLigandDataset(Dataset):
    """Dataset for protein-ligand pairs."""
    
    def __init__(self, data_path: str):
        """Initialize dataset from preprocessed data."""
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
            
        self.sequences = self.data['sequences']
        self.smiles = self.data['smiles']
        self.ic50_values = np.array(self.data['ic50_values'])
        self.protbert_embeddings = np.array(self.data['protbert_embeddings'])
        self.pseq2sites_embeddings = np.array(self.data['pseq2sites_embeddings'])
        self.compound_embeddings = np.array(self.data['compound_embeddings'])
        
        print(f"Loaded dataset with {len(self.sequences)} samples")
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        # Combine protein embeddings
        protein_emb = np.concatenate([
            self.protbert_embeddings[idx],
            self.pseq2sites_embeddings[idx]
        ])
        
        return {
            'protein_embedding': torch.FloatTensor(protein_emb),
            'compound_embedding': torch.FloatTensor(self.compound_embeddings[idx]),
            'ic50_value': torch.FloatTensor([self.ic50_values[idx]]),
            'sequence': self.sequences[idx],
            'smiles': self.smiles[idx]
        }


def validate_smiles(smiles: str) -> bool:
    """
    Validate if a SMILES string represents a valid, organic molecule.
    
    Args:
        smiles: SMILES string to validate
        
    Returns:
        True if valid and organic, False otherwise
    """
    if not RDKIT_AVAILABLE:
        return True  # Skip validation if RDKit is not available
        
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
            
        # Check if molecule is organic (contains only common organic elements)
        organic_elements = {'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I', 'H'}
        atom_symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
        
        if not atom_symbols.issubset(organic_elements):
            return False
            
        # Additional chemical validity checks
        try:
            # Check if we can calculate basic descriptors
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            
            # Basic sanity checks
            if mw <= 0 or mw > 2000:  # Reasonable molecular weight range
                return False
                
            return True
            
        except Exception:
            return False
            
    except Exception:
        return False

def calculate_smiles_validity_metrics(smiles_list: List[str]) -> Dict[str, float]:
    """
    Calculate SMILES validity metrics for a list of SMILES.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        Dictionary with validity metrics
    """
    if not RDKIT_AVAILABLE or not smiles_list:
        return {'validity': 1.0, 'organic_fraction': 1.0}
        
    valid_count = 0
    organic_count = 0
    
    for smiles in smiles_list:
        if validate_smiles(smiles):
            valid_count += 1
            organic_count += 1
        else:
            # Check if it's at least parseable (even if not organic)
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_count += 1
            except Exception:
                pass
    
    total = len(smiles_list)
    return {
        'validity': valid_count / total if total > 0 else 0.0,
        'organic_fraction': organic_count / total if total > 0 else 0.0
    }

class DiffusionTrainer:
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        config: Dict,
        encoder,
        decoder,
        ic50_predictor=None,
        device='cuda'
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        self.ic50_predictor = ic50_predictor
        self.device = device
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # Diffusion parameters
        self.num_timesteps = config.get('num_timesteps', 1000)
        self.beta_start = config.get('beta_start', 1e-4)
        self.beta_end = config.get('beta_end', 0.02)
        
        # Create beta schedule
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
        # SMILES validation parameters
        self.smiles_validation_weight = config.get('smiles_validation_weight', 0.1)
        self.use_smiles_validation = config.get('use_smiles_validation', True) and RDKIT_AVAILABLE
        
        print(f"SMILES validation: {'enabled' if self.use_smiles_validation else 'disabled'}")
        if self.use_smiles_validation:
            print(f"SMILES validation weight: {self.smiles_validation_weight}")
    
    def add_noise(self, x_0, timesteps):
        """Add noise to clean data according to diffusion schedule"""
        noise = torch.randn_like(x_0)
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[timesteps]).unsqueeze(-1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[timesteps]).unsqueeze(-1)
        
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise, noise
    
    def predict_x0_from_noise(self, x_t, noise_pred, timesteps):
        """Predict x_0 from noisy input and predicted noise"""
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[timesteps]).unsqueeze(-1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[timesteps]).unsqueeze(-1)
        
        return (x_t - sqrt_one_minus_alpha_bar * noise_pred) / sqrt_alpha_bar
    
    def calculate_smiles_validation_loss(self, predicted_embeddings, batch_size=8):
        """
        Calculate a loss term based on SMILES validity.
        
        Args:
            predicted_embeddings: Predicted compound embeddings
            batch_size: Batch size for decoding (to avoid memory issues)
            
        Returns:
            Validation loss (scalar tensor)
        """
        if not self.use_smiles_validation:
            return torch.tensor(0.0, device=self.device)
            
        try:
            # Decode embeddings to SMILES in batches
            all_smiles = []
            embeddings_np = predicted_embeddings.detach().cpu().numpy()
            
            for i in range(0, len(embeddings_np), batch_size):
                batch_embeddings = embeddings_np[i:i+batch_size]
                try:
                    batch_smiles = self.decoder(batch_embeddings)
                    if isinstance(batch_smiles, list):
                        all_smiles.extend(batch_smiles)
                    else:
                        all_smiles.append(batch_smiles)
                except Exception as e:
                    print(f"Warning: Decoder failed for batch {i//batch_size}: {e}")
                    # Add placeholder invalid SMILES for failed decodings
                    all_smiles.extend(['[INVALID]'] * len(batch_embeddings))
            
            # Calculate validity metrics
            validity_metrics = calculate_smiles_validity_metrics(all_smiles)
            
            # Convert to loss (1 - validity)
            validity_loss = 1.0 - validity_metrics['organic_fraction']
            
            return torch.tensor(validity_loss, device=self.device, dtype=torch.float32)
            
        except Exception as e:
            print(f"Warning: SMILES validation failed: {e}")
            return torch.tensor(0.0, device=self.device)
        
    def train_step(self, batch: Dict) -> Tuple[float, float, float]:
        """Single training step."""
        self.model.train()
        
        protein_emb = batch['protein_embedding'].to(self.device)
        compound_emb = batch['compound_embedding'].to(self.device)
        sequences = batch['sequence']
        
        # Forward pass
        noise, predicted_noise, timesteps = self.model(compound_emb, protein_emb)
        
        # Diffusion loss (MSE between predicted and actual noise)
        diffusion_loss = nn.MSELoss()(predicted_noise, noise)
        
        # Initialize regularization losses
        ic50_reg = torch.tensor(0.0, device=self.device)
        smiles_validation_loss = torch.tensor(0.0, device=self.device)
        
        # IC50 regularization (optional)
        if self.config['use_ic50_regularization'] and self.step % self.config['ic50_regularization_freq'] == 0:
            # Reconstruct clean compound embeddings from noisy input and predicted noise
            # First, we need to reconstruct the noisy input x_t
            x_t = self.model.scheduler.q_sample(compound_emb, timesteps, noise)
            # Then predict x_0 from x_t and predicted noise
            predicted_compound_emb = self.model.scheduler.predict_start_from_noise(x_t, timesteps, predicted_noise)
            ic50_reg = self.compute_ic50_regularization(predicted_compound_emb, protein_emb, sequences)
        
        # SMILES validation regularization (optional)
        if (self.use_smiles_validation and 
            self.step % self.config.get('smiles_validation_freq', 50) == 0):
            # Reconstruct clean compound embeddings for SMILES validation
            x_t = self.model.scheduler.q_sample(compound_emb, timesteps, noise)
            predicted_compound_emb = self.model.scheduler.predict_start_from_noise(x_t, timesteps, predicted_noise)
            smiles_validation_loss = self.calculate_smiles_validation_loss(predicted_compound_emb)
            smiles_validation_loss = self.smiles_validation_weight * smiles_validation_loss
            
        # Total loss
        total_loss = self.diffusion_weight * diffusion_loss + ic50_reg + smiles_validation_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
        
        self.optimizer.step()
        
        return diffusion_loss.item(), ic50_reg.item(), smiles_validation_loss.item()
        
    def validate(self) -> float:
        """Validation step."""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                protein_emb = batch['protein_embedding'].to(self.device)
                compound_emb = batch['compound_embedding'].to(self.device)
                
                # Forward pass
                noise, predicted_noise, timesteps = self.model(compound_emb, protein_emb)
                
                # Validation loss (only diffusion loss)
                loss = nn.MSELoss()(predicted_noise, noise)
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches
        
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.config['num_epochs']} epochs...")
        
        for epoch in range(self.config['num_epochs']):
            self.epoch = epoch
            
            # Training
            epoch_diffusion_loss = 0.0
            epoch_ic50_loss = 0.0
            epoch_smiles_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}")
            
            for batch in progress_bar:
                diffusion_loss, ic50_loss, smiles_loss = self.train_step(batch)
                
                epoch_diffusion_loss += diffusion_loss
                epoch_ic50_loss += ic50_loss
                epoch_smiles_loss += smiles_loss
                num_batches += 1
                self.step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'diff_loss': f"{diffusion_loss:.4f}",
                    'ic50_loss': f"{ic50_loss:.4f}",
                    'smiles_loss': f"{smiles_loss:.4f}"
                })
                
                # Log to wandb
                if self.config['use_wandb']:
                    wandb.log({
                        'train/diffusion_loss': diffusion_loss,
                        'train/ic50_loss': ic50_loss,
                        'train/smiles_loss': smiles_loss,
                        'train/total_loss': diffusion_loss + ic50_loss + smiles_loss,
                        'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                        'train/step': self.step
                    })
                    
            # Validation
            val_loss = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log epoch metrics
            avg_diffusion_loss = epoch_diffusion_loss / num_batches
            avg_ic50_loss = epoch_ic50_loss / num_batches
            avg_smiles_loss = epoch_smiles_loss / num_batches
            
            print(f"Epoch {epoch+1}: Train Diffusion Loss: {avg_diffusion_loss:.4f}, "
                  f"IC50 Loss: {avg_ic50_loss:.4f}, SMILES Loss: {avg_smiles_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}")
                  
            if self.config['use_wandb']:
                wandb.log({
                    'epoch/train_diffusion_loss': avg_diffusion_loss,
                    'epoch/train_ic50_loss': avg_ic50_loss,
                    'epoch/train_smiles_loss': avg_smiles_loss,
                    'epoch/val_loss': val_loss,
                    'epoch': epoch
                })
                
            # Save checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(is_best=True)
                
            if (epoch + 1) % self.config['save_freq'] == 0:
                self.save_checkpoint(is_best=False)
                
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        
        if is_best:
            path = os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
        else:
            path = os.path.join(self.config['checkpoint_dir'], f'checkpoint_epoch_{self.epoch}.pth')
            
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default_config.py')
    parser.add_argument('--wandb', action='store_true', help='Use wandb logging')
    args = parser.parse_args()
    
    # Default configuration
    config = {
        # Model parameters
        'compound_dim': 512,
        'protbert_dim': 1024,
        'pseq2sites_dim': 256,
        'hidden_dim': 512,
        'num_layers': 6,
        'dropout': 0.1,
        'num_timesteps': 1000,
        
        # Training parameters
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_epochs': 100,
        'max_grad_norm': 1.0,
        'num_workers': 4,
        
        # Loss weights
        'diffusion_weight': 1.0,
        'ic50_weight': 0.1,
        'use_ic50_regularization': True,
        'ic50_regularization_freq': 10,  # Apply IC50 loss every N steps
        
        # Checkpointing
        'checkpoint_dir': '../checkpoints/',
        'save_freq': 10,
        
        # Logging
        'use_wandb': args.wandb,
        'project_name': 'protein-ligand-diffusion'
    }
    
    # Initialize wandb
    if config['use_wandb']:
        wandb.init(
            project=config['project_name'],
            config=config
        )
        
    # Load datasets
    print("Loading datasets...")
    train_dataset = ProteinLigandDataset('../database/train/preprocessed_data.pkl')
    val_dataset = ProteinLigandDataset('../database/test/preprocessed_data.pkl')
    
    # Initialize trainer
    trainer = DiffusionTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Start training
    trainer.train()
    
    print("Training completed!")


if __name__ == "__main__":
    main()
