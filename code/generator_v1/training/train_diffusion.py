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


class DiffusionTrainer:
    """Trainer for protein-ligand diffusion model."""
    
    def __init__(self,
                 config: Dict,
                 train_dataset: ProteinLigandDataset,
                 val_dataset: ProteinLigandDataset,
                 device: str = "cuda"):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
            train_dataset: Training dataset
            val_dataset: Validation dataset
            device: Device for training
        """
        self.config = config
        self.device = device
        
        # Initialize model
        self.model = ProteinLigandDiffusion(
            compound_dim=config['compound_dim'],
            protbert_dim=config['protbert_dim'],
            pseq2sites_dim=config['pseq2sites_dim'],
            num_timesteps=config['num_timesteps'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(device)
        
        # Initialize dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers']
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers']
        )
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs'],
            eta_min=config['learning_rate'] * 0.01
        )
        
        # Initialize IC50 predictor for regularization
        if config['use_ic50_regularization']:
            print("Loading IC50 predictor for regularization...")
            self.ic50_predictor = BindingDBInterface(
                config_path="../BindingDB.yml",
                ki_weights="/home/sarvesh/scratch/GS/negroni_data/Blendnet/model_checkpoint/BindingDB/Ki/random_split/CV1/BlendNet_S.pth",
                ic50_weights="/home/sarvesh/scratch/GS/negroni_data/Blendnet/model_checkpoint/BindingDB/IC50/random_split/CV1/BlendNet_S.pth",
                device=device
            )
            
            # Load smi-TED for decoding with robust path resolution
            print("Loading smi-TED...")
            current_dir = os.path.dirname(os.path.abspath(__file__))
            possible_smi_ted_paths = [
                # Direct relative from working directory
                '../../materials.smi-ted/smi-ted/inference/smi_ted_light',
                # Relative to current file
                os.path.join(current_dir, '../../../materials.smi-ted/smi-ted/inference/smi_ted_light'),
                # Alternative paths
                os.path.join(current_dir, '../../materials.smi-ted/smi-ted/inference/smi_ted_light'),
                '../../../materials.smi-ted/smi-ted/inference/smi_ted_light',
                # Absolute paths based on workspace
                '/home/threesamyak/sura/plm_sura/BlendNet/materials.smi-ted/smi-ted/inference/smi_ted_light',
                '/home/sarvesh/scratch/GS/samyak/.Blendnet/materials.smi-ted/smi-ted/inference/smi_ted_light'
            ]
            
            smi_ted_path = None
            for path in possible_smi_ted_paths:
                abs_path = os.path.abspath(path)
                vocab_test = os.path.join(abs_path, 'bert_vocab_curated.txt')
                if os.path.exists(vocab_test):
                    smi_ted_path = abs_path
                    print(f"Found smi-TED at: {smi_ted_path}")
                    break
            
            if smi_ted_path is None:
                print("Searched paths:")
                for path in possible_smi_ted_paths:
                    print(f"  - {os.path.abspath(path)}")
                raise FileNotFoundError("Could not locate smi-TED files for training.")
            
            self.smi_ted = load_smi_ted(
                folder=smi_ted_path,
                ckpt_filename="smi-ted-Light_40.pt"
            )
        else:
            self.ic50_predictor = None
            self.smi_ted = None
            
        # Loss weights
        self.diffusion_weight = config['diffusion_weight']
        self.ic50_weight = config['ic50_weight']
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
    def compute_ic50_regularization(self, 
                                   generated_embeddings: torch.Tensor,
                                   protein_embeddings: torch.Tensor,
                                   sequences: List[str]) -> torch.Tensor:
        """Compute IC50 regularization term."""
        if self.ic50_predictor is None:
            return torch.tensor(0.0, device=self.device)
            
        regularization_loss = torch.tensor(0.0, device=self.device)
        batch_size = generated_embeddings.shape[0]
        
        # Decode embeddings to SMILES
        try:
            with torch.no_grad():
                # Ensure embeddings are numpy arrays for smi-TED decoding
                if isinstance(generated_embeddings, torch.Tensor):
                    embedding_numpy = generated_embeddings.detach().cpu().numpy()
                else:
                    embedding_numpy = generated_embeddings
                    
                decoded_smiles = self.smi_ted.decode(embedding_numpy)
                
            valid_count = 0
            for i, (smiles, sequence) in enumerate(zip(decoded_smiles, sequences)):
                try:
                    # Predict IC50
                    result = self.ic50_predictor.predict(smiles, sequence)
                    ic50_pred = result['IC50']
                    
                    # Add regularization term: lambda / ic50 (encourage lower IC50)
                    if ic50_pred > 0:
                        # Ensure ic50_pred is a float, not tensor
                        if isinstance(ic50_pred, torch.Tensor):
                            ic50_value = float(ic50_pred.cpu().item())
                        else:
                            ic50_value = float(ic50_pred)
                            
                        reg_term = torch.tensor(self.ic50_weight / (ic50_value + 1e-6), 
                                              device=self.device, dtype=torch.float32)
                        regularization_loss += reg_term
                        valid_count += 1
                        
                except Exception as e:
                    # Skip invalid SMILES or prediction errors
                    print(f"IC50 prediction failed for sample {i}: {e}")
                    continue
                    
            if valid_count > 0:
                regularization_loss = regularization_loss / valid_count
                
        except Exception as e:
            print(f"IC50 regularization failed: {e}")
            regularization_loss = torch.tensor(0.0, device=self.device)
            
        return regularization_loss
        
    def train_step(self, batch: Dict) -> Tuple[float, float]:
        """Single training step."""
        self.model.train()
        
        protein_emb = batch['protein_embedding'].to(self.device)
        compound_emb = batch['compound_embedding'].to(self.device)
        sequences = batch['sequence']
        
        # Forward pass
        noise, predicted_noise, timesteps = self.model(compound_emb, protein_emb)
        
        # Diffusion loss (MSE between predicted and actual noise)
        diffusion_loss = nn.MSELoss()(predicted_noise, noise)
        
        # IC50 regularization (optional)
        if self.config['use_ic50_regularization'] and self.step % self.config['ic50_regularization_freq'] == 0:
            ic50_reg = self.compute_ic50_regularization(predicted_noise, protein_emb, sequences)
        else:
            ic50_reg = torch.tensor(0.0, device=self.device)
            
        # Total loss
        total_loss = self.diffusion_weight * diffusion_loss + ic50_reg
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
        
        self.optimizer.step()
        
        return diffusion_loss.item(), ic50_reg.item()
        
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
            num_batches = 0
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}")
            
            for batch in progress_bar:
                diffusion_loss, ic50_loss = self.train_step(batch)
                
                epoch_diffusion_loss += diffusion_loss
                epoch_ic50_loss += ic50_loss
                num_batches += 1
                self.step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'diff_loss': f"{diffusion_loss:.4f}",
                    'ic50_loss': f"{ic50_loss:.4f}"
                })
                
                # Log to wandb
                if self.config['use_wandb']:
                    wandb.log({
                        'train/diffusion_loss': diffusion_loss,
                        'train/ic50_loss': ic50_loss,
                        'train/total_loss': diffusion_loss + ic50_loss,
                        'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                        'step': self.step
                    })
                    
            # Validation
            val_loss = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log epoch metrics
            avg_diffusion_loss = epoch_diffusion_loss / num_batches
            avg_ic50_loss = epoch_ic50_loss / num_batches
            
            print(f"Epoch {epoch+1}: Train Diffusion Loss: {avg_diffusion_loss:.4f}, "
                  f"IC50 Loss: {avg_ic50_loss:.4f}, Val Loss: {val_loss:.4f}")
                  
            if self.config['use_wandb']:
                wandb.log({
                    'epoch/train_diffusion_loss': avg_diffusion_loss,
                    'epoch/train_ic50_loss': avg_ic50_loss,
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
