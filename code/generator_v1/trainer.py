"""
Trainer module for protein-conditioned ligand diffusion model.

This module handles the training of the diffusion model according to idea.md:
- Uses protein embeddings (ProtBERT + Pseq2Sites) for conditioning
- Uses retrieval-augmented generation with top-k similar proteins
- Implements SMILES validation and IC50 regularization
- Follows the exact loss function from idea.md: Total_Loss = Diffusion_Loss + λ/IC50_predicted

Compliance with idea.md:
- Retrieval-augmented: Get top-k similar proteins, randomly pick 1 of their m ligands
- Conditioning: Es + E1 of input protein  
- Starting point: Mean of top-k retrieved compound embeddings
- Loss function: Diffusion + IC50 guidance
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import random
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import logging

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../materials.smi-ted/smi-ted/'))

# Import required modules
try:
    from models.diffusion_model import ProteinLigandDiffusion
    from inference.smi_ted_light.load import load_smi_ted
    from bindingdb_interface import BindingDBInterface
    # RDKit for SMILES validation
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen
    RDKIT_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    RDKIT_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_smiles(smiles: str) -> bool:
    """Validate if a SMILES string represents a valid, organic molecule."""
    if not RDKIT_AVAILABLE:
        return True
        
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
            
        # Check if molecule is organic (contains only common organic elements)
        organic_elements = {'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I', 'H'}
        atom_symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
        
        if not atom_symbols.issubset(organic_elements):
            return False
            
        # Basic sanity checks
        try:
            mw = Descriptors.MolWt(mol)
            if mw <= 0 or mw > 2000:
                return False
            return True
        except Exception:
            return False
    except Exception:
        return False


class RetrievalAugmentedDataset(Dataset):
    """
    Dataset that implements retrieval-augmented generation according to idea.md.
    
    For each protein:
    1. Retrieve top-k similar proteins using FAISS
    2. Randomly select 1 ligand from the m ligands of each retrieved protein  
    3. Compute mean compound embedding as diffusion starting point
    """
    
    def __init__(self, 
                 protein_database: Dict,
                 protein_sequences: List[str],
                 faiss_index,
                 protein_embeddings: Dict,
                 k_similar: int = 5,
                 alpha: float = 0.5):
        """
        Initialize retrieval-augmented dataset.
        
        Args:
            protein_database: Database of proteins and their ligands
            protein_sequences: List of protein sequences (same order as FAISS index)
            faiss_index: FAISS index for protein similarity search (built with concatenated embeddings)
            protein_embeddings: Dict with 'protbert' and 'pseq2sites' embeddings
            k_similar: Number of similar proteins to retrieve
            alpha: Weight for similarity metric (kept for compatibility, but FAISS uses concatenated embeddings)
        """
        self.protein_database = protein_database
        self.protein_sequences = protein_sequences
        self.faiss_index = faiss_index
        self.protein_embeddings = protein_embeddings
        self.k_similar = k_similar
        self.alpha = alpha
        
        # Create list of all protein-ligand pairs for training
        self.training_pairs = []
        for seq in protein_sequences:
            protein_data = protein_database[seq]
            for ligand in protein_data['ligands']:
                self.training_pairs.append({
                    'protein_sequence': seq,
                    'protbert_embedding': protein_data['protbert_embedding'],
                    'pseq2sites_embedding': protein_data['pseq2sites_embedding'],
                    'target_smiles': ligand['smiles'],
                    'target_compound_embedding': ligand['compound_embedding'],
                    'target_ic50': ligand['ic50']
                })
                
        logger.info(f"Dataset created with {len(self.training_pairs)} training pairs")
        
    def __len__(self):
        return len(self.training_pairs)
        
    def retrieve_similar_compounds(self, query_protein_idx: int) -> np.ndarray:
        """
        Retrieve compound embeddings from top-k similar proteins.
        
        According to idea.md:
        1. Get top-k similar proteins using concatenated embeddings [ProtBERT || Pseq2Sites]
        2. For each protein, randomly select 1 of its m ligands
        3. Return compound embeddings for mean computation
        
        Args:
            query_protein_idx: Index of query protein in the dataset
            
        Returns:
            Array of retrieved compound embeddings
        """
        # Get query protein embeddings
        query_seq = self.protein_sequences[query_protein_idx]
        query_protbert = self.protein_embeddings['protbert'][query_protein_idx:query_protein_idx+1]
        query_pseq2sites = self.protein_embeddings['pseq2sites'][query_protein_idx:query_protein_idx+1]
        
        # Concatenate embeddings (to match FAISS index structure)
        # The FAISS index was built with concatenated embeddings, not weighted combination
        query_combined = np.concatenate([query_protbert, query_pseq2sites], axis=1)
        query_combined = query_combined.astype(np.float32)
        
        # Normalize for cosine similarity
        import faiss
        faiss.normalize_L2(query_combined)
        
        # Search for top-k similar proteins
        similarities, indices = self.faiss_index.search(query_combined, self.k_similar + 1)  # +1 to exclude self
        
        # Remove self if present
        indices = indices[0]
        if query_protein_idx in indices:
            indices = indices[indices != query_protein_idx][:self.k_similar]
        else:
            indices = indices[:self.k_similar]
            
        # Collect compound embeddings from retrieved proteins
        retrieved_embeddings = []
        for protein_idx in indices:
            similar_seq = self.protein_sequences[protein_idx]
            similar_protein = self.protein_database[similar_seq]
            
            # Randomly select 1 ligand from the m ligands of this protein
            if similar_protein['ligands']:
                selected_ligand = random.choice(similar_protein['ligands'])
                retrieved_embeddings.append(selected_ligand['compound_embedding'])
                
        if not retrieved_embeddings:
            # Fallback: use random compound embedding
            logger.warning("No retrieved embeddings, using random fallback")
            retrieved_embeddings = [np.random.randn(768)]  # smi-TED dimension
            
        return np.array(retrieved_embeddings)
        
    def __getitem__(self, idx):
        pair = self.training_pairs[idx]
        
        # Find protein index in the sequence list
        protein_idx = self.protein_sequences.index(pair['protein_sequence'])
        
        # Retrieve similar compound embeddings
        retrieved_embeddings = self.retrieve_similar_compounds(protein_idx)
        
        # Compute mean as starting point for diffusion
        starting_point = np.mean(retrieved_embeddings, axis=0)
        
        # Combine protein embeddings for conditioning
        protein_conditioning = np.concatenate([
            pair['protbert_embedding'],
            pair['pseq2sites_embedding']
        ])
        
        return {
            'protein_embedding': torch.FloatTensor(protein_conditioning),
            'target_compound_embedding': torch.FloatTensor(pair['target_compound_embedding']),
            'starting_point_embedding': torch.FloatTensor(starting_point),
            'target_smiles': pair['target_smiles'],
            'target_ic50': torch.FloatTensor([pair['target_ic50']]),
            'protein_sequence': pair['protein_sequence']
        }


class ProteinLigandDiffusionTrainer:
    """
    Trainer for protein-conditioned ligand diffusion model according to idea.md.
    """
    
    def __init__(self, 
                 config: Dict,
                 protein_database: Dict,
                 protein_sequences: List[str],
                 faiss_index,
                 protein_embeddings: Dict,
                 device: str = "cuda"):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
            protein_database: Database of proteins and their ligands
            protein_sequences: List of protein sequences
            faiss_index: FAISS index for retrieval
            protein_embeddings: Protein embeddings dict
            device: Device for training
        """
        self.config = config
        self.device = device
        
        # Store database components
        self.protein_database = protein_database
        self.protein_sequences = protein_sequences
        self.faiss_index = faiss_index
        self.protein_embeddings = protein_embeddings
        
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
        
        # Initialize IC50 predictor and smi-TED for regularization
        self.ic50_predictor = None
        self.smi_ted = None
        
        if config.get('use_ic50_regularization', False):
            self.initialize_regularization_models()
            
        # SMILES validation settings
        self.use_smiles_validation = config.get('use_smiles_validation', False) and RDKIT_AVAILABLE
        self.smiles_validation_weight = config.get('smiles_validation_weight', 0.1)
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        logger.info(f"Trainer initialized - IC50 regularization: {config.get('use_ic50_regularization', False)}")
        logger.info(f"SMILES validation: {self.use_smiles_validation}")
        
    def initialize_regularization_models(self):
        """Initialize IC50 predictor and smi-TED for regularization."""
        try:
            # Initialize IC50 predictor
            self.ic50_predictor = BindingDBInterface(
                config_path="../BindingDB.yml",
                ic50_weights=self.config.get('ic50_weights_path'),
                device=self.device
            )
            
            # Initialize smi-TED
            smi_ted_paths = [
                '../../materials.smi-ted/smi-ted/inference/smi_ted_light',
                '../materials.smi-ted/smi-ted/inference/smi_ted_light'
            ]
            
            for path in smi_ted_paths:
                abs_path = os.path.abspath(path)
                if os.path.exists(os.path.join(abs_path, 'bert_vocab_curated.txt')):
                    self.smi_ted = load_smi_ted(
                        folder=abs_path,
                        ckpt_filename="smi-ted-Light_40.pt"
                    )
                    break
                    
            if self.smi_ted is None:
                logger.warning("Could not initialize smi-TED for regularization")
                
        except Exception as e:
            logger.warning(f"Failed to initialize regularization models: {e}")
            
    def create_dataloaders(self, train_split: float = 0.8, k_similar: int = 5):
        """Create train and validation dataloaders."""
        # Split protein sequences for train/val
        random.shuffle(self.protein_sequences)
        split_idx = int(len(self.protein_sequences) * train_split)
        
        train_sequences = self.protein_sequences[:split_idx]
        val_sequences = self.protein_sequences[split_idx:]
        
        # Create datasets
        train_dataset = RetrievalAugmentedDataset(
            protein_database=self.protein_database,
            protein_sequences=train_sequences,
            faiss_index=self.faiss_index,
            protein_embeddings=self.protein_embeddings,
            k_similar=k_similar
        )
        
        val_dataset = RetrievalAugmentedDataset(
            protein_database=self.protein_database,
            protein_sequences=val_sequences,
            faiss_index=self.faiss_index,
            protein_embeddings=self.protein_embeddings,
            k_similar=k_similar
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers']
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers']
        )
        
        logger.info(f"Created dataloaders - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
    def compute_ic50_regularization(self, 
                                  generated_embeddings: torch.Tensor,
                                  protein_sequences: List[str]) -> torch.Tensor:
        """
        Compute IC50 regularization term according to idea.md: λ/IC50_predicted
        """
        if self.ic50_predictor is None or self.smi_ted is None:
            return torch.tensor(0.0, device=self.device)
            
        regularization_loss = torch.tensor(0.0, device=self.device)
        valid_predictions = 0
        
        try:
            # Decode embeddings to SMILES
            embeddings_cpu = generated_embeddings.detach().cpu()
            if embeddings_cpu.dtype != torch.float32:
                embeddings_cpu = embeddings_cpu.float()
            if not embeddings_cpu.is_contiguous():
                embeddings_cpu = embeddings_cpu.contiguous()
                
            decoded_smiles = self.smi_ted.decode(embeddings_cpu)
            
            # Compute IC50 regularization for valid SMILES
            for smiles, sequence in zip(decoded_smiles, protein_sequences):
                if not smiles or not validate_smiles(smiles):
                    continue
                    
                try:
                    result = self.ic50_predictor.predict(smiles, sequence)
                    ic50_pred = float(result['IC50'])
                    
                    if ic50_pred > 0:
                        # λ/IC50_predicted - encourages lower IC50 (better binding)
                        reg_term = self.config['ic50_weight'] / (ic50_pred + 1e-6)
                        regularization_loss += reg_term
                        valid_predictions += 1
                        
                except Exception:
                    continue
                    
            if valid_predictions > 0:
                regularization_loss = regularization_loss / valid_predictions
                
        except Exception as e:
            logger.warning(f"IC50 regularization failed: {e}")
            
        return regularization_loss
        
    def compute_smiles_validation_loss(self, generated_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute SMILES validation loss to encourage organic, valid molecules."""
        if not self.use_smiles_validation or self.smi_ted is None:
            return torch.tensor(0.0, device=self.device)
            
        try:
            # Decode embeddings
            embeddings_cpu = generated_embeddings.detach().cpu()
            if embeddings_cpu.dtype != torch.float32:
                embeddings_cpu = embeddings_cpu.float()
            if not embeddings_cpu.is_contiguous():
                embeddings_cpu = embeddings_cpu.contiguous()
                
            decoded_smiles = self.smi_ted.decode(embeddings_cpu)
            
            # Calculate validity
            valid_count = sum(1 for smiles in decoded_smiles if validate_smiles(smiles))
            validity_fraction = valid_count / len(decoded_smiles) if decoded_smiles else 0.0
            
            # Loss is 1 - validity (lower is better)
            validation_loss = 1.0 - validity_fraction
            
            return torch.tensor(validation_loss * self.smiles_validation_weight, 
                              device=self.device, dtype=torch.float32)
                              
        except Exception as e:
            logger.warning(f"SMILES validation failed: {e}")
            return torch.tensor(0.0, device=self.device)
            
    def train_step(self, batch: Dict) -> Tuple[float, float, float]:
        """
        Single training step implementing the loss from idea.md:
        Total_Loss = Diffusion_Loss + λ/IC50_predicted + SMILES_validation
        """
        self.model.train()
        
        # Get batch data
        protein_emb = batch['protein_embedding'].to(self.device)
        target_compound_emb = batch['target_compound_embedding'].to(self.device)
        starting_point_emb = batch['starting_point_embedding'].to(self.device)
        protein_sequences = batch['protein_sequence']
        
        # Forward diffusion: start from retrieved compound mean + noise
        timesteps = torch.randint(0, self.model.num_timesteps, (target_compound_emb.shape[0],), device=self.device)
        noise = torch.randn_like(target_compound_emb)
        
        # Noisy version of target embedding
        noisy_target = self.model.q_sample(target_compound_emb, timesteps, noise)
        
        # Model prediction (predicts noise)
        predicted_noise = self.model(noisy_target, protein_emb, timesteps)
        
        # Diffusion loss (MSE between predicted and actual noise)
        diffusion_loss = nn.MSELoss()(predicted_noise, noise)
        
        # Regularization losses
        ic50_loss = torch.tensor(0.0, device=self.device)
        smiles_loss = torch.tensor(0.0, device=self.device)
        
        # IC50 regularization (according to idea.md: λ/IC50_predicted)
        if (self.config.get('use_ic50_regularization', False) and 
            self.step % self.config.get('ic50_regularization_freq', 10) == 0):
            # Predict clean compound embedding
            predicted_clean = self.model.predict_start_from_noise(noisy_target, timesteps, predicted_noise)
            ic50_loss = self.compute_ic50_regularization(predicted_clean, protein_sequences)
            
        # SMILES validation loss
        if (self.use_smiles_validation and 
            self.step % self.config.get('smiles_validation_freq', 50) == 0):
            predicted_clean = self.model.predict_start_from_noise(noisy_target, timesteps, predicted_noise)
            smiles_loss = self.compute_smiles_validation_loss(predicted_clean)
            
        # Total loss according to idea.md
        total_loss = diffusion_loss + ic50_loss + smiles_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('max_grad_norm', 1.0))
        
        self.optimizer.step()
        
        return diffusion_loss.item(), ic50_loss.item(), smiles_loss.item()
        
    def validate(self) -> float:
        """Validation step."""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                protein_emb = batch['protein_embedding'].to(self.device)
                target_compound_emb = batch['target_compound_embedding'].to(self.device)
                
                # Forward pass
                timesteps = torch.randint(0, self.model.num_timesteps, (target_compound_emb.shape[0],), device=self.device)
                noise = torch.randn_like(target_compound_emb)
                noisy_target = self.model.q_sample(target_compound_emb, timesteps, noise)
                predicted_noise = self.model(noisy_target, protein_emb, timesteps)
                
                # Validation loss (only diffusion loss)
                loss = nn.MSELoss()(predicted_noise, noise)
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches if num_batches > 0 else float('inf')
        
    def train(self):
        """Main training loop."""
        logger.info(f"Starting training for {self.config['num_epochs']} epochs...")
        
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
                
            # Validation
            val_loss = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log epoch metrics
            avg_diffusion_loss = epoch_diffusion_loss / num_batches if num_batches > 0 else 0
            avg_ic50_loss = epoch_ic50_loss / num_batches if num_batches > 0 else 0
            avg_smiles_loss = epoch_smiles_loss / num_batches if num_batches > 0 else 0
            
            logger.info(f"Epoch {epoch+1}: Diffusion Loss: {avg_diffusion_loss:.4f}, "
                       f"IC50 Loss: {avg_ic50_loss:.4f}, SMILES Loss: {avg_smiles_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}")
                       
            # Save checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(is_best=True)
                
            if (epoch + 1) % self.config.get('save_freq', 10) == 0:
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
        logger.info(f"Saved checkpoint: {path}")


def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train protein-conditioned ligand diffusion model")
    parser.add_argument("--embeddings_dir", type=str, required=True,
                       help="Directory containing embeddings from embedder.py")
    parser.add_argument("--config_path", type=str, 
                       help="Path to training config file")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--k_similar", type=int, default=5,
                       help="Number of similar proteins to retrieve")
    parser.add_argument("--use_ic50_regularization", action='store_true')
    parser.add_argument("--use_smiles_validation", action='store_true')
    
    args = parser.parse_args()
    
    # Load embeddings from embedder
    from embedder import ProteinLigandEmbedder
    embedder = ProteinLigandEmbedder(data_path="", output_dir=args.embeddings_dir)
    embedding_data = embedder.load_embeddings()
    
    # Default training configuration
    config = {
        # Model parameters
        'compound_dim': 768,  # smi-TED dimension
        'protbert_dim': 1024,
        'pseq2sites_dim': 256,
        'hidden_dim': 512,
        'num_layers': 8,
        'dropout': 0.1,
        'num_timesteps': 1000,
        
        # Training parameters
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': 1e-5,
        'num_epochs': args.num_epochs,
        'max_grad_norm': 1.0,
        'num_workers': 4,
        
        # Loss weights and regularization
        'use_ic50_regularization': args.use_ic50_regularization,
        'ic50_weight': 0.1,
        'ic50_regularization_freq': 10,
        'use_smiles_validation': args.use_smiles_validation,
        'smiles_validation_weight': 0.1,
        'smiles_validation_freq': 50,
        
        # Checkpointing
        'checkpoint_dir': './checkpoints/',
        'save_freq': 10
    }
    
    # Initialize trainer
    trainer = ProteinLigandDiffusionTrainer(
        config=config,
        protein_database=embedding_data['protein_database'],
        protein_sequences=embedding_data['protein_sequences'],
        faiss_index=embedding_data['faiss_index'],
        protein_embeddings={
            'protbert': embedding_data['protein_protbert_embeddings'],
            'pseq2sites': embedding_data['protein_pseq2sites_embeddings']
        },
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Create dataloaders
    trainer.create_dataloaders(k_similar=args.k_similar)
    
    # Start training
    trainer.train()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
