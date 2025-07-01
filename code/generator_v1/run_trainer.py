#!/usr/bin/env python3
"""
Endpoint script to run the protein-conditioned ligand diffusion model training.

This script trains the diffusion model according to idea.md architecture:
- Uses retrieval-augmented generation with top-k similar proteins
- Implements conditioning with ProtBERT + Pseq2Sites embeddings
- Uses exact loss function: Total_Loss = Diffusion_Loss + λ/IC50_predicted
- Includes SMILES validation for organic molecule generation

Usage:
    # Basic training (diffusion loss only)
    python run_trainer.py --embeddings_dir ./embedder_output --batch_size 32 --num_epochs 100
    
    # With IC50 regularization (requires weight files)
    python run_trainer.py --embeddings_dir ./embedder_output --use_ic50_regularization \\
                          --ic50_weights_path /path/to/ic50_weights.pth \\
                          --ki_weights_path /path/to/ki_weights.pth
"""

import argparse
import sys
import os
import logging
import torch

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trainer import ProteinLigandDiffusionTrainer
from embedder import ProteinLigandEmbedder


def main():
    parser = argparse.ArgumentParser(
        description="Train protein-conditioned ligand diffusion model according to idea.md"
    )
    
    parser.add_argument(
        "--embeddings_dir", 
        type=str, 
        required=True,
        help="Directory containing embeddings from run_embedder.py"
    )
    
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        default="./checkpoints",
        help="Directory to save model checkpoints"
    )
    
    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of layers")
    parser.add_argument("--num_timesteps", type=int, default=1000, help="Diffusion timesteps")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping")
    
    # Retrieval parameters
    parser.add_argument("--k_similar", type=int, default=5, 
                       help="Number of similar proteins to retrieve")
    parser.add_argument("--train_split", type=float, default=0.8, 
                       help="Train/validation split ratio")
    
    # Regularization parameters
    parser.add_argument("--use_ic50_regularization", action='store_true',
                       help="Enable IC50 regularization (requires --ic50_weights_path and --ki_weights_path)")
    parser.add_argument("--ic50_weight", type=float, default=0.1,
                       help="IC50 regularization weight (λ)")
    parser.add_argument("--ic50_regularization_freq", type=int, default=10,
                       help="Apply IC50 regularization every N steps")
    parser.add_argument("--ic50_weights_path", type=str,
                       help="Path to IC50 predictor weights (required for IC50 regularization)")
    parser.add_argument("--ki_weights_path", type=str,
                       help="Path to Ki predictor weights (required for IC50 regularization)")
    
    # SMILES validation parameters
    parser.add_argument("--use_smiles_validation", action='store_true',
                       help="Enable SMILES validation for organic molecules")
    parser.add_argument("--smiles_validation_weight", type=float, default=0.1,
                       help="SMILES validation loss weight")
    parser.add_argument("--smiles_validation_freq", type=int, default=50,
                       help="Apply SMILES validation every N steps")
    
    # Checkpointing
    parser.add_argument("--save_freq", type=int, default=10,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--resume_from", type=str,
                       help="Path to checkpoint to resume from")
    
    # Logging
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    # Device
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        
    # Validate embeddings directory
    if not os.path.exists(args.embeddings_dir):
        logger.error(f"Embeddings directory not found: {args.embeddings_dir}")
        sys.exit(1)
        
    required_files = [
        "protein_database.pkl",
        "protein_faiss_index.faiss", 
        "protein_embeddings.npz"
    ]
    
    for file in required_files:
        file_path = os.path.join(args.embeddings_dir, file)
        if not os.path.exists(file_path):
            logger.error(f"Required file not found: {file_path}")
            logger.error("Please run run_embedder.py first to create embeddings")
            sys.exit(1)
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Validate IC50 regularization arguments
    if args.use_ic50_regularization:
        if not args.ic50_weights_path or not args.ki_weights_path:
            logger.error("IC50 regularization requires both --ic50_weights_path and --ki_weights_path")
            logger.error("Either provide both weight paths or disable IC50 regularization")
            sys.exit(1)
        
        # Check if weight files exist
        if not os.path.exists(args.ic50_weights_path):
            logger.error(f"IC50 weights file not found: {args.ic50_weights_path}")
            sys.exit(1)
        if not os.path.exists(args.ki_weights_path):
            logger.error(f"Ki weights file not found: {args.ki_weights_path}")
            sys.exit(1)
    
    logger.info("="*80)
    logger.info("PROTEIN-LIGAND DIFFUSION MODEL TRAINING")
    logger.info("="*80)
    logger.info(f"Embeddings directory: {args.embeddings_dir}")
    logger.info(f"Checkpoint directory: {args.checkpoint_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Number of epochs: {args.num_epochs}")
    logger.info(f"Retrieval: top-{args.k_similar} similar proteins")
    logger.info(f"IC50 regularization: {args.use_ic50_regularization}")
    logger.info(f"SMILES validation: {args.use_smiles_validation}")
    
    try:
        # Load embeddings
        logger.info("Loading embeddings and database...")
        embedder = ProteinLigandEmbedder(data_path="", output_dir=args.embeddings_dir)
        embedding_data = embedder.load_embeddings()
        
        logger.info(f"✅ Loaded {embedding_data['metadata']['total_proteins']} proteins")
        logger.info(f"✅ Each protein has up to {embedding_data['metadata']['top_m_ligands']} ligands")
        
        # Create training configuration
        config = {
            # Model parameters
            'compound_dim': embedding_data['metadata']['compound_dim'],
            'protbert_dim': embedding_data['metadata']['protbert_dim'],
            'pseq2sites_dim': embedding_data['metadata']['pseq2sites_dim'],
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'num_timesteps': args.num_timesteps,
            
            # Training parameters
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'num_epochs': args.num_epochs,
            'max_grad_norm': args.max_grad_norm,
            'num_workers': args.num_workers,
            
            # Regularization
            'use_ic50_regularization': args.use_ic50_regularization,
            'ic50_weight': args.ic50_weight,
            'ic50_regularization_freq': args.ic50_regularization_freq,
            'ic50_weights_path': args.ic50_weights_path,
            'ki_weights_path': args.ki_weights_path,
            
            'use_smiles_validation': args.use_smiles_validation,
            'smiles_validation_weight': args.smiles_validation_weight,
            'smiles_validation_freq': args.smiles_validation_freq,
            
            # Checkpointing
            'checkpoint_dir': args.checkpoint_dir,
            'save_freq': args.save_freq
        }
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = ProteinLigandDiffusionTrainer(
            config=config,
            protein_database=embedding_data['protein_database'],
            protein_sequences=embedding_data['protein_sequences'],
            faiss_index=embedding_data['faiss_index'],
            protein_embeddings={
                'protbert': embedding_data['protein_protbert_embeddings'],
                'pseq2sites': embedding_data['protein_pseq2sites_embeddings']
            },
            device=device
        )
        
        # Create dataloaders
        logger.info("Creating dataloaders...")
        trainer.create_dataloaders(train_split=args.train_split, k_similar=args.k_similar)
        
        # Resume from checkpoint if specified
        if args.resume_from:
            logger.info(f"Resuming from checkpoint: {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location=device)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            trainer.step = checkpoint['step']
            trainer.epoch = checkpoint['epoch']
            trainer.best_val_loss = checkpoint['best_val_loss']
            logger.info(f"✅ Resumed from epoch {trainer.epoch}, step {trainer.step}")
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        logger.info("="*80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"✅ Best validation loss: {trainer.best_val_loss:.6f}")
        logger.info(f"✅ Checkpoints saved to: {args.checkpoint_dir}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
