#!/usr/bin/env python3
"""
Quick test of training with fixed compound embeddings.
"""
import os
import sys
import pickle
import torch

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from training.train_diffusion import DiffusionTrainer, ProteinLigandDataset

def test_training_with_fixed_data():
    """Test training with the fixed compound embeddings."""
    print("="*80)
    print("TESTING TRAINING WITH FIXED COMPOUND EMBEDDINGS")
    print("="*80)
    
    # Load the fixed data
    fixed_data_path = './preprocessed_data/preprocessed_data_fixed.pkl'
    
    if not os.path.exists(fixed_data_path):
        print(f"❌ Fixed data not found: {fixed_data_path}")
        print("Please run fix_compound_embeddings_v2.py first")
        return
    
    print("Loading fixed data...")
    with open(fixed_data_path, 'rb') as f:
        data = pickle.load(f)
    
    train_data = data['train_data']
    test_data = data['test_data']
    
    print("Verifying data shapes:")
    print(f"Train compound embeddings: {train_data['compound_embeddings'].shape}")
    print(f"Test compound embeddings: {test_data['compound_embeddings'].shape}")
    
    if train_data['compound_embeddings'].shape[1] != test_data['compound_embeddings'].shape[1]:
        print("❌ Embeddings still have shape mismatch!")
        return
    
    # Create small test datasets
    print("Creating small test datasets...")
    
    # Use first 10 samples for train, first 5 for val
    small_train_data = {}
    small_val_data = {}
    
    for key in train_data.keys():
        if key == 'metadata':
            small_train_data[key] = train_data[key].copy()
            small_val_data[key] = test_data[key].copy()
        elif isinstance(train_data[key], list):
            small_train_data[key] = train_data[key][:10]
            small_val_data[key] = test_data[key][:5]
        else:  # numpy arrays
            small_train_data[key] = train_data[key][:10]
            small_val_data[key] = test_data[key][:5]
    
    # Update metadata
    small_train_data['metadata']['n_samples'] = 10
    small_val_data['metadata']['n_samples'] = 5
    
    print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Test train samples: {len(small_train_data['sequences'])}")
    print(f"Test val samples: {len(small_val_data['sequences'])}")
    
    # Save temporary datasets
    train_path = './test_train_data.pkl'
    val_path = './test_val_data.pkl'
    
    with open(train_path, 'wb') as f:
        pickle.dump(small_train_data, f)
    with open(val_path, 'wb') as f:
        pickle.dump(small_val_data, f)
    
    # Create dataset objects
    train_dataset = ProteinLigandDataset(train_path)
    val_dataset = ProteinLigandDataset(val_path)
    
    print(f"Loaded dataset with {len(train_dataset)} samples")
    print(f"Loaded dataset with {len(val_dataset)} samples")
    print(f"✅ Datasets created: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Test configuration
    config = {
        # Model parameters
        'compound_dim': 768,  # Fixed dimension
        'protbert_dim': 1024,
        'pseq2sites_dim': 256,
        'hidden_dim': 128,  # Small for quick test
        'num_layers': 2,    # Small for quick test
        'dropout': 0.1,
        'num_timesteps': 50,  # Small for quick test
        
        # Training parameters
        'batch_size': 2,  # Very small batch
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_epochs': 1,  # Just 1 epoch for test
        'max_grad_norm': 1.0,
        'num_workers': 0,  # No multiprocessing for test
        
        # Loss weights
        'diffusion_weight': 1.0,
        'ic50_weight': 0.1,
        'use_ic50_regularization': True,
        'ic50_regularization_freq': 1,
        
        # Checkpointing
        'checkpoint_dir': './test_checkpoints',
        'save_freq': 1,
        
        # Logging
        'use_wandb': False,
        'project_name': 'test'
    }
    
    print("Initializing trainer...")
    
    # Initialize trainer
    trainer = DiffusionTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("Starting mini training test...")
    
    try:
        # Run training for 1 epoch
        trainer.train()
        print("✅ Training test completed successfully!")
        
        # Clean up
        os.remove(train_path)
        os.remove(val_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up
        if os.path.exists(train_path):
            os.remove(train_path)
        if os.path.exists(val_path):
            os.remove(val_path)
        
        return False

if __name__ == "__main__":
    test_training_with_fixed_data()
