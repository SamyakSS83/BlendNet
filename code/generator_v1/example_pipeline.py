"""
Example script demonstrating the complete protein-ligand diffusion pipeline.
"""
import os
import sys
import torch
import pickle
import numpy as np
import gc
import psutil

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from preprocess.data_preprocessor import DataPreprocessor
from database.vector_database_fixed import ProteinLigandVectorDB, build_database_from_preprocessed_data
from training.train_diffusion import DiffusionTrainer, ProteinLigandDataset
from inference.ligand_generator import LigandGenerator
from evaluation.metrics import evaluate_ligand_generation


def run_complete_pipeline(test_mode=False, num_epochs=50, disable_ic50=False, batch_size=None, num_workers=None, max_samples=None, 
                         learning_rate=1e-4, hidden_dim=None, num_layers=None, num_timesteps=None, use_smiles_validation=True,
                         smiles_validation_weight=0.1, smiles_validation_freq=50):
    """Run the complete pipeline from preprocessing to evaluation."""
    
    print("="*80)
    print("PROTEIN-LIGAND DIFFUSION PIPELINE EXAMPLE")
    if test_mode:
        print("RUNNING IN TEST MODE")
    print("="*80)
    
    # Configuration
    config = {
        'data_paths': {
            'ic50_data': '/home/sarvesh/scratch/GS/negroni_data/Blendnet/input_data/BindingDB/IC50_data.tsv',
            'smi_ted_path': os.path.abspath('../../materials.smi-ted/smi-ted/inference/smi_ted_light'),
            'smi_ted_ckpt': 'smi-ted-Light_40.pt'
        },
        'preprocessing': {
            'output_dir': './preprocessed_data',
            'max_samples': max_samples if max_samples is not None else (10000 if test_mode else None),  # Use provided max_samples or defaults
            'test_split': 0.2
        },
        'training': {
            'output_dir': './trained_models',
            'num_epochs': num_epochs,
            'batch_size': batch_size if batch_size is not None else (4 if test_mode else 8),   # Use provided batch_size or defaults
            'learning_rate': learning_rate,
            'lambda_ic50': 0.0 if disable_ic50 else 0.1  # Disable IC50 if requested
        },
        'inference': {
            'num_samples': 10,   # More samples for evaluation
            'top_k_retrieve': 20,
            'num_inference_steps': 50  # Full inference steps
        },
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Using device: {config['device']}")
    
    # Display configuration
    print(f"Configuration:")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Max samples: {config['preprocessing']['max_samples'] or 'All'}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Number of epochs: {config['training']['num_epochs']}")
    
    # Check system memory
    memory = psutil.virtual_memory()
    print(f"System RAM: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
    if memory.available < 10 * (1024**3):  # Less than 10GB available
        print("⚠️  WARNING: Low system memory. Consider using --test_training mode or adding more RAM.")
        if not test_mode:
            print("Automatically enabling test mode due to low memory...")
            test_mode = True
            if max_samples is None:
                config['preprocessing']['max_samples'] = 10000
    
    # Step 1: Data Preprocessing
    print("\\n" + "="*60)
    print("STEP 1: DATA PREPROCESSING")
    print("="*60)
    
    # Check if we have the fixed preprocessed data
    fixed_data_path = os.path.join(config['preprocessing']['output_dir'], 'preprocessed_data_fixed.pkl')
    
    if os.path.exists(fixed_data_path):
        print("✅ Found fixed preprocessed data, skipping preprocessing.")
        print(f"Using: {fixed_data_path}")
        
        # Verify the fixed data has correct dimensions
        with open(fixed_data_path, 'rb') as f:
            data_check = pickle.load(f)
        
        train_compound_dim = data_check['train_data']['compound_embeddings'].shape[1]
        test_compound_dim = data_check['test_data']['compound_embeddings'].shape[1]
        
        print(f"Data verification:")
        print(f"  Train samples: {len(data_check['train_data']['sequences'])}")
        print(f"  Test samples: {len(data_check['test_data']['sequences'])}")
        print(f"  Train compound dim: {train_compound_dim}")
        print(f"  Test compound dim: {test_compound_dim}")
        
        if train_compound_dim == test_compound_dim == 768:
            print("✅ Data dimensions are correct!")
        else:
            print(f"❌ Data dimensions are incorrect: train={train_compound_dim}, test={test_compound_dim}")
            print("Please run the compound embedding fix script first.")
            return None
            
    else:
        print("No fixed preprocessed data found. Starting full preprocessing...")
        
        preprocessor = DataPreprocessor(
            ic50_data_path=config['data_paths']['ic50_data'],
            device=config['device']
        )
        
        # Check if basic preprocessed data exists
        preprocess_output = os.path.join(config['preprocessing']['output_dir'], 'preprocessed_data.pkl')
        
        if not os.path.exists(preprocess_output):
            print("Preprocessing data from scratch...")
            train_data, val_data = preprocessor.preprocess_and_split(
                output_dir=config['preprocessing']['output_dir'],
                max_samples=config['preprocessing']['max_samples'],
                test_split=config['preprocessing']['test_split']
            )
            print(f"Training samples: {len(train_data)}")
            print(f"Validation samples: {len(val_data)}")
        else:
            print("Basic preprocessed data exists, but need to fix compound embeddings.")
            
        print("❌ You need to run the compound embedding fix script first:")
        print("  python fix_compound_embeddings_v2.py")
        return None
    
    # Step 2: Vector Database Creation
    print("\\n" + "="*60)
    print("STEP 2: VECTOR DATABASE CREATION")
    print("="*60)
    
    # Use the fixed data and vector database
    fixed_data_path = os.path.join(config['preprocessing']['output_dir'], 'preprocessed_data_fixed.pkl')
    db_path = os.path.join(config['preprocessing']['output_dir'], 'vector_database_fixed')
    
    if not os.path.exists(db_path):
        print("Building vector database...")
        
        if not os.path.exists(fixed_data_path):
            print("❌ Fixed preprocessed data not found!")
            print("Please run: python fix_compound_embeddings_v2.py")
            return None
        
        # Build vector database using the fixed data
        print("Building FAISS vector database...")
        vector_db = build_database_from_preprocessed_data(
            data_path=fixed_data_path,
            output_path=config['preprocessing']['output_dir'],
            db_name='vector_database_fixed'
        )
        print("✅ Vector database built successfully!")
    else:
        print("✅ Vector database found, loading...")
        vector_db = ProteinLigandVectorDB()
        vector_db.load_database(db_path)
        print("✅ Vector database loaded successfully!")
        
        # Test the database
        print("Testing vector database...")
        try:
            # Test retrieval with a simple compound query
            test_embedding = np.random.randn(1, 768).astype(np.float32)  # Random test embedding
            indices, scores = vector_db.search_similar_compounds(test_embedding, k=5)
            print(f"✅ Database test successful: retrieved {len(indices)} results")
        except Exception as e:
            print(f"⚠️  Database test warning: {e}")
            print("Database may work but encountered an issue during testing")
    
    # Step 3: Model Training
    print("\\n" + "="*60)
    print("STEP 3: DIFFUSION MODEL TRAINING")
    print("="*60)
    
    # Check for existing trained models
    best_model_path = os.path.join(config['training']['output_dir'], 'best_model.pth')
    checkpoint_path = os.path.join(config['training']['output_dir'], 'checkpoint_epoch_0.pth')
    legacy_path = os.path.join(config['training']['output_dir'], 'diffusion_model.pth')
    
    # Use the best available model
    if os.path.exists(best_model_path):
        model_checkpoint = best_model_path
        print("✅ Found trained model: best_model.pth")
    elif os.path.exists(checkpoint_path):
        model_checkpoint = checkpoint_path
        print("✅ Found trained model: checkpoint_epoch_0.pth")
    elif os.path.exists(legacy_path):
        model_checkpoint = legacy_path
        print("✅ Found trained model: diffusion_model.pth")
    else:
        model_checkpoint = best_model_path  # Default for new training
    
    if not os.path.exists(model_checkpoint):
        print("Training diffusion model...")
        
        # Load datasets for training
        with open(fixed_data_path, 'rb') as f:
            full_data = pickle.load(f)
        
        print(f"✅ Loaded data:")
        print(f"  Train samples: {len(full_data['train_data']['sequences'])}")
        print(f"  Test samples: {len(full_data['test_data']['sequences'])}")
        
        # Verify data dimensions one more time before training
        train_compound_dim = full_data['train_data']['compound_embeddings'].shape[1]
        test_compound_dim = full_data['test_data']['compound_embeddings'].shape[1]
        
        if train_compound_dim != 768 or test_compound_dim != 768:
            print(f"❌ Invalid compound dimensions: train={train_compound_dim}, test={test_compound_dim}")
            print("Expected 768 for both. Please fix the data first.")
            return None
        
        print("✅ All embedding dimensions verified (768)")
        
        # Create training configuration for full dataset
        training_config = {
            # Model parameters
            'compound_dim': 768,  # smi-TED dimension (verified)
            'protbert_dim': 1024,
            'pseq2sites_dim': 256,
            'hidden_dim': hidden_dim if hidden_dim is not None else (256 if test_mode else 512),    # Use provided hidden_dim or defaults
            'num_layers': num_layers if num_layers is not None else (4 if test_mode else 8),        # Use provided num_layers or defaults
            'dropout': 0.1,
            'num_timesteps': num_timesteps if num_timesteps is not None else (100 if test_mode else 1000),  # Use provided num_timesteps or defaults
            
            # Training parameters
            'batch_size': config['training']['batch_size'],
            'learning_rate': config['training']['learning_rate'],
            'weight_decay': 1e-5,
            'num_epochs': config['training']['num_epochs'],
            'max_grad_norm': 1.0,
            'num_workers': num_workers if num_workers is not None else (1 if test_mode else 2),  # Use provided num_workers or defaults
            
            # Loss weights
            'diffusion_weight': 1.0,
            'ic50_weight': config['training']['lambda_ic50'],
            'use_ic50_regularization': not disable_ic50 and config['training']['lambda_ic50'] > 0,
            'ic50_regularization_freq': 20 if test_mode else 10,  # Less frequent for test mode
            
            # SMILES validation parameters
            'use_smiles_validation': use_smiles_validation,
            'smiles_validation_weight': smiles_validation_weight,
            'smiles_validation_freq': smiles_validation_freq,
            
            # Checkpointing
            'checkpoint_dir': config['training']['output_dir'],
            'save_freq': max(1, config['training']['num_epochs'] // 5),  # Save every 20% of epochs
            
            # Logging
            'use_wandb': False,
            'project_name': 'protein-ligand-diffusion-test' if test_mode else 'protein-ligand-diffusion-full'
        }
        
        # Create datasets from the fixed data
        print("Creating training and validation datasets...")
        
        # Memory cleanup before loading large datasets
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        train_data = full_data['train_data']
        val_data = full_data['test_data']
        
        print(f"Memory usage after loading datasets: {psutil.virtual_memory().percent:.1f}%")
        
        # Save temporary dataset files for the trainer
        os.makedirs(config['training']['output_dir'], exist_ok=True)
        train_path = os.path.join(config['training']['output_dir'], 'train_data.pkl')
        val_path = os.path.join(config['training']['output_dir'], 'val_data.pkl')
        
        print("Saving temporary dataset files...")
        with open(train_path, 'wb') as f:
            pickle.dump(train_data, f)
        with open(val_path, 'wb') as f:
            pickle.dump(val_data, f)
            
        # Clear from memory after saving
        del full_data, train_data, val_data
        gc.collect()
        
        # Create dataset objects
        train_dataset = ProteinLigandDataset(train_path)
        val_dataset = ProteinLigandDataset(val_path)
        
        # Initialize the diffusion model
        from models.diffusion_model import ProteinLigandDiffusion
        
        model = ProteinLigandDiffusion(
            compound_dim=training_config['compound_dim'],
            protbert_dim=training_config['protbert_dim'],
            pseq2sites_dim=training_config['pseq2sites_dim'],
            num_timesteps=training_config['num_timesteps'],
            hidden_dim=training_config['hidden_dim'],
            num_layers=training_config['num_layers'],
            dropout=training_config['dropout']
        )
        
        # Load smi-TED for encoding/decoding
        print("Loading smi-TED for compound encoding/decoding...")
        smi_ted_path = None
        current_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            '../materials.smi-ted/smi-ted/inference/smi_ted_light',
            os.path.join(current_dir, '../materials.smi-ted/smi-ted/inference/smi_ted_light'),
            '../../materials.smi-ted/smi-ted/inference/smi_ted_light'
        ]
        
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(os.path.join(abs_path, 'bert_vocab_curated.txt')):
                smi_ted_path = abs_path
                break
                
        if smi_ted_path is None:
            raise FileNotFoundError("Could not locate smi-TED for compound encoding/decoding")
            
        from inference.smi_ted_light.load import load_smi_ted
        smi_ted = load_smi_ted(
            folder=smi_ted_path,
            ckpt_filename="smi-ted-Light_40.pt"
        )
        
        # Initialize trainer
        trainer = DiffusionTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=training_config,
            encoder=smi_ted.encode,  # Use smi-TED encoder
            decoder=smi_ted.decode,  # Use smi-TED decoder
            ic50_predictor=None,     # Will be loaded inside if needed
            device=config['device']
        )
        
        # Start training
        trainer.train()
        
        print("Training completed!")
    else:
        print("Trained model found, skipping training.")
    
    # Step 4: Inference and Generation
    print("\\n" + "="*60)
    print("STEP 4: LIGAND GENERATION")
    print("="*60)
    
    print("Initializing ligand generator...")
    generator = LigandGenerator(
        diffusion_checkpoint=model_checkpoint,
        vector_db_path=db_path,
        smi_ted_path=config['data_paths']['smi_ted_path'],
        smi_ted_ckpt=config['data_paths']['smi_ted_ckpt'],
        device=config['device']
    )
    
    # Example protein sequence (first 200 residues of a real protein)
    example_protein = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQHKLRKLNPPDESGPGCMSCKCVLS"
    
    print(f"Generating ligands for protein sequence (length: {len(example_protein)})...")
    
    generated_ligands, comparison_data = generator.generate_ligands(
        protein_sequence=example_protein,
        num_samples=config['inference']['num_samples'],
        k_retrieve=config['inference']['top_k_retrieve']
    )
    
    print(f"Generated {len(generated_ligands)} ligands!")
    
    # Display top results
    print("\\nTop generated ligands:")
    for i, ligand in enumerate(generated_ligands[:3]):
        print(f"{i+1}. SMILES: {ligand['smiles']}")
        if ligand.get('predicted_ic50') is not None:
            print(f"   Predicted IC50: {ligand['predicted_ic50']:.2f} nM")
        else:
            print("   Predicted IC50: N/A")
        print(f"   Molecular Weight: {ligand.get('molecular_weight', 'N/A'):.2f}" if ligand.get('molecular_weight') else "   Molecular Weight: N/A")
        print(f"   LogP: {ligand.get('logp', 'N/A'):.2f}" if ligand.get('logp') else "   LogP: N/A")
        print()
    
    # Step 5: Evaluation
    print("\\n" + "="*60)
    print("STEP 5: EVALUATION")
    print("="*60)
    
    # Extract SMILES and IC50 predictions for evaluation
    generated_smiles = [ligand['smiles'] for ligand in generated_ligands]
    predicted_ic50s = [ligand.get('predicted_ic50') for ligand in generated_ligands if ligand.get('predicted_ic50') is not None]
    
    print(f"Generated {len(generated_smiles)} SMILES")
    print(f"Got IC50 predictions for {len(predicted_ic50s)} compounds")
    
    if predicted_ic50s:
        # Evaluate
        print("Computing evaluation metrics...")
        metrics = evaluate_ligand_generation(
            generated_smiles=generated_smiles,
            ic50_predictions=predicted_ic50s,
            print_summary=True
        )
    else:
        print("⚠️  No valid IC50 predictions available, skipping detailed evaluation")
        metrics = {
            'num_generated': len(generated_smiles),
            'num_valid_ic50': 0,
            'note': 'IC50 prediction failed for all compounds'
        }
    
    # Step 6: Save Results
    print("\\n" + "="*60)
    print("STEP 6: SAVING RESULTS")
    print("="*60)
    
    results_dir = './example_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save generated ligands
    import json
    results = {
        'input_protein_sequence': example_protein,
        'generated_ligands': generated_ligands,
        'comparison_data': comparison_data,
        'evaluation_metrics': metrics,
        'config': config
    }
    
    with open(os.path.join(results_dir, 'generation_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to {results_dir}/generation_results.json")
    
    print("\\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return results


def run_quick_example():
    """Run a quick example with minimal setup."""
    
    print("="*60)
    print("QUICK PROTEIN-LIGAND GENERATION EXAMPLE")
    print("="*60)
    
    # This assumes you have already run the full pipeline once
    # and have the necessary files
    
    try:
        # Find the best available model
        best_model_path = './trained_models/best_model.pth'
        checkpoint_path = './trained_models/checkpoint_epoch_0.pth'
        legacy_path = './trained_models/diffusion_model.pth'
        
        if os.path.exists(best_model_path):
            model_path = best_model_path
        elif os.path.exists(checkpoint_path):
            model_path = checkpoint_path
        elif os.path.exists(legacy_path):
            model_path = legacy_path
        else:
            raise FileNotFoundError("No trained model found. Please run the full pipeline first.")
        
        # Load pre-trained model and database
        generator = LigandGenerator(
            diffusion_checkpoint=model_path,
            vector_db_path='./preprocessed_data/vector_database_fixed',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Example protein
        protein_seq = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDL"
        
        print(f"Generating ligands for protein (length: {len(protein_seq)})...")
        
        # Generate ligands
        ligands = generator.generate_ligands(
            protein_sequence=protein_seq,
            num_samples=3,
            k_retrieve=5
        )
        
        print("Generated ligands:")
        for i, ligand in enumerate(ligands):
            print(f"{i+1}. {ligand['smiles']} (IC50: {ligand['predicted_ic50']:.2f} nM)")
        
        return ligands
        
    except Exception as e:
        print(f"Quick example failed: {e}")
        print("Please run the full pipeline first with run_complete_pipeline()")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Protein-ligand diffusion pipeline example")
    parser.add_argument("--mode", choices=['full', 'quick'], default='full',
                       help="Run full pipeline or quick example")
    parser.add_argument("--test_training", action='store_true',
                       help="Run in test mode with reduced settings")
    parser.add_argument("--num_epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Training batch size (default: 4 for test mode, 8 for full mode)")
    parser.add_argument("--num_workers", type=int, default=None,
                       help="Number of data loading workers (default: 1 for test mode, 2 for full mode)")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process (default: 10000 for test mode, all for full mode)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate for training")
    parser.add_argument("--hidden_dim", type=int, default=None,
                       help="Hidden dimension of the model (default: 256 for test mode, 512 for full mode)")
    parser.add_argument("--num_layers", type=int, default=None,
                       help="Number of layers in the model (default: 4 for test mode, 8 for full mode)")
    parser.add_argument("--num_timesteps", type=int, default=None,
                       help="Number of diffusion timesteps (default: 100 for test mode, 1000 for full mode)")
    parser.add_argument("--disable_ic50", action='store_true',
                       help="Disable IC50 regularization for testing")
    parser.add_argument("--use_smiles_validation", action='store_true', default=True,
                       help="Use SMILES validation during training")
    parser.add_argument("--smiles_validation_weight", type=float, default=0.1,
                       help="Weight for SMILES validation loss")
    parser.add_argument("--smiles_validation_freq", type=int, default=50,
                       help="Frequency of SMILES validation (every N steps)")
    parser.add_argument("--low_memory", action='store_true',
                       help="Enable low memory mode (reduces batch size and dataset size)")
    
    args = parser.parse_args()
    
    # Override test_mode if low_memory is specified
    if args.low_memory:
        args.test_training = True
    
    # Apply low memory defaults if needed
    if args.low_memory:
        if args.batch_size is None:
            args.batch_size = 2
        if args.num_workers is None:
            args.num_workers = 1
        if args.max_samples is None:
            args.max_samples = 5000
    
    if args.mode == 'full':
        run_complete_pipeline(
            test_mode=args.test_training, 
            num_epochs=args.num_epochs,
            disable_ic50=args.disable_ic50,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_timesteps=args.num_timesteps,
            use_smiles_validation=args.use_smiles_validation,
            smiles_validation_weight=args.smiles_validation_weight,
            smiles_validation_freq=args.smiles_validation_freq
        )
    else:
        run_quick_example()
