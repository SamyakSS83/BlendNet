"""
Example script demonstrating the complete protein-ligand diffusion pipeline.
"""
import os
import sys
import torch

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from preprocess.data_preprocessor import DataPreprocessor
from database.vector_database import ProteinLigandVectorDB
from training.train_diffusion import DiffusionTrainer
from inference.ligand_generator import LigandGenerator
from evaluation.metrics import evaluate_ligand_generation


def run_complete_pipeline():
    """Run the complete pipeline from preprocessing to evaluation."""
    
    print("="*80)
    print("PROTEIN-LIGAND DIFFUSION PIPELINE EXAMPLE")
    print("="*80)
    
    # Configuration
    config = {
        'data_paths': {
            'ic50_data': '/home/sarvesh/scratch/GS/negroni_data/Blendnet/input_data/BindingDB/IC50_data.tsv',
            'smi_ted_path': '../../materials.smi-ted/smi-ted/inference/smi_ted_light',
            'smi_ted_ckpt': 'smi-ted-Light_40.pt'
        },
        'preprocessing': {
            'output_dir': './preprocessed_data',
            'max_samples': 10000,  # Limit for example
            'test_split': 0.2
        },
        'training': {
            'output_dir': './trained_models',
            'num_epochs': 10,  # Reduced for example
            'batch_size': 32,
            'learning_rate': 1e-4,
            'lambda_ic50': 0.1
        },
        'inference': {
            'num_samples': 5,
            'top_k_retrieve': 10,
            'num_inference_steps': 20  # Reduced for speed
        },
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Using device: {config['device']}")
    
    # Step 1: Data Preprocessing
    print("\\n" + "="*60)
    print("STEP 1: DATA PREPROCESSING")
    print("="*60)
    
    preprocessor = DataPreprocessor(
        ic50_data_path=config['data_paths']['ic50_data'],
        device=config['device']
    )
    
    # Check if preprocessed data exists
    preprocess_output = os.path.join(config['preprocessing']['output_dir'], 'preprocessed_data.pkl')
    
    if not os.path.exists(preprocess_output):
        print("Preprocessing data...")
        train_data, val_data = preprocessor.preprocess_and_split(
            output_dir=config['preprocessing']['output_dir'],
            max_samples=config['preprocessing']['max_samples'],
            test_split=config['preprocessing']['test_split']
        )
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
    else:
        print("Preprocessed data found, skipping preprocessing.")
    
    # Step 2: Vector Database Creation
    print("\\n" + "="*60)
    print("STEP 2: VECTOR DATABASE CREATION")
    print("="*60)
    
    from database.vector_database import build_database_from_preprocessed_data
    
    db_path = os.path.join(config['preprocessing']['output_dir'], 'vector_database')
    
    if not os.path.exists(db_path):
        print("Building vector database...")
        vector_db = build_database_from_preprocessed_data(
            data_path=preprocess_output,
            output_path=config['preprocessing']['output_dir'],
            db_name='vector_database'
        )
        print("Vector database built successfully!")
    else:
        print("Vector database found, loading...")
        vector_db = ProteinLigandVectorDB()
        vector_db.load_database(db_path)
        print("Vector database loaded successfully!")
    
    # Step 3: Model Training
    print("\\n" + "="*60)
    print("STEP 3: DIFFUSION MODEL TRAINING")
    print("="*60)
    
    model_checkpoint = os.path.join(config['training']['output_dir'], 'diffusion_model.pth')
    
    if not os.path.exists(model_checkpoint):
        print("Training diffusion model...")
        
        trainer = DiffusionTrainer(
            preprocessed_data_path=preprocess_output,
            vector_db=vector_db,
            device=config['device']
        )
        
        trainer.train(
            num_epochs=config['training']['num_epochs'],
            batch_size=config['training']['batch_size'],
            learning_rate=config['training']['learning_rate'],
            lambda_ic50=config['training']['lambda_ic50'],
            output_dir=config['training']['output_dir']
        )
        
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
        device=config['device']
    )
    
    # Example protein sequence (first 200 residues of a real protein)
    example_protein = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQHKLRKLNPPDESGPGCMSCKCVLS"
    
    print(f"Generating ligands for protein sequence (length: {len(example_protein)})...")
    
    generated_ligands = generator.generate_ligands(
        protein_sequence=example_protein,
        num_samples=config['inference']['num_samples'],
        top_k_retrieve=config['inference']['top_k_retrieve'],
        num_inference_steps=config['inference']['num_inference_steps']
    )
    
    print(f"Generated {len(generated_ligands)} ligands!")
    
    # Display top results
    print("\\nTop generated ligands:")
    for i, ligand in enumerate(generated_ligands[:3]):
        print(f"{i+1}. SMILES: {ligand['smiles']}")
        print(f"   Predicted IC50: {ligand['predicted_ic50']:.2f} nM")
        print(f"   Generation Score: {ligand['generation_score']:.4f}")
        print()
    
    # Step 5: Evaluation
    print("\\n" + "="*60)
    print("STEP 5: EVALUATION")
    print("="*60)
    
    # Extract SMILES and IC50 predictions for evaluation
    generated_smiles = [ligand['smiles'] for ligand in generated_ligands]
    predicted_ic50s = [ligand['predicted_ic50'] for ligand in generated_ligands]
    
    # Evaluate
    print("Computing evaluation metrics...")
    metrics = evaluate_ligand_generation(
        generated_smiles=generated_smiles,
        ic50_predictions=predicted_ic50s,
        print_summary=True
    )
    
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
        # Load pre-trained model and database
        generator = LigandGenerator(
            diffusion_checkpoint='./trained_models/diffusion_model.pth',
            vector_db_path='./preprocessed_data/vector_database',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Example protein
        protein_seq = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDL"
        
        print(f"Generating ligands for protein (length: {len(protein_seq)})...")
        
        # Generate ligands
        ligands = generator.generate_ligands(
            protein_sequence=protein_seq,
            num_samples=3,
            top_k_retrieve=5,
            num_inference_steps=20
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
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        run_complete_pipeline()
    else:
        run_quick_example()
