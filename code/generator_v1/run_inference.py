#!/usr/bin/env python3
"""
Endpoint script to run ligand generation inference.

This script generates ligands for given protein sequences using the trained diffusion model:
- Implements retrieval-augmented generation according to idea.md
- Uses top-k similar proteins to initialize diffusion
- Generates novel ligands conditioned on protein embeddings

Usage:
    python run_inference.py --protein_sequence "MKTAYIA..." --model_path ./checkpoints/best_model.pth
"""

import argparse
import sys
import os
import logging
import torch
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trainer import ProteinLigandDiffusionTrainer
from embedder import ProteinLigandEmbedder
from inference.ligand_generator import LigandGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Generate ligands for protein sequences using trained diffusion model"
    )
    
    parser.add_argument(
        "--protein_sequence", 
        type=str, 
        required=True,
        help="Protein sequence for ligand generation"
    )
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to trained model checkpoint"
    )
    
    parser.add_argument(
        "--embeddings_dir", 
        type=str, 
        required=True,
        help="Directory containing embeddings and database"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./inference_output",
        help="Directory to save generated ligands"
    )
    
    # Generation parameters
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of ligands to generate")
    parser.add_argument("--k_similar", type=int, default=5,
                       help="Number of similar proteins to retrieve")
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                       help="Guidance scale for conditioning")
    parser.add_argument("--num_inference_steps", type=int, default=100,
                       help="Number of denoising steps")
    
    # Filtering parameters
    parser.add_argument("--filter_invalid", action='store_true',
                       help="Filter out invalid SMILES")
    parser.add_argument("--filter_nonorganic", action='store_true',
                       help="Filter out non-organic molecules")
    parser.add_argument("--predict_ic50", action='store_true',
                       help="Predict IC50 for generated ligands")
    
    # Device
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (cuda/cpu/auto)")
    
    # Logging
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
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
        
    # Validate inputs
    if not os.path.exists(args.model_path):
        logger.error(f"Model checkpoint not found: {args.model_path}")
        sys.exit(1)
        
    if not os.path.exists(args.embeddings_dir):
        logger.error(f"Embeddings directory not found: {args.embeddings_dir}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("="*80)
    logger.info("PROTEIN-LIGAND GENERATION INFERENCE")
    logger.info("="*80)
    logger.info(f"Protein sequence: {args.protein_sequence[:50]}{'...' if len(args.protein_sequence) > 50 else ''}")
    logger.info(f"Model checkpoint: {args.model_path}")
    logger.info(f"Embeddings directory: {args.embeddings_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Number of samples: {args.num_samples}")
    logger.info(f"Retrieval: top-{args.k_similar} similar proteins")
    
    try:
        # Load embeddings and database
        logger.info("Loading embeddings and database...")
        embedder = ProteinLigandEmbedder(data_path="", output_dir=args.embeddings_dir)
        embedding_data = embedder.load_embeddings()
        
        # Load model checkpoint
        logger.info("Loading model checkpoint...")
        checkpoint = torch.load(args.model_path, map_location=device)
        config = checkpoint['config']
        
        # Initialize generator
        logger.info("Initializing ligand generator...")
        generator = LigandGenerator(
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
        
        # Load model weights
        generator.model.load_state_dict(checkpoint['model_state_dict'])
        generator.model.eval()
        
        logger.info("✅ Model loaded successfully")
        
        # Generate ligands
        logger.info(f"Generating {args.num_samples} ligands...")
        
        results = generator.generate_ligands(
            protein_sequence=args.protein_sequence,
            num_samples=args.num_samples,
            k_similar=args.k_similar,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            filter_invalid=args.filter_invalid,
            filter_nonorganic=args.filter_nonorganic,
            predict_ic50=args.predict_ic50
        )
        
        # Save results
        results_file = os.path.join(args.output_dir, "generated_ligands.json")
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        # Print summary
        logger.info("="*80)
        logger.info("LIGAND GENERATION COMPLETED!")
        logger.info("="*80)
        logger.info(f"✅ Generated {len(results['ligands'])} ligands")
        logger.info(f"✅ Results saved to: {results_file}")
        
        if results['ligands']:
            logger.info("\nTop 5 generated ligands:")
            for i, ligand in enumerate(results['ligands'][:5]):
                ic50_str = f" (IC50: {ligand.get('predicted_ic50', 'N/A')})" if 'predicted_ic50' in ligand else ""
                logger.info(f"  {i+1}. {ligand['smiles']}{ic50_str}")
                
        if args.filter_invalid or args.filter_nonorganic:
            filtered_count = results.get('filtered_count', 0)
            logger.info(f"\n📊 Filtering statistics: {filtered_count} ligands filtered out")
            
    except Exception as e:
        logger.error(f"Inference failed with error: {e}")
        import traceback
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
