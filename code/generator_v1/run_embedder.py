#!/usr/bin/env python3
"""
Endpoint script to run the protein-ligand embedding pipeline.

This script creates embeddings and vector database according to idea.md architecture:
- Groups data by unique proteins (not molecules)
- Keeps top 3 binding ligands per protein
- Creates FAISS vector database for protein similarity search

Usage:
    python run_embedder.py --data_path /path/to/IC50_data.tsv --output_dir ./embeddings
"""

import argparse
import sys
import os
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from embedder import ProteinLigandEmbedder


def main():
    parser = argparse.ArgumentParser(
        description="Create protein-ligand embeddings and vector database according to idea.md"
    )
    
    parser.add_argument(
        "--data_path", 
        type=str, 
        required=True,
        help="Path to IC50_data.tsv file"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./embedder_output",
        help="Output directory for embeddings and database"
    )
    
    parser.add_argument(
        "--top_m_ligands", 
        type=int, 
        default=3,
        help="Number of top binding ligands to keep per protein (default: 3)"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        help="Device for computations (cuda/cpu)"
    )
    
    parser.add_argument(
        "--log_level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Validate input file
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        sys.exit(1)
        
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("="*80)
    logger.info("PROTEIN-LIGAND EMBEDDING PIPELINE")
    logger.info("="*80)
    logger.info(f"Input data: {args.data_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Top ligands per protein: {args.top_m_ligands}")
    logger.info(f"Device: {args.device}")
    
    try:
        # Initialize embedder
        embedder = ProteinLigandEmbedder(
            data_path=args.data_path,
            output_dir=args.output_dir,
            top_m_ligands=args.top_m_ligands,
            device=args.device
        )
        
        # Run the embedding pipeline
        result = embedder.run_embedding_pipeline()
        
        # Print summary
        logger.info("="*80)
        logger.info("EMBEDDING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"✅ Total proteins processed: {result['metadata']['total_proteins']}")
        logger.info(f"✅ Ligands per protein: up to {result['metadata']['top_m_ligands']}")
        logger.info(f"✅ ProtBERT embedding dimension: {result['metadata']['protbert_dim']}")
        logger.info(f"✅ Pseq2Sites embedding dimension: {result['metadata']['pseq2sites_dim']}")
        logger.info(f"✅ Compound embedding dimension: {result['metadata']['compound_dim']}")
        logger.info(f"✅ Output saved to: {args.output_dir}")
        logger.info("\nFiles created:")
        logger.info(f"  - protein_database.pkl: Protein-ligand database")
        logger.info(f"  - protein_faiss_index.faiss: FAISS similarity index")
        logger.info(f"  - protein_embeddings.npz: Embedding arrays")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        import traceback
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
