#!/usr/bin/env python3
"""
Example usage of Pseq2Sites Embedding Model

This script demonstrates how to use the Pseq2SitesEmbeddings class to extract
binding site embeddings from protein sequences.
"""

import os
import sys
import numpy as np
import pickle
from pathlib import Path

# Add the modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from modules.pocket_modules.pseq2sites_embeddings import Pseq2SitesEmbeddings, extract_binding_site_embeddings


def load_example_data():
    """
    Load example protein data for demonstration.
    This loads actual data from the BlendNet input data if available.
    """
    # Try to load real data from the project
    try:
        # Load protein features (ProtBERT embeddings)
        feature_path = "input_data/PDB/BA/CASF2016_protein_features.pkl"
        if os.path.exists(feature_path):
            with open(feature_path, 'rb') as f:
                protein_features = pickle.load(f)
            print(f"✓ Loaded protein features for {len(protein_features)} proteins")
        else:
            # Create dummy data if real data not available
            protein_features = create_dummy_features()
            
        # Load protein sequences
        import pandas as pd
        seq_path = "input_data/PDB/BA/CASF2016_BA_data.tsv"
        if os.path.exists(seq_path):
            df = pd.read_csv(seq_path, sep='\t')
            protein_sequences = dict(zip(df.iloc[:, 1].values, df.iloc[:, 4].values))
            print(f"✓ Loaded sequences for {len(protein_sequences)} proteins")
        else:
            # Create dummy sequences
            protein_sequences = create_dummy_sequences(list(protein_features.keys()))
            
        return protein_features, protein_sequences
        
    except Exception as e:
        print(f"Could not load real data ({e}), creating dummy data...")
        return create_dummy_data()


def create_dummy_features():
    """Create dummy protein features for testing."""
    protein_ids = ["P12345", "Q67890", "R54321"]
    protein_features = {}
    
    for pid in protein_ids:
        # Create random ProtBERT-like features (1024-dimensional)
        seq_len = np.random.randint(50, 300)  # Random sequence length
        features = np.random.randn(seq_len, 1024).astype(np.float32)
        protein_features[pid] = features
        
    return protein_features


def create_dummy_sequences(protein_ids):
    """Create dummy protein sequences."""
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    protein_sequences = {}
    
    for pid in protein_ids:
        seq_len = np.random.randint(50, 300)
        sequence = ''.join(np.random.choice(list(amino_acids), seq_len))
        protein_sequences[pid] = sequence
        
    return protein_sequences


def create_dummy_data():
    """Create completely dummy data for testing."""
    protein_features = create_dummy_features()
    protein_sequences = create_dummy_sequences(list(protein_features.keys()))
    return protein_features, protein_sequences


def example_basic_usage():
    """Demonstrate basic usage of the embedding model."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Usage")
    print("="*60)
    
    # Load example data
    protein_features, protein_sequences = load_example_data()
    
    # Take a subset for quick demo
    demo_ids = list(protein_features.keys())[:3]
    demo_features = {pid: protein_features[pid] for pid in demo_ids}
    demo_sequences = {pid: protein_sequences[pid] for pid in demo_ids}
    
    print(f"Demo with {len(demo_ids)} proteins: {demo_ids}")
    
    try:
        # Initialize the embedding model
        print("\nInitializing Pseq2Sites embedding model...")
        embedder = Pseq2SitesEmbeddings()
        
        # Extract embeddings
        print("Extracting binding site embeddings...")
        results = embedder.extract_embeddings(
            protein_features=demo_features,
            protein_sequences=demo_sequences,
            batch_size=2,
            return_predictions=True,
            return_attention=True
        )
        
        # Print results summary
        print(f"\n✓ Successfully extracted embeddings for {len(results)} proteins")
        
        for pid, result in results.items():
             print(f"\nProtein {pid}:")
             print(f"  Sequence length: {result['sequence_length']}")
             print(f"  Sequence embeddings shape: {result['sequence_embeddings'].shape}")
             print(f"  Protein embeddings shape: {result['protein_embeddings'].shape}")
             
             if 'binding_site_probabilities' in result:
                 probs = result['binding_site_probabilities']
                 num_sites = np.sum(probs > 0.5)
                 max_prob = np.max(probs)
                 print(f"  Predicted binding sites: {num_sites}")
                 print(f"  Max binding probability: {max_prob:.3f}")
                 
                 # Show top 3 binding sites
                 top_indices = np.argsort(probs)[-3:][::-1]
                 print("  Top binding sites:")
                 for idx in top_indices:
                     print(f"    Position {idx+1}: {probs[idx]:.3f}")
        
        # Generate and print a single embedding vector per protein by mean pooling sequence embeddings
        # 'sequence_embeddings' has shape (sequence_length, embedding_dim), e.g., (L, 1024)
        # Mean pooling across the sequence dimension yields a (embedding_dim,) vector
        print("\nSingle-vector embeddings (mean-pooled) for each protein:")
        for pid, result in results.items():
            seq_emb = result['sequence_embeddings']  # shape (L, D)
            single_vector = np.mean(seq_emb, axis=0)  # shape (D,)
            print(f"Protein {pid} single vector shape: {single_vector.shape}")
            print(f"Protein {pid} single vector: {single_vector}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in basic usage: {str(e)}")
        return None


def example_advanced_usage():
    """Demonstrate advanced features of the embedding model."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Advanced Usage with Analysis")
    print("="*60)
    
    # Load data
    protein_features, protein_sequences = load_example_data()
    demo_ids = list(protein_features.keys())[:5]
    demo_features = {pid: protein_features[pid] for pid in demo_ids}
    demo_sequences = {pid: protein_sequences[pid] for pid in demo_ids}
    
    try:
        # Initialize with custom settings
        embedder = Pseq2SitesEmbeddings(device="auto")
        
        # Extract embeddings with all features
        results = embedder.extract_embeddings(
            protein_features=demo_features,
            protein_sequences=demo_sequences,
            return_predictions=True,
            return_attention=True
        )
        
        # Generate binding site summary
        print("\nGenerating binding site analysis...")
        summary = embedder.get_binding_site_summary(results, threshold=0.5)
        print(summary.to_string(index=False))
        
        # Save embeddings
        output_dir = "results/embeddings/"
        os.makedirs(output_dir, exist_ok=True)
        
        embedder.save_embeddings(
            results, 
            os.path.join(output_dir, "demo_binding_site_embeddings.pkl")
        )
        print(f"\n✓ Embeddings saved to {output_dir}")
        
        # Demonstrate loading
        loaded_results = embedder.load_embeddings(
            os.path.join(output_dir, "demo_binding_site_embeddings.pkl")
        )
        print(f"✓ Loaded embeddings for {len(loaded_results)} proteins")
        
        return results, summary
        
    except Exception as e:
        print(f"❌ Error in advanced usage: {str(e)}")
        return None, None


def example_convenience_function():
    """Demonstrate the convenience function."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Convenience Function")
    print("="*60)
    
    # Load data
    protein_features, protein_sequences = load_example_data()
    demo_ids = list(protein_features.keys())[:2]
    demo_features = {pid: protein_features[pid] for pid in demo_ids}
    demo_sequences = {pid: protein_sequences[pid] for pid in demo_ids}
    
    try:
        # Use convenience function
        print("Using convenience function for quick extraction...")
        results = extract_binding_site_embeddings(
            protein_features=demo_features,
            protein_sequences=demo_sequences,
            batch_size=1
        )
        
        print(f"✓ Quick extraction completed for {len(results)} proteins")
        
        # Simple analysis
        for pid, result in results.items():
            probs = result['binding_site_probabilities']
            sites = np.where(probs > 0.5)[0]
            print(f"Protein {pid}: {len(sites)} binding sites predicted")
            
        return results
        
    except Exception as e:
        print(f"❌ Error in convenience function: {str(e)}")
        return None


def main():
    """Run all examples."""
    print("Pseq2Sites Binding Site Embedding Examples")
    print("=" * 60)
    print("This script demonstrates how to use the Pseq2Sites embedding model")
    print("to extract binding site representations from protein sequences.")
    
    # Run examples
    results1 = example_basic_usage()
    results2, summary = example_advanced_usage()
    results3 = example_convenience_function()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if results1 is not None:
        print("✓ Basic usage example completed successfully")
    if results2 is not None:
        print("✓ Advanced usage example completed successfully")
    if results3 is not None:
        print("✓ Convenience function example completed successfully")
        
    print("\nThe Pseq2Sites embedding model can extract:")
    print("• Sequence-level embeddings (256-dim per residue)")
    print("• Protein-level embeddings (256-dim per residue)")
    print("• Binding site probability predictions")
    print("• Attention weights from the model")
    print("\nThese embeddings capture binding site information and can be used")
    print("for downstream machine learning tasks involving protein-ligand binding.")


if __name__ == "__main__":
    main()
