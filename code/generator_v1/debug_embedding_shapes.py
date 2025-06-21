#!/usr/bin/env python3
"""
Debug embedding shapes in preprocessed data to identify the shape mismatch.
"""
import os
import sys
import pickle
import numpy as np

def check_embedding_shapes(data_path):
    """Check embedding shapes in both train and test data."""
    print(f"Loading data from {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    train_data = data['train_data']
    test_data = data['test_data']
    
    print("\n" + "="*60)
    print("TRAIN DATA EMBEDDING SHAPES")
    print("="*60)
    
    print(f"ProtBERT embeddings: {train_data['protbert_embeddings'].shape}")
    print(f"Pseq2Sites embeddings: {train_data['pseq2sites_embeddings'].shape}")
    print(f"Compound embeddings: {train_data['compound_embeddings'].shape}")
    
    print("\n" + "="*60)
    print("TEST DATA EMBEDDING SHAPES")
    print("="*60)
    
    print(f"ProtBERT embeddings: {test_data['protbert_embeddings'].shape}")
    print(f"Pseq2Sites embeddings: {test_data['pseq2sites_embeddings'].shape}")
    print(f"Compound embeddings: {test_data['compound_embeddings'].shape}")
    
    # Check if there are shape mismatches
    print("\n" + "="*60)
    print("SHAPE CONSISTENCY CHECK")
    print("="*60)
    
    protbert_match = train_data['protbert_embeddings'].shape[1] == test_data['protbert_embeddings'].shape[1]
    pseq2sites_match = train_data['pseq2sites_embeddings'].shape[1] == test_data['pseq2sites_embeddings'].shape[1]
    compound_match = train_data['compound_embeddings'].shape[1] == test_data['compound_embeddings'].shape[1]
    
    print(f"ProtBERT dimensions match: {protbert_match}")
    print(f"Pseq2Sites dimensions match: {pseq2sites_match}")
    print(f"Compound dimensions match: {compound_match}")
    
    if not compound_match:
        print(f"\n❌ COMPOUND EMBEDDING MISMATCH DETECTED!")
        print(f"Train compound embedding dim: {train_data['compound_embeddings'].shape[1]}")
        print(f"Test compound embedding dim: {test_data['compound_embeddings'].shape[1]}")
        
        # Sample some compound data to understand the issue
        print(f"\nTrain compound embeddings (first 3 samples):")
        for i in range(min(3, len(train_data['compound_embeddings']))):
            print(f"  Sample {i}: shape={train_data['compound_embeddings'][i].shape}")
        
        print(f"\nTest compound embeddings (first 3 samples):")
        for i in range(min(3, len(test_data['compound_embeddings']))):
            print(f"  Sample {i}: shape={test_data['compound_embeddings'][i].shape}")
        
        return False
    
    print("✅ All embedding dimensions are consistent!")
    return True

def main():
    """Main function."""
    # Check both original and fixed data
    original_path = './preprocessed_data/preprocessed_data.pkl'
    fixed_path = './preprocessed_data/preprocessed_data_fixed.pkl'
    
    for path, name in [(original_path, "ORIGINAL"), (fixed_path, "FIXED")]:
        if os.path.exists(path):
            print(f"\n{'='*80}")
            print(f"CHECKING {name} DATA: {path}")
            print(f"{'='*80}")
            check_embedding_shapes(path)
        else:
            print(f"\n⚠️  File not found: {path}")

if __name__ == "__main__":
    main()
