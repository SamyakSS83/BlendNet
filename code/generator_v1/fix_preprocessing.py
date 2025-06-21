#!/usr/bin/env python3
"""
Fix preprocessing data to ensure FAISS-GPU compatibility.
This script will regenerate the preprocessed data with arrays that work with FAISS.
"""
import os
import sys
import pickle
import numpy as np

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

def fix_preprocessed_data(input_path, output_path):
    """Fix preprocessed data for FAISS compatibility."""
    print(f"Loading preprocessed data from {input_path}")
    
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    print("Original data structure:")
    for key in data.keys():
        print(f"  {key}: {type(data[key])}")
    
    # Extract train data
    train_data = data['train_data']
    test_data = data['test_data']
    
    print("\nFixing train data embeddings...")
    train_data_fixed = fix_embeddings(train_data)
    
    print("\nFixing test data embeddings...")
    test_data_fixed = fix_embeddings(test_data)
    
    # Create new data structure
    fixed_data = {
        'train_data': train_data_fixed,
        'test_data': test_data_fixed,
        'metadata': data['metadata']
    }
    
    # Save fixed data
    print(f"Saving fixed data to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(fixed_data, f)
    
    print("✅ Fixed preprocessed data saved!")
    return fixed_data

def fix_embeddings(data):
    """Fix embedding arrays for FAISS compatibility."""
    fixed_data = {}
    
    # Copy non-embedding data
    for key, value in data.items():
        if not key.endswith('_embeddings'):
            fixed_data[key] = value
    
    # Fix embeddings with explicit array creation
    print(f"Fixing ProtBERT embeddings...")
    protbert_orig = data['protbert_embeddings']
    protbert_fixed = create_faiss_compatible_array(protbert_orig)
    fixed_data['protbert_embeddings'] = protbert_fixed
    
    print(f"Fixing Pseq2Sites embeddings...")
    pseq2sites_orig = data['pseq2sites_embeddings']
    pseq2sites_fixed = create_faiss_compatible_array(pseq2sites_orig)
    fixed_data['pseq2sites_embeddings'] = pseq2sites_fixed
    
    print(f"Fixing compound embeddings...")
    compound_orig = data['compound_embeddings']
    compound_fixed = create_faiss_compatible_array(compound_orig)
    fixed_data['compound_embeddings'] = compound_fixed
    
    return fixed_data

def create_faiss_compatible_array(original_array):
    """Create a FAISS-compatible numpy array."""
    # Convert to Python list first, then back to numpy with explicit settings
    if isinstance(original_array, np.ndarray):
        # Convert to Python list to completely break any memory layout issues
        as_list = original_array.tolist()
    else:
        as_list = list(original_array)
    
    # Create fresh numpy array with explicit parameters
    new_array = np.array(as_list, dtype=np.float32, order='C')
    
    # Ensure it's properly formatted
    new_array = np.ascontiguousarray(new_array, dtype=np.float32)
    
    print(f"  Original shape: {original_array.shape if hasattr(original_array, 'shape') else 'N/A'}")
    print(f"  Fixed shape: {new_array.shape}")
    print(f"  Fixed dtype: {new_array.dtype}")
    print(f"  C-contiguous: {new_array.flags.c_contiguous}")
    print(f"  Own data: {new_array.flags.owndata}")
    
    return new_array

def test_faiss_compatibility(array, dim_name):
    """Test if array works with FAISS."""
    try:
        import faiss
        
        # Test with CPU first
        print(f"Testing {dim_name} with FAISS-CPU...")
        index_cpu = faiss.IndexFlatIP(array.shape[1])
        index_cpu.add(array)
        print(f"✅ {dim_name} works with FAISS-CPU")
        
        # Test with GPU if available
        try:
            print(f"Testing {dim_name} with FAISS-GPU...")
            res = faiss.StandardGpuResources()
            index_gpu = faiss.GpuIndexFlatIP(res, array.shape[1])
            index_gpu.add(array)
            print(f"✅ {dim_name} works with FAISS-GPU")
            return True, True  # CPU, GPU
        except Exception as e:
            print(f"⚠️  {dim_name} fails with FAISS-GPU: {e}")
            return True, False  # CPU works, GPU doesn't
            
    except Exception as e:
        print(f"❌ {dim_name} fails with FAISS: {e}")
        return False, False

def main():
    """Main function to fix preprocessing data."""
    input_file = './preprocessed_data/preprocessed_data.pkl'
    output_file = './preprocessed_data/preprocessed_data_fixed.pkl'
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    # Fix the data
    fixed_data = fix_preprocessed_data(input_file, output_file)
    
    # Test FAISS compatibility
    print("\n" + "="*60)
    print("TESTING FAISS COMPATIBILITY")
    print("="*60)
    
    train_data = fixed_data['train_data']
    
    protbert_cpu, protbert_gpu = test_faiss_compatibility(
        train_data['protbert_embeddings'], 'ProtBERT'
    )
    pseq2sites_cpu, pseq2sites_gpu = test_faiss_compatibility(
        train_data['pseq2sites_embeddings'], 'Pseq2Sites'
    )
    compound_cpu, compound_gpu = test_faiss_compatibility(
        train_data['compound_embeddings'], 'Compound'
    )
    
    print("\n" + "="*60)
    print("COMPATIBILITY SUMMARY")
    print("="*60)
    print(f"ProtBERT:   CPU={protbert_cpu}, GPU={protbert_gpu}")
    print(f"Pseq2Sites: CPU={pseq2sites_cpu}, GPU={pseq2sites_gpu}")
    print(f"Compound:   CPU={compound_cpu}, GPU={compound_gpu}")
    
    if all([protbert_cpu, pseq2sites_cpu, compound_cpu]):
        print("✅ All embeddings work with FAISS-CPU")
        if all([protbert_gpu, pseq2sites_gpu, compound_gpu]):
            print("✅ All embeddings work with FAISS-GPU")
        else:
            print("⚠️  Some embeddings don't work with FAISS-GPU, will fallback to CPU")
    else:
        print("❌ Some embeddings don't work with FAISS at all")
    
    print(f"\nFixed data saved to: {output_file}")
    print("You can now use this file with the vector database.")

if __name__ == "__main__":
    main()
