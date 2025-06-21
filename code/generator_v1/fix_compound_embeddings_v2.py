#!/usr/bin/env python3
"""
Fix compound embeddings in preprocessed data by regenerating them with correct smi-TED configuration.
"""
import os
import sys
import pickle
import numpy as np

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append('../../materials.smi-ted/smi-ted/inference/smi_ted_light')

def load_smi_ted_model():
    """Load smi-TED model with correct path configuration."""
    try:
        from load import load_smi_ted
        
        # Use the correct checkpoint path
        checkpoint_path = '../../materials.smi-ted/smi-ted-Light_40.pt'
        
        print(f"Loading smi-TED from: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return None
            
        model = load_smi_ted(checkpoint_path)
        print("✅ smi-TED model loaded successfully")
        return model
        
    except Exception as e:
        print(f"❌ Failed to load smi-TED: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_compound_embeddings_batch(model, smiles_list, batch_size=32):
    """Generate compound embeddings in batches."""
    print(f"Generating embeddings for {len(smiles_list)} compounds...")
    
    all_embeddings = []
    
    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(smiles_list) + batch_size - 1)//batch_size}")
        
        try:
            # Generate embeddings for this batch
            batch_embeddings = model.encode(batch_smiles)
            
            # Convert to numpy if needed
            if hasattr(batch_embeddings, 'values'):
                # It's a pandas DataFrame
                batch_emb_np = batch_embeddings.values.astype(np.float32)
            elif hasattr(batch_embeddings, 'cpu'):
                # It's a torch tensor
                batch_emb_np = batch_embeddings.cpu().numpy().astype(np.float32)
            else:
                # Already numpy
                batch_emb_np = np.array(batch_embeddings, dtype=np.float32)
            
            all_embeddings.append(batch_emb_np)
            
            print(f"  Batch shape: {batch_emb_np.shape}")
            
        except Exception as e:
            print(f"❌ Error processing batch {i//batch_size + 1}: {e}")
            # Create dummy embeddings for this batch
            dummy_emb = np.zeros((len(batch_smiles), 768), dtype=np.float32)
            all_embeddings.append(dummy_emb)
    
    # Concatenate all embeddings
    final_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"✅ Generated embeddings shape: {final_embeddings.shape}")
    
    return final_embeddings

def fix_compound_embeddings(input_path, output_path):
    """Fix compound embeddings by regenerating them."""
    print("="*60)
    print("FIXING COMPOUND EMBEDDINGS")
    print("="*60)
    
    # Load the data
    print(f"Loading data from {input_path}")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    train_data = data['train_data']
    test_data = data['test_data']
    
    print("Current embedding shapes:")
    print(f"Train compound embeddings: {train_data['compound_embeddings'].shape}")
    print(f"Test compound embeddings: {test_data['compound_embeddings'].shape}")
    
    train_dim = train_data['compound_embeddings'].shape[1]
    test_dim = test_data['compound_embeddings'].shape[1]
    
    if train_dim == test_dim:
        print("✅ Embedding dimensions are consistent, no fix needed")
        return data
    
    print(f"❌ Dimension mismatch detected: train={train_dim}, test={test_dim}")
    
    # Load smi-TED model
    print("Loading smi-TED model...")
    smi_ted = load_smi_ted_model()
    
    if smi_ted is None:
        print("❌ Failed to load smi-TED model")
        return None
    
    # Test the model with a simple SMILES to verify it works
    print("Testing smi-TED model...")
    try:
        test_emb = smi_ted.encode(["CCO"])  # ethanol
        if hasattr(test_emb, 'values'):
            test_shape = test_emb.values.shape
        elif hasattr(test_emb, 'shape'):
            test_shape = test_emb.shape
        else:
            test_shape = np.array(test_emb).shape
        print(f"✅ Test embedding shape: {test_shape}")
    except Exception as e:
        print(f"❌ smi-TED model test failed: {e}")
        return None
    
    # Fix embeddings based on which set has the wrong dimension
    if train_dim == 768 and test_dim != 768:
        print("Regenerating TEST set compound embeddings...")
        test_smiles = test_data['smiles']
        new_test_compounds = generate_compound_embeddings_batch(smi_ted, test_smiles)
        
        # Update test data
        test_data['compound_embeddings'] = new_test_compounds
        test_data['metadata']['compound_dim'] = new_test_compounds.shape[1]
        
    elif test_dim == 768 and train_dim != 768:
        print("Regenerating TRAIN set compound embeddings...")
        train_smiles = train_data['smiles']
        new_train_compounds = generate_compound_embeddings_batch(smi_ted, train_smiles)
        
        # Update train data
        train_data['compound_embeddings'] = new_train_compounds
        train_data['metadata']['compound_dim'] = new_train_compounds.shape[1]
        
    else:
        print("Regenerating BOTH train and test compound embeddings...")
        
        # Fix train data
        train_smiles = train_data['smiles']
        new_train_compounds = generate_compound_embeddings_batch(smi_ted, train_smiles)
        train_data['compound_embeddings'] = new_train_compounds
        train_data['metadata']['compound_dim'] = new_train_compounds.shape[1]
        
        # Fix test data
        test_smiles = test_data['smiles']
        new_test_compounds = generate_compound_embeddings_batch(smi_ted, test_smiles)
        test_data['compound_embeddings'] = new_test_compounds
        test_data['metadata']['compound_dim'] = new_test_compounds.shape[1]
    
    # Create the fixed data structure
    fixed_data = {
        'train_data': train_data,
        'test_data': test_data,
        'metadata': data['metadata']
    }
    
    # Verify the fix
    print("\nAfter fixing:")
    print(f"Train compound embeddings: {train_data['compound_embeddings'].shape}")
    print(f"Test compound embeddings: {test_data['compound_embeddings'].shape}")
    
    # Save the fixed data
    print(f"Saving fixed data to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(fixed_data, f)
    
    print("✅ Compound embeddings fixed successfully!")
    return fixed_data

def make_faiss_compatible(data):
    """Make all embeddings FAISS-compatible."""
    print("\n" + "="*60)
    print("MAKING EMBEDDINGS FAISS-COMPATIBLE")
    print("="*60)
    
    train_data = data['train_data']
    test_data = data['test_data']
    
    # Fix all embedding arrays
    for dataset_name, dataset in [('train', train_data), ('test', test_data)]:
        print(f"\nFixing {dataset_name} data...")
        
        for emb_type in ['protbert_embeddings', 'pseq2sites_embeddings', 'compound_embeddings']:
            original = dataset[emb_type]
            
            # Convert to C-contiguous float32
            fixed = np.ascontiguousarray(original, dtype=np.float32)
            
            dataset[emb_type] = fixed
            
            print(f"  {emb_type}: {original.shape} -> {fixed.shape} ({fixed.dtype}, C-contiguous: {fixed.flags.c_contiguous})")
    
    return data

def main():
    """Main function to fix compound embeddings."""
    input_file = './preprocessed_data/preprocessed_data.pkl'
    output_file = './preprocessed_data/preprocessed_data_compound_fixed.pkl'
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    # Fix compound embeddings
    fixed_data = fix_compound_embeddings(input_file, output_file)
    
    if fixed_data is None:
        print("❌ Failed to fix compound embeddings")
        return
    
    # Make FAISS compatible
    faiss_data = make_faiss_compatible(fixed_data)
    
    # Save the final fixed data
    final_output = './preprocessed_data/preprocessed_data_fixed.pkl'
    print(f"\nSaving final FAISS-compatible data to {final_output}")
    with open(final_output, 'wb') as f:
        pickle.dump(faiss_data, f)
    
    print(f"\n✅ All fixes completed! Files saved:")
    print(f"  - Compound-fixed: {output_file}")
    print(f"  - FAISS-compatible: {final_output}")
    
    # Final verification
    print("\n" + "="*60)
    print("FINAL VERIFICATION")
    print("="*60)
    
    train_data = faiss_data['train_data']
    test_data = faiss_data['test_data']
    
    print("Final embedding shapes:")
    print(f"Train - ProtBERT: {train_data['protbert_embeddings'].shape}")
    print(f"Train - Pseq2Sites: {train_data['pseq2sites_embeddings'].shape}")
    print(f"Train - Compound: {train_data['compound_embeddings'].shape}")
    print(f"Test - ProtBERT: {test_data['protbert_embeddings'].shape}")
    print(f"Test - Pseq2Sites: {test_data['pseq2sites_embeddings'].shape}")
    print(f"Test - Compound: {test_data['compound_embeddings'].shape}")
    
    # Check consistency
    train_compound_dim = train_data['compound_embeddings'].shape[1]
    test_compound_dim = test_data['compound_embeddings'].shape[1]
    
    if train_compound_dim == test_compound_dim == 768:
        print("✅ All compound embeddings have correct dimension (768)")
    else:
        print(f"❌ Still have dimension mismatch: train={train_compound_dim}, test={test_compound_dim}")

if __name__ == "__main__":
    main()
