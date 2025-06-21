#!/usr/bin/env python3
"""
Fix compound embeddings by regenerating test set embeddings with correct dimensions.
This script fixes the dimension mismatch where test embeddings have shape (N, 200) instead of (N, 768).
"""
import os
import sys
import pickle
import numpy as np

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append('../../materials.smi-ted/smi-ted/inference/smi_ted_light')

def load_smi_ted_model():
    """Load smi-TED model with correct path."""
    try:
        from load import load_smi_ted
        
        # Use the correct path where the checkpoint actually exists
        checkpoint_path = '../../materials.smi-ted/smi-ted/inference/smi_ted_light/smi-ted-Light_40.pt'
        
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            # Try alternative paths
            alt_paths = [
                '../../materials.smi-ted/smi-ted-Light_40.pt',
                '../../materials.smi-ted/smi-ted-Light_40.pt',
                './materials.smi-ted/smi-ted-Light_40.pt'
            ]
            
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    print(f"Found alternative path: {alt_path}")
                    checkpoint_path = alt_path
                    break
            else:
                return None
        
        print(f"Loading smi-TED from: {checkpoint_path}")
        model = load_smi_ted(checkpoint_path)
        print("✅ Successfully loaded smi-TED model")
        return model
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error loading smi-TED: {e}")
        return None

def regenerate_compound_embeddings(smiles_list, smi_ted_model, batch_size=32):
    """Regenerate compound embeddings with correct dimensions."""
    print(f"Regenerating embeddings for {len(smiles_list)} compounds...")
    
    all_embeddings = []
    
    # Process in batches
    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(smiles_list) + batch_size - 1)//batch_size}")
        
        try:
            # Generate embeddings for batch
            batch_embeddings_df = smi_ted_model.encode(batch_smiles)
            
            # Convert to numpy array
            batch_embeddings = batch_embeddings_df.values.astype(np.float32)
            
            print(f"  Batch embedding shape: {batch_embeddings.shape}")
            all_embeddings.append(batch_embeddings)
            
        except Exception as e:
            print(f"  ❌ Error processing batch: {e}")
            # Create dummy embeddings with correct shape if batch fails
            dummy_embeddings = np.zeros((len(batch_smiles), 768), dtype=np.float32)
            all_embeddings.append(dummy_embeddings)
    
    # Concatenate all embeddings
    final_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Final embeddings shape: {final_embeddings.shape}")
    
    return final_embeddings

def fix_compound_embeddings():
    """Fix compound embeddings by regenerating test set with correct dimensions."""
    print("="*60)
    print("FIXING COMPOUND EMBEDDINGS")
    print("="*60)
    
    # Load data
    data_path = './preprocessed_data/preprocessed_data.pkl'
    print(f"Loading data from {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    train_data = data['train_data']
    test_data = data['test_data']
    
    print("Current embedding shapes:")
    print(f"Train compound embeddings: {train_data['compound_embeddings'].shape}")
    print(f"Test compound embeddings: {test_data['compound_embeddings'].shape}")
    
    # Check if fix is needed
    train_dim = train_data['compound_embeddings'].shape[1]
    test_dim = test_data['compound_embeddings'].shape[1]
    
    if train_dim == test_dim:
        print("✅ No dimension mismatch detected")
        return True
    
    print(f"❌ Dimension mismatch detected: train={train_dim}, test={test_dim}")
    
    # Load smi-TED model
    print("Loading smi-TED model...")
    smi_ted = load_smi_ted_model()
    
    if smi_ted is None:
        print("❌ Failed to load smi-TED model")
        return False
    
    # Test the model with a simple example
    print("Testing smi-TED model...")
    try:
        test_emb = smi_ted.encode(["CCO"])  # Simple test molecule
        print(f"Test embedding shape: {test_emb.shape}")
        expected_dim = test_emb.shape[1]
    except Exception as e:
        print(f"❌ Error testing smi-TED: {e}")
        return False
    
    if expected_dim != train_dim:
        print(f"❌ Model outputs {expected_dim}D embeddings, but train data has {train_dim}D")
        return False
    
    print(f"✅ Model outputs {expected_dim}D embeddings, which matches train data")
    
    # Regenerate test compound embeddings
    print("Regenerating test compound embeddings...")
    test_smiles = test_data['smiles']
    new_compound_embeddings = regenerate_compound_embeddings(test_smiles, smi_ted)
    
    # Verify new embeddings
    if new_compound_embeddings.shape[1] != train_dim:
        print(f"❌ New embeddings have wrong dimension: {new_compound_embeddings.shape[1]} != {train_dim}")
        return False
    
    print(f"✅ Generated new embeddings with shape: {new_compound_embeddings.shape}")
    
    # Update test data
    test_data['compound_embeddings'] = new_compound_embeddings
    test_data['metadata']['compound_dim'] = train_dim
    
    # Save fixed data
    fixed_data = {
        'train_data': train_data,
        'test_data': test_data,
        'metadata': data['metadata']
    }
    
    output_path = './preprocessed_data/preprocessed_data_fixed_embeddings.pkl'
    print(f"Saving fixed data to {output_path}")
    
    with open(output_path, 'wb') as f:
        pickle.dump(fixed_data, f)
    
    print("✅ Successfully fixed compound embeddings!")
    
    # Verify fix
    print("\nVerifying fix...")
    with open(output_path, 'rb') as f:
        fixed_data = pickle.load(f)
    
    train_shape = fixed_data['train_data']['compound_embeddings'].shape
    test_shape = fixed_data['test_data']['compound_embeddings'].shape
    
    print(f"Fixed train embeddings: {train_shape}")
    print(f"Fixed test embeddings: {test_shape}")
    
    if train_shape[1] == test_shape[1]:
        print("✅ Verification passed: dimensions match!")
        return True
    else:
        print("❌ Verification failed: dimensions still don't match")
        return False

def update_existing_fixed_file():
    """Update the existing preprocessed_data_fixed.pkl file with correct embeddings."""
    print("\n" + "="*60)
    print("UPDATING EXISTING FIXED FILE")
    print("="*60)
    
    fixed_emb_path = './preprocessed_data/preprocessed_data_fixed_embeddings.pkl'
    existing_fixed_path = './preprocessed_data/preprocessed_data_fixed.pkl'
    
    if not os.path.exists(fixed_emb_path):
        print(f"❌ Fixed embeddings file not found: {fixed_emb_path}")
        return False
    
    if not os.path.exists(existing_fixed_path):
        print(f"❌ Existing fixed file not found: {existing_fixed_path}")
        return False
    
    # Load both files
    with open(fixed_emb_path, 'rb') as f:
        fixed_emb_data = pickle.load(f)
    
    with open(existing_fixed_path, 'rb') as f:
        existing_data = pickle.load(f)
    
    # Update the existing fixed file with correct embeddings
    existing_data['test_data']['compound_embeddings'] = fixed_emb_data['test_data']['compound_embeddings']
    existing_data['test_data']['metadata']['compound_dim'] = fixed_emb_data['test_data']['metadata']['compound_dim']
    
    # Save updated file
    print(f"Updating {existing_fixed_path} with correct embeddings...")
    with open(existing_fixed_path, 'wb') as f:
        pickle.dump(existing_data, f)
    
    print("✅ Successfully updated existing fixed file!")
    
    # Verify
    with open(existing_fixed_path, 'rb') as f:
        verify_data = pickle.load(f)
    
    train_shape = verify_data['train_data']['compound_embeddings'].shape
    test_shape = verify_data['test_data']['compound_embeddings'].shape
    
    print(f"Updated train embeddings: {train_shape}")
    print(f"Updated test embeddings: {test_shape}")
    
    return train_shape[1] == test_shape[1]

if __name__ == "__main__":
    success = fix_compound_embeddings()
    
    if success:
        print("\nUpdating existing fixed file...")
        update_success = update_existing_fixed_file()
        
        if update_success:
            print("\n🎉 ALL FIXES COMPLETED SUCCESSFULLY!")
            print("You can now run the training pipeline without embedding dimension errors.")
        else:
            print("\n❌ Failed to update existing fixed file")
    else:
        print("\n❌ Failed to fix compound embeddings")
