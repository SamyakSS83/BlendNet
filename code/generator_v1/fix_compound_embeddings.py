#!/usr/bin/env python3
"""
Fix compound embedding dimension mismatch by regenerating test embeddings.
"""
import os
import sys
import pickle
import numpy as np
import torch

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

def fix_compound_embeddings(data_path, output_path):
    """Fix compound embeddings to ensure consistent dimensions."""
    print(f"Loading data from {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    train_data = data['train_data']
    test_data = data['test_data']
    
    print("Current embedding shapes:")
    print(f"Train compound embeddings: {train_data['compound_embeddings'].shape}")
    print(f"Test compound embeddings: {test_data['compound_embeddings'].shape}")
    
    # Check if there's a dimension mismatch
    train_dim = train_data['compound_embeddings'].shape[1]
    test_dim = test_data['compound_embeddings'].shape[1]
    
    if train_dim == test_dim:
        print("✅ No dimension mismatch detected")
        return data
    
    print(f"❌ Dimension mismatch detected: train={train_dim}, test={test_dim}")
    
    # Load smi-TED model to regenerate embeddings
    print("Loading smi-TED model...")
    
    # Load smi-TED with proper paths
    sys.path.insert(0, '../../materials.smi-ted/smi-ted/inference/smi_ted_light')
    try:
        from load import load_smi_ted
        smi_ted = load_smi_ted('../../materials.smi-ted/smi-ted-Light_40.pt')
        smi_ted.eval()
        print("✅ smi-TED model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load smi-TED: {e}")
        return None
    
    # Regenerate test compound embeddings
    print("Regenerating test compound embeddings...")
    
    # Get test SMILES (assuming they're stored in the data)
    if 'smiles' in test_data:
        test_smiles = test_data['smiles']
        print(f"Found {len(test_smiles)} test SMILES")
        
        # Generate new embeddings
        try:
            with torch.no_grad():
                new_embeddings = smi_ted.encode(test_smiles, return_torch=False)
            
            # Convert to numpy if needed
            if torch.is_tensor(new_embeddings):
                new_embeddings = new_embeddings.cpu().numpy()
            elif hasattr(new_embeddings, 'values'):  # pandas DataFrame
                new_embeddings = new_embeddings.values
            
            print(f"New test embeddings shape: {new_embeddings.shape}")
            
            # Ensure correct format
            new_embeddings = np.ascontiguousarray(new_embeddings, dtype=np.float32)
            
            # Update test data
            test_data_fixed = test_data.copy()
            test_data_fixed['compound_embeddings'] = new_embeddings
            
            # Create fixed data structure
            fixed_data = {
                'train_data': train_data,
                'test_data': test_data_fixed,
                'metadata': data['metadata']
            }
            
            print("✅ Test compound embeddings regenerated successfully")
            
            # Verify dimensions
            print("Fixed embedding shapes:")
            print(f"Train compound embeddings: {fixed_data['train_data']['compound_embeddings'].shape}")
            print(f"Test compound embeddings: {fixed_data['test_data']['compound_embeddings'].shape}")
            
            # Save fixed data
            print(f"Saving fixed data to {output_path}")
            with open(output_path, 'wb') as f:
                pickle.dump(fixed_data, f)
            
            return fixed_data
            
        except Exception as e:
            print(f"❌ Failed to regenerate embeddings: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        print("❌ No SMILES found in test data - cannot regenerate embeddings")
        return None

def find_smiles_in_data(data_path):
    """Find where SMILES are stored in the data structure."""
    print(f"Analyzing data structure in {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    def analyze_dict(d, path=""):
        """Recursively analyze dictionary structure."""
        for key, value in d.items():
            current_path = f"{path}.{key}" if path else key
            if isinstance(value, dict):
                print(f"{current_path}: dict with keys {list(value.keys())}")
                analyze_dict(value, current_path)
            elif isinstance(value, (list, tuple)):
                print(f"{current_path}: {type(value).__name__} of length {len(value)}")
                if len(value) > 0:
                    print(f"  First element type: {type(value[0])}")
                    if isinstance(value[0], str) and len(value[0]) < 100:
                        print(f"  First few elements: {value[:3]}")
            elif isinstance(value, np.ndarray):
                print(f"{current_path}: numpy array {value.shape}, dtype {value.dtype}")
            else:
                print(f"{current_path}: {type(value).__name__}")
                if isinstance(value, (int, float, str)) and len(str(value)) < 100:
                    print(f"  Value: {value}")
    
    analyze_dict(data)

def main():
    """Main function."""
    input_file = './preprocessed_data/preprocessed_data.pkl'
    output_file = './preprocessed_data/preprocessed_data_embedding_fixed.pkl'
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    # First, analyze the data structure to find SMILES
    print("="*60)
    print("ANALYZING DATA STRUCTURE")
    print("="*60)
    find_smiles_in_data(input_file)
    
    # Try to fix the embeddings
    print("\n" + "="*60)
    print("FIXING COMPOUND EMBEDDINGS")
    print("="*60)
    fixed_data = fix_compound_embeddings(input_file, output_file)
    
    if fixed_data is not None:
        print("✅ Compound embeddings fixed successfully!")
        print(f"Fixed data saved to: {output_file}")
    else:
        print("❌ Failed to fix compound embeddings")

if __name__ == "__main__":
    main()
