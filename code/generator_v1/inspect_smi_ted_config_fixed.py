#!/usr/bin/env python3
"""
Inspect smi-TED model configuration to understand embedding dimensions.
"""
import os
import sys
import torch

# Add paths for smi-TED
sys.path.append('../../materials.smi-ted/smi-ted/inference/smi_ted_light')
sys.path.append('../../materials.smi-ted/smi-ted/finetune/smi_ted_light')
sys.path.append('../../materials.smi-ted/smi-ted')

def inspect_smi_ted_config():
    """Inspect smi-TED model configuration and embedding dimensions."""
    print("="*60)
    print("SMI-TED MODEL CONFIGURATION INSPECTION")
    print("="*60)
    
    try:
        # Try to import smi-TED components
        from load import load_smi_ted
        print("✅ Successfully imported load_smi_ted")
        
        # Load the model
        print("\nLoading smi-TED Light model...")
        smi_ted = load_smi_ted('../../materials.smi-ted/smi-ted-Light_40.pt')
        print("✅ Successfully loaded smi-TED Light model")
        
        # Inspect model architecture
        print("\n" + "="*40)
        print("MODEL ARCHITECTURE INSPECTION")
        print("="*40)
        
        print(f"Model type: {type(smi_ted)}")
        print(f"Model class: {smi_ted.__class__.__name__}")
        
        # Check if model has autoencoder
        if hasattr(smi_ted, 'autoencoder'):
            autoencoder = smi_ted.autoencoder
            print(f"Autoencoder type: {type(autoencoder)}")
            
            if hasattr(autoencoder, 'encoder'):
                encoder = autoencoder.encoder
                print(f"Encoder type: {type(encoder)}")
                
                # Try to get encoder dimensions
                if hasattr(encoder, 'feature_size'):
                    print(f"Encoder feature_size: {encoder.feature_size}")
                if hasattr(encoder, 'latent_size'):
                    print(f"Encoder latent_size: {encoder.latent_size}")
                    
        # Check model attributes
        print("\n" + "="*40)
        print("MODEL ATTRIBUTES")
        print("="*40)
        
        for attr in dir(smi_ted):
            if not attr.startswith('_'):
                try:
                    val = getattr(smi_ted, attr)
                    if isinstance(val, (int, float, str, list, tuple)):
                        print(f"{attr}: {val}")
                    elif hasattr(val, '__class__'):
                        print(f"{attr}: {val.__class__.__name__}")
                except:
                    print(f"{attr}: <cannot access>")
        
        # Test actual embedding generation
        print("\n" + "="*40)
        print("TESTING EMBEDDING GENERATION")
        print("="*40)
        
        # Test with a simple SMILES
        test_smiles = ["CCO", "CC(C)O", "CCCC"]  # ethanol, isopropanol, butane
        
        for i, smiles in enumerate(test_smiles):
            print(f"\nTesting SMILES {i+1}: {smiles}")
            try:
                embedding = smi_ted.encode([smiles])
                print(f"  Embedding shape: {embedding.shape}")
                print(f"  Embedding dtype: {embedding.dtype}")
                print(f"  Embedding device: {embedding.device}")
                
                # Convert to numpy and check
                emb_np = embedding.cpu().numpy()
                print(f"  Numpy shape: {emb_np.shape}")
                print(f"  Numpy dtype: {emb_np.dtype}")
                print(f"  First few values: {emb_np[0][:5]}")
                
            except Exception as e:
                print(f"  ❌ Error encoding {smiles}: {e}")
        
        return smi_ted
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Available Python path:")
        for path in sys.path:
            print(f"  {path}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_different_smi_ted_versions():
    """Test different smi-TED versions to see embedding dimensions."""
    print("\n" + "="*60)
    print("TESTING DIFFERENT SMI-TED VERSIONS")
    print("="*60)
    
    # Test smi-TED Light
    print("\n--- Testing smi-TED Light ---")
    sys.path.insert(0, '../../materials.smi-ted/smi-ted/inference/smi_ted_light')
    try:
        from load import load_smi_ted as load_light
        model_light = load_light('../../materials.smi-ted/smi-ted-Light_40.pt')
        test_emb = model_light.encode(["CCO"])
        print(f"smi-TED Light embedding shape: {test_emb.shape}")
    except Exception as e:
        print(f"smi-TED Light failed: {e}")
    
    # Test smi-TED Large  
    print("\n--- Testing smi-TED Large ---")
    sys.path.insert(0, '../../materials.smi-ted/smi-ted/inference/smi_ted_large')
    try:
        from load import load_smi_ted as load_large
        # Note: We don't have the large model checkpoint, so this will likely fail
        print("smi-TED Large model checkpoint not available")
    except Exception as e:
        print(f"smi-TED Large failed: {e}")

def inspect_checkpoint():
    """Inspect the actual checkpoint file."""
    print("\n" + "="*60)
    print("CHECKPOINT FILE INSPECTION")
    print("="*60)
    
    checkpoint_path = '../../materials.smi-ted/smi-ted-Light_40.pt'
    
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint exists: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"Checkpoint type: {type(checkpoint)}")
            
            if isinstance(checkpoint, dict):
                print("Checkpoint keys:")
                for key in checkpoint.keys():
                    val = checkpoint[key]
                    if torch.is_tensor(val):
                        print(f"  {key}: tensor {val.shape}")
                    else:
                        print(f"  {key}: {type(val)}")
            
            # Look for model state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("\nModel state dict keys:")
                for key, val in list(state_dict.items())[:10]:  # First 10 keys
                    print(f"  {key}: {val.shape}")
                if len(state_dict) > 10:
                    print(f"  ... and {len(state_dict) - 10} more keys")
                    
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")

if __name__ == "__main__":
    inspect_smi_ted_config()
    test_different_smi_ted_versions()
    inspect_checkpoint()
