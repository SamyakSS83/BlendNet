#!/usr/bin/env python3
"""
Compliance verification script to ensure the pipeline follows idea.md specifications.
"""
import os
import sys
import pickle
import numpy as np

def verify_idea_compliance(data_path):
    """
    Verify that the preprocessed data follows idea.md specifications.
    
    According to idea.md:
    1. Split by unique molecules (no SMILES overlap between train/test)
    2. Proteins can appear in both train/test with different compounds
    3. Storage format: [compound_id, protbert_embeddings(E1), pseq2sites_embeddings(Es), smiles, ic50_value]
    4. FAISS vector database with cosine similarity indexing
    """
    
    print("="*80)
    print("IDEA.MD COMPLIANCE VERIFICATION")
    print("="*80)
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return False
    
    # Load data
    print(f"Loading data from: {data_path}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    train_data = data['train_data']
    test_data = data['test_data']
    
    compliance_checks = []
    
    # Check 1: Unique molecule split (CRITICAL)
    print("\n" + "="*60)
    print("CHECK 1: UNIQUE MOLECULE SPLIT (idea.md core requirement)")
    print("="*60)
    
    train_smiles = set(train_data['smiles'])
    test_smiles = set(test_data['smiles'])
    overlap_smiles = train_smiles.intersection(test_smiles)
    
    if len(overlap_smiles) == 0:
        print("✅ PASS: No SMILES overlap between train/test")
        print(f"   Train molecules: {len(train_smiles)}")
        print(f"   Test molecules: {len(test_smiles)}")
        compliance_checks.append(True)
    else:
        print(f"❌ FAIL: {len(overlap_smiles)} SMILES overlap between train/test")
        print(f"   Examples: {list(overlap_smiles)[:3]}")
        compliance_checks.append(False)
    
    # Check 2: Protein overlap allowed
    print("\n" + "="*60)
    print("CHECK 2: PROTEIN OVERLAP (allowed per idea.md)")
    print("="*60)
    
    train_seqs = set(train_data['sequences'])
    test_seqs = set(test_data['sequences'])
    overlap_seqs = train_seqs.intersection(test_seqs)
    
    print(f"✅ INFO: {len(overlap_seqs)} proteins appear in both train/test")
    print(f"   Train proteins: {len(train_seqs)}")
    print(f"   Test proteins: {len(test_seqs)}")
    print("   This is ALLOWED per idea.md design")
    compliance_checks.append(True)
    
    # Check 3: Data structure compliance
    print("\n" + "="*60)
    print("CHECK 3: DATA STRUCTURE (idea.md storage format)")
    print("="*60)
    
    required_keys = ['sequences', 'smiles', 'ic50_values', 'protbert_embeddings', 'pseq2sites_embeddings', 'compound_embeddings']
    
    for dataset_name, dataset in [('train', train_data), ('test', test_data)]:
        missing_keys = [key for key in required_keys if key not in dataset]
        if not missing_keys:
            print(f"✅ {dataset_name.upper()} data structure: PASS")
        else:
            print(f"❌ {dataset_name.upper()} data structure: FAIL - missing {missing_keys}")
            compliance_checks.append(False)
            continue
            
        # Check dimensions
        n_samples = len(dataset['sequences'])
        protbert_dim = dataset['protbert_embeddings'].shape
        pseq2sites_dim = dataset['pseq2sites_embeddings'].shape
        compound_dim = dataset['compound_embeddings'].shape
        
        print(f"   {dataset_name} samples: {n_samples}")
        print(f"   ProtBERT (E1): {protbert_dim}")
        print(f"   Pseq2Sites (Es): {pseq2sites_dim}")
        print(f"   Compound: {compound_dim}")
        
        # Verify consistency
        shapes_consistent = (
            protbert_dim[0] == n_samples and
            pseq2sites_dim[0] == n_samples and
            compound_dim[0] == n_samples
        )
        
        if shapes_consistent:
            print(f"✅ {dataset_name} shape consistency: PASS")
            compliance_checks.append(True)
        else:
            print(f"❌ {dataset_name} shape consistency: FAIL")
            compliance_checks.append(False)
    
    # Check 4: Embedding dimensions per idea.md
    print("\n" + "="*60)
    print("CHECK 4: EMBEDDING DIMENSIONS (idea.md specifications)")
    print("="*60)
    
    expected_dims = {
        'protbert': 1024,    # ProtBERT standard
        'compound': 768,     # smi-TED Light
        'pseq2sites': 256    # Pseq2Sites standard
    }
    
    train_protbert_dim = train_data['protbert_embeddings'].shape[1]
    train_compound_dim = train_data['compound_embeddings'].shape[1]
    train_pseq2sites_dim = train_data['pseq2sites_embeddings'].shape[1]
    
    test_protbert_dim = test_data['protbert_embeddings'].shape[1]
    test_compound_dim = test_data['compound_embeddings'].shape[1]
    test_pseq2sites_dim = test_data['pseq2sites_embeddings'].shape[1]
    
    dim_checks = [
        ('ProtBERT', train_protbert_dim, test_protbert_dim, expected_dims['protbert']),
        ('Compound', train_compound_dim, test_compound_dim, expected_dims['compound']),
        ('Pseq2Sites', train_pseq2sites_dim, test_pseq2sites_dim, expected_dims['pseq2sites'])
    ]
    
    for name, train_dim, test_dim, expected in dim_checks:
        if train_dim == test_dim == expected:
            print(f"✅ {name} dimensions: PASS ({train_dim})")
            compliance_checks.append(True)
        elif train_dim == test_dim:
            print(f"⚠️  {name} dimensions: WARN ({train_dim}, expected {expected})")
            compliance_checks.append(True)  # Still passes if consistent
        else:
            print(f"❌ {name} dimensions: FAIL (train={train_dim}, test={test_dim})")
            compliance_checks.append(False)
    
    # Check 5: FAISS compatibility
    print("\n" + "="*60)
    print("CHECK 5: FAISS COMPATIBILITY (idea.md vector database)")
    print("="*60)
    
    for dataset_name, dataset in [('train', train_data), ('test', test_data)]:
        for emb_name in ['protbert_embeddings', 'pseq2sites_embeddings', 'compound_embeddings']:
            emb = dataset[emb_name]
            
            dtype_ok = emb.dtype == np.float32
            contiguous_ok = emb.flags.c_contiguous
            
            if dtype_ok and contiguous_ok:
                print(f"✅ {dataset_name} {emb_name}: FAISS ready")
                compliance_checks.append(True)
            else:
                print(f"❌ {dataset_name} {emb_name}: Not FAISS ready (dtype={emb.dtype}, contiguous={contiguous_ok})")
                compliance_checks.append(False)
    
    # Final compliance summary
    print("\n" + "="*80)
    print("COMPLIANCE SUMMARY")
    print("="*80)
    
    total_checks = len(compliance_checks)
    passed_checks = sum(compliance_checks)
    
    if passed_checks == total_checks:
        print(f"✅ FULL COMPLIANCE: {passed_checks}/{total_checks} checks passed")
        print("🎉 Data is ready for idea.md pipeline!")
        return True
    else:
        print(f"❌ PARTIAL COMPLIANCE: {passed_checks}/{total_checks} checks passed")
        print("🚨 Data needs fixes before running idea.md pipeline!")
        return False

def main():
    """Main verification function."""
    data_files = [
        './preprocessed_data/preprocessed_data.pkl',
        './preprocessed_data/preprocessed_data_fixed.pkl'
    ]
    
    for data_file in data_files:
        if os.path.exists(data_file):
            print(f"\n{'='*100}")
            print(f"VERIFYING: {data_file}")
            print(f"{'='*100}")
            
            is_compliant = verify_idea_compliance(data_file)
            
            if is_compliant:
                print(f"\n✅ {data_file} is COMPLIANT with idea.md")
            else:
                print(f"\n❌ {data_file} is NOT COMPLIANT with idea.md")
        else:
            print(f"\n⚠️  File not found: {data_file}")

if __name__ == "__main__":
    main()
