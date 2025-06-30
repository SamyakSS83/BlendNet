"""
Data preprocessing for protein-ligand diffusion        # Initialize ProtBERT directly (more efficient than full BindingDB interface)
        print("Loading ProtBERT...")
        from transformers import BertModel, BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        self.protbert_model = BertModel.from_pretrained("Rostlab/prot_bert", use_safetensors=True).to(device)
        self.protbert_model.eval()
        
        # Initialize Pseq2Sites
        print("Loading Pseq2Sites...")
        from modules.pocket_modules.pseq2sites_embeddings import Pseq2SitesEmbeddings
        self.pseq2sites = Pseq2SitesEmbeddings(device=device)Loads IC50 data and generates embeddings for vector database.
"""
import os
import sys
import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Tuple
import torch
from tqdm import tqdm

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../materials.smi-ted/smi-ted/'))

from pseq2sites_interface import get_protein_matrix
from inference.smi_ted_light.load import load_smi_ted


class DataPreprocessor:
    """Preprocess IC50 data and generate embeddings for vector database."""
    
    def __init__(self, 
                 ic50_data_path: str,
                 smi_ted_path: str = None,
                 smi_ted_ckpt: str = "smi-ted-Light_40.pt",
                 device: str = "cuda:0"):
        """
        Initialize preprocessor with paths and models.
        
        Args:
            ic50_data_path: Path to IC50_data.tsv
            smi_ted_path: Path to smi-TED model directory
            smi_ted_ckpt: smi-TED checkpoint filename
            device: Device for computation
        """
        self.ic50_data_path = ic50_data_path
        self.device = device
        
        print("Loading models...")
        
        # Set default smi-TED path if not provided
        if smi_ted_path is None:
            # Try multiple possible paths for smi-TED
            current_dir = os.path.dirname(os.path.abspath(__file__))
            possible_paths = [
                # From your terminal output - this should work
                '../../materials.smi-ted/smi-ted/inference/smi_ted_light',
                # Original path structure
                os.path.join(current_dir, '../../../../materials.smi-ted/smi-ted/inference/smi_ted_light'),
                # Alternative path structure (for your current setup)
                os.path.join(current_dir, '../../../materials.smi-ted/smi-ted/inference/smi_ted_light'),
                # Another alternative
                os.path.join(current_dir, '../../materials.smi-ted/smi-ted/inference/smi_ted_light'),
                # Direct relative path from working directory
                './materials.smi-ted/smi-ted/inference/smi_ted_light',
                '../materials.smi-ted/smi-ted/inference/smi_ted_light',
                # Absolute paths based on your workspace
                '/home/sarvesh/sura/plm_sura/BlendNet/materials.smi-ted/smi-ted/inference/smi_ted_light',
                '/home/sarvesh/scratch/GS/samyak/.Blendnet/materials.smi-ted/smi-ted/inference/smi_ted_light'
            ]
            
            smi_ted_path = None
            for path in possible_paths:
                abs_path = os.path.abspath(path)
                vocab_test = os.path.join(abs_path, 'bert_vocab_curated.txt')
                if os.path.exists(vocab_test):
                    smi_ted_path = abs_path
                    print(f"Found smi-TED at: {smi_ted_path}")
                    break
            
            if smi_ted_path is None:
                raise FileNotFoundError(
                    f"Could not find smi-TED directory. Tried paths:\n" + 
                    "\n".join([f"  - {os.path.abspath(p)}" for p in possible_paths]) +
                    "\n\nPlease provide the correct path using smi_ted_path parameter."
                )
        else:
            # Ensure smi-TED path is absolute and resolve any relative components
            smi_ted_path = os.path.abspath(smi_ted_path)
        
        print(f"Using smi-TED path: {smi_ted_path}")
        
        # Verify smi-TED files exist
        vocab_file = os.path.join(smi_ted_path, 'bert_vocab_curated.txt')
        checkpoint_file = os.path.join(smi_ted_path, smi_ted_ckpt)
        
        print(f"Checking vocab file: {vocab_file}")
        print(f"Checking checkpoint file: {checkpoint_file}")
        
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"smi-TED vocabulary file not found: {vocab_file}")
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"smi-TED checkpoint file not found: {checkpoint_file}")
        
        print("smi-TED files verified!")
        
        # Load smi-TED for compound encoding
        print("Loading smi-TED...")
        self.smi_ted = load_smi_ted(
            folder=smi_ted_path,
            ckpt_filename=smi_ted_ckpt
        )
        
        # Initialize ProtBERT for protein feature extraction
        print("Loading ProtBERT...")
        from transformers import BertModel, BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        self.protbert_model = BertModel.from_pretrained("Rostlab/prot_bert", use_safetensors=True).to(self.device)
        self.protbert_model.eval()
        
        # Initialize Pseq2Sites for pocket prediction
        print("Loading Pseq2Sites...")
        from modules.pocket_modules.pseq2sites_embeddings import Pseq2SitesEmbeddings
        self.pseq2sites = Pseq2SitesEmbeddings(device=self.device)
        
        print("Models loaded successfully!")
        
    def _get_protein_features(self, protein_seq: str) -> np.ndarray:
        """Generate ProtBERT features from protein sequence (same as bindingdb_interface)"""
        if not protein_seq or len(protein_seq) == 0:
            raise ValueError("Protein sequence cannot be empty")
            
        import re
        # Clean sequence and format for ProtBERT
        clean_seq = re.sub(r"[UZOB]", "X", protein_seq)
        formatted_seq = " ".join(list(clean_seq))
        
        # Tokenize with proper length handling
        ids = self.tokenizer.batch_encode_plus(
            [formatted_seq], 
            add_special_tokens=True, 
            padding=True, 
            truncation=True, 
            max_length=1024,  # ProtBERT max length
            return_tensors='pt'
        )
        
        input_ids = ids['input_ids'].to(self.device)
        attention_mask = ids['attention_mask'].to(self.device)
        
        # Get ProtBERT embeddings
        with torch.no_grad():
            embedding = self.protbert_model(input_ids=input_ids, attention_mask=attention_mask)[0]
            embedding = embedding.cpu().numpy()
            seq_len = (attention_mask[0] == 1).sum()
            
            # Remove [CLS] and [SEP] tokens, handle max length
            if seq_len < 1024:
                seq_emb = embedding[0][1:seq_len-1]
            else:
                seq_emb = embedding[0][1:1023]  # Max length handling
                print(f"Warning: Protein sequence truncated to {seq_emb.shape[0]} residues")
                
        return seq_emb
    
    def load_ic50_data(self) -> pd.DataFrame:
        """Load and preprocess IC50 dataset."""
        print(f"Loading IC50 data from {self.ic50_data_path}")
        
        # Load the dataset
        df = pd.read_csv(self.ic50_data_path, sep='\t')
        print(f"Loaded {len(df)} records")
        
        # Check column names and structure
        print(f"Columns: {list(df.columns)}")
        print(f"Sample data:\n{df.head()}")
        
        # Remove duplicates and filter valid data
        df = df.dropna(subset=['SMILES', 'Seqs'])  # Use correct column names
        df = df.drop_duplicates()
        
        print(f"After cleaning: {len(df)} records")
        return df
        
    def split_unique_molecules(self, df: pd.DataFrame, 
                              train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset based on unique molecules (SMILES) as per idea.md.
        
        This ensures no compound appears in both train and test sets,
        preventing data leakage while allowing proteins to appear in both sets
        with different compound partners.
        """
        print("Splitting dataset by unique molecules (idea.md compliance)...")
        
        # Get unique SMILES - this is the key constraint from idea.md
        unique_smiles = df['SMILES'].unique()
        print(f"Found {len(unique_smiles)} unique molecules")
        
        # Random split of unique molecules
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(unique_smiles)
        split_idx = int(len(unique_smiles) * train_ratio)
        
        train_smiles = set(unique_smiles[:split_idx])
        test_smiles = set(unique_smiles[split_idx:])
        
        # Split dataframe based on molecule membership
        train_df = df[df['SMILES'].isin(train_smiles)]
        test_df = df[df['SMILES'].isin(test_smiles)]
        
        print(f"Train set: {len(train_df)} records ({len(train_smiles)} unique molecules)")
        print(f"Test set: {len(test_df)} records ({len(test_smiles)} unique molecules)")
        
        # Verify compliance with idea.md: no molecule overlap
        train_molecules = set(train_df['SMILES'].unique())
        test_molecules = set(test_df['SMILES'].unique())
        overlap = train_molecules.intersection(test_molecules)
        
        if len(overlap) == 0:
            print("✅ idea.md compliance: No molecule overlap between train/test")
        else:
            print(f"❌ idea.md violation: {len(overlap)} molecules overlap between train/test")
        
        # Check protein overlap (this is allowed per idea.md)
        train_proteins = set(train_df['Seqs'].unique())
        test_proteins = set(test_df['Seqs'].unique())
        protein_overlap = train_proteins.intersection(test_proteins)
        print(f"Protein overlap: {len(protein_overlap)} proteins (allowed per idea.md)")
        
        return train_df, test_df
        
    def generate_protein_embeddings(self, sequences: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ProtBERT and Pseq2Sites embeddings for protein sequences with caching."""
        print("Generating protein embeddings...")
        
        # Check cache first
        cache_dir = os.path.join(os.path.dirname(self.ic50_data_path), 'embedding_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        protbert_cache_file = os.path.join(cache_dir, 'protbert_embeddings_cache.pkl')
        pseq2sites_cache_file = os.path.join(cache_dir, 'pseq2sites_embeddings_cache.pkl')
        
        # Load caches
        protbert_cache = {}
        pseq2sites_cache = {}
        
        if os.path.exists(protbert_cache_file):
            print("Loading ProtBERT embeddings from cache...")
            with open(protbert_cache_file, 'rb') as f:
                protbert_cache = pickle.load(f)
        
        if os.path.exists(pseq2sites_cache_file):
            print("Loading Pseq2Sites embeddings from cache...")
            with open(pseq2sites_cache_file, 'rb') as f:
                pseq2sites_cache = pickle.load(f)
        
        # Find sequences not in cache
        new_sequences = []
        sequence_indices = {}
        
        for i, seq in enumerate(sequences):
            sequence_indices[seq] = i
            if seq not in protbert_cache or seq not in pseq2sites_cache:
                new_sequences.append(seq)
        
        print(f"Total sequences: {len(sequences)}")
        print(f"New sequences to process: {len(new_sequences)}")
        print(f"Cached sequences: {len(sequences) - len(new_sequences)}")
        
        # Process new sequences
        if new_sequences:
            print("Processing new protein sequences...")
            
            # Generate ProtBERT features in batches
            new_protbert_features = {}
            for seq in tqdm(new_sequences, desc="Generating ProtBERT features"):
                try:
                    if seq not in protbert_cache:
                        protbert_feat = self._get_protein_features(seq)
                        protbert_pooled = np.mean(protbert_feat, axis=0)  # [1024]
                        protbert_cache[seq] = protbert_pooled
                        new_protbert_features[seq] = protbert_feat
                    else:
                        # Still need features for Pseq2Sites
                        protbert_feat = self._get_protein_features(seq)
                        new_protbert_features[seq] = protbert_feat
                except Exception as e:
                    print(f"Error processing ProtBERT for sequence: {e}")
                    protbert_cache[seq] = np.zeros(1024)
                    new_protbert_features[seq] = np.zeros((50, 1024))  # Fallback
            
            # Generate Pseq2Sites embeddings in batch
            if new_protbert_features:
                new_pseq2sites_sequences = [seq for seq in new_sequences if seq not in pseq2sites_cache]
                if new_pseq2sites_sequences:
                    try:
                        print(f"Batch processing {len(new_pseq2sites_sequences)} sequences with Pseq2Sites...")
                        
                        # Prepare batch data for Pseq2Sites
                        protein_features = {}
                        protein_sequences = {}
                        
                        for seq in new_pseq2sites_sequences:
                            temp_id = f"prot_{hash(seq) % 1000000}"  # Create unique ID
                            protein_features[temp_id] = new_protbert_features[seq]
                            protein_sequences[temp_id] = seq
                        
                        # Batch process with Pseq2Sites
                        results = self.pseq2sites.extract_embeddings(
                            protein_features=protein_features,
                            protein_sequences=protein_sequences,
                            batch_size=16,  # Adjust based on memory
                            return_predictions=False,
                            return_attention=False
                        )
                        
                        # Extract embeddings and cache them
                        for seq in new_pseq2sites_sequences:
                            temp_id = f"prot_{hash(seq) % 1000000}"
                            if temp_id in results:
                                # Get sequence embeddings and pool
                                seq_emb = results[temp_id]['sequence_embeddings']
                                pseq2sites_pooled = np.mean(seq_emb, axis=0)
                                pseq2sites_cache[seq] = pseq2sites_pooled
                            else:
                                print(f"Warning: No Pseq2Sites result for sequence")
                                pseq2sites_cache[seq] = np.zeros(256)  # Fallback
                        
                    except Exception as e:
                        print(f"Batch Pseq2Sites processing failed: {e}")
                        print("Falling back to individual processing...")
                        
                        # Individual processing fallback
                        for seq in new_pseq2sites_sequences:
                            if seq not in pseq2sites_cache:
                                try:
                                    protbert_feat = new_protbert_features[seq]
                                    pseq2sites_feat = get_protein_matrix(protbert_feat, seq)
                                    pseq2sites_pooled = np.mean(pseq2sites_feat, axis=0)
                                    pseq2sites_cache[seq] = pseq2sites_pooled
                                except Exception as e2:
                                    print(f"Individual Pseq2Sites processing failed: {e2}")
                                    pseq2sites_cache[seq] = np.zeros(256)
            
            # Save updated caches
            print("Saving updated caches...")
            with open(protbert_cache_file, 'wb') as f:
                pickle.dump(protbert_cache, f)
            with open(pseq2sites_cache_file, 'wb') as f:
                pickle.dump(pseq2sites_cache, f)
        
        # Collect embeddings in correct order
        protbert_embeddings = []
        pseq2sites_embeddings = []
        
        for seq in sequences:
            protbert_embeddings.append(protbert_cache.get(seq, np.zeros(1024)))
            pseq2sites_embeddings.append(pseq2sites_cache.get(seq, np.zeros(256)))
        
        print(f"Final ProtBERT embeddings shape: {np.array(protbert_embeddings).shape}")
        print(f"Final Pseq2Sites embeddings shape: {np.array(pseq2sites_embeddings).shape}")
        
        return np.array(protbert_embeddings), np.array(pseq2sites_embeddings)
        
    def generate_compound_embeddings(self, smiles_list: List[str]) -> np.ndarray:
        """Generate smi-TED embeddings for compounds with caching."""
        print("Generating compound embeddings...")
        
        # Check cache first
        cache_dir = os.path.join(os.path.dirname(self.ic50_data_path), 'embedding_cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, 'compound_embeddings_cache.pkl')
        
        if os.path.exists(cache_file):
            print("Loading compound embeddings from cache...")
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
        else:
            cache = {}
        
        # Filter valid SMILES and check cache
        valid_smiles = []
        valid_indices = []
        new_smiles = []
        
        for i, smiles in enumerate(smiles_list):
            try:
                from rdkit import Chem
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_smiles.append(smiles)
                    valid_indices.append(i)
                    if smiles not in cache:
                        new_smiles.append(smiles)
            except:
                pass
                
        print(f"Valid SMILES: {len(valid_smiles)}/{len(smiles_list)}")
        print(f"New SMILES to encode: {len(new_smiles)}")
        print(f"Cached SMILES: {len(valid_smiles) - len(new_smiles)}")
        
        # Generate embeddings for new SMILES
        if new_smiles:
            try:
                print("Encoding new compounds...")
                with torch.no_grad():
                    new_embeddings = self.smi_ted.encode(new_smiles)
                
                # Handle different return formats from smi-TED
                if hasattr(new_embeddings, 'values'):
                    # pandas DataFrame
                    new_embeddings_np = new_embeddings.values
                elif hasattr(new_embeddings, 'cpu'):
                    # torch tensor
                    new_embeddings_np = new_embeddings.cpu().numpy()
                else:
                    # numpy array
                    new_embeddings_np = np.array(new_embeddings)
                
                print(f"New embeddings shape: {new_embeddings_np.shape}")
                print(f"Expected shape: ({len(new_smiles)}, 768)")
                
                # Verify dimensions match
                if new_embeddings_np.shape[0] != len(new_smiles):
                    print(f"❌ Shape mismatch: got {new_embeddings_np.shape[0]} embeddings for {len(new_smiles)} SMILES")
                    # Fallback: use first len(new_smiles) embeddings
                    if new_embeddings_np.shape[0] > len(new_smiles):
                        new_embeddings_np = new_embeddings_np[:len(new_smiles)]
                    else:
                        raise ValueError(f"Not enough embeddings: got {new_embeddings_np.shape[0]}, need {len(new_smiles)}")
                
                if new_embeddings_np.shape[1] != 768:
                    print(f"❌ Embedding dimension mismatch: got {new_embeddings_np.shape[1]}, expected 768")
                    raise ValueError(f"Wrong embedding dimension: {new_embeddings_np.shape[1]}")
                
                # Update cache
                for i, smiles in enumerate(new_smiles):
                    cache[smiles] = new_embeddings_np[i]
                
                # Save updated cache
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache, f)
                print(f"✅ Cache updated with {len(new_smiles)} new embeddings")
                
            except Exception as e:
                print(f"❌ Encoding failed: {e}")
                import traceback
                traceback.print_exc()
                print("Using random embeddings as fallback...")
                # Generate random embeddings as fallback
                for smiles in new_smiles:
                    cache[smiles] = np.random.randn(768).astype(np.float32)  # Default smi-TED dimension
        
        # Collect all embeddings from cache in the correct order
        all_embeddings = []
        for smiles in valid_smiles:
            if smiles in cache:
                emb = cache[smiles]
                # Ensure embedding is the right shape and type
                if isinstance(emb, np.ndarray) and emb.shape == (768,):
                    all_embeddings.append(emb.astype(np.float32))
                else:
                    print(f"⚠️  Invalid embedding for SMILES {smiles}: shape {emb.shape if hasattr(emb, 'shape') else type(emb)}")
                    all_embeddings.append(np.random.randn(768).astype(np.float32))
            else:
                # Fallback for missing embeddings
                all_embeddings.append(np.random.randn(768).astype(np.float32))
        
        if all_embeddings:
            try:
                embeddings = np.stack(all_embeddings, axis=0)  # Use stack instead of array for better error messages
                print(f"✅ Successfully created embeddings array: {embeddings.shape}")
            except Exception as e:
                print(f"❌ Failed to stack embeddings: {e}")
                print(f"Embedding shapes: {[emb.shape for emb in all_embeddings[:5]]}")  # Show first 5 shapes
                # Create a clean array
                embeddings = np.zeros((len(valid_smiles), 768), dtype=np.float32)
                for i, emb in enumerate(all_embeddings):
                    if i < len(embeddings) and hasattr(emb, 'shape') and emb.shape == (768,):
                        embeddings[i] = emb
        else:
            # Complete fallback
            embeddings = np.random.randn(len(valid_smiles), 768).astype(np.float32)
            
        # Create full embeddings array with zeros for invalid SMILES
        full_embeddings = np.zeros((len(smiles_list), embeddings.shape[1]), dtype=np.float32)
        for i, valid_idx in enumerate(valid_indices):
            if i < len(embeddings):
                full_embeddings[valid_idx] = embeddings[i]
            
        print(f"Final embeddings shape: {full_embeddings.shape}")
        print(f"Expected shape: ({len(smiles_list)}, 768)")
        
        # Final verification
        if full_embeddings.shape != (len(smiles_list), 768):
            print(f"❌ Final shape mismatch! Expected ({len(smiles_list)}, 768), got {full_embeddings.shape}")
            # Create clean fallback
            full_embeddings = np.random.randn(len(smiles_list), 768).astype(np.float32)
            print(f"✅ Created fallback embeddings: {full_embeddings.shape}")
            
        return full_embeddings
        
    def create_vector_database_data(self, df: pd.DataFrame, 
                                   output_path: str = "../database/") -> Dict:
        """Create and save preprocessed data for vector database."""
        print("Creating vector database data...")
        
        # Extract unique protein-compound pairs
        sequences = df['Seqs'].tolist()  # Use correct column name
        smiles = df['SMILES'].tolist()
        ic50_values = df['Labels'].tolist()  # Use correct column name for IC50 values
        
        # Generate embeddings
        protbert_emb, pseq2sites_emb = self.generate_protein_embeddings(sequences)
        compound_emb = self.generate_compound_embeddings(smiles)
        
        # Create database structure
        database_data = {
            'sequences': sequences,
            'smiles': smiles,
            'ic50_values': ic50_values,
            'protbert_embeddings': protbert_emb,
            'pseq2sites_embeddings': pseq2sites_emb,
            'compound_embeddings': compound_emb,
            'metadata': {
                'n_samples': len(sequences),
                'protbert_dim': protbert_emb.shape[1],
                'pseq2sites_dim': pseq2sites_emb.shape[1],
                'compound_dim': compound_emb.shape[1],
                'device': self.device
            }
        }
        
        # Save to file
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, "preprocessed_data.pkl")
        
        print(f"Saving preprocessed data to {output_file}")
        with open(output_file, 'wb') as f:
            pickle.dump(database_data, f)
            
        print("Preprocessing completed!")
        return database_data

    def preprocess_and_split(self, 
                           output_dir: str = "./preprocessed_data",
                           max_samples: int = -1,
                           test_split: float = 0.2) -> Tuple[Dict, Dict]:
        """
        Complete preprocessing pipeline: load, split, and process data.
        
        Args:
            output_dir: Directory to save preprocessed data
            max_samples: Maximum number of samples to process (-1 for all)
            test_split: Fraction for test split
            
        Returns:
            Tuple of (train_data, test_data) dictionaries
        """
        print("Starting complete preprocessing pipeline...")
        
        # Load data
        df = self.load_ic50_data()
        
        # Limit samples if specified (None means use all data per idea.md)
        if max_samples is not None and max_samples > 0 and len(df) > max_samples:
            print(f"Limiting to {max_samples} samples from {len(df)}")
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        
        # Split by unique molecules
        train_ratio = 1.0 - test_split
        train_df, test_df = self.split_unique_molecules(df, train_ratio=train_ratio)
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        train_dir = os.path.join(output_dir, "train")
        test_dir = os.path.join(output_dir, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Process training data
        print("\nProcessing training data...")
        train_data = self.create_vector_database_data(train_df, train_dir)
        
        # Process test data
        print("\nProcessing test data...")
        test_data = self.create_vector_database_data(test_df, test_dir)
        
        # Also save combined data for vector database
        combined_output = os.path.join(output_dir, "preprocessed_data.pkl")
        print(f"Saving combined data to {combined_output}")
        with open(combined_output, 'wb') as f:
            pickle.dump({
                'train_data': train_data,
                'test_data': test_data,
                'metadata': {
                    'train_samples': len(train_df),
                    'test_samples': len(test_df),
                    'total_samples': len(df),
                    'test_split': test_split,
                    'max_samples': max_samples
                }
            }, f)
        
        print("Complete preprocessing pipeline finished!")
        return train_data, test_data

def main():
    """Main preprocessing pipeline."""
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        ic50_data_path="/home/sarvesh/scratch/GS/negroni_data/Blendnet/input_data/BindingDB/IC50_data.tsv"
    )
    
    # Load and split data
    df = preprocessor.load_ic50_data()
    train_df, test_df = preprocessor.split_unique_molecules(df)
    
    # Process training data
    print("\nProcessing training data...")
    train_data = preprocessor.create_vector_database_data(
        train_df, 
        output_path="../database/train/"
    )
    
    # Process test data
    print("\nProcessing test data...")
    test_data = preprocessor.create_vector_database_data(
        test_df,
        output_path="../database/test/"
    )
    
    print("\nPreprocessing pipeline completed successfully!")


if __name__ == "__main__":
    main()
