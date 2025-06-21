"""
Data preprocessing for protein-ligand diffusion model.
Loads IC50 data and generates embeddings for vector database.
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

from bindingdb_interface import BindingDBInterface
from pseq2sites_interface import get_protein_matrix
from inference.smi_ted_light.load import load_smi_ted


class DataPreprocessor:
    """Preprocess IC50 data and generate embeddings for vector database."""
    
    def __init__(self, 
                 ic50_data_path: str,
                 smi_ted_path: str = "../../../materials.smi-ted/smi-ted/inference/smi_ted_light",
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
        
        # Load smi-TED for compound encoding
        print("Loading smi-TED...")
        self.smi_ted = load_smi_ted(
            folder=smi_ted_path,
            ckpt_filename=smi_ted_ckpt
        )
        
        # Initialize BindingDB interface for protein embeddings
        print("Loading BindingDB interface...")
        self.bindingdb_interface = BindingDBInterface(
            config_path="../../BindingDB.yml",
            ki_weights="/home/sarvesh/scratch/GS/negroni_data/Blendnet/model_checkpoint/BindingDB/Ki/random_split/CV1/BlendNet_S.pth",
            ic50_weights="/home/sarvesh/scratch/GS/negroni_data/Blendnet/model_checkpoint/BindingDB/IC50/random_split/CV1/BlendNet_S.pth",
            device=device
        )
        
        print("Models loaded successfully!")
        
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
        df = df.dropna(subset=['SMILES', 'Sequence'])  # Adjust column names as needed
        df = df.drop_duplicates()
        
        print(f"After cleaning: {len(df)} records")
        return df
        
    def split_unique_molecules(self, df: pd.DataFrame, 
                              train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataset based on unique molecules."""
        print("Splitting dataset by unique molecules...")
        
        # Get unique SMILES
        unique_smiles = df['SMILES'].unique()
        print(f"Found {len(unique_smiles)} unique molecules")
        
        # Random split of unique molecules
        np.random.shuffle(unique_smiles)
        split_idx = int(len(unique_smiles) * train_ratio)
        
        train_smiles = set(unique_smiles[:split_idx])
        test_smiles = set(unique_smiles[split_idx:])
        
        # Split dataframe based on molecule membership
        train_df = df[df['SMILES'].isin(train_smiles)]
        test_df = df[df['SMILES'].isin(test_smiles)]
        
        print(f"Train set: {len(train_df)} records ({len(train_smiles)} unique molecules)")
        print(f"Test set: {len(test_df)} records ({len(test_smiles)} unique molecules)")
        
        return train_df, test_df
        
    def generate_protein_embeddings(self, sequences: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ProtBERT and Pseq2Sites embeddings for protein sequences."""
        print("Generating protein embeddings...")
        
        protbert_embeddings = []
        pseq2sites_embeddings = []
        
        for i, seq in enumerate(tqdm(sequences, desc="Processing proteins")):
            try:
                # Generate ProtBERT features
                protbert_feat = self.bindingdb_interface._get_protein_features(seq)
                
                # Generate Pseq2Sites embeddings using the interface
                pseq2sites_feat = get_protein_matrix(protbert_feat, seq)
                
                # Pool to fixed size (mean pooling)
                protbert_pooled = np.mean(protbert_feat, axis=0)  # [1024]
                pseq2sites_pooled = np.mean(pseq2sites_feat, axis=0)  # [embedding_dim]
                
                protbert_embeddings.append(protbert_pooled)
                pseq2sites_embeddings.append(pseq2sites_pooled)
                
            except Exception as e:
                print(f"Error processing sequence {i}: {e}")
                # Use zero embeddings as fallback
                protbert_embeddings.append(np.zeros(1024))
                pseq2sites_embeddings.append(np.zeros(256))  # Adjust dimension as needed
                
        return np.array(protbert_embeddings), np.array(pseq2sites_embeddings)
        
    def generate_compound_embeddings(self, smiles_list: List[str]) -> np.ndarray:
        """Generate smi-TED embeddings for compounds."""
        print("Generating compound embeddings...")
        
        # Filter valid SMILES
        valid_smiles = []
        valid_indices = []
        
        for i, smiles in enumerate(smiles_list):
            try:
                from rdkit import Chem
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_smiles.append(smiles)
                    valid_indices.append(i)
            except:
                pass
                
        print(f"Valid SMILES: {len(valid_smiles)}/{len(smiles_list)}")
        
        # Generate embeddings
        with torch.no_grad():
            embeddings = self.smi_ted.encode(valid_smiles, return_torch=False)
            
        # Create full embeddings array with zeros for invalid SMILES
        full_embeddings = np.zeros((len(smiles_list), embeddings.shape[1]))
        for i, valid_idx in enumerate(valid_indices):
            full_embeddings[valid_idx] = embeddings[i]
            
        return full_embeddings
        
    def create_vector_database_data(self, df: pd.DataFrame, 
                                   output_path: str = "../database/") -> Dict:
        """Create and save preprocessed data for vector database."""
        print("Creating vector database data...")
        
        # Extract unique protein-compound pairs
        sequences = df['Sequence'].tolist()
        smiles = df['SMILES'].tolist()
        ic50_values = df['IC50'].tolist()  # Adjust column name as needed
        
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
