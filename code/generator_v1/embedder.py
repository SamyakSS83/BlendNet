"""
Embedder module for creating protein-ligand vector database.

This module processes the IC50 dataset, groups by proteins, keeps top-3 binding ligands
per protein, generates embeddings, and creates a FAISS vector database.

Compliance with idea.md:
- Groups data by unique proteins (not molecules)
- Keeps top 3 (m=3) binding ligands per protein based on IC50
- Generates ProtBERT + Pseq2Sites embeddings for proteins
- Generates smi-TED embeddings for ligands
- Creates FAISS database for efficient similarity search
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import torch
from typing import Dict, List, Tuple, Any
import logging
from tqdm import tqdm
import faiss

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../materials.smi-ted/smi-ted/'))

# Import required modules
try:
    from inference.smi_ted_light.load import load_smi_ted
    from preprocess.data_preprocessor import DataPreprocessor
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are available")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProteinLigandEmbedder:
    """
    Creates protein-ligand embeddings and vector database according to idea.md architecture.
    """
    
    def __init__(self, 
                 data_path: str,
                 output_dir: str = "./embedder_output",
                 top_m_ligands: int = 3,
                 device: str = "cuda"):
        """
        Initialize the embedder.
        
        Args:
            data_path: Path to IC50_data.tsv
            output_dir: Directory to save outputs
            top_m_ligands: Number of top binding ligands to keep per protein (m=3)
            device: Device for computations
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.top_m_ligands = top_m_ligands
        self.device = device
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.preprocessor = None
        self.smi_ted = None
        
        logger.info(f"Embedder initialized - keeping top {top_m_ligands} ligands per protein")
        
    def load_and_process_data(self) -> pd.DataFrame:
        """
        Load IC50 data and process according to idea.md specifications.
        
        Returns:
            Processed DataFrame grouped by proteins with top-m ligands
        """
        logger.info("Loading IC50 dataset...")
        
        # Load the dataset
        df = pd.read_csv(self.data_path, sep='\t')
        logger.info(f"Loaded {len(df)} total protein-ligand pairs")
        
        # Check required columns
        required_cols = ['UniProt_IDs', 'CID', 'Labels', 'Seqs', 'SMILES', 'SMILES_iso', 'Reactant_IDs']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Clean data
        df = df.dropna(subset=['UniProt_IDs', 'Seqs', 'SMILES', 'Labels'])
        logger.info(f"After cleaning: {len(df)} pairs")
        
        # Convert IC50 labels to numeric (assuming they are IC50 values)
        df['IC50'] = pd.to_numeric(df['Labels'], errors='coerce')
        df = df.dropna(subset=['IC50'])
        
        # Group by protein sequence and keep top-m binding ligands per protein
        logger.info(f"Grouping by proteins and selecting top {self.top_m_ligands} ligands per protein...")
        
        # Sort by IC50 (lower is better binding)
        df = df.sort_values('IC50')
        
        # Group by protein sequence and keep top-m ligands
        protein_groups = []
        for seq, group in tqdm(df.groupby('Seqs'), desc="Processing proteins"):
            # Keep top-m ligands for this protein (lowest IC50 values)
            top_ligands = group.head(self.top_m_ligands)
            protein_groups.append(top_ligands)
            
        processed_df = pd.concat(protein_groups, ignore_index=True)
        
        # Get unique proteins for statistics
        unique_proteins = processed_df['Seqs'].nunique()
        total_pairs = len(processed_df)
        
        logger.info(f"Final dataset: {unique_proteins} unique proteins, {total_pairs} protein-ligand pairs")
        logger.info(f"Average ligands per protein: {total_pairs/unique_proteins:.2f}")
        
        return processed_df
        
    def initialize_models(self):
        """Initialize ProtBERT, Pseq2Sites, and smi-TED models."""
        logger.info("Initializing embedding models...")
        
        # Initialize data preprocessor (contains ProtBERT and Pseq2Sites)
        self.preprocessor = DataPreprocessor()
        
        # Initialize smi-TED
        logger.info("Loading smi-TED...")
        smi_ted_paths = [
            '../../materials.smi-ted/smi-ted/inference/smi_ted_light',
            '../materials.smi-ted/smi-ted/inference/smi_ted_light',
            './materials.smi-ted/smi-ted/inference/smi_ted_light'
        ]
        
        smi_ted_path = None
        for path in smi_ted_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(os.path.join(abs_path, 'bert_vocab_curated.txt')):
                smi_ted_path = abs_path
                break
                
        if smi_ted_path is None:
            raise FileNotFoundError("Could not locate smi-TED files")
            
        self.smi_ted = load_smi_ted(
            folder=smi_ted_path,
            ckpt_filename="smi-ted-Light_40.pt"
        )
        
        logger.info("All models initialized successfully")
        
    def generate_protein_embeddings(self, sequences: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ProtBERT and Pseq2Sites embeddings for protein sequences.
        
        Args:
            sequences: List of protein sequences
            
        Returns:
            Tuple of (protbert_embeddings, pseq2sites_embeddings)
        """
        logger.info(f"Generating protein embeddings for {len(sequences)} sequences...")
        
        # Get unique sequences to avoid redundant computation
        unique_sequences = list(set(sequences))
        seq_to_idx = {seq: i for i, seq in enumerate(unique_sequences)}
        
        # Generate embeddings for unique sequences
        protbert_embeddings = []
        pseq2sites_embeddings = []
        
        for seq in tqdm(unique_sequences, desc="Generating protein embeddings"):
            # Generate ProtBERT embedding
            protbert_emb = self.preprocessor.get_protbert_embedding(seq)
            protbert_embeddings.append(protbert_emb)
            
            # Generate Pseq2Sites embedding
            pseq2sites_emb = self.preprocessor.get_pseq2sites_embedding(seq)
            pseq2sites_embeddings.append(pseq2sites_emb)
            
        # Convert to arrays
        protbert_embeddings = np.array(protbert_embeddings)
        pseq2sites_embeddings = np.array(pseq2sites_embeddings)
        
        # Map back to original sequence order
        original_protbert = np.array([protbert_embeddings[seq_to_idx[seq]] for seq in sequences])
        original_pseq2sites = np.array([pseq2sites_embeddings[seq_to_idx[seq]] for seq in sequences])
        
        logger.info(f"Generated embeddings - ProtBERT: {original_protbert.shape}, Pseq2Sites: {original_pseq2sites.shape}")
        
        return original_protbert, original_pseq2sites
        
    def generate_compound_embeddings(self, smiles_list: List[str]) -> np.ndarray:
        """
        Generate smi-TED embeddings for SMILES.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Array of compound embeddings
        """
        logger.info(f"Generating compound embeddings for {len(smiles_list)} SMILES...")
        
        # Get unique SMILES to avoid redundant computation
        unique_smiles = list(set(smiles_list))
        smiles_to_idx = {smiles: i for i, smiles in enumerate(unique_smiles)}
        
        # Generate embeddings for unique SMILES
        compound_embeddings = []
        
        batch_size = 32  # Process in batches to manage memory
        for i in tqdm(range(0, len(unique_smiles), batch_size), desc="Generating compound embeddings"):
            batch_smiles = unique_smiles[i:i+batch_size]
            try:
                batch_embeddings = self.smi_ted.encode(batch_smiles)
                if isinstance(batch_embeddings, torch.Tensor):
                    batch_embeddings = batch_embeddings.cpu().numpy()
                compound_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.warning(f"Failed to encode batch {i//batch_size}: {e}")
                # Add zero embeddings for failed batch
                batch_size_actual = len(batch_smiles)
                zero_embeddings = np.zeros((batch_size_actual, 768))  # smi-TED dimension
                compound_embeddings.extend(zero_embeddings)
                
        compound_embeddings = np.array(compound_embeddings)
        
        # Map back to original SMILES order
        original_embeddings = np.array([compound_embeddings[smiles_to_idx[smiles]] for smiles in smiles_list])
        
        logger.info(f"Generated compound embeddings: {original_embeddings.shape}")
        
        return original_embeddings
        
    def create_vector_database(self, 
                             protein_data: Dict[str, Any],
                             use_gpu: bool = True) -> faiss.Index:
        """
        Create FAISS vector database for protein similarity search.
        
        According to idea.md, we use: argmax(alpha * sim(Es) + (1-alpha) * sim(E1))
        where Es = Pseq2Sites, E1 = ProtBERT
        
        Args:
            protein_data: Dictionary containing protein embeddings and ligand data
            use_gpu: Whether to use GPU for FAISS
            
        Returns:
            FAISS index
        """
        logger.info("Creating FAISS vector database...")
        
        # Get protein embeddings
        protbert_embeddings = protein_data['protbert_embeddings']
        pseq2sites_embeddings = protein_data['pseq2sites_embeddings']
        
        # Combine embeddings according to idea.md similarity metric
        # We'll store both separately and combine during search
        # For now, we'll create index on combined embeddings with alpha=0.5
        alpha = 0.5
        combined_embeddings = alpha * pseq2sites_embeddings + (1 - alpha) * protbert_embeddings
        
        # Ensure embeddings are float32 and contiguous
        combined_embeddings = np.ascontiguousarray(combined_embeddings.astype(np.float32))
        
        dimension = combined_embeddings.shape[1]
        n_proteins = combined_embeddings.shape[0]
        
        logger.info(f"Creating index for {n_proteins} proteins with dimension {dimension}")
        
        # Create FAISS index
        try:
            if use_gpu and hasattr(faiss, 'StandardGpuResources'):
                # Try GPU first
                res = faiss.StandardGpuResources()
                index = faiss.GpuIndexFlatIP(res, dimension)
                logger.info("Using FAISS-GPU")
            else:
                raise Exception("GPU not available")
        except Exception:
            # Fall back to CPU
            index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            logger.info("Using FAISS-CPU")
            
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(combined_embeddings)
        
        # Add to index
        index.add(combined_embeddings)
        
        logger.info(f"Vector database created with {index.ntotal} proteins")
        
        return index
        
    def run_embedding_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete embedding pipeline according to idea.md.
        
        Returns:
            Dictionary containing all processed data and embeddings
        """
        logger.info("Starting embedding pipeline...")
        
        # Step 1: Load and process data
        df = self.load_and_process_data()
        
        # Step 2: Initialize models
        self.initialize_models()
        
        # Step 3: Generate protein embeddings
        sequences = df['Seqs'].tolist()
        protbert_embeddings, pseq2sites_embeddings = self.generate_protein_embeddings(sequences)
        
        # Step 4: Generate compound embeddings
        smiles_list = df['SMILES'].tolist()
        compound_embeddings = self.generate_compound_embeddings(smiles_list)
        
        # Step 5: Organize data by proteins with their top-m ligands
        logger.info("Organizing data by proteins...")
        
        protein_database = {}
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building protein database"):
            seq = row['Seqs']
            
            if seq not in protein_database:
                protein_database[seq] = {
                    'protbert_embedding': protbert_embeddings[idx],
                    'pseq2sites_embedding': pseq2sites_embeddings[idx],
                    'ligands': []
                }
                
            # Add ligand information
            ligand_info = {
                'smiles': row['SMILES'],
                'ic50': row['IC50'],
                'compound_embedding': compound_embeddings[idx],
                'uniprot_id': row['UniProt_IDs'],
                'cid': row['CID']
            }
            protein_database[seq]['ligands'].append(ligand_info)
            
        # Step 6: Create arrays for FAISS database
        protein_sequences = list(protein_database.keys())
        protein_protbert_embeddings = np.array([protein_database[seq]['protbert_embedding'] for seq in protein_sequences])
        protein_pseq2sites_embeddings = np.array([protein_database[seq]['pseq2sites_embedding'] for seq in protein_sequences])
        
        # Step 7: Create FAISS vector database
        protein_data = {
            'protbert_embeddings': protein_protbert_embeddings,
            'pseq2sites_embeddings': protein_pseq2sites_embeddings
        }
        faiss_index = self.create_vector_database(protein_data)
        
        # Step 8: Save everything
        output_data = {
            'protein_database': protein_database,
            'protein_sequences': protein_sequences,
            'protein_protbert_embeddings': protein_protbert_embeddings,
            'protein_pseq2sites_embeddings': protein_pseq2sites_embeddings,
            'faiss_index': faiss_index,
            'metadata': {
                'total_proteins': len(protein_sequences),
                'top_m_ligands': self.top_m_ligands,
                'protbert_dim': protein_protbert_embeddings.shape[1],
                'pseq2sites_dim': protein_pseq2sites_embeddings.shape[1],
                'compound_dim': compound_embeddings.shape[1] if len(compound_embeddings) > 0 else 768
            }
        }
        
        # Save to files
        self.save_embeddings(output_data)
        
        logger.info("Embedding pipeline completed successfully!")
        
        return output_data
        
    def save_embeddings(self, data: Dict[str, Any]):
        """Save embeddings and database to files."""
        logger.info("Saving embeddings to files...")
        
        # Save protein database
        protein_db_path = os.path.join(self.output_dir, 'protein_database.pkl')
        with open(protein_db_path, 'wb') as f:
            pickle.dump({
                'protein_database': data['protein_database'],
                'protein_sequences': data['protein_sequences'],
                'metadata': data['metadata']
            }, f)
            
        # Save FAISS index
        faiss_path = os.path.join(self.output_dir, 'protein_faiss_index.faiss')
        faiss.write_index(data['faiss_index'], faiss_path)
        
        # Save embeddings arrays
        embeddings_path = os.path.join(self.output_dir, 'protein_embeddings.npz')
        np.savez_compressed(
            embeddings_path,
            protbert_embeddings=data['protein_protbert_embeddings'],
            pseq2sites_embeddings=data['protein_pseq2sites_embeddings']
        )
        
        logger.info(f"Saved embeddings to {self.output_dir}")
        
    def load_embeddings(self) -> Dict[str, Any]:
        """Load previously saved embeddings."""
        logger.info("Loading embeddings from files...")
        
        # Load protein database
        protein_db_path = os.path.join(self.output_dir, 'protein_database.pkl')
        with open(protein_db_path, 'rb') as f:
            protein_data = pickle.load(f)
            
        # Load FAISS index
        faiss_path = os.path.join(self.output_dir, 'protein_faiss_index.faiss')
        faiss_index = faiss.read_index(faiss_path)
        
        # Load embeddings arrays
        embeddings_path = os.path.join(self.output_dir, 'protein_embeddings.npz')
        embeddings = np.load(embeddings_path)
        
        data = {
            'protein_database': protein_data['protein_database'],
            'protein_sequences': protein_data['protein_sequences'],
            'protein_protbert_embeddings': embeddings['protbert_embeddings'],
            'protein_pseq2sites_embeddings': embeddings['pseq2sites_embeddings'],
            'faiss_index': faiss_index,
            'metadata': protein_data['metadata']
        }
        
        logger.info("Embeddings loaded successfully!")
        
        return data


def main():
    """Main function to run the embedding pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create protein-ligand embeddings and vector database")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to IC50_data.tsv")
    parser.add_argument("--output_dir", type=str, default="./embedder_output",
                       help="Output directory for embeddings")
    parser.add_argument("--top_m_ligands", type=int, default=3,
                       help="Number of top binding ligands to keep per protein")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device for computations")
    
    args = parser.parse_args()
    
    # Create embedder and run pipeline
    embedder = ProteinLigandEmbedder(
        data_path=args.data_path,
        output_dir=args.output_dir,
        top_m_ligands=args.top_m_ligands,
        device=args.device
    )
    
    # Run the pipeline
    result = embedder.run_embedding_pipeline()
    
    print(f"✅ Embedding pipeline completed!")
    print(f"✅ Created database with {result['metadata']['total_proteins']} proteins")
    print(f"✅ Each protein has up to {result['metadata']['top_m_ligands']} ligands")
    print(f"✅ Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
