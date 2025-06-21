"""
FAISS vector database for efficient similarity search.
Stores protein and compound embeddings for retrieval-augmented generation.
"""
import os
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
import faiss


class ProteinLigandVectorDB:
    """FAISS-based vector database for protein-ligand pairs."""
    
    def __init__(self, 
                 protbert_dim: int = 1024,
                 pseq2sites_dim: int = 256,
                 compound_dim: int = 512):
        """
        Initialize vector database.
        
        Args:
            protbert_dim: Dimension of ProtBERT embeddings
            pseq2sites_dim: Dimension of Pseq2Sites embeddings  
            compound_dim: Dimension of compound embeddings
        """
        self.protbert_dim = protbert_dim
        self.pseq2sites_dim = pseq2sites_dim
        self.compound_dim = compound_dim
        
        # Initialize FAISS indices
        self.protbert_index = None
        self.pseq2sites_index = None
        self.compound_index = None
        
        # Store data
        self.sequences = []
        self.smiles = []
        self.ic50_values = []
        self.compound_embeddings = None
        
    def build_index(self, data: Dict):
        """Build FAISS indices from preprocessed data."""
        print("="*50)
        print("STARTING BUILD_INDEX METHOD")
        print("="*50)
        
        # Debug: Check if data is properly passed
        print(f"Data keys: {list(data.keys())}")
        print(f"Data type: {type(data)}")
        
        # Extract data
        self.sequences = data['sequences']
        self.smiles = data['smiles']
        self.ic50_values = np.array(data['ic50_values'])
        print(f"Extracted basic data: {len(self.sequences)} sequences, {len(self.smiles)} SMILES")
        
        # Debug: Check data types and shapes
        print(f"ProtBERT embeddings type: {type(data['protbert_embeddings'])}")
        print(f"Pseq2Sites embeddings type: {type(data['pseq2sites_embeddings'])}")
        print(f"Compound embeddings type: {type(data['compound_embeddings'])}")
        
        # Convert embeddings to numpy arrays with error handling
        print("Converting embeddings to numpy arrays...")
        try:
            protbert_emb = data['protbert_embeddings']
            if not isinstance(protbert_emb, np.ndarray):
                protbert_emb = np.array(protbert_emb)
            protbert_emb = protbert_emb.astype('float32')
            print(f"ProtBERT shape: {protbert_emb.shape}, dtype: {protbert_emb.dtype}")
            
            pseq2sites_emb = data['pseq2sites_embeddings']
            if not isinstance(pseq2sites_emb, np.ndarray):
                pseq2sites_emb = np.array(pseq2sites_emb)
            pseq2sites_emb = pseq2sites_emb.astype('float32')
            print(f"Pseq2Sites shape: {pseq2sites_emb.shape}, dtype: {pseq2sites_emb.dtype}")
            
            compound_emb = data['compound_embeddings']
            if not isinstance(compound_emb, np.ndarray):
                compound_emb = np.array(compound_emb)
            compound_emb = compound_emb.astype('float32')
            print(f"Compound shape: {compound_emb.shape}, dtype: {compound_emb.dtype}")
            
        except Exception as e:
            print(f"Error converting embeddings to numpy arrays: {e}")
            print("Data structure:")
            for key, value in data.items():
                if key.endswith('_embeddings'):
                    print(f"  {key}: type={type(value)}, shape={getattr(value, 'shape', 'N/A')}")
            raise
        
        # Validate embeddings
        if len(protbert_emb.shape) != 2:
            raise ValueError(f"ProtBERT embeddings must be 2D, got shape: {protbert_emb.shape}")
        if len(pseq2sites_emb.shape) != 2:
            raise ValueError(f"Pseq2Sites embeddings must be 2D, got shape: {pseq2sites_emb.shape}")
        if len(compound_emb.shape) != 2:
            raise ValueError(f"Compound embeddings must be 2D, got shape: {compound_emb.shape}")
        
        # Handle NaN/inf values
        protbert_emb = np.nan_to_num(protbert_emb, nan=0.0, posinf=0.0, neginf=0.0)
        pseq2sites_emb = np.nan_to_num(pseq2sites_emb, nan=0.0, posinf=0.0, neginf=0.0)
        compound_emb = np.nan_to_num(compound_emb, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize embeddings for cosine similarity
        print("Normalizing embeddings...")
        protbert_emb = self._normalize_embeddings(protbert_emb)
        pseq2sites_emb = self._normalize_embeddings(pseq2sites_emb)
        compound_emb = self._normalize_embeddings(compound_emb)
        
        # Debug: Check embeddings after normalization
        print(f"After normalization:")
        print(f"  ProtBERT: type={type(protbert_emb)}, shape={protbert_emb.shape}, dtype={protbert_emb.dtype}")
        print(f"  Pseq2Sites: type={type(pseq2sites_emb)}, shape={pseq2sites_emb.shape}, dtype={pseq2sites_emb.dtype}")
        print(f"  Compound: type={type(compound_emb)}, shape={compound_emb.shape}, dtype={compound_emb.dtype}")
        
        # Ensure embeddings are C-contiguous for FAISS
        print("Ensuring arrays are C-contiguous and properly formatted for FAISS...")
        
        # Create completely new arrays to avoid any memory layout issues
        print("Creating fresh arrays from scratch...")
        protbert_emb_new = np.empty((protbert_emb.shape[0], protbert_emb.shape[1]), dtype=np.float32, order='C')
        protbert_emb_new[:] = protbert_emb
        protbert_emb = protbert_emb_new
        
        pseq2sites_emb_new = np.empty((pseq2sites_emb.shape[0], pseq2sites_emb.shape[1]), dtype=np.float32, order='C')
        pseq2sites_emb_new[:] = pseq2sites_emb
        pseq2sites_emb = pseq2sites_emb_new
        
        compound_emb_new = np.empty((compound_emb.shape[0], compound_emb.shape[1]), dtype=np.float32, order='C')
        compound_emb_new[:] = compound_emb
        compound_emb = compound_emb_new
        
        print(f"Final array properties:")
        print(f"  ProtBERT: C_CONTIGUOUS={protbert_emb.flags.c_contiguous}, OWNDATA={protbert_emb.flags.owndata}")
        print(f"  Pseq2Sites: C_CONTIGUOUS={pseq2sites_emb.flags.c_contiguous}, OWNDATA={pseq2sites_emb.flags.owndata}")
        print(f"  Compound: C_CONTIGUOUS={compound_emb.flags.c_contiguous}, OWNDATA={compound_emb.flags.owndata}")
        
        # Verify these are proper numpy arrays that FAISS can handle
        print("Verifying array compatibility with FAISS...")
        print(f"  ProtBERT strides: {protbert_emb.strides}")
        print(f"  Pseq2Sites strides: {pseq2sites_emb.strides}")
        print(f"  Compound strides: {compound_emb.strides}")
        
        # Critical: Test FAISS compatibility before building indices
        print("Testing FAISS-GPU compatibility...")
        try:
            # Test with a simple L2 index first
            test_index = faiss.IndexFlatL2(protbert_emb.shape[1])
            test_sample = protbert_emb[:1].copy()  # Single sample for testing
            test_index.add(test_sample)
            print("✅ FAISS-GPU compatibility test passed!")
            del test_index
        except Exception as test_error:
            print(f"❌ FAISS-GPU compatibility test failed: {test_error}")
            print("Attempting array recreation from raw data...")
            
            # Last resort: completely recreate arrays from Python lists
            print("Converting to Python lists and back to numpy...")
            protbert_data = protbert_emb.tolist()
            pseq2sites_data = pseq2sites_emb.tolist()
            compound_data = compound_emb.tolist()
            
            # Create new arrays from scratch
            protbert_emb = np.array(protbert_data, dtype=np.float32)
            pseq2sites_emb = np.array(pseq2sites_data, dtype=np.float32)
            compound_emb = np.array(compound_data, dtype=np.float32)
            
            # Ensure C-contiguous
            protbert_emb = np.ascontiguousarray(protbert_emb)
            pseq2sites_emb = np.ascontiguousarray(pseq2sites_emb)
            compound_emb = np.ascontiguousarray(compound_emb)
            
            print("Re-testing FAISS compatibility...")
            test_index = faiss.IndexFlatL2(protbert_emb.shape[1])
            test_sample = protbert_emb[:1].copy()
            test_index.add(test_sample)
            print("✅ Array recreation successful!")
            del test_index
        assert isinstance(protbert_emb, np.ndarray), f"ProtBERT is not numpy array: {type(protbert_emb)}"
        assert isinstance(pseq2sites_emb, np.ndarray), f"Pseq2Sites is not numpy array: {type(pseq2sites_emb)}"
        assert isinstance(compound_emb, np.ndarray), f"Compound is not numpy array: {type(compound_emb)}"
        
        # Build FAISS indices (using Inner Product for normalized vectors = cosine similarity)
        print("Creating FAISS indices...")
        self.protbert_index = faiss.IndexFlatIP(self.protbert_dim)
        self.pseq2sites_index = faiss.IndexFlatIP(self.pseq2sites_dim)
        self.compound_index = faiss.IndexFlatIP(self.compound_dim)
        
        # Add vectors to indices with explicit error handling
        print("Adding ProtBERT embeddings to FAISS index...")
        try:
            self.protbert_index.add(protbert_emb)
            print(f"✅ Successfully added {protbert_emb.shape[0]} ProtBERT embeddings")
        except Exception as e:
            print(f"❌ Error adding ProtBERT embeddings: {e}")
            print(f"ProtBERT array info: {protbert_emb.__array_interface__}")
            raise
            
        print("Adding Pseq2Sites embeddings to FAISS index...")
        try:
            self.pseq2sites_index.add(pseq2sites_emb)
            print(f"✅ Successfully added {pseq2sites_emb.shape[0]} Pseq2Sites embeddings")
        except Exception as e:
            print(f"❌ Error adding Pseq2Sites embeddings: {e}")
            raise
            
        print("Adding compound embeddings to FAISS index...")
        try:
            self.compound_index.add(compound_emb)
            print(f"✅ Successfully added {compound_emb.shape[0]} compound embeddings")
        except Exception as e:
            print(f"❌ Error adding compound embeddings: {e}")
            raise
        
        # Store compound embeddings for later use
        self.compound_embeddings = compound_emb
        
        print(f"Built indices with {len(self.sequences)} entries")
        
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms
        
    def search_similar_proteins(self, 
                               protbert_query: np.ndarray,
                               pseq2sites_query: np.ndarray,
                               k: int = 10,
                               alpha: float = 0.5) -> Tuple[List[int], List[float]]:
        """
        Search for similar proteins using combined similarity.
        
        Args:
            protbert_query: ProtBERT embedding of query protein [1024]
            pseq2sites_query: Pseq2Sites embedding of query protein [256] 
            k: Number of top results to return
            alpha: Weight for Pseq2Sites similarity (1-alpha for ProtBERT)
            
        Returns:
            Tuple of (indices, combined_scores)
        """
        # Normalize query embeddings
        protbert_query = protbert_query / np.linalg.norm(protbert_query)
        pseq2sites_query = pseq2sites_query / np.linalg.norm(pseq2sites_query)
        
        # Search in both indices (get more than k to combine results)
        search_k = min(k * 10, len(self.sequences))
        
        protbert_scores, protbert_indices = self.protbert_index.search(
            protbert_query.reshape(1, -1).astype('float32'), search_k
        )
        pseq2sites_scores, pseq2sites_indices = self.pseq2sites_index.search(
            pseq2sites_query.reshape(1, -1).astype('float32'), search_k
        )
        
        # Combine scores
        combined_scores = {}
        
        # Add ProtBERT scores
        for score, idx in zip(protbert_scores[0], protbert_indices[0]):
            combined_scores[idx] = (1 - alpha) * score
            
        # Add Pseq2Sites scores
        for score, idx in zip(pseq2sites_scores[0], pseq2sites_indices[0]):
            if idx in combined_scores:
                combined_scores[idx] += alpha * score
            else:
                combined_scores[idx] = alpha * score
                
        # Sort by combined score and return top k
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_results = sorted_results[:k]
        
        indices = [idx for idx, _ in top_k_results]
        scores = [score for _, score in top_k_results]
        
        return indices, scores
        
    def get_compounds_for_indices(self, indices: List[int]) -> Tuple[List[str], np.ndarray, List[float]]:
        """Get compound data for given indices."""
        smiles_list = [self.smiles[i] for i in indices]
        compound_emb = self.compound_embeddings[indices]
        ic50_vals = [self.ic50_values[i] for i in indices]
        
        return smiles_list, compound_emb, ic50_vals
        
    def save_database(self, save_path: str):
        """Save the vector database to disk."""
        print(f"Saving vector database to {save_path}")
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save FAISS indices
        faiss.write_index(self.protbert_index, os.path.join(save_path, "protbert_index.faiss"))
        faiss.write_index(self.pseq2sites_index, os.path.join(save_path, "pseq2sites_index.faiss"))
        faiss.write_index(self.compound_index, os.path.join(save_path, "compound_index.faiss"))
        
        # Save metadata and data
        metadata = {
            'sequences': self.sequences,
            'smiles': self.smiles,
            'ic50_values': self.ic50_values.tolist(),
            'compound_embeddings': self.compound_embeddings.tolist(),
            'protbert_dim': self.protbert_dim,
            'pseq2sites_dim': self.pseq2sites_dim,
            'compound_dim': self.compound_dim
        }
        
        with open(os.path.join(save_path, "metadata.pkl"), 'wb') as f:
            pickle.dump(metadata, f)
            
        print("Database saved successfully!")
        
    def load_database(self, load_path: str):
        """Load vector database from disk."""
        print(f"Loading vector database from {load_path}")
        
        # Load FAISS indices
        self.protbert_index = faiss.read_index(os.path.join(load_path, "protbert_index.faiss"))
        self.pseq2sites_index = faiss.read_index(os.path.join(load_path, "pseq2sites_index.faiss"))
        self.compound_index = faiss.read_index(os.path.join(load_path, "compound_index.faiss"))
        
        # Load metadata
        with open(os.path.join(load_path, "metadata.pkl"), 'rb') as f:
            metadata = pickle.load(f)
            
        self.sequences = metadata['sequences']
        self.smiles = metadata['smiles']
        self.ic50_values = np.array(metadata['ic50_values'])
        self.compound_embeddings = np.array(metadata['compound_embeddings'])
        
        print(f"Database loaded with {len(self.sequences)} entries")


def build_database_from_preprocessed_data(data_path: str, 
                                         output_path: str,
                                         db_name: str = "vector_db",
                                         use_train_data: bool = True):
    """Build and save vector database from preprocessed data."""
    
    # Load preprocessed data
    print(f"Loading preprocessed data from {data_path}")
    with open(data_path, 'rb') as f:
        loaded_data = pickle.load(f)
    
    # Handle different data structures
    if 'train_data' in loaded_data and 'test_data' in loaded_data:
        # New format with train/test split
        if use_train_data:
            data = loaded_data['train_data']
            print("Using training data for vector database")
        else:
            data = loaded_data['test_data']
            print("Using test data for vector database")
        metadata = data['metadata']
    else:
        # Old format - direct data structure
        data = loaded_data
        metadata = data['metadata']
        
    # Initialize database
    db = ProteinLigandVectorDB(
        protbert_dim=metadata['protbert_dim'],
        pseq2sites_dim=metadata['pseq2sites_dim'],
        compound_dim=metadata['compound_dim']
    )
    
    # Build indices
    db.build_index(data)
    
    # Save database
    save_path = os.path.join(output_path, db_name)
    db.save_database(save_path)
    
    return db


def main():
    """Build vector databases from preprocessed data."""
    
    # Build training database
    print("Building training vector database...")
    train_db = build_database_from_preprocessed_data(
        data_path="../database/train/preprocessed_data.pkl",
        output_path="../database/",
        db_name="train_vector_db"
    )
    
    # Build test database  
    print("\nBuilding test vector database...")
    test_db = build_database_from_preprocessed_data(
        data_path="../database/test/preprocessed_data.pkl", 
        output_path="../database/",
        db_name="test_vector_db"
    )
    
    print("\nVector databases built successfully!")


if __name__ == "__main__":
    main()
