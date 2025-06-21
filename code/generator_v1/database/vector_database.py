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
        print("Building FAISS indices...")
        
        # Extract data
        self.sequences = data['sequences']
        self.smiles = data['smiles']
        self.ic50_values = np.array(data['ic50_values'])
        
        protbert_emb = np.array(data['protbert_embeddings']).astype('float32')
        pseq2sites_emb = np.array(data['pseq2sites_embeddings']).astype('float32')
        self.compound_embeddings = np.array(data['compound_embeddings']).astype('float32')
        
        # Normalize embeddings for cosine similarity
        protbert_emb = self._normalize_embeddings(protbert_emb)
        pseq2sites_emb = self._normalize_embeddings(pseq2sites_emb)
        self.compound_embeddings = self._normalize_embeddings(self.compound_embeddings)
        
        # Build FAISS indices (using Inner Product for normalized vectors = cosine similarity)
        self.protbert_index = faiss.IndexFlatIP(self.protbert_dim)
        self.pseq2sites_index = faiss.IndexFlatIP(self.pseq2sites_dim)
        self.compound_index = faiss.IndexFlatIP(self.compound_dim)
        
        # Add vectors to indices
        self.protbert_index.add(protbert_emb)
        self.pseq2sites_index.add(pseq2sites_emb)
        self.compound_index.add(self.compound_embeddings)
        
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
