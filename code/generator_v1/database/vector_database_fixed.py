"""
FAISS vector database for efficient similarity search - Fixed version with CPU fallback.
"""
import os
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
import faiss


class ProteinLigandVectorDB:
    """FAISS-based vector database for protein-ligand pairs with CPU/GPU auto-detection."""
    
    def __init__(self, 
                 protbert_dim: int = 1024,
                 pseq2sites_dim: int = 256,
                 compound_dim: int = 768):
        """Initialize vector database."""
        self.protbert_dim = protbert_dim
        self.pseq2sites_dim = pseq2sites_dim
        self.compound_dim = compound_dim
        
        # FAISS indices
        self.protbert_index = None
        self.pseq2sites_index = None
        self.compound_index = None
        
        # Data storage
        self.sequences = []
        self.smiles = []
        self.ic50_values = []
        self.compound_embeddings = None
        
        # Backend info
        self.use_gpu = False
        
    def _detect_faiss_backend(self):
        """Detect best available FAISS backend."""
        try:
            if hasattr(faiss, 'StandardGpuResources'):
                # Test GPU
                res = faiss.StandardGpuResources()
                test_index = faiss.IndexFlatL2(10)
                test_index = faiss.index_cpu_to_gpu(res, 0, test_index)
                test_data = np.random.randn(1, 10).astype('float32')
                test_index.add(test_data)
                print("✅ FAISS-GPU detected and working")
                self.use_gpu = True
                del test_index, res
            else:
                print("⚠️  FAISS-GPU not available")
                self.use_gpu = False
        except Exception as e:
            print(f"⚠️  FAISS-GPU failed: {e}, using CPU")
            self.use_gpu = False
        
        # Test CPU fallback
        try:
            test_index = faiss.IndexFlatL2(10)
            test_data = np.random.randn(1, 10).astype('float32')
            test_index.add(test_data)
            print("✅ FAISS-CPU working")
            del test_index
        except Exception as e:
            raise RuntimeError(f"FAISS-CPU also failed: {e}")
        
        backend = "GPU" if self.use_gpu else "CPU"
        print(f"Using FAISS backend: {backend}")
        
    def build_index(self, data: Dict):
        """Build FAISS indices from preprocessed data."""
        print("Building FAISS indices...")
        
        # Detect backend
        self._detect_faiss_backend()
        
        # Extract data
        self.sequences = data['sequences']
        self.smiles = data['smiles']
        self.ic50_values = np.array(data['ic50_values'])
        
        # Get embeddings (should be already clean from fix_preprocessing.py)
        protbert_emb = data['protbert_embeddings'].astype('float32')
        pseq2sites_emb = data['pseq2sites_embeddings'].astype('float32')
        compound_emb = data['compound_embeddings'].astype('float32')
        
        print(f"Embedding shapes:")
        print(f"  ProtBERT: {protbert_emb.shape}")
        print(f"  Pseq2Sites: {pseq2sites_emb.shape}")
        print(f"  Compound: {compound_emb.shape}")
        
        # Normalize embeddings for cosine similarity
        print("Normalizing embeddings...")
        protbert_emb = self._normalize_embeddings(protbert_emb)
        pseq2sites_emb = self._normalize_embeddings(pseq2sites_emb)
        compound_emb = self._normalize_embeddings(compound_emb)
        self.compound_embeddings = compound_emb
        
        # Create indices
        print(f"Creating FAISS indices...")
        if self.use_gpu:
            # GPU indices
            res = faiss.StandardGpuResources()
            cpu_protbert = faiss.IndexFlatIP(self.protbert_dim)
            cpu_pseq2sites = faiss.IndexFlatIP(self.pseq2sites_dim)
            cpu_compound = faiss.IndexFlatIP(self.compound_dim)
            
            self.protbert_index = faiss.index_cpu_to_gpu(res, 0, cpu_protbert)
            self.pseq2sites_index = faiss.index_cpu_to_gpu(res, 0, cpu_pseq2sites)
            self.compound_index = faiss.index_cpu_to_gpu(res, 0, cpu_compound)
        else:
            # CPU indices
            self.protbert_index = faiss.IndexFlatIP(self.protbert_dim)
            self.pseq2sites_index = faiss.IndexFlatIP(self.pseq2sites_dim)
            self.compound_index = faiss.IndexFlatIP(self.compound_dim)
        
        # Add embeddings
        print("Adding embeddings to indices...")
        self.protbert_index.add(protbert_emb)
        print(f"✅ Added {protbert_emb.shape[0]} ProtBERT embeddings")
        
        self.pseq2sites_index.add(pseq2sites_emb)
        print(f"✅ Added {pseq2sites_emb.shape[0]} Pseq2Sites embeddings")
        
        self.compound_index.add(compound_emb)
        print(f"✅ Added {compound_emb.shape[0]} compound embeddings")
        
        print(f"Built indices with {len(self.sequences)} entries using FAISS-{'GPU' if self.use_gpu else 'CPU'}")
        
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
        """Search for similar proteins using combined similarity."""
        # Normalize queries
        protbert_query = self._normalize_embeddings(protbert_query.reshape(1, -1))
        pseq2sites_query = self._normalize_embeddings(pseq2sites_query.reshape(1, -1))
        
        # Search each index
        _, protbert_indices = self.protbert_index.search(protbert_query, k * 2)
        protbert_scores, _ = self.protbert_index.search(protbert_query, k * 2)
        
        _, pseq2sites_indices = self.pseq2sites_index.search(pseq2sites_query, k * 2)
        pseq2sites_scores, _ = self.pseq2sites_index.search(pseq2sites_query, k * 2)
        
        # Combine scores (simple approach)
        combined_scores = {}
        for i, idx in enumerate(protbert_indices[0]):
            combined_scores[idx] = (1 - alpha) * protbert_scores[0][i]
            
        for i, idx in enumerate(pseq2sites_indices[0]):
            if idx in combined_scores:
                combined_scores[idx] += alpha * pseq2sites_scores[0][i]
            else:
                combined_scores[idx] = alpha * pseq2sites_scores[0][i]
        
        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        indices = [idx for idx, _ in sorted_results[:k]]
        scores = [score for _, score in sorted_results[:k]]
        
        return indices, scores
        
    def search_similar_compounds(self, 
                                compound_query: np.ndarray,
                                k: int = 10) -> Tuple[List[int], List[float]]:
        """Search for similar compounds."""
        compound_query = self._normalize_embeddings(compound_query.reshape(1, -1))
        scores, indices = self.compound_index.search(compound_query, k)
        return indices[0].tolist(), scores[0].tolist()
        
    def get_retrieval_context(self, 
                             protbert_query: np.ndarray,
                             pseq2sites_query: np.ndarray,
                             k: int = 10) -> Dict:
        """Get retrieval context for a protein query."""
        indices, scores = self.search_similar_proteins(protbert_query, pseq2sites_query, k)
        
        return {
            'indices': indices,
            'scores': scores,
            'sequences': [self.sequences[i] for i in indices],
            'smiles': [self.smiles[i] for i in indices],
            'ic50_values': [self.ic50_values[i] for i in indices],
            'compound_embeddings': self.compound_embeddings[indices]
        }
        
    def save_database(self, path: str):
        """Save database to disk."""
        os.makedirs(path, exist_ok=True)
        
        # Save indices
        if self.use_gpu:
            # Move to CPU for saving
            faiss.write_index(faiss.index_gpu_to_cpu(self.protbert_index), 
                            os.path.join(path, 'protbert_index.faiss'))
            faiss.write_index(faiss.index_gpu_to_cpu(self.pseq2sites_index), 
                            os.path.join(path, 'pseq2sites_index.faiss'))
            faiss.write_index(faiss.index_gpu_to_cpu(self.compound_index), 
                            os.path.join(path, 'compound_index.faiss'))
        else:
            faiss.write_index(self.protbert_index, os.path.join(path, 'protbert_index.faiss'))
            faiss.write_index(self.pseq2sites_index, os.path.join(path, 'pseq2sites_index.faiss'))
            faiss.write_index(self.compound_index, os.path.join(path, 'compound_index.faiss'))
        
        # Save metadata
        metadata = {
            'sequences': self.sequences,
            'smiles': self.smiles,
            'ic50_values': self.ic50_values,
            'compound_embeddings': self.compound_embeddings,
            'protbert_dim': self.protbert_dim,
            'pseq2sites_dim': self.pseq2sites_dim,
            'compound_dim': self.compound_dim,
            'use_gpu': self.use_gpu
        }
        
        with open(os.path.join(path, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
            
        print(f"Database saved to {path}")
        
    def load_database(self, path: str):
        """Load database from disk."""
        # Load metadata
        with open(os.path.join(path, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
            
        self.sequences = metadata['sequences']
        self.smiles = metadata['smiles']
        self.ic50_values = metadata['ic50_values']
        self.compound_embeddings = metadata['compound_embeddings']
        self.protbert_dim = metadata['protbert_dim']
        self.pseq2sites_dim = metadata['pseq2sites_dim']
        self.compound_dim = metadata['compound_dim']
        
        # Detect backend
        self._detect_faiss_backend()
        
        # Load indices
        protbert_cpu = faiss.read_index(os.path.join(path, 'protbert_index.faiss'))
        pseq2sites_cpu = faiss.read_index(os.path.join(path, 'pseq2sites_index.faiss'))
        compound_cpu = faiss.read_index(os.path.join(path, 'compound_index.faiss'))
        
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.protbert_index = faiss.index_cpu_to_gpu(res, 0, protbert_cpu)
            self.pseq2sites_index = faiss.index_cpu_to_gpu(res, 0, pseq2sites_cpu)
            self.compound_index = faiss.index_cpu_to_gpu(res, 0, compound_cpu)
        else:
            self.protbert_index = protbert_cpu
            self.pseq2sites_index = pseq2sites_cpu
            self.compound_index = compound_cpu
            
        print(f"Database loaded from {path} using FAISS-{'GPU' if self.use_gpu else 'CPU'}")


def build_database_from_preprocessed_data(data_path: str, 
                                        output_path: str, 
                                        db_name: str = "vector_database",
                                        use_train_data: bool = True) -> ProteinLigandVectorDB:
    """Build vector database from preprocessed data."""
    print(f"Loading preprocessed data from {data_path}")
    
    # Load preprocessed data
    with open(data_path, 'rb') as f:
        loaded_data = pickle.load(f)
    
    # Handle both old and new format
    if 'train_data' in loaded_data and 'test_data' in loaded_data:
        if use_train_data:
            data = loaded_data['train_data']
            print("Using training data for vector database")
        else:
            data = loaded_data['test_data']
            print("Using test data for vector database")
        metadata = data['metadata']
    else:
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


if __name__ == "__main__":
    # Test with fixed data
    print("Testing fixed vector database...")
    
    try:
        db = build_database_from_preprocessed_data(
            data_path="./preprocessed_data/preprocessed_data_fixed.pkl",
            output_path="./preprocessed_data",
            db_name="vector_database_fixed"
        )
        print("✅ Fixed vector database created successfully!")
        
        # Test search
        import numpy as np
        test_query = np.random.randn(1024).astype('float32')
        test_query2 = np.random.randn(256).astype('float32')
        
        results = db.search_similar_proteins(test_query, test_query2, k=3)
        print(f"Search test successful: found {len(results[0])} results")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
