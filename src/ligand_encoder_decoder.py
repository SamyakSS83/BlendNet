"""
Ligand Encoder/Decoder Module for BlendNet

This module provides high-level functions to encode ligands (molecular graphs) into 
single vector representations and decode them back to molecular graphs or SMILES.

The module leverages the existing BlendNet architecture:
- PNA/Net3D GNNs for molecular graph encoding
- VectorQuantizer (VQVAE) for discrete latent representations
- GNNDecoder for molecular graph reconstruction
- RDKit for SMILES generation

Author: BlendNet Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict, Tuple, List
import numpy as np
from rdkit import Chem
from rdkit.Chem import MolToSmiles, MolFromSmiles
import warnings

# Import BlendNet modules - with error handling for compatibility
try:
    import sys
    import os
    
    # Add the parent directory to path to find modules
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from modules.compound_modules.pna import PNA
    from modules.compound_modules.models import VectorQuantizer, GNNDecoder, AtomEncoder, BondEncoder
    
    # Try to import feature generation utilities
    try:
        from feature_generation.compound.Get_Mol_features import get_mol_features, allowable_features
    except ImportError:
        # Fallback implementation if module not found
        print("Warning: Could not import Get_Mol_features, using fallback implementation")
        allowable_features = None
        
        def get_mol_features(mol):
            """Simplified fallback implementation"""
            from rdkit import Chem
            
            atom_features = []
            for atom in mol.GetAtoms():
                features = [
                    atom.GetAtomicNum() - 1,  # Atomic number (0-indexed)
                    int(atom.GetChiralTag()),  # Chirality
                    atom.GetTotalDegree(),  # Degree
                    atom.GetFormalCharge() + 5,  # Formal charge (shifted to be positive)
                    atom.GetTotalNumHs(),  # Number of hydrogens
                    atom.GetNumRadicalElectrons(),  # Radical electrons
                    int(atom.GetHybridization()) - 1,  # Hybridization
                    int(atom.GetIsAromatic()),  # Is aromatic
                    int(atom.IsInRing())  # Is in ring
                ]
                atom_features.append(features)
            
            edges_list = []
            edge_features = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                bond_type = int(bond.GetBondType()) - 1
                edge_feature = [bond_type, 0, 0, 0]  # Simplified edge features
                
                edges_list.append([i, j])
                edge_features.append(edge_feature)
                edges_list.append([j, i])
                edge_features.append(edge_feature)
            
            edge_index = torch.tensor(edges_list, dtype=torch.long).T if edges_list else torch.zeros((2, 0), dtype=torch.long)
            
            return atom_features, edge_index, edge_features, len(edges_list)
    
    # Import PyTorch Geometric utilities with compatibility checks
    try:
        from torch_geometric.data import Data, Batch
        from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
    except ImportError as e:
        print(f"Warning: PyTorch Geometric import failed: {e}")
        # Create fallback classes
        class Data:
            def __init__(self, x=None, edge_index=None, edge_attr=None, batch=None):
                self.x = x
                self.edge_index = edge_index
                self.edge_attr = edge_attr
                self.batch = batch if batch is not None else torch.zeros(x.size(0), dtype=torch.long)
            
            def to(self, device):
                self.x = self.x.to(device) if self.x is not None else None
                self.edge_index = self.edge_index.to(device) if self.edge_index is not None else None
                self.edge_attr = self.edge_attr.to(device) if self.edge_attr is not None else None
                self.batch = self.batch.to(device) if self.batch is not None else None
                return self
        
        class Batch(Data):
            @classmethod
            def from_data_list(cls, data_list):
                if not data_list:
                    return cls()
                
                # Simple batching implementation
                x_list = [data.x for data in data_list if data.x is not None]
                edge_index_list = []
                edge_attr_list = []
                batch_list = []
                
                node_offset = 0
                for i, data in enumerate(data_list):
                    if data.x is not None:
                        batch_list.extend([i] * data.x.size(0))
                        if data.edge_index is not None:
                            edge_index_list.append(data.edge_index + node_offset)
                            if data.edge_attr is not None:
                                edge_attr_list.append(data.edge_attr)
                        node_offset += data.x.size(0)
                
                x = torch.cat(x_list, dim=0) if x_list else None
                edge_index = torch.cat(edge_index_list, dim=1) if edge_index_list else None
                edge_attr = torch.cat(edge_attr_list, dim=0) if edge_attr_list else None
                batch = torch.tensor(batch_list, dtype=torch.long) if batch_list else None
                
                return cls(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        
        def global_mean_pool(x, batch, size=None):
            """Fallback implementation for global mean pooling"""
            if batch is None:
                return x.mean(dim=0, keepdim=True)
            
            batch_size = batch.max().item() + 1
            output = torch.zeros(batch_size, x.size(1), device=x.device, dtype=x.dtype)
            
            for i in range(batch_size):
                mask = batch == i
                if mask.any():
                    output[i] = x[mask].mean(dim=0)
            
            return output
        
        def global_add_pool(x, batch, size=None):
            """Fallback implementation for global sum pooling"""
            if batch is None:
                return x.sum(dim=0, keepdim=True)
            
            batch_size = batch.max().item() + 1
            output = torch.zeros(batch_size, x.size(1), device=x.device, dtype=x.dtype)
            
            for i in range(batch_size):
                mask = batch == i
                if mask.any():
                    output[i] = x[mask].sum(dim=0)
            
            return output
        
        def global_max_pool(x, batch, size=None):
            """Fallback implementation for global max pooling"""
            if batch is None:
                return x.max(dim=0, keepdim=True)[0]
            
            batch_size = batch.max().item() + 1
            output = torch.zeros(batch_size, x.size(1), device=x.device, dtype=x.dtype)
            
            for i in range(batch_size):
                mask = batch == i
                if mask.any():
                    output[i] = x[mask].max(dim=0)[0]
            
            return output

except ImportError as e:
    print(f"Critical import error: {e}")
    print("Please ensure you're in the correct environment and all dependencies are installed.")
    raise


class LigandEncoder(nn.Module):
    """
    Encodes molecular graphs into single vector representations using BlendNet's GNN architecture.
    
    This class wraps the PNA or other GNN models to provide a simple interface for 
    encoding ligands into fixed-size vector representations suitable for downstream tasks.
    """
    
    def __init__(
        self,
        model_type: str = "pna",
        hidden_dim: int = 256,
        target_dim: int = 256,
        readout_aggregators: List[str] = ["mean", "sum", "max"],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the ligand encoder.
        
        Args:
            model_type: Type of GNN to use ("pna" or "net3d")
            hidden_dim: Hidden dimension of the GNN
            target_dim: Output dimension of the encoded vector
            readout_aggregators: List of aggregation functions for graph-level representation
            device: Device to run the model on
        """
        super(LigandEncoder, self).__init__()
        
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.target_dim = target_dim
        self.device = device
        self.readout_aggregators = readout_aggregators
        
        # Initialize the GNN backbone
        if model_type == "pna":
            self.gnn = PNA(
                hidden_dim=hidden_dim,
                target_dim=target_dim,
                aggregators=["mean", "sum", "max", "min"],
                scalers=["identity", "amplification", "attenuation"],
                readout_aggregators=readout_aggregators,
                readout_batchnorm=True,
                propagation_depth=5,
                dropout=0.1
            )
        else:
            raise ValueError(f"Model type {model_type} not supported. Use 'pna'.")
        
        # Move model to device
        self.to(device)
    
    def forward(self, molecular_graphs: Union[Data, Batch]) -> torch.Tensor:
        """
        Encode molecular graphs to vector representations.
        
        Args:
            molecular_graphs: PyTorch Geometric Data or Batch of molecular graphs
            
        Returns:
            Tensor of shape [batch_size, target_dim] containing encoded vectors
        """
        # Ensure graphs are on the correct device
        molecular_graphs = molecular_graphs.to(self.device)
        
        # Get node and graph representations from GNN
        node_representations, graph_representations = self.gnn(molecular_graphs)
        
        return graph_representations
    
    def encode_smiles(self, smiles: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode SMILES strings to vector representations.
        
        Args:
            smiles: SMILES string or list of SMILES strings
            
        Returns:
            Tensor of encoded vectors
        """
        if isinstance(smiles, str):
            smiles = [smiles]
        
        # Convert SMILES to molecular graphs
        graphs = []
        for smi in smiles:
            mol = MolFromSmiles(smi)
            if mol is None:
                warnings.warn(f"Could not parse SMILES: {smi}")
                continue
            
            # Get molecular features
            atom_features, edge_index, edge_features, num_edges = get_mol_features(mol)
            
            # Create PyTorch Geometric Data object
            graph = Data(
                x=torch.tensor(atom_features, dtype=torch.long),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_features, dtype=torch.long)
            )
            graphs.append(graph)
        
        if not graphs:
            raise ValueError("No valid SMILES found")
        
        # Create batch
        batch = Batch.from_data_list(graphs)
        
        # Encode
        with torch.no_grad():
            encoded_vectors = self.forward(batch)
        
        return encoded_vectors


class LigandDecoder(nn.Module):
    """
    Decodes vector representations back to molecular graphs and SMILES.
    
    This class uses VectorQuantizer and GNNDecoder to reconstruct molecular 
    features from vector representations.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        embedding_dim: int = 256,
        num_embeddings: int = 512,
        commitment_cost: float = 2.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the ligand decoder.
        
        Args:
            hidden_dim: Hidden dimension for GNN decoder
            embedding_dim: Dimension of VQ embeddings
            num_embeddings: Number of VQ embeddings
            commitment_cost: VQ commitment cost
            device: Device to run the model on
        """
        super(LigandDecoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Vector quantizer
        self.vq_layer = VectorQuantizer(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            commitment_cost=commitment_cost
        )
        
        # Decoders for different molecular features
        self.atom_decoder = GNNDecoder(
            hidden_dim=hidden_dim,
            out_dim=119,  # Number of atom types
            JK="last",
            gnn_type="gcn"
        )
        
        self.atom_chiral_decoder = GNNDecoder(
            hidden_dim=hidden_dim,
            out_dim=5,   # Number of chirality types
            JK="last",
            gnn_type="gcn"
        )
        
        self.bond_decoder = GNNDecoder(
            hidden_dim=hidden_dim,
            out_dim=5,   # Number of bond types
            JK="last",
            gnn_type="linear"
        )
        
        # Projection layer to convert input vectors to node representations
        self.vector_to_nodes = nn.Linear(embedding_dim, hidden_dim)
        
        # Move model to device
        self.to(device)
    
    def forward(
        self,
        encoded_vectors: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        num_nodes: int
    ) -> Dict[str, torch.Tensor]:
        """
        Decode vectors to molecular features.
        
        Args:
            encoded_vectors: Encoded vector representations
            edge_index: Edge connectivity
            edge_attr: Edge features
            num_nodes: Number of nodes in the graph
            
        Returns:
            Dictionary containing decoded atom and bond features
        """
        # Ensure inputs are on correct device
        encoded_vectors = encoded_vectors.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)
        
        # Expand encoded vectors to node representations
        # For simplicity, we broadcast the graph-level vector to all nodes
        batch_size = encoded_vectors.size(0)
        node_representations = encoded_vectors.unsqueeze(1).expand(-1, num_nodes, -1)
        node_representations = node_representations.contiguous().view(-1, self.embedding_dim)
        
        # Project to hidden dimension
        node_representations = self.vector_to_nodes(node_representations)
        
        # Decode molecular features
        atom_logits = self.atom_decoder(node_representations, edge_index, edge_attr)
        atom_chiral_logits = self.atom_chiral_decoder(node_representations, edge_index, edge_attr)
        
        # For bond predictions, use edge representations
        edge_representations = node_representations[edge_index[0]] + node_representations[edge_index[1]]
        bond_logits = self.bond_decoder(edge_representations, edge_index, edge_attr)
        
        return {
            "atom_logits": atom_logits,
            "atom_chiral_logits": atom_chiral_logits,
            "bond_logits": bond_logits,
            "node_representations": node_representations
        }
    
    def decode_to_smiles(
        self,
        encoded_vectors: torch.Tensor,
        template_graph: Optional[Data] = None,
        max_atoms: int = 50
    ) -> List[str]:
        """
        Decode vectors to SMILES strings.
        
        Args:
            encoded_vectors: Encoded vector representations
            template_graph: Template graph structure (if None, uses a simple chain)
            max_atoms: Maximum number of atoms for generated molecules
            
        Returns:
            List of SMILES strings
        """
        # This is a simplified implementation
        # In practice, you would need a more sophisticated graph generation approach
        
        smiles_list = []
        batch_size = encoded_vectors.size(0)
        
        for i in range(batch_size):
            try:
                # Create a simple template if none provided
                if template_graph is None:
                    # Create a simple chain structure
                    num_atoms = min(max_atoms, 10)  # Simple chain of 10 atoms
                    edge_index = torch.tensor([
                        [j for j in range(num_atoms-1)] + [j+1 for j in range(num_atoms-1)],
                        [j+1 for j in range(num_atoms-1)] + [j for j in range(num_atoms-1)]
                    ], dtype=torch.long)
                    edge_attr = torch.zeros((edge_index.size(1), 4), dtype=torch.long)
                else:
                    num_atoms = template_graph.x.size(0)
                    edge_index = template_graph.edge_index
                    edge_attr = template_graph.edge_attr
                
                # Decode features
                decoded = self.forward(
                    encoded_vectors[i:i+1],
                    edge_index,
                    edge_attr,
                    num_atoms
                )
                
                # Get predicted atom and bond types
                atom_types = torch.argmax(decoded["atom_logits"], dim=-1)
                bond_types = torch.argmax(decoded["bond_logits"], dim=-1)
                
                # Convert to RDKit molecule (simplified)
                mol = Chem.RWMol()
                
                # Add atoms
                for atom_idx in range(num_atoms):
                    atom_type = atom_types[atom_idx].item()
                    if atom_type < 118:  # Valid atomic number
                        mol.AddAtom(Chem.Atom(atom_type + 1))  # RDKit uses 1-based atomic numbers
                    else:
                        mol.AddAtom(Chem.Atom(6))  # Default to carbon
                
                # Add bonds (simplified - only single bonds)
                edges_added = set()
                for edge_idx in range(0, edge_index.size(1), 2):  # Skip reverse edges
                    i_atom = edge_index[0, edge_idx].item()
                    j_atom = edge_index[1, edge_idx].item()
                    
                    if (i_atom, j_atom) not in edges_added and (j_atom, i_atom) not in edges_added:
                        if i_atom < num_atoms and j_atom < num_atoms and i_atom != j_atom:
                            mol.AddBond(i_atom, j_atom, Chem.BondType.SINGLE)
                            edges_added.add((i_atom, j_atom))
                
                # Convert to SMILES
                mol = mol.GetMol()
                Chem.SanitizeMol(mol)
                smiles = MolToSmiles(mol)
                smiles_list.append(smiles)
                
            except Exception as e:
                warnings.warn(f"Could not decode vector {i}: {e}")
                smiles_list.append("C")  # Default to methane
        
        return smiles_list


def ligand_encoder(
    molecular_input: Union[str, List[str], Data, Batch],
    model_type: str = "pna",
    hidden_dim: int = 256,
    target_dim: int = 256,
    device: Optional[str] = None
) -> torch.Tensor:
    """
    High-level function to encode ligands to vector representations.
    
    Args:
        molecular_input: SMILES string(s) or PyTorch Geometric graph(s)
        model_type: Type of GNN model to use
        hidden_dim: Hidden dimension of the model
        target_dim: Output vector dimension
        device: Device to run on (auto-detected if None)
        
    Returns:
        Tensor of encoded vectors
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize encoder
    encoder = LigandEncoder(
        model_type=model_type,
        hidden_dim=hidden_dim,
        target_dim=target_dim,
        device=device
    )
    
    # Set to evaluation mode
    encoder.eval()
    
    with torch.no_grad():
        if isinstance(molecular_input, (str, list)):
            # SMILES input
            encoded_vectors = encoder.encode_smiles(molecular_input)
        elif isinstance(molecular_input, (Data, Batch)):
            # Graph input
            encoded_vectors = encoder(molecular_input)
        else:
            raise ValueError("molecular_input must be SMILES string(s) or PyTorch Geometric graph(s)")
    
    return encoded_vectors


def ligand_decoder(
    encoded_vectors: torch.Tensor,
    output_format: str = "smiles",
    template_graph: Optional[Data] = None,
    hidden_dim: int = 256,
    device: Optional[str] = None
) -> Union[List[str], Dict[str, torch.Tensor]]:
    """
    High-level function to decode vector representations back to molecular representations.
    
    Args:
        encoded_vectors: Tensor of encoded vectors
        output_format: Output format ("smiles" or "features")
        template_graph: Template graph structure for decoding
        hidden_dim: Hidden dimension of the decoder
        device: Device to run on (auto-detected if None)
        
    Returns:
        List of SMILES strings or dictionary of molecular features
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize decoder
    decoder = LigandDecoder(
        hidden_dim=hidden_dim,
        embedding_dim=encoded_vectors.size(-1),
        device=device
    )
    
    # Set to evaluation mode
    decoder.eval()
    
    with torch.no_grad():
        if output_format == "smiles":
            return decoder.decode_to_smiles(encoded_vectors, template_graph)
        elif output_format == "features":
            # Need a template graph for feature decoding
            if template_graph is None:
                raise ValueError("template_graph required for features output format")
            
            return decoder(
                encoded_vectors,
                template_graph.edge_index,
                template_graph.edge_attr,
                template_graph.x.size(0)
            )
        else:
            raise ValueError("output_format must be 'smiles' or 'features'")


def load_pretrained_models(
    encoder_path: Optional[str] = None,
    decoder_path: Optional[str] = None,
    vq_path: Optional[str] = None,
    device: Optional[str] = None
) -> Tuple[LigandEncoder, LigandDecoder]:
    """
    Load pre-trained encoder and decoder models.
    
    Args:
        encoder_path: Path to pre-trained encoder checkpoint
        decoder_path: Path to pre-trained decoder checkpoint  
        vq_path: Path to pre-trained VQ checkpoint
        device: Device to load models on
        
    Returns:
        Tuple of (encoder, decoder) models
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize models
    encoder = LigandEncoder(device=device)
    decoder = LigandDecoder(device=device)
    
    # Load pre-trained weights if provided
    if encoder_path:
        checkpoint = torch.load(encoder_path, map_location=device)
        encoder.load_state_dict(checkpoint)
        print(f"Loaded encoder from {encoder_path}")
    
    if decoder_path:
        checkpoint = torch.load(decoder_path, map_location=device)
        decoder.load_state_dict(checkpoint)
        print(f"Loaded decoder from {decoder_path}")
    
    if vq_path:
        checkpoint = torch.load(vq_path, map_location=device)
        decoder.vq_layer.load_state_dict(checkpoint)
        print(f"Loaded VQ layer from {vq_path}")
    
    return encoder, decoder


# Example usage and testing functions
def test_encoder_decoder():
    """Test the encoder and decoder with example molecules."""
    
    # Test SMILES
    test_smiles = [
        "CCO",  # Ethanol
        "c1ccccc1",  # Benzene
        "CC(C)O",  # Isopropanol
    ]
    
    print("Testing Ligand Encoder/Decoder...")
    
    # Test encoding
    print("\n1. Testing encoding...")
    encoded_vectors = ligand_encoder(test_smiles)
    print(f"Encoded {len(test_smiles)} molecules to vectors of shape: {encoded_vectors.shape}")
    
    # Test decoding
    print("\n2. Testing decoding...")
    decoded_smiles = ligand_decoder(encoded_vectors, output_format="smiles")
    print(f"Decoded back to SMILES:")
    for i, (original, decoded) in enumerate(zip(test_smiles, decoded_smiles)):
        print(f"  {i+1}. Original: {original} -> Decoded: {decoded}")
    
    print("\nTest completed!")


if __name__ == "__main__":
    # Run tests
    test_encoder_decoder()
