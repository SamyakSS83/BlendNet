"""
Ligand Encoder/Decoder Module for BlendNet

This module provides high-level functions to encode ligands (molecular graphs) into 
single vector representations and decode them back to molecular graphs or SMILES.

The module leverages the existing BlendNet architecture:
- PNA/Net3D GNNs for molecular graph encoding
- VectorQuantizer (VQVAE) for discrete latent representations
- GNNDecoder for molecular graph reconstruction
- RDKit for SMILES generation

Author: Samyak
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict, Tuple, List
import numpy as np
import sys
import os
from rdkit import Chem
from rdkit.Chem import MolToSmiles, MolFromSmiles
import warnings

# Add parent directory to path to import feature_generation module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import BlendNet modules
from modules.compound_modules.pna import PNA
from modules.compound_modules.models import VectorQuantizer, GNNDecoder, AtomEncoder, BondEncoder
from feature_generation.compound.Get_Mol_features import get_mol_features, allowable_features
from torch_geometric.data import Data, Batch


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
        template_graphs: List[Data]
    ) -> List[str]:
        """
        Decode vectors to SMILES strings using provided template graphs.

        Args:
            encoded_vectors: Encoded vector representations (B x D)
            template_graphs: List of PyG Data objects with x, edge_index, edge_attr

        Returns:
            List of SMILES strings, one per graph
        """
        smiles_list: List[str] = []
        batch_size = encoded_vectors.size(0)

        if not isinstance(template_graphs, list) or len(template_graphs) != batch_size:
            raise ValueError("template_graphs must be a list of Data objects matching batch size")

        # Decode each vector with its corresponding graph structure
        for i, graph in enumerate(template_graphs):
            decoded: Dict[str, torch.Tensor] = self.forward(
                encoded_vectors[i:i+1].to(self.device),
                graph.edge_index.to(self.device),
                graph.edge_attr.to(self.device),
                graph.x.size(0)
            )
            # Convert logits to discrete features
            atom_ids = torch.argmax(decoded["atom_logits"], dim=-1).view(-1).cpu().numpy()
            chirals = torch.argmax(decoded["atom_chiral_logits"], dim=-1).view(-1).cpu().numpy()
            bond_ids = torch.argmax(decoded["bond_logits"], dim=-1).view(-1).cpu().numpy()

            # Construct molecule
            mol = Chem.RWMol()
            # Add atoms with chirality
            for idx, atom_id in enumerate(atom_ids):
                at_list = allowable_features['possible_atomic_num_list']
                atomic_num = at_list[atom_id] if atom_id < len(at_list)-1 else 6
                atom = Chem.Atom(int(atomic_num))
                # Set chiral tag
                ch_list = allowable_features['possible_chirality_list']
                ch_tag = ch_list[chirals[idx]]
                atom.SetChiralTag(getattr(Chem.rdchem.ChiralType, ch_tag))
                mol.AddAtom(atom)

            # Add bonds (undirected, unique)
            bond_list = allowable_features['possible_bond_type_list']
            num_edges = graph.edge_index.size(1)
            seen: set = set()
            for e_idx in range(0, num_edges):
                u = int(graph.edge_index[0, e_idx])
                v = int(graph.edge_index[1, e_idx])
                if (v, u) in seen or u == v:
                    continue
                seen.add((u, v))
                bond_type_str = bond_list[bond_ids[e_idx]]
                bond_type = getattr(Chem.BondType, bond_type_str) if bond_type_str in vars(Chem.BondType) else Chem.BondType.SINGLE
                mol.AddBond(u, v, bond_type)

            # Sanitize and get SMILES
            mol_obj = mol.GetMol()
            Chem.SanitizeMol(mol_obj)
            smiles = MolToSmiles(mol_obj, isomericSmiles=True)
            smiles_list.append(smiles)

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
    template_graphs: Optional[List[Data]] = None,
    hidden_dim: int = 256,
    device: Optional[str] = None
) -> Union[List[str], Dict[str, torch.Tensor]]:
    """
    High-level function to decode vector representations back to molecular representations.
    
    Args:
        encoded_vectors: Tensor of encoded vectors
        output_format: Output format ("smiles" or "features")
        template_graphs: Template graph structure for decoding
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
            # Decode smiles using list of template_graphs
            return decoder.decode_to_smiles(encoded_vectors, template_graphs)
        elif output_format == "features":
            # Need a template graph for feature decoding
            if template_graphs is None or len(template_graphs) != encoded_vectors.size(0):
                raise ValueError("template_graphs required for features output format")
            
            results = []
            for graph in template_graphs:
                result = decoder(
                    encoded_vectors,
                    graph.edge_index,
                    graph.edge_attr,
                    graph.x.size(0)
                )
                results.append(result)
            return results
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
    # Build template graphs from SMILES
    template_graphs = []
    for smi in test_smiles:
        mol = MolFromSmiles(smi)
        atom_feats, edge_idx, edge_feats, _ = get_mol_features(mol)
        num_atoms = len(atom_feats)
        # Create dummy x for Data
        x = torch.zeros((num_atoms, 1), dtype=torch.float)
        data = Data(x=x, edge_index=edge_idx, edge_attr=edge_feats)
        template_graphs.append(data)
    
    print("Testing Ligand Encoder/Decoder...")
    
    # Test encoding
    print("\n1. Testing encoding...")
    encoded_vectors = ligand_encoder(test_smiles)
    print(f"Encoded {len(test_smiles)} molecules to vectors of shape: {encoded_vectors.shape}")
    
    # Test decoding
    print("\n2. Testing decoding...")
    decoded_smiles = ligand_decoder(
        encoded_vectors,
        output_format="smiles",
        template_graphs=template_graphs
    )
    print(f"Decoded back to SMILES:")
    for i, (original, decoded) in enumerate(zip(test_smiles, decoded_smiles)):
        print(f"  {i+1}. Original: {original} -> Decoded: {decoded}")
    
    print("\nTest completed!")


if __name__ == "__main__":
    # Run tests
    test_encoder_decoder()
