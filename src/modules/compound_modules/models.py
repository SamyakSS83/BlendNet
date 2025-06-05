from math import sqrt
from rdkit import Chem
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import add_self_loops, softmax

from .base_layers import MLP

num_atom_type = 119

# use atom number (1~118), masked token number: 119; so 120 atom type; 0 is not used
num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
# 0: single, 1: double, 2: triple, 3: aromatic, 4: self-loop, 5: masked
num_bond_direction = 6 

allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)) + ['misc'],
    'possible_chirality_list' : [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'misc'
    ],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list' : [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
        ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list' : [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ], 
    'possible_is_conjugated_list': [False, True],
    
    'possible_bond_dirs' : [ 
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.BEGINDASH, # new add
        Chem.rdchem.BondDir.BEGINWEDGE, # new add
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
        Chem.rdchem.BondDir.EITHERDOUBLE, # new add
    ]
}

class VectorQuantizer(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized
        - embedding_dim (int): the dimensionality of the tensors in the quantized space. Inputs to the modules must be in this format as well.
        - num_embeddings (int): the number of vectors in the quantized space
        - commitment_cost (float): scaler which controls the weighting of the the loss terms
    """
    def __init__(self, embedding_dim, num_embeddings, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim # (256)
        self.num_embeddings = num_embeddings # (512)
        self.commitment_cost = commitment_cost # (2.0)
        
        #initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim) # (512, 256)
        
    def forward(self, x, e): # (x: atom_features (B, N, 9), e: node_embeddings (B, N, 256))
        encoding_indices = self.get_code_indices(x, e) # x: B * H, encoding_indices: B

        quantized = self.quantize(encoding_indices)

        # embedding loss: move the embeddings towards the encoder's output
        q_latent_loss = F.mse_loss(quantized, e.detach())

        # commitment loss
        e_latent_loss = F.mse_loss(e, quantized.detach())

        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = e + (quantized - e).detach().contiguous()
        return quantized, loss
    
    def get_code_indices(self, x, e):
        flat_x = x.view(-1, x.shape[-1])
        
        distances = (torch.sum(e ** 2, dim = -1, keepdim = True) +
                     torch.sum(self.embeddings.weight ** 2, dim = 1) -
                     2 * torch.matmul(e, self.embeddings.weight.t()))

        encoding_indices = torch.argmin(distances, dim = 1)
        return encoding_indices
    
    def quantize(self, encoding_indices):
        """ Returns embedding tensor for a batch of indices."""
        return self.embeddings(encoding_indices)

class AtomEncoder(nn.Module):
    
    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = nn.ModuleList()
        self.atom_feature_dims = [119, 5, 12, 12, 10, 6, 6, 2, 2]
        for i, dim in enumerate(self.atom_feature_dims):
            emb = nn.Embedding(dim + 1, emb_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        assert x.shape[1] == len(self.atom_embedding_list), 'The input feature numbers does not match.'

        for i in range(len(self.atom_embedding_list)):
            x_embedding += self.atom_embedding_list[i](x[:,i].long())

        return x_embedding

class BondEncoder(nn.Module):
    
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()
        self.bond_embedding_list = nn.ModuleList()
        self.bond_feature_dims = [6, 6, 2, 2, 2]
        for i, dim in enumerate(self.bond_feature_dims):
            emb = nn.Embedding(dim + 1, emb_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        assert edge_attr.shape[1] == len(self.bond_embedding_list), 'The input feature numbers does not match.'

        for i in range(len(self.bond_embedding_list)):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i].long())

        return bond_embedding

class GNNLayer(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GNNLayer, self).__init__(aggr=aggr)
        self.emb_dim = emb_dim
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.edge_embedding = BondEncoder(emb_dim)
        self.norm = nn.BatchNorm1d(emb_dim)
        self.act = nn.ReLU()
        
    def forward(self, x, edge_index, edge_attr):
        # Add self-loops
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, 
                                             new_attr_value=torch.zeros(edge_attr.shape[1]).to(edge_attr.device),
                                             num_nodes=x.size(0))
        
        # Create edge embeddings
        edge_embeddings = self.edge_embedding(edge_attr)
        
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)
    
    def message(self, x_j, edge_attr):
        return x_j + edge_attr
    
    def update(self, aggr_out):
        aggr_out = self.norm(self.linear(aggr_out))
        aggr_out = self.act(aggr_out)
        return aggr_out

class GNNDecoder(nn.Module):
    def __init__(self, hidden_dim, out_dim, JK="last", gnn_type="gcn", num_layer=4):
        super(GNNDecoder, self).__init__()
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.JK = JK
        self.out_dim = out_dim
        
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "linear":
                self.gnns.append(nn.Linear(hidden_dim, hidden_dim))
            else:
                self.gnns.append(GNNLayer(hidden_dim))
        
        # Output layer
        self.out_layer = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, h_node, edge_index, edge_attr=None):
        h_list = [h_node]
        
        for layer in range(self.num_layer):
            if isinstance(self.gnns[layer], nn.Linear):
                h = self.gnns[layer](h_list[layer])
            else:
                h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h_list.append(h)
        
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]
        else:
            raise ValueError("Invalid JK option")
        
        return self.out_layer(node_representation)

class GNN_graphpred(nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.
    Args:
        num_layer (int): the number of GNN layers
        hidden_dim (int): the hidden dim
    """

    def __init__(self, hidden_dim, output_dim, JK="last", gnn_type="gin", num_layer=4, graph_pooling="mean"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.JK = JK
        self.output_dim = output_dim
        self.graph_pooling = graph_pooling
        
        # GNN layers
        self.atom_encoder = AtomEncoder(hidden_dim)
        
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GNNLayer(hidden_dim))
        
        # Graph pooling
        if self.graph_pooling == "mean":
            self.pool = global_mean_pool
        else:
            raise ValueError("Invalid graph pooling type")
        
        # Output MLP
        self.output_mlp = MLP([hidden_dim, hidden_dim//2, output_dim], dropout=0.2)
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Initial node embeddings
        h_node = self.atom_encoder(x)
        
        # GNN layers
        h_list = [h_node]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h_list.append(h)
        
        # Node representation
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]
        else:
            raise ValueError("Invalid JK option")
        
        # Graph representation
        graph_representation = self.pool(node_representation, batch)
        
        # Output prediction
        return self.output_mlp(graph_representation), node_representation, h_list, None
