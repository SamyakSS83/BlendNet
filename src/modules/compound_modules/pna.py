import torch
import numpy as np
from functools import partial
from typing import Dict, List, Union, Callable

from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

from .base_layers import MLP
from .models import AtomEncoder, BondEncoder

EPS = 1e-5

def aggregate_mean(x, index, dim_size):
    return global_mean_pool(x, index, dim_size)

def aggregate_sum(x, index, dim_size):
    return global_add_pool(x, index, dim_size)

def aggregate_max(x, index, dim_size):
    return global_max_pool(x, index, dim_size)

def aggregate_min(x, index, dim_size):
    if index.numel() == 0:
        return torch.zeros((dim_size,) + x.shape[1:], dtype=x.dtype, device=x.device)
    # Use scatter_min for min pooling
    min_vals = torch.ones((dim_size,) + x.shape[1:], dtype=x.dtype, device=x.device) * float('inf')
    
    # Handle the case where x might be multi-dimensional
    if len(x.shape) > 1:
        for i in range(x.shape[1]):
            # For each feature dimension
            feature = x[:, i]
            min_vals_feature = min_vals[:, i]
            # Use index_put_ for scatter min operation
            for idx, val in zip(index, feature):
                if val < min_vals_feature[idx]:
                    min_vals_feature[idx] = val
    else:
        for idx, val in zip(index, x):
            if val < min_vals[idx]:
                min_vals[idx] = val
                
    # Replace infinities with zeros
    min_vals[min_vals == float('inf')] = 0
    return min_vals

def aggregate_std(x, index, dim_size):
    mean = aggregate_mean(x, index, dim_size)
    mean_squares = aggregate_mean(x**2, index, dim_size)
    return torch.sqrt(torch.relu(mean_squares - mean**2) + EPS)

def aggregate_var(x, index, dim_size):
    mean = aggregate_mean(x, index, dim_size)
    mean_squares = aggregate_mean(x**2, index, dim_size)
    return torch.relu(mean_squares - mean**2)

def aggregate_moment(x, index, dim_size, n=3):
    # Compute mean per node
    mean = aggregate_mean(x, index, dim_size)
    
    # Expand mean to match x's shape for broadcasting
    expanded_mean = mean[index]
    
    # Compute (x - mean)^n
    diff_power_n = torch.pow(x - expanded_mean, n)
    
    # Compute mean of the powered differences
    moment_n = aggregate_mean(diff_power_n, index, dim_size)
    
    # Take the nth root with sign preservation
    rooted_moment = torch.sign(moment_n) * torch.pow(torch.abs(moment_n) + EPS, 1.0 / n)
    return rooted_moment

# Scaling functions
def scale_identity(x, D=None, avg_d=None):
    return x

def scale_amplification(x, D, avg_d):
    # log(D + 1) / d * h where d is the average of the log(D + 1) in the training set
    return x * (torch.log(D.view(-1, 1) + 1) / avg_d["log"])

def scale_attenuation(x, D, avg_d):
    # (log(D + 1))^-1 / d * X where d is the average of the (log(D + 1))^-1 in the training set
    return x * (avg_d["log"] / torch.log(D.view(-1, 1) + 1))

PNA_AGGREGATORS = {
    "mean": aggregate_mean,
    "sum": aggregate_sum,
    "max": aggregate_max,
    "min": aggregate_min,
    "std": aggregate_std,
    "var": aggregate_var,
    "moment3": partial(aggregate_moment, n=3),
    "moment4": partial(aggregate_moment, n=4),
    "moment5": partial(aggregate_moment, n=5),
}

PNA_SCALERS = {
    "identity": scale_identity,
    "amplification": scale_amplification,
    "attenuation": scale_attenuation,
}

class PNALayer(MessagePassing):
    def __init__(self, in_dim: int, out_dim: int, in_dim_edges: int, 
                 aggregators: List[str], scalers: List[str],
                 activation: Union[Callable, str] = "relu", 
                 last_activation: Union[Callable, str] = "none",
                 dropout: float = 0.0, residual: bool = True, 
                 pairwise_distances: bool = False,
                 mid_batch_norm: bool = False, last_batch_norm: bool = False, 
                 batch_norm_momentum=0.1,
                 avg_d: Dict[str, float] = {"log": 1.0}, 
                 posttrans_layers: int = 2, pretrans_layers: int = 1):
        
        super(PNALayer, self).__init__(aggr=None)  # No default aggregation
        
        self.aggregators_name = aggregators
        self.scalers_name = scalers
        self.edge_features = in_dim_edges > 0
        self.activation = activation
        self.avg_d = avg_d
        self.pairwise_distances = pairwise_distances
        self.residual = residual
        if in_dim != out_dim:
            self.residual = False

        # Pre-transformation MLP for messages
        self.pretrans = MLP(
            layer_sizes=[(2 * in_dim + in_dim_edges + 1) if self.pairwise_distances else (2 * in_dim + in_dim_edges), in_dim, in_dim], 
            dropout=dropout,
            activation=activation,
            last_layer_activation=last_activation,
            batch_norm=mid_batch_norm
        )

        # Post-transformation MLP
        self.posttrans = MLP(
            layer_sizes=[(len(aggregators) * len(scalers) + 1) * in_dim, out_dim, out_dim],
            dropout=dropout,
            activation=activation,
            last_layer_activation=last_activation,
            batch_norm=last_batch_norm
        )

    def forward(self, x, edge_index, edge_attr, pos=None):
        # x has shape [N, in_dim]
        # edge_index has shape [2, E]
        # edge_attr has shape [E, in_dim_edges]
        
        # Save input for residual connection
        x_in = x
        
        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, pos=pos)
        
        # Concatenate with input features
        out = torch.cat([x, out], dim=-1)
        
        # Apply post-transformation
        out = self.posttrans(out)
        
        # Add residual connection if applicable
        if self.residual:
            out = out + x_in
            
        return out

    def message(self, x_i, x_j, edge_attr, pos_i=None, pos_j=None):
        # Construct message inputs
        if self.edge_features and self.pairwise_distances and pos_i is not None:
            squared_distance = torch.sum((pos_i - pos_j) ** 2, dim=-1, keepdim=True)
            z = torch.cat([x_i, x_j, edge_attr, squared_distance], dim=-1)
        elif not self.edge_features and self.pairwise_distances and pos_i is not None:
            squared_distance = torch.sum((pos_i - pos_j) ** 2, dim=-1, keepdim=True)
            z = torch.cat([x_i, x_j, squared_distance], dim=-1)
        elif self.edge_features and not self.pairwise_distances:
            z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            z = torch.cat([x_i, x_j], dim=-1)
            
        # Apply pretransformation to messages
        return self.pretrans(z)
    
    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        # inputs has shape [E, in_dim]
        # index has shape [E]
        
        # Get degree information for scaling
        if dim_size is None:
            dim_size = int(index.max()) + 1
        
        D = degree(index, dim_size, dtype=inputs.dtype).to(inputs.device)
        
        # Apply different aggregators
        outs = []
        
        for aggr_name in self.aggregators_name:
            aggr_func = PNA_AGGREGATORS[aggr_name]
            
            if aggr_name in ["moment3", "moment4", "moment5"]:
                if aggr_name == "moment3":
                    out = aggr_func(inputs, index, dim_size, n=3)
                elif aggr_name == "moment4":
                    out = aggr_func(inputs, index, dim_size, n=4)
                else:
                    out = aggr_func(inputs, index, dim_size, n=5)
            else:
                out = aggr_func(inputs, index, dim_size)
            
            # Apply different scalers
            for scale_name in self.scalers_name:
                scale_func = PNA_SCALERS[scale_name]
                if scale_name == "identity":
                    outs.append(scale_func(out))
                else:
                    outs.append(scale_func(out, D, self.avg_d))
        
        # Concatenate and return
        return torch.cat(outs, dim=-1) if len(outs) > 0 else torch.zeros_like(inputs[0:dim_size])

    def update(self, aggr_out, x):
        # No additional update needed here
        return aggr_out


class PNAGNN(torch.nn.Module):
    def __init__(self, hidden_dim, aggregators: List[str], scalers: List[str],
                 residual: bool = True, pairwise_distances: bool = False, 
                 activation: Union[Callable, str] = "relu",
                 last_activation: Union[Callable, str] = "none", 
                 mid_batch_norm: bool = False,
                 last_batch_norm: bool = False, batch_norm_momentum=0.1, 
                 propagation_depth: int = 5,
                 dropout: float = 0.0, posttrans_layers: int = 1, 
                 pretrans_layers: int = 1, **kwargs):
        
        super(PNAGNN, self).__init__()

        # Encoders for atom and bond features
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoder = BondEncoder(hidden_dim)
        
        # Message passing layers
        self.mp_layers = torch.nn.ModuleList()
        for _ in range(propagation_depth):
            self.mp_layers.append(
                PNALayer(
                    in_dim=hidden_dim, 
                    out_dim=hidden_dim, 
                    in_dim_edges=hidden_dim,
                    aggregators=aggregators,
                    scalers=scalers,
                    pairwise_distances=pairwise_distances,
                    residual=residual,
                    dropout=dropout,
                    activation=activation,
                    last_activation=last_activation,
                    mid_batch_norm=mid_batch_norm,
                    last_batch_norm=last_batch_norm,
                    avg_d={"log": 1.0},
                    posttrans_layers=posttrans_layers,
                    pretrans_layers=pretrans_layers
                )
            )

    def forward(self, data):
        # Get atom and bond features
        x = self.atom_encoder(data.x)
        edge_index = data.edge_index
        edge_attr = self.bond_encoder(data.edge_attr)
        
        # Apply message passing layers
        for mp_layer in self.mp_layers:
            x = mp_layer(x, edge_index, edge_attr)
        
        # Store result in data object similar to DGL implementation
        data.x = x
        return data


class PNA(torch.nn.Module):
    """
    Message Passing Neural Network that does not use 3D information
    """
    def __init__(self,
                 hidden_dim,
                 target_dim,
                 aggregators: List[str],
                 scalers: List[str],
                 readout_aggregators: List[str],
                 readout_batchnorm: bool = True,
                 readout_hidden_dim=None,
                 readout_layers: int = 2,
                 residual: bool = True,
                 pairwise_distances: bool = False,
                 activation: Union[Callable, str] = "relu",
                 last_activation: Union[Callable, str] = "none",
                 mid_batch_norm: bool = False,
                 last_batch_norm: bool = False,
                 propagation_depth: int = 5,
                 dropout: float = 0.0,
                 posttrans_layers: int = 1,
                 pretrans_layers: int = 1,
                 batch_norm_momentum=0.1,
                 **kwargs):
        super(PNA, self).__init__()

        self.node_gnn = PNAGNN(
            hidden_dim=hidden_dim, 
            aggregators=aggregators,
            scalers=scalers, 
            residual=residual, 
            pairwise_distances=pairwise_distances,
            activation=activation, 
            last_activation=last_activation, 
            mid_batch_norm=mid_batch_norm,
            last_batch_norm=last_batch_norm, 
            propagation_depth=propagation_depth, 
            dropout=dropout,
            posttrans_layers=posttrans_layers, 
            pretrans_layers=pretrans_layers,
            batch_norm_momentum=batch_norm_momentum
        )
        
        if readout_hidden_dim is None:
            readout_hidden_dim = hidden_dim
            
        self.readout_aggregators = readout_aggregators
        
        self.output = MLP(
            layer_sizes=[hidden_dim * len(self.readout_aggregators), readout_hidden_dim, target_dim],
            dropout=dropout,
            activation=activation,
            batch_norm=readout_batchnorm
        )

    def forward(self, data):
        # Process graph with node GNN
        processed_data = self.node_gnn(data)
        
        # Apply different readout aggregators
        readouts = []
        for aggr in self.readout_aggregators:
            if aggr == "mean":
                readouts.append(global_mean_pool(processed_data.x, processed_data.batch))
            elif aggr == "sum":
                readouts.append(global_add_pool(processed_data.x, processed_data.batch))
            elif aggr == "max":
                readouts.append(global_max_pool(processed_data.x, processed_data.batch))
            # Add more aggregators as needed
        
        # Concatenate readouts
        readout = torch.cat(readouts, dim=-1)
        
        # Apply output MLP
        out = self.output(readout)
        
        return processed_data.x, out
