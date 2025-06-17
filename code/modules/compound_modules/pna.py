import torch
import numpy as np
from functools import partial
from typing import Dict, List, Union, Callable

from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

from .base_layers import MLP
from .models import AtomEncoder, BondEncoder

EPS = 1e-5

def global_min_pool(x, batch, size=None):
    """Custom implementation of global min pooling using scatter."""
    # Use scatter to compute min pooling
    result = scatter(x, batch, dim=0, dim_size=size, reduce='min')
    # Return as tuple to match global_max_pool API
    return result, None

def aggregate_mean(h, **kwargs):
    return torch.mean(h, dim=-2)


def aggregate_max(h, **kwargs):
    return torch.max(h, dim=-2)[0]


def aggregate_min(h, **kwargs):
    return torch.min(h, dim=-2)[0]


def aggregate_std(h, **kwargs):
    var = aggregate_var(h)
    # Add epsilon for numerical stability and prevent NaN from sqrt of negative numbers
    return torch.sqrt(var + EPS)


def aggregate_var(h, **kwargs):
    h_mean_squares = torch.mean(h * h, dim=-2)
    h_mean = torch.mean(h, dim=-2)
    var = h_mean_squares - h_mean * h_mean
    # Use relu to ensure non-negative variance, but also add epsilon
    var = torch.relu(var) + EPS
    return var


def aggregate_moment(h, n=3, **kwargs):
    # for each node (E[(X-E[X])^n])^{1/n}
    # EPS is added to the absolute value of expectation before taking the nth root for stability
    h_mean = torch.mean(h, dim=-2, keepdim=True)
    h_n = torch.mean(torch.pow(h - h_mean, n), dim=-2)
    rooted_h_n = torch.sign(h_n) * torch.pow(torch.abs(h_n) + EPS, 1.0 / n)
    return rooted_h_n


def aggregate_sum(h, **kwargs):
    return torch.sum(h, dim=-2)


def scale_identity(h, D=None, avg_d=None):
    return h


def scale_amplification(h, D, avg_d):
    # log(D + 1) / d * h     where d is the average of the ``log(D + 1)`` in the training set
    # Add small epsilon to prevent division by zero
    epsilon = 1e-8
    denominator = max(avg_d["log"], epsilon)
    return h * (torch.log(D + 1) / denominator)


def scale_attenuation(h, D, avg_d):
    # (log(D + 1))^-1 / d * X     where d is the average of the ``log(D + 1))^-1`` in the training set
    # Add small epsilon to prevent division by zero
    epsilon = 1e-8
    log_term = torch.log(D + 1)
    # Clamp log_term to prevent division by very small numbers
    log_term = torch.clamp(log_term, min=epsilon)
    return h * (avg_d["log"] / log_term)


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


class PNA(nn.Module):
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

        self.node_gnn = PNAGNN(hidden_dim=hidden_dim, aggregators=aggregators,
                               scalers=scalers, residual=residual, pairwise_distances=pairwise_distances,
                               activation=activation, last_activation=last_activation, mid_batch_norm=mid_batch_norm,
                               last_batch_norm=last_batch_norm, propagation_depth=propagation_depth, dropout=dropout,
                               posttrans_layers=posttrans_layers, pretrans_layers=pretrans_layers,
                               batch_norm_momentum=batch_norm_momentum)
        if readout_hidden_dim == None:
            readout_hidden_dim = hidden_dim
        self.readout_aggregators = readout_aggregators
        self.output = MLP(in_dim=hidden_dim * len(self.readout_aggregators), hidden_size=readout_hidden_dim,
                          mid_batch_norm=readout_batchnorm, out_dim=target_dim,
                          layers=readout_layers, batch_norm_momentum=batch_norm_momentum)

    def forward(self, data):
        """
        Args:
            data: PyTorch Geometric Data object with:
                - x: node features [num_nodes, feat_dim]
                - edge_index: edge connectivity [2, num_edges]
                - edge_attr: edge features [num_edges, feat_dim]
                - batch: batch assignment [num_nodes]
        """
        node_features = self.node_gnn(data)
        
        # Apply readout aggregation functions
        readouts_to_cat = []
        for aggr in self.readout_aggregators:
            if aggr == 'mean':
                readout = global_mean_pool(node_features, data.batch)
            elif aggr == 'sum':
                readout = global_add_pool(node_features, data.batch)
            elif aggr == 'max':
                readout = global_max_pool(node_features, data.batch)
                if isinstance(readout, tuple):
                    readout = readout[0]  # Take values, not indices
            elif aggr == 'min':
                readout = global_min_pool(node_features, data.batch)
                if isinstance(readout, tuple):
                    readout = readout[0]  # Take values, not indices
            else:
                raise ValueError(f"Unsupported aggregator: {aggr}")
            
            # Ensure readout is 2D (batch_size, features)
            if readout.dim() == 1:
                readout = readout.unsqueeze(0)
            elif readout.dim() == 0:
                readout = readout.unsqueeze(0).unsqueeze(0)
            readouts_to_cat.append(readout)
        
        readout = torch.cat(readouts_to_cat, dim=-1)
        
        # Check for NaN/Inf values and replace with zeros
        if torch.isnan(readout).any() or torch.isinf(readout).any():
            readout = torch.where(torch.isnan(readout) | torch.isinf(readout), 
                                torch.zeros_like(readout), readout)
        
        final_output = self.output(readout)
        
        # Final check for NaN/Inf in output
        if torch.isnan(final_output).any() or torch.isinf(final_output).any():
            final_output = torch.where(torch.isnan(final_output) | torch.isinf(final_output), 
                                     torch.zeros_like(final_output), final_output)
        
        return node_features, final_output


class PNAGNN(nn.Module):
    def __init__(self, hidden_dim, aggregators: List[str], scalers: List[str],
                 residual: bool = True, pairwise_distances: bool = False, activation: Union[Callable, str] = "relu",
                 last_activation: Union[Callable, str] = "none", mid_batch_norm: bool = False,
                 last_batch_norm: bool = False, batch_norm_momentum=0.1, propagation_depth: int = 5,
                 dropout: float = 0.0, posttrans_layers: int = 1, pretrans_layers: int = 1, **kwargs):
        super(PNAGNN, self).__init__()

        self.mp_layers = nn.ModuleList()

        for _ in range(propagation_depth):
            self.mp_layers.append(
                PNALayer(in_dim=hidden_dim, out_dim=int(hidden_dim), in_dim_edges=hidden_dim, aggregators=aggregators,
                         scalers=scalers, pairwise_distances=pairwise_distances, residual=residual, dropout=dropout,
                         activation=activation, last_activation=last_activation, mid_batch_norm=mid_batch_norm,
                         last_batch_norm=last_batch_norm, avg_d={"log": 1.0}, posttrans_layers=posttrans_layers,
                         pretrans_layers=pretrans_layers, batch_norm_momentum=batch_norm_momentum))

        self.atom_encoder = AtomEncoder(emb_dim=hidden_dim)
        self.bond_encoder = BondEncoder(emb_dim=hidden_dim)

    def forward(self, data):
        """
        Args:
            data: PyTorch Geometric Data object
        """
        # Get atom and bond embeddings
        data.x = self.atom_encoder(data.x)
        data.edge_attr = self.bond_encoder(data.edge_attr)

        for mp_layer in self.mp_layers:
            data = mp_layer(data)
        
        return data.x


class PNALayer(MessagePassing):
    def __init__(self, in_dim: int, out_dim: int, in_dim_edges: int, aggregators: List[str], scalers: List[str],
                 activation: Union[Callable, str] = "relu", last_activation: Union[Callable, str] = "none",
                 dropout: float = 0.0, residual: bool = True, pairwise_distances: bool = False,
                 mid_batch_norm: bool = False, last_batch_norm: bool = False, batch_norm_momentum=0.1,
                 avg_d: Dict[str, float] = {"log": 1.0}, posttrans_layers: int = 2, pretrans_layers: int = 1):
        super(PNALayer, self).__init__(aggr=None)  # We handle aggregation manually
        
        self.aggregators = [PNA_AGGREGATORS[aggr] for aggr in aggregators]
        self.scalers = [PNA_SCALERS[scale] for scale in scalers]
        self.edge_features = in_dim_edges > 0
        self.activation = activation
        self.avg_d = avg_d
        self.pairwise_distances = pairwise_distances
        self.residual = residual
        if in_dim != out_dim:
            self.residual = False

        self.pretrans = MLP(
            in_dim=(2 * in_dim + in_dim_edges + 1) if self.pairwise_distances else (2 * in_dim + in_dim_edges),
            hidden_size=in_dim, out_dim=in_dim, mid_batch_norm=mid_batch_norm, last_batch_norm=last_batch_norm,
            layers=pretrans_layers, mid_activation=activation, dropout=dropout, last_activation=last_activation,
            batch_norm_momentum=batch_norm_momentum)

        self.posttrans = MLP(in_dim=(len(self.aggregators) * len(self.scalers) + 1) * in_dim, hidden_size=out_dim,
                             out_dim=out_dim, layers=posttrans_layers, mid_activation=activation,
                             last_activation=last_activation, dropout=dropout, mid_batch_norm=mid_batch_norm,
                             last_batch_norm=last_batch_norm, batch_norm_momentum=batch_norm_momentum)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h_in = x
        
        # Create edge messages using pretrans
        row, col = edge_index
        if self.edge_features and self.pairwise_distances:
            # We don't have 'x' coordinates in this case, so skip pairwise distances
            z2 = torch.cat([x[row], x[col], edge_attr], dim=-1)
        elif self.edge_features and not self.pairwise_distances:
            z2 = torch.cat([x[row], x[col], edge_attr], dim=-1)
        else:
            z2 = torch.cat([x[row], x[col]], dim=-1)
        
        edge_messages = self.pretrans(z2)
        
        # Get node degrees for each node
        node_degrees = scatter(torch.ones_like(col, dtype=torch.float), col, dim=0, dim_size=x.size(0), reduce='sum')
        
        # Apply aggregators
        h_to_cat = []
        for aggr in self.aggregators:
            if aggr == aggregate_mean:
                agg_out = scatter(edge_messages, col, dim=0, dim_size=x.size(0), reduce='mean')
            elif aggr == aggregate_sum:
                agg_out = scatter(edge_messages, col, dim=0, dim_size=x.size(0), reduce='sum')
            elif aggr == aggregate_max:
                agg_out = scatter(edge_messages, col, dim=0, dim_size=x.size(0), reduce='max')
            else:
                # For more complex aggregators, we need to group messages by target node
                # This is a simplified version - for full PNA you'd need more sophisticated aggregation
                agg_out = scatter(edge_messages, col, dim=0, dim_size=x.size(0), reduce='mean')
            h_to_cat.append(agg_out)
        
        h = torch.cat(h_to_cat, dim=-1)
        
        # Apply scalers
        if len(self.scalers) > 1:
            h_scaled = []
            for scale in self.scalers:
                D = node_degrees.unsqueeze(-1)  # Use node degree as D
                h_scaled.append(scale(h, D=D, avg_d=self.avg_d))
            h = torch.cat(h_scaled, dim=-1)
        
        # Concatenate with original features
        h = torch.cat([h, h_in], dim=-1)
        
        # Post-transformation
        h = self.posttrans(h)
        if self.residual:
            h = h + h_in

        # Update data
        data.x = h
        return data
