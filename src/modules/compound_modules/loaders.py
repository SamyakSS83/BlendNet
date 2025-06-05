import os
import copy
import torch
import pickle
import random
import numpy as np
import pandas as pd
from typing import List, Tuple

from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

class MoleculeGraphDataset(Dataset):
    
    def __init__(self, processed_file, device='cuda:0'):
        self.processed_file = processed_file
        self.device = device

        print('Load data ...')
        if not os.path.exists(f"{self.processed_file}"):
            print(f"Check input file: {self.processed_file}")
        data_dict = torch.load(f"{self.processed_file}")
        
        self.features_tensor = data_dict['atom_features']
        self.e_features_tensor = data_dict['edge_features']
        self.edge_indices = data_dict['edge_indices']
        
        self.meta_dict = {k: data_dict[k] for k in ('mol_ids', 'edge_slices', 'atom_slices', 'n_atoms')}

    def __len__(self):
        return len(self.meta_dict['mol_ids'])
    
    def __getitem__(self, idx):
        e_start = self.meta_dict['edge_slices'][idx].item()
        e_end = self.meta_dict['edge_slices'][idx + 1].item()
        start = self.meta_dict['atom_slices'][idx].item()
        n_atoms = self.meta_dict['n_atoms'][idx].item()
        
        return self.data_by_type(e_start, e_end, start, n_atoms)

    def data_by_type(self, e_start, e_end, start, n_atoms):
        g = self.get_graph(e_start, e_end, n_atoms, start)
        return g

    def get_graph(self, e_start, e_end, n_atoms, start):
        edge_indices = self.edge_indices[:, e_start: e_end]
        node_features = self.features_tensor[start: start + n_atoms].to(self.device)
        edge_features = self.e_features_tensor[e_start: e_end].to(self.device)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=node_features,
            edge_index=edge_indices.to(self.device),
            edge_attr=edge_features,
            num_nodes=n_atoms
        )
        return data

def graph_collate(batch):
    return Batch.from_data_list(batch)

class DataLoaderMaskingPred(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle=False, num_atom_type = 118, num_edge_type = 4, 
            mask_rate = 0.0, mask_edge = 0.0, **kwargs):
        
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge
        
        super(DataLoaderMaskingPred, self).__init__(
            dataset, batch_size, shuffle, collate_fn = self.collate_fn,
            **kwargs)
        
    def collate_fn(self, batch):
        batch_masked_atom_indices, batch_masked_edge_indices = list(), list()
        batch_mask_node_labels, batch_mask_edge_labels = list(), list()
        accum_node, accum_edge = 0, 0
        
        modified_batch = []
        
        for graph in batch:
            ### masked atom random sampling
            num_atoms = graph.x.size()[0]
            sample_size = int(num_atoms * self.mask_rate + 1)
            masked_atom_indices = random.sample(range(num_atoms), sample_size)

            for atom_idx in masked_atom_indices:
                batch_mask_node_labels.append(copy.deepcopy(graph.x[atom_idx]).view(1, -1))

            # modify the original node feature of the masked node
            for atom_idx in masked_atom_indices:
                graph.x[atom_idx] = torch.tensor([self.num_atom_type, 0, 0, 0, 0, 0, 0, 0, 0], device=graph.x.device) # original atom type to masked token

            if self.mask_edge:
                # create mask edge labels by copying edge features of edges that are bonded to mask atoms
                connected_edge_indices = []

                for bond_idx, (u, v) in enumerate(graph.edge_index.t().cpu().numpy()):
                    for atom_idx in masked_atom_indices:
                        if atom_idx in set((u, v)) and bond_idx not in connected_edge_indices:
                            connected_edge_indices.append(bond_idx)

                if len(connected_edge_indices) > 0:
                    # create mask edge labels by copying bond features of the bonds connected to the mask atoms
                    for bond_idx in connected_edge_indices[::2]:
                        batch_mask_edge_labels.append(copy.deepcopy(graph.edge_attr[bond_idx]).view(1, -1))
                        
                    # modify the original edge feature of the masked edge
                    for bond_idx in connected_edge_indices:
                        graph.edge_attr[bond_idx][0] = self.num_edge_type # use the last type to indicate the masked edge

                # add masked atom and edge indices for batch
                for atom_idx in masked_atom_indices:
                    batch_masked_atom_indices.append(atom_idx + accum_node)
                for bond_idx in connected_edge_indices:
                    batch_masked_edge_indices.append(bond_idx + accum_edge)

            else:
                # add masked atom indices for batch
                for atom_idx in masked_atom_indices:
                    batch_masked_atom_indices.append(atom_idx + accum_node)

            # accumulate node and edge numbers
            accum_node += num_atoms
            accum_edge += graph.edge_index.shape[1]
            
            modified_batch.append(graph)
            
        # Batch the modified graphs
        batched_graph = Batch.from_data_list(modified_batch)
        
        # Convert lists to tensors if they have elements
        if len(batch_masked_atom_indices) > 0:
            batch_masked_atom_indices = torch.tensor(batch_masked_atom_indices, device=batched_graph.x.device)
        else:
            batch_masked_atom_indices = torch.tensor([], device=batched_graph.x.device)
        
        if len(batch_mask_node_labels) > 0:
            batch_mask_node_labels = torch.cat(batch_mask_node_labels, dim=0)
        else:
            batch_mask_node_labels = torch.tensor([], device=batched_graph.x.device)
        
        if self.mask_edge:
            if len(batch_masked_edge_indices) > 0:
                batch_masked_edge_indices = torch.tensor(batch_masked_edge_indices, device=batched_graph.x.device)
            else:
                batch_masked_edge_indices = torch.tensor([], device=batched_graph.x.device)
            
            if len(batch_mask_edge_labels) > 0:
                batch_mask_edge_labels = torch.cat(batch_mask_edge_labels, dim=0)
            else:
                batch_mask_edge_labels = torch.tensor([], device=batched_graph.x.device)
            
            return batched_graph, batch_masked_atom_indices, batch_masked_edge_indices, batch_mask_node_labels, batch_mask_edge_labels
        
        else:
            return batched_graph, batch_masked_atom_indices, batch_mask_node_labels
