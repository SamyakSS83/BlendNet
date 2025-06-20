#!/usr/bin/env python3
"""
Simple client API for predicting BindingDB affinities (Ki or IC50) using BlendNetS.
"""
import os
import yaml
import torch
from torch.utils.data import DataLoader

from modules.interaction_modules.BDB_loaders import BADataset, pad_data
from modules.interaction_modules.BDB_models import BlendNetS
from modules.common.utils import load_cfg


def _load_bindingdb_config(config_path=None):
    cfg_path = config_path or os.path.join(os.path.dirname(__file__), '/home/sarvesh/scratch/GS/samyak/.Blendnet/code/BindingDB.yml')
    return load_cfg(cfg_path)


class BindingDBClient:
    """
    Client for predicting Ki or IC50 affinities using the BlendNetS model.
    """
    def __init__(self, mode='Ki', config_path=None, device='auto'):
        assert mode in ('Ki', 'IC50'), "mode must be 'Ki' or 'IC50'"
        self.mode = mode
        self.config = _load_bindingdb_config(config_path)
        # Select paths based on mode
        P = self.config['Path']
        feat_path = P[f'{mode}_protein_feat']
        pocket_path = P[f'{mode}_pockets']
        graph_path = P['Ligand_graph']
        teacher_path = P[f'{mode}_interaction_site_predictor']
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() and device!='cpu' else 'cpu')
        # Initialize dataset parameters
        self.feat_path = feat_path
        self.pocket_path = pocket_path
        self.graph_path = graph_path
        # Load model
        self.model = BlendNetS(teacher_path, self.config, self.device)
        self.model.eval()
    
    def predict(
        self,
        protein_compound_pairs,
        batch_size=8
    ):
        """
        Batch predict affinities for a list of (protein_id, compound_id) tuples.
        Returns a list of floats (Ki or IC50 values).
        """
        ids = [f"{pid}_{cid}" for pid, cid in protein_compound_pairs]
        labels = [0.0] * len(ids)
        # Prepare dataset and loader
        dataset = BADataset(
            interaction_IDs=ids,
            labels=labels,
            protein_feature_path=self.feat_path,
            pocket_path=self.pocket_path,
            compound_feature_path=self.graph_path,
            device=self.device
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=pad_data
        )
        preds = []
        with torch.no_grad():
            for prot_data, comp_graphs, labels, pocket_mask, comp_mask in loader:
                prot_data = tuple(x.to(self.device) if isinstance(x, torch.Tensor) else x for x in prot_data)
                comp_graphs = comp_graphs.to(self.device)
                pocket_mask = pocket_mask.to(self.device)
                comp_mask = comp_mask.to(self.device)
                outputs, *_ = self.model(prot_data, comp_graphs, pocket_mask, comp_mask)
                # outputs is tensor of shape (batch,)
                preds.extend(outputs.cpu().numpy().tolist())
        return preds

    def predict_single(self, protein_id, compound_id):
        """
        Predict affinity for a single protein-compound pair.
        Returns a float.
        """
        return self.predict([(protein_id, compound_id)], batch_size=1)[0]


# Convenience functions
_default_kiclient = BindingDBClient('Ki')
_default_ic50client = BindingDBClient('IC50')

def predict_Ki(protein_id, compound_id):
    """Predict Ki for one protein-compound pair."""
    return _default_kiclient.predict_single(protein_id, compound_id)


def predict_IC50(protein_id, compound_id):
    """Predict IC50 for one protein-compound pair."""
    return _default_ic50client.predict_single(protein_id, compound_id)
