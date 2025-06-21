# bindingdb_interface.py
"""
End-to-end inference for one SMILES and one UniProt ID using BlendNetS.
"""
import torch
import pickle
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data

from modules.common.utils import load_cfg
from modules.interaction_modules.BDB_models import BlendNetS
from modules.interaction_modules.BDB_loaders import pad_data
from feature_generation.compound.Get_Mol_features import get_mol_features

class BindingDBInterface:
    def __init__(self,
                 config_path: str,
                 ki_weights: str,
                 ic50_weights: str,
                 device: str = None):
        # load config
        self.cfg = load_cfg(config_path)
        # device
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # load protein features and pocket indices
        with open(self.cfg['Path']['Ki_protein_feat'], 'rb') as f:
            self.protein_feats = pickle.load(f)
        with open(self.cfg['Path']['Ki_pockets'], 'rb') as f:
            self.pocket_inds = pickle.load(f)

        # load Ki model
        ki_teacher = self.cfg['Path']['Ki_interaction_site_predictor']
        self.ki_model = BlendNetS(ki_teacher, self.cfg, self.device).to(self.device)
        self.ki_model.load_state_dict(torch.load(ki_weights, map_location=self.device, weights_only=False))
        self.ki_model.eval()

        # load IC50 model
        ic50_teacher = self.cfg['Path']['IC50_interaction_site_predictor']
        self.ic50_model = BlendNetS(ic50_teacher, self.cfg, self.device).to(self.device)
        self.ic50_model.load_state_dict(torch.load(ic50_weights, map_location=self.device, weights_only=False))
        self.ic50_model.eval()

    def _prepare_sample(self, smiles: str, protein_id: str):
        # protein features
        pfeat = self.protein_feats[protein_id]  # numpy array [seq_len, feat_dim]
        pocket = self.pocket_inds[protein_id]   # list of residue indices
        seqlen = pfeat.shape[0]
        # build compound graph via RDKit + Get_Mol_features
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        atom_feats, edge_index, edge_feats, _ = get_mol_features(mol)
        graph = Data(
            # atom feature indices must be long for embedding layers
            x=torch.tensor(atom_feats, dtype=torch.long, device=self.device),
            edge_index=edge_index.to(self.device),
            edge_attr=edge_feats.to(self.device),
            num_nodes=len(atom_feats)
        )
        return {
            'pfeat': pfeat,
            'seqlength': seqlen,
            'pocket': pocket,
            'compound_graph': graph,
            'num_node': len(atom_feats),
            'label': 0.0
        }

    def predict(self, smiles: str, protein_id: str) -> dict:
        """
        Returns predicted Ki and IC50 for one SMILES and UniProt ID.
        """
        sample = self._prepare_sample(smiles, protein_id)
        # pad single sample into batch
        (resi_feat, _), comp_batch, _, pocket_mask, comp_mask = pad_data([sample])
        print(resi_feat,comp_batch,pocket_mask,comp_mask)
        with torch.no_grad():
            # Debug: check tensor types
            print(f"comp_batch.x dtype: {comp_batch.x.dtype}")
            print(f"comp_batch.edge_attr dtype: {comp_batch.edge_attr.dtype}")
            
            # Ensure compound graph tensors are correct type and create copies for each model
            comp_batch.x = comp_batch.x.long()
            comp_batch.edge_attr = comp_batch.edge_attr.long()
            
            # Create a deep copy for Ki model to avoid in-place modifications
            import copy
            comp_batch_ki = copy.deepcopy(comp_batch)
            comp_batch_ic50 = copy.deepcopy(comp_batch)
            
            # Try Ki model first with fresh copy
            ki_pred, *_ = self.ki_model(resi_feat, comp_batch_ki, pocket_mask, comp_mask)
            print("Ki prediction successful")
            
            # Try IC50 with its own copy
            ic50_pred, *_ = self.ic50_model(resi_feat, comp_batch_ic50, pocket_mask, comp_mask)
            print("IC50 prediction successful")
            
        return {
            'Ki': float(ki_pred.cpu().item()),
            'IC50': float(ic50_pred.cpu().item())
        }

# from bindingdb_interface import BindingDBInterface

iface = BindingDBInterface(
    config_path="BindingDB.yml",
    ki_weights="/home/sarvesh/scratch/GS/negroni_data/Blendnet/model_checkpoint/BindingDB/Ki/random_split/CV1/BlendNet_S.pth",
    ic50_weights="/home/sarvesh/scratch/GS/negroni_data/Blendnet/model_checkpoint/BindingDB/IC50/random_split/CV1/BlendNet_S.pth",
    device="cuda:0"
)

out = iface.predict("CCO", "O00408")
print(out)  # {'Ki': ..., 'IC50': ...}