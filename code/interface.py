# inference_interface.py

import torch
import rdkit.Chem as Chem
import numpy as np
import pickle
from modules.pocket_modules.loaders import PocketDataset
from modules.compound_modules.loaders import MoleculeGraphDataset, graph_collate
from modules.interaction_modules.BDB_models import BlendNetS
from modules.common.utils import load_cfg
from torch.utils.data import DataLoader

class BindingAffinityPredictor:
    """
    Unified interface to predict Ki and IC50 from SMILES and protein sequence.
    """

    def __init__(self,
                 pocket_cfg: str,
                 compound_cfg: str,
                 interaction_cfg: str,
                 ki_model_path: str,
                 ic50_model_path: str,
                 device: str = None):
        # load configs
        self.pocket_cfg = load_cfg(pocket_cfg)
        self.compound_cfg = load_cfg(compound_cfg)
        self.inter_cfg   = load_cfg(interaction_cfg)

        # set device
        self.device = torch.device(device or
                        ("cuda" if torch.cuda.is_available() else "cpu"))

        # load pocket extractor features (precomputed)
        with open(self.pocket_cfg["Path"]["prot_feats"], "rb") as f:
            self.prot_feats = pickle.load(f)

        # instantiate compound‐graph loader (no dataset path needed, we'll override)
        self._compound_dataset = MoleculeGraphDataset(
            processed_file=self.compound_cfg["Path"]["dataset_path"]
        )

        # load interaction models
        ki_teacher_path = self.inter_cfg['Path']['Ki_interaction_site_predictor']
        self.ki_model = BlendNetS(ki_teacher_path, self.inter_cfg, self.device)
        self.ki_model.load_state_dict(torch.load(ki_model_path, map_location=self.device))
        self.ki_model.eval()

        ic50_teacher_path = self.inter_cfg['Path']['IC50_interaction_site_predictor']
        self.ic50_model = BlendNetS(ic50_teacher_path, self.inter_cfg, self.device)
        self.ic50_model.load_state_dict(torch.load(ic50_model_path, map_location=self.device))
        self.ic50_model.eval()

    def _smiles_to_graph(self, smiles: str):
        """Turn a SMILES string into the same graph‐tensor tuple used by MoleculeGraphDataset."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        # MoleculeGraphDataset does its own parsing. Here we monkey‐patch it:
        idx = len(self._compound_dataset)
        self._compound_dataset._cache[idx] = mol  # assume internal API
        loader = DataLoader([idx],
                            batch_size=1,
                            collate_fn=graph_collate)
        return next(iter(loader))

    def _seq_to_features(self, seq: str):
        """Fetch or compute protein features for a raw sequence."""
        # if precomputed
        if seq in self.prot_feats:
            return self.prot_feats[seq]
        # otherwise fall back to PocketDataset machinery:
        ds = PocketDataset(PID=[seq], Pseqs=[seq],
                           Pfeatures=None, Labels=[0])
        return ds[0][1]  # returns (id, features, label)

    def predict_ki(self, smiles: str, protein_seq: str) -> float:
        """
        Returns the predicted Ki (in log‐units) for a single SMILES+sequence.
        """
        graph_batch = self._smiles_to_graph(smiles)
        prot_feat   = self._seq_to_features(protein_seq)
        with torch.no_grad():
            inputs = {'compound': graph_batch,
                      'protein': prot_feat.to(self.device)}
            return float(self.ki_model(**inputs).cpu().squeeze())

    def predict_ic50(self, smiles: str, protein_seq: str) -> float:
        """
        Returns the predicted IC50 (in log‐units) for a single SMILES+sequence.
        """
        graph_batch = self._smiles_to_graph(smiles)
        prot_feat   = self._seq_to_features(protein_seq)
        with torch.no_grad():
            inputs = {'compound': graph_batch,
                      'protein': prot_feat.to(self.device)}
            return float(self.ic50_model(**inputs).cpu().squeeze())

    def predict(self, smiles: str, protein_seq: str) -> dict:
        """
        Returns both Ki and IC50 predictions in one call.
        """
        return {
            'Ki':   self.predict_ki(smiles, protein_seq),
            'IC50': self.predict_ic50(smiles, protein_seq)
        }
    
predictor = BindingAffinityPredictor(
    pocket_cfg="pocket_extractor_config.yml",
    compound_cfg="compound_VQVAE.yml",
    interaction_cfg="BindingDB.yml",
    ki_model_path="path/to/your/Ki_model.pth",
    ic50_model_path="path/to/your/IC50_model.pth",
    device="cuda:0"
)

result = predictor.predict("CCO", "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGE")
print("Predicted Ki:", result['Ki'])
print("Predicted IC50:", result['IC50'])