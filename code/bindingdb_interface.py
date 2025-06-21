# bindingdb_interface.py
"""
End-to-end inference for one SMILES and one protein sequence using BlendNetS.
"""
import torch
import pickle
from rdkit import Chem
import numpy as np
from torch_geometric.data import Data
from feature_generation.compound.Get_Mol_features import get_mol_features
from transformers import BertModel, BertTokenizer

from modules.common.utils import load_cfg
from modules.interaction_modules.BDB_models import BlendNetS
from modules.pocket_modules.models import Pseq2Sites
from modules.interaction_modules.BDB_loaders import pad_data

class BindingDBInterface:
    def __init__(self,
                 bindingdb_cfg: str,
                 pocket_cfg: str,
                 ki_weights: str,
                 ic50_weights: str,
                 device: str = None):
        # load configs
        self.cfg = load_cfg(bindingdb_cfg)
        self.pocket_cfg = load_cfg(pocket_cfg)
        # device
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # load ProtBERT for protein embedding
        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        self.protbert = BertModel.from_pretrained("Rostlab/prot_bert").to(self.device)
        self.protbert.eval()

        # load pocket extractor Pseq2Sites
        self.pocket_model = Pseq2Sites(self.pocket_cfg).to(self.device)
        ckpt = self.pocket_cfg['Path']['check_point']
        self.pocket_model.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.pocket_model.eval()

        # load Ki and IC50 BlendNetS models
        ki_teacher = self.cfg['Path']['Ki_interaction_site_predictor']
        self.ki_model = BlendNetS(ki_teacher, self.cfg, self.device).to(self.device)
        self.ki_model.load_state_dict(torch.load(ki_weights, map_location=self.device))
        self.ki_model.eval()

        ic50_teacher = self.cfg['Path']['IC50_interaction_site_predictor']
        self.ic50_model = BlendNetS(ic50_teacher, self.cfg, self.device).to(self.device)
        self.ic50_model.load_state_dict(torch.load(ic50_weights, map_location=self.device))
        self.ic50_model.eval()

    def _get_protein_feats_and_pocket(self, seq: str):
        # tokenize and embed
        seq_clean = " ".join(list(seq))
        ids = self.tokenizer.batch_encode_plus([seq_clean], add_special_tokens=True, padding='max_length',
                                               max_length=self.pocket_cfg['Architecture']['max_lengths'],
                                               truncation=True, return_tensors='pt')
        input_ids = ids['input_ids'].to(self.device)
        att_mask = ids['attention_mask'].to(self.device)
        with torch.no_grad():
            emb = self.protbert(input_ids=input_ids, attention_mask=att_mask)[0]  # [1, L, 1024]
        # remove CLS/SEP, recover true length
        true_len = (att_mask[0]==1).sum().item() - 2
        pfeat = emb[0,1:1+true_len].cpu().numpy()  # [true_len, 1024]

        # build Pseq2Sites inputs
        maxL = self.pocket_cfg['Architecture']['max_lengths']
        inputD = self.pocket_cfg['Architecture']['prots_input_dim']
        seq_len = pfeat.shape[0]
        # prot_feat padded
        prot_feat = np.zeros((maxL, inputD), dtype=np.float32)
        prot_feat[:seq_len] = pfeat
        # total_prot_feat = sum embeddings replic.
        sum_feat = pfeat.sum(axis=0)
        total_feat = np.zeros((maxL, inputD), dtype=np.float32)
        total_feat[:seq_len] = sum_feat
        # masks and position ids
        mask = np.array([1]*seq_len + [0]*(maxL-seq_len), dtype=np.int64)
        pos_ids = np.arange(maxL, dtype=np.int64)
        # to tensors
        t_feat = torch.tensor(prot_feat, dtype=torch.float32, device=self.device)
        t_total = torch.tensor(total_feat, dtype=torch.float32, device=self.device)
        t_mask = torch.tensor(mask, dtype=torch.long, device=self.device)
        t_pos = torch.tensor(pos_ids, dtype=torch.long, device=self.device)
        # get pocket logits
        with torch.no_grad():
            _, _, logits, _ = self.pocket_model(t_feat.unsqueeze(0), t_total.unsqueeze(0), t_mask.unsqueeze(0), t_pos.unsqueeze(0))
        probs = torch.sigmoid(logits[0])  # [maxL]
        pocket_inds = (probs>0.5).nonzero().flatten().cpu().tolist()
        return prot_feat, pocket_inds

    def _smiles_to_graph(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        atom_feats, edge_idx, edge_feats, _ = get_mol_features(mol)
        # convert indices to int
        x = torch.tensor(atom_feats, dtype=torch.long, device=self.device)
        return Data(
            x=x,
            edge_index=edge_idx.to(self.device),
            edge_attr=edge_feats.to(self.device),
            num_nodes=x.size(0)
        )

    def predict(self, smiles: str, protein_seq: str):
        # get protein features and pocket indices
        pfeat, pocket = self._get_protein_feats_and_pocket(protein_seq)
        seqlen = pfeat.shape[0]
        # build compound graph
        comp_graph = self._smiles_to_graph(smiles)
        num_node = comp_graph.num_nodes
        # sample dict
        sample = {'pfeat': pfeat, 'seqlength': seqlen, 'pocket': pocket,
                  'compound_graph': comp_graph, 'num_node': num_node, 'label':0.0}
        (resi, _), batch_graph, _, p_mask, c_mask = pad_data([sample])
        # cast node indices to Long for embedding
        batch_graph.x = batch_graph.x.long()
        batch_graph.edge_attr = batch_graph.edge_attr.long()
        with torch.no_grad():
            ki, *_ = self.ki_model(resi, batch_graph, p_mask, c_mask)
            ic50, *_ = self.ic50_model(resi, batch_graph, p_mask, c_mask)
        return {'Ki': float(ki.cpu().item()), 'IC50': float(ic50.cpu().item())}

if __name__ == '__main__':
    # example usage:
    iface = BindingDBInterface(
        bindingdb_cfg="BindingDB.yml",
        pocket_cfg="pocket_extractor_config.yml",
        ki_weights="/home/sarvesh/scratch/GS/negroni_data/Blendnet/model_checkpoint/BindingDB/Ki/random_split/CV1/BlendNet_S.pth",
        ic50_weights="/home/sarvesh/scratch/GS/negroni_data/Blendnet/model_checkpoint/BindingDB/IC50/random_split/CV1/BlendNet_S.pth",
        device="cuda:0"
    )

    # replace second arg with actual protein sequence (amino-acid letters)
    out = iface.predict("CCO", "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGE")
    print(out)  # {'Ki': ..., 'IC50': ...}