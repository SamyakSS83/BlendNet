#!/usr/bin/env python3
import torch
import numpy as np
from rdkit import Chem
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from modules.common.utils import load_cfg
from modules.interaction_modules.BDB_models import BlendNetS
from modules.interaction_modules.BDB_loaders import pad_data
from feature_generation.compound.Get_Mol_features import get_mol_features, remove_hydrogen
from pseq2sites_interface import get_protein_matrix

# --- build a little helper to turn a SMILES into the exact same PyG graph BlendNet uses ---
def smiles_to_graph(smiles: str, device: torch.device):
    """
    Parse a SMILES string via RDKit, build atom/edge features and return a PyG Data graph plus atom mask.
    """
    mol = Chem.MolFromSmiles(smiles)
    atom_feats_list, edge_idx, edge_feats, _ = get_mol_features(mol)
    # drop hydrogens if present
    n_atoms, atom_feats_list, edge_feats, edge_idx, _ = remove_hydrogen(atom_feats_list, edge_idx, edge_feats)
    # convert to tensors
    x = torch.tensor(atom_feats_list, dtype=torch.float32, device=device)
    edge_index = torch.tensor(edge_idx, dtype=torch.long, device=device)
    edge_attr = torch.tensor(edge_feats, dtype=torch.float32, device=device)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n_atoms)
    mask = torch.ones(n_atoms, dtype=torch.float32, device=device)
    return data, mask.unsqueeze(0)

def prepare_protein(seq: str, device: torch.device):
    # use the Pseq2SitesClient under‐the‐hood to get per‐residue embeddings
    # get_protein_matrix wants a ProtBERT feature matrix, but Pseq2SitesEmbeddings will
    # actually compute it from seq internally if you pass any dummy feature array of right shape.
    feats = np.zeros((len(seq), 1024), dtype=np.float32)
    matrix = get_protein_matrix(feats, seq)           # returns (L x D) numpy
    prot = torch.from_numpy(matrix).to(device)        # (L x D)
    mask = torch.ones(len(seq), device=device)        # (L,)
    return prot.unsqueeze(0), mask.unsqueeze(0)       # add batch dim

# Move config, device, compound_data and models to module scope to load once
cfg = load_cfg("BindingDB.yml")
device = torch.device(f"cuda:{cfg['Train']['device']}") if torch.cuda.is_available() else torch.device("cpu")
# load ligand graph via memory map to avoid full deserialization overhead
compound_data = torch.load(cfg['Path']['Ligand_graph'], map_location='cpu')
# instantiate models once
model_ki = BlendNetS(cfg["Path"]["Ki_interaction_site_predictor"], cfg, device).to(device).eval()
model_ic50 = BlendNetS(cfg["Path"]["IC50_interaction_site_predictor"], cfg, device).to(device).eval()

def predict(seq: str, smiles: str):
    # reuse globally loaded compound_data and models
     
    # build our single‐sample inputs
    prot_feat, prot_mask = prepare_protein(seq, device)
    comp_graph, comp_mask = smiles_to_graph(smiles, device)

    # run
    with torch.no_grad():
        ki_pred,    *_ = model_ki(prot_feat,    comp_graph, prot_mask,    comp_mask)
        ic50_pred, *_ = model_ic50(prot_feat, comp_graph, prot_mask, comp_mask)

    return ki_pred.item(), ic50_pred.item()

if __name__ == "__main__":
    seq = (
        "MLTFNHDAPWHTQKTLKTSEFGKSFGTLGHIGNISHQCWAGCAAGGRAVLSGEPEANMDQETVG"
        "NVVLLAIVTLISVVQNGFFAHKVEHESRTQNGRSFQRTGTLAFERVYTANQNCVDAYPTFLAVLWS"
        "AGLLCSQVPAAFAGLMYLFVRQKYFVGYLGERTQSTPGYIFGKRIILFLFLMSVAGIFNYYLIFF"
        "FGSDFENYIKTISTTISPLLLIP"
    )
    smiles = "Cc1ccc(COc2ccc3nc(C4C(C(=O)O)C4(C)C)n(Cc4ccc(Br)cc4)c3c2)nc1"

    ki, ic50 = predict(seq, smiles)
    print(f"Predicted Ki:  {ki:.4f}")
    print(f"Predicted IC50:{ic50:.4f}")