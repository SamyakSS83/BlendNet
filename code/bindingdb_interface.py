# bindingdb_interface.py
"""
End-to-end inference for one SMILES and one protein sequence using BlendNetS.
"""
import re
import torch
import pickle
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data
from transformers import BertModel, BertTokenizer

from modules.common.utils import load_cfg
from modules.interaction_modules.BDB_models import BlendNetS
from modules.interaction_modules.BDB_loaders import pad_data
from feature_generation.compound.Get_Mol_features import get_mol_features
from modules.pocket_modules.pseq2sites_embeddings import Pseq2SitesEmbeddings

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

        # Initialize ProtBERT for protein feature extraction
        print("Loading ProtBERT...")
        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        self.protbert_model = BertModel.from_pretrained("Rostlab/prot_bert", use_safetensors=True).to(self.device)
        self.protbert_model.eval()
        
        # Initialize Pseq2Sites for pocket prediction
        print("Loading Pseq2Sites...")
        self.pseq2sites = Pseq2SitesEmbeddings(device=self.device)

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

    def _get_protein_features(self, protein_seq: str):
        """Generate ProtBERT features from protein sequence"""
        if not protein_seq or len(protein_seq) == 0:
            raise ValueError("Protein sequence cannot be empty")
            
        # Clean sequence and format for ProtBERT
        clean_seq = re.sub(r"[UZOB]", "X", protein_seq)
        formatted_seq = " ".join(list(clean_seq))
        
        # Tokenize with proper length handling
        ids = self.tokenizer.batch_encode_plus(
            [formatted_seq], 
            add_special_tokens=True, 
            padding=True, 
            truncation=True, 
            max_length=1024,  # ProtBERT max length
            return_tensors='pt'
        )
        
        input_ids = ids['input_ids'].to(self.device)
        attention_mask = ids['attention_mask'].to(self.device)
        
        # Get ProtBERT embeddings
        with torch.no_grad():
            embedding = self.protbert_model(input_ids=input_ids, attention_mask=attention_mask)[0]
            embedding = embedding.cpu().numpy()
            seq_len = (attention_mask[0] == 1).sum()
            
            # Remove [CLS] and [SEP] tokens, handle max length
            if seq_len < 1024:
                seq_emb = embedding[0][1:seq_len-1]
            else:
                seq_emb = embedding[0][1:1023]  # Max length handling
                print(f"Warning: Protein sequence truncated to {seq_emb.shape[0]} residues")
                
        return seq_emb
    
    def _predict_pockets(self, protein_features, protein_seq):
        """Predict pocket indices using Pseq2Sites"""
        try:
            # Prepare data for Pseq2Sites
            temp_id = "temp_protein"
            protein_feats = {temp_id: protein_features}
            protein_seqs = {temp_id: protein_seq}
            
            # Get predictions
            results = self.pseq2sites.extract_embeddings(
                protein_features=protein_feats,
                protein_sequences=protein_seqs,
                return_predictions=True,
                batch_size=1
            )
            
            # Extract pocket indices (binding sites with probability > 0.5)
            binding_probs = results[temp_id]['binding_site_probabilities']
            pocket_indices = np.where(binding_probs > 0.5)[0].tolist()
            
            # If no binding sites predicted, use top 10% of residues
            if len(pocket_indices) == 0:
                n_top = max(1, len(binding_probs) // 10)
                top_indices = np.argsort(binding_probs)[-n_top:]
                pocket_indices = top_indices.tolist()
                print(f"Warning: No high-confidence binding sites found. Using top {len(pocket_indices)} residues.")
                
            print(f"Predicted {len(pocket_indices)} pocket residues")
            return pocket_indices
            
        except Exception as e:
            print(f"Warning: Pseq2Sites prediction failed: {e}. Using fallback pocket prediction.")
            # Fallback: use every 10th residue as a simple pocket prediction
            seq_len = len(protein_seq)
            fallback_indices = list(range(0, seq_len, 10))
            return fallback_indices

    def _prepare_sample(self, smiles: str, protein_seq: str):
        # Generate protein features from sequence
        pfeat = self._get_protein_features(protein_seq)
        seqlen = pfeat.shape[0]
        
        # Predict pocket indices
        pocket = self._predict_pockets(pfeat, protein_seq)
        
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

    def predict(self, smiles: str, protein_seq: str) -> dict:
        """
        Returns predicted Ki and IC50 for one SMILES and protein sequence.
        
        Args:
            smiles: SMILES string of the compound
            protein_seq: Amino acid sequence of the protein
            
        Returns:
            Dict with 'Ki' and 'IC50' predictions
        """
        sample = self._prepare_sample(smiles, protein_seq)
        # pad single sample into batch
        (resi_feat, _), comp_batch, _, pocket_mask, comp_mask = pad_data([sample])
        
        with torch.no_grad():
            # Ensure compound graph tensors are correct type and create copies for each model
            comp_batch.x = comp_batch.x.long()
            comp_batch.edge_attr = comp_batch.edge_attr.long()
            
            # Create a deep copy for Ki model to avoid in-place modifications
            import copy
            comp_batch_ki = copy.deepcopy(comp_batch)
            comp_batch_ic50 = copy.deepcopy(comp_batch)
            
            # Get predictions from both models
            ki_pred, *_ = self.ki_model(resi_feat, comp_batch_ki, pocket_mask, comp_mask)
            ic50_pred, *_ = self.ic50_model(resi_feat, comp_batch_ic50, pocket_mask, comp_mask)
            
            print(f"Predictions completed successfully")
            
        return {
            'Ki': float(ki_pred.cpu().item()),
            'IC50': float(ic50_pred.cpu().item())
        }

# Example usage with a real protein sequence
if __name__ == "__main__":
    # Example protein sequence (first 200 residues of a real protein)
    example_protein_seq = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQHKLRKLNPPDESGPGCMSCKCVLS"
    
    iface = BindingDBInterface(
        config_path="BindingDB.yml",
        ki_weights="/home/sarvesh/scratch/GS/negroni_data/Blendnet/model_checkpoint/BindingDB/Ki/random_split/CV1/BlendNet_S.pth",
        ic50_weights="/home/sarvesh/scratch/GS/negroni_data/Blendnet/model_checkpoint/BindingDB/IC50/random_split/CV1/BlendNet_S.pth",
        device="cuda:0"
    )

    out = iface.predict("CCO", example_protein_seq)
    print(out)  # {'Ki': ..., 'IC50': ...}