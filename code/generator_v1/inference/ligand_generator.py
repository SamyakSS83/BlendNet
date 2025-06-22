"""
Inference script for protein-conditioned ligand generation.
"""
import os
import sys
import torch
import numpy as np
import pickle
from typing import List, Optional, Dict, Tuple
from rdkit import Chem
from rdkit.Chem import Descriptors

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../materials.smi-ted/smi-ted/'))

from models.diffusion_model import ProteinLigandDiffusion
from database.vector_database import ProteinLigandVectorDB
from bindingdb_interface import BindingDBInterface
from inference.smi_ted_light.load import load_smi_ted


class LigandGenerator:
    """Complete pipeline for protein-conditioned ligand generation."""
    
    def __init__(self,
                 diffusion_checkpoint: str,
                 vector_db_path: str,
                 smi_ted_path: str = "../../../materials.smi-ted/smi-ted/inference/smi_ted_light",
                 smi_ted_ckpt: str = "smi-ted-Light_40.pt",
                 device: str = "cuda"):
        """
        Initialize ligand generator.
        
        Args:
            diffusion_checkpoint: Path to trained diffusion model
            vector_db_path: Path to vector database
            smi_ted_path: Path to smi-TED model
            smi_ted_ckpt: smi-TED checkpoint filename
            device: Device for computation
        """
        self.device = device
        
        print("Loading models...")
        
        # Load diffusion model
        print("Loading diffusion model...")
        checkpoint = torch.load(diffusion_checkpoint, map_location=device)
        config = checkpoint['config']
        
        self.diffusion_model = ProteinLigandDiffusion(
            compound_dim=config['compound_dim'],
            protbert_dim=config['protbert_dim'],
            pseq2sites_dim=config['pseq2sites_dim'],
            num_timesteps=config['num_timesteps'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(device)
        
        self.diffusion_model.load_state_dict(checkpoint['model_state_dict'])
        self.diffusion_model.eval()
        
        # Load vector database
        print("Loading vector database...")
        self.vector_db = ProteinLigandVectorDB()
        self.vector_db.load_database(vector_db_path)
        
        # Load smi-TED
        print("Loading smi-TED...")
        self.smi_ted = load_smi_ted(
            folder=smi_ted_path,
            ckpt_filename=smi_ted_ckpt
        )
        
        # Load BindingDB interface for protein embeddings and IC50 prediction
        print("Loading BindingDB interface...")
        self.bindingdb_interface = BindingDBInterface(
            config_path="../BindingDB.yml",
            ki_weights="/home/sarvesh/scratch/GS/negroni_data/Blendnet/model_checkpoint/BindingDB/Ki/random_split/CV1/BlendNet_S.pth",
            ic50_weights="/home/sarvesh/scratch/GS/negroni_data/Blendnet/model_checkpoint/BindingDB/IC50/random_split/CV1/BlendNet_S.pth",
            device=device
        )
        
        print("All models loaded successfully!")
        
    def generate_protein_embeddings(self, protein_sequence: str) -> Tuple[np.ndarray, np.ndarray]:
        """Generate protein embeddings for input sequence."""
        print("Generating protein embeddings...")
        
        # Generate ProtBERT features
        protbert_feat = self.bindingdb_interface._get_protein_features(protein_sequence)
        
        # Generate Pseq2Sites features (using existing interface)
        from pseq2sites_interface import get_protein_matrix
        pseq2sites_feat = get_protein_matrix(protbert_feat, protein_sequence)
        
        # Pool to fixed size
        protbert_pooled = np.mean(protbert_feat, axis=0)  # [1024]
        pseq2sites_pooled = np.mean(pseq2sites_feat, axis=0)  # [embedding_dim]
        
        return protbert_pooled, pseq2sites_pooled
        
    def retrieve_similar_compounds(self,
                                  protbert_emb: np.ndarray,
                                  pseq2sites_emb: np.ndarray,
                                  k: int = 10,
                                  alpha: float = 0.5) -> Tuple[List[str], np.ndarray, List[float]]:
        """Retrieve similar compounds from vector database."""
        print(f"Retrieving top-{k} similar compounds...")
        
        # Search vector database
        indices, scores = self.vector_db.search_similar_proteins(
            protbert_emb, pseq2sites_emb, k=k, alpha=alpha
        )
        
        # Get compound data
        smiles_list, compound_embeddings, ic50_values = self.vector_db.get_compounds_for_indices(indices)
        
        print(f"Retrieved {len(smiles_list)} compounds")
        print(f"IC50 range: {min(ic50_values):.2f} - {max(ic50_values):.2f}")
        
        return smiles_list, compound_embeddings, ic50_values
        
    def generate_ligands(self,
                        protein_sequence: str,
                        num_samples: int = 5,
                        k_retrieve: int = 10,
                        alpha: float = 0.5,
                        use_retrieval_init: bool = True) -> List[Dict]:
        """
        Generate ligands for a given protein sequence.
        
        Args:
            protein_sequence: Target protein sequence
            num_samples: Number of ligands to generate
            k_retrieve: Number of similar compounds to retrieve
            alpha: Weight for similarity combination
            use_retrieval_init: Whether to use retrieved compounds as initialization
            
        Returns:
            List of generated ligand data
        """
        print(f"Generating {num_samples} ligands for protein sequence...")
        print(f"Protein length: {len(protein_sequence)}")
        
        # Step 1: Generate protein embeddings
        protbert_emb, pseq2sites_emb = self.generate_protein_embeddings(protein_sequence)
        
        # Step 2: Retrieve similar compounds
        retrieved_smiles, retrieved_embeddings, retrieved_ic50 = self.retrieve_similar_compounds(
            protbert_emb, pseq2sites_emb, k=k_retrieve, alpha=alpha
        )
        
        # Step 3: Prepare protein condition for diffusion
        protein_condition = torch.FloatTensor(
            np.concatenate([protbert_emb, pseq2sites_emb])
        ).unsqueeze(0).to(self.device)  # [1, protbert_dim + pseq2sites_dim]
        
        # Step 4: Prepare initial compound embedding
        initial_compound = None
        if use_retrieval_init and len(retrieved_embeddings) > 0:
            # Use mean of top-k retrieved compounds as initialization
            mean_embedding = np.mean(retrieved_embeddings, axis=0)
            initial_compound = torch.FloatTensor(mean_embedding).unsqueeze(0).to(self.device)
            print(f"Using mean of {len(retrieved_embeddings)} retrieved compounds as initialization")
        
        # Step 5: Generate new compound embeddings using diffusion
        print("Running diffusion generation...")
        with torch.no_grad():
            generated_embeddings = self.diffusion_model.sample(
                protein_condition=protein_condition,
                initial_compound=initial_compound,
                num_samples=num_samples
            )
            
        # Step 6: Decode embeddings to SMILES
        print("Decoding embeddings to SMILES...")
        # Keep as torch tensor for smi-TED decoder (it expects torch tensor, not numpy)
        if isinstance(generated_embeddings, torch.Tensor):
            # Move to CPU but keep as tensor for smi-TED
            generated_embeddings_cpu = generated_embeddings.cpu()
        else:
            # Convert numpy to tensor if needed
            generated_embeddings_cpu = torch.from_numpy(generated_embeddings).cpu()
            
        decoded_smiles = self.smi_ted.decode(generated_embeddings_cpu)
        
        # Step 7: Validate and score generated ligands
        results = []
        for i, smiles in enumerate(decoded_smiles):
            try:
                # Validate SMILES
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                    
                # Calculate molecular properties
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)
                
                # Predict IC50
                try:
                    prediction = self.bindingdb_interface.predict(smiles, protein_sequence)
                    predicted_ic50 = prediction['IC50']
                    predicted_ki = prediction['Ki']
                except Exception as e:
                    print(f"IC50 prediction failed for SMILES {i}: {e}")
                    predicted_ic50 = None
                    predicted_ki = None
                    
                result = {
                    'smiles': smiles,
                    'embedding': generated_embeddings_cpu[i].numpy(),  # Convert to numpy for storage
                    'molecular_weight': mw,
                    'logp': logp,
                    'hbd': hbd,
                    'hba': hba,
                    'predicted_ic50': predicted_ic50,
                    'predicted_ki': predicted_ki,
                    'valid': True
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing generated SMILES {i}: {e}")
                results.append({
                    'smiles': smiles,
                    'valid': False,
                    'error': str(e)
                })
                
        # Step 8: Sort by predicted IC50 (lower is better)
        valid_results = [r for r in results if r.get('valid', False) and r.get('predicted_ic50') is not None]
        valid_results.sort(key=lambda x: x['predicted_ic50'])
        
        print(f"Generated {len(valid_results)} valid ligands out of {num_samples}")
        
        if valid_results:
            best_ic50 = valid_results[0]['predicted_ic50']
            print(f"Best predicted IC50: {best_ic50:.4f}")
            
        # Add retrieved compounds for comparison
        comparison_data = {
            'retrieved_smiles': retrieved_smiles,
            'retrieved_ic50': retrieved_ic50,
            'protein_sequence': protein_sequence,
            'generation_params': {
                'num_samples': num_samples,
                'k_retrieve': k_retrieve,
                'alpha': alpha,
                'use_retrieval_init': use_retrieval_init
            }
        }
        
        return valid_results, comparison_data
        
    def batch_generate(self,
                      protein_sequences: List[str],
                      output_path: str,
                      **generation_kwargs) -> Dict:
        """Generate ligands for multiple protein sequences."""
        
        all_results = {}
        
        for i, seq in enumerate(protein_sequences):
            print(f"\nProcessing protein {i+1}/{len(protein_sequences)}")
            
            try:
                results, comparison = self.generate_ligands(seq, **generation_kwargs)
                all_results[f"protein_{i}"] = {
                    'sequence': seq,
                    'generated_ligands': results,
                    'comparison_data': comparison
                }
                
            except Exception as e:
                print(f"Error processing protein {i}: {e}")
                all_results[f"protein_{i}"] = {
                    'sequence': seq,
                    'error': str(e)
                }
                
        # Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(all_results, f)
            
        print(f"Results saved to {output_path}")
        return all_results


def main():
    """Example usage of ligand generator."""
    
    # Initialize generator
    generator = LigandGenerator(
        diffusion_checkpoint="../checkpoints/best_model.pth",
        vector_db_path="../database/train_vector_db/",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Example protein sequence (first 200 residues of a kinase)
    example_sequence = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQHKLRKLNPPDESGPGCMSCKCVLS"
    
    # Generate ligands
    results, comparison = generator.generate_ligands(
        protein_sequence=example_sequence,
        num_samples=10,
        k_retrieve=5,
        alpha=0.6,
        use_retrieval_init=True
    )
    
    # Print results
    print("\n" + "="*50)
    print("GENERATION RESULTS")
    print("="*50)
    
    print(f"\nGenerated {len(results)} valid ligands:")
    for i, result in enumerate(results[:5]):  # Show top 5
        print(f"\n{i+1}. SMILES: {result['smiles']}")
        print(f"   Predicted IC50: {result['predicted_ic50']:.4f}")
        print(f"   MW: {result['molecular_weight']:.1f}, LogP: {result['logp']:.2f}")
        print(f"   HBD: {result['hbd']}, HBA: {result['hba']}")
        
    print(f"\nComparison - Retrieved compounds:")
    for i, (smiles, ic50) in enumerate(zip(comparison['retrieved_smiles'][:3], comparison['retrieved_ic50'][:3])):
        print(f"{i+1}. {smiles} (IC50: {ic50:.4f})")


if __name__ == "__main__":
    main()
