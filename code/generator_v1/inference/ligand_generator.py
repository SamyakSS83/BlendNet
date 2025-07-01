"""
Ligand generation module for protein-conditioned SMILES generation.

This module implements the ligand generation pipeline according to idea.md:
- Retrieval-augmented diffusion using top-k similar proteins
- Protein embedding generation (ProtBERT + Pseq2Sites)
- SMILES validation and IC50 prediction
- Modular and compliant with the overall architecture
"""

import os
import sys
import torch
import numpy as np
import faiss
from typing import List, Optional, Dict, Tuple, Any
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors
import random

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../materials.smi-ted/smi-ted/'))

try:
    from models.diffusion_model import ProteinLigandDiffusion
    from preprocess.data_preprocessor import DataPreprocessor
except ImportError as e:
    logging.warning(f"Import error: {e}")

# Import SMILES validator
try:
    from utils.smiles_validator import SMILESValidator
except ImportError as e:
    logging.warning(f"SMILES validator import error: {e}")
    SMILESValidator = None

# Import smi-TED with correct path
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../materials.smi-ted/smi-ted/inference/'))
    from smi_ted_light.load import load_smi_ted
except ImportError as e:
    logging.warning(f"smi-TED import error: {e}")
    load_smi_ted = None

logger = logging.getLogger(__name__)


class LigandGenerator:
    """
    Complete pipeline for protein-conditioned ligand generation.
    
    Implements retrieval-augmented diffusion according to idea.md:
    - Uses FAISS index for efficient protein similarity search
    - Retrieves top-k similar proteins and their ligands
    - Generates ligands using diffusion model with protein conditioning
    """
    
    def __init__(self,
                 config: Dict[str, Any],
                 protein_database: List[Dict[str, Any]],
                 protein_sequences: List[str],
                 faiss_index: faiss.Index,
                 protein_embeddings: Dict[str, np.ndarray],
                 device: str = "cuda"):
        """
        Initialize ligand generator.
        
        Args:
            config: Model configuration dictionary
            protein_database: List of protein data entries
            protein_sequences: List of protein sequences
            faiss_index: FAISS index for similarity search
            protein_embeddings: Dict with 'protbert' and 'pseq2sites' embeddings
            device: Device for computation
        """
        self.device = device
        self.config = config
        self.protein_database = protein_database
        self.protein_sequences = protein_sequences
        self.faiss_index = faiss_index
        self.protein_embeddings = protein_embeddings
        
        logger.info("Initializing LigandGenerator...")
        
        # Initialize diffusion model
        self.model = ProteinLigandDiffusion(
            compound_dim=config['compound_dim'],
            protbert_dim=config['protbert_dim'],
            pseq2sites_dim=config['pseq2sites_dim'],
            num_timesteps=config['num_timesteps'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(device)
        
        # Initialize other components
        self._initialize_models()
        
        logger.info("LigandGenerator initialized successfully!")
        
    def _initialize_models(self):
        """Initialize smi-TED and other models."""
        try:
            # Load smi-TED
            if load_smi_ted is not None:
                # CORRECTED: Use root materials.smi-ted directory, not subdirectory
                # The checkpoint file is in the root directory
                smi_ted_path = '/home/sarvesh/scratch/GS/samyak/.Blendnet/materials.smi-ted'
                self.smi_ted = load_smi_ted(
                    folder=smi_ted_path,
                    ckpt_filename="smi-ted-Light_40.pt"
                )
                logger.info("✅ smi-TED loaded successfully")
            else:
                logger.warning("smi-TED load function not available")
                self.smi_ted = None
        except Exception as e:
            logger.warning(f"Failed to load smi-TED: {e}")
            self.smi_ted = None
            
        try:
            # For inference, we don't need the full DataPreprocessor
            # We'll use direct protein embedding generation
            from transformers import BertTokenizer, BertModel
            
            # Initialize ProtBERT
            self.protbert_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
            self.protbert_model = BertModel.from_pretrained("Rostlab/prot_bert").to(self.device)
            self.protbert_model.eval()
            
            # Initialize Pseq2Sites
            sys.path.append('../../')  # Add root path for modules
            from modules.pocket_modules.pseq2sites_embeddings import Pseq2SitesEmbeddings
            self.pseq2sites = Pseq2SitesEmbeddings(device=self.device)
            
            logger.info("✅ Protein embedding models initialized")
            self.data_preprocessor = True  # Mark as available
        except Exception as e:
            logger.warning(f"Failed to initialize protein embedding models: {e}")
            self.data_preprocessor = None
            
        # Initialize SMILES validator
        if SMILESValidator is not None:
            self.smiles_validator = SMILESValidator()
            logger.info("✅ SMILES validator initialized")
        else:
            logger.warning("SMILES validator not available")
            self.smiles_validator = None
        
    def generate_protein_embeddings(self, protein_sequence: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate protein embeddings for input sequence.
        
        Args:
            protein_sequence: Input protein sequence
            
        Returns:
            Tuple of (protbert_embedding, pseq2sites_embedding)
        """
        logger.info("Generating protein embeddings...")
        
        if self.data_preprocessor is None:
            raise RuntimeError("Protein embedding models not initialized")
            
        try:
            # Generate ProtBERT embedding
            with torch.no_grad():
                # Add spaces between amino acids for ProtBERT
                spaced_sequence = ' '.join(list(protein_sequence))
                
                # Tokenize and encode
                inputs = self.protbert_tokenizer(
                    spaced_sequence, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=1024
                ).to(self.device)
                
                # Get ProtBERT embedding
                outputs = self.protbert_model(**inputs)
                protbert_emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                
            # Generate Pseq2Sites embedding
            with torch.no_grad():
                pseq2sites_emb = self.pseq2sites.get_embeddings([protein_sequence])[0].cpu().numpy()
            
            logger.info(f"✅ Generated embeddings: ProtBERT {protbert_emb.shape}, Pseq2Sites {pseq2sites_emb.shape}")
            return protbert_emb, pseq2sites_emb
            
        except Exception as e:
            logger.error(f"Failed to generate protein embeddings: {e}")
            raise
            
    def retrieve_similar_proteins(self,
                                protbert_emb: np.ndarray,
                                pseq2sites_emb: np.ndarray,
                                k: int = 5,
                                alpha: float = 0.5) -> Tuple[List[str], List[Dict]]:
        """
        Retrieve top-k similar proteins and their ligands.
        
        Args:
            protbert_emb: ProtBERT embedding of query protein
            pseq2sites_emb: Pseq2Sites embedding of query protein
            k: Number of similar proteins to retrieve
            alpha: Weight for combining similarity scores
            
        Returns:
            Tuple of (similar_sequences, similar_protein_data)
        """
        logger.info(f"Retrieving top-{k} similar proteins...")
        
        # Combine embeddings with weighting according to idea.md
        # sim = alpha * sim(pseq2sites) + (1-alpha) * sim(protbert)
        combined_query = np.concatenate([
            alpha * pseq2sites_emb,
            (1 - alpha) * protbert_emb
        ])
        
        # Search FAISS index
        scores, indices = self.faiss_index.search(
            combined_query.reshape(1, -1).astype('float32'), k
        )
        
        # Get corresponding protein data
        similar_sequences = []
        similar_protein_data = []
        
        for idx in indices[0]:
            if idx < len(self.protein_sequences):
                seq = self.protein_sequences[idx]
                protein_data = self.protein_database[idx]
                
                similar_sequences.append(seq)
                similar_protein_data.append(protein_data)
                
        logger.info(f"✅ Retrieved {len(similar_sequences)} similar proteins")
        
        return similar_sequences, similar_protein_data
        
    def select_initialization_ligand(self, similar_protein_data: List[Dict]) -> str:
        """
        Randomly select one ligand from top-k similar proteins.
        
        According to idea.md: "use any 1 of the m smiles for that protein randomly"
        
        Args:
            similar_protein_data: List of protein data dictionaries
            
        Returns:
            Selected SMILES string
        """
        # Collect all ligands from similar proteins
        all_ligands = []
        for protein_data in similar_protein_data:
            ligands = protein_data.get('ligands', [])
            all_ligands.extend(ligands)
            
        if not all_ligands:
            logger.warning("No ligands found in similar proteins")
            return None
            
        # Randomly select one ligand
        selected_ligand = random.choice(all_ligands)
        smiles = selected_ligand.get('smiles', '')
        
        logger.info(f"Selected initialization ligand: {smiles}")
        return smiles
        
    def encode_smiles_to_embedding(self, smiles: str) -> np.ndarray:
        """
        Encode SMILES to smi-TED embedding.
        
        Args:
            smiles: SMILES string
            
        Returns:
            smi-TED embedding
        """
        if self.smi_ted is None:
            raise RuntimeError("smi-TED model not loaded")
            
        try:
            # Encode using smi-TED
            embedding = self.smi_ted.encode([smiles])
            
            # Handle different output types from smi-TED encode
            if hasattr(embedding, 'values'):  # DataFrame
                embedding = embedding.values
            elif isinstance(embedding, torch.Tensor):
                embedding = embedding.cpu().numpy()
            # If already numpy array, keep as is
            
            return embedding[0]  # Return single embedding
            
        except Exception as e:
            logger.error(f"Failed to encode SMILES '{smiles}': {e}")
            raise
        
    def generate_ligands(self,
                        protein_sequence: str,
                        num_samples: int = 5,
                        k_similar: int = 5,
                        guidance_scale: float = 1.0,
                        num_inference_steps: int = 100,
                        filter_invalid: bool = True,
                        filter_nonorganic: bool = True,
                        predict_ic50: bool = False) -> Dict[str, Any]:
        """
        Generate ligands for a given protein sequence.
        
        Args:
            protein_sequence: Target protein sequence
            num_samples: Number of ligands to generate
            k_similar: Number of similar proteins to retrieve
            guidance_scale: Guidance scale for conditioning
            num_inference_steps: Number of denoising steps
            filter_invalid: Whether to filter invalid SMILES
            filter_nonorganic: Whether to filter non-organic molecules
            predict_ic50: Whether to predict IC50 values
            
        Returns:
            Dictionary with generated ligands and metadata
        """
        logger.info(f"Generating {num_samples} ligands for protein sequence...")
        logger.info(f"Protein length: {len(protein_sequence)}")
        
        results = {
            'ligands': [],
            'protein_sequence': protein_sequence,
            'generation_params': {
                'num_samples': num_samples,
                'k_similar': k_similar,
                'guidance_scale': guidance_scale,
                'num_inference_steps': num_inference_steps
            },
            'filtered_count': 0
        }
        
        try:
            # Step 1: Generate protein embeddings
            logger.info("Step 1: Generating protein embeddings...")
            protbert_emb, pseq2sites_emb = self.generate_protein_embeddings(protein_sequence)
            
            # Step 2: Retrieve similar proteins
            logger.info("Step 2: Retrieving similar proteins...")
            similar_sequences, similar_protein_data = self.retrieve_similar_proteins(
                protbert_emb, pseq2sites_emb, k=k_similar
            )
            
            # Step 3: Select initialization ligand
            logger.info("Step 3: Selecting initialization ligand...")
            init_smiles = self.select_initialization_ligand(similar_protein_data)
            
            if init_smiles is None:
                logger.warning("No initialization ligand found, using random initialization")
                init_embedding = None
            else:
                init_embedding = self.encode_smiles_to_embedding(init_smiles)
                logger.info(f"Using initialization ligand: {init_smiles}")
            
            # Step 4: Prepare conditioning
            logger.info("Step 4: Preparing diffusion conditioning...")
            protein_condition = torch.FloatTensor(
                np.concatenate([protbert_emb, pseq2sites_emb])
            ).unsqueeze(0).to(self.device)
            
            if init_embedding is not None:
                init_compound = torch.FloatTensor(init_embedding).unsqueeze(0).to(self.device)
            else:
                init_compound = None
            
            # Step 5: Generate compound embeddings
            logger.info("Step 5: Running diffusion generation...")
            with torch.no_grad():
                generated_embeddings = self.model.sample(
                    protein_condition=protein_condition,
                    initial_compound=init_compound,
                    num_samples=num_samples,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                )
            
            # Step 6: Decode to SMILES
            logger.info("Step 6: Decoding embeddings to SMILES...")
            if self.smi_ted is None:
                raise RuntimeError("smi-TED not loaded for decoding")
                
            if isinstance(generated_embeddings, torch.Tensor):
                generated_embeddings_cpu = generated_embeddings.cpu()
            else:
                generated_embeddings_cpu = torch.from_numpy(generated_embeddings)
                
            decoded_smiles = self.smi_ted.decode(generated_embeddings_cpu)
            
            # Step 7: Process and validate results
            logger.info("Step 7: Processing and validating results...")
            for i, smiles in enumerate(decoded_smiles):
                try:
                    ligand_data = {
                        'smiles': smiles,
                        'index': i,
                        'embedding': generated_embeddings_cpu[i].numpy() if isinstance(generated_embeddings_cpu, torch.Tensor) else generated_embeddings_cpu[i]
                    }
                    
                    # Validate SMILES
                    if filter_invalid and self.smiles_validator is not None:
                        if not self.smiles_validator.is_valid(smiles):
                            results['filtered_count'] += 1
                            continue
                            
                    # Check if organic
                    if filter_nonorganic and self.smiles_validator is not None:
                        if not self.smiles_validator.is_organic(smiles):
                            results['filtered_count'] += 1
                            continue
                    
                    # Calculate molecular properties
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        ligand_data.update({
                            'molecular_weight': Descriptors.MolWt(mol),
                            'logp': Descriptors.MolLogP(mol),
                            'hbd': Descriptors.NumHDonors(mol),
                            'hba': Descriptors.NumHAcceptors(mol)
                        })
                    
                    # Predict IC50 if requested
                    if predict_ic50:
                        try:
                            # Placeholder for IC50 prediction
                            # In a full implementation, this would use BlendNet
                            ligand_data['predicted_ic50'] = None
                            logger.debug(f"IC50 prediction not implemented yet")
                        except Exception as e:
                            logger.warning(f"IC50 prediction failed for {smiles}: {e}")
                            ligand_data['predicted_ic50'] = None
                    
                    ligand_data['valid'] = True
                    results['ligands'].append(ligand_data)
                    
                except Exception as e:
                    logger.warning(f"Error processing SMILES {i} ({smiles}): {e}")
                    results['filtered_count'] += 1
                    continue
            
            # Sort by molecular weight (or other criteria)
            results['ligands'].sort(key=lambda x: x.get('molecular_weight', float('inf')))
            
            logger.info(f"✅ Generated {len(results['ligands'])} valid ligands")
            logger.info(f"📊 Filtered out {results['filtered_count']} invalid/unwanted molecules")
            
            return results
            
        except Exception as e:
            logger.error(f"Ligand generation failed: {e}")
            raise


def main():
    """Example usage of ligand generator."""
    
    # This is a placeholder main function for testing
    print("LigandGenerator module loaded successfully!")
    print("Use the LigandGenerator class for protein-conditioned ligand generation.")
    

if __name__ == "__main__":
    main()
