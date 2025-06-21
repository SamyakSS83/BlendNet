"""
Evaluation metrics for generated ligands.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
from scipy.spatial.distance import pdist, squareform
import torch


class LigandEvaluationMetrics:
    """Comprehensive evaluation metrics for generated ligands."""
    
    def __init__(self):
        self.metrics = {}
        
    def compute_validity(self, smiles_list: List[str]) -> Dict[str, float]:
        """
        Compute chemical validity metrics.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Dictionary with validity metrics
        """
        valid_count = 0
        valid_smiles = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_count += 1
                valid_smiles.append(smiles)
                
        validity_rate = valid_count / len(smiles_list) if smiles_list else 0.0
        
        return {
            'validity_rate': validity_rate,
            'valid_count': valid_count,
            'total_count': len(smiles_list),
            'valid_smiles': valid_smiles
        }
    
    def compute_uniqueness(self, smiles_list: List[str]) -> Dict[str, float]:
        """
        Compute uniqueness metrics.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Dictionary with uniqueness metrics
        """
        # Canonicalize SMILES
        canonical_smiles = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                canonical = Chem.MolToSmiles(mol, canonical=True)
                canonical_smiles.append(canonical)
        
        unique_smiles = list(set(canonical_smiles))
        uniqueness_rate = len(unique_smiles) / len(canonical_smiles) if canonical_smiles else 0.0
        
        return {
            'uniqueness_rate': uniqueness_rate,
            'unique_count': len(unique_smiles),
            'valid_count': len(canonical_smiles),
            'unique_smiles': unique_smiles
        }
    
    def compute_diversity(self, smiles_list: List[str]) -> Dict[str, float]:
        """
        Compute diversity metrics using Tanimoto similarity.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Dictionary with diversity metrics
        """
        # Generate fingerprints
        fps = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fps.append(fp)
        
        if len(fps) < 2:
            return {
                'diversity_tanimoto': 0.0,
                'pairwise_similarities': [],
                'mean_similarity': 0.0,
                'std_similarity': 0.0
            }
        
        # Compute pairwise similarities
        similarities = []
        for i in range(len(fps)):
            for j in range(i+1, len(fps)):
                sim = TanimotoSimilarity(fps[i], fps[j])
                similarities.append(sim)
        
        # Diversity is 1 - mean similarity
        mean_sim = np.mean(similarities)
        diversity = 1.0 - mean_sim
        
        return {
            'diversity_tanimoto': diversity,
            'pairwise_similarities': similarities,
            'mean_similarity': mean_sim,
            'std_similarity': np.std(similarities)
        }
    
    def compute_drug_likeness(self, smiles_list: List[str]) -> Dict[str, float]:
        """
        Compute drug-likeness metrics (Lipinski's Rule of Five, etc.).
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Dictionary with drug-likeness metrics
        """
        lipinski_pass = 0
        qed_scores = []
        molecular_weights = []
        logp_values = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Lipinski's Rule of Five
                mw = Descriptors.MolWt(mol)
                logp = Crippen.MolLogP(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)
                
                if (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10):
                    lipinski_pass += 1
                
                # QED score
                qed_score = QED.qed(mol)
                qed_scores.append(qed_score)
                
                molecular_weights.append(mw)
                logp_values.append(logp)
        
        lipinski_rate = lipinski_pass / len(smiles_list) if smiles_list else 0.0
        
        return {
            'lipinski_pass_rate': lipinski_rate,
            'mean_qed': np.mean(qed_scores) if qed_scores else 0.0,
            'std_qed': np.std(qed_scores) if qed_scores else 0.0,
            'mean_molecular_weight': np.mean(molecular_weights) if molecular_weights else 0.0,
            'mean_logp': np.mean(logp_values) if logp_values else 0.0,
            'qed_scores': qed_scores
        }
    
    def compute_binding_affinity_metrics(self, 
                                       smiles_list: List[str], 
                                       ic50_predictions: List[float]) -> Dict[str, float]:
        """
        Compute binding affinity metrics.
        
        Args:
            smiles_list: List of SMILES strings
            ic50_predictions: List of predicted IC50 values
            
        Returns:
            Dictionary with binding affinity metrics
        """
        if not ic50_predictions:
            return {
                'mean_ic50': 0.0,
                'std_ic50': 0.0,
                'min_ic50': 0.0,
                'max_ic50': 0.0,
                'high_affinity_count': 0,
                'high_affinity_rate': 0.0
            }
        
        # Convert to numpy for easier computation
        ic50_array = np.array(ic50_predictions)
        
        # Count high-affinity compounds (IC50 < 100 nM)
        high_affinity_count = np.sum(ic50_array < 100)
        high_affinity_rate = high_affinity_count / len(ic50_predictions)
        
        return {
            'mean_ic50': np.mean(ic50_array),
            'std_ic50': np.std(ic50_array),
            'min_ic50': np.min(ic50_array),
            'max_ic50': np.max(ic50_array),
            'high_affinity_count': high_affinity_count,
            'high_affinity_rate': high_affinity_rate
        }
    
    def compute_novelty(self, 
                       generated_smiles: List[str], 
                       reference_smiles: List[str]) -> Dict[str, float]:
        """
        Compute novelty metrics by comparing against reference dataset.
        
        Args:
            generated_smiles: List of generated SMILES
            reference_smiles: List of reference SMILES (training data)
            
        Returns:
            Dictionary with novelty metrics
        """
        # Canonicalize all SMILES
        gen_canonical = set()
        for smiles in generated_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                canonical = Chem.MolToSmiles(mol, canonical=True)
                gen_canonical.add(canonical)
        
        ref_canonical = set()
        for smiles in reference_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                canonical = Chem.MolToSmiles(mol, canonical=True)
                ref_canonical.add(canonical)
        
        # Compute novelty
        novel_compounds = gen_canonical - ref_canonical
        novelty_rate = len(novel_compounds) / len(gen_canonical) if gen_canonical else 0.0
        
        return {
            'novelty_rate': novelty_rate,
            'novel_count': len(novel_compounds),
            'generated_unique_count': len(gen_canonical),
            'novel_smiles': list(novel_compounds)
        }
    
    def compute_comprehensive_metrics(self, 
                                    generated_smiles: List[str],
                                    ic50_predictions: Optional[List[float]] = None,
                                    reference_smiles: Optional[List[str]] = None) -> Dict[str, any]:
        """
        Compute all evaluation metrics comprehensively.
        
        Args:
            generated_smiles: List of generated SMILES
            ic50_predictions: Optional list of IC50 predictions
            reference_smiles: Optional reference dataset for novelty computation
            
        Returns:
            Dictionary with all metrics
        """
        print("Computing comprehensive evaluation metrics...")
        
        # Basic validity and uniqueness
        validity_metrics = self.compute_validity(generated_smiles)
        valid_smiles = validity_metrics['valid_smiles']
        
        uniqueness_metrics = self.compute_uniqueness(valid_smiles)
        unique_smiles = uniqueness_metrics['unique_smiles']
        
        # Diversity
        diversity_metrics = self.compute_diversity(unique_smiles)
        
        # Drug-likeness
        drug_likeness_metrics = self.compute_drug_likeness(unique_smiles)
        
        # Binding affinity (if predictions available)
        binding_metrics = {}
        if ic50_predictions:
            binding_metrics = self.compute_binding_affinity_metrics(
                generated_smiles, ic50_predictions
            )
        
        # Novelty (if reference available)
        novelty_metrics = {}
        if reference_smiles:
            novelty_metrics = self.compute_novelty(unique_smiles, reference_smiles)
        
        # Combine all metrics
        all_metrics = {
            'validity': validity_metrics,
            'uniqueness': uniqueness_metrics,
            'diversity': diversity_metrics,
            'drug_likeness': drug_likeness_metrics,
            'binding_affinity': binding_metrics,
            'novelty': novelty_metrics,
            'summary': {
                'total_generated': len(generated_smiles),
                'valid_count': validity_metrics['valid_count'],
                'unique_count': uniqueness_metrics['unique_count'],
                'validity_rate': validity_metrics['validity_rate'],
                'uniqueness_rate': uniqueness_metrics['uniqueness_rate'],
                'diversity_score': diversity_metrics['diversity_tanimoto'],
                'lipinski_pass_rate': drug_likeness_metrics['lipinski_pass_rate'],
                'mean_qed': drug_likeness_metrics['mean_qed']
            }
        }
        
        # Add binding affinity to summary if available
        if ic50_predictions:
            all_metrics['summary']['mean_ic50'] = binding_metrics['mean_ic50']
            all_metrics['summary']['high_affinity_rate'] = binding_metrics['high_affinity_rate']
        
        # Add novelty to summary if available
        if reference_smiles:
            all_metrics['summary']['novelty_rate'] = novelty_metrics['novelty_rate']
        
        return all_metrics
    
    def print_summary(self, metrics: Dict[str, any]) -> None:
        """Print a summary of evaluation metrics."""
        print("\n" + "="*60)
        print("LIGAND GENERATION EVALUATION SUMMARY")
        print("="*60)
        
        summary = metrics['summary']
        
        print(f"Total Generated:     {summary['total_generated']}")
        print(f"Valid Compounds:     {summary['valid_count']} ({summary['validity_rate']:.3f})")
        print(f"Unique Compounds:    {summary['unique_count']} ({summary['uniqueness_rate']:.3f})")
        print(f"Diversity Score:     {summary['diversity_score']:.3f}")
        print(f"Lipinski Pass Rate:  {summary['lipinski_pass_rate']:.3f}")
        print(f"Mean QED Score:      {summary['mean_qed']:.3f}")
        
        if 'mean_ic50' in summary:
            print(f"Mean IC50:           {summary['mean_ic50']:.2f} nM")
            print(f"High Affinity Rate:  {summary['high_affinity_rate']:.3f}")
        
        if 'novelty_rate' in summary:
            print(f"Novelty Rate:        {summary['novelty_rate']:.3f}")
        
        print("="*60)


def evaluate_ligand_generation(generated_smiles: List[str],
                             ic50_predictions: Optional[List[float]] = None,
                             reference_smiles: Optional[List[str]] = None,
                             print_summary: bool = True) -> Dict[str, any]:
    """
    Convenience function to evaluate generated ligands.
    
    Args:
        generated_smiles: List of generated SMILES
        ic50_predictions: Optional list of IC50 predictions
        reference_smiles: Optional reference dataset
        print_summary: Whether to print summary
        
    Returns:
        Dictionary with all evaluation metrics
    """
    evaluator = LigandEvaluationMetrics()
    metrics = evaluator.compute_comprehensive_metrics(
        generated_smiles, ic50_predictions, reference_smiles
    )
    
    if print_summary:
        evaluator.print_summary(metrics)
    
    return metrics
