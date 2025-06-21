"""
Benchmark script for protein-ligand diffusion model evaluation.
"""
import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import json

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../materials.smi-ted/smi-ted/'))

from inference.ligand_generator import LigandGenerator
from evaluation.metrics import evaluate_ligand_generation


class ProteinLigandBenchmark:
    """Benchmark class for evaluating protein-ligand diffusion model."""
    
    def __init__(self, 
                 generator: LigandGenerator,
                 test_data_path: str,
                 output_dir: str = "./benchmark_results"):
        """
        Initialize benchmark.
        
        Args:
            generator: Trained ligand generator
            test_data_path: Path to test dataset
            output_dir: Directory to save results
        """
        self.generator = generator
        self.test_data_path = test_data_path
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load test data
        print(f"Loading test data from {test_data_path}")
        self.test_data = pd.read_csv(test_data_path, sep='\t')
        print(f"Loaded {len(self.test_data)} test samples")
    
    def run_single_protein_benchmark(self, 
                                   protein_sequence: str,
                                   reference_smiles: str,
                                   reference_ic50: Optional[float] = None,
                                   num_generations: int = 10,
                                   top_k: int = 5) -> Dict:
        """
        Run benchmark for a single protein.
        
        Args:
            protein_sequence: Input protein sequence
            reference_smiles: Known ligand SMILES for comparison
            reference_ic50: Known IC50 value
            num_generations: Number of ligands to generate
            top_k: Number of retrieved compounds for conditioning
            
        Returns:
            Dictionary with benchmark results
        """
        print(f"Benchmarking protein (length: {len(protein_sequence)})")
        
        start_time = time.time()
        
        # Generate ligands
        generated_ligands = self.generator.generate_ligands(
            protein_sequence=protein_sequence,
            num_samples=num_generations,
            top_k_retrieve=top_k,
            num_inference_steps=50
        )
        
        generation_time = time.time() - start_time
        
        # Extract generated SMILES and predicted IC50s
        generated_smiles = [ligand['smiles'] for ligand in generated_ligands]
        predicted_ic50s = [ligand['predicted_ic50'] for ligand in generated_ligands]
        
        # Evaluate generated ligands
        evaluation_metrics = evaluate_ligand_generation(
            generated_smiles=generated_smiles,
            ic50_predictions=predicted_ic50s,
            reference_smiles=[reference_smiles] if reference_smiles else None,
            print_summary=False
        )
        
        # Additional metrics
        best_ic50 = min(predicted_ic50s) if predicted_ic50s else float('inf')
        improvement_over_reference = None
        if reference_ic50 and best_ic50 != float('inf'):
            improvement_over_reference = reference_ic50 / best_ic50  # >1 means improvement
        
        return {
            'protein_length': len(protein_sequence),
            'generation_time_seconds': generation_time,
            'num_generated': len(generated_smiles),
            'generated_smiles': generated_smiles,
            'predicted_ic50s': predicted_ic50s,
            'best_predicted_ic50': best_ic50,
            'reference_smiles': reference_smiles,
            'reference_ic50': reference_ic50,
            'improvement_factor': improvement_over_reference,
            'evaluation_metrics': evaluation_metrics,
            'top_ligands': generated_ligands[:3]  # Top 3 by IC50
        }
    
    def run_benchmark_suite(self, 
                          num_proteins: int = 50,
                          num_generations_per_protein: int = 10,
                          top_k: int = 5) -> Dict:
        """
        Run comprehensive benchmark on multiple proteins.
        
        Args:
            num_proteins: Number of proteins to test
            num_generations_per_protein: Number of ligands per protein
            top_k: Number of retrieved compounds for conditioning
            
        Returns:
            Dictionary with comprehensive benchmark results
        """
        print(f"Running benchmark suite on {num_proteins} proteins...")
        
        # Sample test proteins
        test_proteins = self.test_data.sample(n=min(num_proteins, len(self.test_data)))
        
        benchmark_results = []
        total_start_time = time.time()
        
        for idx, row in test_proteins.iterrows():
            protein_seq = row.get('ProteinSequence', row.get('protein_sequence', ''))
            reference_smiles = row.get('SMILES', row.get('smiles', ''))
            reference_ic50 = row.get('IC50', row.get('ic50', None))
            
            if not protein_seq or not reference_smiles:
                print(f"Skipping row {idx} due to missing data")
                continue
            
            print(f"\\nBenchmarking protein {len(benchmark_results)+1}/{num_proteins}")
            
            try:
                result = self.run_single_protein_benchmark(
                    protein_sequence=protein_seq,
                    reference_smiles=reference_smiles,
                    reference_ic50=reference_ic50,
                    num_generations=num_generations_per_protein,
                    top_k=top_k
                )
                
                result['protein_id'] = idx
                benchmark_results.append(result)
                
            except Exception as e:
                print(f"Error benchmarking protein {idx}: {e}")
                continue
        
        total_time = time.time() - total_start_time
        
        # Aggregate results
        aggregate_metrics = self._aggregate_results(benchmark_results)
        
        final_results = {
            'benchmark_config': {
                'num_proteins_requested': num_proteins,
                'num_proteins_completed': len(benchmark_results),
                'num_generations_per_protein': num_generations_per_protein,
                'top_k_retrieval': top_k,
                'total_time_seconds': total_time
            },
            'individual_results': benchmark_results,
            'aggregate_metrics': aggregate_metrics
        }
        
        return final_results
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate individual benchmark results."""
        if not results:
            return {}
        
        # Extract metrics
        generation_times = [r['generation_time_seconds'] for r in results]
        validity_rates = [r['evaluation_metrics']['summary']['validity_rate'] for r in results]
        uniqueness_rates = [r['evaluation_metrics']['summary']['uniqueness_rate'] for r in results]
        diversity_scores = [r['evaluation_metrics']['summary']['diversity_score'] for r in results]
        lipinski_rates = [r['evaluation_metrics']['summary']['lipinski_pass_rate'] for r in results]
        qed_scores = [r['evaluation_metrics']['summary']['mean_qed'] for r in results]
        best_ic50s = [r['best_predicted_ic50'] for r in results if r['best_predicted_ic50'] != float('inf')]
        
        # Improvement factors (where available)
        improvements = [r['improvement_factor'] for r in results if r['improvement_factor'] is not None]
        
        # High affinity rates
        high_affinity_rates = []
        for r in results:
            if 'high_affinity_rate' in r['evaluation_metrics']['summary']:
                high_affinity_rates.append(r['evaluation_metrics']['summary']['high_affinity_rate'])
        
        aggregate = {
            'performance_metrics': {
                'mean_generation_time': np.mean(generation_times),
                'std_generation_time': np.std(generation_times),
                'total_proteins_tested': len(results)
            },
            'chemical_validity': {
                'mean_validity_rate': np.mean(validity_rates),
                'std_validity_rate': np.std(validity_rates),
                'mean_uniqueness_rate': np.mean(uniqueness_rates),
                'std_uniqueness_rate': np.std(uniqueness_rates)
            },
            'molecular_diversity': {
                'mean_diversity_score': np.mean(diversity_scores),
                'std_diversity_score': np.std(diversity_scores)
            },
            'drug_likeness': {
                'mean_lipinski_rate': np.mean(lipinski_rates),
                'std_lipinski_rate': np.std(lipinski_rates),
                'mean_qed_score': np.mean(qed_scores),
                'std_qed_score': np.std(qed_scores)
            },
            'binding_affinity': {
                'mean_best_ic50': np.mean(best_ic50s) if best_ic50s else None,
                'std_best_ic50': np.std(best_ic50s) if best_ic50s else None,
                'median_best_ic50': np.median(best_ic50s) if best_ic50s else None,
                'mean_high_affinity_rate': np.mean(high_affinity_rates) if high_affinity_rates else None
            }
        }
        
        # Add improvement metrics if available
        if improvements:
            aggregate['improvement_over_reference'] = {
                'mean_improvement_factor': np.mean(improvements),
                'std_improvement_factor': np.std(improvements),
                'proteins_with_improvement': sum(1 for imp in improvements if imp > 1.0),
                'improvement_rate': sum(1 for imp in improvements if imp > 1.0) / len(improvements)
            }
        
        return aggregate
    
    def save_results(self, results: Dict, filename: str = "benchmark_results.json"):
        """Save benchmark results to file."""
        output_path = os.path.join(self.output_dir, filename)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        def deep_convert(obj):
            if isinstance(obj, dict):
                return {k: deep_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deep_convert(v) for v in obj]
            else:
                return convert_numpy(obj)
        
        results_serializable = deep_convert(results)
        
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"Results saved to {output_path}")
    
    def print_summary(self, results: Dict):
        """Print benchmark summary."""
        print("\\n" + "="*80)
        print("PROTEIN-LIGAND DIFFUSION MODEL BENCHMARK SUMMARY")
        print("="*80)
        
        config = results['benchmark_config']
        aggregate = results['aggregate_metrics']
        
        print(f"Proteins Tested:     {config['num_proteins_completed']}/{config['num_proteins_requested']}")
        print(f"Total Time:          {config['total_time_seconds']:.1f} seconds")
        print(f"Avg Time/Protein:    {aggregate['performance_metrics']['mean_generation_time']:.2f} ± {aggregate['performance_metrics']['std_generation_time']:.2f} seconds")
        
        print("\\nChemical Validity:")
        print(f"  Validity Rate:     {aggregate['chemical_validity']['mean_validity_rate']:.3f} ± {aggregate['chemical_validity']['std_validity_rate']:.3f}")
        print(f"  Uniqueness Rate:   {aggregate['chemical_validity']['mean_uniqueness_rate']:.3f} ± {aggregate['chemical_validity']['std_uniqueness_rate']:.3f}")
        
        print("\\nMolecular Properties:")
        print(f"  Diversity Score:   {aggregate['molecular_diversity']['mean_diversity_score']:.3f} ± {aggregate['molecular_diversity']['std_diversity_score']:.3f}")
        print(f"  Lipinski Rate:     {aggregate['drug_likeness']['mean_lipinski_rate']:.3f} ± {aggregate['drug_likeness']['std_lipinski_rate']:.3f}")
        print(f"  Mean QED:          {aggregate['drug_likeness']['mean_qed_score']:.3f} ± {aggregate['drug_likeness']['std_qed_score']:.3f}")
        
        if aggregate['binding_affinity']['mean_best_ic50'] is not None:
            print("\\nBinding Affinity:")
            print(f"  Mean Best IC50:    {aggregate['binding_affinity']['mean_best_ic50']:.2f} ± {aggregate['binding_affinity']['std_best_ic50']:.2f} nM")
            print(f"  Median Best IC50:  {aggregate['binding_affinity']['median_best_ic50']:.2f} nM")
            if aggregate['binding_affinity']['mean_high_affinity_rate'] is not None:
                print(f"  High Affinity Rate: {aggregate['binding_affinity']['mean_high_affinity_rate']:.3f}")
        
        if 'improvement_over_reference' in aggregate:
            improvement = aggregate['improvement_over_reference']
            print("\\nImprovement over Reference:")
            print(f"  Mean Improvement:  {improvement['mean_improvement_factor']:.2f}x")
            print(f"  Improvement Rate:  {improvement['improvement_rate']:.3f}")
            print(f"  Proteins Improved: {improvement['proteins_with_improvement']}")
        
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark protein-ligand diffusion model")
    parser.add_argument("--diffusion_checkpoint", required=True, help="Path to trained diffusion model")
    parser.add_argument("--vector_db", required=True, help="Path to vector database")
    parser.add_argument("--test_data", required=True, help="Path to test dataset")
    parser.add_argument("--output_dir", default="./benchmark_results", help="Output directory")
    parser.add_argument("--num_proteins", type=int, default=50, help="Number of proteins to test")
    parser.add_argument("--num_generations", type=int, default=10, help="Number of ligands per protein")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k retrieval")
    parser.add_argument("--device", default="cuda", help="Device for computation")
    
    args = parser.parse_args()
    
    # Initialize generator
    print("Initializing ligand generator...")
    generator = LigandGenerator(
        diffusion_checkpoint=args.diffusion_checkpoint,
        vector_db_path=args.vector_db,
        device=args.device
    )
    
    # Initialize benchmark
    benchmark = ProteinLigandBenchmark(
        generator=generator,
        test_data_path=args.test_data,
        output_dir=args.output_dir
    )
    
    # Run benchmark
    results = benchmark.run_benchmark_suite(
        num_proteins=args.num_proteins,
        num_generations_per_protein=args.num_generations,
        top_k=args.top_k
    )
    
    # Save and print results
    benchmark.save_results(results)
    benchmark.print_summary(results)


if __name__ == "__main__":
    main()
