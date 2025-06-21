# Protein-Ligand Diffusion Generator v1

A modular, efficient, and retrieval-augmented diffusion model pipeline for protein-specific ligand generation. Given a protein sequence, this system generates optimal ligands (SMILES) predicted to bind with high affinity using protein embeddings (ProtBERT, Pseq2Sites), compound embeddings (smi-TED), and binding affinity prediction (BlendNet IC50).

## Architecture Overview

The pipeline consists of six main components:

1. **Data Preprocessing** (`preprocess/`): Generates protein and compound embeddings
2. **Vector Database** (`database/`): FAISS-based similarity search for retrieval
3. **Diffusion Model** (`models/`): Protein-conditioned diffusion model
4. **Training** (`training/`): Training loop with IC50-based regularization  
5. **Inference** (`inference/`): End-to-end ligand generation pipeline
6. **Evaluation** (`evaluation/`): Comprehensive metrics for generated ligands

## Quick Start

### 1. Installation

```bash
# Clone repository and navigate to generator directory
cd /home/threesamyak/sura/plm_sura/BlendNet/code/generator_v1

# Install dependencies
pip install -r requirements.txt

# Install additional dependencies for your system
# For GPU support:
pip install faiss-gpu torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Run Complete Pipeline

```bash
# Run the complete pipeline example
python example_pipeline.py --mode full

# Or run a quick example (after full pipeline has been run once)
python example_pipeline.py --mode quick
```

### 3. Generate Ligands for Your Protein

```python
from inference.ligand_generator import LigandGenerator

# Initialize generator (assumes trained model exists)
generator = LigandGenerator(
    diffusion_checkpoint='./trained_models/diffusion_model.pth',
    vector_db_path='./preprocessed_data/vector_database',
    device='cuda'
)

# Your protein sequence
protein_sequence = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEE..."

# Generate ligands
ligands = generator.generate_ligands(
    protein_sequence=protein_sequence,
    num_samples=10,
    top_k_retrieve=5,
    num_inference_steps=50
)

# Print results
for i, ligand in enumerate(ligands):
    print(f"{i+1}. SMILES: {ligand['smiles']}")
    print(f"   Predicted IC50: {ligand['predicted_ic50']:.2f} nM")
```

## Detailed Usage

### Data Preprocessing

```python
from preprocess.data_preprocessor import DataPreprocessor

preprocessor = DataPreprocessor(
    ic50_data_path='/path/to/IC50_data.tsv',
    smi_ted_path='../../materials.smi-ted/smi-ted/inference/smi_ted_light',
    device='cuda'
)

train_data, val_data = preprocessor.preprocess_and_split(
    output_dir='./preprocessed_data',
    max_samples=10000,
    test_split=0.2
)
```

### Vector Database Creation

```python
from database.vector_database import ProteinLigandVectorDB

vector_db = ProteinLigandVectorDB()
vector_db.build_database(
    preprocessed_data_path='./preprocessed_data/preprocessed_data.pkl',
    output_dir='./preprocessed_data/vector_database'
)
```

### Model Training

```python
from training.train_diffusion import DiffusionTrainer

trainer = DiffusionTrainer(
    preprocessed_data_path='./preprocessed_data/preprocessed_data.pkl',
    vector_db=vector_db,
    device='cuda'
)

trainer.train(
    num_epochs=100,
    batch_size=32,
    learning_rate=1e-4,
    lambda_ic50=0.1,
    output_dir='./trained_models'
)
```

### Evaluation

```python
from evaluation.metrics import evaluate_ligand_generation

# Evaluate generated ligands
metrics = evaluate_ligand_generation(
    generated_smiles=generated_smiles,
    ic50_predictions=predicted_ic50s,
    reference_smiles=reference_dataset,
    print_summary=True
)
```

### Benchmarking

```python
from evaluation.benchmark import ProteinLigandBenchmark

benchmark = ProteinLigandBenchmark(
    generator=generator,
    test_data_path='./test_data.tsv',
    output_dir='./benchmark_results'
)

results = benchmark.run_benchmark_suite(
    num_proteins=50,
    num_generations_per_protein=10
)
```

## Configuration

Modify `config.yaml` to customize the pipeline:

```yaml
# Data paths
data:
  ic50_data_path: "/path/to/IC50_data.tsv"
  smi_ted_path: "../../materials.smi-ted/smi-ted/inference/smi_ted_light"

# Training parameters
training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 100
  lambda_ic50: 0.1

# Inference parameters
inference:
  num_samples: 10
  top_k_retrieve: 10
  num_inference_steps: 50
```

## File Structure

```
generator_v1/
├── idea.md                     # Core concept and architecture
├── config.yaml                # Configuration file
├── requirements.txt            # Python dependencies
├── example_pipeline.py         # Complete pipeline example
├── README.md                   # This file
├── preprocess/
│   └── data_preprocessor.py    # Data preprocessing pipeline
├── database/
│   └── vector_database.py      # FAISS vector database
├── models/
│   └── diffusion_model.py      # Diffusion model implementation
├── training/
│   └── train_diffusion.py      # Training script
├── inference/
│   └── ligand_generator.py     # Inference pipeline
└── evaluation/
    ├── metrics.py              # Evaluation metrics
    └── benchmark.py            # Benchmarking suite
```

## Key Features

### Retrieval-Augmented Generation
- Uses FAISS for efficient similarity search
- Combines protein sequence and pocket site similarities
- Leverages known protein-ligand pairs as generation context

### IC50-Based Regularization
- Integrates BlendNet IC50 predictor during training
- Loss function includes binding affinity optimization
- Filters generated candidates by predicted binding strength

### Comprehensive Evaluation
- Chemical validity and uniqueness metrics
- Molecular diversity and drug-likeness scores
- Binding affinity predictions and improvements
- Novelty compared to training data

### Modular Design
- Each component can be used independently
- Easy to swap different encoders or models
- Configurable via YAML files
- Extensible architecture

## Requirements

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- 50GB+ storage space

### Key Dependencies
- PyTorch 1.9+
- RDKit 2022.9+
- Transformers 4.21+
- FAISS 1.7+
- NumPy, Pandas, SciPy

### Pre-trained Models Required
- ProtBERT (automatically downloaded)
- smi-TED Light model
- BlendNet IC50/Ki predictors
- Pseq2Sites embeddings

## Performance

Expected performance on modern hardware:

- **Preprocessing**: ~2-4 hours for 50K compounds
- **Training**: ~8-12 hours for 100 epochs (GPU)
- **Inference**: ~30-60 seconds per protein
- **Memory**: ~8-12GB GPU RAM during training

## Evaluation Metrics

The system provides comprehensive evaluation:

1. **Chemical Validity**: Percentage of valid SMILES
2. **Uniqueness**: Percentage of unique compounds
3. **Diversity**: Tanimoto-based diversity score
4. **Drug-likeness**: Lipinski's Rule of Five, QED scores
5. **Binding Affinity**: IC50 predictions and improvements
6. **Novelty**: Comparison against reference datasets

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch_size in config.yaml
2. **Invalid SMILES generated**: Increase num_inference_steps
3. **Poor binding affinity**: Increase lambda_ic50 weight
4. **Slow inference**: Reduce top_k_retrieve or num_inference_steps

### Performance Optimization

1. **Use mixed precision**: Set `mixed_precision: true`
2. **Optimize batch size**: Enable `max_batch_size_auto`
3. **Use gradient checkpointing**: For memory-constrained training
4. **Pre-compute embeddings**: Store embeddings on fast storage

## Citation

If you use this code in your research, please cite:

```bibtex
@software{protein_ligand_diffusion_v1,
  title={Protein-Ligand Diffusion Generator v1},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/protein-ligand-diffusion}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the example scripts
3. Open an issue on GitHub
4. Contact the maintainers

## Acknowledgments

- ProtBERT team for protein embeddings
- smi-TED team for compound embeddings  
- BlendNet team for binding affinity prediction
- Pseq2Sites team for pocket site prediction
- RDKit community for molecular computing tools
