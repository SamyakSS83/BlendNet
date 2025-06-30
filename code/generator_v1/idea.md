# Protein-Specific Ligand Generation via Diffusion Model

## Core Idea

Design a diffusion model that generates optimal ligands for given protein sequences by leveraging:
1. **Protein embeddings** (ProtBERT + Pseq2Sites)
2. **Compound embeddings** (smi-TED)
3. **Binding affinity prediction** (BlendNet IC50 predictor)

## Architecture Overview

### Data Pipeline
1. **Input Data**: `/home/sarvesh/scratch/GS/negroni_data/Blendnet/input_data/BindingDB/IC50_data.tsv`
2. **Vector Database**: FAISS for efficient similarity search
3. **Storage Format**: [compound_id, protbert_embeddings(E1), pseq2sites_embeddings(Es), smiles, ic50_value]

### Model Workflow

#### Phase 1: Preprocessing & Database Creation
1. Load IC50 dataset and extract unique molecules
2. Generate ProtBERT embeddings (E1) for all protein sequences
3. Generate Pseq2Sites embeddings (Es) for all protein sequences  
4. Encode all SMILES using smi-TED
5. Store in FAISS vector database with cosine similarity indexing

#### Phase 2: Retrieval-Augmented Generation
**Input**: New protein sequence
1. Generate E1 (ProtBERT) and Es (Pseq2Sites) for input protein
2. Retrieve top-k similar compounds using: `argmax(alpha * sim(Es) + (1-alpha) * sim(E1))`
3. Encode retrieved SMILES using smi-TED
4. Compute mean of top-k compound embeddings as diffusion starting point

#### Phase 3: Diffusion Generation
**Conditioning**: Es + E1 of input protein
**Starting Point**: Mean of top-k retrieved compound embeddings
**Process**: 
1. Forward diffusion: Add noise to mean compound embedding
2. Reverse diffusion: Denoise while conditioning on protein embeddings
3. Generate new compound embedding in smi-TED space
4. Decode to SMILES using smi-TED decoder

#### Phase 4: Loss Function
```
Total_Loss = Diffusion_Loss + λ/IC50_predicted
```
Where:
- `Diffusion_Loss`: Standard denoising objective
- `IC50_predicted`: BlendNet prediction for (generated_smile, input_protein)
- `λ`: Regularization weight (encourages better binding)

## Key Design Principles

1. **Efficiency**: Precompute all embeddings, use FAISS for fast retrieval
2. **Modularity**: Separate preprocessing, retrieval, diffusion, and evaluation
3. **Conditioning**: Strong protein-specific conditioning via dual embeddings
4. **Guidance**: IC50-based regularization for binding optimization
5. **Retrieval-Augmented**: Leverage existing successful compounds as starting points

## File Structure
```
generator_v1/
├── idea.md                 # This file
├── preprocess/            # Data preprocessing
├── database/              # FAISS vector database
├── models/               # Diffusion model components
├── training/             # Training scripts
├── inference/            # Generation pipeline
└── evaluation/           # Metrics and validation
```

## Success Criteria
1. Generated ligands have better predicted IC50 than baseline
2. Generated SMILES are chemically valid (>95%)
3. Diverse ligand generation for different protein families
4. Fast inference (<1 minute per protein)
