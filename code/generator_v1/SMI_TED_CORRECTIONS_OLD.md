# smi-TED Checkpoint Path Corrections

## Summary of Changes Made

### Issue Identified
The smi-TED checkpoint loading was incorrectly assuming the checkpoint file (`smi-ted-Light_40.pt`) was located in the inference subdirectory, when it's actually located in the ROOT materials.smi-ted directory.

### Files Corrected

#### 1. `test_smited_loading.ipynb` ✅ FIXED
- **Before**: Using `/path/to/inference/smi_ted_light/` as folder parameter
- **After**: Using `/home/sarvesh/scratch/GS/samyak/.Blendnet/materials.smi-ted` as folder parameter
- **Change**: Moved checkpoint lookup to root directory
- **Diagnostic**: Added comprehensive path checking and troubleshooting

#### 2. `inference/ligand_generator.py` ✅ FIXED
- **Before**: Potentially incorrect path handling
- **After**: Explicitly using root materials.smi-ted directory
- **Change**: Updated `_initialize_models()` method with correct path and added comment

#### 3. Already Correct (using search mechanisms)
- `embedder.py` ✅ Uses search mechanism to find correct path
- `trainer.py` ✅ Uses search mechanism to find correct path  
- `example_pipeline.py` ✅ Uses search mechanism to find correct path

### Correct smi-TED Loading Pattern

```python
from smi_ted_light.load import load_smi_ted

# CORRECT: Use ROOT directory as folder parameter
model = load_smi_ted(
    folder='/path/to/materials.smi-ted',  # ROOT directory
    ckpt_filename='smi-ted-Light_40.pt'   # File located in ROOT
)
```

### File Structure Understanding

```
materials.smi-ted/
├── smi-ted-Light_40.pt          # ← Checkpoint file HERE (ROOT)
├── bert_vocab_curated.txt
├── config.json
├── model_weights.bin
├── smi-ted/
│   ├── inference/
│   │   └── smi_ted_light/
│   │       └── load.py          # ← Import from HERE
│   ├── training/
│   └── finetune/
└── ...
```

### Key Points
1. **Import path**: Use `smi-ted/inference/smi_ted_light/load.py`
2. **Folder parameter**: Use ROOT `materials.smi-ted` directory
3. **Checkpoint file**: Located in ROOT, not in subdirectories
4. **Search mechanism**: Most files already use robust search to find correct paths

### Testing
- ✅ Corrected test notebook shows proper diagnostic and loading
- ✅ All modular pipeline components use correct paths
- ✅ Error handling covers path issues and missing files

### Integration Status
🎉 **ALL COMPONENTS NOW USE CORRECT SMI-TED LOADING PATTERN**

The modular protein-ligand diffusion pipeline is fully corrected and ready for production use.
