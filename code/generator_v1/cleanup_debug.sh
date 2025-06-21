#!/bin/bash
"""
Clean up debug files and prepare for full pipeline run.
This script moves debug files to a debug folder and removes test data/checkpoints.
"""

set -e  # Exit on any error

echo "=========================================="
echo "CLEANING UP DEBUG FILES"
echo "=========================================="

# Create debug directory if it doesn't exist
mkdir -p debug

echo "Moving debug files to debug folder..."

# Move debug and test files
if [ -f "debug_embedding_shapes.py" ]; then
    mv debug_embedding_shapes.py debug/
    echo "✓ Moved debug_embedding_shapes.py"
fi

if [ -f "fix_compound_embeddings.py" ]; then
    mv fix_compound_embeddings.py debug/
    echo "✓ Moved fix_compound_embeddings.py"
fi

if [ -f "fix_compound_embeddings_v2.py" ]; then
    mv fix_compound_embeddings_v2.py debug/
    echo "✓ Moved fix_compound_embeddings_v2.py"
fi

if [ -f "inspect_smi_ted_config.py" ]; then
    mv inspect_smi_ted_config.py debug/
    echo "✓ Moved inspect_smi_ted_config.py"
fi

if [ -f "inspect_smi_ted_config_fixed.py" ]; then
    mv inspect_smi_ted_config_fixed.py debug/
    echo "✓ Moved inspect_smi_ted_config_fixed.py"
fi

if [ -f "test_training_fixed.py" ]; then
    mv test_training_fixed.py debug/
    echo "✓ Moved test_training_fixed.py"
fi

if [ -f "fix_preprocessing.py" ]; then
    mv fix_preprocessing.py debug/
    echo "✓ Moved fix_preprocessing.py"
fi

echo ""
echo "=========================================="
echo "REMOVING OLD PREPROCESSED DATA"
echo "=========================================="

# Remove old preprocessed data files
if [ -d "preprocessed_data" ]; then
    echo "Removing old preprocessed data..."
    
    # Keep only the final fixed version
    if [ -f "preprocessed_data/preprocessed_data_fixed.pkl" ]; then
        echo "✓ Keeping preprocessed_data_fixed.pkl (the working version)"
    fi
    
    # Remove intermediate files
    rm -f preprocessed_data/preprocessed_data.pkl
    rm -f preprocessed_data/preprocessed_data_compound_fixed.pkl
    
    # Remove old vector database if it exists
    if [ -d "preprocessed_data/vector_database" ]; then
        rm -rf preprocessed_data/vector_database
        echo "✓ Removed old vector_database"
    fi
    
    # Keep the working vector database
    if [ -d "preprocessed_data/vector_database_fixed" ]; then
        echo "✓ Keeping vector_database_fixed (the working version)"
    fi
    
    echo "✓ Cleaned up preprocessed_data directory"
else
    echo "No preprocessed_data directory found"
fi

echo ""
echo "=========================================="
echo "REMOVING TEST CHECKPOINTS"
echo "=========================================="

# Remove test checkpoints
if [ -d "test_checkpoints" ]; then
    rm -rf test_checkpoints
    echo "✓ Removed test_checkpoints"
fi

if [ -d "trained_models" ]; then
    echo "Cleaning trained_models directory..."
    rm -f trained_models/train_data.pkl
    rm -f trained_models/val_data.pkl
    rm -f trained_models/diffusion_model.pth
    echo "✓ Cleaned trained_models directory"
fi

echo ""
echo "=========================================="
echo "REMOVING EXAMPLE RESULTS"
echo "=========================================="

if [ -d "example_results" ]; then
    rm -rf example_results
    echo "✓ Removed example_results"
fi

echo ""
echo "=========================================="
echo "FINAL CLEANUP STATUS"
echo "=========================================="

echo "Files in current directory:"
ls -la *.py 2>/dev/null || echo "No Python files in current directory"

echo ""
echo "Files in preprocessed_data:"
if [ -d "preprocessed_data" ]; then
    ls -la preprocessed_data/
else
    echo "No preprocessed_data directory"
fi

echo ""
echo "Files in debug:"
if [ -d "debug" ]; then
    ls -la debug/
else
    echo "No debug directory"
fi

echo ""
echo "=========================================="
echo "✅ CLEANUP COMPLETED"
echo "=========================================="
echo ""
echo "Ready for full pipeline run!"
echo "The following files are ready to use:"
echo "  - preprocessed_data/preprocessed_data_fixed.pkl (if exists)"
echo "  - preprocessed_data/vector_database_fixed/ (if exists)"
echo ""
echo "To run the full pipeline:"
echo "  python example_pipeline.py --mode full"
echo ""
