#!/bin/bash

echo "============================================================"
echo "CLEANING UP DEBUG FILES AND RESETTING FOR FULL PIPELINE"
echo "============================================================"

# Create debug directory if it doesn't exist
mkdir -p debug

echo "Moving debug and test files to debug folder..."

# Move all debug and test files
mv debug_*.py debug/ 2>/dev/null || true
mv inspect_*.py debug/ 2>/dev/null || true
mv test_*.py debug/ 2>/dev/null || true
mv fix_*.py debug/ 2>/dev/null || true
mv *_test.py debug/ 2>/dev/null || true
mv *_debug.py debug/ 2>/dev/null || true

# Move temporary data files
mv preprocessed_data_*.pkl debug/ 2>/dev/null || true

echo "Removing old preprocessed data and checkpoints..."

# Remove old preprocessed data
rm -rf preprocessed_data/ 2>/dev/null || true
rm -rf trained_models/ 2>/dev/null || true
rm -rf test_checkpoints/ 2>/dev/null || true
rm -rf example_results/ 2>/dev/null || true

# Remove any leftover pickle files
rm -f *.pkl 2>/dev/null || true

# Remove any temporary directories
rm -rf __pycache__/ 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "Creating fresh directories..."

# Create fresh directories
mkdir -p preprocessed_data
mkdir -p trained_models
mkdir -p example_results

echo "✅ Cleanup completed!"
echo ""
echo "Files moved to debug/:"
ls debug/ 2>/dev/null | head -10 || echo "  (no files moved)"
echo ""
echo "Ready for full pipeline execution!"
echo "Run: python example_pipeline.py --mode full"
