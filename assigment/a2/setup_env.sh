#!/bin/bash
# Setup script to activate the course Python environment
# Usage: source setup_env.sh

echo "Activating course Python environment..."
source /data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate

echo "✓ Environment activated"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""
echo "You can now run:"
echo "  python sanity_check.py          # Test all components"
echo "  python test_integration.py      # Test A1/A2 integration"
echo "  python train_a2.py --help       # See training options"
echo "  sbatch run_a2_slurm.sh          # Submit training job"
