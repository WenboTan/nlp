#!/bin/bash
#SBATCH --job-name=assignment5_rag_full
#SBATCH --output=slurm_output_%j.txt
#SBATCH --error=slurm_error_%j.txt
#SBATCH --partition=long
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L4:1

# Load any necessary modules (if needed)
# module load CUDA/11.8.0

# Activate the virtual environment
source /data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate

# Print some information
echo "Job started on $(hostname) at $(date)"
echo "Python version: $(python --version)"
echo "GPU information:"
nvidia-smi

# Change to the assignment directory
cd /data/users/wenbota/nlp/assigment/5

# Run the assignment
python assignment5.py 2>&1 | tee output_gpu.txt

echo "Job finished at $(date)"
