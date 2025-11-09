#!/bin/bash
#SBATCH --job-name=a1_full
#SBATCH --gres=gpu:L4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=short
#SBATCH --account=student
#SBATCH --output=%x-%j.out

# Initialize/activate conda environment (robust to missing `module` on compute nodes)
# We try common conda locations first, then fall back to `conda` in PATH.
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
  . /opt/conda/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  # Initialize conda for this shell session
  eval "$(conda shell.bash hook 2>/dev/null || true)"
else
  echo "WARNING: conda not found. Ensure conda is available on the node or use a module to load it." >&2
fi

# Activate the environment we will create from conda-forge only
conda activate dat450
export MPLBACKEND=Agg
echo "[Diag] Python: $(python -V)"
python - <<'PY'
import torch, nltk, matplotlib
print('[Diag] torch', torch.__version__, 'cuda_available=', torch.cuda.is_available())
if torch.cuda.is_available():
  try:
    print('[Diag] device:', torch.cuda.get_device_name(0))
  except Exception as e:
    print('[Diag] device query error:', e)
print('[Diag] nltk', nltk.__version__, 'matplotlib', matplotlib.__version__)
PY
cd /data/users/wenbota/nlp/assigment/1/a1

# Run training; adjust args if you want different hyperparameters
python3 train_full.py \
  --epochs 3 \
  --train_batch 64 \
  --eval_batch 64 \
  --lr 5e-4 \
  --max_voc_size 20000 \
  --model_max_length 128 \
  --embedding_size 256 \
  --hidden_size 512 \
  --output_dir ./a1_model_full
