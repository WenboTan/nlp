#!/bin/bash
#SBATCH --job-name=a2_test
#SBATCH --partition=long
#SBATCH --account=student
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out

# Use course dat450 environment's Python directly
PYTHON=/data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/python

# Non-interactive & cache settings
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export HF_HOME=$HOME/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export NLTK_DATA=$HOME/.cache/nltk

# Quick diagnostic
echo "[Diag] Python: $($PYTHON -V)"
$PYTHON -c "import torch; print('[Diag] torch', torch.__version__, 'cuda=', torch.cuda.is_available())" || echo "[Warning] torch check failed"

# Paths
PROJECT_DIR="/data/users/wenbota/nlp/assigment/a2"
TRAIN_FILE="/data/users/wenbota/nlp/assigment/1/a1/train.txt"
VAL_FILE="/data/users/wenbota/nlp/assigment/1/a1/val.txt"
OUTPUT_DIR="${PROJECT_DIR}/a2_model_test"
TOKENIZER_FILE="${PROJECT_DIR}/a2_tokenizer_test.pkl"

cd "$PROJECT_DIR"

# Small test parameters
EPOCHS=2
LR=1e-4
TRAIN_BS=8
EVAL_BS=8
HIDDEN=128           # Smaller model for quick test
LAYERS=2
HEADS=4
MAX_VOC=5000         # Smaller vocab for quick test
MAXLEN=128
SUBSAMPLE=1000       # Only use 1000 samples for quick test

echo "[Info] Quick test with small model and dataset..."
echo "[Info] Train file: $TRAIN_FILE"
echo "[Info] Val file: $VAL_FILE"
echo "[Info] Subsample: $SUBSAMPLE examples"
echo "[Info] Model: hidden=$HIDDEN, layers=$LAYERS, heads=$HEADS"

# Run training
$PYTHON train_a2.py \
    --train_file "$TRAIN_FILE" \
    --val_file "$VAL_FILE" \
    --save_tokenizer "$TOKENIZER_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --subsample "$SUBSAMPLE" \
    --epochs "$EPOCHS" \
    --train_batch "$TRAIN_BS" \
    --eval_batch "$EVAL_BS" \
    --lr "$LR" \
    --max_voc_size "$MAX_VOC" \
    --model_max_length "$MAXLEN" \
    --hidden_size "$HIDDEN" \
    --num_layers "$LAYERS" \
    --num_heads "$HEADS" \
    --seed 2025 \
    --predict_prompt "She lives in San"

echo "[Done] Test completed!"
echo "[Info] If this works, you can run the full training with: sbatch run_a2_slurm.sh"
