#!/bin/bash
#SBATCH --job-name=a2_transformer
#SBATCH --partition=long
#SBATCH --account=student
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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
OUTPUT_DIR="${PROJECT_DIR}/a2_model_lr1e-4_b16_h256_l4"
TOKENIZER_FILE="${PROJECT_DIR}/a2_tokenizer.pkl"

cd "$PROJECT_DIR"

# Hyperparameters - Transformer settings
EPOCHS=5
LR=1e-4              # Transformer通常用较小的学习率
TRAIN_BS=16          # 根据GPU内存调整，如果OOM可以减小
EVAL_BS=16
HIDDEN=256           # Hidden size
LAYERS=4             # Number of Transformer layers
HEADS=4              # Number of attention heads
MAX_VOC=20000
MAXLEN=128

echo "[Info] Starting A2 Transformer training..."
echo "[Info] Train file: $TRAIN_FILE"
echo "[Info] Val file: $VAL_FILE"
echo "[Info] Output: $OUTPUT_DIR"
echo "[Info] Model: hidden=$HIDDEN, layers=$LAYERS, heads=$HEADS"
echo "[Info] Training: epochs=$EPOCHS, lr=$LR, batch=$TRAIN_BS"

# Run training
$PYTHON train_a2.py \
    --train_file "$TRAIN_FILE" \
    --val_file "$VAL_FILE" \
    --save_tokenizer "$TOKENIZER_FILE" \
    --output_dir "$OUTPUT_DIR" \
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

echo "[Done] Training completed!"
echo "[Info] Model saved to: $OUTPUT_DIR"
echo "[Info] Tokenizer saved to: $TOKENIZER_FILE"
