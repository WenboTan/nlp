#!/bin/bash
#SBATCH --job-name=a1_rnnlm_5ep
#SBATCH --partition=long
#SBATCH --account=student
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1                 # 如果你确定是 L4，可写成: --gres=gpu:L4:1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G                    # batch=4 + hidden=128 足够；不够可以提到 32G
#SBATCH --output=%x-%j.out

# Use dat450 environment's Python directly (bypasses conda activate issues in SLURM)
PYTHON=/data/users/wenbota/miniconda3/envs/dat450/bin/python

# 2) Non-interactive & cache (避免权限问题)
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export HF_HOME=$HOME/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export NLTK_DATA=$HOME/.cache/nltk

# 3) Quick diagnostic (check if torch available)
echo "[Diag] Python: $($PYTHON -V)"
$PYTHON -c "import torch; print('[Diag] torch', torch.__version__, 'cuda=', torch.cuda.is_available())" || echo "[Warning] torch check failed"

# 4) Paths
PROJECT_DIR="/data/users/wenbota/nlp/assigment/1/a1"
TRAIN_FILE="/data/courses/2025_dat450_dit247/assignments/a1/train.txt"
VAL_FILE="/data/courses/2025_dat450_dit247/assignments/a1/val.txt"
OUTPUT_DIR="${PROJECT_DIR}/a1_model_lr2e-3_b4_h128_e5"

cd "$PROJECT_DIR"

# 5) Hyperparameters (teacher's suggestion as defaults)
EPOCHS=5                 # 训练5个epochs获得更好效果
LR=2e-3                  # 如果 loss 抖或上升，试 1e-3 或 5e-4
TRAIN_BS=4               # 老师建议的 batch
EVAL_BS=4
EMB=128
HID=128
MAX_VOC=20000
MAXLEN=128

# 6) Run - 使用 dat450 环境的 Python
$PYTHON train_full.py \
  --train_file "$TRAIN_FILE" \
  --val_file "$VAL_FILE" \
  --epochs "$EPOCHS" \
  --train_batch "$TRAIN_BS" \
  --eval_batch "$EVAL_BS" \
  --lr "$LR" \
  --max_voc_size "$MAX_VOC" \
  --model_max_length "$MAXLEN" \
  --embedding_size "$EMB" \
  --hidden_size "$HID" \
  --output_dir "$OUTPUT_DIR" \
  --save_tokenizer a1_tokenizer.pkl
