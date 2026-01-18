#!/bin/bash
#SBATCH --job-name=rag_openai
#SBATCH --output=logs/rag_openai_%j.out
#SBATCH --error=logs/rag_openai_%j.err
#SBATCH --partition=cpu           # OpenAI 版本不需要 GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00

echo "=========================================="
echo "Chalmers Course RAG - OpenAI Version"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

# 激活虚拟环境
VENV_PATH="/data/courses/2025_dat450_dit247/venvs/dat450_venv"
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "✓ Activated dat450_venv"
    echo "Python: $(which python3)"
else
    echo "❌ Virtual environment not found at $VENV_PATH"
    exit 1
fi

# 检查 .env 文件
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "Please create .env file with your OPENAI_API_KEY"
    echo "Run: cp .env.example .env"
    exit 1
fi

echo ""
echo "✓ .env file found"

# 运行 RAG 系统
echo ""
echo "Starting RAG system with OpenAI..."
echo "=========================================="
echo ""

python rag_query_system_openai.py

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
