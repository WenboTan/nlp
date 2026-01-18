#!/bin/bash
#SBATCH --job-name=rag_local
#SBATCH --partition=long
#SBATCH --gres=gpu:L4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rag_%j.out
#SBATCH --error=logs/rag_%j.err

# 运行本地模型 RAG 系统

echo "=========================================="
echo "Chalmers Course RAG - Local Model"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# 激活课程虚拟环境
source /data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate
echo "✓ Activated dat450_venv"
echo "Python: $(which python3)"
echo ""

# 创建日志目录
mkdir -p logs

# 切换到项目目录
cd /data/users/wenbota/nlp/project

# 检查 GPU
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# 检查向量数据库
if [ ! -d "chalmers_chroma_db" ]; then
    echo "❌ Error: Vector database not found!"
    echo "Please run: sbatch run_build_db.sh first"
    exit 1
fi

echo "✓ Vector database found ($(du -sh chalmers_chroma_db | cut -f1))"
echo ""

# 运行 RAG 系统
echo "Starting RAG system with local model..."
echo "=========================================="
echo ""

python rag_query_system_local.py

echo ""
echo "=========================================="
echo "End time: $(date)"
