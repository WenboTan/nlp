#!/bin/bash
#SBATCH --job-name=rag_test
#SBATCH --partition=long
#SBATCH --gres=gpu:L4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/rag_test_%j.out
#SBATCH --error=logs/rag_test_%j.err

echo "=========================================="
echo "Chalmers Course RAG - Batch Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# 激活虚拟环境
source /data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate
echo "✓ Activated dat450_venv"
echo "Python: $(which python3)"
echo ""

# 显示 GPU 信息
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# 检查向量数据库
if [ -d "./chalmers_chroma_db" ]; then
    echo "✓ Vector database found ($(du -sh ./chalmers_chroma_db | cut -f1))"
else
    echo "❌ Vector database not found!"
    echo "Please run build_vector_db.py first"
    exit 1
fi

echo ""
echo "Starting RAG batch test..."
echo "=========================================="
echo ""

# 运行批处理测试
python3 test_rag_batch.py

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
