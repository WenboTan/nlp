#!/bin/bash
#SBATCH --job-name=rag_build_db
#SBATCH --partition=short
#SBATCH --gres=gpu:L4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/build_db_%j.out
#SBATCH --error=logs/build_db_%j.err

# 构建向量数据库（使用 GPU 加速）

echo "=========================================="
echo "Building Chalmers Course Vector Database"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# 激活课程虚拟环境
source /data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate
echo "✓ Activated dat450_venv"
echo "Python: $(which python3)"
echo "Python version: $(python3 --version)"
echo ""

# 创建日志目录
mkdir -p logs

# 切换到项目目录
cd /data/users/wenbota/nlp/project

# 检查 GPU
echo "Checking GPU..."
nvidia-smi
echo ""

# 检查数据文件
if [ ! -f "chalmers_courses_full_scraped.json" ]; then
    echo "❌ Error: chalmers_courses_full_scraped.json not found!"
    exit 1
fi

echo "Data file found: chalmers_courses_full_scraped.json ($(du -h chalmers_courses_full_scraped.json | cut -f1))"
echo ""

# 运行构建脚本
echo "Starting vector database build..."
python build_vector_db.py

# 检查结果
if [ -d "chalmers_chroma_db" ]; then
    echo ""
    echo "✅ Vector database created successfully!"
    echo "Database size: $(du -sh chalmers_chroma_db | cut -f1)"
    echo "Files in database: $(find chalmers_chroma_db -type f | wc -l)"
else
    echo ""
    echo "❌ Error: Vector database not created!"
    exit 1
fi

echo ""
echo "End time: $(date)"
echo "=========================================="
