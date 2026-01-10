#!/bin/bash
#SBATCH --job-name=rag_gemini
#SBATCH --output=logs/rag_gemini_%j.out
#SBATCH --error=logs/rag_gemini_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Google Gemini 版本不需要 GPU，只需要 CPU 即可

echo "=========================================="
echo "Chalmers Course RAG - Gemini Interactive"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo ""

# 加载模块
module load Python/3.10.4
echo "✓ Python module loaded"

# 激活虚拟环境（如果有）
if [ -d "/data/courses/2025_dat450_dit247/venvs/dat450_venv" ]; then
    source /data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate
    echo "✓ Activated dat450_venv"
    echo "Python: $(which python3)"
fi

# 检查向量数据库
if [ -d "chalmers_chroma_db" ]; then
    echo ""
    echo "✓ Vector database found ($(du -sh chalmers_chroma_db/ | cut -f1))"
else
    echo ""
    echo "❌ Vector database not found!"
    echo "Please run: sbatch run_build_db.sh"
    exit 1
fi

# 检查 API Key
if [ -z "$GOOGLE_API_KEY" ] && [ ! -f ".env" ]; then
    echo ""
    echo "❌ GOOGLE_API_KEY not configured!"
    echo "Please create .env file with:"
    echo "  GOOGLE_API_KEY=your-api-key-here"
    echo ""
    echo "Get your API key at: https://makersuite.google.com/app/apikey"
    exit 1
fi

echo ""
echo "Starting RAG system with Google Gemini..."
echo "=========================================="
echo ""

# 运行 Gemini RAG 系统（交互式）
python rag_query_system_gemini.py

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
