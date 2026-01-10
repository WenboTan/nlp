# 🚀 快速启动指南

## 立即开始（3 条命令）

### 1️⃣ 安装依赖（5 分钟）

```bash
cd /data/users/wenbota/nlp/project
module load Python/3.10.4 CUDA/11.7
pip install --user -r requirements_local.txt
```

### 2️⃣ 构建向量数据库（10-15 分钟）

```bash
sbatch run_build_db.sh
# 等待完成后查看日志
tail -f logs/build_db_*.out
```

### 3️⃣ 启动 RAG 系统（首次 20 分钟，之后秒启）

#### 方式 A: 通过任务提交（适合长时间运行）
```bash
sbatch run_rag_local.sh
tail -f logs/rag_*.out
```

#### 方式 B: 交互式会话（推荐，实时问答）
```bash
# 申请 GPU 节点
srun -p gpu --gres=gpu:1 --mem=32G --time=2:00:00 --pty bash

# 加载模块
module load Python/3.10.4 CUDA/11.7

# 启动系统
cd /data/users/wenbota/nlp/project
python rag_query_system_local.py

# 开始提问！
💬 You: What machine learning courses are available?
```

---

## 💬 示例问题

```
💬 You: Can I take TDA357 and DAT450 together?
💬 You: What are the prerequisites for database courses?
💬 You: Show me 7.5 credit AI courses in Block C
💬 You: Which courses are open for exchange students?
```

---

## 📚 完整文档

- **README.md** - 完整使用指南
- **LOCAL_MODEL_GUIDE.md** - 本地模型详解
- **PROJECT_OVERVIEW.md** - 项目架构

---

## 🆘 遇到问题？

```bash
# 检查任务状态
squeue -u $USER

# 查看日志
ls -lt logs/

# 检查 GPU
nvidia-smi
```

**常见问题**: 见 README.md 故障排除章节

---

✨ **就是这么简单！3 步启动你的 AI 课程助手！**
