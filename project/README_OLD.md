# 🚀 Chalmers Course RAG 系统 - 本地模型版

使用开源模型的课程问答系统，完全在学校 GPU 集群上运行。

---

## 📁 项目结构（清理后）

```
/data/users/wenbota/nlp/project/
│
├── 🔧 核心脚本
│   ├── build_vector_db.py              # 构建向量数据库
│   ├── rag_query_system_local.py       # 本地模型 RAG 系统
│   └── test_rag_setup.py               # 系统验证工具
│
├── 📊 数据文件
│   └── chalmers_courses_full_scraped.json  # 1122 门课程（114MB）
│
├── 🔨 工具脚本
│   ├── syllabus_scraper.py             # 课程爬虫
│   └── deduplicate_courses.py          # 去重工具
│
├── 🚀 SLURM 任务脚本
│   ├── run_build_db.sh                 # 构建数据库任务
│   └── run_rag_local.sh                # 运行 RAG 系统任务
│
├── 📖 文档
│   ├── README.md                       # 本文件
│   ├── LOCAL_MODEL_GUIDE.md            # 本地模型详细指南
│   ├── PROJECT_OVERVIEW.md             # 项目总览
│   └── RAG_README.md                   # RAG 技术文档
│
└── 📦 归档（可忽略）
    └── archive/                        # 旧版本和测试文件
```

---

## 🎯 快速开始（3 步）

### Step 1: 安装依赖

```bash
cd /data/users/wenbota/nlp/project

# 检查 Python 和 CUDA
module load Python/3.10.4
module load CUDA/11.7

# 安装依赖
pip install --user -r requirements_local.txt
```

**预计时间**: 5-10 分钟

---

### Step 2: 构建向量数据库

```bash
# 创建日志目录
mkdir -p logs

# 提交构建任务
sbatch run_build_db.sh

# 查看任务状态
squeue -u $USER

# 查看日志（等任务完成后）
tail -f logs/build_db_*.out
```

**预计时间**: 10-15 分钟  
**生成内容**: `chalmers_chroma_db/` 目录（约 500MB）

---

### Step 3: 运行 RAG 系统

```bash
# 提交 RAG 任务
sbatch run_rag_local.sh

# 查看输出
tail -f logs/rag_*.out
```

**首次运行**: 会下载模型（Phi-3-mini 约 7GB），需要 15-20 分钟  
**后续运行**: 立即启动

---

## 💬 使用方式

### 方式 A: 交互式模式（通过 SLURM 作业）

启动后在日志中提问：

```
💬 You: What machine learning courses are available?

🤖 Assistant: Here are the machine learning courses:
- TDA233: Algoritmer för maskininlärning och slutledning (7.5 credits, Block D)
- TIF285: Bayesiansk dataanalys och maskininlärning (7.5 credits, Block C)
...
```

### 方式 B: 交互式终端（推荐）

如果需要实时交互，申请交互式 GPU 节点：

```bash
# 申请交互式会话
srun -p gpu --gres=gpu:1 --mem=32G --time=2:00:00 --pty bash

# 加载模块
module load Python/3.10.4 CUDA/11.7

# 运行 RAG 系统
cd /data/users/wenbota/nlp/project
python rag_query_system_local.py

# 直接提问
💬 You: Can I take TDA357 and DAT450 together?
```

---

## 🔍 示例查询

### 1. 查找课程
```
💬 You: What courses are available about artificial intelligence?
```

### 2. 检查时间冲突（核心功能）
```
💬 You: Can I take TDA357 and TDA233 at the same time?
🤖 Assistant: Let me check the schedules:
- TDA357 (Databases): Block D
- TDA233 (Machine Learning): Block D
⚠️ TIME CONFLICT: Both courses are in Block D
```

### 3. 查询先修课程
```
💬 You: What are the prerequisites for database courses?
```

### 4. 查询交换生资格
```
💬 You: Which data science courses are open for exchange students?
```

---

## 🎛️ 配置选项

### 更换模型

编辑 `rag_query_system_local.py` 第 18 行：

```python
# 小模型（默认，推荐）
LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"      # ~8GB 显存

# 中型模型（更好的质量）
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # ~14GB 显存

# 大型模型（最佳质量）
LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct" # ~16GB 显存

# 中英双语
LLM_MODEL = "Qwen/Qwen2-7B-Instruct"              # ~14GB 显存
```

### 调整检索数量

```python
RETRIEVAL_K = 5  # 每次查询检索 5 个最相关的课程
```

### 关闭量化（如果有大显存）

```python
use_8bit = False  # 在 load_local_llm() 调用中
```

---

## 📊 资源需求

| 组件 | CPU | 内存 | GPU 显存 | 磁盘 |
|------|-----|------|----------|------|
| **构建数据库** | 4-8 核 | 16GB | 可选 | 500MB |
| **RAG 运行 (Phi-3)** | 8 核 | 16GB | 8GB | 7GB |
| **RAG 运行 (Mistral)** | 8 核 | 24GB | 14GB | 14GB |
| **RAG 运行 (Llama-3)** | 8 核 | 32GB | 16GB | 16GB |

**Minerva 集群**: 通常提供 24GB+ 显存，可运行任何模型。

---

## 🔧 故障排除

### 问题 1: 向量数据库未找到
```
❌ FileNotFoundError: chalmers_chroma_db
```
**解决**: 先运行 `sbatch run_build_db.sh`

### 问题 2: GPU 内存不足
```
❌ CUDA out of memory
```
**解决**: 
1. 使用更小的模型（Phi-3-mini）
2. 启用 8-bit 量化：`use_8bit=True`
3. 减少 batch size

### 问题 3: 模型下载失败
```
❌ Connection timeout
```
**解决**:
1. 检查网络连接
2. 使用国内镜像：`export HF_ENDPOINT=https://hf-mirror.com`
3. 手动下载模型到本地

### 问题 4: 依赖包缺失
```
❌ ImportError: No module named 'transformers'
```
**解决**: `pip install --user -r requirements_local.txt`

---

## 📈 性能指标

基于 Phi-3-mini 模型：

- **向量检索**: < 100ms
- **LLM 生成**: 2-5 秒（取决于答案长度）
- **总响应时间**: 通常 < 6 秒
- **内存占用**: ~8GB GPU 显存

---

## 📚 文档

- **快速开始**: 本文件
- **详细指南**: `LOCAL_MODEL_GUIDE.md`
- **技术文档**: `RAG_README.md`
- **项目总览**: `PROJECT_OVERVIEW.md`

---

## 🎓 技术栈

- **向量数据库**: Chroma
- **嵌入模型**: sentence-transformers/all-MiniLM-L6-v2
- **LLM**: HuggingFace 开源模型（Phi-3/Mistral/Llama）
- **框架**: LangChain
- **量化**: bitsandbytes (8-bit)

---

## ✅ 检查清单

构建和运行前确认：

- [ ] 已安装所有依赖 (`requirements_local.txt`)
- [ ] 数据文件存在 (`chalmers_courses_full_scraped.json`)
- [ ] 已创建日志目录 (`mkdir -p logs`)
- [ ] 有 GPU 访问权限
- [ ] 磁盘空间充足（至少 10GB）

---

## 🚀 进阶功能

### 1. 微调模型

使用课程数据微调模型以提升专业性：

```bash
python finetune_model.py  # 见 LOCAL_MODEL_GUIDE.md
```

### 2. Web 界面

部署 Gradio/Streamlit 界面：

```bash
pip install gradio
python app.py  # 需要创建
```

### 3. 批量查询

创建脚本批量处理问题列表。

---

## 📞 获取帮助

- **查看日志**: `tail -f logs/*.out`
- **检查任务**: `squeue -u $USER`
- **取消任务**: `scancel <job_id>`
- **查看 GPU**: `nvidia-smi`

---

**最后更新**: 2025-12-02  
**版本**: v1.0 - 本地模型版  
**状态**: ✅ 生产就绪
