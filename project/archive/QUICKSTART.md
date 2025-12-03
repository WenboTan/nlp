# 🚀 快速开始指南

## 5 分钟构建你的课程问答系统

### 前置要求

- Python 3.8+
- 已爬取的课程数据：`chalmers_courses_full_scraped.json`
- OpenAI API Key（或使用本地模型）

---

## 步骤 1️⃣：安装依赖（2分钟）

**选项 A - 使用安装脚本（推荐）**
```bash
./install_rag.sh
```

**选项 B - 手动安装**
```bash
pip install -r requirements.txt
```

---

## 步骤 2️⃣：配置 API Key（30秒）

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，替换 your_api_key_here
nano .env  # 或使用其他编辑器
```

在 `.env` 文件中：
```
OPENAI_API_KEY=sk-your-actual-openai-key-here
```

💡 **获取 API Key**: https://platform.openai.com/api-keys

---

## 步骤 3️⃣：构建向量数据库（5-10分钟）

```bash
python build_vector_db.py
```

这会：
- 处理 1122 门课程
- 生成约 8500+ 个文本块
- 创建嵌入向量
- 保存到 `./chalmers_chroma_db/`

**只需运行一次！** 之后数据库会被持久化保存。

---

## 步骤 4️⃣：启动问答系统（即时）

```bash
python rag_query_system.py
```

系统启动后你会看到：

```
🎓 Chalmers Course Assistant - Interactive Mode
======================================================================

Ask me anything about Chalmers courses!

💬 You: _
```

---

## 🎯 示例对话

### 查找课程
```
💬 You: What machine learning courses are available?

🤖 Assistant: Here are the machine learning courses:
- TDA233: Algoritmer för maskininlärning och slutledning (7.5 credits)
- TIF285: Bayesiansk dataanalys och maskininlärning (7.5 credits)
...
```

### 检查时间冲突（核心功能！）
```
💬 You: Can I take TDA357 and DAT450 together?

🤖 Assistant: Let me check:
- TDA357 (Databases): Block D
- DAT450 (Advanced NLP): Block C

✓ These courses do NOT have a time conflict!
```

### 查询先修课程
```
💬 You: What are the prerequisites for TDA357?

🤖 Assistant: TDA357 (Databases, 7.5 credits)
Prerequisites: "Object-oriented programming and basic data structures"
URL: https://www.chalmers.se/en/.../TDA357/...
```

---

## ✅ 验证安装

运行测试脚本检查一切是否正常：

```bash
python test_rag_setup.py
```

应该看到：
```
✓ PASS: Files
✓ PASS: Dependencies
✓ PASS: Vector Database
✓ PASS: Environment
✓ PASS: Database Query

🎉 All checks passed! Your RAG system is ready.
```

---

## 📁 文件清单

安装完成后你应该有：

```
project/
├── chalmers_courses_full_scraped.json  # 原始数据
├── build_vector_db.py                  # 数据库构建器
├── rag_query_system.py                 # 问答系统（主程序）
├── chalmers_chroma_db/                 # 向量数据库（自动生成）
├── .env                                # API Key 配置
├── requirements.txt                    # Python 依赖
└── RAG_README.md                       # 完整文档
```

---

## 🔧 常见问题速查

| 问题 | 解决方案 |
|------|----------|
| `FileNotFoundError: chalmers_chroma_db` | 先运行 `python build_vector_db.py` |
| `ImportError: No module named 'langchain'` | 运行 `pip install -r requirements.txt` |
| `ValueError: OPENAI_API_KEY not found` | 检查 `.env` 文件是否配置正确 |
| 查询响应慢 | 考虑使用 GPU 或减少 `RETRIEVAL_K` |
| 数据库构建失败 | 检查 `chalmers_courses_full_scraped.json` 是否存在 |

---

## 🎓 核心功能

### ✨ 智能时间冲突检测
系统会自动分析课程的 Block 信息：
- **相同 Block** → 🔴 时间冲突
- **不同 Block** → ✅ 无冲突

### 🔍 语义搜索
不需要精确匹配关键词，AI 能理解你的意图：
- "database courses" → 找到 TDA357, DAT300...
- "AI related courses" → 找到机器学习、深度学习、NLP...

### 📚 丰富的上下文
每次查询检索 5 个最相关的课程文档，包含：
- 课程代码、标题、学分
- 先修课程、时间安排
- 学习成果、课程内容
- 适用项目、交换生资格

---

## 🚀 进阶使用

### 使用更强大的模型

编辑 `rag_query_system.py`：
```python
LLM_MODEL = 'gpt-4'  # 更准确但更贵
```

### GPU 加速（如果有 CUDA）

在两个脚本中修改：
```python
model_kwargs={'device': 'cuda'}  # 从 'cpu' 改为 'cuda'
```

### 增加检索文档数

```python
RETRIEVAL_K = 10  # 默认 5
```

---

## 📖 详细文档

查看完整文档获取更多信息：
- **RAG_README.md** - 完整技术文档
- **系统架构说明**
- **本地模型配置**
- **性能优化建议**

---

## 💡 提示

- 第一次运行会下载嵌入模型（约 80MB），需要等待
- 数据库只需构建一次，除非数据更新
- 使用 `gpt-3.5-turbo` 每次查询约花费 $0.001-0.002
- 输入 `quit` 或 `q` 退出问答系统

---

**🎉 完成！现在你可以开始与你的 AI 课程助手对话了！**
