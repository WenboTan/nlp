# Chalmers Course RAG System

完整的 RAG (Retrieval-Augmented Generation) 问答系统，用于查询 Chalmers 大学课程信息。

## 系统架构

```
chalmers_courses_full_scraped.json  (原始数据)
           ↓
    build_vector_db.py  (构建向量数据库)
           ↓
    chalmers_chroma_db/  (持久化向量数据库)
           ↓
    rag_query_system.py  (RAG 问答系统)
```

## 安装依赖

```bash
pip install langchain langchain-community langchain-chroma langchain-openai
pip install chromadb sentence-transformers openai python-dotenv
```

## 使用步骤

### 第一步：构建向量数据库

这一步将课程 JSON 数据转换为向量数据库：

```bash
python build_vector_db.py
```

**功能说明：**
- 加载 `chalmers_courses_full_scraped.json` (1122 门课程)
- 为每门课程创建 LangChain Document，包含：
  - **page_content**: 课程代码、标题、学分、语言、时间块、学习成果、先修课程等
  - **metadata**: 结构化信息（课程代码、Block、学分、URL 等）用于过滤
- 使用 `RecursiveCharacterTextSplitter` 切分文本 (chunk_size=1000, overlap=200)
- 使用 `sentence-transformers/all-MiniLM-L6-v2` 模型生成嵌入向量
- 保存到 `./chalmers_chroma_db/` 目录

**预期输出：**
```
Building Chalmers Course Vector Database
======================================================================
Loading course data from chalmers_courses_full_scraped.json...
✓ Loaded 1122 courses

Converting courses to LangChain Documents...
  Processed 100/1122 courses
  ...
✓ Created 1122 documents

Splitting documents (chunk_size=1000, overlap=200)...
✓ Split into 8500+ chunks

Creating Chroma vector store...
✓ Vector database created and persisted to ./chalmers_chroma_db
```

### 第二步：配置 OpenAI API

复制环境变量模板：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 OpenAI API Key：

```
OPENAI_API_KEY=sk-your-actual-key-here
```

获取 API Key：https://platform.openai.com/api-keys

### 第三步：运行 RAG 问答系统

```bash
python rag_query_system.py
```

**系统启动后进入交互模式：**

```
🎓 Chalmers Course Assistant - Interactive Mode
======================================================================

Ask me anything about Chalmers courses!
Examples:
  - What machine learning courses are available?
  - Can I take DAT450 and TDA357 together?
  - Tell me about courses in block C
  - What are the prerequisites for database courses?
  - Which courses are open for exchange students?

Type 'quit', 'exit', or 'q' to stop.
======================================================================

💬 You: 
```

## 示例查询

### 1. 查找特定主题的课程
```
💬 You: What machine learning courses are available?

🤖 Assistant: Based on the course database, here are machine learning courses:
- **TDA233** Algoritmer för maskininlärning och slutledning (7.5 credits)
- **TIF285** Bayesiansk dataanalys och maskininlärning (7.5 credits)
...
```

### 2. 检查课程时间冲突（核心功能）
```
💬 You: Can I take TDA357 and DAT450 together?

🤖 Assistant: Let me check the schedule blocks:
- TDA357 (Databases): Block D
- DAT450 (Advanced NLP): Block C

✓ These courses do NOT conflict - they are in different blocks!
```

### 3. 查询先修课程
```
💬 You: What are the prerequisites for database courses?

🤖 Assistant: For **TDA357** (Databases):
Prerequisites: "Object-oriented programming and basic data structures"
Course URL: https://www.chalmers.se/en/.../TDA357/...
```

### 4. 查询交换生可选课程
```
💬 You: Which AI courses are open for exchange students?

🤖 Assistant: The following AI courses are open for exchange students:
- **DAT450** Advanced NLP (Block C, 7.5 credits)
  Open for Exchange: Yes
...
```

## RAG 系统核心功能

### 智能调度冲突检测

系统会自动检查课程的 **Schedule Block** 字段：
- 相同 Block → **时间冲突**（如 Block C vs Block C）
- 不同 Block → 无冲突（如 Block C vs Block D）
- Block C+/D+ → 提示可能的部分冲突

### 精确的上下文检索

每次查询检索 **Top-5** 最相关的课程文档，使用：
- **语义相似度搜索**：基于 sentence-transformers 嵌入向量
- **元数据过滤**：可按 Block、学分、语言等过滤

### 结构化输出

AI 助手会在回答中包含：
- ✅ 课程代码（如 TDA357）
- ✅ 课程 URL（供用户深入了解）
- ✅ 先修课程、学分、语言、项目等关键信息
- ✅ 时间冲突警告（加粗显示）

## 技术细节

### Document 结构

每门课程被转换为包含以下内容的 Document：

**page_content（用于向量化）:**
```
Course Code: DAT450
Title: Advanced NLP
Credits: 7.5
Language: English
Schedule Block: C
Study Period: Sp1, Sp2

Prerequisites: Machine learning basics

Eligibility: ...
Open for Exchange Students: Yes

Programs:
MPDSC - Data Science and AI, Year 2

Learning Outcomes:
The student will be able to...
[详细的学习成果文本]
```

**metadata（用于过滤）:**
```json
{
  "course_code": "DAT450",
  "title": "Advanced NLP",
  "url": "https://...",
  "block": "C",
  "credits": 7.5,
  "language": "English",
  "sp": "Sp1, Sp2",
  "open_for_exchange": true
}
```

### Prompt 工程

系统使用精心设计的 Prompt 模板指导 AI：

1. **仅使用检索到的上下文**：避免幻觉
2. **时间冲突检测逻辑**：明确定义如何比较 Block 字段
3. **格式化要求**：加粗关键信息、使用列表
4. **引用要求**：包含课程代码和 URL

### 文本切分策略

- **chunk_size=1000**: 每块最多 1000 字符
- **overlap=200**: 块之间重叠 200 字符，避免信息断裂
- **递归分隔符**: 优先按段落 `\n\n`，再按句子 `\n`，最后按单词 ` `

## 文件说明

| 文件 | 用途 |
|------|------|
| `build_vector_db.py` | 构建向量数据库（运行一次） |
| `rag_query_system.py` | 交互式问答系统（主程序） |
| `chalmers_courses_full_scraped.json` | 原始课程数据（1122 门） |
| `chalmers_chroma_db/` | 持久化的向量数据库 |
| `.env` | OpenAI API Key 配置 |

## 性能优化建议

### 使用 GPU 加速

修改两个脚本中的 `model_kwargs`：
```python
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device': 'cuda'},  # 改为 cuda
    encode_kwargs={'normalize_embeddings': True}
)
```

### 使用更强大的 LLM

在 `rag_query_system.py` 中：
```python
LLM_MODEL = 'gpt-4'  # 或 'gpt-4-turbo'
```

### 调整检索数量

增加检索文档数量以获得更全面的上下文：
```python
RETRIEVAL_K = 10  # 默认为 5
```

## 本地模型选项（无需 OpenAI）

如果不想使用 OpenAI API，可以配置本地模型。在 `rag_query_system.py` 中取消注释 `create_llm()` 函数中的 HuggingFace 部分：

```python
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
# ... (见代码中的完整实现)
```

**注意**：本地模型需要大量内存和 GPU。

## 常见问题

### Q: 构建数据库需要多久？
A: 对于 1122 门课程，首次运行约需 **5-10 分钟**（取决于 CPU/GPU）。后续使用已保存的数据库无需重建。

### Q: 如何更新数据库？
A: 重新运行 `python build_vector_db.py`，旧数据库会自动被覆盖。

### Q: 查询响应慢？
A: 
- 检查是否使用 GPU (`device='cuda'`)
- 减少 `RETRIEVAL_K` 值
- 考虑使用更快的嵌入模型（但可能牺牲精度）

### Q: OpenAI API 费用？
A: 使用 `gpt-3.5-turbo` 每次查询约 $0.001-0.002。可设置预算限制。

## 扩展功能建议

1. **Web 界面**：使用 Streamlit 或 Gradio 创建可视化界面
2. **多语言支持**：添加中文翻译功能
3. **课程推荐**：基于学生背景推荐课程组合
4. **时间表生成**：自动生成无冲突的课程计划
5. **历史记录**：保存对话历史供后续参考

## 致谢

- **LangChain**: RAG 框架
- **Chroma**: 向量数据库
- **Sentence Transformers**: 文本嵌入模型
- **OpenAI**: GPT 语言模型
