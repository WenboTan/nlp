# 🚀 使用 Google Gemini API 快速指南

## 为什么选择 Gemini？

✅ **优势**：
- 免费额度高（Gemini Flash 每分钟15个请求免费）
- 回答质量接近 GPT-4o-mini
- 速度快（Flash 版本）
- 支持中文

❌ **缺点**：
- 需要网络连接
- 免费额度有限制

---

## 📋 快速开始（3步）

### 1️⃣ 获取 Gemini API Key

访问以下任一链接：
- https://makersuite.google.com/app/apikey
- https://aistudio.google.com/app/apikey

点击 "Create API Key" 获取你的密钥（格式类似：`AIzaSyC...`）

### 2️⃣ 配置 API Key

```bash
cd /data/users/wenbota/nlp/project

# 复制配置模板
cp .env.example .env

# 编辑 .env 文件
nano .env
```

在 .env 文件中添加：
```bash
GOOGLE_API_KEY=your-actual-api-key-here
```

保存并退出（Ctrl+X, 然后 Y, 然后 Enter）

### 3️⃣ 安装依赖

```bash
# 如果你已经安装了其他版本的依赖，直接安装 Gemini 额外依赖
pip install -U langchain-google-genai

# 或者完整安装
pip install --user -r requirements_gemini.txt
```

---

## 🎯 使用方式

### 方式 A: 交互式（推荐）

```bash
# 直接运行（不需要 GPU！）
python rag_query_system_gemini.py
```

### 方式 B: SLURM 批处理

```bash
# 提交任务
sbatch run_rag_gemini.sh

# 查看日志
tail -f logs/rag_gemini_*.out
```

### 方式 C: 批量测试

```bash
# 运行10个预设测试
python test_rag_batch_gemini.py
```

---

## 💬 示例对话

```
💬 You: What machine learning courses are available?

🤖 Assistant: Based on the course database, here are machine learning 
courses at Chalmers:

1. TDA231 - Machine Learning (7.5 credits)
   - Block: C
   - Language: English
   - Prerequisites: Basic programming, linear algebra
   
2. DAT340 - Applied Machine Learning (7.5 credits)
   - Block: A
   - Prerequisites: TDA231 or equivalent
   ...
```

---

## 📊 模型选择

编辑 `rag_query_system_gemini.py` 中的模型：

```python
# 推荐（免费额度高，速度快）
GEMINI_MODEL = "gemini-2.5-flash"

# 质量更高但较慢
# GEMINI_MODEL = "gemini-2.5-pro"

# 实验版本
# GEMINI_MODEL = "gemini-2.0-flash-exp"
```

---

## 💰 免费额度

| 模型 | 免费额度（每分钟） | 质量 | 速度 |
|------|-------------------|------|------|
| gemini-2.5-flash | 15 请求 | ⭐⭐⭐⭐⭐ | 🚀🚀🚀 |
| gemini-2.5-pro | 2 请求 | ⭐⭐⭐⭐⭐ | 🚀🚀 |
| gemini-2.0-flash-exp | 10 请求 | ⭐⭐⭐⭐ | 🚀🚀🚀 |

详见: https://ai.google.dev/pricing

---

## 🆘 常见问题

### Q: 如何检查 API Key 是否有效？

```bash
python -c "
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
print(f'API Key: {api_key[:10]}...{api_key[-4:]}')

llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', google_api_key=api_key)
response = llm.invoke('Hello!')
print('✅ API Key is valid!')
print(f'Response: {response.content}')
"
```

### Q: 出现 "API Key not found" 错误？

确保：
1. `.env` 文件存在于项目根目录
2. 文件中有 `GOOGLE_API_KEY=...`
3. API Key 没有多余的空格或引号

### Q: 速率限制错误？

免费额度有限制，可以：
1. 等待几分钟后重试
2. 升级到付费账户
3. 使用本地模型（免费无限制）

### Q: 对比 OpenAI 和 Gemini 哪个更好？

| 特性 | OpenAI GPT-4o-mini | Gemini 1.5 Flash |
|------|-------------------|------------------|
| 质量 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 速度 | 🚀🚀🚀 | 🚀🚀🚀 |
| 免费额度 | 无免费额度 | 每分钟15次 |
| 价格 | $0.15/$0.60 per 1M tokens | 免费或更便宜 |

**建议**：先用 Gemini（免费），如果需要更高质量再用 OpenAI。

---

## ✨ 完整工作流

```bash
# 1. 获取 API Key
# 访问: https://aistudio.google.com/app/apikey

# 2. 配置
cd /data/users/wenbota/nlp/project
cp .env.example .env
nano .env  # 添加 GOOGLE_API_KEY

# 3. 安装依赖
pip install -U langchain-google-genai

# 4. 测试
python rag_query_system_gemini.py

# 5. 开始提问！
💬 You: Tell me about database courses
```

---

## 📚 相关文件

- `rag_query_system_gemini.py` - 交互式 Gemini RAG 系统
- `test_rag_batch_gemini.py` - 批量测试脚本
- `requirements_gemini.txt` - Gemini 版本依赖
- `run_rag_gemini.sh` - SLURM 批处理脚本
- `.env` - API 密钥配置（需自己创建）

---

需要帮助？查看主文档：
- README.md
- MODEL_COMPARISON.md
- PROJECT_INTRODUCTION.txt
