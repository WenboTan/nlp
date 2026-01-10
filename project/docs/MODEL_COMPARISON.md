# 🤖 模型选择指南

本项目提供两种 RAG 实现方式，各有优缺点。

---

## 📊 对比表格

| 特性 | **本地模型** (Local) | **OpenAI API** |
|------|---------------------|----------------|
| **脚本文件** | `rag_query_system_local.py` | `rag_query_system_openai.py` |
| **成本** | ✅ 完全免费 | ❌ 按使用付费 (~$0.01-0.04/查询) |
| **回答质量** | ⚠️ 良好 (Mistral-7B) | ✅ 优秀 (GPT-4o-mini) |
| **启动时间** | ❌ 慢 (首次 5-10 分钟，之后 ~1 分钟) | ✅ 快 (~5 秒) |
| **运行速度** | ⚠️ 中等 (~10 秒/查询) | ✅ 快 (~2-3 秒/查询) |
| **GPU 需求** | ✅ 需要 (~14GB 显存) | ✅ 不需要 |
| **网络需求** | ✅ 仅首次下载模型 | ❌ 每次查询都需要 |
| **推荐场景** | 开发、调试、大量测试 | 演示、最终提交、少量查询 |

---

## 🚀 使用方法

### 方式 A：本地模型（免费）

#### 1. 安装依赖
```bash
pip install -r requirements_local.txt
```

#### 2. 运行
```bash
# 交互式（推荐）
srun -p gpu --gres=gpu:1 --mem=32G --time=1:00:00 --pty bash
python rag_query_system_local.py

# 批量任务
sbatch run_rag_local.sh
```

#### 3. 模型选择
编辑 [`rag_query_system_local.py`](rag_query_system_local.py#L26)：
```python
# 当前使用（推荐）
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # ~14GB 显存

# 备选模型
# LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"  # ~8GB 显存
# LLM_MODEL = "Qwen/Qwen2-7B-Instruct"            # 支持中文
```

---

### 方式 B：OpenAI API（付费）

#### 1. 安装依赖
```bash
pip install -r requirements_openai.txt
```

#### 2. 配置 API Key
```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 文件，填入你的 API Key
nano .env
```

`.env` 内容：
```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
```

获取 API Key：https://platform.openai.com/api-keys

#### 3. 运行
```bash
# 交互式（推荐）
python rag_query_system_openai.py

# 批量任务
sbatch run_rag_openai.sh
```

#### 4. 模型选择
编辑 [`rag_query_system_openai.py`](rag_query_system_openai.py#L28)：
```python
# 当前使用（推荐，性价比最高）
OPENAI_MODEL = "gpt-4o-mini"  # $0.15/$0.60 per 1M tokens

# 备选模型
# OPENAI_MODEL = "gpt-3.5-turbo"  # 更便宜但质量稍差
# OPENAI_MODEL = "gpt-4o"         # 质量最好但较贵
```

---

## 💰 OpenAI 成本估算

### 价格（2026年1月）
- **GPT-4o-mini**: $0.15 输入 / $0.60 输出（每百万 tokens）
- **GPT-3.5-turbo**: $0.50 输入 / $1.50 输出
- **GPT-4o**: $2.50 输入 / $10.00 输出

### 实际使用
每次查询约 3000-5000 tokens（检索上下文 + 问题 + 回答）

| 场景 | 查询次数 | 估算成本 (GPT-4o-mini) |
|------|---------|------------------------|
| 调试测试 | 10 次 | ~$0.10 |
| 项目演示 | 20 次 | ~$0.30 |
| 完整评估 | 100 次 | ~$1.50 |
| 大量使用 | 500 次 | ~$7.50 |

---

## 🎯 推荐策略

### 开发阶段 ✅
使用**本地模型**：
- 完全免费
- 可以无限测试
- 质量足够好（Mistral-7B）

```bash
python rag_query_system_local.py
```

### 演示/提交阶段 💎
使用 **OpenAI API**：
- 回答质量最高
- 启动速度快
- 少量查询成本很低

```bash
python rag_query_system_openai.py
```

### 混合方案 🎭
- 平时开发用本地模型（免费）
- 重要演示用 OpenAI（少量成本）
- 两个脚本都保留，随时切换

---

## 🔧 故障排除

### 本地模型问题

**问题**: 显存不足
```
RuntimeError: CUDA out of memory
```
**解决**: 
1. 编辑脚本，改用更小的模型：
   ```python
   LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"  # 只需 8GB
   ```
2. 或启用 8-bit 量化：
   ```python
   llm = load_local_llm(LLM_MODEL, use_8bit=True)
   ```

**问题**: 模型下载慢
```
Downloading...
```
**解决**: 首次运行需要下载 ~13GB 模型，需耐心等待 5-10 分钟

---

### OpenAI 问题

**问题**: API Key 无效
```
❌ OPENAI_API_KEY not found!
```
**解决**: 
```bash
cp .env.example .env
nano .env  # 填入真实 API Key
```

**问题**: 余额不足
```
You exceeded your current quota
```
**解决**: 
1. 访问 https://platform.openai.com/account/billing
2. 添加付款方式
3. 充值（建议 $5-10 起）

---

## 📚 更多信息

- **本地模型详解**: [LOCAL_MODEL_GUIDE.md](LOCAL_MODEL_GUIDE.md)
- **完整使用指南**: [README.md](README.md)
- **快速启动**: [START_HERE.md](START_HERE.md)
