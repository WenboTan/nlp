# 🚀 使用本地模型构建 RAG 系统（无需 OpenAI）

## 为什么使用本地模型？

### OpenAI 方案的问题
- ❌ 需要 API Key 和付费
- ❌ 数据发送到外部服务器
- ❌ 受网络限制
- ❌ 每次查询都有成本

### 本地模型的优势
- ✅ **完全免费**（一次性下载）
- ✅ **数据隐私**（本地运行）
- ✅ **无网络依赖**
- ✅ **可以在学校 GPU 集群训练**
- ✅ **完全可控**（可以微调模型）

---

## 🎯 两种本地方案

### 方案 1: 使用预训练开源模型（推荐）
**无需训练，直接使用**

适合的开源模型：
- **Mistral-7B-Instruct** - 7B 参数，质量高
- **Llama-3-8B-Instruct** - Meta 的最新模型
- **Qwen2-7B-Instruct** - 阿里巴巴的中英双语模型
- **Phi-3-mini** - 微软的 3.8B 轻量模型

### 方案 2: 训练/微调自己的模型
**基于你的课程数据微调**

---

## 📦 方案 1：直接使用开源模型

### Step 1: 检查 GPU 可用性

```bash
# 检查 CUDA
nvidia-smi

# 检查 PyTorch GPU 支持
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU数量: {torch.cuda.device_count()}')"
```

### Step 2: 安装依赖

```bash
cd /data/users/wenbota/nlp/project

# 安装必要的包
pip install torch transformers accelerate bitsandbytes
pip install langchain langchain-community langchain-chroma chromadb sentence-transformers
```

### Step 3: 修改 `rag_query_system.py` 使用本地模型

创建新文件 `rag_query_system_local.py`：

```python
"""
本地模型版本的 RAG 问答系统
使用 HuggingFace 模型，无需 OpenAI API
"""

import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


# ===== 配置 =====
DB_PATH = './chalmers_chroma_db'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

# 选择你的 LLM 模型（根据 GPU 显存选择）
# LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # 需要 ~14GB 显存
# LLM_MODEL = "meta-llama/Llama-3-8B-Instruct"      # 需要 ~16GB 显存
LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"      # 只需 ~8GB 显存（推荐）
# LLM_MODEL = "Qwen/Qwen2-7B-Instruct"              # 中英双语

RETRIEVAL_K = 5
MAX_NEW_TOKENS = 512


def load_local_llm(model_name: str, device: str = 'cuda'):
    """
    加载本地 LLM 模型
    
    Args:
        model_name: HuggingFace 模型名称
        device: 'cuda' 或 'cpu'
    
    Returns:
        HuggingFacePipeline 实例
    """
    print(f"Loading local LLM: {model_name}")
    print(f"Device: {device}")
    print("This may take a few minutes on first run (downloading model)...")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 加载模型
    # 使用 8-bit 量化以减少显存占用
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',              # 自动分配到 GPU
        torch_dtype=torch.float16,      # 使用 FP16 减少显存
        load_in_8bit=True,              # 8-bit 量化（可选，进一步减少显存）
        trust_remote_code=True          # 某些模型需要
    )
    
    print(f"✓ Model loaded on {device}")
    print(f"  - Parameters: ~{sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
    print(f"  - Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # 创建 text-generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.3,                # 低温度 = 更确定的回答
        top_p=0.9,
        repetition_penalty=1.1
    )
    
    # 包装为 LangChain LLM
    llm = HuggingFacePipeline(pipeline=pipe)
    
    return llm


def load_vector_store(db_path: str = DB_PATH) -> Chroma:
    """加载向量数据库"""
    print(f"Loading vector database from {db_path}...")
    
    if not Path(db_path).exists():
        raise FileNotFoundError(
            f"Vector database not found at {db_path}. "
            "Please run build_vector_db.py first."
        )
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name='chalmers_courses'
    )
    
    print(f"✓ Vector store loaded")
    return vectorstore


def create_rag_prompt() -> ChatPromptTemplate:
    """创建 RAG Prompt"""
    system_message = """You are a helpful Chalmers University Course Assistant. 
Answer questions about courses using the provided context.

IMPORTANT:
1. Use ONLY the retrieved context below
2. For schedule conflicts: Check "Block" field
   - Same Block = TIME CONFLICT
   - Different Block = NO CONFLICT
3. Include course codes and URLs in answers
4. If unsure, say "I don't have enough information"

Context:
{context}

Question: {question}

Answer:"""
    
    return ChatPromptTemplate.from_template(system_message)


def format_docs(docs) -> str:
    """格式化检索到的文档"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        formatted.append(f"--- Document {i} ---")
        formatted.append(f"Course: {doc.metadata.get('course_code', 'Unknown')}")
        formatted.append(f"Block: {doc.metadata.get('block', 'Unknown')}")
        formatted.append(f"URL: {doc.metadata.get('url', 'N/A')}")
        formatted.append(f"\n{doc.page_content}\n")
    return "\n".join(formatted)


def create_rag_chain(vectorstore: Chroma, llm, k: int = RETRIEVAL_K):
    """创建 RAG 链"""
    retriever = vectorstore.as_retriever(
        search_type='similarity',
        search_kwargs={'k': k}
    )
    
    prompt = create_rag_prompt()
    
    rag_chain = (
        {
            'context': retriever | format_docs,
            'question': RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


def interactive_query_loop(rag_chain):
    """交互式问答循环"""
    print("\n" + "=" * 70)
    print("🎓 Chalmers Course Assistant - Local Model Mode")
    print("=" * 70)
    print("\nAsk me anything about Chalmers courses!")
    print("Type 'quit', 'exit', or 'q' to stop.")
    print("=" * 70 + "\n")
    
    while True:
        try:
            question = input("\n💬 You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q', 'bye']:
                print("\n👋 Goodbye!")
                break
            
            if not question:
                continue
            
            print("\n🤖 Assistant: ", end='', flush=True)
            response = rag_chain.invoke(question)
            print(response)
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n⚠ Error: {e}")


def main():
    """主函数"""
    print("=" * 70)
    print("Chalmers Course RAG System - Local Model")
    print("=" * 70)
    
    # 检查 GPU
    if torch.cuda.is_available():
        print(f"\n✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        device = 'cuda'
    else:
        print("\n⚠ No GPU detected, using CPU (will be slow)")
        device = 'cpu'
    
    try:
        # 1. 加载向量数据库
        vectorstore = load_vector_store()
        
        # 2. 加载本地 LLM
        print(f"\nInitializing local LLM...")
        llm = load_local_llm(LLM_MODEL, device=device)
        print("✓ LLM ready")
        
        # 3. 创建 RAG 链
        print(f"\nBuilding RAG chain (k={RETRIEVAL_K})...")
        rag_chain = create_rag_chain(vectorstore, llm, k=RETRIEVAL_K)
        print("✓ RAG chain ready")
        
        # 4. 启动交互式问答
        interactive_query_loop(rag_chain)
    
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\nPlease run: python build_vector_db.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
```

### Step 4: 运行本地 RAG 系统

```bash
# 首先构建向量数据库（如果还没有）
python build_vector_db.py

# 使用本地模型启动
python rag_query_system_local.py
```

**首次运行会下载模型（约 5-15 分钟，取决于网速）**

---

## 📊 模型选择指南

| 模型 | 参数量 | 显存需求 | 质量 | 推荐场景 |
|------|--------|----------|------|----------|
| **Phi-3-mini** | 3.8B | ~8GB | 中等 | GPU 显存有限 |
| **Mistral-7B** | 7B | ~14GB | 高 | 有足够显存 |
| **Llama-3-8B** | 8B | ~16GB | 很高 | 最佳质量 |
| **Qwen2-7B** | 7B | ~14GB | 高 | 中英双语 |

**在 Minerva 集群上**: 通常有 24GB+ 显存，可以选择任何模型。

---

## 🔧 GPU 优化技巧

### 1. 使用量化减少显存

```python
# 8-bit 量化（推荐）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,  # 显存减半
    device_map='auto'
)

# 4-bit 量化（更激进）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,  # 显存减至 1/4
    device_map='auto'
)
```

### 2. 批处理嵌入

在 `build_vector_db.py` 中使用 GPU：

```python
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device': 'cuda'},  # 使用 GPU
    encode_kwargs={
        'normalize_embeddings': True,
        'batch_size': 32  # 批量处理
    }
)
```

### 3. 在 SLURM 集群上运行

创建 `run_rag.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=rag_chalmers
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

module load Python/3.10.4
module load CUDA/11.7

cd /data/users/wenbota/nlp/project

# 构建数据库（如果需要）
python build_vector_db.py

# 启动 RAG 系统
python rag_query_system_local.py
```

提交任务：
```bash
sbatch run_rag.sh
```

---

## 🎯 方案 2：微调自己的模型

### 为什么要微调？

使用 1122 门课程数据微调模型，可以：
- ✅ 更好地理解 Chalmers 课程术语
- ✅ 更准确的 Block 冲突检测
- ✅ 学习瑞典语课程名称
- ✅ 记住常见课程代码

### 微调步骤

#### 1. 准备训练数据

创建 `prepare_finetune_data.py`:

```python
import json
from datasets import Dataset

# 加载课程数据
with open('chalmers_courses_full_scraped.json', 'r') as f:
    courses = json.load(f)

# 创建 QA 对
qa_pairs = []

for course in courses:
    code = course['course_code']
    title = course['title']
    block = course['logistics']['block']
    credits = course['logistics']['credits']
    prereq = course['constraints']['prerequisites']
    
    # 生成多种问题
    qa_pairs.extend([
        {
            'instruction': f"What is {code}?",
            'response': f"{code} is {title}, a {credits} credit course in Block {block}."
        },
        {
            'instruction': f"Tell me about course {code}",
            'response': course['rag_text']['learning_outcomes'][:500]
        },
        {
            'instruction': f"What are the prerequisites for {code}?",
            'response': f"Prerequisites: {prereq}"
        },
        {
            'instruction': f"What block is {code} in?",
            'response': f"Block {block}"
        }
    ])

# 保存为 JSONL
dataset = Dataset.from_list(qa_pairs)
dataset.save_to_disk('chalmers_course_qa_dataset')
print(f"Created {len(qa_pairs)} QA pairs")
```

#### 2. 微调模型

使用 HuggingFace `trl` 库：

```bash
pip install trl peft
```

创建 `finetune_model.py`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer
from datasets import load_from_disk

# 加载基础模型
model_name = "microsoft/Phi-3-mini-4k-instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 加载数据
dataset = load_from_disk('chalmers_course_qa_dataset')

# 训练
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    max_seq_length=512,
    args=TrainingArguments(
        output_dir="./chalmers_course_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        fp16=True
    )
)

trainer.train()
trainer.save_model("./chalmers_course_model_final")
```

#### 3. 使用微调后的模型

在 `rag_query_system_local.py` 中：

```python
LLM_MODEL = "./chalmers_course_model_final"  # 使用你微调的模型
```

---

## 🆚 对比总结

| 方案 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| **OpenAI GPT** | 最高质量，即开即用 | 需付费，依赖网络 | ⭐⭐⭐ |
| **预训练开源模型** | 免费，质量好 | 需要 GPU | ⭐⭐⭐⭐⭐ |
| **微调模型** | 最适配你的数据 | 需要训练时间 | ⭐⭐⭐⭐ |

---

## 🚀 推荐路线

### 对于你的情况（学校集群 + GPU）：

1. **第一步**: 使用预训练模型（Phi-3-mini）
   - 快速验证 RAG 系统可行性
   - 不需要训练

2. **第二步**（可选）: 微调模型
   - 使用课程数据微调
   - 提升专业领域表现

3. **第三步**（可选）: 部署到 Web
   - 使用 Gradio/Streamlit
   - 让其他人也能使用

---

## ✅ 完整工作流程

```bash
# 1. 构建向量数据库（CPU 即可）
python build_vector_db.py

# 2. 创建本地模型版本（上面提供的代码）
# 保存为 rag_query_system_local.py

# 3. 在 GPU 节点上运行
srun -p gpu --gres=gpu:1 --mem=32G python rag_query_system_local.py

# 4. 开始提问！
```

---

需要我帮你创建完整的 `rag_query_system_local.py` 文件吗？
