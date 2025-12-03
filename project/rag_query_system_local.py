"""
本地模型版本的 RAG 问答系统
使用 HuggingFace 开源模型，无需 OpenAI API

适合在学校 GPU 集群上运行
"""

import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ===== 配置 =====
DB_PATH = './chalmers_chroma_db'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

# 选择你的 LLM 模型（根据 GPU 显存选择）
# 小型模型（推荐开始用这个）:
LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"      # ~8GB 显存

# 中型模型（如果有足够显存）:
# LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # ~14GB 显存
# LLM_MODEL = "Qwen/Qwen2-7B-Instruct"              # ~14GB 显存，支持中文

# 大型模型（如果有 24GB+ 显存）:
# LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct" # ~16GB 显存

RETRIEVAL_K = 5
MAX_NEW_TOKENS = 512


def load_local_llm(model_name: str, device: str = 'cuda', use_8bit: bool = True):
    """
    加载本地 LLM 模型
    
    Args:
        model_name: HuggingFace 模型名称或本地路径
        device: 'cuda' 或 'cpu'
        use_8bit: 是否使用 8-bit 量化以减少显存
    
    Returns:
        HuggingFacePipeline 实例
    """
    print(f"\n{'='*70}")
    print(f"Loading local LLM: {model_name}")
    print(f"Device: {device}")
    print(f"8-bit quantization: {use_8bit}")
    print("This may take a few minutes on first run (downloading model)...")
    print(f"{'='*70}\n")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # 确保有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型配置
    model_kwargs = {
        'trust_remote_code': True,
        'torch_dtype': torch.float16 if device == 'cuda' else torch.float32,
    }
    
    if device == 'cuda':
        model_kwargs['device_map'] = 'auto'
        if use_8bit:
            model_kwargs['load_in_8bit'] = True
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    # 如果是 CPU 模式，手动移到 CPU
    if device == 'cpu':
        model = model.to('cpu')
    
    print(f"✓ Model loaded successfully!")
    if device == 'cuda':
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
        print(f"  - Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  - Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    # 创建 text-generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.3,                # 低温度 = 更确定性的回答
        top_p=0.9,
        repetition_penalty=1.15,        # 避免重复
        do_sample=True
    )
    
    # 包装为 LangChain LLM
    llm = HuggingFacePipeline(pipeline=pipe)
    
    return llm


def load_vector_store(db_path: str = DB_PATH) -> Chroma:
    """加载向量数据库"""
    print(f"Loading vector database from {db_path}...")
    
    if not Path(db_path).exists():
        raise FileNotFoundError(
            f"Vector database not found at {db_path}.\n"
            "Please run: python build_vector_db.py"
        )
    
    # 使用 GPU 加速嵌入（如果可用）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name='chalmers_courses'
    )
    
    print(f"✓ Vector store loaded (using {device} for embeddings)")
    return vectorstore


def create_rag_prompt() -> ChatPromptTemplate:
    """
    创建 RAG Prompt 模板
    针对课程助手优化
    """
    system_message = """You are a helpful Chalmers University Course Assistant. 
Your job is to answer questions about courses using the provided context.

IMPORTANT INSTRUCTIONS:

1. **Use Only Retrieved Context**: Base your answers strictly on the context provided below. 
   If you cannot find the answer in the context, say "I don't have enough information about that in the course database."

2. **Schedule Conflict Detection**: 
   - When a user asks if they can take multiple courses together (e.g., "Can I take course A and B?"), 
     CHECK THE "Schedule Block" field in the context.
   - If two courses have the **same Block** (e.g., both are "Block C"), they have a TIME CONFLICT.
   - Example: Block "C" conflicts with "C", Block "D" conflicts with "D"
   - If blocks are different, courses do NOT conflict.
   
3. **Include Key Information**:
   - Always mention the course code when discussing a course
   - Include the course URL if available
   - Mention prerequisites, credits, language when relevant

4. **Be Concise and Clear**:
   - Use bullet points for multiple courses
   - Highlight important information like conflicts or prerequisites
   - Provide direct, actionable answers

Context from course database:
{context}

Question: {question}

Answer:"""
    
    return ChatPromptTemplate.from_template(system_message)


def format_docs(docs) -> str:
    """格式化检索到的文档为上下文字符串"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        formatted.append(f"--- Document {i} ---")
        formatted.append(f"Course Code: {doc.metadata.get('course_code', 'Unknown')}")
        formatted.append(f"Schedule Block: {doc.metadata.get('block', 'Unknown')}")
        formatted.append(f"Credits: {doc.metadata.get('credits', 'Unknown')}")
        formatted.append(f"URL: {doc.metadata.get('url', 'N/A')}")
        formatted.append(f"\n{doc.page_content}\n")
    return "\n".join(formatted)


def create_rag_chain(vectorstore: Chroma, llm, k: int = RETRIEVAL_K):
    """
    创建 RAG 链（使用 LCEL）
    
    Args:
        vectorstore: Chroma 向量数据库
        llm: 语言模型
        k: 检索的文档数量
    
    Returns:
        可执行的 RAG 链
    """
    # 创建检索器
    retriever = vectorstore.as_retriever(
        search_type='similarity',
        search_kwargs={'k': k}
    )
    
    # 创建 prompt
    prompt = create_rag_prompt()
    
    # 构建链: Retrieve -> Format -> Prompt -> LLM -> Parse
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
    print("\nExamples:")
    print("  - What machine learning courses are available?")
    print("  - Can I take TDA357 and DAT450 together?")
    print("  - Tell me about courses in block C")
    print("  - What are the prerequisites for database courses?")
    print("\nType 'quit', 'exit', or 'q' to stop.")
    print("=" * 70 + "\n")
    
    query_count = 0
    
    while True:
        try:
            # 获取用户输入
            question = input("\n💬 You: ").strip()
            
            # 检查退出命令
            if question.lower() in ['quit', 'exit', 'q', 'bye']:
                print(f"\n👋 Goodbye! Answered {query_count} questions.")
                break
            
            # 跳过空输入
            if not question:
                continue
            
            # 查询 RAG 系统
            print("\n🤖 Assistant: ", end='', flush=True)
            
            try:
                response = rag_chain.invoke(question)
                
                # 清理输出（移除可能的 prompt 残留）
                if "Answer:" in response:
                    response = response.split("Answer:")[-1].strip()
                
                print(response)
                query_count += 1
            
            except Exception as e:
                print(f"\n⚠ Error processing query: {e}")
                print("Please try again with a different question.")
        
        except KeyboardInterrupt:
            print(f"\n\n👋 Interrupted. Answered {query_count} questions. Goodbye!")
            break
        except Exception as e:
            print(f"\n⚠ Unexpected error: {e}")
            print("Continuing...")


def main():
    """主函数"""
    print("=" * 70)
    print("Chalmers Course RAG System - Local Model")
    print("=" * 70)
    
    # 检查 GPU 可用性
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n✓ GPU detected: {gpu_name}")
        print(f"  Total memory: {gpu_memory:.1f} GB")
        device = 'cuda'
        use_8bit = False  # 禁用 8-bit 量化（需要 bitsandbytes）
    else:
        print("\n⚠ No GPU detected, using CPU")
        print("  WARNING: CPU inference will be VERY slow!")
        print("  Recommendation: Use a GPU or switch to OpenAI API version")
        device = 'cpu'
        use_8bit = False
        
        # 询问是否继续
        response = input("\nContinue with CPU? (y/n): ").strip().lower()
        if response != 'y':
            print("Exiting. Please use a GPU or OpenAI version.")
            return
    
    try:
        # Step 1: 加载向量数据库
        print("\n" + "=" * 70)
        print("Step 1: Loading Vector Database")
        print("=" * 70)
        vectorstore = load_vector_store()
        
        # Step 2: 加载本地 LLM
        print("\n" + "=" * 70)
        print("Step 2: Initializing Local LLM")
        print("=" * 70)
        llm = load_local_llm(LLM_MODEL, device=device, use_8bit=use_8bit)
        print("✓ LLM ready")
        
        # Step 3: 创建 RAG 链
        print("\n" + "=" * 70)
        print("Step 3: Building RAG Chain")
        print("=" * 70)
        print(f"Retrieval strategy: Top-{RETRIEVAL_K} most relevant documents")
        rag_chain = create_rag_chain(vectorstore, llm, k=RETRIEVAL_K)
        print("✓ RAG chain ready")
        
        # Step 4: 启动交互式问答
        interactive_query_loop(rag_chain)
    
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\nPlease run the following first:")
        print("  python build_vector_db.py")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n" + "=" * 70)
        print("Troubleshooting Tips:")
        print("=" * 70)
        print("1. Check if you have enough GPU memory")
        print("2. Try a smaller model (e.g., Phi-3-mini)")
        print("3. Enable 8-bit quantization: use_8bit=True")
        print("4. Check internet connection (first run downloads model)")
        print("5. Install missing packages: pip install transformers accelerate bitsandbytes")


if __name__ == '__main__':
    main()
