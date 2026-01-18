"""
Local RAG Query System using HuggingFace Models
Uses open-source models without API requirements

Suitable for running on GPU clusters
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


# Configuration
DB_PATH = './chalmers_chroma_db'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

# Select LLM model (based on GPU memory)
# Recommended models (by quality):
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # ~14GB VRAM, best quality

# Alternative models:
# LLM_MODEL = "microsoft/Phi-3-mini-4k-instruct"      # ~8GB VRAM, good quality
# LLM_MODEL = "Qwen/Qwen2-7B-Instruct"              # ~14GB VRAM, supports Chinese
# LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct" # ~16GB VRAM, requires auth

RETRIEVAL_K = 5
MAX_NEW_TOKENS = 512


def load_local_llm(model_name: str, device: str = 'cuda', use_8bit: bool = True):
    """
    Load local LLM model
    
    Args:
        model_name: HuggingFace model name or local path
        device: 'cuda' or 'cpu'
        use_8bit: Use 8-bit quantization to reduce VRAM
    
    Returns:
        HuggingFacePipeline instance
    """
    print(f"\n{'='*70}")
    print(f"Loading local LLM: {model_name}")
    print(f"Device: {device}")
    print(f"8-bit quantization: {use_8bit}")
    print("This may take a few minutes on first run (downloading model)...")
    print(f"{'='*70}\n")
    
    # Load分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # 确保有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load模型配置
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
    
    # Create text-generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.7,                # 适中温度，保持创造性
        top_p=0.9,
        top_k=50,                       # 限制候选词数量
        repetition_penalty=1.2,         # 强力避免重复
        no_repeat_ngram_size=3,         # 禁止3-gram重复
        do_sample=True,
        return_full_text=False          # 只返回新生成的文本
    )
    
    # 包装为 LangChain LLM
    llm = HuggingFacePipeline(pipeline=pipe)
    
    return llm


def load_vector_store(db_path: str = DB_PATH) -> Chroma:
    """Load vector database"""
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
    Create RAG prompt template
    Optimized for course assistant
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
    """Format retrieved documents as context string"""
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
    Create RAG chain (using LCEL)
    
    Args:
        vectorstore: Chroma Vector database
        llm: Language model
        k: Number of documents to retrieve
    
    Returns:
        Executable RAG chain
    """
    # CreateRetriever
    retriever = vectorstore.as_retriever(
        search_type='similarity',
        search_kwargs={'k': k}
    )
    
    # Create prompt
    prompt = create_rag_prompt()
    
    # Build chain: Retrieve -> Format -> Prompt -> LLM -> Parse
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
    """Interactive QA loop"""
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
            # GetUser input
            question = input("\n💬 You: ").strip()
            
            # CheckExit command
            if question.lower() in ['quit', 'exit', 'q', 'bye']:
                print(f"\n👋 Goodbye! Answered {query_count} questions.")
                break
            
            # SkipEmpty input
            if not question:
                continue
            
            # 查询 RAG system
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
    """Main function"""
    print("=" * 70)
    print("Chalmers Course RAG System - Local Model")
    print("=" * 70)
    
    # Check GPU 可用性
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
        # Step 1: Load vector database
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
