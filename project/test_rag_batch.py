"""
批处理测试脚本 - 用于在 SLURM 任务中测试 RAG 系统
不需要交互输入，直接运行预设的测试查询
"""

import os
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline
)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ===== 配置 =====
DB_PATH = './chalmers_chroma_db'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
# 使用更稳定兼容的模型
LLM_MODEL = "google/flan-t5-large"  # 更稳定，transformers 兼容性好
RETRIEVAL_K = 5
MAX_NEW_TOKENS = 512

# 测试查询列表
TEST_QUERIES = [
    "What machine learning courses are available?",
    "Tell me about database courses at Chalmers.",
    "What are the prerequisites for advanced programming courses?",
]


def load_local_llm(model_name: str, device: str = 'cuda', use_8bit: bool = False):
    """加载本地 LLM 模型"""
    print(f"\n{'='*70}")
    print(f"Loading local LLM: {model_name}")
    print(f"Device: {device}")
    print(f"8-bit quantization: {use_8bit}")
    print("This may take a few minutes on first run (downloading model)...")
    print(f"{'='*70}\n")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model_kwargs = {
        'trust_remote_code': True,
        'torch_dtype': torch.float16 if device == 'cuda' else torch.float32,
    }
    
    if device == 'cuda':
        model_kwargs['device_map'] = 'auto'
        if use_8bit:
            model_kwargs['load_in_8bit'] = True
    
    # 自动检测模型类型
    if 't5' in model_name.lower() or 'flan' in model_name.lower():
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
        task = "text2text-generation"
        pipeline_kwargs = {
            'max_new_tokens': MAX_NEW_TOKENS,
            'do_sample': True,
            'temperature': 0.1,
            'top_p': 0.95
        }
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
            attn_implementation='eager'
        )
        task = "text-generation"
        pipeline_kwargs = {
            'max_new_tokens': MAX_NEW_TOKENS,
            'do_sample': True,
            'temperature': 0.1,
            'top_p': 0.95,
            'return_full_text': False
        }
    
    text_generation_pipeline = pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        **pipeline_kwargs
    )
    
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    
    return llm


def load_vector_store():
    """加载向量数据库"""
    print(f"Loading vector database from {DB_PATH}...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': device}
    )
    
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    
    print(f"✓ Vector store loaded (using {device} for embeddings)")
    return vectorstore


def format_docs(docs):
    """格式化检索到的文档"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata
        course_code = metadata.get('course_code', 'N/A')
        title = metadata.get('title', 'N/A')
        block = metadata.get('block', 'N/A')
        url = metadata.get('url', '')
        
        doc_str = f"Course {i}: {course_code} - {title}\n"
        doc_str += f"Block: {block}\n"
        if url:
            doc_str += f"URL: {url}\n"
        doc_str += f"Content: {doc.page_content[:500]}...\n"
        
        formatted.append(doc_str)
    
    return "\n".join(formatted)


def create_rag_chain(vectorstore, llm, k: int = 5):
    """创建 RAG 链"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    
    template = """You are a helpful academic advisor at Chalmers University of Technology.
Use the following course information to answer the student's question.

If you notice that multiple courses are scheduled in the same block (e.g., both in Block A or Block B), 
explicitly warn the student that these courses may have scheduling conflicts.

Course Information:
{context}

Question: {question}

Answer: Provide a helpful, detailed answer based on the course information. If there are 
scheduling conflicts (courses in the same block), mention this explicitly.
Include relevant course codes and links when applicable."""

    prompt = ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


def main():
    """主函数 - 运行批处理测试"""
    print("=" * 70)
    print("Chalmers Course RAG System - Batch Test Mode")
    print("=" * 70)
    
    # 检查 GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n✓ GPU detected: {gpu_name}")
        print(f"  Total memory: {gpu_memory:.1f} GB")
        device = 'cuda'
    else:
        print("\n⚠ No GPU detected, using CPU")
        device = 'cpu'
    
    try:
        # 加载组件
        print("\n" + "=" * 70)
        print("Step 1: Loading Vector Database")
        print("=" * 70)
        vectorstore = load_vector_store()
        
        print("\n" + "=" * 70)
        print("Step 2: Initializing Local LLM")
        print("=" * 70)
        llm = load_local_llm(LLM_MODEL, device=device, use_8bit=False)
        print("✓ LLM ready")
        
        print("\n" + "=" * 70)
        print("Step 3: Building RAG Chain")
        print("=" * 70)
        rag_chain = create_rag_chain(vectorstore, llm, k=RETRIEVAL_K)
        print("✓ RAG chain ready")
        
        # 运行测试查询
        print("\n" + "=" * 70)
        print("Running Test Queries")
        print("=" * 70)
        
        for i, query in enumerate(TEST_QUERIES, 1):
            print(f"\n{'='*70}")
            print(f"Test Query {i}/{len(TEST_QUERIES)}")
            print(f"{'='*70}")
            print(f"Question: {query}")
            print(f"\nGenerating answer...")
            
            try:
                answer = rag_chain.invoke(query)
                print(f"\n🤖 Answer:\n{answer}")
            except Exception as e:
                print(f"\n❌ Error generating answer: {str(e)}")
        
        print("\n" + "=" * 70)
        print("✅ All tests completed!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
