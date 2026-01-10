"""
OpenAI API 版本的 RAG 问答系统
使用 OpenAI GPT 模型，需要 API Key

优点：
- 回答质量最高
- 推理能力强
- 启动速度快（无需加载模型）

缺点：
- 需要付费（约 $0.01-0.04 每次查询）
- 需要网络连接
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ===== 配置 =====
DB_PATH = './chalmers_chroma_db'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

# OpenAI 模型选择
OPENAI_MODEL = "gpt-4o-mini"  # 推荐：性价比最高（$0.15/$0.60 per 1M tokens）
# OPENAI_MODEL = "gpt-3.5-turbo"  # 便宜但质量稍差（$0.50/$1.50 per 1M tokens）
# OPENAI_MODEL = "gpt-4o"         # 最好但较贵（$2.50/$10.00 per 1M tokens）

RETRIEVAL_K = 5


def load_vector_store(db_path: str = DB_PATH) -> Chroma:
    """加载向量数据库"""
    print(f"Loading vector database from {db_path}...")
    
    if not Path(db_path).exists():
        raise FileNotFoundError(
            f"Vector database not found at {db_path}.\n"
            "Please run: python build_vector_db.py"
        )
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name='chalmers_courses'
    )
    
    print(f"✓ Vector store loaded")
    return vectorstore


def load_openai_llm(model: str = OPENAI_MODEL) -> ChatOpenAI:
    """
    加载 OpenAI LLM
    
    需要设置环境变量 OPENAI_API_KEY
    或在 .env 文件中配置
    """
    # 加载 .env 文件
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError(
            "\n❌ OPENAI_API_KEY not found!\n\n"
            "Please set your OpenAI API key:\n"
            "  1. Create .env file: cp .env.example .env\n"
            "  2. Edit .env and add: OPENAI_API_KEY=sk-your-key-here\n"
            "  3. Or set environment variable: export OPENAI_API_KEY=sk-your-key-here\n\n"
            "Get your API key at: https://platform.openai.com/api-keys"
        )
    
    print(f"\n{'='*70}")
    print(f"Using OpenAI Model: {model}")
    print(f"API Key: {api_key[:10]}...{api_key[-4:]}")
    print(f"{'='*70}\n")
    
    llm = ChatOpenAI(
        model=model,
        temperature=0.3,  # 低温度 = 更事实性的回答
        api_key=api_key
    )
    
    return llm


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
    print("🎓 Chalmers Course Assistant - OpenAI Mode")
    print("=" * 70)
    print("\nAsk me anything about Chalmers courses!")
    print("\nExamples:")
    print("  - What machine learning courses are available?")
    print("  - Can I take TDA357 and DAT450 together?")
    print("  - Tell me about courses in block C")
    print("  - What are the prerequisites for database courses?")
    print("  - Which courses are open for exchange students?")
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
            
            # 生成答案
            print("\n🤖 Assistant: ", end="", flush=True)
            answer = rag_chain.invoke(question)
            print(answer)
            
            query_count += 1
            
        except KeyboardInterrupt:
            print(f"\n\n👋 Interrupted. Answered {query_count} questions.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'quit' to exit.")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("Chalmers Course RAG System - OpenAI Version")
    print("="*70)
    
    try:
        # Step 1: 加载向量数据库
        print("\n" + "="*70)
        print("Step 1: Loading Vector Database")
        print("="*70)
        vectorstore = load_vector_store()
        
        # Step 2: 初始化 OpenAI LLM
        print("\n" + "="*70)
        print("Step 2: Initializing OpenAI LLM")
        print("="*70)
        llm = load_openai_llm()
        print("✓ LLM ready")
        
        # Step 3: 构建 RAG 链
        print("\n" + "="*70)
        print("Step 3: Building RAG Chain")
        print("="*70)
        rag_chain = create_rag_chain(vectorstore, llm)
        print("✓ RAG chain ready")
        
        # Step 4: 进入交互模式
        interactive_query_loop(rag_chain)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
