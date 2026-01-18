"""
RAG Query System using Google Gemini API
Requires Google API Key

Pros:
- High quality responses
- Strong reasoning ability
- Fast startup (no model loading)
- Higher free tier (Gemini Flash is free)

Cons:
- Requires internet connection
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# Configuration
DB_PATH = './chalmers_chroma_db'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

# Gemini model selection
GEMINI_MODEL = "gemini-2.5-flash"  # Recommended: fast with free tier
# GEMINI_MODEL = "gemini-2.5-pro"    # Higher quality but slower
# GEMINI_MODEL = "gemini-2.0-flash-exp"  # Experimental version

RETRIEVAL_K = 10  # Increased for better coverage


def load_vector_store(db_path: str = DB_PATH) -> Chroma:
    """Load vector database"""
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


def load_gemini_llm(model: str = GEMINI_MODEL) -> ChatGoogleGenerativeAI:
    """
    加载 Google Gemini LLM
    
    Set environment variable GOOGLE_API_KEY
    or in .env fileconfiguration
    """
    # Load .env file
    load_dotenv()
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError(
            "\n❌ GOOGLE_API_KEY not found!\n\n"
            "Please set your Google API key:\n"
            "  1. Create .env file: cp .env.example .env\n"
            "  2. Edit .env and add: GOOGLE_API_KEY=your-key-here\n"
            "  3. Or set environment variable: export GOOGLE_API_KEY=your-key-here\n\n"
            "Get your API key at: https://makersuite.google.com/app/apikey"
        )
    
    print(f"\n{'='*70}")
    print(f"Using Google Gemini Model: {model}")
    print(f"API Key: {api_key[:10]}...{api_key[-4:]}")
    print(f"{'='*70}\n")
    
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=0.3,  # Lower temperature = more factual
        google_api_key=api_key,
        convert_system_message_to_human=True  # Gemini 需要转换系统消息
    )
    
    return llm


def create_rag_prompt() -> ChatPromptTemplate:
    """
    Create RAG prompt template
    Optimized for course assistant
    """
    system_message = """You are a helpful Chalmers University Course Assistant. 
Your job is to answer questions about courses using the provided context.

IMPORTANT INSTRUCTIONS:

1. **Use Retrieved Context Wisely**: 
   - Base your answers primarily on the context provided below
   - If you find partial information, provide what you can and clearly state what's missing
   - Only say "I don't have enough information" if the context is completely irrelevant
   - If a specific course code is mentioned but not in context, acknowledge this explicitly

2. **Schedule Conflict Detection**: 
   - When asked if courses can be taken together, CHECK THE "Schedule Block" field
   - Same Block = TIME CONFLICT (e.g., both in "Block C")
   - Different Blocks = No conflict
   - If one course is missing, state: "I found info for [Course A] but not [Course B]"
   
3. **Include Key Information**:
   - Always mention the course code when discussing a course
   - Include the course URL if available
   - Mention prerequisites, credits, language when relevant

4. **Be Helpful and Practical**:
   - Use bullet points for multiple courses
   - If information is incomplete, provide what you have
   - Suggest checking the official website for missing details
   - Be direct and actionable

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
    import re
    
    def smart_retriever(question: str):
        """智能Retriever：提取课程代码并进行针对性检索"""
        # 提取可能的课程代码 (如 TDA357, DAT450)
        course_codes = re.findall(r'\b[A-Z]{3}\d{3}\b', question.upper())
        
        all_docs = []
        
        # 如果提到了特定课程代码，使用metadata filter精确检索
        if course_codes:
            for code in course_codes:
                try:
                    # 使用metadata filter精确查找课程
                    code_docs = vectorstore.similarity_search(
                        question,  # 使用原始问题以保持相关性
                        k=5,  # 每个课程获取5个chunks
                        filter={'course_code': code}
                    )
                    all_docs.extend(code_docs)
                    print(f"✓ 找到课程 {code}: {len(code_docs)} chunks")
                except Exception as e:
                    print(f"⚠️  课程 {code} filter失败: {e}")
        
        # 如果没找到足够文档，或者没有课程代码，使用更大的K值进行语义检索
        if len(all_docs) < 5:
            semantic_docs = vectorstore.similarity_search(question, k=k*2)  # K值翻倍
            # 合并并去重
            for doc in semantic_docs:
                if doc not in all_docs:
                    all_docs.append(doc)
        
        # 返回最多k*2个文档
        return all_docs[:k*2]
    
    # Create prompt
    prompt = create_rag_prompt()
    
    # Build chain: Question -> Smart Retrieve -> Format -> Prompt -> LLM -> Parse
    rag_chain = (
        {
            'context': lambda x: format_docs(smart_retriever(x)),
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
    print("🎓 Chalmers Course Assistant - Google Gemini Mode")
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
            # GetUser input
            question = input("\n💬 You: ").strip()
            
            # CheckExit command
            if question.lower() in ['quit', 'exit', 'q', 'bye']:
                print(f"\n👋 Goodbye! Answered {query_count} questions.")
                break
            
            # SkipEmpty input
            if not question:
                continue
            
            # GenerateAnswer
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
    """Main function"""
    print("\n" + "="*70)
    print("Chalmers Course RAG System - Google Gemini Version")
    print("="*70)
    
    try:
        # Step 1: Load vector database
        print("\n" + "="*70)
        print("Step 1: Loading Vector Database")
        print("="*70)
        vectorstore = load_vector_store()
        
        # Step 2: Initialize Gemini LLM
        print("\n" + "="*70)
        print("Step 2: Initializing Google Gemini LLM")
        print("="*70)
        llm = load_gemini_llm()
        print("✓ LLM ready")
        
        # Step 3: Build RAG chain
        print("\n" + "="*70)
        print("Step 3: Building RAG Chain")
        print("="*70)
        rag_chain = create_rag_chain(vectorstore, llm)
        print("✓ RAG chain ready")
        
        # Step 4: Enter interactive mode
        interactive_query_loop(rag_chain)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
