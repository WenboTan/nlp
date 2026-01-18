"""
Batch test script - Google Geminiversion
For testing in SLURM jobs Gemini RAG system
No interactive input required, runs predefined test queries
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# Configuration
DB_PATH = './chalmers_chroma_db'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
GEMINI_MODEL = "gemini-2.5-flash"  # 使用快速version进行测试
RETRIEVAL_K = 10  # 增加检索数量以提高覆盖率

# 测试查询列表（覆盖多种查询场景）
TEST_QUERIES = [
    # 1. 课程搜索 - 按主题
    "What machine learning courses are available?",
    
    # 2. 领域查询
    "Tell me about database courses at Chalmers.",
    
    # 3. 时间冲突检测（核心功能）
    "Can I take TDA357 and DAT450 together? Do they have schedule conflicts?",
    
    # 4. 先修课程查询
    "What are the prerequisites for advanced programming courses?",
    
    # 5. 按Block和学分筛选
    "Show me all 7.5 credit courses offered in Block C.",
    
    # 6. 语言要求查询
    "Which courses are taught in English and suitable for international students?",
    
    # 7. 具体课程详情
    "Tell me everything about the course TDA357.",
    
    # 8. 课程比较
    "What's the difference between DAT450 and TDA362?",
    
    # 9. 项目型课程
    "Are there any courses that involve working on real projects?",
    
    # 10. 推荐课程（综合查询）
    "I'm interested in AI and want to take courses in spring (Block 3 and 4). What do you recommend?"
]


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
    """加载 Google Gemini LLM"""
    load_dotenv()
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError(
            "\n❌ GOOGLE_API_KEY not found!\n"
            "Please set it in .env file or as environment variable.\n"
            "Get your API key at: https://makersuite.google.com/app/apikey"
        )
    
    print(f"\n{'='*70}")
    print(f"Loading Google Gemini: {model}")
    print(f"API Key: {api_key[:10]}...{api_key[-4:]}")
    print(f"{'='*70}\n")
    
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=0.3,
        google_api_key=api_key,
        convert_system_message_to_human=True
    )
    
    return llm


def create_rag_prompt() -> ChatPromptTemplate:
    """Create RAG prompt template"""
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
    """Create RAG chain (using LCEL)- 使用smart retriever"""
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
                except Exception as e:
                    pass  # 静默失败，在测试中
        
        # 如果没找到足够文档，或者没有课程代码，使用更大的K值进行语义检索
        if len(all_docs) < 5:
            semantic_docs = vectorstore.similarity_search(question, k=k*2)  # K值翻倍
            # 合并并去重
            for doc in semantic_docs:
                if doc not in all_docs:
                    all_docs.append(doc)
        
        # 返回最多k*2个文档
        return all_docs[:k*2]
    
    prompt = create_rag_prompt()
    
    rag_chain = (
        {
            'context': RunnableLambda(smart_retriever) | format_docs,
            'question': RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


def run_batch_test(rag_chain, queries: list):
    """运行批量测试"""
    print("\n" + "=" * 70)
    print(f"Running batch test with {len(queries)} queries")
    print("=" * 70 + "\n")
    
    results = []
    
    for i, query in enumerate(queries, 1):
        print("=" * 70)
        print(f"Test Query {i}/{len(queries)}")
        print("=" * 70)
        print(f"Question: {query}\n")
        
        try:
            print("Generating answer...")
            answer = rag_chain.invoke(query)
            
            print("\n🤖 Answer:\n")
            print(answer)
            print("\n")
            
            results.append({
                'query': query,
                'answer': answer,
                'status': 'success'
            })
            
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            results.append({
                'query': query,
                'error': str(e),
                'status': 'failed'
            })
    
    # 打印总结
    print("\n" + "=" * 70)
    print("✅ All tests completed!")
    print("=" * 70)
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"\nResults: {success_count}/{len(queries)} successful")
    
    return results


def main():
    """Main function"""
    print("\n" + "="*70)
    print("Chalmers Course RAG System - Batch Test Mode")
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
        
        # Step 4: 运行批量测试
        results = run_batch_test(rag_chain, TEST_QUERIES)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
