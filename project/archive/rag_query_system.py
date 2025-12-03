"""
Step 2: RAG Query System for Chalmers Courses

This script creates an interactive RAG-based chatbot that can answer questions
about Chalmers courses using the vector database created in Step 1.

Requirements:
    pip install langchain langchain-community langchain-chroma chromadb 
    pip install sentence-transformers openai python-dotenv
    
    # Create a .env file with:
    OPENAI_API_KEY=your_api_key_here
"""

import os
import sys
from pathlib import Path
from typing import List
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# LLM options
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠ langchain-openai not installed. Run: pip install langchain-openai")


def load_vector_store(
    db_path: str = './chalmers_chroma_db',
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
) -> Chroma:
    """
    Load the existing Chroma vector database.
    
    Args:
        db_path: Path to the persisted Chroma database
        embedding_model: Must match the model used during indexing
    
    Returns:
        Chroma vector store instance
    """
    print(f"Loading vector database from {db_path}...")
    
    if not Path(db_path).exists():
        raise FileNotFoundError(
            f"Vector database not found at {db_path}. "
            "Please run build_vector_db.py first."
        )
    
    # Initialize the same embeddings model used during indexing
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': 'cpu'},  # Change to 'cuda' for GPU
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Load the persisted vector store
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name='chalmers_courses'
    )
    
    print(f"✓ Vector store loaded successfully")
    return vectorstore


def create_llm(use_openai: bool = True, model: str = 'gpt-3.5-turbo'):
    """
    Create the LLM instance.
    
    Args:
        use_openai: If True, use OpenAI's API. If False, use local model.
        model: Model name to use
    
    Returns:
        LLM instance
    """
    if use_openai:
        if not OPENAI_AVAILABLE:
            raise ImportError("langchain-openai not installed. Run: pip install langchain-openai")
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it in .env file or environment variable."
            )
        
        print(f"Using OpenAI model: {model}")
        return ChatOpenAI(
            model=model,
            temperature=0.3,  # Lower temperature for more factual responses
            api_key=api_key
        )
    else:
        # Alternative: Use local HuggingFace model
        # Uncomment and modify as needed:
        """
        from langchain_community.llms import HuggingFacePipeline
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # or another model
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            torch_dtype='auto'
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.3
        )
        
        return HuggingFacePipeline(pipeline=pipe)
        """
        raise NotImplementedError(
            "Local model option not implemented. "
            "Uncomment and configure the HuggingFacePipeline code above."
        )


def create_rag_prompt() -> ChatPromptTemplate:
    """
    Create the prompt template for RAG queries.
    
    The prompt instructs the AI to:
    - Act as a Chalmers University course assistant
    - Use retrieved context to answer questions
    - Check for schedule conflicts based on Block information
    - Include course codes and URLs when available
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
   - Example conflicts: Block "C" conflicts with "C", Block "D" conflicts with "D"
   - Block "C+" or "D+" may span multiple periods - mention this as potential partial conflict.
   - If blocks are different, courses do NOT conflict.
   
3. **Include Key Information**:
   - Always mention the course code when discussing a course
   - Include the course URL if available for users to learn more
   - Mention prerequisites, credits, language, and program availability when relevant

4. **Be Precise and Helpful**:
   - If asked about prerequisites, eligibility, or program availability, extract exact information from context
   - For exchange students, check the "Open for Exchange Students" field
   - Provide concrete details like credit amounts, study periods, and assessment methods

5. **Format**: 
   - Use clear, well-structured responses
   - Use bullet points for listing multiple courses
   - Bold important information like **TIME CONFLICT** or **course codes**

Context from course database:
{context}

Question: {question}

Answer:"""
    
    return ChatPromptTemplate.from_template(system_message)


def format_docs(docs: List) -> str:
    """
    Format retrieved documents into a single context string.
    
    Args:
        docs: List of retrieved Document objects
    
    Returns:
        Formatted string containing all document contents
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        formatted.append(f"--- Document {i} ---")
        formatted.append(f"Course: {doc.metadata.get('course_code', 'Unknown')}")
        formatted.append(f"Block: {doc.metadata.get('block', 'Unknown')}")
        formatted.append(f"URL: {doc.metadata.get('url', 'N/A')}")
        formatted.append(f"\n{doc.page_content}\n")
    
    return "\n".join(formatted)


def create_rag_chain(vectorstore: Chroma, llm, k: int = 5):
    """
    Create the RAG chain using LCEL (LangChain Expression Language).
    
    Args:
        vectorstore: Chroma vector store instance
        llm: Language model instance
        k: Number of documents to retrieve
    
    Returns:
        Runnable RAG chain
    """
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type='similarity',
        search_kwargs={'k': k}
    )
    
    # Create prompt
    prompt = create_rag_prompt()
    
    # Build the chain using LCEL
    # Chain: Retrieve -> Format -> Prompt -> LLM -> Parse
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
    """
    Run an interactive loop for querying the RAG system.
    
    Args:
        rag_chain: The RAG chain to use for queries
    """
    print("\n" + "=" * 70)
    print("🎓 Chalmers Course Assistant - Interactive Mode")
    print("=" * 70)
    print("\nAsk me anything about Chalmers courses!")
    print("Examples:")
    print("  - What machine learning courses are available?")
    print("  - Can I take DAT450 and TDA357 together?")
    print("  - Tell me about courses in block C")
    print("  - What are the prerequisites for database courses?")
    print("  - Which courses are open for exchange students?")
    print("\nType 'quit', 'exit', or 'q' to stop.")
    print("=" * 70 + "\n")
    
    while True:
        try:
            # Get user input
            question = input("\n💬 You: ").strip()
            
            # Check for exit commands
            if question.lower() in ['quit', 'exit', 'q', 'bye']:
                print("\n👋 Goodbye! Thank you for using Chalmers Course Assistant.")
                break
            
            # Skip empty inputs
            if not question:
                continue
            
            # Query the RAG chain
            print("\n🤖 Assistant: ", end='', flush=True)
            
            # Stream the response (if supported by LLM)
            try:
                response = rag_chain.invoke(question)
                print(response)
            except Exception as e:
                print(f"\n⚠ Error processing query: {e}")
                print("Please try again with a different question.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n⚠ Unexpected error: {e}")
            print("Continuing...")


def main():
    """Main function to set up and run the RAG query system."""
    # Load environment variables from .env file
    load_dotenv()
    
    print("=" * 70)
    print("Chalmers Course RAG Query System")
    print("=" * 70)
    
    # Configuration
    DB_PATH = './chalmers_chroma_db'
    EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
    USE_OPENAI = True  # Set to False to use local model (requires configuration)
    LLM_MODEL = 'gpt-3.5-turbo'  # or 'gpt-4' for better results
    RETRIEVAL_K = 5  # Number of documents to retrieve
    
    try:
        # Step 1: Load vector store
        vectorstore = load_vector_store(DB_PATH, EMBEDDING_MODEL)
        
        # Step 2: Initialize LLM
        print(f"\nInitializing language model...")
        llm = create_llm(use_openai=USE_OPENAI, model=LLM_MODEL)
        print("✓ LLM initialized")
        
        # Step 3: Create RAG chain
        print(f"\nBuilding RAG chain (retrieving top-{RETRIEVAL_K} documents per query)...")
        rag_chain = create_rag_chain(vectorstore, llm, k=RETRIEVAL_K)
        print("✓ RAG chain ready")
        
        # Step 4: Run interactive query loop
        interactive_query_loop(rag_chain)
    
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\nPlease run build_vector_db.py first to create the vector database.")
        sys.exit(1)
    
    except ValueError as e:
        print(f"\n❌ {e}")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
