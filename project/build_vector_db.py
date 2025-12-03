"""
Step 1: Build Vector Database for Chalmers Course RAG System

This script loads the scraped course data, processes it into LangChain Documents,
splits the text, and creates a persistent Chroma vector database.

Requirements:
    pip install langchain langchain-community langchain-chroma chromadb sentence-transformers
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def load_course_data(json_path: str) -> List[Dict[str, Any]]:
    """Load course data from JSON file."""
    print(f"Loading course data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} courses")
    return data


def create_document_from_course(course: Dict[str, Any]) -> Document:
    """
    Convert a course JSON object into a LangChain Document.
    
    Page Content: Concatenates key course information with clear labels
    Metadata: Stores structured data for filtering and retrieval
    """
    # Extract fields safely with defaults
    course_code = course.get('course_code', 'Unknown')
    title = course.get('title', 'No title')
    url = course.get('metadata', {}).get('url', '')
    
    # Logistics
    logistics = course.get('logistics', {})
    block = logistics.get('block', 'Unknown')
    sp = ', '.join(logistics.get('sp', ['Unknown']))
    language = logistics.get('language', 'Unknown')
    credits = logistics.get('credits', 'Unknown')
    
    # RAG text content
    rag_text = course.get('rag_text', {})
    learning_outcomes = rag_text.get('learning_outcomes', '')
    content = rag_text.get('content', '')
    
    # Constraints
    constraints = course.get('constraints', {})
    prerequisites = constraints.get('prerequisites', 'No prerequisites')
    eligibility = constraints.get('eligibility', {})
    eligibility_text = eligibility.get('text', '')
    open_for_exchange = eligibility.get('open_for_exchange', False)
    programs = constraints.get('programs', [])
    programs_text = '\n'.join(programs) if programs else 'No specific program'
    
    # Build page_content with clear labels
    page_content_parts = [
        f"Course Code: {course_code}",
        f"Title: {title}",
        f"Credits: {credits}",
        f"Language: {language}",
        f"Schedule Block: {block}",
        f"Study Period: {sp}",
        f"\nPrerequisites: {prerequisites}",
        f"\nEligibility: {eligibility_text}" if eligibility_text else "",
        f"Open for Exchange Students: {'Yes' if open_for_exchange else 'No'}",
        f"\nPrograms:\n{programs_text}",
    ]
    
    # Add learning outcomes if available
    if learning_outcomes:
        page_content_parts.append(f"\nLearning Outcomes:\n{learning_outcomes}")
    
    # Add content if available
    if content:
        page_content_parts.append(f"\nCourse Content:\n{content}")
    
    page_content = '\n'.join(filter(None, page_content_parts))
    
    # Create metadata for filtering
    metadata = {
        'course_code': course_code,
        'title': title,
        'url': url,
        'block': block,
        'credits': credits,
        'language': language,
        'sp': sp,
        'open_for_exchange': open_for_exchange,
        'prerequisites': prerequisites[:200] if prerequisites else '',  # Truncate for metadata
    }
    
    return Document(page_content=page_content, metadata=metadata)


def build_vector_database(
    json_path: str = 'chalmers_courses_full_scraped.json',
    db_path: str = './chalmers_chroma_db',
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
):
    """
    Main function to build the vector database.
    
    Args:
        json_path: Path to the course JSON file
        db_path: Directory to save the Chroma database
        chunk_size: Maximum size of text chunks
        chunk_overlap: Overlap between consecutive chunks
        embedding_model: HuggingFace model for embeddings
    """
    print("=" * 70)
    print("Building Chalmers Course Vector Database")
    print("=" * 70)
    
    # Step 1: Load data
    courses = load_course_data(json_path)
    
    # Step 2: Convert to LangChain Documents
    print("\nConverting courses to LangChain Documents...")
    documents = []
    for i, course in enumerate(courses):
        try:
            doc = create_document_from_course(course)
            documents.append(doc)
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(courses)} courses")
        except Exception as e:
            print(f"  ⚠ Error processing course {course.get('course_code', 'Unknown')}: {e}")
    
    print(f"✓ Created {len(documents)} documents")
    
    # Step 3: Split documents into chunks
    print(f"\nSplitting documents (chunk_size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    split_docs = text_splitter.split_documents(documents)
    print(f"✓ Split into {len(split_docs)} chunks")
    
    # Step 4: Create embeddings
    print(f"\nInitializing embeddings model: {embedding_model}")
    print("(This may take a while on first run as it downloads the model...)")
    
    # Auto-detect GPU
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': device},  # Auto-detect GPU
        encode_kwargs={
            'normalize_embeddings': True,
            'batch_size': 32 if device == 'cuda' else 8  # Larger batch on GPU
        }
    )
    print("✓ Embeddings model loaded")
    
    # Step 5: Create and persist Chroma vector store
    print(f"\nCreating Chroma vector store at {db_path}...")
    print("(This will take several minutes depending on dataset size...)")
    
    # Remove existing database if it exists
    db_dir = Path(db_path)
    if db_dir.exists():
        print(f"  ⚠ Existing database found at {db_path}, will be overwritten")
        import shutil
        shutil.rmtree(db_path)
    
    # Create vector store with persistence
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=db_path,
        collection_name='chalmers_courses'
    )
    
    print(f"✓ Vector database created and persisted to {db_path}")
    
    # Step 6: Test the database
    print("\n" + "=" * 70)
    print("Testing the database with a sample query...")
    print("=" * 70)
    
    test_query = "machine learning courses"
    results = vectorstore.similarity_search(test_query, k=3)
    
    print(f"\nQuery: '{test_query}'")
    print(f"Top {len(results)} results:\n")
    
    for i, doc in enumerate(results, 1):
        print(f"{i}. Course: {doc.metadata.get('course_code')} - {doc.metadata.get('title')}")
        print(f"   Block: {doc.metadata.get('block')}, Credits: {doc.metadata.get('credits')}")
        print(f"   Preview: {doc.page_content[:200]}...")
        print()
    
    print("=" * 70)
    print("✅ Vector database build complete!")
    print("=" * 70)
    print(f"\nDatabase location: {db_path}")
    print(f"Total documents: {len(documents)}")
    print(f"Total chunks: {len(split_docs)}")
    print(f"Embedding model: {embedding_model}")
    print("\nYou can now use this database in your RAG query system.")


if __name__ == '__main__':
    # Configuration
    JSON_FILE = 'chalmers_courses_full_scraped.json'
    DB_PATH = './chalmers_chroma_db'
    
    # Check if JSON file exists
    if not Path(JSON_FILE).exists():
        print(f"❌ Error: Course data file not found: {JSON_FILE}")
        print("Please make sure you have run the scraper first.")
        exit(1)
    
    # Build the database
    try:
        build_vector_database(
            json_path=JSON_FILE,
            db_path=DB_PATH,
            chunk_size=1000,
            chunk_overlap=200
        )
    except Exception as e:
        print(f"\n❌ Error building database: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
