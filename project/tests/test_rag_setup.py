"""
Quick test script to verify RAG system setup

This script runs a few test queries to ensure everything is working correctly.
"""

import sys
from pathlib import Path

def check_files():
    """Check if required files exist."""
    print("Checking required files...")
    
    required = {
        'chalmers_courses_full_scraped.json': 'Course data file',
        'build_vector_db.py': 'Vector DB builder script',
        'rag_query_system.py': 'RAG query system script',
    }
    
    all_exist = True
    for filename, description in required.items():
        if Path(filename).exists():
            print(f"  ✓ {filename} ({description})")
        else:
            print(f"  ✗ {filename} ({description}) - MISSING")
            all_exist = False
    
    return all_exist


def check_dependencies():
    """Check if required packages are installed."""
    print("\nChecking dependencies...")
    
    packages = {
        'langchain': 'LangChain framework',
        'langchain_community': 'LangChain community integrations',
        'chromadb': 'Chroma vector database',
        'sentence_transformers': 'Sentence Transformers for embeddings',
    }
    
    all_installed = True
    for package, description in packages.items():
        try:
            __import__(package)
            print(f"  ✓ {package} ({description})")
        except ImportError:
            print(f"  ✗ {package} ({description}) - NOT INSTALLED")
            all_installed = False
    
    return all_installed


def check_vector_db():
    """Check if vector database exists."""
    print("\nChecking vector database...")
    
    db_path = Path('./chalmers_chroma_db')
    if db_path.exists() and db_path.is_dir():
        # Check if it contains data
        files = list(db_path.glob('**/*'))
        if files:
            print(f"  ✓ Vector database exists ({len(files)} files)")
            return True
        else:
            print(f"  ⚠ Vector database directory exists but is empty")
            return False
    else:
        print(f"  ✗ Vector database not found at {db_path}")
        print(f"    Run: python build_vector_db.py")
        return False


def test_vector_db():
    """Test loading and querying the vector database."""
    print("\nTesting vector database...")
    
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
        
        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'}
        )
        
        vectorstore = Chroma(
            persist_directory='./chalmers_chroma_db',
            embedding_function=embeddings,
            collection_name='chalmers_courses'
        )
        
        # Test query
        results = vectorstore.similarity_search("machine learning", k=3)
        
        if results:
            print(f"  ✓ Successfully retrieved {len(results)} results")
            print(f"  Sample: {results[0].metadata.get('course_code')} - {results[0].metadata.get('title')[:50]}...")
            return True
        else:
            print(f"  ⚠ No results returned (database may be empty)")
            return False
            
    except Exception as e:
        print(f"  ✗ Error testing database: {e}")
        return False


def check_env_file():
    """Check if .env file exists and has API key."""
    print("\nChecking environment configuration...")
    
    env_path = Path('.env')
    if not env_path.exists():
        print(f"  ⚠ .env file not found")
        print(f"    Copy .env.example to .env and add your OpenAI API key")
        return False
    
    # Check if it contains an API key
    content = env_path.read_text()
    if 'OPENAI_API_KEY=sk-' in content or 'OPENAI_API_KEY="sk-' in content:
        print(f"  ✓ .env file exists with API key configured")
        return True
    elif 'your_api_key_here' in content:
        print(f"  ⚠ .env file exists but API key not set")
        print(f"    Replace 'your_api_key_here' with your actual OpenAI API key")
        return False
    else:
        print(f"  ⚠ .env file exists but API key format unclear")
        return False


def main():
    """Run all checks."""
    print("=" * 60)
    print("RAG System Setup Verification")
    print("=" * 60)
    print()
    
    checks = [
        ("Files", check_files),
        ("Dependencies", check_dependencies),
        ("Vector Database", check_vector_db),
        ("Environment", check_env_file),
    ]
    
    results = {}
    for name, check_func in checks:
        results[name] = check_func()
        print()
    
    # Try to test the database if dependencies are installed
    if results["Dependencies"] and results["Vector Database"]:
        results["Database Query"] = test_vector_db()
        print()
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print()
    
    if all_passed:
        print("🎉 All checks passed! Your RAG system is ready.")
        print("\nTo start the query system:")
        print("  python rag_query_system.py")
    else:
        print("⚠ Some checks failed. Please address the issues above.")
        print("\nQuick fix guide:")
        print("  - Missing files: Make sure you're in the correct directory")
        print("  - Missing dependencies: Run ./install_rag.sh or pip install -r requirements.txt")
        print("  - No vector database: Run python build_vector_db.py")
        print("  - No API key: Copy .env.example to .env and add your OpenAI key")
    
    print("\nFor detailed instructions, see RAG_README.md")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
