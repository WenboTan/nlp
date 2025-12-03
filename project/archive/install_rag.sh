#!/bin/bash

echo "========================================"
echo "Installing Chalmers Course RAG System"
echo "========================================"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment (optional but recommended)
read -p "Create virtual environment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Creating virtual environment..."
    python3 -m venv rag_env
    source rag_env/bin/activate
    echo "✓ Virtual environment activated"
fi

echo ""
echo "Installing dependencies..."
echo "This may take several minutes..."
echo ""

# Install dependencies
pip install --upgrade pip

echo "Installing LangChain packages..."
pip install langchain langchain-community langchain-chroma langchain-openai

echo "Installing vector store and embeddings..."
pip install chromadb sentence-transformers

echo "Installing utilities..."
pip install openai python-dotenv

echo ""
echo "========================================"
echo "✓ Installation complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Copy .env.example to .env and add your OpenAI API key"
echo "2. Run: python build_vector_db.py"
echo "3. Run: python rag_query_system.py"
echo ""
echo "See RAG_README.md for detailed instructions."
