# Chalmers Course RAG System - Code Submission

**Course:** DAT450/DIT247 - Independent Project  
**Project:** Retrieval-Augmented Generation System for University Course Information  
**Date:** January 18, 2026

---

## 📦 Submission Contents

This code submission includes all source files necessary to reproduce the RAG system described in the project report.

### File Structure

```
├── src/                              # Core system implementation
│   ├── build_vector_db.py            # Vector database construction
│   ├── rag_query_system_gemini.py    # Gemini API version (main)
│   ├── rag_query_system_openai.py    # OpenAI API version
│   ├── rag_query_system_local.py     # Local Mistral-7B version
│   ├── syllabus_scraper.py           # Course data scraper
│   └── deduplicate_courses.py        # Data preprocessing
│
├── tests/                            # Evaluation and testing
│   ├── run_full_test.py              # Complete test suite
│   ├── test_rag_batch_gemini.py      # Gemini batch testing
│   ├── test_rag_batch.py             # General batch testing
│   └── test_rag_setup.py             # System verification
│
├── requirements/                     # Dependencies
│   ├── requirements_gemini.txt       # For Gemini version
│   ├── requirements_openai.txt       # For OpenAI version
│   └── requirements_local.txt        # For local model
│
├── scripts/                          # SLURM job scripts
│   ├── run_build_db.sh               # Build vector database
│   ├── run_rag_gemini.sh             # Run Gemini RAG
│   ├── run_rag_openai.sh             # Run OpenAI RAG
│   └── run_rag_local.sh              # Run local model
│
├── test_results/                     # Evaluation results
│   ├── TEST_REPORT_GEMINI_IMPROVED.md
│   ├── test_results_gemini_improved_20260110.json
│   └── TEST_RESULTS_README.md
│
├── README.md                         # Main documentation
└── CODE_SUBMISSION_README.md         # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google Gemini API key (free tier available)
- 16GB+ RAM for vector database
- GPU recommended for local model (optional)

### Installation

1. **Install dependencies:**
```bash
pip install -r requirements/requirements_gemini.txt
```

2. **Set API key:**
```bash
export GOOGLE_API_KEY="your_api_key_here"
```

3. **Download pre-built database** (if provided) or build from scratch:
```bash
cd src
python build_vector_db.py
```

### Running the System

**Interactive mode:**
```bash
cd src
python rag_query_system_gemini.py
```

**Batch testing:**
```bash
cd tests
python test_rag_batch_gemini.py
```

---

## 📊 Key Files for Evaluation

### Main System Files

1. **`src/rag_query_system_gemini.py`** (Lines 1-300)
   - RAG system implementation
   - Smart retrieval with course code extraction
   - Gemini API integration
   - This is the **primary system** evaluated in the report

2. **`src/build_vector_db.py`** (Lines 1-200)
   - Vector database construction
   - Document chunking and embedding
   - ChromaDB integration

3. **`tests/test_rag_batch_gemini.py`** (Lines 1-150)
   - 10 test queries used in evaluation
   - Automated testing framework
   - Results logging

### Data Files

**Note:** Due to size constraints, the following are **NOT included** in this submission but can be provided upon request:

- `data/chalmers_courses_full_scraped.json` (114 MB) - Raw course data
- `chalmers_chroma_db/` (14 GB) - Pre-built vector database

**These files can be:**
1. Downloaded from [provide link if available]
2. Regenerated using the provided scripts
3. Requested from the authors

---

## 🔬 Reproducing Results

### Step 1: Build Vector Database

```bash
cd src
python build_vector_db.py
```

**Expected output:**
- Vector database created in `chalmers_chroma_db/`
- ~8,500 document chunks
- Processing time: ~30 minutes

### Step 2: Run Evaluation

```bash
cd tests
python test_rag_batch_gemini.py
```

**Expected output:**
- 10 test queries processed
- Results saved to `test_results/`
- Success rate: ~60%

### Step 3: Interactive Testing

```bash
cd src
python rag_query_system_gemini.py
```

Try example queries:
- "What machine learning courses are available?"
- "Tell me about database courses at Chalmers"
- "What are the prerequisites for DAT450?"

---

## 🛠️ System Configuration

### Gemini Version (Recommended)

**Hyperparameters:**
- Retrieval K: 10
- Chunk size: 1000 characters
- Chunk overlap: 200 characters
- Embedding model: `all-MiniLM-L6-v2`
- LLM: `gemini-2.5-flash`
- Temperature: 0.7

**Configuration location:** `src/rag_query_system_gemini.py`, lines 20-30

### OpenAI Version (Alternative)

Requires OpenAI API key and billing setup:
```bash
export OPENAI_API_KEY="your_key"
cd src
python rag_query_system_openai.py
```

### Local Model Version (GPU Required)

Requires ~14GB GPU memory:
```bash
cd scripts
sbatch run_rag_local.sh  # For SLURM
# or
cd src
python rag_query_system_local.py  # Direct execution
```

---

## 📈 Expected Results

Based on our evaluation (detailed in the report):

| Query Type | Success Rate | Examples |
|------------|--------------|----------|
| Topic search | 80-100% | ML courses, database courses |
| Prerequisites | 70-80% | Course requirements |
| Schedule conflicts | 0-20% | Time conflict checking |
| Filtered search | 20-40% | Credit + block filters |

**Overall accuracy: 60% (6/10 test queries)**

---

## 🐛 Troubleshooting

### Common Issues

1. **"API key not found"**
   - Solution: Set `export GOOGLE_API_KEY="your_key"`

2. **"ChromaDB not found"**
   - Solution: Run `python src/build_vector_db.py` first
   - Or download pre-built database

3. **"Out of memory"**
   - Solution: Reduce chunk size or use smaller K value
   - Requires 16GB+ RAM for full database

4. **"Model hallucinating"**
   - This is expected behavior for complex queries
   - See Limitations section in report

---

## 📝 Testing the System

### Minimal Test (5 minutes)

```bash
# 1. Install dependencies
pip install chromadb langchain sentence-transformers google-generativeai

# 2. Set API key
export GOOGLE_API_KEY="your_key"

# 3. Run single query test
cd src
python -c "
from rag_query_system_gemini import *
# Test if system loads
print('System loaded successfully!')
"
```

### Full Evaluation (30 minutes)

```bash
# 1. Build database
cd src
python build_vector_db.py

# 2. Run all tests
cd tests
python test_rag_batch_gemini.py

# 3. Check results
cat test_results/test_results_gemini_improved_*.json
```

---

## 📚 Additional Resources

- **Main documentation:** [README.md](README.md)
- **Test results:** [test_results/TEST_REPORT_GEMINI_IMPROVED.md](test_results/TEST_REPORT_GEMINI_IMPROVED.md)
- **Project overview:** [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)
- **Model comparison:** [docs/MODEL_COMPARISON.md](docs/MODEL_COMPARISON.md)

---

## 👥 Contact

For questions about the code or reproduction issues, please contact:

- [Your Name] - [your.email@student.chalmers.se]
- [Team Member 2] - [member2@student.chalmers.se]

---

## 📄 License & Academic Integrity

This code is submitted as part of the DAT450/DIT247 course project at Chalmers University of Technology. It is provided for evaluation purposes only.

**External libraries used:**
- LangChain (MIT License)
- ChromaDB (Apache 2.0)
- sentence-transformers (Apache 2.0)
- Google Generative AI Python SDK (Apache 2.0)
- BeautifulSoup4 (MIT License)

All external libraries are properly attributed in the report bibliography.
