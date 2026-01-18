# Chalmers Course RAG System

A Retrieval-Augmented Generation (RAG) system for querying Chalmers University course information in natural language.

## 🚀 Quick Start

- **New user?** → [docs/START_HERE.md](docs/START_HERE.md)
- **Use Gemini API?** → [docs/GEMINI_QUICKSTART.txt](docs/GEMINI_QUICKSTART.txt)
- **View test results?** → [test_results/TEST_RESULTS_README.md](test_results/TEST_RESULTS_README.md)

## 📁 Project Structure

```
nlp/project/
├── 📦 SUBMISSION FILES (DAT450/DIT247 Course Project)
│   ├── submission/                  # Final submission package
│   │   ├── code_submission.zip      # Code package (51KB) ⭐
│   │   ├── code_submission.tar.gz   # Alternative format
│   │   ├── report.tex               # ACL format report
│   │   └── *.md                     # Submission documentation
│   ├── code_submission/             # Clean code for submission
│   │   ├── src/                     # 6 Python source files
│   │   ├── tests/                   # 4 test scripts
│   │   ├── requirements/            # Dependencies
│   │   └── README.md                # How to run
│   └── presentation/                # Presentation materials
│       ├── PRESENTATION_SCRIPT.md   # Presentation slides
│       ├── test_case_*.png          # Demo screenshots (5 images)
│       └── presentation_demo.py     # Live demo script
│
├── 🔬 CORE SYSTEM
│   ├── src/                         # Source code
│   │   ├── build_vector_db.py       # Vector database builder
│   │   ├── rag_query_system_gemini.py   # Gemini API (recommended)
│   │   ├── rag_query_system_openai.py   # OpenAI API
│   │   ├── rag_query_system_local.py    # Local Mistral model
│   │   ├── syllabus_scraper.py      # Course data scraper
│   │   └── deduplicate_courses.py   # Data preprocessing
│   ├── scripts/                     # SLURM job scripts
│   ├── tests/                       # Test suite
│   └── requirements/                # Python dependencies
│
├── 📊 DATA & RESULTS
│   ├── data/                        # Course data (1,122 courses)
│   ├── chalmers_chroma_db/          # Vector database (14GB)
│   ├── test_results/                # Evaluation results
│   └── logs/                        # SLURM logs
│
├── 📚 DOCUMENTATION
│   ├── docs/                        # Technical documentation
│   ├── demo_tests/                  # Demo test files
│   ├── archived_docs/               # Old documentation
│   └── archive/                     # Historical files
│
└── README.md                        # This file
```

## 🎯 Features

- **Natural Language Queries**: Ask questions like "What ML courses are available?"
- **Multi-Model Support**: Choose between Gemini API, OpenAI, or local Mistral
- **Rich Course Information**: Credits, prerequisites, schedules, URLs
- **Vector Search**: Semantic search with ChromaDB + sentence-transformers
- **Batch Testing**: Automated test suite with 10 standard queries

## 📊 Performance

**Current Status (Gemini API with Improved Retrieval):**
- ✅ Success Rate: **60%** (6/10 queries)
- ✅ Retrieval K: 10 (increased from 5)
- ✅ Smart Retriever: Course code extraction + targeted search

**Model Comparison:**
- **Gemini gemini-2.5-flash**: Best quality, 15 req/min free tier
- **OpenAI GPT-4o-mini**: Highest quality, $0.01-0.04/query
- **Local Mistral-7B**: Free but slower, hallucination issues

See [docs/MODEL_COMPARISON.md](docs/MODEL_COMPARISON.md) for details.

## 🚀 Usage

### Option 1: Gemini API (Recommended)

```bash
# 1. Set up environment
export GOOGLE_API_KEY="your_api_key_here"

# 2. Run interactive mode
cd src
python3 rag_query_system_gemini.py

# 3. Or submit SLURM job
cd scripts
sbatch run_rag_gemini.sh
```

### Option 2: Local Model

```bash
# 1. Ensure model is cached (~14GB)
# 2. Submit SLURM job (requires GPU)
cd scripts
sbatch run_rag_local.sh
```

### Option 3: OpenAI API

```bash
# 1. Set API key
export OPENAI_API_KEY="your_api_key_here"

# 2. Run
cd src
python3 rag_query_system_openai.py
```

## 🧪 Testing

Run comprehensive test suite:

```bash
cd tests
python3 run_full_test.py
```

This will:
- Run 10 standard test queries
- Generate detailed report (Markdown)
- Save results (JSON)
- Create timestamped logs

View latest results: [test_results/TEST_RESULTS_README.md](test_results/TEST_RESULTS_README.md)

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [PROJECT_INTRODUCTION.txt](docs/PROJECT_INTRODUCTION.txt) | Complete project overview (598 lines) |
| [START_HERE.md](docs/START_HERE.md) | Quick start guide |
| [GEMINI_GUIDE.md](docs/GEMINI_GUIDE.md) | Gemini API setup and usage |
| [GEMINI_QUICKSTART.txt](docs/GEMINI_QUICKSTART.txt) | Quick Gemini reference |
| [LOCAL_MODEL_GUIDE.md](docs/LOCAL_MODEL_GUIDE.md) | Local Mistral model guide |
| [MODEL_COMPARISON.md](docs/MODEL_COMPARISON.md) | Model performance comparison |
| [RAG_README.md](docs/RAG_README.md) | RAG system architecture |

## 🔧 Development

### Build Vector Database

```bash
cd scripts
sbatch run_build_db.sh
```

This creates `chalmers_chroma_db/` (14GB) with ~8,500 document chunks.

### Scrape New Course Data

```bash
cd src
python3 syllabus_scraper.py
```

### Run Tests

```bash
# Quick test
cd tests
python3 test_rag_setup.py

# Full batch test (Gemini)
python3 test_rag_batch_gemini.py

# Comprehensive test with report
python3 run_full_test.py
```

## 📈 Recent Improvements

**v1.0 (Jan 10, 2026) - Improved Retrieval:**
- ✅ Increased retrieval K from 5 to 10 (+100% documents)
- ✅ Relaxed prompt: allows partial information
- ✅ Smart retriever: extracts course codes (TDA357, DAT450) with regex
- ✅ Success rate: 60% (up from ~40-50%)
- ✅ TDA357 now successfully retrieved in general queries

**Known Issues:**
- ❌ Single course queries still fail (e.g., "Tell me about TDA357")
- ❌ Multi-course comparisons need improvement
- ❌ Complex filtering not fully supported

See [test_results/TEST_REPORT_GEMINI_IMPROVED.md](test_results/TEST_REPORT_GEMINI_IMPROVED.md) for detailed analysis.

## 🗺️ Roadmap

**High Priority:**
1. Fix metadata filtering for single course queries
2. Implement dynamic K for multi-course retrieval
3. Add Block/LP mapping documentation

**Medium Priority:**
4. Hybrid search (BM25 + semantic)
5. Query expansion for better course code matching
6. Support complex metadata filtering

## 💡 Example Queries

```
✅ What machine learning courses are available?
✅ Tell me about database courses at Chalmers.
✅ What are the prerequisites for advanced programming courses?
✅ Which courses are taught in English?
✅ Are there courses involving real projects?

❌ Can I take TDA357 and DAT450 together? (multi-course - WIP)
❌ Tell me everything about TDA357. (single course - WIP)
❌ Show me 7.5 credit courses in Block C. (complex filter - WIP)
```

## 🤝 Contributing

This is a research project. For questions or suggestions, refer to the documentation in `docs/`.

## 📄 License

Educational/Research use only. Course data sourced from Chalmers University of Technology.

---

**Last Updated:** January 10, 2026  
**Version:** 1.0 (Improved Retrieval)  
**Status:** Functional (60% success rate, ready for basic use)
