# Project Directory Structure - Clean & Organized

## 📂 Directory Overview

```
/data/users/wenbota/nlp/project/
│
├── 📦 SUBMISSION (Course Project Files)
│   ├── submission/                  # Final submission package ⭐
│   │   ├── code_submission.zip      # Ready to upload (51KB)
│   │   ├── code_submission.tar.gz   # Alternative format (34KB)
│   │   ├── report.tex               # ACL format LaTeX report
│   │   ├── CODE_READY_TO_SUBMIT.md
│   │   ├── CODE_SUBMISSION_CHECKLIST.md
│   │   ├── CODE_SUBMISSION_README.md
│   │   ├── SUBMISSION_CHECKLIST.md
│   │   ├── SUBMISSION_VERIFICATION.md
│   │   └── prepare_code_submission.sh
│   │
│   ├── code_submission/             # Clean submission code
│   │   ├── src/                     # 6 Python files
│   │   ├── tests/                   # 4 test files
│   │   ├── requirements/            # 3 dependency files
│   │   ├── scripts/                 # 5 SLURM scripts
│   │   ├── test_results/            # 2 result files
│   │   └── README.md
│   │
│   └── presentation/                # Presentation materials
│       ├── PRESENTATION_SCRIPT.md
│       ├── PRESENTATION_READINESS.md
│       ├── SPEAKER_NOTES_TEST_CASES.md
│       ├── TEST_CASES_FOR_SLIDES.md
│       ├── presentation_demo.py
│       ├── presentation_demo_output.txt
│       ├── generate_test_case_images.py
│       ├── test_case_1_ai_ml.png
│       ├── test_case_2_schedule.png
│       ├── test_case_3_comparison.png
│       ├── test_case_4_learning_path.png
│       └── test_case_5_limitation.png
│
├── 🔬 CORE SYSTEM (Production Code)
│   ├── src/                         # Main source code
│   │   ├── build_vector_db.py
│   │   ├── rag_query_system_gemini.py
│   │   ├── rag_query_system_openai.py
│   │   ├── rag_query_system_local.py
│   │   ├── syllabus_scraper.py
│   │   └── deduplicate_courses.py
│   │
│   ├── scripts/                     # SLURM job scripts
│   │   ├── run_build_db.sh
│   │   ├── run_rag_gemini.sh
│   │   ├── run_rag_openai.sh
│   │   ├── run_rag_local.sh
│   │   └── run_rag_test.sh
│   │
│   ├── tests/                       # Test suite
│   │   ├── run_full_test.py
│   │   ├── test_rag_batch_gemini.py
│   │   ├── test_rag_batch.py
│   │   └── test_rag_setup.py
│   │
│   └── requirements/                # Dependencies
│       ├── requirements_gemini.txt
│       ├── requirements_openai.txt
│       └── requirements_local.txt
│
├── 📊 DATA & RESULTS
│   ├── data/                        # Course data
│   │   └── chalmers_courses_full_scraped.json (114MB, 1,122 courses)
│   │
│   ├── chalmers_chroma_db/          # Vector database (14GB)
│   │   └── ~8,500 document chunks
│   │
│   ├── test_results/                # Evaluation results
│   │   ├── TEST_RESULTS_README.md
│   │   ├── TEST_REPORT_GEMINI_IMPROVED.md
│   │   └── test_results_gemini_improved_20260110.json
│   │
│   └── logs/                        # SLURM job logs
│       └── *.err files
│
├── 📚 DOCUMENTATION
│   ├── docs/                        # Technical docs
│   │   ├── START_HERE.md
│   │   ├── PROJECT_OVERVIEW.md
│   │   ├── PROJECT_INTRODUCTION.txt
│   │   ├── GEMINI_GUIDE.md
│   │   ├── GEMINI_QUICKSTART.txt
│   │   ├── LOCAL_MODEL_GUIDE.md
│   │   ├── MODEL_COMPARISON.md
│   │   ├── RAG_README.md
│   │   └── README.md
│   │
│   ├── demo_tests/                  # Demo/test files
│   │   ├── test_hardware_courses.py
│   │   ├── test_hardware_output.txt
│   │   └── q3_ai_courses_query.txt
│   │
│   ├── archived_docs/               # Old documentation
│   │   ├── README_OLD.md
│   │   └── REORGANIZATION_LOG.md
│   │
│   └── archive/                     # Historical files
│       ├── courses_code.py
│       ├── courses_code_selenium.py
│       ├── rag_query_system.py
│       ├── install_rag.sh
│       ├── QUICKSTART.md
│       ├── data/
│       └── test/
│
├── ⚙️ CONFIGURATION
│   ├── .env                         # API keys (not in git)
│   ├── .env.example                 # Template
│   ├── .gitignore
│   └── .venv/                       # Python virtual env
│
└── README.md                        # Main readme

```

## 📋 File Count Summary

| Category | Count | Size |
|----------|-------|------|
| **Python source files** | 6 | ~60KB |
| **Test files** | 4 | ~30KB |
| **Scripts** | 5 | ~7KB |
| **Documentation** | 15+ | ~200KB |
| **Presentation files** | 10 | ~320KB |
| **Submission package** | 21 files | 51KB (zip) |
| **Course data** | 1,122 courses | 114MB |
| **Vector database** | ~8,500 chunks | 14GB |

## 🎯 Key Files for Submission

### To Upload to Canvas:
1. **submission/code_submission.zip** (51KB) - Main code package
2. **submission/report.tex** - Project report (compile to PDF)

### Documentation References:
- **submission/CODE_READY_TO_SUBMIT.md** - Submission checklist
- **submission/SUBMISSION_VERIFICATION.md** - File hashes & GitHub links

## 🔗 Quick Links

- **GitHub:** https://github.com/WenboTan/nlp
- **Commit:** 64a4d8d (Streamlined submission)
- **SHA-256:** (see SUBMISSION_VERIFICATION.md)

## 📝 Notes

- All Chinese comments removed from code
- No hardcoded API keys
- Redundant docs removed (from 32 → 21 files)
- Clean, professional submission ready
- GitHub backup available

---

**Last Updated:** January 18, 2026  
**Status:** ✅ Ready for submission
