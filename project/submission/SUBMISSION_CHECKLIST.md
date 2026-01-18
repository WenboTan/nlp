# DAT450/DIT247 Project Submission Checklist

**Student:** [Your Name]  
**Date:** January 18, 2026  
**Project:** Chalmers Course RAG System

---

## ✅ Submission Requirements

### 1. Report Submission (PDF/Word - SEPARATE UPLOAD)

- [ ] **Format:** Using ACL 2023 template ✓
- [ ] **File format:** PDF or Word document
- [ ] **Main body:** Maximum 4 pages ✓
- [ ] **Structure includes:**
  - [ ] Abstract ✓
  - [ ] Introduction ✓
  - [ ] Related Work ✓
  - [ ] Methodology ✓
  - [ ] Results ✓
  - [ ] Discussion ✓
  - [ ] Conclusion ✓
- [ ] **Limitations section** (not counted in 4 pages) ✓
- [ ] **Bibliography** (not counted) ✓
- [ ] **Appendix** (optional, not counted) ✓
- [ ] **Submission method:** Direct upload to Canvas (NOT in zip file)
- [ ] **Deadline:** January 18, 2026 ⚠️ TODAY

**File to submit:** `report.pdf`

---

### 2. Code Submission (SEPARATE UPLOAD)

- [ ] **Core source files included:**
  - [ ] `src/build_vector_db.py` ✓
  - [ ] `src/rag_query_system_gemini.py` ✓
  - [ ] `src/rag_query_system_openai.py` ✓
  - [ ] `src/rag_query_system_local.py` ✓
  - [ ] `src/syllabus_scraper.py` ✓
  - [ ] `src/deduplicate_courses.py` ✓

- [ ] **Test files included:**
  - [ ] `tests/run_full_test.py` ✓
  - [ ] `tests/test_rag_batch_gemini.py` ✓
  - [ ] `tests/test_rag_setup.py` ✓

- [ ] **Documentation included:**
  - [ ] `README.md` ✓
  - [ ] `CODE_SUBMISSION_README.md` ✓
  - [ ] `requirements/requirements_gemini.txt` ✓

- [ ] **Test results included:**
  - [ ] `test_results/TEST_REPORT_GEMINI_IMPROVED.md` ✓
  - [ ] `test_results/test_results_gemini_improved_20260110.json` ✓

- [ ] **Submission method:** Upload to Canvas (source files or zip)
- [ ] **Deadline:** January 18, 2026 ⚠️ TODAY

**Files to submit:** Code package (zip) or individual source files

---

### 3. Oral Presentation (ALREADY DUE)

- [ ] **Duration:** Maximum 10 minutes
- [ ] **Format:** Recorded video
- [ ] **Submission:** Canvas upload
- [ ] **Deadline:** January 13, 2026 ⚠️ OVERDUE

**Status:** [ ] Submitted / [ ] Late submission needed

---

### 4. Peer Review (UPCOMING)

- [ ] **Task:** Watch 2 other group presentations
- [ ] **Task:** Provide written feedback
- [ ] **Submission:** Canvas
- [ ] **Deadline:** January 21, 2026

---

## 📋 Pre-Submission Checks

### Report Quality Checks

- [ ] All author names and emails filled in
- [ ] No placeholder text (e.g., "Your Name Here")
- [ ] All references properly formatted
- [ ] Page count verified (main body ≤ 4 pages)
- [ ] Figures/tables have captions and are referenced in text
- [ ] No spelling or grammar errors
- [ ] Compiled successfully to PDF

### Code Quality Checks

- [ ] All code files include docstrings/comments
- [ ] No hardcoded API keys in source files
- [ ] README explains how to run the code
- [ ] Requirements files are complete
- [ ] Code follows PEP 8 style guidelines (Python)
- [ ] No unnecessary debug print statements

### Technical Verification

- [ ] Can compile LaTeX to PDF without errors
- [ ] Code runs without errors (tested)
- [ ] All dependencies listed in requirements.txt
- [ ] Data availability explained (if not included)
- [ ] Results are reproducible

---

## 📦 How to Package Submission

### Option 1: Zip File (Recommended)

```bash
cd /data/users/wenbota/nlp/project

# Create submission package
zip -r code_submission.zip \
  src/ \
  tests/ \
  requirements/ \
  scripts/ \
  test_results/ \
  README.md \
  CODE_SUBMISSION_README.md \
  -x "*.pyc" "*__pycache__*" "*.log"
```

**Verify zip contents:**
```bash
unzip -l code_submission.zip
```

### Option 2: Individual Files

Upload directly to Canvas:
- All files from `src/`
- All files from `tests/`
- README files
- Requirements files

---

## 🚀 Final Steps Before Submission

### For Report:

1. **Compile LaTeX to PDF:**
```bash
cd /data/users/wenbota/nlp/project
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
```

2. **Verify PDF:**
   - Open `report.pdf`
   - Check all pages render correctly
   - Verify page count: main body ≤ 4 pages
   - Check that references are linked properly

3. **Submit to Canvas:**
   - Go to Canvas assignment page
   - Upload `report.pdf` (or `report.docx`)
   - **DO NOT** include in zip with code
   - Submit before deadline

### For Code:

1. **Create submission package:**
```bash
cd /data/users/wenbota/nlp/project
zip -r code_submission.zip src/ tests/ requirements/ README.md CODE_SUBMISSION_README.md test_results/
```

2. **Test the package:**
```bash
# Extract to temp directory and test
unzip code_submission.zip -d /tmp/test_submission
cd /tmp/test_submission
pip install -r requirements/requirements_gemini.txt
# Try running a simple test
```

3. **Submit to Canvas:**
   - Upload `code_submission.zip`
   - Or upload individual source files
   - Submit before deadline

---

## ⏰ Deadlines Summary

| Item | Deadline | Status |
|------|----------|--------|
| Report (PDF) | January 18, 2026 | ⚠️ TODAY |
| Code | January 18, 2026 | ⚠️ TODAY |
| Presentation | January 13, 2026 | 🔴 OVERDUE |
| Peer Review | January 21, 2026 | ⏳ 3 days left |

---

## 📧 What to Submit to Canvas

### Upload 1: Report (Separate)
- **File:** `report.pdf`
- **Upload location:** Canvas → Report Submission
- **Format:** PDF or Word
- **Size:** Should be < 5 MB

### Upload 2: Code (Separate)
- **File:** `code_submission.zip` or individual files
- **Upload location:** Canvas → Code Submission
- **Format:** .zip, .py, .txt files
- **Size:** Should be < 50 MB (without data/database)

### Upload 3: Presentation (If not submitted yet)
- **File:** `presentation_video.mp4` (or similar)
- **Upload location:** Canvas → Presentation Submission
- **Format:** Video file (MP4, MOV, etc.)
- **Duration:** ≤ 10 minutes

---

## ✨ Final Quality Check

Before clicking "Submit":

1. **Report:**
   - [ ] Opens correctly in PDF reader
   - [ ] All sections present
   - [ ] Page count correct
   - [ ] No compilation errors visible

2. **Code:**
   - [ ] Can extract zip file without errors
   - [ ] README explains how to run
   - [ ] No sensitive information (API keys, passwords)
   - [ ] All necessary files included

3. **General:**
   - [ ] Names/emails correct on all documents
   - [ ] Files named appropriately
   - [ ] Deadline confirmed (today!)

---

## 🆘 Emergency Contacts

If you encounter issues:

1. **Canvas technical issues:** Contact Chalmers IT support
2. **Deadline extension:** Email course instructors immediately
3. **File size too large:** Remove large data files, explain in README

---

**Good luck with your submission! 🎓**
