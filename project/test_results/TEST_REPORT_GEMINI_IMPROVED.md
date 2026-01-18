# Chalmers Course RAG System - Test Report
## Gemini API with Improved Retrieval (K=10 + Smart Retriever)

---

**Test Date:** January 10, 2026  
**Model:** Google Gemini gemini-2.5-flash  
**Configuration:**
- Retrieval K: **10** (increased from 5)
- Prompt Strategy: **Relaxed** (provide partial info instead of strict "no info")
- Retrieval Method: **Smart Retriever** with course code regex extraction

---

## Executive Summary

| Metric | Value | Percentage |
|--------|-------|------------|
| **Total Queries** | 10 | 100% |
| **Successful** | 6 | **60%** |
| **Failed** | 4 | 40% |

**Improvement vs. Baseline:**
- Previous success rate: ~40-50%
- Current success rate: **60%**
- **Improvement: +10-20%** 

---

## Detailed Test Results

### ✅ Query 1: Machine Learning Courses
**Question:** "What machine learning courses are available?"

**Status:** ✓ **SUCCESS**

**Answer Summary:**
- Found **6 relevant courses**: EEN175, DAT441, DAT341, TIF360, DAT635, MMS131
- Provided complete information including:
  - Course codes and names
  - Credits (6.0-7.5)
  - Prerequisites
  - Course URLs

**Quality:** Excellent - comprehensive list with all details

---

### ✅ Query 2: Database Courses
**Question:** "Tell me about database courses at Chalmers."

**Status:** ✓ **SUCCESS**

**Answer Summary:**
- Found **3 database courses**: DAT335, TDA357, DAT475
- Key course: **TDA357 Databases** correctly identified
- Included:
  - Credits and schedule blocks
  - Prerequisites (e.g., 7.5 credits programming, logic concepts)
  - Content overview (relational algebra, ER diagrams, SQL, JDBC)
  - URLs for each course

**Quality:** Excellent - this is a major improvement! TDA357 is now correctly retrieved

---

### ❌ Query 3: Schedule Conflict Check
**Question:** "Can I take TDA357 and DAT450 together in the same semester?"

**Status:** ✗ **FAILED**

**Answer:** "I don't have enough information about that in the course database."

**Analysis:**
- Query requires checking schedule blocks for both TDA357 and DAT450
- System needs to retrieve both courses and compare their blocks
- **Root Cause:** Multi-course retrieval insufficient

---

### ✅ Query 4: Prerequisites for Advanced Programming
**Question:** "What are the prerequisites for advanced programming courses?"

**Status:** ✓ **SUCCESS**

**Answer Summary:**
- Found **6 advanced programming courses**: DAT151, TDA342, DAT575, DAT480, EDA093, DAT516
- Detailed prerequisites for each, including:
  - Programming languages (Haskell, Java, C/C++, Python)
  - Mathematical requirements (calculus, linear algebra)
  - Prior course requirements (DAT452, TMV210, etc.)
  - Recommended background

**Quality:** Excellent - comprehensive and well-organized

---

### ❌ Query 5: Block C Courses
**Question:** "Show me all 7.5 credit courses offered in Block C."

**Status:** ✗ **FAILED**

**Answer:** "I don't have enough information about that in the course database. The provided context does not list any 7.5 credit courses specifically scheduled in 'Block C'."

**Analysis:**
- Query combines two filters: credits=7.5 AND block=C
- **Root Cause:** Block metadata may not be well-represented in retrieved chunks

---

### ✅ Query 6: English Courses for International Students
**Question:** "Which courses are taught in English and suitable for international students?"

**Status:** ✓ **SUCCESS**

**Answer Summary:**
- Found **3 courses** explicitly marked for exchange students:
  - TRA345 (Bachelor level, requires English 6/B)
  - TEK665 (Master's level, requires English 6)
  - IMS160 (Master's level, requires English 6)
- All courses: 7.5 credits, taught in English
- Included prerequisites and URLs

**Quality:** Good - found relevant courses, though could potentially find more

---

### ❌ Query 7: Single Course Deep Dive
**Question:** "Tell me everything about the course TDA357."

**Status:** ✗ **FAILED**

**Answer:** "I don't have enough information about that in the course database."

**Analysis:**
- **Critical Issue:** Even though Q2 successfully found TDA357 in a general query, this specific course code query fails
- **Root Cause:** Smart retriever's course code extraction + filtering not working correctly
- Query explicitly mentions "TDA357" but system cannot retrieve it
- **High Priority Fix Needed**

---

### ❌ Query 8: Course Comparison
**Question:** "What's the difference between DAT450 and TDA362?"

**Status:** ✗ **FAILED**

**Answer:** "I don't have enough information about that in the course database."

**Analysis:**
- Requires retrieving information about BOTH courses
- Then comparing content, prerequisites, focus areas
- **Root Cause:** Multi-course retrieval insufficient, similar to Q3

---

### ✅ Query 9: Project-Based Courses
**Question:** "Are there any courses that involve working on real projects?"

**Status:** ✓ **SUCCESS**

**Answer Summary:**
- Found **6 project-based courses**: MVE405, TME131, TRA385, TEK495, TEK830, TEK486
- Highlighted real-world collaboration aspects:
  - Industry partnerships (TEK830, TEK486)
  - Real company challenges (TEK495)
  - Team-based projects (TME131, TRA385)
- Included credits, blocks, and URLs

**Quality:** Excellent - diverse selection with clear project focus

---

### ❌ Query 10: Block Mapping Issue
**Question:** "I'm interested in AI and want to take courses in spring (Block 3 and 4). What do you recommend?"

**Status:** ✗ **FAILED**

**Answer:** "I don't have enough information about the specific mapping of 'Block 3 and 4' to the course schedule blocks (A, B, C, D)..."

**Analysis:**
- System correctly identifies the problem: "Block 3 and 4" vs. "A, B, C, D" terminology mismatch
- **Root Cause:** Missing LP (study period) to Block mapping documentation
- Note: System's LP codes are LP1, LP2, LP3, LP4, but schedule uses A, B, C, D

---

## Problem Categories and Analysis

### Category 1: Single Course Queries (1 failure)
**Affected:** Q7

**Issue:** Specific course code query fails despite course being in database

**Technical Analysis:**
```python
# Current smart_retriever logic (lines 175-192):
if course_codes:
    for code in course_codes:
        code_docs = vectorstore.similarity_search(
            f"Course code: {code}",
            k=3,
            filter={'course_code': code}  # ⚠️ This filter may not work
        )
```

**Problem:** 
- Metadata filtering not functioning correctly
- Query "Course code: TDA357" doesn't match document embeddings well
- Need to verify course_code metadata is properly indexed

**Recommended Fix:**
1. Test metadata filter separately
2. If filtering doesn't work, increase K and post-filter in Python
3. Add explicit BM25 keyword search for course codes

---

### Category 2: Multi-Course Queries (2 failures)
**Affected:** Q3 (schedule conflict), Q8 (comparison)

**Issue:** Need to retrieve complete information about 2+ courses simultaneously

**Technical Analysis:**
- Current K=10 may retrieve only partial info for multiple courses
- Smart retriever extracts codes but may not get enough chunks per course

**Recommended Fix:**
1. When multiple course codes detected, increase K dynamically (K = base_k * num_codes)
2. Ensure at least 3-5 chunks per course code
3. Group chunks by course_code before sending to LLM

---

### Category 3: Complex Filtering (1 failure)
**Affected:** Q5 (Block C + 7.5 credits)

**Issue:** Combining multiple metadata filters

**Technical Analysis:**
- Query needs: `block='C' AND credits=7.5`
- Current retrieval doesn't support complex filters well

**Recommended Fix:**
1. Implement multi-filter support in smart retriever
2. Or retrieve broader set and filter in Python
3. Consider indexing combined metadata (e.g., "block_credits" field)

---

### Category 4: Domain Knowledge Gap (1 failure)
**Affected:** Q10 (Block mapping)

**Issue:** LP vs Block terminology mismatch

**Recommended Fix:**
1. Add mapping documentation to vector store:
   - "Block 3 corresponds to LP3"
   - "Spring semester includes LP3 (Block C) and LP4 (Block D)"
2. Or add preprocessing logic to convert block references

---

## Comparison: Before vs After Improvements

| Aspect | Before | After | Change |
|--------|--------|-------|--------|
| **Retrieval K** | 5 | 10 | +100% |
| **Prompt Strategy** | Strict "no info" | Flexible partial info | Improved |
| **Retrieval Method** | Basic similarity | Smart + course code | Enhanced |
| **Success Rate** | ~40-50% | 60% | +10-20% |
| **TDA357 in general query** | ❌ | ✅ | Fixed! |
| **TDA357 in specific query** | ❌ | ❌ | Still broken |

---

## Recommendations (Prioritized)

### 🔴 High Priority

**1. Fix Single Course Code Queries (Q7)**
- Debug metadata filtering in Chroma
- Add explicit course code index
- Test: `vectorstore.similarity_search("", k=20, filter={'course_code': 'TDA357'})`

**2. Improve Multi-Course Retrieval (Q3, Q8)**
- Dynamic K based on number of course codes: `K = BASE_K * max(1, len(course_codes))`
- Ensure minimum 5 chunks per detected course code
- Example: For Q8 with 2 courses, use K=20 instead of K=10

### 🟡 Medium Priority

**3. Add Block/LP Mapping Documentation**
- Create synthetic documents explaining block system:
  ```
  "Study periods at Chalmers:
   - Autumn: LP1 (Block A), LP2 (Block B)
   - Spring: LP3 (Block C), LP4 (Block D)"
  ```
- Add to vector store as metadata documents

**4. Implement Hybrid Search**
- Combine semantic search (current) with BM25 keyword search
- Especially useful for course codes (TDA357, DAT450, etc.)
- Libraries: `rank-bm25` or `whoosh`

### 🟢 Low Priority

**5. Complex Metadata Filtering**
- Support queries like "credits=7.5 AND block=C"
- Post-filtering in Python if Chroma doesn't support complex filters

**6. Query Expansion**
- For course code queries, also search for:
  - Course name
  - Related keywords
  - Department abbreviation

---

## Next Steps

1. **Immediate:** Run fix for metadata filtering and retest Q7
2. **Short-term:** Implement dynamic K for multi-course queries, retest Q3 & Q8
3. **Medium-term:** Add block mapping docs, retest Q10
4. **Long-term:** Consider hybrid search system upgrade

---

## Conclusion

The improvements made (K=10, relaxed prompt, smart retriever) have resulted in a **10-20% improvement** in success rate (from ~40-50% to 60%). Key achievements:

✅ **Successfully retrieves TDA357** in general database course queries  
✅ **Comprehensive results** for ML courses, prerequisites, project courses  
✅ **Better context** provided to LLM (10 docs vs 5)

However, critical issues remain:

❌ **Single course queries still fail** despite course being in database  
❌ **Multi-course comparisons** need more robust retrieval  
❌ **Metadata filtering** not working as expected

**The system is functional for 60% of queries and ready for basic use**, but requires the recommended fixes for production deployment.

---

**Report Generated:** January 10, 2026  
**Test Version:** Improved (K=10 + Smart Retriever)  
**Next Test:** After implementing High Priority fixes
