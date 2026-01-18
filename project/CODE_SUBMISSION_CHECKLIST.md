## 代码提交清单 - DAT450/DIT247

### 📋 需要提交的文件

#### 核心源代码 (src/)
- [x] `src/build_vector_db.py` - 构建向量数据库
- [x] `src/rag_query_system_gemini.py` - Gemini版本RAG系统（主要）
- [x] `src/rag_query_system_openai.py` - OpenAI版本
- [x] `src/rag_query_system_local.py` - 本地Mistral版本
- [x] `src/syllabus_scraper.py` - 课程数据爬虫
- [x] `src/deduplicate_courses.py` - 数据去重

#### 测试代码 (tests/)
- [x] `tests/run_full_test.py` - 完整测试套件
- [x] `tests/test_rag_batch_gemini.py` - Gemini批量测试
- [x] `tests/test_rag_batch.py` - 通用批量测试
- [x] `tests/test_rag_setup.py` - 系统验证

#### 配置文件 (requirements/)
- [x] `requirements/requirements_gemini.txt`
- [x] `requirements/requirements_openai.txt`
- [x] `requirements/requirements_local.txt`

#### 脚本 (scripts/)
- [x] `scripts/run_build_db.sh`
- [x] `scripts/run_rag_gemini.sh`
- [x] `scripts/run_rag_openai.sh`
- [x] `scripts/run_rag_local.sh`

#### 测试结果 (test_results/)
- [x] `test_results/TEST_REPORT_GEMINI_IMPROVED.md`
- [x] `test_results/test_results_gemini_improved_20260110.json`
- [x] `test_results/TEST_RESULTS_README.md`

#### 文档 (docs/)
- [x] `docs/PROJECT_OVERVIEW.md`
- [x] `docs/GEMINI_GUIDE.md`
- [x] `docs/MODEL_COMPARISON.md`
- [x] `docs/START_HERE.md`

#### README文件
- [x] `README.md` - 主文档
- [x] `CODE_SUBMISSION_README.md` - 代码说明

---

### ❌ 不需要提交的文件

- `data/chalmers_courses_full_scraped.json` (114MB - 太大)
- `chalmers_chroma_db/` (14GB - 太大)
- `logs/*.err` (日志文件)
- `src/__pycache__/` (Python缓存)
- `*.pyc` (编译文件)
- `archive/` (旧版本文件)

---

### 🚀 快速打包提交

#### 方法1: 使用提供的脚本
```bash
cd /data/users/wenbota/nlp/project
bash prepare_code_submission.sh
```

#### 方法2: 手动打包
```bash
cd /data/users/wenbota/nlp/project

# 创建提交文件夹
mkdir code_submission
cp -r src tests requirements scripts test_results docs code_submission/
cp README.md CODE_SUBMISSION_README.md code_submission/

# 清理
find code_submission -name "*.pyc" -delete
find code_submission -name "__pycache__" -type d -rm -rf

# 压缩（可选）
tar -czf code_submission.tar.gz code_submission/
```

#### 方法3: 直接上传文件
直接选择这些文件夹上传到Canvas:
- `src/`
- `tests/`
- `requirements/`
- `scripts/`
- `test_results/`
- `docs/`
- `README.md`
- `CODE_SUBMISSION_README.md`

---

### 📝 提交前检查

- [ ] 所有源代码文件都包含在内
- [ ] 没有包含敏感信息（API keys等）
- [ ] README清楚说明如何运行代码
- [ ] 测试结果文件已包含
- [ ] 没有包含大文件（data/，数据库等）
- [ ] Python缓存文件已清理

---

### 📤 Canvas提交步骤

1. **进入Canvas课程页面**
2. **找到代码提交作业**
3. **上传文件**:
   - 选项A: 上传 `code_submission.tar.gz`
   - 选项B: 上传 `code_submission/` 文件夹内所有文件
   - 选项C: 分别上传各个文件夹
4. **确认上传成功**
5. **点击提交**

---

### 📊 预期文件大小

- 源代码: ~500KB
- 测试文件: ~200KB
- 文档: ~100KB
- 测试结果: ~50KB
- **总计: < 1MB** (不含数据和数据库)

---

### ⏰ 截止时间

**今天: January 18, 2026**

---

### 💡 提示

如果Canvas有文件大小限制，记得：
1. 不要包含 `data/` 和 `chalmers_chroma_db/`
2. 在 `CODE_SUBMISSION_README.md` 中说明这些文件可按需提供
3. 或提供Google Drive/GitHub链接
