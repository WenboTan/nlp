# 项目整理说明 (2026-01-10)

## ✅ 整理完成

项目文件已按功能分类整理到专门的目录中。

## 📁 新目录结构

```
nlp/project/
├── data/           # 数据文件（1个：courses JSON）
├── src/            # 源代码（6个Python文件）
├── scripts/        # Shell脚本（5个SLURM脚本）
├── tests/          # 测试代码（4个测试文件）
├── test_results/   # 测试结果和报告（3-4个文件）
├── docs/           # 文档（9个文档文件）
├── requirements/   # 依赖文件（3个requirements）
├── chalmers_chroma_db/  # 向量数据库（14GB，保持不动）
├── logs/           # SLURM日志（保持不动）
├── archive/        # 旧文件备份（保持不动）
└── README.md       # 更新的主README
```

## 🗑️ 已删除

- `__pycache__/` - Python缓存目录
- 重复的旧测试日志（保留最新的）

## 📋 文件映射

### 数据文件
- `chalmers_courses_full_scraped.json` → `data/`

### 源代码
- `build_vector_db.py` → `src/`
- `deduplicate_courses.py` → `src/`
- `syllabus_scraper.py` → `src/`
- `rag_query_system_gemini.py` → `src/`
- `rag_query_system_local.py` → `src/`
- `rag_query_system_openai.py` → `src/`

### 脚本
- `run_*.sh` → `scripts/`

### 测试
- `test_*.py` → `tests/`
- `run_full_test.py` → `tests/`

### 测试结果
- `TEST_REPORT_*.md` → `test_results/`
- `TEST_RESULTS_README.md` → `test_results/`
- `test_results_*.json` → `test_results/`
- `test_results_*.log` → `test_results/`

### 文档
- `*.md` (除了README.md) → `docs/`
- `*.txt` → `docs/`

### 依赖
- `requirements_*.txt` → `requirements/`

## 📝 重要变更

1. **README.md已更新**：新版本更清晰，包含完整的项目结构和导航
2. **旧README已备份**：保存为 `README_OLD.md`
3. **路径更改**：所有导入路径和脚本需要相应更新

## ⚠️ 注意事项

运行脚本时可能需要更新路径：

```bash
# 旧的
python3 rag_query_system_gemini.py

# 新的
cd src && python3 rag_query_system_gemini.py
# 或
python3 src/rag_query_system_gemini.py
```

SLURM脚本已在 `scripts/` 目录中，使用时：

```bash
cd scripts
sbatch run_rag_gemini.sh
```

## 🚀 快速开始

- 新用户：`cat docs/START_HERE.md`
- Gemini用户：`cat docs/GEMINI_QUICKSTART.txt`
- 查看测试结果：`cat test_results/TEST_RESULTS_README.md`

## 📊 文件统计

- 总文件数：~35个（不含向量DB和日志）
- 源代码：6个
- 脚本：5个
- 测试：4个
- 文档：9个
- 依赖：3个
- 数据：1个（114MB）

---

整理完成时间：2026年1月10日  
整理目的：提高项目可维护性和可读性
