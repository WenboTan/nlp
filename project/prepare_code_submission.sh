#!/bin/bash
# 代码提交打包脚本
# 运行方式: bash prepare_code_submission.sh

echo "=== 准备DAT450项目代码提交 ==="
echo ""

# 创建临时目录
SUBMISSION_DIR="code_submission"
rm -rf $SUBMISSION_DIR
mkdir -p $SUBMISSION_DIR

echo "📦 复制核心代码文件..."

# 复制源代码
cp -r src $SUBMISSION_DIR/
echo "  ✓ src/ 已复制"

# 复制测试文件
cp -r tests $SUBMISSION_DIR/
echo "  ✓ tests/ 已复制"

# 复制依赖文件
cp -r requirements $SUBMISSION_DIR/
echo "  ✓ requirements/ 已复制"

# 复制脚本
cp -r scripts $SUBMISSION_DIR/
echo "  ✓ scripts/ 已复制"

# 复制测试结果
cp -r test_results $SUBMISSION_DIR/
echo "  ✓ test_results/ 已复制"

# 复制文档
cp -r docs $SUBMISSION_DIR/
echo "  ✓ docs/ 已复制"

# 复制README文件
cp README.md $SUBMISSION_DIR/
cp CODE_SUBMISSION_README.md $SUBMISSION_DIR/
echo "  ✓ README文件已复制"

# 清理不必要的文件
echo ""
echo "🧹 清理临时文件..."
find $SUBMISSION_DIR -name "*.pyc" -delete
find $SUBMISSION_DIR -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find $SUBMISSION_DIR -name "*.log" -delete
find $SUBMISSION_DIR -name "*.err" -delete
echo "  ✓ 已清理 .pyc, __pycache__, .log, .err 文件"

# 统计文件
echo ""
echo "📊 文件统计:"
echo "  Python文件: $(find $SUBMISSION_DIR -name "*.py" | wc -l)"
echo "  文档文件: $(find $SUBMISSION_DIR -name "*.md" | wc -l)"
echo "  总文件数: $(find $SUBMISSION_DIR -type f | wc -l)"

# 显示目录结构
echo ""
echo "📁 提交包结构:"
tree -L 2 $SUBMISSION_DIR 2>/dev/null || find $SUBMISSION_DIR -maxdepth 2 -type d

echo ""
echo "✅ 代码已准备好，位于: $SUBMISSION_DIR/"
echo ""
echo "📤 下一步:"
echo "  1. 检查 $SUBMISSION_DIR/ 中的文件"
echo "  2. 压缩文件夹: tar -czf code_submission.tar.gz $SUBMISSION_DIR/"
echo "  3. 或直接上传 $SUBMISSION_DIR/ 中的所有文件到Canvas"
echo ""
