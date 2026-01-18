# Speaker Notes for Test Case Images
**演讲者备注 - 测试用例图片**

快速讲解版本（每个测试用例约30-40秒）

---

## Test Case 1: AI/ML Course Recommendation
**Image:** test_case_1_ai_ml.png

### English Speaker Notes (30 sec):
"Let's start with our first test case. A student wants to specialize in machine learning. Notice how our system provides a clear learning path: start with DAT340 for ML fundamentals, move to TDA234 for deep learning, and DAT565 for broader AI concepts. The system correctly identifies prerequisites - you need programming and statistics first. All responses include course URLs for verification. This shows our system understands course dependencies."

### 中文演讲备注 (30秒):
"我们看第一个测试用例。学生想专攻机器学习。注意系统给出了清晰的学习路径：从DAT340开始学ML基础，然后TDA234深度学习，再DAT565学更广的AI概念。系统正确识别了先修要求 - 需要编程和统计基础。所有回答都包含课程URL可供验证。这展示了系统理解课程依赖关系。"

### Key Points to Emphasize:
- ✅ Structured learning path (Foundation → Advanced → Specialized)
- ✅ Correct prerequisite identification
- ✅ All answers include verifiable URLs
- ✅ Natural language understanding

---

## Test Case 2: Schedule Conflict Detection
**Image:** test_case_2_schedule.png

### English Speaker Notes (30 sec):
"Second case demonstrates schedule conflict detection - a critical feature. The student asks if they can take two courses together. Our system checks the Block schedules: DAT565 is Block C, but MVE137 runs the entire semester, creating an overlap. The system immediately flags this as a potential conflict. This prevents students from making costly enrollment mistakes. Manual checking would require opening multiple pages and comparing schedules yourself - our system does this instantly."

### 中文演讲备注 (30秒):
"第二个案例展示时间冲突检测 - 这是关键功能。学生问能否同时选两门课。系统检查Block时间：DAT565是Block C，但MVE137是全学期课程，会产生重叠。系统立即标记为潜在冲突。这防止学生犯代价高昂的选课错误。手动检查需要打开多个页面自己比较 - 我们的系统瞬间完成。"

### Key Points to Emphasize:
- ⚠️ Automatic conflict detection
- ✅ 100% accuracy in schedule checking
- 💡 Prevents enrollment mistakes
- ⚡ Instant vs manual checking (seconds vs minutes)

---

## Test Case 3: Course Comparison & Analysis
**Image:** test_case_3_comparison.png

### English Speaker Notes (35 sec):
"Third case shows course comparison. The student wants to know which course is more hands-on. The system compares DAT340 and TDA234 in detail: DAT340 has balanced practical assignments covering classical algorithms, while TDA234 is more theory-focused on neural networks. It even compares assessment methods - DAT340 has projects plus exams, TDA234 is primarily exam-based. This level of analysis would take hours of manual research. Our system provides structured, actionable comparisons instantly."

### 中文演讲备注 (35秒):
"第三个案例展示课程比较。学生想知道哪门课更注重实践。系统详细比较DAT340和TDA234：DAT340有平衡的实践作业覆盖经典算法，而TDA234更偏理论侧重神经网络。甚至比较考核方式 - DAT340有项目加考试，TDA234主要是考试。这种深度分析手动需要几小时研究。我们的系统瞬间提供结构化、可操作的比较。"

### Key Points to Emphasize:
- 📊 Deep course comparison
- ✅ Multiple dimensions: content, assessment, focus
- 💡 Actionable insights for decision-making
- ⏱️ Saves hours of manual research

---

## Test Case 4: Data Science Learning Path
**Image:** test_case_4_learning_path.png

### English Speaker Notes (40 sec):
"Fourth case demonstrates learning path planning - the most complex scenario. A student wants to become a data scientist. The system generates a complete 3-phase roadmap: Foundation courses for statistics and programming, Core Skills with multiple options at the same level, and Specialization courses including a 15-credit project. Notice it also identifies schedule conflicts - DAT340 and MVE441 both in Block C. This is comprehensive curriculum planning powered by our RAG system. It's like having an academic advisor available 24/7."

### 中文演讲备注 (40秒):
"第四个案例展示学习路径规划 - 最复杂的场景。学生想成为数据科学家。系统生成完整的3阶段路线图：基础课程学统计和编程，核心技能有多个同级选项，专业化课程包括15学分大项目。注意它还识别时间冲突 - DAT340和MVE441都在Block C。这是由RAG系统驱动的全面课程规划。就像有个24/7在线的学术顾问。"

### Key Points to Emphasize:
- 🎓 Complete multi-phase learning roadmap
- ✅ Identifies conflicts across entire path
- 💡 Provides choice at each level
- 🤖 Like having a 24/7 academic advisor

**[Optional if time permits]:**
"This is where RAG really shines - combining information from multiple courses to create a coherent learning path."

---

## Test Case 5: Honest Limitation Handling
**Image:** test_case_5_limitation.png

### English Speaker Notes (35 sec):
"Finally, our fifth case demonstrates something crucial - honest limitation handling. When asked about TDA567, the system admits it doesn't have complete information. It tells you what it knows - the course is about testing and debugging - but directs you to official sources for complete details. This is GOOD design. Many AI systems hallucinate fake information when uncertain. Our system builds trust by being transparent about its limitations. This is a key feature, not a bug."

### 中文演讲备注 (35秒):
"最后，第五个案例展示关键的一点 - 诚实的局限性处理。当问及TDA567时，系统承认没有完整信息。它告诉你知道的内容 - 课程关于测试调试 - 但引导你去官方查完整信息。这是好的设计。很多AI系统在不确定时会编造假信息。我们的系统通过对局限性保持透明来建立信任。这是关键特性，不是缺陷。"

### Key Points to Emphasize:
- ⚠️ Honest admission of limitations
- ✅ No hallucinated information
- 🔗 Directs to authoritative sources
- 💡 Builds user trust through transparency

**[TRANSITION TO LIMITATIONS SLIDE]:**
"This leads us naturally to discussing our system's limitations..."

---

## Overall Delivery Strategy (Total: ~3 minutes for all 5 cases)

### Pacing:
- Case 1-2: Quick (30 sec each) - establish credibility
- Case 3-4: Medium (35-40 sec) - show depth
- Case 5: Important (35 sec) - transition to limitations

### What to Highlight Across All Cases:
1. **100% Success Rate**: All 5 cases passed testing
2. **No Hallucinations**: System never made up course codes
3. **Real Value**: Saves hours of manual research
4. **Trust Building**: Honest about limitations

### Body Language Tips:
- Point to specific parts of the image when mentioning them
- Use hand gestures to show "flow" for learning paths
- Show enthusiasm for success cases
- Pause briefly when showing the limitation case - it's a strength

### Backup Responses for Q&A:
**Q: "How did you verify accuracy?"**
A: "We manually checked every response against official Chalmers course pages and TimeEdit schedules."

**Q: "What if the database is outdated?"**
A: "That's a valid limitation. We recommend re-scraping annually, or we could implement automatic updates from the official API."

**Q: "Can it handle Chinese queries?"**
A: "Currently English only. Multilingual support is in our future roadmap."

---

## Quick Reference Card (Print This!)

```
TEST 1: ML Path → Foundation→Advanced→Specialized ✅
TEST 2: Schedule Check → Instant conflict detection ⚠️
TEST 3: Course Compare → Multi-dimension analysis 📊
TEST 4: Learning Path → 3-phase roadmap with options 🎓
TEST 5: Honesty → Admits limitations, builds trust 💡

KEY MESSAGE: Not just accurate - TRUSTWORTHY
```

---

## Time Management:

- **If running short on time**: Skip detailed explanation of Test 3, just say "The system also excels at course comparisons" and move to Test 4
- **If audience is engaged**: Add the optional RAG comment after Test 4
- **Always include**: Test 1 (establishes capability), Test 2 (practical value), Test 5 (shows maturity)

Good luck! 🎯
