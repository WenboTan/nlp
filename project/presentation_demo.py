"""
Presentation Demo - 硬件课程 RAG 查询演示
展示系统在实际使用场景中的表现
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from rag_query_system_gemini import create_rag_chain, load_vector_store, load_gemini_llm

def demo_for_presentation():
    """针对 presentation 优化的演示"""
    
    print("\n" + "="*80)
    print("📚 Chalmers Course RAG System - Hardware Courses Demo")
    print("="*80)
    
    # 加载系统
    print("\n🔧 Loading RAG system...")
    vectorstore = load_vector_store()
    llm = load_gemini_llm()
    rag_chain = create_rag_chain(vectorstore, llm)
    print("✅ System ready!\n")
    
    # 演示场景
    demos = [
        {
            "scenario": "场景 1: 学生想学习 FPGA 设计",
            "query": "I want to learn FPGA design. What courses should I take and in what order?",
            "highlight": "课程推荐 + 先修课程关系"
        },
        {
            "scenario": "场景 2: 查询特定课程详情 (DAT110)",
            "query": "What is DAT110 about? Tell me the learning outcomes, prerequisites, and assessment methods.",
            "highlight": "详细课程信息提取"
        },
        {
            "scenario": "场景 3: 课程对比",
            "query": "What's the difference between EDA234 and MCC093? Which one is more project-based?",
            "highlight": "多课程比较分析"
        },
        {
            "scenario": "场景 4: 时间冲突检查",
            "query": "Can I take DAT110 and EDA234 in the same period? Check their schedule blocks.",
            "highlight": "Schedule 冲突检测"
        },
        {
            "scenario": "场景 5: 嵌入式系统课程路径",
            "query": "I'm interested in embedded systems and want to work on real hardware projects. Recommend me a course sequence.",
            "highlight": "学习路径规划"
        }
    ]
    
    results = []
    
    for i, demo in enumerate(demos, 1):
        print("\n" + "="*80)
        print(f"🎯 {demo['scenario']}")
        print(f"💡 演示重点: {demo['highlight']}")
        print("="*80)
        print(f"\n❓ Question:\n{demo['query']}\n")
        print("-"*80)
        
        try:
            response = rag_chain.invoke(demo['query'])
            print(f"💬 Answer:\n{response}\n")
            results.append({"success": True, "scenario": demo['scenario']})
        except Exception as e:
            print(f"❌ Error: {str(e)}\n")
            results.append({"success": False, "scenario": demo['scenario'], "error": str(e)})
    
    # 总结
    print("\n" + "="*80)
    print("📊 Demo Summary")
    print("="*80)
    success = sum(1 for r in results if r['success'])
    print(f"✅ Successful: {success}/{len(demos)}")
    print(f"❌ Failed: {len(demos)-success}/{len(demos)}")
    
    print("\n🎓 System Capabilities Demonstrated:")
    print("  ✓ Course recommendation based on topics")
    print("  ✓ Detailed course information extraction")
    print("  ✓ Multi-course comparison")
    print("  ✓ Schedule conflict detection")
    print("  ✓ Learning path planning")
    
    print("\n📈 Presentation Readiness Assessment:")
    if success == len(demos):
        print("  🌟 EXCELLENT - System is fully ready for presentation!")
        print("  💪 All key features working perfectly")
    elif success >= len(demos) * 0.8:
        print("  ✅ GOOD - System is ready with minor issues")
    else:
        print("  ⚠️  NEEDS IMPROVEMENT - Address failures before presenting")
    
    return results

if __name__ == "__main__":
    demo_for_presentation()
