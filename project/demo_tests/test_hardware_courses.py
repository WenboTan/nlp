"""
测试硬件相关课程的 RAG 查询
用于 presentation 演示
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from rag_query_system_gemini import create_rag_chain, load_vector_store, load_gemini_llm

def test_hardware_courses():
    """测试硬件相关课程查询"""
    
    print("=" * 80)
    print("硬件课程 RAG 系统测试")
    print("=" * 80)
    print()
    
    # 加载系统
    print("🔧 正在加载 RAG 系统...")
    vectorstore = load_vector_store()
    llm = load_gemini_llm()
    rag_chain = create_rag_chain(vectorstore, llm)
    print("✓ RAG 系统加载完成\n")
    
    # 测试查询
    test_queries = [
        {
            "query": "I want to learn digital circuit design and FPGA. Which courses should I take?",
            "description": "数字电路设计和 FPGA"
        },
        {
            "query": "Tell me about DAT110. What will I learn and what are the prerequisites?",
            "description": "DAT110 课程详情"
        },
        {
            "query": "I'm interested in embedded systems. What courses are available?",
            "description": "嵌入式系统相关课程"
        },
        {
            "query": "Compare EDA234 and MCC093. Which one focuses more on hands-on projects?",
            "description": "课程对比 (EDA234 vs MCC093)"
        },
        {
            "query": "What is the course content and grading system for DAT105 Computer Architecture?",
            "description": "DAT105 课程内容和评分"
        }
    ]
    
    results = []
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n{'=' * 80}")
        print(f"测试 {i}/{len(test_queries)}: {test['description']}")
        print(f"{'=' * 80}")
        print(f"问题: {test['query']}")
        print(f"{'-' * 80}")
        
        try:
            response = rag_chain.invoke(test['query'])
            print(f"回答:\n{response}")
            results.append({
                "query": test['query'],
                "description": test['description'],
                "response": response,
                "success": True
            })
        except Exception as e:
            error_msg = f"❌ 错误: {str(e)}"
            print(error_msg)
            results.append({
                "query": test['query'],
                "description": test['description'],
                "error": str(e),
                "success": False
            })
        
        print()
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    success_count = sum(1 for r in results if r['success'])
    print(f"成功: {success_count}/{len(test_queries)}")
    print(f"失败: {len(test_queries) - success_count}/{len(test_queries)}")
    
    if success_count == len(test_queries):
        print("\n✅ 所有测试通过！系统已准备好用于 presentation！")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息")
    
    return results

if __name__ == "__main__":
    test_hardware_courses()
