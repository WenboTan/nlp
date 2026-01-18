#!/usr/bin/env python3
"""
Full RAG System Batch Test with Detailed Logging and Report Generation
"""

import os
import sys
import json
from datetime import datetime
from test_rag_batch_gemini import (
    load_vector_store, 
    initialize_llm, 
    create_rag_chain,
    TEST_QUERIES
)

def run_full_test_with_logging():
    """Run full test and save detailed logs"""
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"test_results_gemini_{timestamp}.log"
    report_file = f"test_report_gemini_{timestamp}.txt"
    json_file = f"test_results_gemini_{timestamp}.json"
    
    print(f"\n{'='*70}")
    print(f"Chalmers RAG System - Full Test Report")
    print(f"{'='*70}")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_file}")
    print(f"Report file: {report_file}")
    print(f"{'='*70}\n")
    
    # Initialize system
    print("Initializing system...")
    vectorstore = load_vector_store()
    llm = initialize_llm()
    rag_chain = create_rag_chain(vectorstore, llm)
    print("✓ System ready\n")
    
    # Store test results
    results = []
    success_count = 0
    failed_queries = []
    
    # Run all tests
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n{'='*70}")
        print(f"Test Query {i}/{len(TEST_QUERIES)}")
        print(f"{'='*70}")
        print(f"Question: {query}\n")
        
        try:
            print("Generating answer...")
            answer = rag_chain.invoke(query)
            
            print(f"\n🤖 Answer:\n{answer}\n")
            
            # Check success
            is_success = "I don't have enough information" not in answer
            if is_success:
                success_count += 1
                status = "✓ SUCCESS"
            else:
                status = "✗ FAILED - Insufficient information"
                failed_queries.append({
                    'number': i,
                    'query': query,
                    'answer': answer[:200] + "..." if len(answer) > 200 else answer
                })
            
            print(f"Status: {status}")
            
            # Save results
            results.append({
                'query_number': i,
                'query': query,
                'answer': answer,
                'status': 'success' if is_success else 'failed',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            error_msg = f"ERROR: {str(e)}"
            print(f"\n❌ {error_msg}\n")
            failed_queries.append({
                'number': i,
                'query': query,
                'error': str(e)
            })
            results.append({
                'query_number': i,
                'query': query,
                'error': str(e),
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            })
    
    # Generate summary report
    print(f"\n{'='*70}")
    print(f"📊 TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Total queries: {len(TEST_QUERIES)}")
    print(f"Successful: {success_count} ({success_count/len(TEST_QUERIES)*100:.1f}%)")
    print(f"Failed: {len(failed_queries)} ({len(failed_queries)/len(TEST_QUERIES)*100:.1f}%)")
    print(f"{'='*70}\n")
    
    if failed_queries:
        print("Failed Queries:")
        for item in failed_queries:
            print(f"  Q{item['number']}: {item['query']}")
    
    # Save JSON results
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'total': len(TEST_QUERIES),
            'success': success_count,
            'failed': len(failed_queries),
            'results': results
        }, f, indent=2, ensure_ascii=False)
    print(f"\n✓ JSON results saved to: {json_file}")
    
    # Generate detailed report
    generate_detailed_report(report_file, results, success_count, failed_queries, timestamp)
    print(f"✓ Detailed report saved to: {report_file}")
    
    print(f"\n{'='*70}")
    print("Test completed successfully!")
    print(f"{'='*70}\n")
    
    return success_count, len(failed_queries)


def generate_detailed_report(filename, results, success_count, failed_queries, timestamp):
    """Generate detailed test report"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Chalmers Course RAG System - Test Report\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuration:\n")
        f.write(f"  - Model: Google Gemini gemini-2.5-flash\n")
        f.write(f"  - Retrieval K: 10 (up from 5)\n")
        f.write(f"  - Improvements: Relaxed prompt + Smart retriever with course code extraction\n\n")
        
        f.write("="*70 + "\n")
        f.write("SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        total = len(results)
        f.write(f"Total Queries: {total}\n")
        f.write(f"Successful: {success_count} ({success_count/total*100:.1f}%)\n")
        f.write(f"Failed: {len(failed_queries)} ({len(failed_queries)/total*100:.1f}%)\n\n")
        
        f.write("="*70 + "\n")
        f.write("FAILED QUERIES\n")
        f.write("="*70 + "\n\n")
        
        if failed_queries:
            for item in failed_queries:
                f.write(f"Query #{item['number']}:\n")
                f.write(f"  Question: {item['query']}\n")
                if 'error' in item:
                    f.write(f"  Error: {item['error']}\n")
                else:
                    f.write(f"  Answer: {item.get('answer', 'N/A')}\n")
                f.write("\n")
        else:
            f.write("No failed queries! All tests passed.\n\n")
        
        f.write("="*70 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("="*70 + "\n\n")
        
        for result in results:
            f.write(f"Query #{result['query_number']}\n")
            f.write(f"{'='*70}\n")
            f.write(f"Question: {result['query']}\n\n")
            f.write(f"Status: {result['status'].upper()}\n\n")
            
            if 'answer' in result:
                f.write(f"Answer:\n{result['answer']}\n\n")
            elif 'error' in result:
                f.write(f"Error: {result['error']}\n\n")
            
            f.write(f"{'-'*70}\n\n")
        
        f.write("="*70 + "\n")
        f.write("IMPROVEMENT RECOMMENDATIONS\n")
        f.write("="*70 + "\n\n")
        
        f.write("Based on failed queries, recommended improvements:\n\n")
        f.write("1. Fix metadata filtering for single course code queries\n")
        f.write("   - Queries like 'Tell me about TDA357' should trigger exact course lookup\n\n")
        f.write("2. Improve multi-course retrieval for comparison queries\n")
        f.write("   - Ensure both courses are retrieved when comparing (e.g., DAT450 vs TDA362)\n\n")
        f.write("3. Add Block/LP mapping documentation\n")
        f.write("   - Help system understand 'Block 3 and 4' means specific schedule blocks\n\n")
        f.write("4. Consider hybrid search (BM25 + semantic)\n")
        f.write("   - Improve keyword matching for course codes and specific terms\n\n")


if __name__ == "__main__":
    try:
        success, failed = run_full_test_with_logging()
        sys.exit(0 if failed == 0 else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
