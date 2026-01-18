"""
Course Deduplication by Course Name
Keep all course codes for each course name
"""

import json
from collections import defaultdict

def deduplicate_courses_by_name(input_file, output_file):
    """
    Deduplicate courses by name, keep all possible course codes
    
    Args:
        input_file: Input JSON file path
        output_file: Output JSON file path
    """
    print("=" * 60)
    print("Course Deduplication - Merge by Name")
    print("=" * 60)
    
    # 1. Load data
    print(f"\n[Step 1] Loading data: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        courses = json.load(f)
    print(f"✓ Original courses: {len(courses)}")
    
    # 2. Group by course name
    print("\n[Step 2] Grouping by course name...")
    course_groups = defaultdict(list)
    
    for course in courses:
        course_name = course['courseName']
        course_groups[course_name].append(course)
    
    print(f"✓ Unique course names: {len(course_groups)}")
    
    # 3. Analysis
    print("\n[Step 3] Analyzing duplicates...")
    duplicate_names = {name: codes for name, codes in course_groups.items() if len(codes) > 1}
    print(f"  Courses with multiple codes: {len(duplicate_names)}")
    
    # Show top 10 duplicates
    print("\n  Top 10 duplicate examples:")
    for i, (name, course_list) in enumerate(list(duplicate_names.items())[:10], 1):
        codes = [c['courseCode'] for c in course_list]
        print(f"    {i}. {name}")
        print(f"       Codes: {', '.join(codes)}")
    
    # 4. Create deduplicated structure
    print("\n[Step 4] Creating deduplicated data...")
    deduplicated_courses = []
    
    for course_name, course_list in course_groups.items():
        # Sort by course code
        course_list_sorted = sorted(course_list, key=lambda x: x['courseCode'], reverse=True)
        
        # Merge entry with all codes
        merged_course = {
            "courseName": course_name,
            "primaryCourseCode": course_list_sorted[0]['courseCode'],
            "alternativeCodes": [c['courseCode'] for c in course_list_sorted[1:]],
            "allCodes": [c['courseCode'] for c in course_list_sorted],
            "programShort": course_list_sorted[0]['programShort'],
            "programLong": course_list_sorted[0]['programLong'],
            "passRate": course_list_sorted[0]['passRate'],
            "averageGrade": course_list_sorted[0]['averageGrade'],
            "totalPass": course_list_sorted[0]['totalPass'],
            "totalFail": course_list_sorted[0]['totalFail'],
            "total": course_list_sorted[0]['total']
        }
        
        deduplicated_courses.append(merged_course)
    
    # Sort by primary code
    deduplicated_courses.sort(key=lambda x: x['primaryCourseCode'])
    
    print(f"✓ Deduplicated courses: {len(deduplicated_courses)}")
    print(f"✓ Removed: {len(courses) - len(deduplicated_courses)} duplicates")
    
    # 5. Save results
    print(f"\n[Step 5] Saving data: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(deduplicated_courses, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved to: {output_file}")
    
    # 6. Statistics
    print("\n" + "=" * 60)
    print("Statistics:")
    print(f"  Original courses: {len(courses)}")
    print(f"  Deduplicated: {len(deduplicated_courses)}")
    print(f"  Reduction rate: {(1 - len(deduplicated_courses) / len(courses)) * 100:.2f}%")
    print(f"  With alternatives: {sum(1 for c in deduplicated_courses if c['alternativeCodes'])}")
    
    # Show course with most alternatives
    max_alternatives = max(deduplicated_courses, key=lambda x: len(x['alternativeCodes']))
    print(f"\n  Most alternatives:")
    print(f"    {max_alternatives['courseName']}")
    print(f"    Primary: {max_alternatives['primaryCourseCode']}")
    print(f"    Alternatives: {', '.join(max_alternatives['alternativeCodes'])}")
    
    print("=" * 60)
    
    return deduplicated_courses


if __name__ == "__main__":
    input_file = "chalmers_courses_details.json"
    output_file = "chalmers_courses_deduplicated.json"
    
    deduplicated_courses = deduplicate_courses_by_name(input_file, output_file)
    
    print("\n✓ 去重完成！")
    print(f"\n下一步：使用 {output_file} 进行爬取")
    print(f"  python syllabus_scraper.py")
