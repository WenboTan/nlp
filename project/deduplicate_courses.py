"""
根据课程名称对课程代码进行去重
保留每个课程名称对应的所有课程代码（作为备选）
"""

import json
from collections import defaultdict

def deduplicate_courses_by_name(input_file, output_file):
    """
    根据课程名称去重，为每个课程名称保留所有可能的课程代码
    
    Args:
        input_file: 输入的 JSON 文件路径
        output_file: 输出的去重后 JSON 文件路径
    """
    print("=" * 60)
    print("课程去重工具 - 根据课程名称合并")
    print("=" * 60)
    
    # 1. 读取原始数据
    print(f"\n[步骤 1] 读取原始数据: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        courses = json.load(f)
    print(f"✓ 原始课程数: {len(courses)}")
    
    # 2. 按课程名称分组
    print("\n[步骤 2] 按课程名称分组...")
    course_groups = defaultdict(list)
    
    for course in courses:
        course_name = course['courseName']
        course_groups[course_name].append(course)
    
    print(f"✓ 唯一课程名称数: {len(course_groups)}")
    
    # 3. 统计分析
    print("\n[步骤 3] 分析重复情况...")
    duplicate_names = {name: codes for name, codes in course_groups.items() if len(codes) > 1}
    print(f"  有多个课程代码的课程: {len(duplicate_names)}")
    
    # 显示前 10 个重复的示例
    print("\n  前 10 个重复课程示例:")
    for i, (name, course_list) in enumerate(list(duplicate_names.items())[:10], 1):
        codes = [c['courseCode'] for c in course_list]
        print(f"    {i}. {name}")
        print(f"       课程代码: {', '.join(codes)}")
    
    # 4. 创建去重后的数据结构
    print("\n[步骤 4] 创建去重数据...")
    deduplicated_courses = []
    
    for course_name, course_list in course_groups.items():
        # 按课程代码排序（通常新代码在后面）
        course_list_sorted = sorted(course_list, key=lambda x: x['courseCode'], reverse=True)
        
        # 创建一个合并的条目，包含所有可能的课程代码
        merged_course = {
            "courseName": course_name,
            "primaryCourseCode": course_list_sorted[0]['courseCode'],  # 主课程代码（最新的）
            "alternativeCodes": [c['courseCode'] for c in course_list_sorted[1:]],  # 备选代码
            "allCodes": [c['courseCode'] for c in course_list_sorted],  # 所有代码
            "programShort": course_list_sorted[0]['programShort'],
            "programLong": course_list_sorted[0]['programLong'],
            "passRate": course_list_sorted[0]['passRate'],
            "averageGrade": course_list_sorted[0]['averageGrade'],
            "totalPass": course_list_sorted[0]['totalPass'],
            "totalFail": course_list_sorted[0]['totalFail'],
            "total": course_list_sorted[0]['total']
        }
        
        deduplicated_courses.append(merged_course)
    
    # 按主课程代码排序
    deduplicated_courses.sort(key=lambda x: x['primaryCourseCode'])
    
    print(f"✓ 去重后课程数: {len(deduplicated_courses)}")
    print(f"✓ 减少了: {len(courses) - len(deduplicated_courses)} 个重复课程")
    
    # 5. 保存结果
    print(f"\n[步骤 5] 保存去重数据: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(deduplicated_courses, f, ensure_ascii=False, indent=2)
    print(f"✓ 已保存到: {output_file}")
    
    # 6. 输出统计信息
    print("\n" + "=" * 60)
    print("统计信息:")
    print(f"  原始课程总数: {len(courses)}")
    print(f"  去重后课程数: {len(deduplicated_courses)}")
    print(f"  去重率: {(1 - len(deduplicated_courses) / len(courses)) * 100:.2f}%")
    print(f"  有备选代码的课程: {sum(1 for c in deduplicated_courses if c['alternativeCodes'])}")
    
    # 显示备选代码最多的课程
    max_alternatives = max(deduplicated_courses, key=lambda x: len(x['alternativeCodes']))
    print(f"\n  备选代码最多的课程:")
    print(f"    {max_alternatives['courseName']}")
    print(f"    主代码: {max_alternatives['primaryCourseCode']}")
    print(f"    备选: {', '.join(max_alternatives['alternativeCodes'])}")
    
    print("=" * 60)
    
    return deduplicated_courses


if __name__ == "__main__":
    input_file = "chalmers_courses_details.json"
    output_file = "chalmers_courses_deduplicated.json"
    
    deduplicated_courses = deduplicate_courses_by_name(input_file, output_file)
    
    print("\n✓ 去重完成！")
    print(f"\n下一步：使用 {output_file} 进行爬取")
    print(f"  python syllabus_scraper.py")
