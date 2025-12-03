import requests
import re
import json
from bs4 import BeautifulSoup

def get_courses_with_details_from_api():
    """
    从 API 直接获取课程详细数据（包括通过率、平均成绩等）
    API 文档: https://github.com/Fysikteknologsektionen/chalmers-course-stats/blob/master/API.md
    """
    print("正在从 API 获取课程详细数据...")
    
    try:
        # 使用官方 API 端点获取所有课程
        # 设置 items=10000 确保获取所有课程（网站大概有 3000+ 门课程）
        api_url = "https://stats.ftek.se/courses?items=10000&page=0"
        
        print(f"  请求: {api_url}")
        response = requests.get(api_url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ 成功从 API 获取数据！")
            
            # API 返回格式: {"courses": [...], "metadata": [{"count": xxx}]}
            courses_info = []
            
            if 'courses' in data:
                for course in data['courses']:
                    # 提取课程信息
                    course_data = {
                        'courseCode': course.get('courseCode', 'N/A'),
                        'courseName': course.get('courseName', 'N/A'),
                        'programShort': course.get('programShort', 'N/A'),
                        'programLong': course.get('programLong', 'N/A'),
                        'passRate': round(course.get('passRate', 0) * 100, 2),  # 转换为百分比
                        'averageGrade': round(course.get('averageGrade', 0), 2),
                        'totalPass': course.get('totalPass', 0),
                        'totalFail': course.get('U', 0),  # U 表示不及格
                        'total': course.get('total', 0)
                    }
                    courses_info.append(course_data)
                
                total_count = data.get('metadata', [{}])[0].get('count', 0)
                print(f"  数据库中共有 {total_count} 门课程")
                print(f"  成功提取 {len(courses_info)} 门课程的详细信息")
                
                return courses_info
            else:
                print("  ✗ API 返回数据格式不符")
                return None
        else:
            print(f"  ✗ API 请求失败，状态码: {response.status_code}")
            return None
        
    except Exception as e:
        print(f"  ✗ API 获取失败: {e}")
        return None

def get_unique_course_codes_from_stats(url):
    """
    从统计页面暴力提取所有符合 Chalmers 格式的课程代码
    """
    print(f"正在访问统计页面: {url} ...")
    
    try:
        # 1. 获取网页内容
        # 这里的 headers 是为了模拟浏览器，防止被简单的反爬虫拦截
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status() # 如果状态码不是200，抛出异常
        
        content = response.text
        
        # 2. 尝试从页面源码中查找数据
        # 有些网站会在 <script> 标签中嵌入 JSON 数据
        soup = BeautifulSoup(content, 'html.parser')
        
        # 查找所有 script 标签中的课程代码
        codes_from_scripts = set()
        for script in soup.find_all('script'):
            script_content = script.string
            if script_content:
                # 提取所有可能的课程代码
                pattern = r'[A-Z]{3}\d{3}'
                codes_from_scripts.update(re.findall(pattern, script_content))
        
        # 同时从整个页面文本中提取
        pattern = r'\b[A-Z]{3}\d{3}\b'
        all_matches = re.findall(pattern, content)
        
        # 合并结果
        all_codes = list(set(all_matches) | codes_from_scripts)
        
        # 3. 使用 set() 自动去重
        unique_codes = sorted(list(set(all_codes)))
        
        print(f"原始匹配数: {len(all_codes)}")
        print(f"从 scripts 中找到: {len(codes_from_scripts)}")
        print(f"去重后唯一代码数: {len(unique_codes)}")
        
        return unique_codes

    except Exception as e:
        print(f"发生错误: {e}")
        return []

# --- 执行 ---
if __name__ == "__main__":
    print("=" * 60)
    print("Chalmers 课程详细信息爬虫")
    print("=" * 60)
    
    # 从 API 获取课程详细信息
    courses_data = get_courses_with_details_from_api()
    
    # 输出结果
    print("\n" + "=" * 60)
    if courses_data:
        print(f"共获取 {len(courses_data)} 门课程的详细信息")
        print("=" * 60)
        
        # 显示前 10 个示例
        print("\n前 10 门课程示例:")
        for i, course in enumerate(courses_data[:10], 1):
            print(f"\n{i:2d}. {course['courseCode']} - {course['courseName']}")
            print(f"    项目: {course['programShort']}")
            print(f"    通过率: {course['passRate']}%")
            print(f"    平均成绩: {course['averageGrade']}")
            print(f"    通过人数: {course['totalPass']}, 挂科人数: {course['totalFail']}")
        
        # 保存到 JSON 文件
        json_file = "chalmers_courses_details.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(courses_data, f, ensure_ascii=False, indent=2)
        print(f"\n✓ 详细信息已保存到: {json_file}")
        
        # 同时保存一个只有课程代码的简化版本
        codes_only = [course['courseCode'] for course in courses_data]
        txt_file = "chalmers_course_codes.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            for code in codes_only:
                f.write(code + '\n')
        print(f"✓ 课程代码列表已保存到: {txt_file}")
        
        # 统计信息
        print(f"\n课程统计信息:")
        print(f"  总课程数: {len(courses_data)}")
        
        # 通过率统计
        pass_rates = [c['passRate'] for c in courses_data if c['passRate'] > 0]
        if pass_rates:
            print(f"  平均通过率: {sum(pass_rates) / len(pass_rates):.2f}%")
            print(f"  最高通过率: {max(pass_rates):.2f}%")
            print(f"  最低通过率: {min(pass_rates):.2f}%")
        
        # 平均成绩统计
        avg_grades = [c['averageGrade'] for c in courses_data if c['averageGrade'] > 0]
        if avg_grades:
            print(f"  平均成绩: {sum(avg_grades) / len(avg_grades):.2f}")
            print(f"  最高平均成绩: {max(avg_grades):.2f}")
            print(f"  最低平均成绩: {min(avg_grades):.2f}")
        
        # 课程代码前缀统计
        print(f"\n课程代码前缀统计 (Top 10):")
        prefix_count = {}
        for course in courses_data:
            prefix = course['courseCode'][:3]
            prefix_count[prefix] = prefix_count.get(prefix, 0) + 1
        
        for prefix, count in sorted(prefix_count.items(), key=lambda x: -x[1])[:10]:
            print(f"  {prefix}: {count} 门课程")
        
    else:
        print("未能获取课程数据")
        print("=" * 60)
