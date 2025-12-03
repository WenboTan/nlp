"""
Chalmers 课程大纲爬虫 (Syllabus Scraper)
用于从 Chalmers Student Portal 提取结构化课程信息，构建 RAG 训练数据集
"""

import requests
import json
import re
from bs4 import BeautifulSoup
from datetime import datetime
from typing import Dict, List, Optional, Any
import time


class ChalmersSyllabusScraper:
    """Chalmers 课程大纲爬虫类"""
    
    def __init__(self, academic_year: str = "2025/2026"):
        self.base_url = "https://www.chalmers.se/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus"
        self.academic_year = academic_year
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def fetch_course_page(self, course_code: str) -> Optional[str]:
        """
        获取课程页面的 HTML 内容
        
        Args:
            course_code: 课程代码 (e.g., "TEK285")
            
        Returns:
            HTML 字符串，或 None（如果请求失败）
        """
        # 新的 URL 格式: https://www.chalmers.se/en/.../course-syllabus/COURSE_CODE/?acYear=2025%2F2026
        encoded_year = self.academic_year.replace('/', '%2F')
        url = f"{self.base_url}/{course_code}/?acYear={encoded_year}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                return response.text
            else:
                print(f"✗ HTTP {response.status_code}")
                return None
        except Exception as e:
            print(f"✗ {e}")
            return None
    
    def clean_text(self, text: Optional[str]) -> str:
        """
        清洗文本：移除多余空白、换行符
        
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本
        """
        if not text:
            return ""
        
        # 移除 HTML 标签（如果有残留）
        text = re.sub(r'<[^>]+>', '', text)
        
        # 移除导航噪音
        noise_patterns = [
            r'Print page.*?', r'To personal page.*?', 
            r'Go to coursepage.*?', r'Opens in new tab.*?',
            r'Back to search.*?', r'Contact.*?'
        ]
        for pattern in noise_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # 统一换行符，移除连续换行
        text = re.sub(r'\n\s*\n+', '\n', text)
        
        # 移除多余空格
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def extract_text_after_label(self, soup: BeautifulSoup, label: str) -> Optional[str]:
        """
        提取标签后的文本内容（通用方法）
        
        Args:
            soup: BeautifulSoup 对象
            label: 要查找的标签文本 (e.g., "Eligibility")
            
        Returns:
            提取的文本，或 None
        """
        # 查找包含标签的元素
        label_element = soup.find(string=re.compile(label, re.IGNORECASE))
        if not label_element:
            return None
        
        # 获取父元素
        parent = label_element.find_parent()
        if not parent:
            return None
        
        # 尝试获取下一个兄弟元素的文本
        next_sibling = parent.find_next_sibling()
        if next_sibling:
            text = next_sibling.get_text(separator=' ', strip=True)
            return self.clean_text(text)
        
        # 如果没有兄弟元素，尝试获取父元素的文本
        text = parent.get_text(separator=' ', strip=True)
        # 移除标签本身
        text = re.sub(label, '', text, flags=re.IGNORECASE)
        return self.clean_text(text)
    
    def extract_prerequisites(self, soup: BeautifulSoup) -> str:
        """提取先修课程要求"""
        # 尝试查找 "Specific entry requirements" 或 "Course specific prerequisites"
        prerequisites = []
        
        for label in ["Specific entry requirements", "Course specific prerequisites", "Prerequisites"]:
            text = self.extract_text_after_label(soup, label)
            if text and text.lower() != "none":
                prerequisites.append(text)
        
        return " | ".join(prerequisites) if prerequisites else "None"
    
    def extract_eligibility(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """提取选课资格信息"""
        eligibility_text = self.extract_text_after_label(soup, "Eligibility")
        
        # 检测是否对交换生开放
        is_open_for_exchange = False
        if eligibility_text:
            if re.search(r'open for exchange', eligibility_text, re.IGNORECASE):
                is_open_for_exchange = "Yes" in eligibility_text or "open" in eligibility_text.lower()
        
        return {
            "text": eligibility_text or "",
            "open_for_exchange": is_open_for_exchange
        }
    
    def extract_programs(self, soup: BeautifulSoup) -> List[str]:
        """提取课程所属项目及其必修/选修状态"""
        programs = []
        
        # 查找 "In programmes" 部分
        in_programmes_header = soup.find(string=re.compile("In programmes", re.IGNORECASE))
        if in_programmes_header:
            parent = in_programmes_header.find_parent()
            if parent:
                # 查找该部分下的所有列表项
                ul = parent.find_next('ul')
                if ul:
                    for li in ul.find_all('li'):
                        text = li.get_text(strip=True)
                        programs.append(self.clean_text(text))
        
        return programs
    
    def extract_block_schedule(self, soup: BeautifulSoup) -> Optional[str]:
        """提取 Block schedule (关键字段，用于冲突检测)"""
        block_text = self.extract_text_after_label(soup, "Block schedule")
        if block_text:
            # 清洗：保留字母、加号、减号
            block_text = re.sub(r'[^A-Za-z+\-]', '', block_text)
        return block_text
    
    def extract_study_period(self, soup: BeautifulSoup) -> List[str]:
        """提取学习期间 (Study Period)"""
        study_periods = []
        
        # 查找 Module 表格中高亮的 SP
        # 通常高亮的单元格有特殊的 class 或 style
        table = soup.find('table', class_=re.compile('module', re.IGNORECASE))
        if table:
            for cell in table.find_all(['td', 'th']):
                # 检查是否包含 Sp1, Sp2 等
                text = cell.get_text(strip=True)
                if re.match(r'Sp\d', text):
                    # 检查是否高亮（背景色、粗体等）
                    if cell.get('class') or cell.get('style'):
                        study_periods.append(text)
        
        return study_periods if study_periods else ["Unknown"]
    
    def extract_exam_dates(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """提取考试日期和时间段"""
        exam_dates = []
        
        exam_header = soup.find(string=re.compile("Examination dates", re.IGNORECASE))
        if exam_header:
            parent = exam_header.find_parent()
            if parent:
                # 查找下一个包含日期的元素
                next_elem = parent.find_next()
                if next_elem:
                    text = next_elem.get_text(separator=' ', strip=True)
                    # 解析日期格式 (e.g., "2025-10-10 am")
                    date_pattern = r'(\d{4}-\d{2}-\d{2})\s*(am|pm|morning|afternoon)?'
                    matches = re.findall(date_pattern, text, re.IGNORECASE)
                    for date, time_period in matches:
                        exam_dates.append({
                            "date": date,
                            "time": time_period.lower() if time_period else "unknown"
                        })
        
        return exam_dates
    
    def extract_teaching_language(self, soup: BeautifulSoup) -> str:
        """提取授课语言"""
        language = self.extract_text_after_label(soup, "Teaching language")
        return language or "Unknown"
    
    def extract_credits(self, soup: BeautifulSoup) -> float:
        """提取学分"""
        credits_text = self.extract_text_after_label(soup, "Credits")
        if credits_text:
            # 提取数字
            match = re.search(r'(\d+\.?\d*)', credits_text)
            if match:
                return float(match.group(1))
        return 0.0
    
    def extract_grading_scale(self, soup: BeautifulSoup) -> str:
        """提取评分标准"""
        grading = self.extract_text_after_label(soup, "Grading scale")
        return grading or "Unknown"
    
    def extract_learning_outcomes(self, soup: BeautifulSoup) -> str:
        """提取学习成果 (用于 RAG Embedding)"""
        outcomes = []
        
        outcomes_header = soup.find(string=re.compile("Learning outcomes", re.IGNORECASE))
        if outcomes_header:
            parent = outcomes_header.find_parent()
            if parent:
                # 查找下一个包含内容的元素（可能是 ul, ol, p）
                next_elem = parent.find_next(['ul', 'ol', 'p', 'div'])
                if next_elem:
                    # 如果是列表，提取所有列表项
                    if next_elem.name in ['ul', 'ol']:
                        for li in next_elem.find_all('li'):
                            outcomes.append(li.get_text(strip=True))
                    else:
                        outcomes.append(next_elem.get_text(strip=True))
        
        combined_text = " ".join(outcomes)
        return self.clean_text(combined_text)
    
    def extract_content_summary(self, soup: BeautifulSoup) -> str:
        """提取课程内容概述 (用于 RAG Embedding)"""
        content = self.extract_text_after_label(soup, "Content")
        
        # 如果找不到，尝试查找 "Course description"
        if not content:
            content = self.extract_text_after_label(soup, "Course description")
        
        return content or ""
    
    def extract_examiner(self, soup: BeautifulSoup) -> str:
        """提取考官姓名"""
        examiner = self.extract_text_after_label(soup, "Examiner")
        return examiner or "Unknown"
    
    def detect_language(self, soup: BeautifulSoup) -> str:
        """检测页面语言"""
        # 查找瑞典语关键词
        swedish_keywords = ["Behörighet", "Förutsättningar", "Kursplan", "Lärandemål"]
        page_text = soup.get_text()
        
        for keyword in swedish_keywords:
            if keyword in page_text:
                return "sv"
        
        return "en"
    
    def parse_course_page(self, html_content: str, course_code: str) -> Optional[Dict[str, Any]]:
        """
        解析课程页面，提取结构化数据
        
        Args:
            html_content: HTML 源码
            course_code: 课程代码
            
        Returns:
            结构化的课程数据字典
        """
        if not html_content:
            return None
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 检测语言
        detected_lang = self.detect_language(soup)
        
        # 提取课程标题
        title_elem = soup.find('h1')
        title = title_elem.get_text(strip=True) if title_elem else "Unknown"
        
        # 构建 URL
        encoded_year = self.academic_year.replace('/', '%2F')
        url = f"{self.base_url}/{course_code}/?acYear={encoded_year}"
        
        # 提取所有字段
        try:
            course_data = {
                "id": f"{course_code}_{self.academic_year.replace('/', '_')}",
                "course_code": course_code,
                "title": self.clean_text(title),
                "logistics": {
                    "block": self.extract_block_schedule(soup),
                    "sp": self.extract_study_period(soup),
                    "language": self.extract_teaching_language(soup),
                    "credits": self.extract_credits(soup)
                },
                "assessment": {
                    "exam_dates": self.extract_exam_dates(soup),
                    "grading_scale": self.extract_grading_scale(soup)
                },
                "constraints": {
                    "prerequisites": self.extract_prerequisites(soup),
                    "eligibility": self.extract_eligibility(soup),
                    "programs": self.extract_programs(soup)
                },
                "rag_text": {
                    "learning_outcomes": self.extract_learning_outcomes(soup),
                    "content": self.extract_content_summary(soup)
                },
                "metadata": {
                    "examiner": self.extract_examiner(soup),
                    "url": url,
                    "scraped_at": datetime.now().strftime("%Y-%m-%d"),
                    "detected_language": detected_lang
                }
            }
            
            return course_data
            
        except Exception as e:
            print(f"  ✗ 解析失败 {course_code}: {e}")
            return None
    
    def scrape_course(self, course_code: str) -> Optional[Dict[str, Any]]:
        """
        爬取单个课程的完整信息
        
        Args:
            course_code: 课程代码
            
        Returns:
            课程数据字典
        """
        html = self.fetch_course_page(course_code)
        if html:
            return self.parse_course_page(html, course_code)
        return None
    
    def scrape_course_with_fallback(self, course_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        爬取单个课程，如果主代码失败则尝试备选代码
        
        Args:
            course_data: 包含 primaryCourseCode 和 alternativeCodes 的字典
            
        Returns:
            课程数据字典
        """
        primary_code = course_data['primaryCourseCode']
        alternative_codes = course_data.get('alternativeCodes', [])
        all_codes = [primary_code] + alternative_codes
        
        for i, code in enumerate(all_codes):
            result = self.scrape_course(code)
            if result:
                # 添加原始课程信息到结果中
                result['courseName'] = course_data['courseName']
                result['allCourseCodes'] = course_data['allCodes']
                result['usedCourseCode'] = code
                result['isAlternativeCode'] = (i > 0)
                return result
        
        return None
    
    def scrape_multiple_courses(self, course_list: List[Dict[str, Any]], delay: float = 0.5) -> List[Dict[str, Any]]:
        """
        批量爬取多个课程（支持备选代码）
        
        Args:
            course_list: 课程数据列表（包含 primaryCourseCode 和 alternativeCodes）
            delay: 每次请求之间的延迟（秒），避免被封
            
        Returns:
            课程数据列表
        """
        results = []
        total = len(course_list)
        
        print(f"\n开始爬取 {total} 门课程（去重后）...")
        print("=" * 60)
        
        for i, course_data in enumerate(course_list, 1):
            course_name = course_data['courseName']
            primary_code = course_data['primaryCourseCode']
            alternatives = course_data.get('alternativeCodes', [])
            
            print(f"[{i}/{total}] {course_name} ({primary_code}", end="")
            if alternatives:
                print(f" +{len(alternatives)} 备选)", end=" ")
            else:
                print(")", end=" ")
            
            # 使用带备选的爬取方法
            result = self.scrape_course_with_fallback(course_data)
            if result:
                results.append(result)
                if result['isAlternativeCode']:
                    print(f"✓ (使用 {result['usedCourseCode']})")
                else:
                    print("✓")
            else:
                print("✗ (所有代码都失败)")
            
            # 延迟，避免过于频繁的请求
            if i < total:
                time.sleep(delay)
        
        print("=" * 60)
        print(f"✓ 成功爬取 {len(results)}/{total} 门课程")
        print(f"✓ 成功率: {len(results)/total*100:.2f}%")
        
        return results


def load_deduplicated_courses(json_file: str) -> List[Dict[str, Any]]:
    """
    从去重后的 JSON 文件中加载课程数据
    
    Args:
        json_file: JSON 文件路径
        
    Returns:
        课程数据列表（包含主代码和备选代码）
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# --- 主程序 ---
if __name__ == "__main__":
    print("=" * 60)
    print("Chalmers 课程大纲爬虫 (Syllabus Scraper)")
    print("用于构建 RAG 训练数据集")
    print("=" * 60)
    
    # 1. 加载去重后的课程数据
    print("\n[步骤 1] 加载去重后的课程数据...")
    deduplicated_file = "chalmers_courses_deduplicated.json"
    
    try:
        all_courses = load_deduplicated_courses(deduplicated_file)
        print(f"✓ 已加载 {len(all_courses)} 门去重后的课程")
        
        # 统计备选代码数量
        courses_with_alternatives = sum(1 for c in all_courses if c.get('alternativeCodes'))
        print(f"  其中 {courses_with_alternatives} 门课程有备选代码")
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        print(f"  请先运行: python deduplicate_courses.py")
        exit(1)
    
    # 2. 默认使用测试模式（前 10 门课程）
    print("\n[步骤 2] 测试模式：爬取前 10 门课程")
    target_courses = all_courses[:10]
    print(f"✓ 将爬取 {len(target_courses)} 门课程")
    print("\n  课程列表:")
    for i, course in enumerate(target_courses, 1):
        alt_count = len(course.get('alternativeCodes', []))
        alt_info = f" (+{alt_count} 备选)" if alt_count > 0 else ""
        print(f"    {i}. {course['courseName']} [{course['primaryCourseCode']}{alt_info}]")
    
    # 3. 创建爬虫实例并开始爬取
    print("\n[步骤 3] 开始爬取课程大纲...")
    scraper = ChalmersSyllabusScraper(academic_year="2025/2026")
    
    # 爬取课程数据（自动尝试备选代码）
    scraped_data = scraper.scrape_multiple_courses(target_courses, delay=0.5)
    
    # 4. 保存结果
    print("\n[步骤 4] 保存爬取结果...")
    output_file = "chalmers_syllabus_rag_dataset.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(scraped_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 数据已保存到: {output_file}")
    
    # 5. 输出统计信息
    print("\n" + "=" * 60)
    print("数据统计:")
    print(f"  总爬取数: {len(scraped_data)} / {len(target_courses)}")
    print(f"  成功率: {len(scraped_data) / len(target_courses) * 100:.2f}%")
    
    # 统计使用备选代码的情况
    used_alternative = sum(1 for c in scraped_data if c.get('isAlternativeCode', False))
    if used_alternative > 0:
        print(f"  使用备选代码: {used_alternative} 门课程")
    
    # 统计有学习成果的课程数
    if len(scraped_data) > 0:
        with_learning_outcomes = sum(1 for c in scraped_data if c['rag_text']['learning_outcomes'])
        print(f"  含学习成果: {with_learning_outcomes} ({with_learning_outcomes / len(scraped_data) * 100:.2f}%)")
    else:
        print(f"  含学习成果: 0 (0.00%)")
    
    # 统计语言分布
    languages = {}
    for course in scraped_data:
        lang = course['logistics']['language']
        languages[lang] = languages.get(lang, 0) + 1
    
    print(f"\n  授课语言分布:")
    for lang, count in sorted(languages.items(), key=lambda x: -x[1]):
        print(f"    {lang}: {count} 门")
    
    print("\n" + "=" * 60)
    print("✓ 爬取完成！数据已准备好用于 RAG 训练。")
    print("=" * 60)
