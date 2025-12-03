"""
使用 Selenium 爬取 https://stats.ftek.se/ 的课程代码
需要安装: pip install selenium
需要下载: ChromeDriver 或 Firefox GeckoDriver
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import re
import time

def get_course_codes_with_selenium(url="https://stats.ftek.se/"):
    """
    使用 Selenium 获取课程代码
    """
    print(f"正在使用 Selenium 访问: {url}")
    
    # 配置 Chrome 选项（无头模式）
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # 无界面模式
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    
    driver = None
    course_codes = set()
    
    try:
        # 初始化 Chrome 驱动
        driver = webdriver.Chrome(options=chrome_options)
        
        # 访问网页
        driver.get(url)
        
        # 等待页面加载（等待表格出现）
        print("等待页面加载...")
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "table"))
        )
        
        # 额外等待确保 JavaScript 执行完成
        time.sleep(3)
        
        # 获取页面源代码
        page_source = driver.page_source
        
        # 方法1: 从页面源代码中提取
        pattern = r'\b[A-Z]{3}\d{3}\b'
        codes_from_page = set(re.findall(pattern, page_source))
        print(f"从页面源代码提取到: {len(codes_from_page)} 个代码")
        
        # 方法2: 直接从表格元素中提取
        try:
            # 查找所有表格行
            rows = driver.find_elements(By.TAG_NAME, "tr")
            for row in rows:
                cells = row.find_elements(By.TAG_NAME, "td")
                if cells:
                    # 第一列通常是课程代码
                    first_cell = cells[0].text.strip()
                    if re.match(r'^[A-Z]{3}\d{3}$', first_cell):
                        course_codes.add(first_cell)
            
            print(f"从表格元素提取到: {len(course_codes)} 个代码")
        except Exception as e:
            print(f"从表格提取失败: {e}")
        
        # 合并两种方法的结果
        course_codes = course_codes | codes_from_page
        
        return sorted(list(course_codes))
        
    except Exception as e:
        print(f"发生错误: {e}")
        print("\n可能的问题:")
        print("  1. 未安装 ChromeDriver - 下载地址: https://chromedriver.chromium.org/")
        print("  2. ChromeDriver 版本与 Chrome 不匹配")
        print("  3. 使用 Firefox: 将 webdriver.Chrome 改为 webdriver.Firefox")
        return []
    
    finally:
        if driver:
            driver.quit()

if __name__ == "__main__":
    print("=" * 60)
    print("Selenium 版本 - Chalmers 课程代码爬虫")
    print("=" * 60)
    print("\n注意: 需要安装 selenium 和 ChromeDriver")
    print("安装命令: pip install selenium")
    print("=" * 60)
    
    try:
        course_codes = get_course_codes_with_selenium()
        
        print(f"\n共找到 {len(course_codes)} 个唯一课程代码")
        
        if course_codes:
            print("\n前 20 个课程代码:")
            for i, code in enumerate(course_codes[:20], 1):
                print(f"  {i:2d}. {code}")
            
            # 保存到文件
            output_file = "chalmers_course_codes.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                for code in course_codes:
                    f.write(code + '\n')
            print(f"\n✓ 已保存到文件: {output_file}")
            
            # 统计
            prefix_count = {}
            for code in course_codes:
                prefix = code[:3]
                prefix_count[prefix] = prefix_count.get(prefix, 0) + 1
            
            print(f"\n课程代码前缀统计 (Top 10):")
            for prefix, count in sorted(prefix_count.items(), key=lambda x: -x[1])[:10]:
                print(f"  {prefix}: {count} 门课程")
        else:
            print("\n未能获取课程代码")
    
    except ImportError:
        print("\n✗ 错误: 未安装 selenium")
        print("请运行: pip install selenium")
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
