#!/usr/bin/env python3
"""
Generate conversation-style images for test cases
"""
from PIL import Image, ImageDraw, ImageFont
import textwrap

def create_chat_bubble_image(user_question, system_response, case_number, case_title, output_path):
    """Create a chat-style image with user question and system response"""
    
    # Image settings
    width = 1200
    padding = 40
    line_spacing = 8
    bubble_padding = 20
    
    # Colors
    bg_color = (245, 245, 250)  # Light gray background
    user_bubble_color = (66, 133, 244)  # Blue
    system_bubble_color = (255, 255, 255)  # White
    user_text_color = (255, 255, 255)  # White text
    system_text_color = (33, 33, 33)  # Dark text
    border_color = (220, 220, 220)  # Light border
    header_color = (51, 51, 51)
    
    # Try to load fonts, fallback to default if not available
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        question_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        response_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        # Fallback to default font
        title_font = ImageFont.load_default()
        question_font = ImageFont.load_default()
        response_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    
    # Create a temporary image to measure text
    temp_img = Image.new('RGB', (width, 100))
    temp_draw = ImageDraw.Draw(temp_img)
    
    # Wrap text
    max_text_width = width - 2 * padding - 2 * bubble_padding - 100
    
    # Wrap user question
    user_lines = textwrap.wrap(user_question, width=int(max_text_width / 12))
    
    # Wrap system response (preserve bullet points)
    response_lines = []
    for line in system_response.split('\n'):
        if line.strip():
            wrapped = textwrap.wrap(line, width=int(max_text_width / 10))
            response_lines.extend(wrapped)
        else:
            response_lines.append('')
    
    # Calculate heights (removed label heights for cleaner look)
    header_height = 80
    user_bubble_height = len(user_lines) * 30 + 2 * bubble_padding
    system_bubble_height = len(response_lines) * 28 + 2 * bubble_padding
    total_height = header_height + user_bubble_height + system_bubble_height + padding * 3
    
    # Create image
    img = Image.new('RGB', (width, total_height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Draw header
    draw.text((padding, padding), f"Test Case {case_number}: {case_title}", 
              fill=header_color, font=title_font)
    
    current_y = header_height + padding
    
    # Draw user bubble (right-aligned, blue)
    user_bubble_width = width - 2 * padding - 100
    user_bubble_x = padding + 100
    draw.rounded_rectangle(
        [(user_bubble_x, current_y), 
         (user_bubble_x + user_bubble_width, current_y + user_bubble_height)],
        radius=15, fill=user_bubble_color
    )
    
    # Draw user question text (no label for natural conversation)
    text_y = current_y + bubble_padding
    for line in user_lines:
        draw.text((user_bubble_x + bubble_padding, text_y), 
                  line, fill=user_text_color, font=question_font)
        text_y += 30
    
    current_y += user_bubble_height + padding
    
    # Draw system bubble (left-aligned, white with border)
    system_bubble_width = width - 2 * padding - 100
    system_bubble_x = padding
    draw.rounded_rectangle(
        [(system_bubble_x, current_y), 
         (system_bubble_x + system_bubble_width, current_y + system_bubble_height)],
        radius=15, fill=system_bubble_color, outline=border_color, width=2
    )
    
    # Draw system response text (no label for natural conversation)
    text_y = current_y + bubble_padding
    for line in response_lines:
        draw.text((system_bubble_x + bubble_padding, text_y), 
                  line, fill=system_text_color, font=response_font)
        text_y += 28
    
    # Save image
    img.save(output_path, quality=95)
    print(f"✅ Created: {output_path}")


def main():
    """Generate all test case images"""
    
    # Test Case 1: AI/ML Course Recommendation
    create_chat_bubble_image(
        user_question="I want to specialize in machine learning and deep learning. What courses should I take and in what order?",
        system_response="""✅ Course Recommendations:

• DAT340 (Applied Machine Learning) - Foundational course
  Covers ML fundamentals, classical algorithms
  
• TDA234 (Deep Learning) - Advanced follow-up
  Focuses on neural networks and deep learning techniques
  
• DAT565 (Data Science and AI) - Broader AI concepts
  Comprehensive AI and data science coverage

📋 Prerequisites: 
  DAT340 requires programming and statistics background

🔗 Full course URLs and details provided""",
        case_number=1,
        case_title="AI/ML Course Recommendation",
        output_path="test_case_1_ai_ml.png"
    )
    
    # Test Case 2: Schedule Conflict Detection
    create_chat_bubble_image(
        user_question="Can I take DAT565 (Data Science and AI) and MVE137 (Probability and Statistics) in the same period? Check their schedule blocks.",
        system_response="""📅 Schedule Analysis:

• DAT565: Block C (7.5 credits, English)
• MVE137: Block A (15 credits, full semester)

⚠️ Conclusion: POTENTIAL CONFLICT!
MVE137 runs the entire semester and overlaps with Block C

📋 Additional Info:
  DAT565 prerequisites: Programming experience required

💡 Recommendation: Consider taking these courses in different semesters""",
        case_number=2,
        case_title="Schedule Conflict Detection",
        output_path="test_case_2_schedule.png"
    )
    
    # Test Case 3: Course Comparison
    create_chat_bubble_image(
        user_question="Compare DAT340 (Applied Machine Learning) and TDA234 (Deep Learning). Which one focuses more on hands-on projects?",
        system_response="""📊 Course Comparison:

DAT340 (Applied Machine Learning):
• Balanced approach with practical assignments
• Covers classical ML algorithms (SVM, decision trees, etc.)
• Assessment: Project + Exam
• Focus: Broader ML applications

TDA234 (Deep Learning):
• More theory-heavy approach
• Focuses on neural network architectures
• Assessment: Primarily exam-based with coding assignments
• Focus: Specialized deep learning techniques

✅ Conclusion: DAT340 offers more hands-on project experience""",
        case_number=3,
        case_title="Course Comparison & Analysis",
        output_path="test_case_3_comparison.png"
    )
    
    # Test Case 4: Learning Path Planning
    create_chat_bubble_image(
        user_question="I'm interested in becoming a data scientist and want to work on real-world data projects. Recommend me a course sequence.",
        system_response="""🎓 3-Phase Learning Path:

Phase 1 - Foundation:
• MVE137 (Probability and Statistics)
• TDA540 (Programming fundamentals)

Phase 2 - Core Skills (Choose):
• DAT340 (Applied Machine Learning)
• DAT565 (Data Science and AI)
• MVE441 (Statistical Learning for Big Data)

Phase 3 - Specialization (Choose):
• TDA234 (Deep Learning)
• DAT450 (Data Mining and Visualization)
• DAT290 (15 credits!) - Complete data science project

⚠️ Note: DAT340 and MVE441 both in Block C - cannot take together""",
        case_number=4,
        case_title="Data Science Learning Path",
        output_path="test_case_4_learning_path.png"
    )
    
    # Test Case 5: Honest Limitation
    create_chat_bubble_image(
        user_question="What is TDA567 (Testing and Debugging) about? Tell me the learning outcomes, prerequisites, and assessment methods.",
        system_response="""⚠️ Limited Information Available:

I don't have enough detailed information about TDA567's specific 
learning outcomes in the provided context.

✅ What I can confirm:
• Course is about software testing and debugging techniques
• Part of software engineering curriculum

🔗 For complete details:
Please check the official Chalmers University website for:
  - Detailed learning outcomes
  - Specific prerequisites
  - Assessment methods
  - Course schedule

Note: System honestly admits lack of information instead of 
hallucinating details - this builds trust!""",
        case_number=5,
        case_title="Honest Limitation Handling",
        output_path="test_case_5_limitation.png"
    )
    
    print("\n✨ All test case images generated successfully!")
    print("\nGenerated files:")
    print("  • test_case_1_ai_ml.png")
    print("  • test_case_2_schedule.png")
    print("  • test_case_3_comparison.png")
    print("  • test_case_4_learning_path.png")
    print("  • test_case_5_limitation.png")


if __name__ == "__main__":
    main()
