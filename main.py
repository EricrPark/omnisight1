import streamlit as st
import os
from openai import OpenAI
from docx import Document
from PyPDF2 import PdfReader
import spacy
import io
import re
import json
import uuid
from datetime import datetime, timezone
import glob

# System prompts for resume evaluation agents
PRIMARY_EVALUATOR_PROMPT = '''You are a detailed, structured resume reviewer.

Evaluate the candidate based on the parsed resume JSON. Assess them using the following categories:

1. **Believability** — Are the listed roles, timelines, and skills plausible given age and background?
2. **Role Depth & Function** — Did the candidate do real work (e.g. SWE vs warehouse)? Ignore fluff.
  - For upcoming internships (e.g., "Incoming Intern at X"), the candidate has not yet started the role. Evaluate based on company and expected role alone, as detailed responsibilities are unlikely to be included.
3. **Pedigree (Contextualized)** — Are the schools or firms impressive given the specific role (e.g. Harvard certificate ≠ Harvard degree)?
4. **Impact & Specificity** — Do bullets show ownership, measurable results, or vague participation?
5. **Writing & Communication** — Is the resume coherent, well-structured, and sharp?
6. **Consistency** — Do listed skills match experiences? Are roles and trajectory aligned?
7. **Trajectory** — Are they leveling up? Increasing complexity, responsibility, or firm quality over time?
8. **Recommended Role Fit** — Suggest 1–2 ideal job types based on their background.

  - Ignore summary sections unless they contain significant new information not already found in education or experience.

  Use the following scale for all ratings:
- 1–3: Very weak — clear issues, low relevance or credibility
- 4–5: Below average — limited signal or serious gaps
- 6–7: Solid — typical for a reasonable candidate
- 8–9: Strong — clearly above average with strong supporting evidence
- 10: Exceptional — rare quality, top 1% signal in this dimension

Detailed guidance per category:

**Believability (1–10):**
- 1–3: Resume feels inflated or implausible; red flags around timelines or claims.
- 4–5: Some questionable experiences or resume "padding."
- 6–7: Believable for someone at this stage; no major concerns.
- 8–9: Well-supported timeline, logical progression, and credible claims.
- 10: Everything fits perfectly with no fluff; impeccable consistency.

**Role Depth & Function (1–10):**
- 1–3: Fluffy roles with unclear responsibilities (e.g. "interned at startup" with no substance).
- 4–5: Some work done, but hard to assess real contributions.
- 6–7: Describes real responsibilities aligned with title.
- 8–9: Clear ownership of real work, often with outcomes.
- 10: Demonstrated impact or leadership well beyond role expectations.

**Pedigree (Contextualized) (1–10):**
- 1–3: No signal from school or firms.
- 4–5: Mid-tier school or unknown companies.
- 6–7: Strong university or decent firm in a relevant role.
- 8–9: Top-tier school and/or name-brand firms in relevant roles.
- 10: Multiple elite institutions and top firms (e.g. Stanford + Goldman + Meta SWE).

**Impact & Specificity (1–10):**
- 1–3: Vague bullets with no action or result.
- 4–5: Some action verbs, but little detail or metrics.
- 6–7: Specific contributions with some results or insight.
- 8–9: Bullets that reflect real impact with data or complexity.
- 10: Clear, quantified results that show outstanding impact.

**Writing & Communication (1–10):**
- 1–3: Hard to follow, inconsistent formatting, grammar errors.
- 4–5: Some structure, but unclear or clunky.
- 6–7: Clear and competent writing.
- 8–9: Clean, sharp phrasing throughout.
- 10: Elite clarity, polish, and professionalism.

**Consistency (1–10):**
- 1–3: Skills, roles, and claims don't line up at all.
- 4–5: Some contradictions or resume feels patched together.
- 6–7: Reasonably aligned story and skill use.
- 8–9: Well-integrated experiences and skills that reinforce each other.
- 10: Extremely coherent and consistent across the entire resume.

**Trajectory (1–10):**
- 1–3: Flat or declining; no signs of growth.
- 4–5: Some movement, but slow or unclear.
- 6–7: Standard upward progression.
- 8–9: Accelerated growth, stronger firms, or increased scope over time.
- 10: Steep trajectory with clear signals of rapid development.

**Recommended Role Fit:**
Base this on demonstrated skills, prior roles, and communication—not just company names.

Respond in this format:

### Resume Evaluation
**Believability (1–10):**  
**Role Depth & Function (1–10):**  
**Pedigree (Contextualized) (1–10):**  
**Impact & Specificity (1–10):**  
**Writing & Communication (1–10):**  
**Consistency (1–10):**  
**Trajectory (1–10):**  
**Recommended Role Types:** [list of 1–2 roles]

### Summary
Write 2–3 sentences explaining your overall opinion of this candidate based on the resume.'''

SKEPTIC_PROMPT = '''You are a resume red teamer.

Your job is to review a candidate's parsed resume AND the primary evaluation output to identify anything that seems suspicious, exaggerated, vague, or implausible.
Incoming internship offers from prestigious firms (e.g., ‘Incoming Summer Investment Associate at Bridgewater’) are common among high-performing students and do not require prior experience at that firm. Do not question how the offer was secured unless there is a direct contradiction or implausibility elsewhere in the resume.
  - For incoming internships (e.g., "Incoming Intern at X"), the candidate has not yet started the role. Evaluate based on company and expected role alone - not that it is skeptical to not yet be working, as detailed responsibilities are unlikely to be included. 
  - It is very normal to list an incoming role and this alone is not suspicious. Avoid questioning role details, expectations, or workload balance for future positions.
  - Do not confuse internship titles such as ‘Incoming Summer Associate’ or ‘Incoming Summer Investment Analyst’ with full-time roles like ‘Associate.’ These are standard internship titles for students and should not be interpreted as signs of unusually fast progression or inflated seniority.
  - Incoming roles often have general descriptions (e.g., ‘Investment research at X’). This is expected and should not be flagged as vague or inflated unless the candidate claims specific achievements in a role that hasn’t started.
  - Do not flag overlapping dates for future roles, including internships and full-time positions. Students often list the date they received an offer rather than the actual start date, which can cause apparent overlap. This is common and should not be treated as a red flag unless multiple full-time roles overlap in a way that is truly implausible or contradictory.

Return the following:

### Skepticism Score (1–10)
- 1–3: Very suspicious; likely exaggerated or dishonest
- 4–6: Mixed; some questionable elements
- 7–8: Generally believable with some minor stretch
- 9–10: Highly believable and well supported

### Summary
Write a 1–2 sentence summary explaining your skepticism score.

### Red Flags
List 2–4 resume elements that seem suspicious. Be specific and explain why.
- Bullet points that lack context or results
- Titles too senior for candidate's age
- Skills listed but never used
- Timing/experience gaps'''

SYNTHESIZER_PROMPT = '''You are a hiring manager making a final judgment.

You've received the primary resume evaluation and the skeptic's critique. Your task is to summarize both perspectives and produce:
- Do not penalize candidates for omitting a summary section unless the resume is otherwise sparse. Most real-world evaluators do not prioritize this section.
- Do not question progression from internship to incoming role at the same firm unless there are clear contradictions. This is a common pattern and not inherently suspicious.

### Final Resume Score (20–80)
Use this scale:
- 20–30: Weak; little credible or relevant experience
- 31–50: Below average or inconsistent
- 51–65: Solid; some strengths but room for doubt or growth
- 66–75: Very strong; high confidence in capabilities
- 76–80: Exceptional; rare clarity, credibility, and fit

### Final Summary
Write a short paragraph summarizing the candidate's perceived quality based on the full picture.

### Follow-Up Questions
List 1–3 probing questions you would ask the candidate in an interview to clarify doubts or explore interesting claims.'''

OVERALL_ASSESSMENT_PROMPT = '''You are a senior hiring manager making a final, comprehensive assessment of a candidate.

You have access to:
1. The candidate's resume evaluation (including primary evaluator, skeptic, and synthesizer outputs)
2. Their reasoning assessment results
3. Their complete candidate profile

Your task is to synthesize all this information into a final, holistic assessment that prioritizes the resume evaluation and reasoning assessment, while considering the full context.

Provide your assessment in this format:

### Overall Candidate Score (20–80)
Weight the resume evaluation at 60% and the reasoning assessment at 40% in determining this score.

### Key Strengths
List 2-3 major strengths demonstrated across both the resume and reasoning assessment.

### Areas for Development
List 2-3 areas where the candidate could improve or develop further.

### Role Recommendations
Based on the complete picture, suggest 1-2 specific roles where this candidate would excel.

### Interview Focus Areas
List 3-4 key areas to explore in an interview, including:
- Any inconsistencies or gaps in the resume
- Specific reasoning responses that warrant deeper discussion
- Areas where more context is needed

### Final Recommendation
Provide a clear, concise recommendation on whether to proceed with this candidate, including any specific concerns or reservations.'''

# Create candidates directory if it doesn't exist
CANDIDATES_DIR = "candidates"
os.makedirs(CANDIDATES_DIR, exist_ok=True)

def save_candidate_to_file(candidate_data):
    """Save candidate data to a JSON file."""
    # Generate a UUID for the candidate if not provided
    if "candidate_id" not in candidate_data:
        candidate_data["candidate_id"] = str(uuid.uuid4())
    
    # Add timestamp if not provided
    if "timestamp" not in candidate_data:
        candidate_data["timestamp"] = datetime.now(timezone.utc).isoformat()
    
    # Set default reason if not provided
    if "reason" not in candidate_data:
        candidate_data["reason"] = "TESTING"
    
    # Ensure test_type is provided
    if "test_type" not in candidate_data:
        candidate_data["test_type"] = "reasoning"
    
    # Create the file path
    file_path = os.path.join(CANDIDATES_DIR, f"{candidate_data['candidate_id']}.json")
    
    # Write to file
    with open(file_path, 'w') as f:
        json.dump(candidate_data, f, indent=2)
    
    return candidate_data["candidate_id"]

def load_candidate_data(candidate_id):
    """Load candidate data from a JSON file."""
    file_path = os.path.join(CANDIDATES_DIR, f"{candidate_id}.json")
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def get_all_candidates():
    """Get a list of all candidate files."""
    return glob.glob(os.path.join(CANDIDATES_DIR, "*.json"))

def format_candidate_display_name(candidate_data):
    """Format a display name for the candidate in dropdowns."""
    timestamp = datetime.fromisoformat(candidate_data["timestamp"])
    formatted_time = timestamp.strftime("%Y-%m-%d %H:%M")
    return f"{candidate_data['candidate_id'][:8]}... ({formatted_time})"

# Define the reasoning questions
reasoning_questions = [
    {
        "id": "ped_testing",
        "text": """You are designing a PED testing strategy for the Olympic Games. You have access to a 100%-accurate drug test, but due to budget constraints, you can only test 30% of athletes. 

Design a strategy to maximize the probability of detecting PED users. Be specific: what data would you use, what criteria would drive your selection, and what are potential drawbacks to your strategy?

Assume this strategy will be implemented exactly as described—do not rely on follow-up clarification or future adjustments. This is not about what system the Olympics should implement in practice—it is purely about designing the system that would catch the most PED users."""
    },
    {
        "id": "iphone_rebuild",
        "text": """The entire modern human population is suddenly transported 10,000 years into the past. Everyone retains their memories, knowledge, and skills—but no modern tools, infrastructure, or devices make the trip.

Assume that over time, humanity begins rebuilding civilization. Your task is to estimate how long it would take for someone to build a fully functioning iPhone from scratch.

Consider the major scientific and technological milestones required, what resources would need to be discovered and refined, and what steps would be essential before manufacturing could even begin. Be realistic and specific—focus on bottlenecks, necessary prerequisites, and potential acceleration strategies."""
    }
]

def generate_combined_evaluation(responses, evaluations):
    """Generate a synthesized evaluation of the candidate based on all responses."""
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return "Error: OPENAI_API_KEY environment variable not found."

    system_prompt = """You are analyzing a candidate's responses to multiple reasoning questions to create a comprehensive profile.
Your task is to synthesize insights across all answers to identify patterns, strengths, and areas for improvement.

Consider:
1. Overall reasoning style and approach
2. Consistency in thinking across different scenarios
3. Key strengths demonstrated (if any)
4. Areas where thinking could be improved (if any)
5. Potential implications for real-world problem-solving
6. Note any skipped questions and their impact on the overall assessment

Format your response as follows:
    
### Final Score (20–80)
Provide a single numerical score between 20 and 80 based on your holistic evaluation of the candidate's reasoning skills across all responses. Use the following rough guide:
- 20–30: Weak reasoning; unclear, inconsistent, or superficial thinking.
- 31–40: Below average; some logic present but lacks clarity, originality, or specificity.
- 41–50: Average; competent but unremarkable reasoning, may lack depth or structure.
- 51–60: Strong reasoning; clear, structured, and generally thoughtful responses.
- 61–70: Very strong reasoning; insightful, well-structured, and original thinking.
- 71–80: Exceptional reasoning; rare clarity, depth, and creativity in responses.

Note: For skipped questions, adjust the score range downward by 5-10 points depending on the number of skips and their importance to the overall assessment.

### Score Rationale
Explain why you assigned this score using examples from the candidate's responses. Identify any consistent patterns in reasoning quality, strengths, and weaknesses. Note any skipped questions and how they affected your evaluation.

Be specific and evidence-based, referencing particular aspects of the candidate's responses to support your analysis.
If the candidate skipped questions, explain how this impacts your ability to fully assess their reasoning capabilities."""

    # Format the responses and evaluations for GPT
    context = "Here are the candidate's responses and evaluations:\n\n"
    for q_id, response in responses.items():
        question = next(q for q in reasoning_questions if q["id"] == q_id)
        context += f"Question: {question['text']}\n"
        context += f"Response: {response}\n"
        context += f"Evaluation: {evaluations[q_id]}\n\n"

    try:
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context}
            ],
            temperature=0.3,
            max_tokens=800  # Reduced from 1000 to encourage more concise responses
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating combined evaluation: {str(e)}"

# If you encounter "Client.init() got an unexpected keyword argument 'proxies'",
# ensure you have the latest version of the openai library installed:
# pip install --upgrade openai

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file with basic formatting preservation."""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def extract_text_from_docx(docx_file):
    """Extract text from DOCX file with basic formatting preservation."""
    doc = Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            text += paragraph.text.strip() + "\n"
    return text.strip()

def parse_resume_with_gpt(text):
    """Use GPT to parse resume text into structured sections."""
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return "Error: OPENAI_API_KEY environment variable not found."

    system_prompt = """You are a resume parser. Your task is to analyze the resume text and extract structured information.
Please identify and organize the following information:

1. Contact Information:
   - Full name
   - Email address
   - Phone number
   - Location (city, state)

2. Professional Summary:
   - Look for a brief summary of the candidate's background and objectives
   - This may be labeled as "Summary", "Professional Summary", "Objective", "Career Objective", or similar
   - If no explicit summary section exists, use the first paragraph if it appears to be a summary
   - If no suitable summary is found, leave this field empty

3. Education:
   - List each educational institution with:
     * Institution name
     * Location (city, state)
     * Degree and major(s) (include all majors and minors if multiple)
     * Expected graduation date (ONLY include the end date, not the start date or "Present". If no end date is found, leave this field empty)
     * GPA (if available)
     * Test scores (if available)
     * Relevant coursework (if available)
     * Honors and awards (if available)

4. Professional Experience:
   - List each job with:
     * Company name
     * Location (city, state)
     * Position title
     * Dates
     * Key responsibilities as bullet points
     - IMPORTANT: Each bullet point must be clearly associated with its specific experience
     - Do not mix or combine bullet points from different experiences
     - If a bullet point's context is unclear, do not include it

5. Projects:
   - List any significant projects with:
     * Project name
     * Description
     * Technologies used
     * Key achievements or outcomes

6. Extracurricular Activities:
   - List any leadership roles or significant involvement in:
     * Clubs
     * Organizations
     * Student government
     * Other activities
   - Include role, dates, and key responsibilities/achievements

7. Sports & Athletics:
   - List any sports participation or athletic achievements
   - Include level of participation, achievements, and dates

8. Skills & Additional Information:
   - List technical skills, tools, and technologies
   - Group related skills together
   - Include interests and hobbies if listed
   - Preserve the original section labels from the resume
   - Make sure bullet points are correctly attributed to their respective experiences

IMPORTANT: Your response must be a valid JSON object. Do not include any text before or after the JSON object.
Do not use markdown code blocks or any other formatting. Just output the raw JSON.

Example format:
{
    "contact_info": {
        "name": "John Doe",
        "email": "john@example.com",
        "phone": "123-456-7890",
        "location": "New York, NY"
    },
    "summary": "Experienced software engineer...",
    "education": [
        {
            "institution": "University of Example",
            "location": "Boston, MA",
            "degree": "Bachelor of Science",
            "major": "Computer Science",
            "graduation_date": "2024",
            "gpa": "3.8",
            "test_scores": "SAT: 1500",
            "relevant_coursework": ["Data Structures", "Algorithms"],
            "honors": ["Dean's List"]
        }
    ],
    "experience": [
        {
            "company": "Tech Corp",
            "location": "San Francisco, CA",
            "position": "Software Engineer",
            "dates": "2020-2023",
            "responsibilities": ["Developed web applications"]
        }
    ],
    "projects": [
        {
            "name": "Project X",
            "description": "Description here",
            "technologies": ["Python", "React"],
            "achievements": ["Achievement 1"]
        }
    ],
    "extracurriculars": [
        {
            "organization": "Club Name",
            "role": "President",
            "dates": "2022-2023",
            "achievements": ["Achievement 1"]
        }
    ],
    "sports": [
        {
            "activity": "Soccer",
            "level": "Varsity",
            "dates": "2020-2023",
            "achievements": ["Team Captain"]
        }
    ],
    "skills": [
        {
            "category": "Programming Languages",
            "items": ["Python", "Java"]
        },
        {
            "category": "Interests",
            "items": ["Photography", "Travel"]
        }
    ]
}"""

    try:
        # Initialize client with minimal configuration
        client = OpenAI(
            api_key=api_key,
            http_client=None
        )
        
        # Create a debug section that's collapsed by default
        with st.expander("Debug Information", expanded=False):
            # Debug: Show the text being sent to GPT
            st.write("Text being sent to GPT:")
            st.text(text[:10000] + "..." if len(text) > 10000 else text)
            
            # Debug: Show the messages being sent
            st.write("Messages being sent to GPT:")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ]
            st.json(messages)
            
            st.write("Making API call...")
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.1,
                max_tokens=2000  # Increased token limit
            )
            
            st.write("API call completed")
            
            # Get the response content
            response_content = completion.choices[0].message.content.strip()
            
            # Debug: Show raw GPT response
            st.write("GPT Response:")
            st.text(response_content)
            
            # Try to parse the JSON response
            try:
                st.write("Attempting to parse JSON response...")
                
                # Clean the response content
                cleaned_content = response_content
                # Remove any markdown code block markers
                cleaned_content = cleaned_content.replace("```json", "").replace("```", "").strip()
                
                # Try to find JSON content between curly braces
                json_match = re.search(r'\{[\s\S]*\}', cleaned_content)
                if json_match:
                    cleaned_content = json_match.group(0)
                
                # Clean up any trailing commas
                cleaned_content = re.sub(r',(\s*[}\]])', r'\1', cleaned_content)
                
                # Only show cleaned content if it's different from original
                if cleaned_content != response_content:
                    st.write("Cleaned JSON content (changes detected):")
                    st.text(cleaned_content)
                
                parsed_data = json.loads(cleaned_content)
                st.write("JSON parsing successful")
                return parsed_data
            except json.JSONDecodeError as e:
                st.error(f"JSON Parsing Error: {str(e)}")
                st.error("Raw response that failed to parse:")
                st.text(response_content)
                st.error("Error position:")
                st.text(f"Line {e.lineno}, Column {e.pos}")
                # Try to fix common JSON issues
                try:
                    # Remove any trailing commas
                    cleaned_content = re.sub(r',(\s*[}\]])', r'\1', cleaned_content)
                    # Ensure all quotes are properly escaped
                    cleaned_content = cleaned_content.replace('"', '\\"').replace('\\"', '"')
                    # Try parsing again
                    if cleaned_content != response_content:
                        st.write("Cleaned JSON content after fixes (changes detected):")
                        st.text(cleaned_content)
                    parsed_data = json.loads(cleaned_content)
                    st.write("JSON parsing successful after fixes")
                    return parsed_data
                except json.JSONDecodeError as e2:
                    st.error("Failed to parse even after fixes")
                    return None

    except Exception as e:
        with st.expander("Error Details", expanded=False):
            st.error(f"GPT API Error: {str(e)}")
            st.error("Full error details:")
            st.exception(e)
        return None

def parse_resume(text):
    """Parse resume text into structured sections using GPT."""
    parsed_data = parse_resume_with_gpt(text)
    if parsed_data is None:
        st.error("Failed to parse resume with GPT")
        return None
    return parsed_data

def format_resume(parsed_data):
    """Format parsed resume data into markdown."""
    if parsed_data is None:
        return "Error: Failed to parse resume"
        
    output = "### Resume Summary\n\n"
    
    # Contact Information
    contact_info = parsed_data.get("contact_info", {})
    if any(contact_info.values()):
        output += "#### Contact Information\n"
        if contact_info.get("name"):
            output += f"- **Name:** {contact_info['name']}\n"
        if contact_info.get("email"):
            output += f"- **Email:** {contact_info['email']}\n"
        if contact_info.get("phone"):
            output += f"- **Phone:** {contact_info['phone']}\n"
        if contact_info.get("location"):
            output += f"- **Location:** {contact_info['location']}\n"
        output += "\n"
    
    # Summary
    if parsed_data.get("summary"):
        output += "#### Professional Summary\n"
        output += parsed_data["summary"] + "\n\n"
    
    # Education
    if parsed_data.get("education"):
        output += "#### Education\n"
        for edu in parsed_data["education"]:
            output += f"**{edu.get('institution', '')}**\n"
            if edu.get('location'):
                output += f"- Location: {edu['location']}\n"
            
            # Handle degree and majors/minors
            degree = edu.get('degree', '')
            major = edu.get('major', '')
            minor = edu.get('minor', '')
            additional_majors = edu.get('additional_majors', [])
            additional_minors = edu.get('additional_minors', [])
            
            if degree:
                output += f"- {degree}"
                if major:
                    output += f" in {major}"
                if minor:
                    output += f" with a minor in {minor}"
                if additional_majors:
                    output += f" and {', '.join(additional_majors)}"
                if additional_minors:
                    output += f" with additional minors in {', '.join(additional_minors)}"
                output += "\n"
            
            if edu.get('graduation_date'):
                # Only show graduation date if it's not "Present" or a date range
                grad_date = edu['graduation_date']
                if grad_date.lower() != "present" and "-" not in grad_date:
                    output += f"- Expected Graduation: {grad_date}\n"
            if edu.get('gpa'):
                output += f"- GPA: {edu['gpa']}\n"
            if edu.get('test_scores'):
                output += f"- Test Scores: {edu['test_scores']}\n"
            if edu.get('relevant_coursework'):
                output += "- Relevant Coursework:\n"
                for course in edu['relevant_coursework']:
                    output += f"  * {course}\n"
            if edu.get('honors'):
                output += "- Honors & Awards:\n"
                for honor in edu['honors']:
                    output += f"  * {honor}\n"
            output += "\n"
    
    # Experience
    if parsed_data.get("experience"):
        output += "#### Professional Experience\n"
        for exp in parsed_data["experience"]:
            output += f"**{exp.get('company', '')}**\n"
            if exp.get('location'):
                output += f"- Location: {exp['location']}\n"
            if exp.get('position'):
                output += f"- Position: {exp['position']}\n"
            if exp.get('dates'):
                output += f"- Dates: {exp['dates']}\n"
            if exp.get('responsibilities'):
                output += "- Key Responsibilities:\n"
                for resp in exp['responsibilities']:
                    output += f"  * {resp}\n"
            output += "\n"
    
    # Projects
    if parsed_data.get("projects"):
        output += "#### Projects\n"
        for proj in parsed_data["projects"]:
            output += f"**{proj.get('name', '')}**\n"
            if proj.get('description'):
                output += f"- {proj['description']}\n"
            if proj.get('technologies'):
                output += "- Technologies:\n"
                for tech in proj['technologies']:
                    output += f"  * {tech}\n"
            if proj.get('achievements'):
                output += "- Key Achievements:\n"
                for achievement in proj['achievements']:
                    output += f"  * {achievement}\n"
            output += "\n"
    
    # Extracurriculars
    if parsed_data.get("extracurriculars"):
        output += "#### Extracurricular Activities\n"
        for extra in parsed_data["extracurriculars"]:
            output += f"**{extra.get('organization', '')}**\n"
            if extra.get('role'):
                output += f"- Role: {extra['role']}\n"
            if extra.get('dates'):
                output += f"- Dates: {extra['dates']}\n"
            if extra.get('achievements'):
                output += "- Key Achievements:\n"
                for achievement in extra['achievements']:
                    output += f"  * {achievement}\n"
            output += "\n"
    
    # Sports
    if parsed_data.get("sports"):
        output += "#### Sports & Athletics\n"
        for sport in parsed_data["sports"]:
            output += f"**{sport.get('activity', '')}**\n"
            if sport.get('level'):
                output += f"- Level: {sport['level']}\n"
            if sport.get('dates'):
                output += f"- Dates: {sport['dates']}\n"
            if sport.get('achievements'):
                output += "- Achievements:\n"
                for achievement in sport['achievements']:
                    output += f"  * {achievement}\n"
            output += "\n"
    
    # Skills & Additional Information
    if parsed_data.get("skills"):
        output += "#### Skills & Additional Information\n"
        for skill_group in parsed_data["skills"]:
            if skill_group.get('category'):
                output += f"**{skill_group['category']}**\n"
                for item in skill_group.get('items', []):
                    output += f"- {item}\n"
                output += "\n"
    
    return output

def get_completion_evaluation(response):
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return "Error: OPENAI_API_KEY environment variable not found."

    system_prompt = """You are evaluating a job candidate's response to a high-level reasoning prompt. 
Your job is to assess the quality of their thinking using the rubric below.

Score the answer from 0–10 in each of the following categories:
1. **Clarity** — Is the response clearly written, well-structured, and easy to follow?
2. **Logical Reasoning** — Is the argument internally consistent, and are the assumptions coherent?
3. **Originality** — Does the response show creativity, non-obvious ideas, or unique perspectives?
4. **Specificity and Realism of Strategy** — Does the candidate present a specific, implementable, and realistic plan given the scenario constraints?

Use this exact format for output:
Clarity: [score]
Logical reasoning: [score]
Originality: [score]
Specificity and realism of strategy: [score]

Feedback: [short paragraph, 2–4 sentences]

Be honest and specific—do not inflate scores. A 7 or 8 reflects strong thinking. A 10 should be rare and exceptional. Penalize responses that:
- Avoid answering the question directly
- Make claims without any reasoning or justification
- Focus on tangents like fairness, emotional appeal, or vague opinions instead of directly addressing the objective in the prompt
Reward answers that support their approach with logic, data proxies, or clear prioritization.

Here are example answers to the following prompt used in the Olympic Games context:
"You are designing a PED testing strategy for the Olympic Games. You have access to a 100%-accurate drug test, but due to budget constraints, you can only test 30% of athletes. Design a strategy to maximize the probability of detecting PED users.""

--- Good Answer ---
"I would prioritize testing athletes with statistically abnormal improvements in performance over time, especially in sports with high historical PED usage. Additionally, I would create a model based on risk indicators like training location, previous suspicions, or affiliations with known violators. This approach focuses resources where the probability of catching a cheater is highest."

Clarity: 9
Logical reasoning: 9
Originality: 8
Specificity and realism of strategy: 7

--- Mediocre Answer ---
"I would focus on top performers and some random athletes from high-risk sports. This would probably catch a few cheaters."

Clarity: 6
Logical reasoning: 4
Originality: 3
Specificity and realism of strategy: 3

--- Poor Answer ---
"I would randomly test athletes because that's the fairest way to do it. Everyone should have the same chance of being tested."

Clarity: 2
Logical reasoning: 1
Originality: 2
Specificity and realism of strategy: 1

--- Insightful but Unstructured Answer ---
"I think people often cheat when there's high financial or national pressure. So, I'd look at the countries with the most to gain—those who win disproportionately or host events. Also, I'd scan for outliers in bio-passport data and prioritize those with unexplained anomalies."

Clarity: 5
Logical reasoning: 7
Originality: 8
Specificity and realism of strategy: 5

--- Jargon-Heavy but Underdeveloped Answer ---
"I would apply a Bayesian decision network to athlete training logs, combined with latent class analysis to infer hidden variables indicating PED probability. The top 30% posterior scores would be targeted. This would be optimized weekly using dynamic reinforcement modeling."

Clarity: 3
Logical reasoning: 4
Originality: 5
Specificity and realism of strategy: 4
"""

    try:
        # Create a client object with your API key
        client = OpenAI(api_key=api_key)

        # Use chat.completions instead of completions
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": response}
            ],
            temperature=0.3,
            max_tokens=300
        )
        evaluated_text = completion.choices[0].message.content.strip()
        return evaluated_text

    except Exception as e:
        return f"Error during evaluation: {str(e)}"

def run_resume_evaluation_agents(parsed_resume_data):
    """Run the three resume evaluation agents and return their outputs."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "Error: OPENAI_API_KEY environment variable not found."

    try:
        client = OpenAI(api_key=api_key)
        
        # Run Primary Evaluator
        primary_evaluation = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": PRIMARY_EVALUATOR_PROMPT},
                {"role": "user", "content": json.dumps(parsed_resume_data, indent=2)}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        primary_output = primary_evaluation.choices[0].message.content.strip()
        
        # Run Skeptic
        skeptic_evaluation = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SKEPTIC_PROMPT},
                {"role": "user", "content": f"Resume Data:\n{json.dumps(parsed_resume_data, indent=2)}\n\nPrimary Evaluation:\n{primary_output}"}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        skeptic_output = skeptic_evaluation.choices[0].message.content.strip()
        
        # Run Synthesizer with resume data included
        synthesizer_evaluation = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYNTHESIZER_PROMPT},
                {"role": "user", "content": f"Resume Data:\n{json.dumps(parsed_resume_data, indent=2)}\n\nPrimary Evaluation:\n{primary_output}\n\nSkeptic Evaluation:\n{skeptic_output}"}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        synthesizer_output = synthesizer_evaluation.choices[0].message.content.strip()
        
        return primary_output, skeptic_output, synthesizer_output
        
    except Exception as e:
        return f"Error during evaluation: {str(e)}", "", ""

def generate_overall_assessment(candidate_data):
    """Generate a comprehensive assessment of the candidate using all available data."""
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return "Error: OPENAI_API_KEY environment variable not found."

    try:
        client = OpenAI(api_key=api_key)
        
        # Prepare the context for the assessment
        context = f"""Candidate Profile:
{json.dumps(candidate_data, indent=2)}

Resume Evaluation:
Primary Evaluator: {candidate_data.get('primary_evaluator_output', 'Not available')}
Skeptic: {candidate_data.get('skeptic_evaluator_output', 'Not available')}
Synthesizer: {candidate_data.get('resume_synthesis', 'Not available')}

Reasoning Assessment:
Final Evaluation: {candidate_data.get('final_evaluation', 'Not available')}"""

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": OVERALL_ASSESSMENT_PROMPT},
                {"role": "user", "content": context}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        return completion.choices[0].message.content.strip()
        
    except Exception as e:
        return f"Error generating overall assessment: {str(e)}"

def main():
    # Set page title and configuration
    st.set_page_config(
        page_title="Omnisight: Thinking Test MVP",
        page_icon="🧠",
        layout="centered"
    )

    # Initialize session state variables
    if 'resume_parsed' not in st.session_state:
        st.session_state.resume_parsed = False
    if 'question_index' not in st.session_state:
        st.session_state.question_index = 0
    if 'responses' not in st.session_state:
        st.session_state.responses = {}
    if 'evaluations' not in st.session_state:
        st.session_state.evaluations = {}
    if 'combined_evaluation' not in st.session_state:
        st.session_state.combined_evaluation = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'resume'
    if 'parsed_resume_data' not in st.session_state:
        st.session_state.parsed_resume_data = None
    # Add multi-agent evaluation session state variables
    if 'primary_evaluator_output' not in st.session_state:
        st.session_state.primary_evaluator_output = ""
    if 'skeptic_evaluator_output' not in st.session_state:
        st.session_state.skeptic_evaluator_output = ""
    if 'resume_synthesized_evaluation' not in st.session_state:
        st.session_state.resume_synthesized_evaluation = ""

    # Display app title
    st.title("Omnisight: Thinking Test MVP")

    # Add navigation sidebar
    with st.sidebar:
        st.markdown("## Navigation")
        if st.button("New Assessment", key="new_assessment_sidebar"):
            st.session_state.current_page = 'resume'
            st.rerun()
        if st.button("View All Candidates", key="view_candidates_sidebar"):
            st.session_state.current_page = 'browser'
            st.rerun()

    # Step 1: Resume Upload Page
    if st.session_state.current_page == 'resume':
        st.title("Resume Upload")
        
        # Create form for resume upload
        with st.form("resume_form"):
            uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
            resume_text = st.text_area("Or paste resume text here")
            submitted = st.form_submit_button("Submit")
        
        # Handle form submission
        if submitted:
            resume_text = ""
            
            if uploaded_file is not None:
                if uploaded_file.type == "application/pdf":
                    resume_text = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    resume_text = extract_text_from_docx(uploaded_file)
            elif resume_text.strip():
                resume_text = resume_text
            
            if resume_text:
                # Debug: Show raw text
                with st.expander("Debug - Raw Resume Text"):
                    st.text(resume_text)
                
                # Parse and format resume
                parsed_data = parse_resume(resume_text)
                formatted_text = format_resume(parsed_data)
                
                # Display formatted resume
                st.markdown(formatted_text)
                st.success("Resume processed successfully!")
                
                # Set resume_parsed to True and store parsed data
                st.session_state.resume_parsed = True
                st.session_state.formatted_resume = formatted_text
                st.session_state.parsed_resume_data = parsed_data
        
        # Show evaluation button and results outside the form
        if st.session_state.resume_parsed and st.session_state.parsed_resume_data:
            if not st.session_state.primary_evaluator_output:  # Only show button if evaluation hasn't been run
                if st.button("🧠 Run Evaluation"):
                    with st.spinner("Evaluating resume..."):
                        # Run evaluation
                        primary_output, skeptic_output, synthesizer_output = run_resume_evaluation_agents(st.session_state.parsed_resume_data)
                        
                        if isinstance(primary_output, str) and primary_output.startswith("Error"):
                            st.error(primary_output)
                        else:
                            # Update session state
                            st.session_state.primary_evaluator_output = primary_output
                            st.session_state.skeptic_evaluator_output = skeptic_output
                            st.session_state.resume_synthesized_evaluation = synthesizer_output
                            
                            # Show results in containers
                            with st.expander("🔍 Primary Evaluator Output", expanded=False):
                                st.markdown(primary_output)
                            
                            with st.expander("🚨 Skeptic Evaluator Output", expanded=False):
                                st.markdown(skeptic_output)
                            
                            with st.expander("🧠 Synthesized Evaluation", expanded=False):
                                st.markdown(synthesizer_output)
                            
                            st.success("Evaluation complete!")
            else:
                # Show results if evaluation has been run
                with st.expander("🔍 Primary Evaluator Output", expanded=False):
                    st.markdown(st.session_state.primary_evaluator_output)
                
                with st.expander("🚨 Skeptic Evaluator Output", expanded=False):
                    st.markdown(st.session_state.skeptic_evaluator_output)
                
                with st.expander("🧠 Synthesized Evaluation", expanded=False):
                    st.markdown(st.session_state.resume_synthesized_evaluation)
        
        # Show Take Assessment button
        st.markdown("---")  # Add a separator
        if st.button("Take Assessment", key="take_assessment"):
            st.session_state.current_page = 'assessment'
            st.rerun()

    # Step 2: Assessment Page
    elif st.session_state.current_page == 'assessment':
        st.title("Reasoning Assessment")
        current_question = reasoning_questions[st.session_state.question_index]
        
        st.write(f"Question {st.session_state.question_index + 1} of {len(reasoning_questions)}")
        st.write(current_question['text'])
        
        with st.form(f"assessment_form_{current_question['id']}"):
            user_answer = st.text_area("Your answer:", key=f"answer_{current_question['id']}")
            
            # Create columns for submit and skip buttons
            col1, col2 = st.columns([3, 1])
            with col1:
                submitted = st.form_submit_button("Submit Answer")
            with col2:
                skip_submitted = st.form_submit_button("Skip Question")
        
        if submitted or skip_submitted:
            if skip_submitted:
                user_answer = "[SKIPPED]"
                st.info("Question skipped.")
            elif not user_answer.strip():
                st.warning("Please enter a response before submitting.")
                return
            
            st.success("Thank you for your response!")
            
            # Get GPT-4 evaluation
            with st.spinner("Evaluating your response..."):
                evaluation = get_completion_evaluation(user_answer)
            
            # Store the answer and evaluation
            st.session_state.responses[current_question["id"]] = user_answer
            st.session_state.evaluations[current_question["id"]] = evaluation
            
            # Move to next question
            if st.session_state.question_index < len(reasoning_questions) - 1:
                st.session_state.question_index += 1
                # Clear the text box by setting the response to empty string
                st.session_state.responses[reasoning_questions[st.session_state.question_index]["id"]] = ""
                st.rerun()
            else:
                # Generate combined evaluation
                with st.spinner("Generating combined evaluation..."):
                    st.session_state.combined_evaluation = generate_combined_evaluation(
                        st.session_state.responses,
                        st.session_state.evaluations
                    )
                
                # Save candidate data
                candidate_data = {
                    "reason": "TESTING",  # Can be made configurable later
                    "test_type": "reasoning",
                    "resume": st.session_state.parsed_resume_data,
                    "responses": st.session_state.responses,
                    "evaluations": st.session_state.evaluations,
                    "final_evaluation": st.session_state.combined_evaluation,
                    "resume_synthesis": st.session_state.resume_synthesized_evaluation,
                    "primary_evaluator_output": st.session_state.primary_evaluator_output,
                    "skeptic_evaluator_output": st.session_state.skeptic_evaluator_output
                }
                
                candidate_id = save_candidate_to_file(candidate_data)
                st.success(f"Candidate data saved with ID: {candidate_id}")
                
                st.session_state.current_page = 'combined_evaluation'
                st.rerun()

    # Step 3: Combined Evaluation Page
    elif st.session_state.current_page == 'combined_evaluation' and st.session_state.combined_evaluation:
        st.markdown("## Step 3: Your Complete Evaluation")
        
        # Create candidate data dictionary
        candidate_data = {
            "reason": "TESTING",  # Can be made configurable later
            "test_type": "reasoning",
            "resume": st.session_state.parsed_resume_data,
            "responses": st.session_state.responses,
            "evaluations": st.session_state.evaluations,
            "final_evaluation": st.session_state.combined_evaluation,
            "resume_synthesis": st.session_state.resume_synthesized_evaluation,
            "primary_evaluator_output": st.session_state.primary_evaluator_output,
            "skeptic_evaluator_output": st.session_state.skeptic_evaluator_output
        }
        
        # Display resume
        with st.expander("Resume", expanded=False):
            st.markdown(format_resume(candidate_data["resume"]))
        
        # Display resume evaluations first
        with st.expander("🧠 Synthesized Resume Evaluation", expanded=False):
            st.markdown(candidate_data.get("resume_synthesis", "Not available"))
        
        with st.expander("🔍 Primary Evaluator Output", expanded=False):
            st.markdown(candidate_data.get("primary_evaluator_output", "Not available"))
        
        with st.expander("🚨 Skeptic Evaluator Output", expanded=False):
            st.markdown(candidate_data.get("skeptic_evaluator_output", "Not available"))
        
        # Display responses and evaluations
        for q_id, response in candidate_data["responses"].items():
            question = next(q for q in reasoning_questions if q["id"] == q_id)
            with st.expander(f"Question {reasoning_questions.index(question) + 1}", expanded=False):
                st.markdown("### Question")
                st.write(question["text"])
                st.markdown("### Response")
                st.write(response)
                st.markdown("### Evaluation")
                st.markdown(candidate_data["evaluations"][q_id])
        
        # Display final evaluation
        with st.expander("🎯 Final Assessment Evaluation", expanded=False):
            st.markdown(candidate_data.get("final_evaluation", "Not available"))
        
        # Add Overall Assessment last
        st.markdown("---")
        with st.expander("🌟 Overall Candidate Assessment", expanded=False):
            if 'overall_assessment' not in candidate_data:
                with st.spinner("Generating overall assessment..."):
                    overall_assessment = generate_overall_assessment(candidate_data)
                    candidate_data['overall_assessment'] = overall_assessment
                    # Save the updated candidate data
                    save_candidate_to_file(candidate_data)
            st.markdown(candidate_data.get('overall_assessment', 'Not available'))
        
        # Add navigation button (removed the second column and "View All Candidates" button)
        if st.button("Start Over", key="start_over_eval"):
            # Reset all session state variables
            st.session_state.resume_parsed = False
            st.session_state.question_index = 0
            st.session_state.responses = {}
            st.session_state.evaluations = {}
            st.session_state.combined_evaluation = None
            st.session_state.current_page = 'resume'
            st.rerun()

    # Step 4: Candidate Browser Page
    elif st.session_state.current_page == 'browser':
        st.markdown("## Candidate Browser")
        
        # Get all candidate files
        candidate_files = get_all_candidates()
        
        if not candidate_files:
            st.info("No candidate data found.")
        else:
            # Load all candidate data for the dropdown
            candidates = []
            for file_path in candidate_files:
                with open(file_path, 'r') as f:
                    candidate_data = json.load(f)
                    candidates.append(candidate_data)
            
            # Sort candidates by timestamp (newest first)
            candidates.sort(key=lambda x: x["timestamp"], reverse=True)
            
            # Create dropdown options
            options = {format_candidate_display_name(c): c["candidate_id"] for c in candidates}
            
            # Add dropdown
            selected_display = st.selectbox(
                "Select a candidate:",
                options=list(options.keys())
            )
            
            if selected_display:
                selected_id = options[selected_display]
                candidate_data = next(c for c in candidates if c["candidate_id"] == selected_id)
                
                # Display candidate information
                st.markdown("### Candidate Information")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**ID:** {candidate_data['candidate_id']}")
                with col2:
                    st.markdown(f"**Reason:** {candidate_data['reason']}")
                with col3:
                    st.markdown(f"**Test Type:** {candidate_data['test_type']}")
                
                st.markdown(f"**Timestamp:** {datetime.fromisoformat(candidate_data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Display resume
                with st.expander("Resume", expanded=False):
                    st.markdown(format_resume(candidate_data["resume"]))
                
                # Display resume evaluations first
                with st.expander("🧠 Synthesized Resume Evaluation", expanded=False):
                    st.markdown(candidate_data.get("resume_synthesis", "Not available"))
                
                with st.expander("🔍 Primary Evaluator Output", expanded=False):
                    st.markdown(candidate_data.get("primary_evaluator_output", "Not available"))
                
                with st.expander("🚨 Skeptic Evaluator Output", expanded=False):
                    st.markdown(candidate_data.get("skeptic_evaluator_output", "Not available"))
                
                # Display responses and evaluations
                for q_id, response in candidate_data["responses"].items():
                    question = next(q for q in reasoning_questions if q["id"] == q_id)
                    with st.expander(f"Question {reasoning_questions.index(question) + 1}", expanded=False):
                        st.markdown("### Question")
                        st.write(question["text"])
                        st.markdown("### Response")
                        st.write(response)
                        st.markdown("### Evaluation")
                        st.markdown(candidate_data["evaluations"][q_id])
                
                # Display final evaluation
                with st.expander("🎯 Final Assessment Evaluation", expanded=False):
                    st.markdown(candidate_data.get("final_evaluation", "Not available"))
                
                # Add Overall Assessment last
                st.markdown("---")
                with st.expander("🌟 Overall Candidate Assessment", expanded=False):
                    if 'overall_assessment' not in candidate_data:
                        with st.spinner("Generating overall assessment..."):
                            overall_assessment = generate_overall_assessment(candidate_data)
                            candidate_data['overall_assessment'] = overall_assessment
                            # Save the updated candidate data
                            save_candidate_to_file(candidate_data)
                    st.markdown(candidate_data.get('overall_assessment', 'Not available'))

    # Add keyboard shortcut for submitting response
    st.components.v1.html(
        """
        <script>
        document.addEventListener('keydown', function(event) {
            if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
                // Find the submit button by its text content and click it
                var btn = Array.from(document.getElementsByTagName('button'))
                    .find(b => b.innerText.includes('Submit Answer'));
                if (btn) { btn.click(); }
            }
        });
        </script>
        """,
        height=0
    )

if __name__ == "__main__":
    main()