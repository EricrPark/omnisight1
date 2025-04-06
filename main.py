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

Format your response as follows:

### Overall Assessment
[1-2 sentences summarizing the candidate's general approach and thinking style]

### Key Strengths
- [Only include if there are clear strengths demonstrated]
- [Limit to 1-2 most significant strengths]

### Areas for Development
- [Only include if there are clear areas for improvement]
- [Limit to 1-2 most significant areas]

### Implications
[1-2 sentences about how this thinking style might translate to real-world scenarios]

Be specific and evidence-based, referencing particular aspects of the candidate's responses to support your analysis.
If the candidate provided minimal or non-substantive responses, keep your evaluation brief and focus on the lack of engagement or depth.
For skipped questions, note this in your analysis but focus on the quality of the responses that were provided."""

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
        with st.form("resume_form"):
            uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
            resume_text = st.text_area("Or paste resume text here")
            submitted = st.form_submit_button("Submit")
        
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
        
        # Show Take Assessment button (moved outside the resume_parsed condition)
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
            submitted = st.form_submit_button("Submit Answer")
        
        if submitted:
            if user_answer.strip():
                st.success("Thank you for your response!")
                
                # Get GPT-4 evaluation
                with st.spinner("Evaluating your response..."):
                    evaluation = get_completion_evaluation(user_answer)
                
                # Store the answer and evaluation
                st.session_state.responses[current_question["id"]] = user_answer
                st.session_state.evaluations[current_question["id"]] = evaluation
                
                # Check if all questions are answered
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
                        "final_evaluation": st.session_state.combined_evaluation
                    }
                    
                    candidate_id = save_candidate_to_file(candidate_data)
                    st.success(f"Candidate data saved with ID: {candidate_id}")
                    
                    st.session_state.current_page = 'combined_evaluation'
                    st.rerun()
            else:
                st.warning("Please enter a response before submitting.")

    # Step 3: Combined Evaluation Page
    elif st.session_state.current_page == 'combined_evaluation' and st.session_state.combined_evaluation:
        st.markdown("## Step 3: Your Complete Evaluation")
        
        # Display individual responses and evaluations
        for q_id, response in st.session_state.responses.items():
            question = next(q for q in reasoning_questions if q["id"] == q_id)
            with st.expander(f"Question {reasoning_questions.index(question) + 1}", expanded=False):
                st.markdown("### Question")
                st.write(question["text"])
                st.markdown("### Your Response")
                st.write(response)
                st.markdown("### Evaluation")
                st.markdown(st.session_state.evaluations[q_id])
        
        # Display combined evaluation
        st.markdown(st.session_state.combined_evaluation)
        
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
                st.markdown("### Final Evaluation")
                st.markdown(candidate_data["final_evaluation"])

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