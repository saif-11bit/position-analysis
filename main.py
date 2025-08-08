import streamlit as st
import pandas as pd
import ast
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import List, Dict
import openai
import re
import time

# ---- Data Classes ----
@dataclass
class Project:
    title: str
    description: str

@dataclass
class Experience:
    company: str
    role: str
    details: str

@dataclass
class Student:
    name: str
    email: str
    education: str  # Added education field
    degree: str     # Added degree field
    skills: List[str]
    projects: List[Project]
    experience: List[Experience]
    achievements: List[str] = None  # Added achievements
    extracurricular: List[str] = None  # Added extracurricular

@dataclass
class RoleEvaluation:
    question: str
    weight: float  # Weight for this evaluation criteria

@dataclass
class InternshipRequirement:
    job_title: str
    company: str
    required_skills: List[str]
    preferred_skills: List[str]
    role_type: str
    relevant_degrees: List[str]  # Added relevant degrees
    evaluation_criteria: List[RoleEvaluation]  # Added evaluation criteria

@dataclass
class EvaluationResult:
    question: str
    answer: bool
    reasoning: str
    weight: float

@dataclass
class MatchResult:
    student: Student
    score: float
    skill_match_score: float
    education_match_score: float
    evaluation_scores: List[EvaluationResult]
    overall_evaluation_score: float
    matched_skills: List[str]
    missing_skills: List[str]
    overall_reasoning: str

# ---- Enhanced Matcher Class ----
class EnhancedInternshipMatcher:
    def __init__(self, openai_api_key: str = None, use_llm=False):
        self.openai_api_key = openai_api_key
        self.use_llm = use_llm
        if openai_api_key:
            openai.api_key = openai_api_key
        
        self.skill_synonyms = {
            'javascript': ['js', 'node.js', 'nodejs', 'react', 'angular', 'vue'],
            'python': ['django', 'flask', 'fastapi', 'pandas', 'numpy', 'scikit-learn'],
            'java': ['spring', 'hibernate', 'maven', 'gradle'],
            'machine learning': ['ml', 'deep learning', 'neural networks', 'tensorflow', 'pytorch', 'ai'],
            'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'nosql'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes'],
            'frontend': ['html', 'css', 'react', 'angular', 'vue', 'javascript'],
            'backend': ['api', 'server', 'database', 'microservices'],
            'marketing': ['digital marketing', 'social media', 'seo', 'sem', 'content marketing'],
            'finance': ['financial modeling', 'excel', 'accounting', 'budgeting', 'forecasting'],
            'design': ['figma', 'photoshop', 'illustrator', 'sketch', 'wireframing', 'prototyping']
        }
        
        # Education matching keywords
        self.education_keywords = {
            'computer science': ['computer science', 'cs', 'cse', 'computer engineering', 'software engineering'],
            'information technology': ['information technology', 'it', 'information systems'],
            'engineering': ['engineering', 'btech', 'b.tech', 'be', 'b.e'],
            'business': ['business', 'mba', 'bba', 'commerce', 'management'],
            'marketing': ['marketing', 'advertising', 'communications', 'media'],
            'finance': ['finance', 'accounting', 'economics', 'chartered accountant', 'ca', 'cfa'],
            'design': ['design', 'arts', 'graphic design', 'visual arts', 'fine arts'],
            'data science': ['data science', 'statistics', 'mathematics', 'analytics']
        }

    def normalize_skills(self, skills: List[str]) -> List[str]:
        normalized = []
        for skill in skills:
            skill_lower = skill.lower().strip()
            normalized.append(skill_lower)
            for main_skill, synonyms in self.skill_synonyms.items():
                if skill_lower in synonyms or skill_lower == main_skill:
                    normalized.extend([main_skill] + synonyms)
        return list(set(normalized))
    
    def calculate_education_match_score(self, student_education: str, student_degree: str, 
                                      relevant_degrees: List[str]) -> float:
        if not relevant_degrees:
            return 1.0  # If no specific degree required, all are equally valid
        
        education_text = f"{student_education} {student_degree}".lower()
        max_score = 0.0
        
        for required_degree in relevant_degrees:
            required_lower = required_degree.lower()
            
            # Check for exact matches first
            if required_lower in education_text:
                max_score = max(max_score, 1.0)
                continue
            
            # Check for keyword matches
            for category, keywords in self.education_keywords.items():
                if required_lower in keywords or any(keyword in required_lower for keyword in keywords):
                    if any(keyword in education_text for keyword in keywords):
                        max_score = max(max_score, 0.8)
        
        return max_score
    
    def calculate_skill_match_score(self, student_skills: List[str], 
                                  required_skills: List[str], 
                                  preferred_skills: List[str]) -> (float, List[str], List[str]):
        student_skills_norm = self.normalize_skills(student_skills)
        required_skills_norm = self.normalize_skills(required_skills)
        preferred_skills_norm = self.normalize_skills(preferred_skills)
        
        matched_required = set(student_skills_norm) & set(required_skills_norm)
        matched_preferred = set(student_skills_norm) & set(preferred_skills_norm)
        
        required_score = len(matched_required) / len(required_skills_norm) if required_skills_norm else 0
        preferred_score = len(matched_preferred) / len(preferred_skills_norm) if preferred_skills_norm else 0
        
        skill_score = (required_score * 0.7) + (preferred_score * 0.3)
        matched_skills = list(matched_required | matched_preferred)
        missing_skills = list(set(required_skills_norm) - set(student_skills_norm))
        
        return skill_score, matched_skills, missing_skills
    
    def evaluate_student_against_criteria(self, student: Student, 
                                        evaluation_criteria: List[RoleEvaluation]) -> (List[EvaluationResult], float):
        if not self.use_llm or not self.openai_api_key:
            # Simple rule-based evaluation for non-LLM mode
            results = []
            for criteria in evaluation_criteria:
                results.append(EvaluationResult(
                    question=criteria.question,
                    answer=True,
                    reasoning="Rule-based evaluation not available without LLM",
                    weight=criteria.weight
                ))
            return results, 0.7  # Default score
        
        try:
            student_profile = f"""
            Student: {student.name}
            Education: {student.education}
            Degree: {student.degree}
            Skills: {', '.join(student.skills)}
            Projects: {[f"{p.title}: {p.description[:200]}..." for p in student.projects]}
            Experience: {[f"{e.role} at {e.company}: {e.details[:200]}..." for e in student.experience]}
            Achievements: {', '.join(student.achievements or [])}
            Extracurricular: {', '.join(student.extracurricular or [])}
            """
            
            evaluation_results = []
            total_weighted_score = 0
            total_weight = 0
            
            for criteria in evaluation_criteria:
                prompt = f"""
                Evaluate the following student against this specific criterion:
                
                CRITERION: {criteria.question}
                
                STUDENT PROFILE:
                {student_profile}
                
                Based on the student's education, skills, projects, experience, achievements, and extracurricular activities, 
                can this student fulfill this criterion? 
                
                Respond with:
                ANSWER: YES or NO
                REASONING: [Detailed reasoning based on specific evidence from the student profile]
                
                Be specific about which aspects of the student's profile support your decision.
                """
                
                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.3
                )
                
                content = response.choices[0].message.content.strip()
                answer_match = re.search(r'ANSWER:\s*(YES|NO)', content, re.IGNORECASE)
                reasoning_match = re.search(r'REASONING:\s*(.+)', content, re.DOTALL)
                
                answer = answer_match.group(1).upper() == "YES" if answer_match else False
                reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
                
                evaluation_results.append(EvaluationResult(
                    question=criteria.question,
                    answer=answer,
                    reasoning=reasoning,
                    weight=criteria.weight
                ))
                
                score_contribution = 1.0 if answer else 0.0
                total_weighted_score += score_contribution * criteria.weight
                total_weight += criteria.weight
                
                # Rate limiting
                time.sleep(0.2)
            
            overall_evaluation_score = total_weighted_score / total_weight if total_weight > 0 else 0
            return evaluation_results, overall_evaluation_score
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            # Fallback evaluation
            results = []
            for criteria in evaluation_criteria:
                results.append(EvaluationResult(
                    question=criteria.question,
                    answer=False,
                    reasoning=f"Error in evaluation: {str(e)}",
                    weight=criteria.weight
                ))
            return results, 0.0

    def generate_overall_reasoning(self, student: Student, internship_req: InternshipRequirement, 
                                 skill_score: float, education_score: float, 
                                 evaluation_score: float, evaluation_results: List[EvaluationResult]) -> str:
        if not self.use_llm or not self.openai_api_key:
            return f"Basic matching: Skills {skill_score:.2f}, Education {education_score:.2f}, Evaluation {evaluation_score:.2f}"
        
        try:
            evaluation_summary = "\n".join([
                f"- {result.question}: {'✓' if result.answer else '✗'} ({result.reasoning})"
                for result in evaluation_results
            ])
            
            prompt = f"""
            Provide a concise overall reasoning for why this student scored {(skill_score * 0.3 + education_score * 0.2 + evaluation_score * 0.5):.2f} 
            for the {internship_req.job_title} role.
            
            Breakdown:
            - Skill Match: {skill_score:.2f}
            - Education Match: {education_score:.2f}
            - Role Evaluation: {evaluation_score:.2f}
            
            Evaluation Details:
            {evaluation_summary}
            
            Provide a 2-3 sentence summary highlighting the main strengths and areas for improvement.
            """
            
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating reasoning: {str(e)}"

    def match_student_to_internship(self, student: Student, internship_req: InternshipRequirement) -> MatchResult:
        # Calculate skill match
        skill_score, matched_skills, missing_skills = self.calculate_skill_match_score(
            student.skills, internship_req.required_skills, internship_req.preferred_skills
        )
        
        # Calculate education match
        education_score = self.calculate_education_match_score(
            student.education, student.degree, internship_req.relevant_degrees
        )
        
        # Evaluate against role criteria
        evaluation_results, evaluation_score = self.evaluate_student_against_criteria(
            student, internship_req.evaluation_criteria
        )
        
        # Calculate overall score (weighted)
        overall_score = (skill_score * 0.3 + education_score * 0.2 + evaluation_score * 0.5)
        
        # Generate reasoning
        reasoning = self.generate_overall_reasoning(
            student, internship_req, skill_score, education_score, evaluation_score, evaluation_results
        )
        
        return MatchResult(
            student=student,
            score=overall_score,
            skill_match_score=skill_score,
            education_match_score=education_score,
            evaluation_scores=evaluation_results,
            overall_evaluation_score=evaluation_score,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            overall_reasoning=reasoning
        )

# ---- CSV Parser ----
def extract_text_from_html(html_str):
    if not html_str or not isinstance(html_str, str):
        return ""
    return BeautifulSoup(html_str, "html.parser").get_text(separator=" ").strip()

def parse_student_csv(file_obj):
    df = pd.read_csv(file_obj)
    students = []
    for _, row in df.iterrows():
        # Parse skills
        try:
            skills = ast.literal_eval(row['skill'])
            if not isinstance(skills, list):
                skills = []
        except:
            skills = []
        
        # Parse projects
        projects = []
        try:
            project_list = ast.literal_eval(row['projects'])
            for proj in project_list:
                name = proj.get('name', '') if isinstance(proj, dict) else ''
                desc_html = proj.get('description', '') if isinstance(proj, dict) else ''
                desc = extract_text_from_html(desc_html)
                projects.append(Project(title=name, description=desc))
        except:
            pass
        
        # Parse experience
        experience = []
        try:
            exp_list = ast.literal_eval(row['experience'])
            for exp in exp_list:
                company = exp.get('company', '') if isinstance(exp, dict) else ''
                role = exp.get('role', '') if isinstance(exp, dict) else ''
                desc_html = exp.get('details', '') if isinstance(exp, dict) else ''
                desc = extract_text_from_html(desc_html)
                experience.append(Experience(company=company, role=role, details=desc))
        except:
            pass
        
        # Parse achievements and extracurricular (if available)
        achievements = []
        extracurricular = []
        
        try:
            if 'achievements' in row:
                achievements = ast.literal_eval(row['achievements']) if pd.notna(row['achievements']) else []
        except:
            achievements = []
            
        try:
            if 'extracurricular' in row:
                extracurricular = ast.literal_eval(row['extracurricular']) if pd.notna(row['extracurricular']) else []
        except:
            extracurricular = []
        
        students.append(Student(
            name=row.get('StudentName', row.get('name', '')),
            email=row.get('Email Address', row.get('email', '')),
            education=row.get('education', row.get('Education', '')),
            degree=row.get('degree', row.get('Degree', '')),
            skills=skills,
            projects=projects,
            experience=experience,
            achievements=achievements,
            extracurricular=extracurricular
        ))
    return students

# ---- Streamlit App ----
st.set_page_config(page_title="Enhanced Student-Job Role Matcher", layout="wide")
st.title("🎓 Enhanced Student-Job Matching with Education & Role Evaluation")

# Sidebar - CSV Upload
st.sidebar.header("Step 1: Upload Student CSV")
students_csv = st.sidebar.file_uploader("Students CSV", type="csv")
students = []
if students_csv:
    students = parse_student_csv(students_csv)
    st.sidebar.success(f"{len(students)} students loaded!")

# Sidebar - Enhanced Job Roles
st.sidebar.header("Step 2: Define Job Roles")

if "enhanced_roles" not in st.session_state:
    st.session_state["enhanced_roles"] = [
        {
            "job_title": "Software Engineer Intern",
            "required_skills": "Python, Java, Data Structures, Algorithms, Git",
            "preferred_skills": "Django, React, SQL",
            "relevant_degrees": "Computer Science, Software Engineering, Information Technology",
            "evaluation_criteria": "Can write clean, maintainable code?|0.3,Can solve algorithmic problems?|0.2,Can work with databases?|0.2,Can collaborate in a team using version control?|0.15,Can learn new technologies quickly?|0.15"
        },
        {
            "job_title": "AI/ML Intern",
            "required_skills": "Python, Machine Learning, Data Analysis, Statistics",
            "preferred_skills": "TensorFlow, PyTorch, Pandas, NumPy",
            "relevant_degrees": "Computer Science, Data Science, Mathematics, Statistics, Electrical Engineering",
            "evaluation_criteria": "Can implement machine learning algorithms?|0.25,Can perform data preprocessing and analysis?|0.25,Can interpret model results and metrics?|0.2,Can work with large datasets?|0.15,Can communicate technical findings clearly?|0.15"
        },
        {
            "job_title": "UI/UX Design Intern",
            "required_skills": "Figma, User Research, Wireframing, Prototyping",
            "preferred_skills": "Adobe Creative Suite, Sketch, User Testing",
            "relevant_degrees": "Design, Arts, Human-Computer Interaction, Psychology",
            "evaluation_criteria": "Can create user-centered designs?|0.3,Can conduct user research and usability testing?|0.25,Can create wireframes and prototypes?|0.2,Can collaborate with developers and stakeholders?|0.15,Can iterate designs based on feedback?|0.1"
        }
    ]

enhanced_roles = st.session_state["enhanced_roles"]

def add_enhanced_role():
    enhanced_roles.append({
        "job_title": "",
        "required_skills": "",
        "preferred_skills": "",
        "relevant_degrees": "",
        "evaluation_criteria": ""
    })
    st.session_state["enhanced_roles"] = enhanced_roles

def remove_enhanced_role(idx):
    del enhanced_roles[idx]
    st.session_state["enhanced_roles"] = enhanced_roles

for idx, role in enumerate(enhanced_roles):
    with st.sidebar.expander(f"Role {idx+1}: {role['job_title'] or 'New Role'}", expanded=False):
        job_title = st.text_input(f"Job Title {idx+1}", value=role["job_title"], key=f"enh_job_title_{idx}")
        required_skills = st.text_input(f"Required Skills {idx+1}", value=role["required_skills"], key=f"enh_req_skills_{idx}")
        preferred_skills = st.text_input(f"Preferred Skills {idx+1}", value=role["preferred_skills"], key=f"enh_pref_skills_{idx}")
        relevant_degrees = st.text_input(f"Relevant Degrees {idx+1}", value=role["relevant_degrees"], key=f"enh_degrees_{idx}")
        evaluation_criteria = st.text_area(f"Evaluation Criteria {idx+1} (question|weight)", 
                                         value=role["evaluation_criteria"], key=f"enh_criteria_{idx}",
                                         help="Format: 'Can do X?|0.3' - Use newlines OR commas to separate criteria. Weights should sum to 1.0")
        
        enhanced_roles[idx].update({
            "job_title": job_title,
            "required_skills": required_skills,
            "preferred_skills": preferred_skills,
            "relevant_degrees": relevant_degrees,
            "evaluation_criteria": evaluation_criteria
        })
        
        if st.button(f"Remove Role {idx+1}", key=f"enh_remove_{idx}"):
            remove_enhanced_role(idx)
            st.rerun()

st.sidebar.button("Add New Role", on_click=add_enhanced_role)

# Process job roles
job_roles = []
for role in enhanced_roles:
    if role["job_title"] and role["required_skills"]:
        # Parse evaluation criteria (handle both newline and comma separation)
        criteria = []
        if role["evaluation_criteria"]:
            # First try splitting by newlines, then by commas if no newlines found
            lines = role["evaluation_criteria"].split('\n')
            if len(lines) == 1 and ',' in role["evaluation_criteria"]:
                lines = role["evaluation_criteria"].split(',')
            
            for line in lines:
                line = line.strip()
                if '|' in line:
                    try:
                        question, weight = line.split('|', 1)
                        criteria.append(RoleEvaluation(
                            question=question.strip(),
                            weight=float(weight.strip())
                        ))
                    except ValueError:
                        st.warning(f"Invalid weight format in role {role['job_title']}: {line}")
                        continue
        
        job_roles.append(
            InternshipRequirement(
                job_title=role["job_title"],
                company="",
                required_skills=[s.strip() for s in role["required_skills"].split(",") if s.strip()],
                preferred_skills=[s.strip() for s in role["preferred_skills"].split(",") if s.strip()],
                role_type=role["job_title"],
                relevant_degrees=[d.strip() for d in role["relevant_degrees"].split(",") if d.strip()],
                evaluation_criteria=criteria
            )
        )

# LLM Configuration
st.sidebar.header("Step 3: AI Configuration")
use_llm = st.sidebar.checkbox("Enable AI for detailed role evaluation", value=False)
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if use_llm and not openai_api_key:
    st.warning("Please provide your OpenAI API key to use AI evaluation.")

# Main matching logic
matcher = EnhancedInternshipMatcher(openai_api_key=openai_api_key if use_llm else None, use_llm=use_llm)

if students and job_roles and (not use_llm or (use_llm and openai_api_key)):
    st.header("Enhanced Matching Results")
    
    # Create a unique key for current configuration
    config_key = f"{len(students)}_{len(job_roles)}_{use_llm}_{hash(str([r.job_title for r in job_roles]))}"
    
    # Check if we need to reprocess or if results are cached
    if "last_config_key" not in st.session_state or st.session_state.last_config_key != config_key:
        with st.spinner("Processing student-job matches..."):
            all_matches = {}
            matrix_rows = []
            
            progress_bar = st.progress(0)
            total_combinations = len(students) * len(job_roles)
            processed = 0
            
            for student in students:
                student_results = {}
                row = {"Student": student.name, "Email": student.email, "Education": f"{student.education} - {student.degree}"}
                
                for role in job_roles:
                    result = matcher.match_student_to_internship(student, role)
                    student_results[role.job_title] = result
                    
                    row[f"{role.job_title} - Score"] = f"{result.score:.3f}"
                    row[f"{role.job_title} - Skills"] = f"{result.skill_match_score:.2f}"
                    row[f"{role.job_title} - Education"] = f"{result.education_match_score:.2f}"
                    row[f"{role.job_title} - Role Eval"] = f"{result.overall_evaluation_score:.2f}"
                    
                    # Add detailed evaluation results
                    if result.evaluation_scores:
                        eval_details = []
                        for eval_result in result.evaluation_scores:
                            status = "YES" if eval_result.answer else "NO"
                            eval_details.append(f"{eval_result.question} → {status} (Reason: {eval_result.reasoning[:100]}...)")
                        row[f"{role.job_title} - Evaluation Details"] = " | ".join(eval_details)
                    else:
                        row[f"{role.job_title} - Evaluation Details"] = "No detailed evaluation available"
                    
                    # Add matched and missing skills
                    row[f"{role.job_title} - Matched Skills"] = ", ".join(result.matched_skills)
                    row[f"{role.job_title} - Missing Skills"] = ", ".join(result.missing_skills)
                    
                    # Add overall reasoning
                    row[f"{role.job_title} - Overall Reasoning"] = result.overall_reasoning
                    
                    processed += 1
                    progress_bar.progress(processed / total_combinations)
                    
                    if use_llm:
                        time.sleep(0.1)  # Rate limiting for API calls
                
                all_matches[student.email] = student_results
                matrix_rows.append(row)
            
            progress_bar.empty()
            
            # Cache the results in session state
            st.session_state.all_matches = all_matches
            st.session_state.matrix_rows = matrix_rows
            st.session_state.last_config_key = config_key
    else:
        # Use cached results
        all_matches = st.session_state.all_matches
        matrix_rows = st.session_state.matrix_rows
    
    # Display results matrix
    df_results = pd.DataFrame(matrix_rows)
    st.dataframe(df_results, use_container_width=True)
    
    # Detailed student analysis
    st.header("Detailed Student Analysis")
    
    selected_email = st.selectbox(
        "Select a student for detailed analysis",
        [s.email for s in students],
        format_func=lambda email: f"{next((s.name for s in students if s.email == email), email)} ({email})"
    )
    
    if selected_email:
        selected_student = next((s for s in students if s.email == selected_email), None)
        selected_results = all_matches[selected_email]
        
        st.subheader(f"Detailed Analysis: {selected_student.name}")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Student Profile:**")
            st.write(f"📚 Education: {selected_student.education}")
            st.write(f"🎓 Degree: {selected_student.degree}")
            st.write(f"💼 Skills: {', '.join(selected_student.skills)}")
            st.write(f"📊 Projects: {len(selected_student.projects)} projects")
            st.write(f"💻 Experience: {len(selected_student.experience)} experiences")
        
        with col2:
            # Role ranking
            role_scores = [(role.job_title, result.score) for role, result in 
                          zip(job_roles, [selected_results[r.job_title] for r in job_roles])]
            role_scores.sort(key=lambda x: x[1], reverse=True)
            
            st.write("**Best Role Matches:**")
            for i, (role_name, score) in enumerate(role_scores[:3]):
                st.write(f"{i+1}. {role_name}: {score:.3f}")
        
        # Detailed role analysis
        selected_role_name = st.selectbox("Select role for detailed breakdown", 
                                        [role.job_title for role in job_roles])
        
        if selected_role_name:
            result = selected_results[selected_role_name]
            selected_role = next((r for r in job_roles if r.job_title == selected_role_name), None)
            
            st.subheader(f"Analysis for {selected_role_name}")
            
            # Score breakdown
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Overall Score", f"{result.score:.3f}")
            with col2:
                st.metric("Skills Match", f"{result.skill_match_score:.3f}")
            with col3:
                st.metric("Education Match", f"{result.education_match_score:.3f}")
            with col4:
                st.metric("Role Evaluation", f"{result.overall_evaluation_score:.3f}")
            
            # Detailed evaluation results
            if result.evaluation_scores:
                st.write("**Role-Specific Evaluation:**")
                for eval_result in result.evaluation_scores:
                    status = "✅ YES" if eval_result.answer else "❌ NO"
                    st.write(f"**{eval_result.question}** {status} (Weight: {eval_result.weight})")
                    st.write(f"*Reasoning:* {eval_result.reasoning}")
                    st.write("---")
            
            # Skills analysis
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Matched Skills:**")
                for skill in result.matched_skills:
                    st.write(f"✅ {skill}")
            
            with col2:
                st.write("**Missing Skills:**")
                for skill in result.missing_skills:
                    st.write(f"❌ {skill}")
            
            # Overall reasoning
            if result.overall_reasoning:
                st.write("**Overall Assessment:**")
                st.info(result.overall_reasoning)
    
    # Download results
    st.header("Export Results")
    
    # Create CSV content
    csv_content = df_results.to_csv(index=False)
    
    st.download_button(
        label="📊 Download Results as CSV",
        data=csv_content,
        file_name=f"student_job_matching_results_{time.strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        help="Download the matching results as a CSV file"
    )

else:
    st.info("Please upload student data, define job roles, and configure AI settings to begin matching.")

# Instructions
with st.expander("📋 Instructions & CSV Format"):
    st.markdown("""
    **Required CSV Columns for Students:**
    - `StudentName` or `name`: Student's full name
    - `Email Address` or `email`: Student's email
    - `education`: Educational institution/background
    - `degree`: Degree program (e.g., "B.Tech Computer Science")
    - `skill`: List of skills (Python list format)
    - `projects`: List of project dictionaries with 'name' and 'description'
    - `experience`: List of experience dictionaries with 'company', 'role', 'details'
    - `achievements` (optional): List of achievements
    - `extracurricular` (optional): List of extracurricular activities
    
    **Job Role Configuration:**
    - **Required Skills**: Comma-separated core skills
    - **Preferred Skills**: Comma-separated nice-to-have skills  
    - **Relevant Degrees**: Comma-separated degree programs
    - **Evaluation Criteria**: One per line OR comma-separated, format: "Question?|weight"
      - Example: "Can write clean code?|0.3" or use commas: "Question1?|0.3,Question2?|0.4,Question3?|0.3"
      - Weights should sum to 1.0
    
    **Scoring System:**
    - Skills Match: 30% (required skills weighted 70%, preferred 30%)
    - Education Match: 20% (degree relevance to role)
    - Role Evaluation: 50% (capability assessment via AI)
    """)

st.markdown("---")
st.caption("Enhanced Student-Job Matching System with Education & Role-Based Evaluation 🚀")