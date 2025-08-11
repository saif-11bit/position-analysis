import streamlit as st
import pandas as pd
import ast
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
import openai
import re
import time
import json
import hashlib
import json

def compute_config_fingerprint(students, job_roles, use_llm: bool) -> str:
    """Stable hash of inputs to detect changes."""
    stu_repr = [
        {
            "email": s.email,
            "name": s.name,
            "education": s.education,
            "degree": s.degree,
            "skills": sorted([str(x).lower() for x in (s.skills or [])]),
            "projects": [(p.title, p.description[:200]) for p in (s.projects or [])],
            "experience": [(e.company, e.role, e.details[:200]) for e in (s.experience or [])],
        }
        for s in students
    ]
    roles_repr = [
        {
            "job_title": r.job_title,
            "relevant_degrees": sorted([d.lower() for d in r.relevant_degrees]),
            "evaluation_criteria": [(c.question, c.weight) for c in r.evaluation_criteria],
            "technical_stack": r.technical_stack,  # already dict
        }
        for r in job_roles
    ]
    payload = {
        "students": stu_repr,
        "roles": roles_repr,
        "use_llm": use_llm,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

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
    education: str
    degree: str
    skills: List[str]
    projects: List[Project]
    experience: List[Experience]
    achievements: List[str] = None
    extracurricular: List[str] = None

@dataclass
class RoleEvaluation:
    question: str
    weight: float

@dataclass
class InternshipRequirement:
    job_title: str
    company: str
    role_type: str
    relevant_degrees: List[str]
    evaluation_criteria: List[RoleEvaluation]
    # NEW: technical stack definition
    # {
    #   "Programming Language": {"options": ["python","r"], "mandatory": true, "min_match": 1},
    #   "Libraries & Frameworks": {"options": ["tensorflow","pytorch"], "mandatory": true, "min_match": 1},
    #   "Data Libraries": {"options": ["pandas","numpy"], "mandatory": false, "min_match": 2},
    #   "Software & Platform": {"options": ["jupyter notebook"], "mandatory": false, "min_match": 1},
    #   "Database & API": {"options": ["sql","nosql"], "mandatory": true, "min_match": 1}
    # }
    technical_stack: Dict[str, Dict[str, Any]]

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
    # replaced "skill_match_score" with stack coverage
    stack_pass: bool
    stack_coverage_score: float
    education_match_score: float
    evaluation_scores: List[EvaluationResult]
    overall_evaluation_score: float
    matched_skills: List[str]
    missing_skills: List[str]
    overall_reasoning: str
    stack_details: Dict[str, Dict[str, Any]]

# ---- Enhanced Matcher Class ----
class EnhancedInternshipMatcher:
    def __init__(self, openai_api_key: str = None, use_llm=False):
        self.openai_api_key = openai_api_key
        self.use_llm = use_llm
        if openai_api_key:
            openai.api_key = openai_api_key

        # Synonyms / expansions to normalize skills
        self.skill_synonyms = {
            'javascript': ['js', 'node.js', 'nodejs', 'react', 'angular', 'vue'],
            'python': ['django', 'flask', 'fastapi', 'pandas', 'numpy', 'scikit-learn', 'sklearn'],
            'r': ['r language'],
            'java': ['spring', 'hibernate', 'maven', 'gradle'],
            'machine learning': ['ml', 'deep learning', 'neural networks', 'tensorflow', 'pytorch', 'ai'],
            'tensorflow': ['tf'],
            'pytorch': ['torch'],
            'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'nosql'],
            'sql': ['postgres', 'postgresql', 'mysql', 'mariadb', 'sqlite'],
            'nosql': ['mongodb', 'dynamodb', 'cassandra', 'couchdb'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes'],
            'frontend': ['html', 'css', 'react', 'angular', 'vue', 'javascript'],
            'backend': ['api', 'server', 'microservices', 'fastapi', 'django', 'flask'],
            'jupyter notebook': ['jupyter', 'ipynb', 'notebook']
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
        normalized = set()
        for skill in skills or []:
            s = (skill or '').lower().strip()
            if not s:
                continue
            normalized.add(s)
            # Expand synonyms
            for main_skill, synonyms in self.skill_synonyms.items():
                if s == main_skill or s in synonyms:
                    normalized.add(main_skill)
                    for syn in synonyms:
                        normalized.add(syn.lower())
        return list(normalized)

    # -------- NEW: Technical Stack Check --------
    def check_technical_stack(
        self,
        student_skills: List[str],
        technical_stack: Dict[str, Dict[str, Any]]
    ) -> Tuple[bool, float, Dict[str, Dict[str, Any]], List[str], List[str]]:
        """
        Returns:
          passed (bool): all mandatory categories satisfied
          coverage_score (float 0..1): overall stack completeness across categories
          details (dict): per-category required, matched, missing, mandatory, met
          flat_matched (list): flattened list of all matched options
          flat_missing (list): flattened list of still-missing options across ALL categories (prioritizing mandatory first)
        """
        student_norm = set(self.normalize_skills(student_skills))
        details = {}
        all_matched = set()
        all_missing = []

        if not technical_stack:
            # No stack constraints means pass with full coverage
            return True, 1.0, {}, [], []

        categories = list(technical_stack.keys())
        if not categories:
            return True, 1.0, {}, [], []

        passed = True
        category_scores = []

        for category, req in technical_stack.items():
            options = [str(o).lower().strip() for o in req.get("options", []) if str(o).strip()]
            mandatory = bool(req.get("mandatory", False))
            min_match = max(int(req.get("min_match", 1)), 1)

            # Expand options via synonyms too (simple: keep canonical option labels)
            # matching happens by intersection with normalized skills
            matches = set()
            for opt in options:
                # Consider either exact presence of opt or its normalized family in student_norm
                if opt in student_norm:
                    matches.add(opt)
                else:
                    # if opt is a main skill, its synonyms may be in student_norm already due to normalize_skills
                    # so we rely on opt-in-student_norm primarily; optionally we can alias common pairs
                    pass

            met = len(matches) >= min_match
            if mandatory and not met:
                passed = False

            details[category] = {
                "required": options,
                "min_match": min_match,
                "matched": sorted(list(matches)),
                "missing": sorted(list(set(options) - matches)),
                "mandatory": mandatory,
                "met": met
            }

            # coverage: cap by min_match (1.0 when min_match satisfied)
            denom = float(max(min_match, 1))
            category_score = min(len(matches), min_match) / denom
            category_scores.append(category_score)

            all_matched |= matches
            # accumulate missing (for display) — keep mandatory first
            if not met:
                # add all still-missing options for this category
                all_missing.extend([f"{category}: {m}" for m in sorted(list(set(options) - matches))])

        coverage_score = sum(category_scores) / len(category_scores) if category_scores else 1.0
        return passed, coverage_score, details, sorted(list(all_matched)), all_missing

    # -------- Education (unchanged) --------
    def calculate_education_match_score(self, student_education: str, student_degree: str, relevant_degrees: List[str]) -> float:
        if not relevant_degrees:
            return 1.0
        education_text = f"{student_education} {student_degree}".lower()
        max_score = 0.0
        for required_degree in relevant_degrees:
            required_lower = required_degree.lower()
            if required_lower in education_text:
                max_score = max(max_score, 1.0)
                continue
            for category, keywords in self.education_keywords.items():
                if required_lower in keywords or any(keyword in required_lower for keyword in keywords):
                    if any(keyword in education_text for keyword in keywords):
                        max_score = max(max_score, 0.8)
        return max_score

    # -------- Role Criteria (kept) --------
    def evaluate_student_against_criteria(self, student: Student, evaluation_criteria: List[RoleEvaluation]) -> (List[EvaluationResult], float):
        if not self.use_llm or not self.openai_api_key:
            results = []
            for criteria in evaluation_criteria:
                results.append(EvaluationResult(
                    question=criteria.question,
                    answer=True,
                    reasoning="Rule-based evaluation not available without LLM",
                    weight=criteria.weight
                ))
            return results, 0.7  # default baseline for non-LLM
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
            total_weighted_score = 0.0
            total_weight = 0.0
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
                time.sleep(0.2)
            overall_evaluation_score = total_weighted_score / total_weight if total_weight > 0 else 0
            return evaluation_results, overall_evaluation_score
        except Exception as e:
            results = []
            for criteria in evaluation_criteria:
                results.append(EvaluationResult(
                    question=criteria.question,
                    answer=False,
                    reasoning=f"Error in evaluation: {str(e)}",
                    weight=criteria.weight
                ))
            return results, 0.0

    def generate_overall_reasoning(
        self, student: Student, internship_req: InternshipRequirement,
        stack_pass: bool, stack_score: float, education_score: float,
        evaluation_score: float, evaluation_results: List[EvaluationResult],
        stack_details: Dict[str, Dict[str, Any]]
    ) -> str:
        if not self.use_llm or not self.openai_api_key:
            if not stack_pass:
                return "Failed mandatory technical stack requirements."
            return f"Stack {stack_score:.2f}, Education {education_score:.2f}, Role Eval {evaluation_score:.2f}."
        try:
            evaluation_summary = "\n".join([
                f"- {result.question}: {'✓' if result.answer else '✗'} ({result.reasoning})"
                for result in evaluation_results
            ])
            stack_lines = []
            for cat, d in stack_details.items():
                mark = "✓" if d.get("met") else "✗"
                stack_lines.append(f"{mark} {cat}: need {d.get('min_match',1)}, matched {len(d.get('matched',[]))} ({', '.join(d.get('matched',[])) or '—'})")
            stack_block = "\n".join(stack_lines)
            prompt = f"""
            Provide a concise overall reasoning for why this student scored {(stack_score * 0.3 + education_score * 0.2 + evaluation_score * 0.5):.2f}
            for the {internship_req.job_title} role.

            Breakdown:
            - Technical Stack Coverage: {stack_score:.2f}
            - Education Match: {education_score:.2f}
            - Role Evaluation: {evaluation_score:.2f}

            Technical Stack:
            {stack_block}

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
        # 1) Technical Stack Gate + Coverage
        stack_pass, stack_cov, stack_details, flat_matched, flat_missing = self.check_technical_stack(
            student.skills, internship_req.technical_stack
        )

        if not stack_pass:
            return MatchResult(
                student=student,
                score=0.0,
                stack_pass=False,
                stack_coverage_score=stack_cov,
                education_match_score=0.0,
                evaluation_scores=[],
                overall_evaluation_score=0.0,
                matched_skills=flat_matched,
                missing_skills=flat_missing,
                overall_reasoning="Failed mandatory technical stack requirements.",
                stack_details=stack_details
            )

        # 2) Education
        education_score = self.calculate_education_match_score(
            student.education, student.degree, internship_req.relevant_degrees
        )

        # 3) Role evaluation
        evaluation_results, evaluation_score = self.evaluate_student_against_criteria(
            student, internship_req.evaluation_criteria
        )

        # Final weighted score (replaces old Skills 30% with Stack Coverage 30%)
        overall_score = (stack_cov * 0.3) + (education_score * 0.2) + (evaluation_score * 0.5)

        reasoning = self.generate_overall_reasoning(
            student, internship_req,
            stack_pass, stack_cov, education_score, evaluation_score,
            evaluation_results, stack_details
        )

        return MatchResult(
            student=student,
            score=overall_score,
            stack_pass=True,
            stack_coverage_score=stack_cov,
            education_match_score=education_score,
            evaluation_scores=evaluation_results,
            overall_evaluation_score=evaluation_score,
            matched_skills=flat_matched,
            missing_skills=flat_missing,
            overall_reasoning=reasoning,
            stack_details=stack_details
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
            for proj in project_list or []:
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
            for exp in exp_list or []:
                company = exp.get('company', '') if isinstance(exp, dict) else ''
                role = exp.get('role', '') if isinstance(exp, dict) else ''
                desc_html = exp.get('details', '') if isinstance(exp, dict) else ''
                desc = extract_text_from_html(desc_html)
                experience.append(Experience(company=company, role=role, details=desc))
        except:
            pass

        achievements = []
        extracurricular = []
        try:
            if 'achievements' in row and pd.notna(row['achievements']):
                achievements = ast.literal_eval(row['achievements'])
        except:
            achievements = []
        try:
            if 'extracurricular' in row and pd.notna(row['extracurricular']):
                extracurricular = ast.literal_eval(row['extracurricular'])
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
st.set_page_config(page_title="Enhanced Student-Job Matcher (Technical Stack)", layout="wide")
st.title("🎓 Student-Job Matching with Technical Stack Gate + Role Evaluation")

# Sidebar - CSV Upload
st.sidebar.header("Step 1: Upload Student CSV")
students_csv = st.sidebar.file_uploader("Students CSV", type="csv")
students = []
if students_csv:
    students = parse_student_csv(students_csv)
    st.sidebar.success(f"{len(students)} students loaded!")

# Sidebar - Job Roles with Technical Stack JSON
st.sidebar.header("Step 2: Define Job Roles (with Technical Stack)")

# Default examples (Data Science + SWE)
default_roles = [
    {
        "job_title": "Data Science Intern",
        "relevant_degrees": "Computer Science, Data Science, Mathematics, Statistics, Electrical Engineering",
        "evaluation_criteria": "Can implement ML algorithms?|0.25,Can do data preprocessing & analysis?|0.25,Understands metrics & model evaluation?|0.2,Can work with large datasets?|0.15,Communicates findings clearly?|0.15",
        "technical_stack": {
            "Programming Language": {
                "options": [
                    "python"
                ],
                "mandatory": True,
                "min_match": 1
            },
            "Libraries & Frameworks": {
                "options": [
                    "tensorflow",
                    "pytorch"
                ],
                "mandatory": True,
                "min_match": 1
            },
            "Data Libraries": {
                "options": [
                    "pandas",
                    "numpy"
                ],
                "mandatory": False,
                "min_match": 1
            },
            "Software & Platform": {
                "options": [
                    "jupyter notebook",
                    "Google Colab"
                ],
                "mandatory": False,
                "min_match": 1
            },
            "Database & API": {
                "options": [
                    "sql",
                    "nosql"
                ],
                "mandatory": True,
                "min_match": 1
            }
        }
    },
    {
        "job_title": "Software Engineer Intern",
        "relevant_degrees": "Computer Science, Software Engineering, Information Technology",
        "evaluation_criteria": "Can write clean, maintainable code?|0.3,Can solve algorithmic problems?|0.2,Can work with databases?|0.2,Can collaborate using Git?|0.15,Can learn quickly?|0.15",
        "technical_stack": {
            "Programming Language": {
                "options": [
                    "python",
                    "java",
                    "c++",
                    "javascript"
                ],
                "mandatory": True,
                "min_match": 1
            },
            "Web/Frameworks": {
                "options": [
                    "django",
                    "flask",
                    "fastapi",
                    "spring",
                    "nodejs",
                    "react"
                ],
                "mandatory": True,
                "min_match": 1
            },
            "Databases": {
                "options": [
                    "sql",
                    "nosql"
                ],
                "mandatory": True,
                "min_match": 1
            },
            "Tools & Platform (Optional)": {
                "options": [
                    "docker",
                    "kubernetes",
                    "aws",
                    "gcp",
                    "azure",
                    "git"
                ],
                "mandatory": False,
                "min_match": 1
            }
        }
    },
    {
        "job_title": "UI/UX Design Intern",
        "relevant_degrees": "Design, Arts, Human-Computer Interaction, Psychology",
        "evaluation_criteria": "Can create user-centered designs?|0.3,Can conduct user research?|0.25,Can create wireframes and prototypes?|0.2,Can collaborate with developers?|0.15,Can iterate based on feedback?|0.1",
        "technical_stack": {
            "Design Tools": {
                "options": [
                    "figma",
                    "sketch",
                    "adobe xd"
                ],
                "mandatory": True,
                "min_match": 1
            },
            "Prototyping (Optional)": {
                "options": [
                    "invision",
                    "principle",
                    "framer"
                ],
                "mandatory": False,
                "min_match": 1
            },
            "Research Tools (Optional)": {
                "options": [
                    "miro",
                    "usertesting",
                    "hotjar"
                ],
                "mandatory": False,
                "min_match": 1
            },
            "Basic Technical (Optional)": {
                "options": [
                    "html",
                    "css",
                    "javascript"
                ],
                "mandatory": False,
                "min_match": 1
            }
        }
    }
]

if "enhanced_roles" not in st.session_state:
    st.session_state["enhanced_roles"] = default_roles

enhanced_roles = st.session_state["enhanced_roles"]

def add_enhanced_role():
    enhanced_roles.append({
        "job_title": "",
        "relevant_degrees": "",
        "evaluation_criteria": "",
        "technical_stack": {}
    })
    st.session_state["enhanced_roles"] = enhanced_roles

def remove_enhanced_role(idx):
    del enhanced_roles[idx]
    st.session_state["enhanced_roles"] = enhanced_roles

for idx, role in enumerate(enhanced_roles):
    with st.sidebar.expander(f"Role {idx+1}: {role.get('job_title') or 'New Role'}", expanded=False):
        job_title = st.text_input(f"Job Title {idx+1}", value=role.get("job_title", ""), key=f"enh_job_title_{idx}")
        relevant_degrees = st.text_input(f"Relevant Degrees {idx+1}", value=role.get("relevant_degrees",""), key=f"enh_degrees_{idx}")
        evaluation_criteria = st.text_area(
            f"Evaluation Criteria {idx+1} (question|weight, comma or newline separated)",
            value=role.get("evaluation_criteria",""),
            key=f"enh_criteria_{idx}"
        )
        tech_stack_str = st.text_area(
            f"Technical Stack {idx+1} (JSON)",
            value=json.dumps(role.get("technical_stack", {}), indent=2),
            key=f"tech_stack_{idx}",
            help='Example:\n{\n  "Programming Language": {"options": ["python","r"], "mandatory": true, "min_match": 1},\n  "Libraries & Frameworks": {"options": ["tensorflow","pytorch"], "mandatory": true, "min_match": 1}\n}'
        )

        # persist edits
        enhanced_roles[idx].update({
            "job_title": job_title,
            "relevant_degrees": relevant_degrees,
            "evaluation_criteria": evaluation_criteria
        })
        try:
            enhanced_roles[idx]["technical_stack"] = json.loads(tech_stack_str)
        except Exception as e:
            st.warning(f"Invalid Technical Stack JSON for role {job_title or idx+1}: {e}")

        if st.button(f"Remove Role {idx+1}", key=f"enh_remove_{idx}"):
            remove_enhanced_role(idx)
            st.rerun()

st.sidebar.button("Add New Role", on_click=add_enhanced_role)

# Process job roles
job_roles: List[InternshipRequirement] = []
for role in enhanced_roles:
    if role.get("job_title"):
        # Parse evaluation criteria
        criteria = []
        crit_text = role.get("evaluation_criteria", "")
        if crit_text:
            lines = crit_text.split('\n')
            if len(lines) == 1 and ',' in crit_text:
                lines = crit_text.split(',')
            for line in lines:
                line = line.strip()
                if '|' in line:
                    try:
                        q, w = line.split('|', 1)
                        criteria.append(RoleEvaluation(question=q.strip(), weight=float(w.strip())))
                    except Exception:
                        st.warning(f"Invalid weight format in role {role['job_title']}: {line}")
                        continue

        job_roles.append(
            InternshipRequirement(
                job_title=role.get("job_title",""),
                company="",
                role_type=role.get("job_title",""),
                relevant_degrees=[d.strip() for d in role.get("relevant_degrees","").split(",") if d.strip()],
                evaluation_criteria=criteria,
                technical_stack=role.get("technical_stack", {}) or {}
            )
        )

# LLM Configuration
st.sidebar.header("Step 3: AI Configuration")
use_llm = st.sidebar.checkbox("Enable AI for detailed role evaluation", value=False)
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if use_llm and not openai_api_key:
    st.warning("Please provide your OpenAI API key to use AI evaluation.")


# After you've built `students`, `job_roles`, and `use_llm`:
current_fp = compute_config_fingerprint(students, job_roles, use_llm)

# Initialize session keys
if "last_run_fp" not in st.session_state:
    st.session_state["last_run_fp"] = None
if "run_clicked_at" not in st.session_state:
    st.session_state["run_clicked_at"] = None
if "matrix_rows" not in st.session_state:
    st.session_state["matrix_rows"] = None
if "all_matches" not in st.session_state:
    st.session_state["all_matches"] = None

st.sidebar.header("Step 4: Run")
if st.session_state["last_run_fp"] and st.session_state["last_run_fp"] != current_fp:
    st.sidebar.warning("Inputs changed since last run. Click **Run / Re-run Evaluation** to update results.")

run_btn = st.sidebar.button("▶ Run / Re-run Evaluation")

if run_btn:
    # mark this fingerprint as the latest confirmed input snapshot
    st.session_state["last_run_fp"] = current_fp
    st.session_state["run_clicked_at"] = time.time()
    # clear cache so we recompute with fresh inputs
    st.session_state["matrix_rows"] = None
    st.session_state["all_matches"] = None


# Main matching logic
matcher = EnhancedInternshipMatcher(openai_api_key=openai_api_key if use_llm else None, use_llm=use_llm)

ready_to_run = (
    students
    and job_roles
    and (not use_llm or (use_llm and openai_api_key))
    and st.session_state["last_run_fp"] == current_fp
    and st.session_state["run_clicked_at"] is not None
)

if ready_to_run:
    st.header("Matching Results (Technical Stack → Education → Role Evaluation)")

    # Use cached compute when available; recompute if cache cleared by button
    if st.session_state["matrix_rows"] is None or st.session_state["all_matches"] is None:
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
                    row[f"{role.job_title} - Stack Pass"] = "Yes" if result.stack_pass else "No"
                    row[f"{role.job_title} - Stack Coverage"] = f"{result.stack_coverage_score:.2f}"
                    row[f"{role.job_title} - Education"] = f"{result.education_match_score:.2f}"
                    row[f"{role.job_title} - Role Eval"] = f"{result.overall_evaluation_score:.2f}"

                    unmet_cats = [c for c, d in result.stack_details.items() if not d.get("met")]
                    row[f"{role.job_title} - Unmet Stack Categories"] = ", ".join(unmet_cats) if unmet_cats else ""

                    row[f"{role.job_title} - Matched Stack Skills"] = ", ".join(result.matched_skills)
                    row[f"{role.job_title} - Missing Stack Skills"] = ", ".join(result.missing_skills)
                    row[f"{role.job_title} - Overall Reasoning"] = result.overall_reasoning

                    processed += 1
                    progress_bar.progress(processed / total_combinations)
                    if use_llm:
                        time.sleep(0.1)

                all_matches[student.email] = student_results
                matrix_rows.append(row)

            progress_bar.empty()
            st.session_state["all_matches"] = all_matches
            st.session_state["matrix_rows"] = matrix_rows

    # Display results
    df_results = pd.DataFrame(st.session_state["matrix_rows"])
    st.dataframe(df_results, use_container_width=True)


    all_matches = st.session_state.get("all_matches") or {}
    if not all_matches:
        st.info("No results cached yet. Click **Run / Re-run Evaluation** after setting inputs.")
        st.stop()  # avoid NameError / KeyError below
    # Detailed student analysis
    st.header("Detailed Student Analysis")

    emails_with_results = list(all_matches.keys())
    if not emails_with_results:
        st.info("No evaluated students found. Please re-run evaluation.")
        st.stop()

    # Optional: keep dropdown in sync with current students
    valid_current_emails = [s.email for s in students if s.email in all_matches]
    options = valid_current_emails or emails_with_results  # fallback to all evaluated

    selected_email = st.selectbox(
        "Select a student for detailed analysis",
        [s.email for s in students],
        format_func=lambda email: f"{next((s.name for s in students if s.email == email), email)} ({email})"
    )

    # Safety guard
    if selected_email not in all_matches:
        st.warning("Selected student has no results. Please re-run evaluation.")
        st.stop()

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
            role_scores = [(role.job_title, selected_results[role.job_title].score) for role in job_roles]
            role_scores.sort(key=lambda x: x[1], reverse=True)
            st.write("**Best Role Matches:**")
            for i, (role_name, score) in enumerate(role_scores[:3]):
                st.write(f"{i+1}. {role_name}: {score:.3f}")

        selected_role_name = st.selectbox("Select role for detailed breakdown", [role.job_title for role in job_roles])
        if selected_role_name:
            result = selected_results[selected_role_name]
            st.subheader(f"Analysis for {selected_role_name}")

            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Overall Score", f"{result.score:.3f}")
            with c2: st.metric("Stack Coverage", f"{result.stack_coverage_score:.3f}")
            with c3: st.metric("Education Match", f"{result.education_match_score:.3f}")
            with c4: st.metric("Role Evaluation", f"{result.overall_evaluation_score:.3f}")

            # Technical stack breakdown
            st.write("**Technical Stack Breakdown:**")
            for cat, d in result.stack_details.items():
                status = "✅ Met" if d.get("met") else "❌ Not Met"
                st.write(f"- **{cat}** ({'Mandatory' if d.get('mandatory') else 'Optional'}, need {d.get('min_match',1)}): {status}")
                st.write(f"  - Matched: {', '.join(d.get('matched', [])) or '—'}")
                st.write(f"  - Missing: {', '.join(d.get('missing', [])) or '—'}")

            if result.stack_pass and result.evaluation_scores:
                st.write("**Role-Specific Evaluation:**")
                for eval_result in result.evaluation_scores:
                    status = "✅ YES" if eval_result.answer else "❌ NO"
                    st.write(f"**{eval_result.question}** {status} (Weight: {eval_result.weight})")
                    st.write(f"*Reasoning:* {eval_result.reasoning}")
                    st.write("---")

            # Overall reasoning
            if result.overall_reasoning:
                st.write("**Overall Assessment:**")
                st.info(result.overall_reasoning)

    # Export
    st.header("Export Results")
    csv_content = df_results.to_csv(index=False)
    st.download_button(
        label="📊 Download Results as CSV",
        data=csv_content,
        file_name=f"student_job_matching_results_{time.strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# else:
else:
    # Guide the user to click the button or finish inputs
    if not students:
        st.info("Upload student CSV to begin.")
    elif not job_roles:
        st.info("Define at least one job role with a technical stack.")
    elif use_llm and not openai_api_key:
        st.info("Enter OpenAI API key or disable AI evaluation.")
    else:
        st.info("Ready to run. Click **Run / Re-run Evaluation** in the sidebar.")
# Instructions
with st.expander("📋 Instructions & CSV Format"):
    st.markdown("""
    **Required CSV Columns for Students:**
    - `StudentName` or `name`
    - `Email Address` or `email`
    - `education`
    - `degree`
    - `skill` (Python list)
    - `projects` (list of dicts with 'name' and 'description')
    - `experience` (list of dicts with 'company', 'role', 'details')
    - `achievements` (optional list)
    - `extracurricular` (optional list)

    **Job Role Configuration:**
    - **Relevant Degrees**: Comma-separated values
    - **Evaluation Criteria**: "Question?|weight" per line or comma-separated (weights sum to 1.0)
    - **Technical Stack (JSON)**: Category-based, each with:
      - `options`: list of acceptable skills
      - `mandatory`: true/false (gate)
      - `min_match`: integer (≥1)
      Example:
      ```
      {
        "Programming Language": {"options": ["python", "r"], "mandatory": true, "min_match": 1},
        "Libraries & Frameworks": {"options": ["tensorflow", "pytorch"], "mandatory": true, "min_match": 1},
        "Data Libraries": {"options": ["pandas", "numpy"], "mandatory": false, "min_match": 2},
        "Software & Platform": {"options": ["jupyter notebook"], "mandatory": false, "min_match": 1},
        "Database & API": {"options": ["sql", "nosql"], "mandatory": true, "min_match": 1}
      }
      ```

    **Scoring System:**
    - Technical Stack Coverage: 30% (only computed if all mandatory categories pass)
    - Education Match: 20%
    - Role Evaluation: 50%
    - If any mandatory stack category fails → Overall score = 0 and evaluation is skipped.
    """)
st.markdown("---")
st.caption("Technical Stack–gated Matching with Education & Role-Based Evaluation 🚀")
