import streamlit as st
import pandas as pd
import ast
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
import re
import time
import json
import hashlib

def compute_config_fingerprint(students, job_roles) -> str:
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
            "technical_stack": r.technical_stack,
        }
        for r in job_roles
    ]
    payload = {
        "students": stu_repr,
        "roles": roles_repr,
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
class InternshipRequirement:
    job_title: str
    company: str
    role_type: str
    relevant_degrees: List[str]
    technical_stack: Dict[str, Dict[str, Any]]

@dataclass
class SkillMatch:
    """Represents a matched skill with its hierarchy information"""
    skill_name: str
    parent_skill: str = ""
    is_parent: bool = False
    matched_children: List[str] = None
    
    def __post_init__(self):
        if self.matched_children is None:
            self.matched_children = []

@dataclass
class MatchResult:
    student: Student
    score: float
    stack_pass: bool
    stack_coverage_score: float
    education_match_score: float
    matched_skills: List[SkillMatch]  # Updated to use SkillMatch objects
    missing_skills: List[str]
    overall_reasoning: str
    stack_details: Dict[str, Dict[str, Any]]

# ---- Simplified Matcher Class ----
class SimplifiedInternshipMatcher:
    def __init__(self, openai_api_key: str = None, use_llm=False):
        self.openai_api_key = openai_api_key
        self.use_llm = use_llm
        
        # Load skill hierarchy
        self.skills_data = self.load_skills("skill_taxonomy_565_skills.json")
        self.child_to_parent_map = self.build_child_to_parent(self.skills_data)
        
        # Synonyms / expansions to normalize skills
        self.skill_synonyms = {
            # Keep your existing synonyms here if needed
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

    def _norm(self, s: str) -> str:
        return s.strip().lower()

    def load_skills(self, path: str) -> List[dict]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            st.warning(f"Skill taxonomy file '{path}' not found. Skill hierarchy features will be disabled.")
            return []
        except Exception as e:
            st.warning(f"Error loading skill taxonomy: {e}. Skill hierarchy features will be disabled.")
            return []

    def build_child_to_parent(self, skills: List[dict]) -> Dict[str, dict]:
        """
        Builds a mapping from child label/name/alias -> parent skill object.
        Uses both:
          1) child's hierarchy.parent (id or name/alias), and
          2) parent's hierarchy.children (string labels).
        Also honors aliases for both child and parent.
        """
        if not skills:
            return {}
            
        by_id: Dict[str, dict] = {}
        by_name: Dict[str, dict] = {}
        by_alias: Dict[str, dict] = {}

        for sk in skills:
            if sk.get("id"):
                by_id[sk["id"]] = sk
            if sk.get("name"):
                by_name[self._norm(sk["name"])] = sk
            for a in (sk.get("aliases") or []):
                by_alias[self._norm(a)] = sk

        child_to_parent: Dict[str, dict] = {}

        # Pass 1: child's explicit hierarchy.parent (could be id or name/alias)
        for sk in skills:
            parent_ref = (sk.get("hierarchy") or {}).get("parent")
            if not parent_ref:
                continue
            parent = None
            if isinstance(parent_ref, str):
                parent = by_id.get(parent_ref) or by_name.get(self._norm(parent_ref)) or by_alias.get(self._norm(parent_ref))
            if parent:
                # map child's name + aliases to parent
                child_to_parent[self._norm(sk["name"])] = parent
                for a in (sk.get("aliases") or []):
                    child_to_parent[self._norm(a)] = parent

        # Pass 2: parent's hierarchy.children (usually string labels)
        for parent in skills:
            children = (parent.get("hierarchy") or {}).get("children") or []
            for child_label in children:
                if not isinstance(child_label, str):
                    continue
                nlabel = self._norm(child_label)
                # map the label directly
                child_to_parent[nlabel] = parent
                # if that label corresponds to a concrete skill (name/alias), map those too
                child_skill = by_name.get(nlabel) or by_alias.get(nlabel)
                if child_skill:
                    child_to_parent[self._norm(child_skill["name"])] = parent
                    for a in (child_skill.get("aliases") or []):
                        child_to_parent[self._norm(a)] = parent

        return child_to_parent

    def get_skill_hierarchy_info(self, skill_name: str) -> Tuple[str, bool, List[str]]:
        """
        Returns tuple of (parent_name, is_parent_skill, children_list)
        """
        skill_lower = self._norm(skill_name)
        parent_skill = self.child_to_parent_map.get(skill_lower)
        parent_name = parent_skill.get("name", "") if parent_skill else ""
        
        # Check if this skill is itself a parent
        is_parent = False
        children = []
        for skill_data in self.skills_data:
            if self._norm(skill_data.get("name", "")) == skill_lower:
                hierarchy = skill_data.get("hierarchy", {})
                children = hierarchy.get("children", [])
                is_parent = len(children) > 0
                break
        
        return parent_name, is_parent, children

    def create_skill_match_objects(self, matched_skill_names: List[str]) -> List[SkillMatch]:
        """Convert list of skill names to SkillMatch objects with hierarchy info"""
        skill_matches = []
        processed_skills = set()
        
        for skill_name in matched_skill_names:
            if skill_name in processed_skills:
                continue
                
            parent_name, is_parent, children = self.get_skill_hierarchy_info(skill_name)
            
            # Find any children that are also in the matched skills
            matched_children = []
            if is_parent:
                for child in children:
                    if child in matched_skill_names and child != skill_name:
                        matched_children.append(child)
                        processed_skills.add(child)
            
            skill_match = SkillMatch(
                skill_name=skill_name,
                parent_skill=parent_name,
                is_parent=is_parent,
                matched_children=matched_children
            )
            
            skill_matches.append(skill_match)
            processed_skills.add(skill_name)
        
        return skill_matches

    def format_skill_hierarchy_display(self, skill_matches: List[SkillMatch]) -> str:
        """Format skill matches for display with hierarchy information"""
        if not skill_matches:
            return ""
        
        display_parts = []
        for skill_match in skill_matches:
            if skill_match.is_parent and skill_match.matched_children:
                # Parent with children: "Python (→ Django, Flask)"
                children_str = ", ".join(skill_match.matched_children)
                display_parts.append(f"{skill_match.skill_name} (→ {children_str})")
            elif skill_match.parent_skill and not skill_match.is_parent:
                # Child with parent: "Django (← Python)"
                display_parts.append(f"{skill_match.skill_name} (← {skill_match.parent_skill})")
            else:
                # Standalone skill
                display_parts.append(skill_match.skill_name)
        
        return ", ".join(display_parts)

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

    def calculate_skill_match_with_llm(
        self, 
        student_skills: List[str], 
        required_skills: List[str],
        use_llm: bool = False,
        openai_api_key: str = None
    ) -> Tuple[float, List[str], List[str], List[str]]:
        """
        Enhanced skill matching that can use LLM for semantic understanding
        Returns: (match_score, actual_student_matched_skills, missing_required_skills, llm_matched_skills)
        """
        # Keep original student skills for display
        student_skills_lower = [s.lower().strip() for s in student_skills if s.strip()]
        required_skills_lower = [s.lower().strip() for s in required_skills if s.strip()]
        
        # Normalize for matching logic
        student_norm = set(self.normalize_skills(student_skills))
        required_norm = set(self.normalize_skills(required_skills))
        
        # Find actual student skills that match (not the expanded normalized ones)
        actual_matched = []
        for student_skill in student_skills_lower:
            student_expanded = set(self.normalize_skills([student_skill]))
            if student_expanded.intersection(required_norm):
                actual_matched.append(student_skill)
        
        # Find which required skills are still missing
        matched_requirements = set()
        for req_skill in required_skills_lower:
            req_expanded = set(self.normalize_skills([req_skill]))
            if req_expanded.intersection(student_norm):
                matched_requirements.add(req_skill)
        
        missing_requirements = [r for r in required_skills_lower if r not in matched_requirements]
        llm_matched = []
        
        # If LLM is available, enhance the matching
        if use_llm and openai_api_key and missing_requirements:
            try:
                import openai
                openai.api_key = openai_api_key
                
                student_skills_str = ", ".join(student_skills)
                missing_skills_str = ", ".join(missing_requirements)
                
                prompt = f"""
                Analyze if the student's skills can satisfy the missing requirements through related/transferable skills:
                
                Student's Skills: {student_skills_str}
                Missing Required Skills: {missing_skills_str}
                
                For each missing skill, determine if the student has related skills that demonstrate similar capabilities.
                Consider:
                - Similar technologies (e.g., MySQL experience for SQL requirement)
                - Related frameworks (e.g., React experience for frontend requirement)
                - Transferable skills (e.g., Python for programming requirement)
                - Academic knowledge that applies
                
                Respond in this format:
                STUDENT_SKILL->REQUIREMENT: actual_student_skill->missing_requirement
                
                Example: MySQL->SQL, React->frontend development
                
                If no additional matches found, respond with:
                ADDITIONAL_MATCHES: none
                """
                
                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.3
                )
                
                content = response.choices[0].message.content.strip()
                
                # Parse LLM response for student_skill->requirement pairs
                if "none" not in content.lower():
                    lines = content.split('\n')
                    for line in lines:
                        if '->' in line and ':' in line:
                            try:
                                # Extract the mapping after the colon
                                mapping = line.split(':', 1)[1].strip()
                                if '->' in mapping:
                                    student_part, req_part = mapping.split('->', 1)
                                    student_skill = student_part.strip()
                                    req_skill = req_part.strip()
                                    
                                    # Check if this student skill exists and requirement was missing
                                    if any(s.lower() == student_skill.lower() for s in student_skills):
                                        if any(r.lower() == req_skill.lower() for r in missing_requirements):
                                            llm_matched.append(f"{student_skill} (satisfies {req_skill})")
                                            # Remove from missing
                                            missing_requirements = [r for r in missing_requirements if r.lower() != req_skill.lower()]
                            except:
                                continue
                
                time.sleep(0.2)  # Rate limiting
                
            except Exception as e:
                print(f"LLM skill matching failed: {e}")
        
        # Calculate match score
        total_required = len(required_skills_lower) if required_skills_lower else 1
        total_matched = len(actual_matched) + len(llm_matched)
        match_score = min(total_matched / total_required, 1.0)
        
        return match_score, actual_matched, missing_requirements, llm_matched

    def check_technical_stack(
        self,
        student_skills: List[str],
        technical_stack: Dict[str, Dict[str, Any]],
        use_llm: bool = False,
        openai_api_key: str = None
    ) -> Tuple[bool, float, Dict[str, Dict[str, Any]], List[SkillMatch], List[str]]:
        """
        Enhanced technical stack checking with optional LLM skill matching
        Returns SkillMatch objects instead of plain strings
        """
        details = {}
        all_matched_names = []
        all_missing = []

        if not technical_stack:
            return True, 1.0, {}, [], []

        categories = list(technical_stack.keys())
        if not categories:
            return True, 1.0, {}, [], []

        passed = True
        category_scores = []

        for category, req in technical_stack.items():
            options = [str(o).strip() for o in req.get("options", []) if str(o).strip()]
            mandatory = bool(req.get("mandatory", False))
            min_match = max(int(req.get("min_match", 1)), 1)

            # Use enhanced skill matching for this category
            match_score, actual_matched, missing_reqs, llm_matched = self.calculate_skill_match_with_llm(
                student_skills, options, use_llm, openai_api_key
            )
            
            # Combine actual matches and LLM matches for display
            all_category_matches = actual_matched + llm_matched
            total_matches = len(actual_matched) + len(llm_matched)
            
            met = total_matches >= min_match
            
            if mandatory and not met:
                passed = False

            details[category] = {
                "required": options,
                "min_match": min_match,
                "matched": all_category_matches,  # Show actual student skills, not normalized
                "missing": missing_reqs,  # Show missing requirements, not expanded
                "mandatory": mandatory,
                "met": met,
                "llm_enhanced": use_llm and openai_api_key is not None and len(llm_matched) > 0
            }

            # Coverage score based on actual matches vs minimum required
            denom = float(max(min_match, 1))
            category_score = min(total_matches, min_match) / denom
            category_scores.append(category_score)

            all_matched_names.extend(all_category_matches)
            if not met:
                all_missing.extend([f"{category}: {m}" for m in missing_reqs])

        coverage_score = sum(category_scores) / len(category_scores) if category_scores else 1.0
        
        # Convert matched skill names to SkillMatch objects
        skill_matches = self.create_skill_match_objects(all_matched_names)
        
        return passed, coverage_score, details, skill_matches, all_missing

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

    def generate_overall_reasoning(
        self, student: Student, internship_req: InternshipRequirement,
        stack_pass: bool, stack_score: float, education_score: float,
        stack_details: Dict[str, Dict[str, Any]]
    ) -> str:
        """Generate simple reasoning without LLM"""
        if not stack_pass:
            unmet_categories = [cat for cat, d in stack_details.items() if d.get("mandatory") and not d.get("met")]
            return f"Failed mandatory technical requirements in: {', '.join(unmet_categories)}"
        
        # Determine scoring logic used
        if len(internship_req.relevant_degrees) == 0:
            final_score = (stack_score * 0.6) + (1.0 * 0.4)  # No degree requirements
            scoring_note = "No specific degree requirements"
        elif education_score == 0:
            final_score = (stack_score * 0.8) + (education_score * 0.2)  # Adjusted weights
            scoring_note = "Adjusted scoring due to education mismatch"
        else:
            final_score = (stack_score * 0.6) + (education_score * 0.4)  # Normal scoring
            scoring_note = "Standard scoring"
        
        if final_score >= 0.8:
            level = "Excellent match"
        elif final_score >= 0.6:
            level = "Good match"
        elif final_score >= 0.4:
            level = "Fair match"
        else:
            level = "Limited match"
            
        # Highlight strengths and gaps
        strengths = []
        gaps = []
        
        if stack_score >= 0.8:
            strengths.append("strong technical skills")
        elif stack_score >= 0.6:
            strengths.append("good technical skills")
        elif stack_score < 0.5:
            gaps.append("technical skills")
            
        if education_score >= 0.8:
            strengths.append("relevant educational background")
        elif education_score == 0:
            if len(internship_req.relevant_degrees) > 0:
                gaps.append("educational alignment")
            # Don't mention education gap if no requirements specified
        elif education_score < 0.5:
            gaps.append("educational alignment")
        
        reasoning = f"{level} (Score: {final_score:.2f}). "
        if strengths:
            reasoning += f"Strengths: {', '.join(strengths)}. "
        if gaps:
            reasoning += f"Areas for improvement: {', '.join(gaps)}. "
        if education_score == 0 and len(internship_req.relevant_degrees) > 0:
            reasoning += f"({scoring_note} - technical skills weighted higher due to education mismatch). "
            
        return reasoning.strip()

    def match_student_to_internship(self, student: Student, internship_req: InternshipRequirement) -> MatchResult:
        # 1) Technical Stack Gate + Coverage (with optional LLM enhancement)
        stack_pass, stack_cov, stack_details, skill_matches, flat_missing = self.check_technical_stack(
            student.skills, internship_req.technical_stack, self.use_llm, self.openai_api_key
        )

        # 2) Education
        education_score = self.calculate_education_match_score(
            student.education, student.degree, internship_req.relevant_degrees
        )

        # 3) Adjusted scoring logic to handle zero education score better
        if len(internship_req.relevant_degrees) == 0:
            # No specific degree requirements - give full education credit
            education_score = 1.0
            overall_score = (stack_cov * 0.6) + (education_score * 0.4)
        elif education_score == 0:
            # Has degree requirements but no match - reduce education weight, increase stack weight
            # This prevents complete failure when technical skills are strong
            overall_score = (stack_cov * 0.8) + (education_score * 0.2)
        else:
            # Normal case with some education match
            overall_score = (stack_cov * 0.6) + (education_score * 0.4)

        reasoning = self.generate_overall_reasoning(
            student, internship_req, stack_pass, stack_cov, education_score, stack_details
        )

        return MatchResult(
            student=student,
            score=overall_score,
            stack_pass=True,
            stack_coverage_score=stack_cov,
            education_match_score=education_score,
            matched_skills=skill_matches,  # Now SkillMatch objects
            missing_skills=flat_missing,
            overall_reasoning=reasoning,
            stack_details=stack_details
        )

# ---- CSV Parser ----
def extract_text_from_html(html_str):
    if not html_str or not isinstance(html_str, str):
        return ""
    return BeautifulSoup(html_str, "html.parser").get_text(separator=" ").strip()

def _norm(s: str) -> str:
    return s.strip().lower()

def load_skills(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_child_to_parent(skills: List[dict]) -> Dict[str, dict]:
    """
    Builds a mapping from child label/name/alias -> parent skill object.
    Uses both:
      1) child's hierarchy.parent (id or name/alias), and
      2) parent's hierarchy.children (string labels).
    Also honors aliases for both child and parent.
    """
    by_id: Dict[str, dict] = {}
    by_name: Dict[str, dict] = {}
    by_alias: Dict[str, dict] = {}

    for sk in skills:
        if sk.get("id"):
            by_id[sk["id"]] = sk
        if sk.get("name"):
            by_name[_norm(sk["name"])] = sk
        for a in (sk.get("aliases") or []):
            by_alias[_norm(a)] = sk

    child_to_parent: Dict[str, dict] = {}

    # Pass 1: child's explicit hierarchy.parent (could be id or name/alias)
    for sk in skills:
        parent_ref = (sk.get("hierarchy") or {}).get("parent")
        if not parent_ref:
            continue
        parent = None
        if isinstance(parent_ref, str):
            parent = by_id.get(parent_ref) or by_name.get(_norm(parent_ref)) or by_alias.get(_norm(parent_ref))
        if parent:
            # map child's name + aliases to parent
            child_to_parent[_norm(sk["name"])] = parent
            for a in (sk.get("aliases") or []):
                child_to_parent[_norm(a)] = parent

    # Pass 2: parent's hierarchy.children (usually string labels)
    for parent in skills:
        children = (parent.get("hierarchy") or {}).get("children") or []
        for child_label in children:
            if not isinstance(child_label, str):
                continue
            nlabel = _norm(child_label)
            # map the label directly
            child_to_parent[nlabel] = parent
            # if that label corresponds to a concrete skill (name/alias), map those too
            child_skill = by_name.get(nlabel) or by_alias.get(nlabel)
            if child_skill:
                child_to_parent[_norm(child_skill["name"])] = parent
                for a in (child_skill.get("aliases") or []):
                    child_to_parent[_norm(a)] = parent

    return child_to_parent

def parent_names_for_children(
    skills_json_path: str,
    child_names: List[str],
    empty_return: str = ""
) -> List[str]:
    """
    Returns a list of parent skill NAMES aligned with child_names.
    If no parent found (root or unknown), returns empty_return (default "") for that position.
    """
    try:
        skills = load_skills(skills_json_path)
        child_to_parent = build_child_to_parent(skills)
        result: List[str] = []
        for raw in child_names:
            parent = child_to_parent.get(_norm(raw))
            result.append(parent.get("name") if parent else empty_return)
        return result
    except:
        return [""] * len(child_names)

def parse_student_csv(file_obj):
    df = pd.read_csv(file_obj)
    students = []
    for _, row in df.iterrows():
        # Parse skills
        try:
            skills = ast.literal_eval(row['skill'])
            # add parent skills
            parents = parent_names_for_children( "skill_taxonomy_565_skills.json",skills)
            skills += parents
            skills = list(set(skills))
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
st.set_page_config(page_title="Simplified Student-Job Matcher", layout="wide")
st.title("🎓 Simplified Student-Job Matching (Technical Stack + Education Only)")

# Sidebar - CSV Upload
st.sidebar.header("Step 1: Upload Student CSV")
students_csv = st.sidebar.file_uploader("Students CSV", type="csv")
students = []
if students_csv:
    students = parse_student_csv(students_csv)
    st.sidebar.success(f"{len(students)} students loaded!")

# Sidebar - Job Roles with Technical Stack JSON
st.sidebar.header("Step 2: Define Job Roles")

# Default examples
default_roles = [
    {
        "job_title": "Python Backend Intern",
        "relevant_degrees": "Computer Science, Software Engineering, Information Technology, Data Science",
        "evaluation_criteria": "Can design RESTful APIs?|0.3,Can work with databases and ORMs?|0.25,Can implement authentication & security?|0.2,Can write clean, maintainable code?|0.15,Can handle testing and debugging?|0.1",
        "technical_stack": {
            "Programming Language": {
                "options": [
                    "python"
                ],
                "mandatory": True,
                "min_match": 1
            },
            "Frameworks": {
                "options": [
                    "django",
                    "flask",
                    "fastapi"
                ],
                "mandatory": True,
                "min_match": 1
            },
            "Databases": {
                "options": [
                    "postgresql",
                    "mysql",
                    "sqlite",
                    "mongodb"
                ],
                "mandatory": True,
                "min_match": 1
            },
            "Testing & Quality": {
                "options": [
                    "pytest",
                    "unittest",
                    "black",
                    "flake8"
                ],
                "mandatory": False,
                "min_match": 1
            },
            "Authentication & Security": {
                "options": [
                    "jwt",
                    "oauth",
                    "bcrypt",
                    "django auth"
                ],
                "mandatory": False,
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
        "job_title": "MLOps Intern",
        "relevant_degrees": "Computer Science, Data Science, Machine Learning, Software Engineering, Statistics",
        "evaluation_criteria": "Can deploy ML models to production?|0.25,Can set up CI/CD pipelines for ML?|0.2,Can monitor model performance?|0.2,Can work with cloud platforms?|0.15,Can manage data pipelines?|0.15,Can collaborate with data scientists?|0.05",
        "technical_stack": {
            "Programming Language": {
                "options": [
                    "python",
                    "bash",
                    "sql"
                ],
                "mandatory": True,
                "min_match": 2
            },
            "ML Frameworks": {
                "options": [
                    "tensorflow",
                    "pytorch",
                    "scikit-learn",
                    "xgboost"
                ],
                "mandatory": True,
                "min_match": 1
            },
            "MLOps Tools": {
                "options": [
                    "mlflow",
                    "kubeflow",
                    "airflow",
                    "dvc",
                    "wandb"
                ],
                "mandatory": True,
                "min_match": 1
            },
            "Cloud Platforms": {
                "options": [
                    "aws",
                    "gcp",
                    "azure"
                ],
                "mandatory": True,
                "min_match": 1
            },
            "Containerization & Orchestration": {
                "options": [
                    "docker",
                    "kubernetes",
                    "helm"
                ],
                "mandatory": True,
                "min_match": 1
            },
            "Monitoring & Observability (Optional)": {
                "options": [
                    "prometheus",
                    "grafana",
                    "elasticsearch",
                    "kibana"
                ],
                "mandatory": False,
                "min_match": 1
            },
            "Version Control & CI/CD (Optional)": {
                "options": [
                    "git",
                    "jenkins",
                    "github actions",
                    "gitlab ci"
                ],
                "mandatory": False,
                "min_match": 1
            }
        }
    },
    {
        "job_title": "MERN Stack Intern",
        "relevant_degrees": "Computer Science, Software Engineering, Information Technology, Web Development",
        "evaluation_criteria": "Can build full-stack web applications?|0.3,Can create responsive UIs with React?|0.25,Can design RESTful APIs?|0.2,Can work with databases?|0.15,Can implement authentication?|0.1",
        "technical_stack": {
            "Frontend": {
                "options": [
                    "react",
                    "javascript",
                    "html",
                    "css"
                ],
                "mandatory": True,
                "min_match": 3
            },
            "Backend": {
                "options": [
                    "nodejs",
                    "express"
                ],
                "mandatory": True,
                "min_match": 2
            },
            "Database": {
                "options": [
                    "mongodb",
                    "mongoose"
                ],
                "mandatory": True,
                "min_match": 1
            },
            "State Management": {
                "options": [
                    "redux",
                    "context api",
                    "zustand"
                ],
                "mandatory": True,
                "min_match": 1
            },
            "Styling & UI": {
                "options": [
                    "bootstrap",
                    "tailwind css",
                    "material-ui",
                    "styled-components"
                ],
                "mandatory": False,
                "min_match": 1
            },
            "Tools & Others (Optional)": {
                "options": [
                    "git",
                    "postman",
                    "jwt",
                    "bcrypt",
                    "cors"
                ],
                "mandatory": False,
                "min_match": 1
            }
        }
    },
    {
        "job_title": "Next.js Intern",
        "relevant_degrees": "Computer Science, Software Engineering, Information Technology, Web Development",
        "evaluation_criteria": "Can build SSR/SSG applications?|0.3,Can optimize web performance?|0.25,Can implement routing and navigation?|0.2,Can work with APIs and data fetching?|0.15,Can deploy applications?|0.1",
        "technical_stack": {
            "Core Framework": {
                "options": [
                    "nextjs",
                    "react",
                    "javascript",
                    "typescript"
                ],
                "mandatory": True,
                "min_match": 3
            },
            "Styling & UI": {
                "options": [
                    "tailwind css",
                    "css modules",
                    "styled-components",
                    "sass"
                ],
                "mandatory": True,
                "min_match": 1
            },
            "Data Fetching": {
                "options": [
                    "swr",
                    "react query",
                    "axios",
                    "fetch api"
                ],
                "mandatory": True,
                "min_match": 1
            },
            "Backend Integration": {
                "options": [
                    "api routes",
                    "prisma",
                    "mongodb",
                    "postgresql"
                ],
                "mandatory": False,
                "min_match": 1
            },
            "Deployment & Performance": {
                "options": [
                    "vercel",
                    "netlify",
                    "aws",
                    "lighthouse"
                ],
                "mandatory": False,
                "min_match": 1
            },
            "Tools & Others (Optional)": {
                "options": [
                    "git",
                    "eslint",
                    "prettier",
                    "next-auth"
                ],
                "mandatory": False,
                "min_match": 1
            }
        }
    },
    {
        "job_title": "Node.js Backend Intern",
        "relevant_degrees": "Computer Science, Software Engineering, Information Technology, Backend Development",
        "evaluation_criteria": "Can design RESTful APIs?|0.3,Can work with databases efficiently?|0.25,Can implement authentication & authorization?|0.2,Can handle error management?|0.15,Can write unit tests?|0.1",
        "technical_stack": {
            "Core Backend": {
                "options": [
                    "nodejs",
                    "express",
                    "javascript",
                    "typescript"
                ],
                "mandatory": True,
                "min_match": 3
            },
            "Databases": {
                "options": [
                    "mongodb",
                    "postgresql",
                    "mysql",
                    "redis"
                ],
                "mandatory": True,
                "min_match": 1
            },
            "Testing & Validation": {
                "options": [
                    "jest",
                    "mocha",
                    "joi",
                    "express-validator"
                ],
                "mandatory": False,
                "min_match": 1
            },
            "Tools & Middleware (Optional)": {
                "options": [
                    "cors",
                    "helmet",
                    "morgan",
                    "multer",
                    "nodemon"
                ],
                "mandatory": False,
                "min_match": 1
            },
            "Deployment & DevOps (Optional)": {
                "options": [
                    "docker",
                    "aws",
                    "heroku",
                    "pm2"
                ],
                "mandatory": False,
                "min_match": 1
            }
        }
    }
]

if "simplified_roles" not in st.session_state:
    st.session_state["simplified_roles"] = default_roles

simplified_roles = st.session_state["simplified_roles"]

def add_simplified_role():
    simplified_roles.append({
        "job_title": "",
        "relevant_degrees": "",
        "technical_stack": {}
    })
    st.session_state["simplified_roles"] = simplified_roles

def remove_simplified_role(idx):
    del simplified_roles[idx]
    st.session_state["simplified_roles"] = simplified_roles

for idx, role in enumerate(simplified_roles):
    with st.sidebar.expander(f"Role {idx+1}: {role.get('job_title') or 'New Role'}", expanded=False):
        job_title = st.text_input(f"Job Title {idx+1}", value=role.get("job_title", ""), key=f"simp_job_title_{idx}")
        relevant_degrees = st.text_input(f"Relevant Degrees {idx+1}", value=role.get("relevant_degrees",""), key=f"simp_degrees_{idx}")
        tech_stack_str = st.text_area(
            f"Technical Stack {idx+1} (JSON)",
            value=json.dumps(role.get("technical_stack", {}), indent=2),
            key=f"simp_tech_stack_{idx}",
            help='Example:\n{\n  "Programming Language": {"options": ["python","r"], "mandatory": true, "min_match": 1},\n  "Libraries & Frameworks": {"options": ["tensorflow","pytorch"], "mandatory": true, "min_match": 1}\n}'
        )

        # persist edits
        simplified_roles[idx].update({
            "job_title": job_title,
            "relevant_degrees": relevant_degrees
        })
        try:
            simplified_roles[idx]["technical_stack"] = json.loads(tech_stack_str)
        except Exception as e:
            st.warning(f"Invalid Technical Stack JSON for role {job_title or idx+1}: {e}")

        if st.button(f"Remove Role {idx+1}", key=f"simp_remove_{idx}"):
            remove_simplified_role(idx)
            st.rerun()

st.sidebar.button("Add New Role", on_click=add_simplified_role)

# LLM Configuration (Optional)
st.sidebar.header("Step 3: AI Configuration (Optional)")
use_llm = st.sidebar.checkbox("Enable AI for enhanced skill matching", value=False, 
                             help="Uses AI to find transferable/related skills beyond exact matches")
openai_api_key = None
if use_llm:
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.sidebar.warning("Please provide your OpenAI API key to use AI-enhanced skill matching.")
        use_llm = False

job_roles: List[InternshipRequirement] = []
for role in simplified_roles:
    if role.get("job_title"):
        job_roles.append(
            InternshipRequirement(
                job_title=role.get("job_title",""),
                company="",
                role_type=role.get("job_title",""),
                relevant_degrees=[d.strip() for d in role.get("relevant_degrees","").split(",") if d.strip()],
                technical_stack=role.get("technical_stack", {}) or {}
            )
        )

# Compute fingerprint and manage state
current_fp = compute_config_fingerprint(students, job_roles)

if "simp_last_run_fp" not in st.session_state:
    st.session_state["simp_last_run_fp"] = None
if "simp_run_clicked_at" not in st.session_state:
    st.session_state["simp_run_clicked_at"] = None
if "simp_matrix_rows" not in st.session_state:
    st.session_state["simp_matrix_rows"] = None
if "simp_all_matches" not in st.session_state:
    st.session_state["simp_all_matches"] = None

st.sidebar.header("Step 4: Run")
if st.session_state["simp_last_run_fp"] and st.session_state["simp_last_run_fp"] != current_fp:
    st.sidebar.warning("Inputs changed since last run. Click **Run Evaluation** to update results.")

run_btn = st.sidebar.button("▶ Run Evaluation")

if run_btn:
    st.session_state["simp_last_run_fp"] = current_fp
    st.session_state["simp_run_clicked_at"] = time.time()
    st.session_state["simp_matrix_rows"] = None
    st.session_state["simp_all_matches"] = None

# Main matching logic
matcher = SimplifiedInternshipMatcher(openai_api_key=openai_api_key, use_llm=use_llm)

ready_to_run = (
    students
    and job_roles
    and st.session_state["simp_last_run_fp"] == current_fp
    and st.session_state["simp_run_clicked_at"] is not None
)

if ready_to_run:
    st.header("Matching Results (Technical Stack + Education)")

    # Use cached compute when available
    if st.session_state["simp_matrix_rows"] is None or st.session_state["simp_all_matches"] is None:
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

                    unmet_cats = [c for c, d in result.stack_details.items() if not d.get("met")]
                    row[f"{role.job_title} - Unmet Stack Categories"] = ", ".join(unmet_cats) if unmet_cats else ""

                    # Format matched skills with hierarchy
                    matched_skills_display = matcher.format_skill_hierarchy_display(result.matched_skills)
                    row[f"{role.job_title} - Matched Skills"] = matched_skills_display
                    row[f"{role.job_title} - Missing Skills"] = ", ".join(result.missing_skills[:10])  # Limit display
                    row[f"{role.job_title} - Reasoning"] = result.overall_reasoning

                    processed += 1
                    progress_bar.progress(processed / total_combinations)

                all_matches[student.email] = student_results
                matrix_rows.append(row)

            progress_bar.empty()
            st.session_state["simp_all_matches"] = all_matches
            st.session_state["simp_matrix_rows"] = matrix_rows

    # Display results
    df_results = pd.DataFrame(st.session_state["simp_matrix_rows"])
    st.dataframe(df_results, use_container_width=True)

    all_matches = st.session_state.get("simp_all_matches") or {}
    if not all_matches:
        st.info("No results cached yet. Click **Run Evaluation** after setting inputs.")
        st.stop()

    # Detailed student analysis
    st.header("Detailed Student Analysis")

    emails_with_results = list(all_matches.keys())
    if not emails_with_results:
        st.info("No evaluated students found. Please re-run evaluation.")
        st.stop()

    valid_current_emails = [s.email for s in students if s.email in all_matches]
    options = valid_current_emails or emails_with_results

    selected_email = st.selectbox(
        "Select a student for detailed analysis",
        [s.email for s in students],
        format_func=lambda email: f"{next((s.name for s in students if s.email == email), email)} ({email})"
    )

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

            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Overall Score", f"{result.score:.3f}")
            with c2: st.metric("Stack Coverage", f"{result.stack_coverage_score:.3f}")
            with c3: st.metric("Education Match", f"{result.education_match_score:.3f}")

            # Enhanced matched skills display with hierarchy
            st.write("**🎯 Matched Skills with Hierarchy:**")
            if result.matched_skills:
                matched_display = matcher.format_skill_hierarchy_display(result.matched_skills)
                st.success(f"✅ {matched_display}")
                
                # Show detailed hierarchy breakdown
                with st.expander("📊 Detailed Skill Hierarchy Breakdown"):
                    for skill_match in result.matched_skills:
                        if skill_match.is_parent and skill_match.matched_children:
                            st.write(f"**🌳 Parent Skill:** {skill_match.skill_name}")
                            for child in skill_match.matched_children:
                                st.write(f"   └── 🌱 Child: {child}")
                        elif skill_match.parent_skill and not skill_match.is_parent:
                            st.write(f"**🌱 Child Skill:** {skill_match.skill_name} (under {skill_match.parent_skill})")
                        else:
                            st.write(f"**⭐ Standalone Skill:** {skill_match.skill_name}")
            else:
                st.warning("No skills matched for this role.")

            # Technical stack breakdown
            st.write("**Technical Stack Breakdown:**")
            for cat, d in result.stack_details.items():
                status = "✅ Met" if d.get("met") else "❌ Not Met"
                llm_indicator = " 🤖" if d.get("llm_enhanced") else ""
                st.write(f"- **{cat}** ({'Mandatory' if d.get('mandatory') else 'Optional'}, need {d.get('min_match',1)}): {status}{llm_indicator}")
                st.write(f"  - Matched: {', '.join(d.get('matched', [])) or '—'}")
                st.write(f"  - Missing: {', '.join(d.get('missing', [])) or '—'}")

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
        file_name=f"simplified_student_job_matching_{time.strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

else:
    # Guide the user
    if not students:
        st.info("Upload student CSV to begin.")
    elif not job_roles:
        st.info("Define at least one job role with a technical stack.")
    else:
        st.info("Ready to run. Click **Run Evaluation** in the sidebar.")

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
    - **Technical Stack (JSON)**: Category-based, each with:
      - `options`: list of acceptable skills
      - `mandatory`: true/false (gate requirement)
      - `min_match`: integer (≥1, minimum skills needed from this category)
      
      Example:
      ```json
      {
        "Programming Language": {"options": ["python", "r"], "mandatory": true, "min_match": 1},
        "Libraries & Frameworks": {"options": ["tensorflow", "pytorch"], "mandatory": true, "min_match": 1},
        "Data Libraries": {"options": ["pandas", "numpy"], "mandatory": false, "min_match": 1},
        "Software & Platform": {"options": ["jupyter notebook"], "mandatory": false, "min_match": 1},
        "Database & API": {"options": ["sql", "nosql"], "mandatory": true, "min_match": 1}
      }
      ```

    **Simplified Scoring System:**
    - Technical Stack Coverage: 60%
    - Education Match: 40%
    - **Optional AI Enhancement**: Enable to find transferable skills (e.g., MySQL experience satisfying SQL requirement)
    - If any mandatory stack category fails → Overall score = 0
    
    **🌳 Skill Hierarchy Display:**
    - **Parent Skills** with children shown as: `Python (→ Django, Flask)`
    - **Child Skills** with parent shown as: `Django (← Python)`
    - **Standalone Skills** displayed normally
    - Detailed breakdown available in the analysis section
    
    **AI-Enhanced Skill Matching:**
    When enabled, the system uses AI to identify:
    - Similar technologies (MySQL → SQL)
    - Related frameworks (React → Frontend)
    - Transferable skills (Python → Programming)
    - Academic knowledge that applies to industry requirements
    """)

st.markdown("---")
st.caption("Enhanced Student-Job Matcher with Skill Hierarchy 🚀 🌳")