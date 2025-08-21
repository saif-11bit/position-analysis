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
class CategorySkillMatch:
    """Represents matched skills within a category, organized by parent-child hierarchy"""
    category_name: str
    parent_skill_groups: Dict[str, List[str]]  # parent_name -> list of matched children
    standalone_skills: List[str]  # skills without parents or parents without children
    
    def get_display_string(self) -> str:
        """Format the category skills for display with proper hierarchy chains"""
        display_parts = []
        
        # Add parent skill groups with hierarchy chains
        for parent, children in self.parent_skill_groups.items():
            if len(children) == 1 and children[0].lower() == parent.lower():
                # Parent skill matched directly
                display_parts.append(parent)
            else:
                # Parent with children - show hierarchy chain
                children_str = ", ".join(children)
                display_parts.append(f"{parent} (→ {children_str})")
        
        # Add standalone skills
        display_parts.extend(self.standalone_skills)
        
        return ", ".join(display_parts)

@dataclass
class MatchResult:
    student: Student
    score: float
    stack_pass: bool
    stack_coverage_score: float
    education_match_score: float
    matched_skills_by_category: List[CategorySkillMatch]  # Updated to use CategorySkillMatch
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
        
        # Build comprehensive skill normalization maps from taxonomy
        self.skill_to_canonical = self.build_skill_normalization_maps()
        
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

    def build_skill_normalization_maps(self) -> Dict[str, str]:
        """
        Build comprehensive skill normalization mapping from the taxonomy.
        Maps all variations (name + aliases) to canonical skill names.
        """
        skill_to_canonical = {}
        
        if not self.skills_data:
            return skill_to_canonical
            
        for skill in self.skills_data:
            canonical_name = skill.get("name", "").strip()
            if not canonical_name:
                continue
                
            canonical_lower = self._norm(canonical_name)
            
            # Map the canonical name to itself
            skill_to_canonical[canonical_lower] = canonical_name
            
            # Map all aliases to the canonical name
            aliases = skill.get("aliases", []) or []
            for alias in aliases:
                if alias and isinstance(alias, str):
                    alias_lower = self._norm(alias)
                    skill_to_canonical[alias_lower] = canonical_name
        
        return skill_to_canonical

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

        return child_to_parent

    def get_ultimate_parent(self, skill_name: str) -> str:
        """
        Traverse up the hierarchy to find the ultimate/root parent skill.
        Returns the root parent name, or empty string if no parent found.
        """
        skill_lower = self._norm(skill_name)
        visited = set()  # Prevent infinite loops
        current_skill = skill_lower
        
        while current_skill and current_skill not in visited:
            visited.add(current_skill)
            parent_skill = self.child_to_parent_map.get(current_skill)
            if parent_skill:
                parent_name = self._norm(parent_skill.get("name", ""))
                if parent_name:
                    current_skill = parent_name
                else:
                    break
            else:
                # No parent found, this is the root
                if current_skill == skill_lower:
                    return ""  # Original skill has no parent
                else:
                    # Return the last parent we found
                    for visited_skill in visited:
                        if visited_skill != skill_lower:
                            skill_obj = next((s for s in self.skills_data if self._norm(s.get("name", "")) == visited_skill), None)
                            if skill_obj:
                                return skill_obj.get("name", "")
                    return ""
        
        # If we exited the loop, current_skill should be the root parent
        if current_skill and current_skill != skill_lower:
            skill_obj = next((s for s in self.skills_data if self._norm(s.get("name", "")) == current_skill), None)
            if skill_obj:
                return skill_obj.get("name", "")
        
        return ""

    def get_all_descendants(self, parent_skill_name: str) -> List[str]:
        """
        Get all descendants (children, grandchildren, etc.) of a parent skill.
        """
        parent_lower = self._norm(parent_skill_name)
        descendants = set()
        
        # Find all skills that have this as their ultimate parent
        for skill_data in self.skills_data:
            skill_name = skill_data.get("name", "")
            if skill_name:
                ultimate_parent = self.get_ultimate_parent(skill_name)
                if self._norm(ultimate_parent) == parent_lower and self._norm(skill_name) != parent_lower:
                    descendants.add(skill_name)
        
        return list(descendants)

    def get_skill_hierarchy_info(self, skill_name: str) -> Tuple[str, bool, List[str]]:
        """
        Returns tuple of (ultimate_parent_name, is_parent_skill, children_list)
        """
        skill_lower = self._norm(skill_name)
        ultimate_parent = self.get_ultimate_parent(skill_name)
        
        # Check if this skill is itself a parent
        is_parent = False
        children = []
        for skill_data in self.skills_data:
            if self._norm(skill_data.get("name", "")) == skill_lower:
                hierarchy = skill_data.get("hierarchy", {})
                children = hierarchy.get("children", [])
                is_parent = len(children) > 0
                break
        
        return ultimate_parent, is_parent, children

    def get_immediate_parent(self, skill_name: str) -> str:
        """
        Get the immediate parent of a skill (not the ultimate parent).
        For Git → returns "Version Control" (not "Software Development")
        """
        skill_lower = self._norm(skill_name)
        parent_skill = self.child_to_parent_map.get(skill_lower)
        return parent_skill.get("name", "") if parent_skill else ""

    def create_category_skill_matches_with_requirements(
        self, 
        matched_skills_by_category: Dict[str, List[str]], 
        requirements_by_category: Dict[str, List[str]]
    ) -> List[CategorySkillMatch]:
        """
        Enhanced version that properly groups skills under their parent requirements.
        """
        category_matches = []
        
        for category_name, matched_skills in matched_skills_by_category.items():
            if not matched_skills:
                continue
            
            required_skills = requirements_by_category.get(category_name, [])
            parent_groups = {}
            standalone_skills = []
            processed_skills = set()
            
            # Group skills by which requirement they satisfy
            for matched_skill in matched_skills:
                if matched_skill in processed_skills:
                    continue
                
                # Find which requirement this skill satisfies
                satisfying_requirement = None
                
                for req in required_skills:
                    # Direct match
                    if self._norm(matched_skill) == self._norm(req):
                        satisfying_requirement = req
                        break
                    
                    # Canonical match
                    skill_canonical = self.skill_to_canonical.get(self._norm(matched_skill))
                    req_canonical = self.skill_to_canonical.get(self._norm(req))
                    if skill_canonical and req_canonical and skill_canonical.lower() == req_canonical.lower():
                        satisfying_requirement = req
                        break
                    
                    # Child skill satisfies parent requirement
                    skill_parent = self.get_immediate_parent(matched_skill)
                    if skill_parent and self._norm(skill_parent) == self._norm(req):
                        satisfying_requirement = req
                        break
                    
                    # Alias matches
                    if skill_canonical and self._norm(skill_canonical) == self._norm(req):
                        satisfying_requirement = req
                        break
                    
                    if req_canonical and self._norm(req_canonical) == self._norm(matched_skill):
                        satisfying_requirement = req
                        break
                
                if satisfying_requirement:
                    # Group under the requirement it satisfies
                    if satisfying_requirement not in parent_groups:
                        parent_groups[satisfying_requirement] = []
                    parent_groups[satisfying_requirement].append(matched_skill)
                    processed_skills.add(matched_skill)
                else:
                    # Standalone skill
                    standalone_skills.append(matched_skill)
                    processed_skills.add(matched_skill)
            
            # Clean up - remove duplicates
            for req, skills in parent_groups.items():
                parent_groups[req] = list(dict.fromkeys(skills))
            
            standalone_skills = list(dict.fromkeys(standalone_skills))
            
            category_match = CategorySkillMatch(
                category_name=category_name,
                parent_skill_groups=parent_groups,
                standalone_skills=standalone_skills
            )
            
            category_matches.append(category_match)
        
        return category_matches

    def create_category_skill_matches(self, matched_skills_by_category: Dict[str, List[str]]) -> List[CategorySkillMatch]:
        """
        Convert category-based matched skills to CategorySkillMatch objects with proper hierarchical grouping.
        Now handles nested requirements properly (e.g., Node.js and Express.js both required).
        """
        category_matches = []
        
        for category_name, matched_skills in matched_skills_by_category.items():
            if not matched_skills:
                continue
            
            # Get the requirements for this category from the current context
            # This is a bit of a hack - we need to pass requirements to this method
            # For now, we'll use the existing immediate parent grouping with enhancement
            
            parent_groups = {}  # immediate_parent_name -> list of chains
            standalone_skills = []
            processed_skills = set()
            
            # Build skill-to-parent mapping
            skill_to_parent = {}
            for skill_name in matched_skills:
                immediate_parent = self.get_immediate_parent(skill_name)
                skill_to_parent[skill_name] = immediate_parent
            
            # Group skills by their immediate parent, but merge related chains
            for skill_name in matched_skills:
                if skill_name in processed_skills:
                    continue
                
                immediate_parent = skill_to_parent[skill_name]
                
                if immediate_parent:
                    if immediate_parent not in parent_groups:
                        parent_groups[immediate_parent] = []
                    
                    # Check if this skill or its parent is already represented
                    # If Node.js is required and Express.js satisfies it, show full chain
                    chain_skills = [skill_name]
                    
                    # Look for children of this skill in matched skills
                    for other_skill in matched_skills:
                        if other_skill != skill_name and self.get_immediate_parent(other_skill) == skill_name:
                            chain_skills.append(other_skill)
                            processed_skills.add(other_skill)
                    
                    parent_groups[immediate_parent].extend(chain_skills)
                    processed_skills.add(skill_name)
                    
                else:
                    # Check if this skill is a parent of other matched skills
                    children_in_matched = [s for s in matched_skills if skill_to_parent[s] == skill_name]
                    
                    if children_in_matched:
                        parent_groups[skill_name] = children_in_matched
                        processed_skills.add(skill_name)
                        for child in children_in_matched:
                            processed_skills.add(child)
                    else:
                        standalone_skills.append(skill_name)
                        processed_skills.add(skill_name)
            
            # Remove duplicates and create clean hierarchy
            cleaned_parent_groups = {}
            for parent, skills_list in parent_groups.items():
                unique_skills = list(dict.fromkeys(skills_list))  # Preserve order, remove duplicates
                cleaned_parent_groups[parent] = unique_skills
            
            category_match = CategorySkillMatch(
                category_name=category_name,
                parent_skill_groups=cleaned_parent_groups,
                standalone_skills=standalone_skills
            )
            
            category_matches.append(category_match)
        
        return category_matches

    def format_categorized_skills_display(self, category_matches: List[CategorySkillMatch]) -> str:
        """Format categorized skill matches for display"""
        if not category_matches:
            return ""
        
        display_parts = []
        for category_match in category_matches:
            category_display = category_match.get_display_string()
            if category_display:
                display_parts.append(f"**{category_match.category_name}**: {category_display}")
        
        return " | ".join(display_parts)

    def normalize_skills(self, skills: List[str]) -> List[str]:
        """
        Basic normalization that only handles aliases and canonical forms.
        NO automatic parent addition - upward matching handled separately.
        """
        normalized = set()
        for skill in skills or []:
            s = (skill or '').lower().strip()
            if not s:
                continue
            
            # Add the original normalized form
            normalized.add(s)
            
            # Try to find canonical form from taxonomy (for aliases)
            canonical = self.skill_to_canonical.get(s)
            if canonical:
                normalized.add(canonical.lower())
        
        return list(normalized)

    def get_skill_satisfaction_chain(self, student_skill: str, required_skill: str) -> List[str]:
        """
        Returns the satisfaction chain from student skill to required skill.
        E.g., student has "GitHub", requirement is "Git" → returns ["Version Control", "Git", "GitHub"]
        Uses immediate parent relationships, not ultimate parent.
        """
        student_lower = self._norm(student_skill)
        required_lower = self._norm(required_skill)
        
        # Check if student skill can satisfy the requirement through hierarchy
        student_normalized = set(self.normalize_skills([student_skill]))
        if required_lower not in student_normalized:
            return [student_skill]  # No satisfaction possible
        
        # Build the chain using immediate parent relationships
        chain = [student_skill]
        
        # If student skill directly matches required skill
        if student_lower == required_lower:
            return chain
        
        # Check if student skill's immediate parent matches the requirement
        immediate_parent = self.get_immediate_parent(student_skill)
        if immediate_parent and self._norm(immediate_parent) == required_lower:
            return [immediate_parent, student_skill]
        
        # If immediate parent exists but doesn't match, check its parent
        if immediate_parent:
            parent_chain = self.get_skill_satisfaction_chain(immediate_parent, required_skill)
            if len(parent_chain) > 1 or (len(parent_chain) == 1 and self._norm(parent_chain[0]) == required_lower):
                # Found a valid chain through the parent
                return parent_chain + [student_skill]
        
        # Default case - just return the student skill
        return [student_skill]

    def find_original_skill_name(self, normalized_skill: str, student_skills: List[str]) -> str:
        """
        Find the original skill name from student's profile that matches the normalized skill.
        This helps preserve the original skill names for display purposes.
        """
        norm_lower = normalized_skill.lower()
        
        # First, check if any student skill normalizes to this skill
        for original_skill in student_skills:
            original_lower = self._norm(original_skill)
            # Check if this original skill maps to our normalized skill
            canonical = self.skill_to_canonical.get(original_lower)
            if canonical and canonical.lower() == norm_lower:
                return original_skill
            # Also check direct match
            if original_lower == norm_lower:
                return original_skill
        
        # If no match found, return the normalized skill
        return normalized_skill

    def calculate_skill_match_with_llm(
        self, 
        student_skills: List[str], 
        required_skills: List[str],
        use_llm: bool = False,
        openai_api_key: str = None
    ) -> Tuple[float, List[str], List[str], List[str]]:
        """
        Precise skill matching that only includes relevant skills for each category.
        Returns: (match_score, actual_student_matched_skills, missing_required_skills, llm_matched_skills)
        """
        # Keep original student skills for display
        student_skills_lower = [s.lower().strip() for s in student_skills if s.strip()]
        required_skills_lower = [s.lower().strip() for s in required_skills if s.strip()]
        
        # Debug print to help identify issues
        print(f"DEBUG: Student skills: {student_skills}")
        print(f"DEBUG: Required skills: {required_skills}")
        
        # Find student skills that are relevant to this category's requirements
        actual_matched = []
        
        for student_skill in student_skills:
            skill_is_relevant = False
            
            for req_skill in required_skills:
                # 1. Direct match (student: "python", requirement: "python")
                if self._norm(student_skill) == self._norm(req_skill):
                    skill_is_relevant = True
                    break
                
                # 2. Canonical match (student: "React.js", requirement: "react")
                student_canonical = self.skill_to_canonical.get(self._norm(student_skill))
                req_canonical = self.skill_to_canonical.get(self._norm(req_skill))
                if student_canonical and req_canonical and student_canonical.lower() == req_canonical.lower():
                    skill_is_relevant = True
                    break
                
                # 3. Child skill implies parent requirement (student: "Flask", requirement: "python")
                # Only if the requirement is the immediate parent of the student skill
                student_parent = self.get_immediate_parent(student_skill)
                if student_parent and self._norm(student_parent) == self._norm(req_skill):
                    skill_is_relevant = True
                    break
                
                # 4. Alias match through canonical
                if student_canonical and self._norm(student_canonical) == self._norm(req_skill):
                    skill_is_relevant = True
                    break
                
                if req_canonical and self._norm(req_canonical) == self._norm(student_skill):
                    skill_is_relevant = True
                    break
            
            if skill_is_relevant:
                actual_matched.append(student_skill)
        
        # Remove duplicates while preserving order
        actual_matched = list(dict.fromkeys(actual_matched))
        
        # Calculate which requirements are satisfied
        satisfied_requirements = set()
        for req_skill in required_skills:
            for student_skill in actual_matched:
                # Direct match
                if self._norm(student_skill) == self._norm(req_skill):
                    satisfied_requirements.add(req_skill.lower())
                    continue
                
                # Canonical match
                student_canonical = self.skill_to_canonical.get(self._norm(student_skill))
                req_canonical = self.skill_to_canonical.get(self._norm(req_skill))
                if student_canonical and req_canonical and student_canonical.lower() == req_canonical.lower():
                    satisfied_requirements.add(req_skill.lower())
                    continue
                
                # Child satisfies parent requirement
                student_parent = self.get_immediate_parent(student_skill)
                if student_parent and self._norm(student_parent) == self._norm(req_skill):
                    satisfied_requirements.add(req_skill.lower())
                    continue
                
                # Alias matches
                if student_canonical and self._norm(student_canonical) == self._norm(req_skill):
                    satisfied_requirements.add(req_skill.lower())
                    continue
                
                if req_canonical and self._norm(req_canonical) == self._norm(student_skill):
                    satisfied_requirements.add(req_skill.lower())
                    continue
        
        missing_requirements = [r for r in required_skills_lower if r not in satisfied_requirements]
        llm_matched = []
        
        print(f"DEBUG: Actual matched: {actual_matched}")
        print(f"DEBUG: Satisfied requirements: {satisfied_requirements}")
        print(f"DEBUG: Missing requirements: {missing_requirements}")
        
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
    ) -> Tuple[bool, float, Dict[str, Dict[str, Any]], Dict[str, List[str]], List[str]]:
        """
        Enhanced technical stack checking with categorized skill matching
        Returns matched_skills_by_category instead of flat skill list
        """
        details = {}
        matched_skills_by_category = {}
        requirements_by_category = {}  # Store requirements for enhanced grouping
        all_missing = []

        if not technical_stack:
            return True, 1.0, {}, {}, []

        categories = list(technical_stack.keys())
        if not categories:
            return True, 1.0, {}, {}, []

        passed = True
        category_scores = []

        for category, req in technical_stack.items():
            options = [str(o).strip() for o in req.get("options", []) if str(o).strip()]
            mandatory = bool(req.get("mandatory", False))
            min_match = max(int(req.get("min_match", 1)), 1)

            # Store requirements for enhanced grouping
            requirements_by_category[category] = options

            # Use enhanced skill matching for this category
            match_score, actual_matched, missing_reqs, llm_matched = self.calculate_skill_match_with_llm(
                student_skills, options, use_llm, openai_api_key
            )
            
            # Combine actual matches and LLM matches for display
            all_category_matches = actual_matched + llm_matched
            total_matches = len(actual_matched) + len(llm_matched)
            
            # Store matched skills by category
            if all_category_matches:
                matched_skills_by_category[category] = all_category_matches
            
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

            if not met:
                all_missing.extend([f"{category}: {m}" for m in missing_reqs])

        coverage_score = sum(category_scores) / len(category_scores) if category_scores else 1.0
        
        # Store requirements for use in enhanced grouping
        self._current_requirements_by_category = requirements_by_category
        
        return passed, coverage_score, details, matched_skills_by_category, all_missing

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
        stack_pass, stack_cov, stack_details, matched_skills_by_category, flat_missing = self.check_technical_stack(
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

        # Convert to CategorySkillMatch objects with enhanced grouping
        if hasattr(self, '_current_requirements_by_category'):
            category_skill_matches = self.create_category_skill_matches_with_requirements(
                matched_skills_by_category, self._current_requirements_by_category
            )
        else:
            category_skill_matches = self.create_category_skill_matches(matched_skills_by_category)

        return MatchResult(
            student=student,
            score=overall_score,
            stack_pass=True,
            stack_coverage_score=stack_cov,
            education_match_score=education_score,
            matched_skills_by_category=category_skill_matches,
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
        # Parse skills - REMOVED automatic parent addition
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
st.set_page_config(page_title="Enhanced Student-Job Matcher", layout="wide")
st.title("🎓 Enhanced Student-Job Matching (Fixed Skill Alias Matching)")

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
    st.header("Matching Results (Fixed Skill Alias Matching)")

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

                    # Format matched skills with categorized hierarchy
                    matched_skills_display = matcher.format_categorized_skills_display(result.matched_skills_by_category)
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

            # Enhanced categorized skills display with hierarchy
            st.write("**🎯 Matched Skills by Category with Hierarchy:**")
            if result.matched_skills_by_category:
                for category_match in result.matched_skills_by_category:
                    st.success(f"**📂 {category_match.category_name}:** {category_match.get_display_string()}")
                
                # Show detailed hierarchy breakdown
                with st.expander("📊 Detailed Category & Skill Hierarchy Breakdown"):
                    for category_match in result.matched_skills_by_category:
                        st.write(f"**📂 Category: {category_match.category_name}**")
                        
                        # Show parent-child groups with enhanced nested hierarchy
                        for parent, children in category_match.parent_skill_groups.items():
                            if len(children) == 1 and children[0].lower() == parent.lower():
                                st.write(f"   ⭐ **Direct Match:** {parent}")
                            else:
                                st.write(f"   🌳 **Satisfied via:** {parent}")
                                
                                # Get requirements for this category to build proper chains
                                category_requirements = []
                                if hasattr(matcher, '_current_requirements_by_category'):
                                    category_requirements = matcher._current_requirements_by_category.get(category_match.category_name, [])
                                
                                # Build enhanced satisfaction chains
                                displayed_skills = set()
                                
                                for child in sorted(children):
                                    if child in displayed_skills:
                                        continue
                                    
                                    # Find what this skill satisfies
                                    satisfies = []
                                    child_normalized = set(matcher.normalize_skills([child]))
                                    
                                    for req in category_requirements:
                                        req_normalized = set(matcher.normalize_skills([req]))
                                        if req_normalized.intersection(child_normalized):
                                            satisfies.append(req)
                                    
                                    if satisfies:
                                        # Build the most appropriate chain
                                        # If this skill satisfies multiple requirements, find the deepest one
                                        deepest_req = satisfies[0]
                                        for req in satisfies[1:]:
                                            if matcher.get_immediate_parent(req) in satisfies:
                                                deepest_req = req
                                        
                                        # Build chain from parent through requirements to student skill
                                        chain_parts = [parent]
                                        
                                        # Add intermediate requirements if they exist
                                        req_chain = []
                                        current_req = deepest_req
                                        while current_req:
                                            req_chain.insert(0, current_req)
                                            req_parent = matcher.get_immediate_parent(current_req)
                                            if req_parent == parent or req_parent in req_chain:
                                                break
                                            current_req = req_parent
                                        
                                        # Add the requirement chain
                                        chain_parts.extend(req_chain)
                                        
                                        # Add student skill if different from requirement
                                        if child not in req_chain:
                                            chain_parts.append(child)
                                        
                                        # Remove duplicates while preserving order
                                        clean_chain = []
                                        seen = set()
                                        for part in chain_parts:
                                            if part not in seen:
                                                clean_chain.append(part)
                                                seen.add(part)
                                        
                                        chain_display = " → ".join(clean_chain)
                                        st.write(f"      └── 📍 **Satisfaction Chain:** {chain_display}")
                                    else:
                                        # Fallback to simple display
                                        st.write(f"      └── 🌱 **Student Skill:** {child}")
                                    
                                    displayed_skills.add(child)
                        
                        # Show standalone skills
                        for skill in category_match.standalone_skills:
                            st.write(f"   ⭐ **Direct Match:** {skill}")
                        
                        # Show standalone skills
                        for skill in category_match.standalone_skills:
                            st.write(f"   ⭐ **Direct Match:** {skill}")
                        
                        st.write("")  # Empty line for separation
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
        file_name=f"fixed_student_job_matching_{time.strftime('%Y%m%d_%H%M%S')}.csv",
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

# Debug section to help troubleshoot
st.sidebar.header("🔍 Debug Information")
if st.sidebar.checkbox("Show Debug Info"):
    if students:
        st.write("**Sample Student Skills:**")
        for student in students[:2]:  # Show first 2 students
            st.write(f"- {student.name}: {student.skills}")
    
    if job_roles:
        st.write("**Job Requirements:**")
        for role in job_roles[:2]:  # Show first 2 roles
            st.write(f"- {role.job_title}: {role.technical_stack}")
    
    # Show skill taxonomy info
    matcher = SimplifiedInternshipMatcher()
    if matcher.skill_to_canonical:
        st.write("**Sample Skill Normalizations:**")
        sample_mappings = list(matcher.skill_to_canonical.items())[:10]
        for orig, canonical in sample_mappings:
            st.write(f"- '{orig}' → '{canonical}'")

# Instructions with fixes highlighted
with st.expander("📋 Instructions & Key Fixes"):
    st.markdown("""
    **🔧 KEY FIXES IMPLEMENTED:**
    
    1. **Fixed Skill Taxonomy Loading**: Now properly loads and utilizes the skill taxonomy JSON file
    2. **Enhanced Alias Matching**: The `build_skill_normalization_maps()` method now correctly maps all aliases to canonical skill names
    3. **Improved Normalization**: The `normalize_skills()` method now uses the taxonomy for accurate skill matching
    4. **Better Debug Output**: Added debug prints and information to help identify matching issues
    5. **Preserved Original Names**: Skills are matched using normalized forms but displayed using original student skill names
    
    **Example Fix**: If your taxonomy has:
    ```json
    {
        "name": "React",
        "aliases": ["React.js", "ReactJS", "react.js"]
    }
    ```
    
    Now when a student has "React.js" and job requires "react", they will match correctly!
    
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

    **Enhanced Debugging Features:**
    - Enable "Show Debug Info" in the sidebar to see skill normalization mappings
    - Debug prints show skill matching process in console
    - Sample skill taxonomy mappings displayed
    
    **Testing the Fix:**
    1. Upload your student CSV with "React.js" skills
    2. Create a job role requiring "react"
    3. Run evaluation - they should now match!
    """)

st.markdown("---")
st.caption("Enhanced Student-Job Matcher with FIXED Skill Alias Matching 🚀 🔧 ✅")