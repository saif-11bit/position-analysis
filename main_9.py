import streamlit as st
import pandas as pd
import ast
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Set
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

# ---- Enhanced Data Classes ----
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
    student_skill: str  # Original student skill name
    matched_requirement: str  # The requirement it satisfies
    match_type: str  # 'direct', 'parent_match', 'child_match', 'alias'
    parent_skill: str = ""  # Parent skill if applicable
    category_metadata: str = ""  # Category from requirements as metadata
    
    def get_display_string(self) -> str:
        """Format the skill match showing SATISFIED REQUIREMENTS"""
        if self.match_type == 'direct':
            return self.matched_requirement  # Show what requirement was satisfied
        elif self.match_type == 'child_to_parent':
            return f"{self.matched_requirement} (via {self.student_skill})"  # Show satisfied requirement (via student skill)
        elif self.match_type == 'alias':
            return f"{self.matched_requirement}"  # Show what requirement was satisfied
        elif self.match_type == 'llm_enhanced':
            return f"{self.student_skill} (AI-enhanced)"
        return self.matched_requirement

@dataclass
class CategorySkillMatch:
    """Enhanced to show parent-child relationships based on actual matches"""
    category_name: str
    skill_matches: List[SkillMatch]  # List of SkillMatch objects
    
    def get_display_string(self) -> str:
        """Format the category skills for display showing SATISFIED REQUIREMENTS"""
        if not self.skill_matches:
            return ""
        
        # Group matches by type - show what requirements were satisfied
        direct_matches = []
        child_to_parent_matches = []  # Show the satisfied parent requirement, not child skill
        alias_matches = []
        other_matches = []
        
        for skill_match in self.skill_matches:
            if skill_match.match_type == 'direct':
                direct_matches.append(skill_match.matched_requirement)  # Show requirement, not student skill
            elif skill_match.match_type == 'child_to_parent':
                # Show: satisfied_parent_requirement (via student_child_skill)
                child_to_parent_matches.append(f"{skill_match.matched_requirement} (via {skill_match.student_skill})")
            elif skill_match.match_type == 'alias':
                alias_matches.append(f"{skill_match.matched_requirement}")  # Show requirement, not student skill
            else:
                other_matches.append(skill_match.get_display_string())
        
        display_parts = []
        
        # Add different match types - showing satisfied requirements
        if direct_matches:
            display_parts.extend(direct_matches)
        if child_to_parent_matches:
            display_parts.extend(child_to_parent_matches)
        if alias_matches:
            display_parts.extend(alias_matches)
        if other_matches:
            display_parts.extend(other_matches)
        
        return ", ".join(display_parts)

@dataclass
class MatchResult:
    student: Student
    score: float
    stack_pass: bool
    stack_coverage_score: float
    education_match_score: float
    matched_skills_by_category: List[CategorySkillMatch]
    missing_skills: List[str]
    overall_reasoning: str
    stack_details: Dict[str, Dict[str, Any]]

# ---- Enhanced Matcher Class ----
class EnhancedInternshipMatcher:
    def __init__(self, openai_api_key: str = None, use_llm=False):
        self.openai_api_key = openai_api_key
        self.use_llm = use_llm
        
        # Load skill hierarchy
        self.skills_data = self.load_skills("skill_taxonomy_565_skills.json")
        self.child_to_parent_map = self.build_child_to_parent(self.skills_data)
        self.parent_to_children_map = self.build_parent_to_children(self.skills_data)
        
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
        """Build comprehensive skill normalization mapping from the taxonomy."""
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
        """Builds a mapping from child label/name/alias -> parent skill object."""
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

        # Pass 1: child's explicit hierarchy.parent
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

    def build_parent_to_children(self, skills: List[dict]) -> Dict[str, List[str]]:
        """Build mapping from parent skill name to list of child skill names."""
        parent_to_children = {}
        
        if not skills:
            return parent_to_children
            
        for skill in skills:
            skill_name = skill.get("name", "").strip()
            if not skill_name:
                continue
                
            skill_lower = self._norm(skill_name)
            parent_skill = self.child_to_parent_map.get(skill_lower)
            
            if parent_skill:
                parent_name = parent_skill.get("name", "")
                if parent_name:
                    parent_lower = self._norm(parent_name)
                    if parent_lower not in parent_to_children:
                        parent_to_children[parent_lower] = []
                    parent_to_children[parent_lower].append(skill_name)
        
        return parent_to_children

    def get_immediate_parent(self, skill_name: str) -> str:
        """Get the immediate parent of a skill."""
        skill_lower = self._norm(skill_name)
        parent_skill = self.child_to_parent_map.get(skill_lower)
        return parent_skill.get("name", "") if parent_skill else ""

    def get_all_children(self, parent_skill_name: str) -> List[str]:
        """Get all immediate children of a parent skill."""
        parent_lower = self._norm(parent_skill_name)
        return self.parent_to_children_map.get(parent_lower, [])

    def match_skills_with_hierarchy(
        self, 
        student_skills: List[str], 
        required_skills: List[str],
        category_name: str = ""
    ) -> Tuple[List[SkillMatch], List[str]]:
        """
        Enhanced skill matching with CORRECT hierarchy logic:
        - Child → Parent: Student has Django, requirement is Python = MATCH ✅
        - Parent → Child: Student has Python, requirement is Django = NO MATCH ❌
        """
        skill_matches = []
        matched_requirements = set()
        
        # Convert to lowercase for matching but keep originals for display
        student_skills_map = {self._norm(skill): skill for skill in student_skills}
        required_skills_lower = [self._norm(req) for req in required_skills]
        
        print(f"DEBUG - Category: {category_name}")
        print(f"DEBUG - Student skills: {student_skills}")
        print(f"DEBUG - Required skills: {required_skills}")
        
        # 1. Direct matches (including aliases)
        for req_skill in required_skills:
            req_lower = self._norm(req_skill)
            
            # Check direct match
            if req_lower in student_skills_map:
                skill_matches.append(SkillMatch(
                    student_skill=student_skills_map[req_lower],
                    matched_requirement=req_skill,
                    match_type='direct',
                    category_metadata=category_name
                ))
                matched_requirements.add(req_lower)
                continue
            
            # Check canonical/alias match
            req_canonical = self.skill_to_canonical.get(req_lower)
            if req_canonical:
                req_canonical_lower = self._norm(req_canonical)
                if req_canonical_lower in student_skills_map:
                    skill_matches.append(SkillMatch(
                        student_skill=student_skills_map[req_canonical_lower],
                        matched_requirement=req_skill,
                        match_type='alias',
                        category_metadata=category_name
                    ))
                    matched_requirements.add(req_lower)
                    continue
            
            # Check if any student skill's canonical form matches requirement
            for student_skill_lower, student_skill_orig in student_skills_map.items():
                student_canonical = self.skill_to_canonical.get(student_skill_lower)
                if student_canonical and self._norm(student_canonical) == req_lower:
                    skill_matches.append(SkillMatch(
                        student_skill=student_skill_orig,
                        matched_requirement=req_skill,
                        match_type='alias',
                        category_metadata=category_name
                    ))
                    matched_requirements.add(req_lower)
                    break
        
        # 2. CORRECT Hierarchical matching: ONLY Child → Parent (Student child skill satisfies parent requirement)
        for req_skill in required_skills:
            req_lower = self._norm(req_skill)
            if req_lower in matched_requirements:
                continue
                
            # Check if any student skill is a CHILD of this requirement (requirement is PARENT)
            for student_skill_lower, student_skill_orig in student_skills_map.items():
                student_parent = self.get_immediate_parent(student_skill_orig)
                
                if student_parent:
                    # Direct parent match: Student has child skill, requirement is parent
                    if self._norm(student_parent) == req_lower:
                        skill_matches.append(SkillMatch(
                            student_skill=student_skill_orig,
                            matched_requirement=req_skill,
                            match_type='child_to_parent',  # Renamed for clarity
                            parent_skill=student_parent,
                            category_metadata=category_name
                        ))
                        matched_requirements.add(req_lower)
                        break
                    
                    # Canonical parent match
                    student_parent_canonical = self.skill_to_canonical.get(self._norm(student_parent))
                    req_canonical = self.skill_to_canonical.get(req_lower)
                    
                    if student_parent_canonical and req_canonical:
                        if self._norm(student_parent_canonical) == self._norm(req_canonical):
                            skill_matches.append(SkillMatch(
                                student_skill=student_skill_orig,
                                matched_requirement=req_skill,
                                match_type='child_to_parent',
                                parent_skill=student_parent,
                                category_metadata=category_name
                            ))
                            matched_requirements.add(req_lower)
                            break
        
        # NOTE: Removed Parent → Child matching as it's logically incorrect
        # Having Python doesn't mean you know Django specifically
        
        # Find unmatched requirements
        unmatched_requirements = [req for req in required_skills if self._norm(req) not in matched_requirements]
        
        print(f"DEBUG - Skill matches: {[(sm.student_skill, sm.matched_requirement, sm.match_type) for sm in skill_matches]}")
        print(f"DEBUG - Unmatched requirements: {unmatched_requirements}")
        
        return skill_matches, unmatched_requirements

    def check_technical_stack(
        self,
        student_skills: List[str],
        technical_stack: Dict[str, Dict[str, Any]],
        use_llm: bool = False,
        openai_api_key: str = None
    ) -> Tuple[bool, float, Dict[str, Dict[str, Any]], List[CategorySkillMatch], List[str]]:
        """
        Enhanced technical stack checking with hierarchical skill matching.
        """
        details = {}
        category_skill_matches = []
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

            # Use enhanced hierarchical skill matching
            skill_matches, unmatched_reqs = self.match_skills_with_hierarchy(
                student_skills, options, category
            )
            
            # Apply LLM enhancement if available
            llm_enhanced_matches = []
            if use_llm and openai_api_key and unmatched_reqs:
                llm_enhanced_matches = self.enhance_with_llm(student_skills, unmatched_reqs, openai_api_key)
            
            # Combine matches
            total_matches = len(skill_matches) + len(llm_enhanced_matches)
            met = total_matches >= min_match
            
            if mandatory and not met:
                passed = False

            # Create CategorySkillMatch object
            if skill_matches or llm_enhanced_matches:
                # Convert LLM matches to SkillMatch objects
                for llm_match in llm_enhanced_matches:
                    skill_matches.append(SkillMatch(
                        student_skill=llm_match,
                        matched_requirement="LLM Enhanced",
                        match_type='llm_enhanced',
                        category_metadata=category
                    ))
                
                category_skill_matches.append(CategorySkillMatch(
                    category_name=category,
                    skill_matches=skill_matches
                ))

            details[category] = {
                "required": options,
                "min_match": min_match,
                "matched": [sm.student_skill for sm in skill_matches] + llm_enhanced_matches,
                "missing": unmatched_reqs,
                "mandatory": mandatory,
                "met": met,
                "llm_enhanced": len(llm_enhanced_matches) > 0
            }

            # Coverage score
            category_score = min(total_matches, min_match) / max(min_match, 1)
            category_scores.append(category_score)

            if not met:
                all_missing.extend([f"{category}: {m}" for m in unmatched_reqs])

        coverage_score = sum(category_scores) / len(category_scores) if category_scores else 1.0
        
        return passed, coverage_score, details, category_skill_matches, all_missing

    def enhance_with_llm(self, student_skills: List[str], missing_requirements: List[str], openai_api_key: str) -> List[str]:
        """Enhanced LLM skill matching"""
        if not missing_requirements:
            return []
            
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
            
            Respond with only the student skills that can satisfy missing requirements, one per line.
            If no additional matches found, respond with: NONE
            """
            
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
            
            content = response.choices[0].message.content.strip()
            
            if "NONE" in content.upper():
                return []
                
            # Parse response
            llm_matches = []
            lines = content.split('\n')
            for line in lines:
                skill = line.strip()
                if skill and any(s.lower() == skill.lower() for s in student_skills):
                    llm_matches.append(f"{skill} (AI-enhanced)")
            
            time.sleep(0.2)  # Rate limiting
            return llm_matches
            
        except Exception as e:
            print(f"LLM enhancement failed: {e}")
            return []

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
        """Generate comprehensive reasoning"""
        if not stack_pass:
            unmet_categories = [cat for cat, d in stack_details.items() if d.get("mandatory") and not d.get("met")]
            return f"Failed mandatory technical requirements in: {', '.join(unmet_categories)}"
        
        # Determine scoring logic
        if len(internship_req.relevant_degrees) == 0:
            final_score = (stack_score * 0.6) + (1.0 * 0.4)
            scoring_note = "No specific degree requirements"
        elif education_score == 0:
            final_score = (stack_score * 0.8) + (education_score * 0.2)
            scoring_note = "Adjusted scoring due to education mismatch"
        else:
            final_score = (stack_score * 0.6) + (education_score * 0.4)
            scoring_note = "Standard scoring"
        
        if final_score >= 0.8:
            level = "Excellent match"
        elif final_score >= 0.6:
            level = "Good match"
        elif final_score >= 0.4:
            level = "Fair match"
        else:
            level = "Limited match"
            
        # Analyze match types
        hierarchy_matches = 0
        direct_matches = 0
        
        for category, details in stack_details.items():
            matched_skills = details.get('matched', [])
            for skill in matched_skills:
                if '→' in skill or 'satisfies' in skill:
                    hierarchy_matches += 1
                else:
                    direct_matches += 1
        
        reasoning = f"{level} (Score: {final_score:.2f}). "
        if direct_matches > 0:
            reasoning += f"Direct skill matches: {direct_matches}. "
        if hierarchy_matches > 0:
            reasoning += f"Hierarchical skill matches: {hierarchy_matches}. "
            
        return reasoning.strip()

    def match_student_to_internship(self, student: Student, internship_req: InternshipRequirement) -> MatchResult:
        # Enhanced technical stack checking
        stack_pass, stack_cov, stack_details, category_skill_matches, flat_missing = self.check_technical_stack(
            student.skills, internship_req.technical_stack, self.use_llm, self.openai_api_key
        )

        # Education matching
        education_score = self.calculate_education_match_score(
            student.education, student.degree, internship_req.relevant_degrees
        )

        # Score calculation with enhanced logic
        if len(internship_req.relevant_degrees) == 0:
            education_score = 1.0
            overall_score = (stack_cov * 0.6) + (education_score * 0.4)
        elif education_score == 0:
            overall_score = (stack_cov * 0.8) + (education_score * 0.2)
        else:
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
            matched_skills_by_category=category_skill_matches,
            missing_skills=flat_missing,
            overall_reasoning=reasoning,
            stack_details=stack_details
        )

    def create_enhanced_mindmap_html(self, result, job_title: str):
        """Create enhanced mind map showing parent-child relationships without duplicates"""
        categories = result.matched_skills_by_category
        if not categories:
            return "<div>No data to visualize</div>"
        
        # Build enhanced D3 data structure
        nodes = [{'id': 'center', 'name': job_title, 'type': 'center', 'group': 0}]
        links = []
        
        node_id = 1
        created_parents = {}  # Track created parent nodes to avoid duplicates
        
        for i, category_match in enumerate(categories):
            cat_id = f'cat_{i}'
            has_matches = bool(category_match.skill_matches)
            
            nodes.append({
                'id': cat_id,
                'name': category_match.category_name,
                'type': 'category',
                'group': i + 1,
                'satisfied': has_matches
            })
            
            links.append({'source': 'center', 'target': cat_id, 'value': 3})
            
            # Build internal hierarchy within category based on CANONICAL REQUIREMENTS
            # This handles aliases properly when building parent-child relationships
            all_matched_requirements = [sm.matched_requirement for sm in category_match.skill_matches]
            
            # Remove duplicates while preserving order
            unique_requirements = []
            seen = set()
            for req in all_matched_requirements:
                if req not in seen:
                    unique_requirements.append(req)
                    seen.add(req)
            
            # Normalize requirements to their canonical forms for hierarchy analysis
            req_to_canonical = {}
            canonical_to_reqs = {}  # canonical -> [original_requirements]
            
            for req in unique_requirements:
                # Get canonical form (or use original if no canonical found)
                canonical = self.skill_to_canonical.get(self._norm(req), req)
                req_to_canonical[req] = canonical
                
                if canonical not in canonical_to_reqs:
                    canonical_to_reqs[canonical] = []
                canonical_to_reqs[canonical].append(req)
            
            print(f"DEBUG - Category {category_match.category_name}:")
            print(f"  Original requirements: {unique_requirements}")
            print(f"  Requirement to canonical: {req_to_canonical}")
            print(f"  Canonical to requirements: {canonical_to_reqs}")
            
            # Build hierarchy map using canonical forms: parent_canonical -> [child_canonicals]
            canonical_hierarchy_map = {}
            processed_canonicals = set()
            
            # Get unique canonical forms
            unique_canonicals = list(canonical_to_reqs.keys())
            
            # Find parent-child relationships among canonical forms
            for canonical in unique_canonicals:
                children_canonicals = []
                
                # Check if this canonical is a parent of any other canonical
                for other_canonical in unique_canonicals:
                    if other_canonical != canonical:
                        # Check if other_canonical is child of canonical (using skill taxonomy)
                        other_canonical_parent = self.get_immediate_parent(other_canonical)
                        
                        # Also check through canonical mapping
                        other_canonical_parent_canonical = self.skill_to_canonical.get(
                            self._norm(other_canonical_parent) if other_canonical_parent else "", 
                            other_canonical_parent or ""
                        )
                        
                        # Match if direct parent or canonical parent matches
                        if ((other_canonical_parent and self._norm(other_canonical_parent) == self._norm(canonical)) or
                            (other_canonical_parent_canonical and self._norm(other_canonical_parent_canonical) == self._norm(canonical))):
                            children_canonicals.append(other_canonical)
                            processed_canonicals.add(other_canonical)
                
                if children_canonicals:
                    canonical_hierarchy_map[canonical] = children_canonicals
            
            # Convert back to original requirements for display
            hierarchy_map = {}
            processed_as_children = set()
            
            for parent_canonical, child_canonicals in canonical_hierarchy_map.items():
                # Get original requirements for this parent canonical
                parent_reqs = canonical_to_reqs.get(parent_canonical, [])
                # Use the first one as representative (could be enhanced to pick best)
                if parent_reqs:
                    parent_req = parent_reqs[0]
                    
                    # Get original requirements for child canonicals
                    child_reqs = []
                    for child_canonical in child_canonicals:
                        child_reqs_for_canonical = canonical_to_reqs.get(child_canonical, [])
                        child_reqs.extend(child_reqs_for_canonical)
                        processed_as_children.update(child_reqs_for_canonical)
                    
                    if child_reqs:
                        hierarchy_map[parent_req] = child_reqs
            
            # Standalone requirements (not part of any parent-child relationship)
            standalone_requirements = [req for req in unique_requirements 
                                     if req not in hierarchy_map and req not in processed_as_children]
            
            print(f"  Final hierarchy map: {hierarchy_map}")
            print(f"  Standalone: {standalone_requirements}")
            
            # Create hierarchical visualization
            # 1. Add hierarchical branches (parent -> children)
            for parent_req, child_reqs in hierarchy_map.items():
                # Find the match info for parent requirement
                parent_match = next((sm for sm in category_match.skill_matches 
                                   if sm.matched_requirement == parent_req), None)
                if not parent_match:
                    continue
                
                # Create parent node
                parent_id = f'parent_{node_id}'
                parent_display = parent_req
                
                # Build tooltip based on how parent was satisfied
                if parent_match.match_type == 'child_to_parent':
                    parent_tooltip = f"Parent requirement '{parent_req}' satisfied by student's '{parent_match.student_skill}' skill (child→parent match)"
                elif parent_match.match_type == 'direct':
                    parent_tooltip = f"Parent requirement '{parent_req}' directly matched by student's '{parent_match.student_skill}' skill"
                elif parent_match.match_type == 'alias':
                    parent_tooltip = f"Parent requirement '{parent_req}' matched by student's '{parent_match.student_skill}' skill (alias match)"
                else:
                    parent_tooltip = f"Parent requirement '{parent_req}' satisfied by '{parent_match.student_skill}' ({parent_match.match_type})"
                
                nodes.append({
                    'id': parent_id,
                    'name': parent_display,
                    'type': 'hierarchy_parent',
                    'group': i + 1,
                    'match_type': parent_match.match_type,
                    'requirement': parent_req,
                    'student_skill': parent_match.student_skill,
                    'tooltip': parent_tooltip
                })
                links.append({'source': cat_id, 'target': parent_id, 'value': 3})
                node_id += 1
                
                # Add child nodes connected to parent
                for child_req in child_reqs:
                    child_match = next((sm for sm in category_match.skill_matches 
                                      if sm.matched_requirement == child_req), None)
                    if not child_match:
                        continue
                    
                    child_id = f'child_{node_id}'
                    child_display = child_req
                    
                    # Build tooltip based on how child was satisfied
                    if child_match.match_type == 'child_to_parent':
                        child_tooltip = f"Child requirement '{child_req}' satisfied by student's '{child_match.student_skill}' skill (child→parent match)"
                    elif child_match.match_type == 'direct':
                        child_tooltip = f"Child requirement '{child_req}' directly matched by student's '{child_match.student_skill}' skill"
                    elif child_match.match_type == 'alias':
                        child_tooltip = f"Child requirement '{child_req}' matched by student's '{child_match.student_skill}' skill (alias match)"
                    else:
                        child_tooltip = f"Child requirement '{child_req}' satisfied by '{child_match.student_skill}' ({child_match.match_type})"
                    
                    nodes.append({
                        'id': child_id,
                        'name': child_display,
                        'type': 'hierarchy_child',
                        'group': i + 1,
                        'match_type': child_match.match_type,
                        'requirement': child_req,
                        'student_skill': child_match.student_skill,
                        'tooltip': child_tooltip,
                        'parent_node': parent_id
                    })
                    links.append({'source': parent_id, 'target': child_id, 'value': 2})
                    node_id += 1
            
            # 2. Add standalone requirements (no parent-child relationships in this category)
            for requirement in standalone_requirements:
                req_match = next((sm for sm in category_match.skill_matches 
                               if sm.matched_requirement == requirement), None)
                if not req_match:
                    continue
                
                skill_id = f'skill_{node_id}'
                display_name = requirement
                
                # Build tooltip based on how requirement was satisfied
                if req_match.match_type == 'child_to_parent':
                    tooltip_info = f"Requirement '{requirement}' satisfied by student's '{req_match.student_skill}' skill (child→parent match)"
                elif req_match.match_type == 'direct':
                    tooltip_info = f"Requirement '{requirement}' directly matched by student's '{req_match.student_skill}' skill"
                elif req_match.match_type == 'alias':
                    tooltip_info = f"Requirement '{requirement}' matched by student's '{req_match.student_skill}' skill (alias match)"
                else:
                    tooltip_info = f"Requirement '{requirement}' satisfied by '{req_match.student_skill}' ({req_match.match_type})"
                
                # Determine node type for coloring
                node_type = 'standalone_match'
                if req_match.match_type == 'direct':
                    node_type = 'direct_match'
                elif req_match.match_type == 'child_to_parent':
                    node_type = 'child_to_parent_match'
                elif req_match.match_type == 'alias':
                    node_type = 'alias_match'
                
                nodes.append({
                    'id': skill_id,
                    'name': display_name,
                    'type': node_type,
                    'group': i + 1,
                    'match_type': req_match.match_type,
                    'requirement': requirement,
                    'student_skill': req_match.student_skill,
                    'tooltip': tooltip_info
                })
                links.append({'source': cat_id, 'target': skill_id, 'value': 2})
                node_id += 1
        
        import json
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
            <style>
                #mindmap {{
                    width: 100%;
                    height: 700px;
                    border: 2px solid #ddd;
                    border-radius: 10px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }}
                
                .controls {{
                    margin-bottom: 10px;
                }}
                
                .control-btn {{
                    background: #2E86AB;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    margin: 0 5px;
                    border-radius: 5px;
                    cursor: pointer;
                }}
                
                .tooltip {{
                    position: absolute;
                    text-align: center;
                    padding: 8px;
                    font: 12px sans-serif;
                    background: rgba(0, 0, 0, 0.8);
                    color: white;
                    border-radius: 5px;
                    pointer-events: none;
                    opacity: 0;
                }}
            </style>
        </head>
        <body>
            <div class="controls">
                <button class="control-btn" onclick="restart()">🔄 Restart</button>
                <button class="control-btn" onclick="togglePhysics()">⚡ Physics</button>
                <button class="control-btn" onclick="centerView()">🎯 Center</button>
            </div>
            
            <svg id="mindmap"></svg>
            <div class="tooltip" id="tooltip"></div>
            
            <script>
                const nodes = {json.dumps(nodes, indent=2)};
                const links = {json.dumps(links, indent=2)};
                
                const width = 800;
                const height = 700;
                
                const svg = d3.select("#mindmap")
                    .attr("width", width)
                    .attr("height", height);
                
                const g = svg.append("g");
                
                // Enhanced color scale for hierarchical relationships within categories
                const getNodeColor = (d) => {{
                    if (d.type === 'center') return '#2E86AB';
                    if (d.type === 'category') return d.satisfied ? '#28A745' : '#DC3545';
                    
                    // Hierarchical nodes within categories
                    if (d.type === 'hierarchy_parent') return '#FFC107';  // Gold for parent in hierarchy
                    if (d.type === 'hierarchy_child') return '#17A2B8';   // Blue for child in hierarchy
                    
                    // Color by match type for standalone requirements
                    if (d.match_type === 'direct') return '#28A745';  // Green for direct
                    if (d.match_type === 'child_to_parent') return '#17A2B8';  // Blue for child→parent
                    if (d.match_type === 'alias') return '#FD7E14';        // Orange for alias
                    if (d.match_type === 'llm_enhanced') return '#DC3545';  // Red for AI
                    
                    // Fallback colors for node types
                    if (d.type === 'direct_match') return '#28A745';
                    if (d.type === 'child_to_parent_match') return '#17A2B8';
                    if (d.type === 'alias_match') return '#FD7E14';
                    if (d.type === 'standalone_match') return '#6C757D';
                    
                    return '#6C757D';  // Default gray
                }};
                
                // Force simulation
                let simulation = d3.forceSimulation(nodes)
                    .force("link", d3.forceLink(links).id(d => d.id).distance(120))
                    .force("charge", d3.forceManyBody().strength(-400))
                    .force("center", d3.forceCenter(width / 2, height / 2))
                    .force("collision", d3.forceCollide().radius(40));
                
                // Links with enhanced styling
                const link = g.append("g")
                    .selectAll("line")
                    .data(links)
                    .enter().append("line")
                    .attr("stroke", "rgba(255, 255, 255, 0.6)")
                    .attr("stroke-width", d => d.value * 2)
                    .attr("stroke-opacity", 0.8);
                
                // Enhanced nodes
                const node = g.append("g")
                    .selectAll("g")
                    .data(nodes)
                    .enter().append("g")
                    .call(d3.drag()
                        .on("start", dragstarted)
                        .on("drag", dragged)
                        .on("end", dragended));
                
                // Node circles with hierarchical sizing
                node.append("circle")
                    .attr("r", d => {{
                        if (d.type === 'center') return 35;
                        if (d.type === 'category') return 28;
                        if (d.type === 'hierarchy_parent') return 24;  // Larger for parent nodes
                        if (d.type === 'hierarchy_child') return 20;   // Medium for child nodes
                        return 18;  // Standard size for standalone nodes
                    }})
                    .attr("fill", getNodeColor)
                    .attr("stroke", "white")
                    .attr("stroke-width", 3)
                    .attr("opacity", 0.9);
                
                // Enhanced node labels
                node.append("text")
                    .attr("dy", 0.35)
                    .attr("text-anchor", "middle")
                    .attr("fill", "white")
                    .attr("font-size", d => {{
                        if (d.type === 'center') return "14px";
                        if (d.type === 'category') return "11px";
                        return "9px";
                    }})
                    .attr("font-weight", "bold")
                    .text(d => d.name.length > 15 ? d.name.substring(0, 15) + "..." : d.name);
                
                // Enhanced tooltip with match type information
                const tooltip = d3.select("#tooltip");
                
                node.on("mouseover", function(event, d) {{
                    let tooltipText = `<strong>${{d.name}}</strong><br>`;
                    tooltipText += `Type: ${{d.type.replace('_', ' ')}}<br>`;
                    
                    if (d.match_type) {{
                        const matchTypeLabels = {{
                            'direct': 'Direct Match ✅',
                            'child_to_parent': 'Child→Parent Match 🔗 (Student skill satisfies parent requirement)',
                            'alias': 'Alias Match 🔄',
                            'llm_enhanced': 'AI Enhanced Match 🤖'
                        }};
                        tooltipText += `Match: ${{matchTypeLabels[d.match_type] || d.match_type}}<br>`;
                    }}
                    
                    if (d.type === 'hierarchy_parent') {{
                        tooltipText += `Role: Parent in category hierarchy 👑<br>`;
                    }}
                    
                    if (d.type === 'hierarchy_child') {{
                        tooltipText += `Role: Child in category hierarchy 🌿<br>`;
                    }}
                    
                    if (d.student_skill) {{
                        tooltipText += `Student Skill: ${{d.student_skill}}<br>`;
                    }}
                    
                    if (d.requirement) {{
                        tooltipText += `Satisfied Requirement: ${{d.requirement}}<br>`;
                    }}
                    
                    if (d.parent_skill) {{
                        tooltipText += `Parent Skill: ${{d.parent_skill}}<br>`;
                    }}
                    
                    if (d.tooltip) {{
                        tooltipText += `Info: ${{d.tooltip}}<br>`;
                    }}
                    
                    if (d.satisfied !== undefined) {{
                        tooltipText += `Status: ${{d.satisfied ? 'Satisfied ✅' : 'Not Satisfied ❌'}}<br>`;
                    }}
                    
                    tooltip.html(tooltipText)
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 10) + "px")
                        .style("opacity", 1);
                }})
                .on("mouseout", function() {{
                    tooltip.style("opacity", 0);
                }});
                
                // Animation on load
                node.selectAll("circle")
                    .attr("r", 0)
                    .transition()
                    .duration(1000)
                    .delay((d, i) => i * 50)
                    .attr("r", d => {{
                        if (d.type === 'center') return 35;
                        if (d.type === 'category') return 28;
                        if (d.type === 'parent_skill') return 22;
                        return 18;
                    }});
                
                // Simulation tick
                simulation.on("tick", () => {{
                    link
                        .attr("x1", d => d.source.x)
                        .attr("y1", d => d.source.y)
                        .attr("x2", d => d.target.x)
                        .attr("y2", d => d.target.y);
                    
                    node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
                }});
                
                // Drag functions
                function dragstarted(event) {{
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    event.subject.fx = event.subject.x;
                    event.subject.fy = event.subject.y;
                }}
                
                function dragged(event) {{
                    event.subject.fx = event.x;
                    event.subject.fy = event.y;
                }}
                
                function dragended(event) {{
                    if (!event.active) simulation.alphaTarget(0);
                    event.subject.fx = null;
                    event.subject.fy = null;
                }}
                
                // Control functions
                function restart() {{
                    simulation.alpha(1).restart();
                }}
                
                let physicsEnabled = true;
                function togglePhysics() {{
                    if (physicsEnabled) {{
                        simulation.stop();
                        physicsEnabled = false;
                    }} else {{
                        simulation.restart();
                        physicsEnabled = true;
                    }}
                }}
                
                function centerView() {{
                    const bounds = g.node().getBBox();
                    const parent = g.node().parentElement;
                    const fullWidth = parent.clientWidth;
                    const fullHeight = parent.clientHeight;
                    const width = bounds.width;
                    const height = bounds.height;
                    const midX = bounds.x + width / 2;
                    const midY = bounds.y + height / 2;
                    
                    const scale = 0.75 / Math.max(width / fullWidth, height / fullHeight);
                    const translate = [fullWidth / 2 - scale * midX, fullHeight / 2 - scale * midY];
                    
                    svg.transition()
                        .duration(750)
                        .call(d3.zoom().transform, d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale));
                }}
                
                // Add zoom functionality
                svg.call(d3.zoom()
                    .extent([[0, 0], [width, height]])
                    .scaleExtent([0.1, 8])
                    .on("zoom", function(event) {{
                        g.attr("transform", event.transform);
                    }}));
            </script>
        </body>
        </html>
        """
        
        return html_content

# ---- Rest of the Application ----

# CSV Parser functions remain the same
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
st.set_page_config(page_title="Enhanced Hierarchical Student-Job Matcher", layout="wide")
st.title("🎓 Enhanced Hierarchical Student-Job Matching System 🧠")

# Sidebar - CSV Upload
st.sidebar.header("Step 1: Upload Student CSV")
students_csv = st.sidebar.file_uploader("Students CSV", type="csv")
students = []
if students_csv:
    students = parse_student_csv(students_csv)
    st.sidebar.success(f"{len(students)} students loaded!")

# Sidebar - Job Roles
st.sidebar.header("Step 2: Define Job Roles")

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


if "enhanced_roles" not in st.session_state:
    st.session_state["enhanced_roles"] = default_roles

enhanced_roles = st.session_state["enhanced_roles"]

def add_enhanced_role():
    enhanced_roles.append({
        "job_title": "",
        "relevant_degrees": "",
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
        tech_stack_str = st.text_area(
            f"Technical Stack {idx+1} (JSON)",
            value=json.dumps(role.get("technical_stack", {}), indent=2),
            key=f"enh_tech_stack_{idx}",
            help='Example:\n{\n  "Programming Language": {"options": ["python"], "mandatory": true, "min_match": 1}\n}'
        )

        enhanced_roles[idx].update({
            "job_title": job_title,
            "relevant_degrees": relevant_degrees
        })
        try:
            enhanced_roles[idx]["technical_stack"] = json.loads(tech_stack_str)
        except Exception as e:
            st.warning(f"Invalid Technical Stack JSON for role {job_title or idx+1}: {e}")

        if st.button(f"Remove Role {idx+1}", key=f"enh_remove_{idx}"):
            remove_enhanced_role(idx)
            st.rerun()

st.sidebar.button("Add New Role", on_click=add_enhanced_role)

# LLM Configuration
st.sidebar.header("Step 3: AI Configuration (Optional)")
use_llm = st.sidebar.checkbox("Enable AI for enhanced skill matching", value=False)
openai_api_key = None
if use_llm:
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.sidebar.warning("Please provide your OpenAI API key to use AI-enhanced skill matching.")
        use_llm = False

# Create job roles
job_roles: List[InternshipRequirement] = []
for role in enhanced_roles:
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

# Session state management
current_fp = compute_config_fingerprint(students, job_roles)

if "enh_last_run_fp" not in st.session_state:
    st.session_state["enh_last_run_fp"] = None
if "enh_run_clicked_at" not in st.session_state:
    st.session_state["enh_run_clicked_at"] = None
if "enh_matrix_rows" not in st.session_state:
    st.session_state["enh_matrix_rows"] = None
if "enh_all_matches" not in st.session_state:
    st.session_state["enh_all_matches"] = None

st.sidebar.header("Step 4: Run Enhanced Matching")
if st.session_state["enh_last_run_fp"] and st.session_state["enh_last_run_fp"] != current_fp:
    st.sidebar.warning("Inputs changed since last run. Click **Run Enhanced Matching** to update results.")

run_btn = st.sidebar.button("▶ Run Enhanced Matching")

if run_btn:
    st.session_state["enh_last_run_fp"] = current_fp
    st.session_state["enh_run_clicked_at"] = time.time()
    st.session_state["enh_matrix_rows"] = None
    st.session_state["enh_all_matches"] = None

# Main matching logic
matcher = EnhancedInternshipMatcher(openai_api_key=openai_api_key, use_llm=use_llm)

ready_to_run = (
    students
    and job_roles
    and st.session_state["enh_last_run_fp"] == current_fp
    and st.session_state["enh_run_clicked_at"] is not None
)

if ready_to_run:
    st.header("🎯 Enhanced Hierarchical Matching Results")

    # Use cached compute when available
    if st.session_state["enh_matrix_rows"] is None or st.session_state["enh_all_matches"] is None:
        with st.spinner("Processing enhanced hierarchical student-job matches..."):
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

                    # Enhanced skill display with hierarchy info
                    matched_skills_display = []
                    for category_match in result.matched_skills_by_category:
                        category_display = category_match.get_display_string()
                        if category_display:
                            matched_skills_display.append(f"**{category_match.category_name}**: {category_display}")
                    
                    row[f"{role.job_title} - Matched Skills"] = " | ".join(matched_skills_display)
                    row[f"{role.job_title} - Missing Skills"] = ", ".join(result.missing_skills[:10])
                    row[f"{role.job_title} - Reasoning"] = result.overall_reasoning

                    processed += 1
                    progress_bar.progress(processed / total_combinations)

                all_matches[student.email] = student_results
                matrix_rows.append(row)

            progress_bar.empty()
            st.session_state["enh_all_matches"] = all_matches
            st.session_state["enh_matrix_rows"] = matrix_rows

    # Display results
    df_results = pd.DataFrame(st.session_state["enh_matrix_rows"])
    st.dataframe(df_results, use_container_width=True)

    all_matches = st.session_state.get("enh_all_matches") or {}
    if not all_matches:
        st.info("No results cached yet. Click **Run Enhanced Matching** after setting inputs.")
        st.stop()

    # Enhanced detailed analysis
    st.header("🔍 Enhanced Detailed Student Analysis")

    valid_emails = [s.email for s in students if s.email in all_matches]
    if not valid_emails:
        st.info("No evaluated students found. Please re-run evaluation.")
        st.stop()

    selected_email = st.selectbox(
        "Select a student for detailed hierarchical analysis",
        valid_emails,
        format_func=lambda email: f"{next((s.name for s in students if s.email == email), email)} ({email})"
    )

    if selected_email:
        selected_student = next((s for s in students if s.email == selected_email), None)
        selected_results = all_matches[selected_email]

        st.subheader(f"🎓 Enhanced Analysis: {selected_student.name}")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write("**Student Profile:**")
            st.write(f"📚 Education: {selected_student.education}")
            st.write(f"🎓 Degree: {selected_student.degree}")
            st.write(f"💼 Skills: {', '.join(selected_student.skills)}")
            
            # Show skill hierarchy analysis
            st.write("**🌳 Skill Hierarchy Analysis:**")
            parent_skills = set()
            child_skills = set()
            for skill in selected_student.skills:
                parent = matcher.get_immediate_parent(skill)
                if parent:
                    parent_skills.add(f"{parent} (via {skill})")
                    child_skills.add(skill)
            
            if parent_skills:
                st.write(f"🔗 **Implied Parent Skills**: {', '.join(list(parent_skills)[:5])}")
            if child_skills:
                st.write(f"🌱 **Child Skills**: {', '.join(list(child_skills)[:5])}")
        
        with col2:
            role_scores = [(role.job_title, selected_results[role.job_title].score) for role in job_roles]
            role_scores.sort(key=lambda x: x[1], reverse=True)
            st.write("**🏆 Best Role Matches:**")
            for i, (role_name, score) in enumerate(role_scores[:3]):
                st.write(f"{i+1}. {role_name}: {score:.3f}")

        selected_role_name = st.selectbox("Select role for detailed hierarchical breakdown", [role.job_title for role in job_roles])
        if selected_role_name:
            result = selected_results[selected_role_name]
            st.subheader(f"🎯 Enhanced Analysis for {selected_role_name}")

            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Overall Score", f"{result.score:.3f}")
            with c2: st.metric("Stack Coverage", f"{result.stack_coverage_score:.3f}")
            with c3: st.metric("Education Match", f"{result.education_match_score:.3f}")

            # Enhanced categorized skills display with detailed hierarchy
            st.write("**🎯 Enhanced Skill Matches with Hierarchy:**")
            if result.matched_skills_by_category:
                for category_match in result.matched_skills_by_category:
                    st.success(f"**📂 {category_match.category_name}:** {category_match.get_display_string()}")
                
                # Enhanced detailed hierarchy breakdown
                with st.expander("🔬 Detailed Hierarchical Match Analysis"):
                    for category_match in result.matched_skills_by_category:
                        st.write(f"**📂 Category: {category_match.category_name}**")
                        
                        for skill_match in category_match.skill_matches:
                            match_type_icons = {
                                'direct': '🎯',
                                'parent_match': '🔗',
                                'child_match': '🌱',
                                'alias': '🔄',
                                'llm_enhanced': '🤖'
                            }
                            
                            icon = match_type_icons.get(skill_match.match_type, '📍')
                            
                            if skill_match.match_type == 'direct':
                                st.write(f"   {icon} **Direct Match**: {skill_match.student_skill} ↔ {skill_match.matched_requirement}")
                            elif skill_match.match_type == 'parent_match':
                                st.write(f"   {icon} **Parent→Child**: {skill_match.parent_skill} → {skill_match.student_skill} (satisfies {skill_match.matched_requirement})")
                            elif skill_match.match_type == 'child_match':
                                st.write(f"   {icon} **Child→Parent**: {skill_match.student_skill} → satisfies {skill_match.matched_requirement}")
                            elif skill_match.match_type == 'alias':
                                st.write(f"   {icon} **Alias Match**: {skill_match.student_skill} ≈ {skill_match.matched_requirement}")
                            elif skill_match.match_type == 'llm_enhanced':
                                st.write(f"   {icon} **AI Enhanced**: {skill_match.student_skill} (AI-detected relevance)")
                            
                            if skill_match.category_metadata:
                                st.write(f"       📋 *Category Metadata: {skill_match.category_metadata}*")
                        
                        st.write("")
            else:
                st.warning("No hierarchical skill matches found for this role.")

            # Technical stack breakdown with match type analysis
            st.write("**⚙️ Enhanced Technical Stack Analysis:**")
            for cat, d in result.stack_details.items():
                status = "✅ Met" if d.get("met") else "❌ Not Met"
                llm_indicator = " 🤖" if d.get("llm_enhanced") else ""
                st.write(f"- **{cat}** ({'Mandatory' if d.get('mandatory') else 'Optional'}, need {d.get('min_match',1)}): {status}{llm_indicator}")
                
                matched_skills = d.get('matched', [])
                if matched_skills:
                    # Analyze match types in matched skills
                    direct_matches = [s for s in matched_skills if '→' not in s and 'satisfies' not in s and '(AI-' not in s]
                    hierarchy_matches = [s for s in matched_skills if '→' in s or 'satisfies' in s]
                    ai_matches = [s for s in matched_skills if '(AI-' in s]
                    
                    if direct_matches:
                        st.write(f"    🎯 **Direct**: {', '.join(direct_matches)}")
                    if hierarchy_matches:
                        st.write(f"    🔗 **Hierarchical**: {', '.join(hierarchy_matches)}")
                    if ai_matches:
                        st.write(f"    🤖 **AI Enhanced**: {', '.join(ai_matches)}")
                else:
                    st.write(f"    📝 **Matched**: —")
                
                missing = d.get('missing', [])
                if missing:
                    st.write(f"    ❌ **Missing**: {', '.join(missing)}")

            # Enhanced reasoning
            if result.overall_reasoning:
                st.write("**🧠 Enhanced Assessment:**")
                st.info(result.overall_reasoning)

            # Enhanced mind map
            st.subheader(f"🧠 Enhanced Hierarchical Mind Map: {selected_role_name}")
            
            enhanced_html = matcher.create_enhanced_mindmap_html(result, selected_role_name)
            
            import streamlit.components.v1 as components
            components.html(enhanced_html, height=720, width=720, scrolling=False)
            
            st.success("🎯 Enhanced hierarchical mind map with internal category hierarchy!")
            st.write("**Node Types:**")
            st.write("- 🔵 **Blue (Center)**: Job Role")
            st.write("- 🟢 **Green (Categories)**: Requirement categories")
            st.write("- 🟡 **Gold**: Parent requirements in category hierarchy")
            st.write("- 🔵 **Blue**: Child requirements in category hierarchy")
            st.write("- 🟢 **Green**: Direct matches")
            st.write("- 🟠 **Orange**: Alias matches")
            st.write("- 🔴 **Red**: AI-enhanced matches")
            st.write("")
            st.write("**Hierarchy Logic:**")
            st.write("- Within each category, if both parent and child requirements are satisfied")
            st.write("- They are connected: Category → Parent Requirement → Child Requirement")
            st.write("- Shows clear skill progression and relationships")

    # Enhanced Export
    st.header("📊 Export Enhanced Results")
    csv_content = df_results.to_csv(index=False)
    st.download_button(
        label="📊 Download Enhanced Results as CSV",
        data=csv_content,
        file_name=f"enhanced_hierarchical_student_job_matching_{time.strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

else:
    # Guide the user
    if not students:
        st.info("Upload student CSV to begin enhanced hierarchical matching.")
    elif not job_roles:
        st.info("Define at least one job role with a technical stack.")
    else:
        st.info("Ready to run enhanced matching. Click **Run Enhanced Matching** in the sidebar.")

# Enhanced Debug section
st.sidebar.header("🔍 Enhanced Debug Information")
if st.sidebar.checkbox("Show Enhanced Debug Info"):
    st.write("### 🧬 Enhanced Skill Hierarchy Analysis")
    
    if students:
        st.write("**Sample Student Skills with Hierarchy:**")
        sample_student = students[0] if students else None
        if sample_student:
            st.write(f"**Student**: {sample_student.name}")
            for skill in sample_student.skills[:5]:  # Show first 5 skills
                parent = matcher.get_immediate_parent(skill)
                children = matcher.get_all_children(skill)
                canonical = matcher.skill_to_canonical.get(matcher._norm(skill), skill)
                
                hierarchy_info = []
                if parent:
                    hierarchy_info.append(f"Parent: {parent}")
                if children:
                    hierarchy_info.append(f"Children: {', '.join(children[:3])}")
                if canonical != skill:
                    hierarchy_info.append(f"Canonical: {canonical}")
                
                if hierarchy_info:
                    st.write(f"- **{skill}** → {' | '.join(hierarchy_info)}")
                else:
                    st.write(f"- **{skill}** → Root skill (no hierarchy)")
    
    if job_roles:
        st.write("**Sample Job Requirements with Expected Hierarchy:**")
        sample_role = job_roles[0] if job_roles else None
        if sample_role:
            st.write(f"**Role**: {sample_role.job_title}")
            for category, req_data in sample_role.technical_stack.items():
                st.write(f"- **{category}**: {req_data.get('options', [])}")
                
                # Show what these requirements could match
                for req_skill in req_data.get('options', [])[:3]:  # First 3 requirements
                    children = matcher.get_all_children(req_skill)
                    parent = matcher.get_immediate_parent(req_skill)
                    if children:
                        st.write(f"    └─ **{req_skill}** can be satisfied by: {', '.join(children[:5])}")
                    elif parent:
                        st.write(f"    └─ **{req_skill}** is child of: {parent}")
    
    # Show enhanced skill taxonomy info
    if matcher.skill_to_canonical:
        st.write("**Enhanced Skill Normalizations (Sample):**")
        sample_mappings = list(matcher.skill_to_canonical.items())[:15]
        for orig, canonical in sample_mappings:
            parent = matcher.get_immediate_parent(canonical)
            children = matcher.get_all_children(canonical)
            
            info_parts = [f"'{orig}' → '{canonical}'"]
            if parent:
                info_parts.append(f"Parent: {parent}")
            if children:
                info_parts.append(f"Children: {len(children)}")
            
            st.write(f"- {' | '.join(info_parts)}")

# Enhanced Instructions
with st.expander("📋 Enhanced Hierarchical Matching Instructions"):
    st.markdown("""
    ## 🔧 KEY ENHANCED FEATURES:
    
    ### 🌳 **Hierarchical Skill Matching**
    - **Parent→Child Matching**: If student has "Flask" and job requires "Python", it matches (Flask is child of Python)
    - **Child→Parent Matching**: If student has "Python" and job requires "Flask", it matches (Python can satisfy Flask requirement)
    - **Direct Matching**: Exact skill matches (e.g., "React" ↔ "React")
    - **Alias Matching**: Different names for same skill (e.g., "React.js" ↔ "React")
    - **AI Enhanced Matching**: Optional AI analysis for transferable skills
    
    ### 🎯 **Enhanced Display Features**
    - **Detailed Match Types**: See exactly how each skill matches (direct, hierarchical, alias, AI)
    - **Visual Hierarchy**: Mind map shows parent-child relationships with different colors
    - **Category Metadata**: Requirements are organized by category with skill context
    - **Match Chain Display**: See full satisfaction chains (e.g., "Python → Django → REST API")
    
    ### 🧠 **Enhanced Mind Map Legend**
    - 🔵 **Center (Blue)**: Job Role
    - 🟢 **Green Categories**: Requirement categories (satisfied)
    - 🔴 **Red Categories**: Requirement categories (not satisfied)
    - 🟡 **Yellow**: Parent skills
    - 🔵 **Blue**: Parent→Child matches (student has child, satisfies parent requirement)
    - 🟣 **Purple**: Child→Parent matches (student has parent, satisfies child requirement)
    - 🟢 **Green**: Direct matches
    - 🟠 **Orange**: Alias matches
    - 🔴 **Red**: AI-enhanced matches
    
    ### 📊 **Enhanced Matching Logic**
    
    **Example Scenario:**
    - **Student Skills**: ["Flask", "SQLAlchemy", "React.js", "Git"]
    - **Job Requirements**: 
      - Programming Language: ["python"] (mandatory)
      - Frontend: ["react"] (mandatory)
      - Database: ["sql"] (optional)
      - Version Control: ["version control"] (optional)
    
    **Enhanced Matching Results:**
    - ✅ **Python**: Satisfied by "Flask" (child→parent match)
    - ✅ **React**: Satisfied by "React.js" (alias match)  
    - ✅ **SQL**: Satisfied by "SQLAlchemy" (child→parent match)
    - ✅ **Version Control**: Satisfied by "Git" (child→parent match)
    
    **Hierarchy Display:**
    - Programming Language: Python (→ Flask)
    - Frontend: React (≈ React.js)
    - Database: SQL (→ SQLAlchemy)
    - Version Control: Version Control (→ Git)
    
    ### 🔍 **Debug Features**
    - **Skill Hierarchy Analysis**: See parent-child relationships for student skills
    - **Requirement Analysis**: Understand what job requirements can match
    - **Enhanced Normalization**: View canonical forms and aliases
    - **Match Type Breakdown**: Detailed analysis of how each skill matches
    
    ### 📝 **Required CSV Format**
    Same as before, but now the system will:
    - Automatically detect parent-child relationships
    - Show detailed match types in results
    - Provide hierarchical visualization
    - Display skill satisfaction chains
    
    ### 🎯 **Enhanced Benefits**
    1. **More Accurate Matching**: Considers skill relationships, not just exact matches
    2. **Better Insights**: See how skills relate to requirements
    3. **Visual Understanding**: Mind map shows skill hierarchy and relationships
    4. **Detailed Analysis**: Understand exactly why students match or don't match
    5. **Future-Proof**: System grows with your skill taxonomy
    
    **Testing Enhanced Matching:**
    1. Upload CSV with hierarchical skills (e.g., "Flask", "Django", "React.js")
    2. Create job requiring parent skills (e.g., "python", "react")
    3. Run enhanced matching - see hierarchical relationships in action!
    4. Explore mind map to visualize skill relationships
    5. Check detailed analysis for match type breakdown
    """)

st.markdown("---")
st.caption("Enhanced Hierarchical Student-Job Matcher with Parent-Child Skill Relationships 🚀 🌳 ✅")