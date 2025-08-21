sample_response = {"matching_results": {
    "python_backend_intern": {
      "job_title": "Python Backend Intern",
      "overall_score": 0.875,
      "stack_pass": True,
      "stack_coverage_score": 0.92,
      "education_match_score": 1.0,
      "scoring_breakdown": {
        "technical_weight": 0.6,
        "education_weight": 0.4,
        "calculation": "(0.92 * 0.6) + (1.0 * 0.4) = 0.875"
      },
      "matched_skills_by_category": [
        {
          "category_name": "Programming Language",
          "required_skills": [
            {
              "id": "sk_001python",
              "name": "python"
            }
          ],
          "matched_skills": [
            {
              "id": "sk_001python",
              "name": "Python",
              "original_input": "Python",
              "match_type": "exact"
            }
          ],
          "match_score": 1.0,
          "hierarchy_trees": [
            {
              "skill_id": "sk_root_prog_lang",
              "skill_name": "Programming Languages",
              "children": [
                {
                  "skill_id": "sk_001python",
                  "skill_name": "Python",
                  "is_matched": True,
                  "children": []
                }
              ],
              "is_matched": False
            }
          ],
          "standalone_skills": []
        },
        {
          "category_name": "Frameworks",
          "required_skills": [
            {
              "id": "sk_006django",
              "name": "django"
            },
            {
              "id": "sk_007flask",
              "name": "flask"
            },
            {
              "id": "sk_017fastapi",
              "name": "fastapi"
            }
          ],
          "matched_skills": [
            {
              "id": "sk_006django",
              "name": "Django",
              "original_input": "Django",
              "match_type": "exact"
            },
            {
              "id": "sk_007flask",
              "name": "Flask",
              "original_input": "Flask",
              "match_type": "exact"
            }
          ],
          "match_score": 0.67,
          "hierarchy_trees": [
            {
              "skill_id": "sk_root_web_fw",
              "skill_name": "Web Frameworks",
              "children": [
                {
                  "skill_id": "sk_python_fw",
                  "skill_name": "Python Frameworks",
                  "children": [
                    {
                      "skill_id": "sk_006django",
                      "skill_name": "Django",
                      "is_matched": True,
                      "children": []
                    },
                    {
                      "skill_id": "sk_007flask",
                      "skill_name": "Flask",
                      "is_matched": True,
                      "children": []
                    }
                  ],
                  "is_matched": False
                }
              ],
              "is_matched": False
            }
          ],
          "standalone_skills": []
        },
        {
          "category_name": "Databases",
          "required_skills": [
            {
              "id": "sk_005postgres",
              "name": "postgresql"
            },
            {
              "id": "sk_016mysql",
              "name": "mysql"
            },
            {
              "id": "sk_018sqlite",
              "name": "sqlite"
            },
            {
              "id": "sk_004mongo",
              "name": "mongodb"
            }
          ],
          "matched_skills": [
            {
              "id": "sk_005postgres",
              "name": "PostgreSQL",
              "original_input": "PostgreSQL",
              "match_type": "exact"
            },
            {
              "id": "sk_016mysql",
              "name": "MySQL",
              "original_input": "MySQL",
              "match_type": "exact"
            },
            {
              "id": "sk_004mongo",
              "name": "MongoDB",
              "original_input": "MongoDB",
              "match_type": "exact"
            }
          ],
          "match_score": 0.75,
          "hierarchy_trees": [
            {
              "skill_id": "sk_root_db",
              "skill_name": "Database Systems",
              "children": [
                {
                  "skill_id": "sk_sql_db",
                  "skill_name": "Relational Databases",
                  "children": [
                    {
                      "skill_id": "sk_005postgres",
                      "skill_name": "PostgreSQL",
                      "is_matched": True,
                      "children": []
                    },
                    {
                      "skill_id": "sk_016mysql",
                      "skill_name": "MySQL",
                      "is_matched": True,
                      "children": []
                    }
                  ],
                  "is_matched": False
                },
                {
                  "skill_id": "sk_nosql_db",
                  "skill_name": "NoSQL Databases",
                  "children": [
                    {
                      "skill_id": "sk_004mongo",
                      "skill_name": "MongoDB",
                      "is_matched": True,
                      "children": []
                    }
                  ],
                  "is_matched": False
                }
              ],
              "is_matched": False
            }
          ],
          "standalone_skills": []
        },
        {
          "category_name": "Tools & Platform (Optional)",
          "required_skills": [
            {
              "id": "sk_009docker",
              "name": "docker"
            },
            {
              "id": "sk_019k8s",
              "name": "kubernetes"
            },
            {
              "id": "sk_010aws",
              "name": "aws"
            },
            {
              "id": "sk_020gcp",
              "name": "gcp"
            },
            {
              "id": "sk_021azure",
              "name": "azure"
            },
            {
              "id": "sk_008git",
              "name": "git"
            }
          ],
          "matched_skills": [
            {
              "id": "sk_009docker",
              "name": "Docker",
              "original_input": "Docker",
              "match_type": "exact"
            },
            {
              "id": "sk_010aws",
              "name": "AWS",
              "original_input": "AWS",
              "match_type": "exact"
            },
            {
              "id": "sk_008git",
              "name": "Git",
              "original_input": "Git",
              "match_type": "exact"
            }
          ],
          "match_score": 0.5,
          "hierarchy_trees": [
            {
              "skill_id": "sk_root_dev",
              "skill_name": "Software Development",
              "children": [
                {
                  "skill_id": "sk_vcs",
                  "skill_name": "Version Control Systems",
                  "children": [
                    {
                      "skill_id": "sk_008git",
                      "skill_name": "Git",
                      "is_matched": True,
                      "children": []
                    }
                  ],
                  "is_matched": False
                },
                {
                  "skill_id": "sk_container",
                  "skill_name": "Containerization",
                  "children": [
                    {
                      "skill_id": "sk_009docker",
                      "skill_name": "Docker",
                      "is_matched": True,
                      "children": []
                    }
                  ],
                  "is_matched": False
                }
              ],
              "is_matched": False
            },
            {
              "skill_id": "sk_cloud",
              "skill_name": "Cloud Platforms",
              "children": [
                {
                  "skill_id": "sk_010aws",
                  "skill_name": "AWS",
                  "is_matched": True,
                  "children": []
                }
              ],
              "is_matched": False
            }
          ],
          "standalone_skills": []
        }
      ],
      "missing_skills": [
        {
          "id": "sk_017fastapi",
          "name": "FastAPI",
          "category": "Frameworks"
        },
        {
          "id": "sk_019k8s",
          "name": "Kubernetes",
          "category": "Tools & Platform (Optional)"
        }
      ],
      "category_requirements": {
        "Programming Language": {
          "mandatory": True,
          "min_match": 1,
          "met": True,
          "matched_count": 1
        },
        "Frameworks": {
          "mandatory": True,
          "min_match": 1,
          "met": True,
          "matched_count": 2
        },
        "Databases": {
          "mandatory": True,
          "min_match": 1,
          "met": True,
          "matched_count": 3
        },
        "Tools & Platform (Optional)": {
          "mandatory": False,
          "min_match": 1,
          "met": True,
          "matched_count": 3
        }
      },
      "overall_reasoning": "Excellent match (Score: 0.875). Strengths: strong technical skills, relevant educational background. Student demonstrates proficiency in Python ecosystem with multiple frameworks and database experience."
    },
    "mern_stack_intern": {
      "job_title": "MERN Stack Intern",
      "overall_score": 0.783,
      "stack_pass": True,
      "stack_coverage_score": 0.83,
      "education_match_score": 1.0,
      "scoring_breakdown": {
        "technical_weight": 0.6,
        "education_weight": 0.4,
        "calculation": "(0.83 * 0.6) + (1.0 * 0.4) = 0.783"
      },
      "matched_skills_by_category": [
        {
          "category_name": "Frontend",
          "required_skills": [
            {
              "id": "sk_11223jklmn",
              "name": "react"
            },
            {
              "id": "sk_011javascript",
              "name": "javascript"
            },
            {
              "id": "sk_012html",
              "name": "html"
            },
            {
              "id": "sk_013css",
              "name": "css"
            }
          ],
          "matched_skills": [
            {
              "id": "sk_11223jklmn",
              "name": "React",
              "original_input": "React.js",
              "match_type": "alias"
            },
            {
              "id": "sk_011javascript",
              "name": "JavaScript",
              "original_input": "JavaScript",
              "match_type": "exact"
            },
            {
              "id": "sk_012html",
              "name": "HTML",
              "original_input": "HTML",
              "match_type": "exact"
            },
            {
              "id": "sk_013css",
              "name": "CSS",
              "original_input": "CSS",
              "match_type": "exact"
            }
          ],
          "match_score": 1.0,
          "hierarchy_trees": [
            {
              "skill_id": "sk_web_dev",
              "skill_name": "Web Development",
              "children": [
                {
                  "skill_id": "sk_frontend",
                  "skill_name": "Frontend Technologies",
                  "children": [
                    {
                      "skill_id": "sk_011javascript",
                      "skill_name": "JavaScript",
                      "children": [
                        {
                          "skill_id": "sk_11223jklmn",
                          "skill_name": "React",
                          "is_matched": True,
                          "children": []
                        }
                      ],
                      "is_matched": True
                    },
                    {
                      "skill_id": "sk_012html",
                      "skill_name": "HTML",
                      "is_matched": True,
                      "children": []
                    },
                    {
                      "skill_id": "sk_013css",
                      "skill_name": "CSS",
                      "is_matched": True,
                      "children": []
                    }
                  ],
                  "is_matched": False
                }
              ],
              "is_matched": False
            }
          ],
          "standalone_skills": []
        },
        {
          "category_name": "Backend",
          "required_skills": [
            {
              "id": "sk_002nodejs",
              "name": "nodejs"
            },
            {
              "id": "sk_003express",
              "name": "express"
            }
          ],
          "matched_skills": [
            {
              "id": "sk_002nodejs",
              "name": "Node.js",
              "original_input": "Node.js",
              "match_type": "exact"
            },
            {
              "id": "sk_003express",
              "name": "Express.js",
              "original_input": "Express.js",
              "match_type": "exact"
            }
          ],
          "match_score": 1.0,
          "hierarchy_trees": [
            {
              "skill_id": "sk_011javascript",
              "skill_name": "JavaScript",
              "children": [
                {
                  "skill_id": "sk_002nodejs",
                  "skill_name": "Node.js",
                  "children": [
                    {
                      "skill_id": "sk_003express",
                      "skill_name": "Express.js",
                      "is_matched": True,
                      "children": []
                    }
                  ],
                  "is_matched": True
                }
              ],
              "is_matched": False
            }
          ],
          "standalone_skills": []
        },
        {
          "category_name": "Database",
          "required_skills": [
            {
              "id": "sk_004mongo",
              "name": "mongodb"
            },
            {
              "id": "sk_022mongoose",
              "name": "mongoose"
            }
          ],
          "matched_skills": [
            {
              "id": "sk_004mongo",
              "name": "MongoDB",
              "original_input": "MongoDB",
              "match_type": "exact"
            }
          ],
          "match_score": 0.5,
          "hierarchy_trees": [
            {
              "skill_id": "sk_nosql_db",
              "skill_name": "NoSQL Databases",
              "children": [
                {
                  "skill_id": "sk_004mongo",
                  "skill_name": "MongoDB",
                  "is_matched": True,
                  "children": []
                }
              ],
              "is_matched": False
            }
          ],
          "standalone_skills": []
        },
        {
          "category_name": "State Management",
          "required_skills": [
            {
              "id": "sk_014redux",
              "name": "redux"
            },
            {
              "id": "sk_023context",
              "name": "context api"
            },
            {
              "id": "sk_024zustand",
              "name": "zustand"
            }
          ],
          "matched_skills": [
            {
              "id": "sk_014redux",
              "name": "Redux",
              "original_input": "Redux",
              "match_type": "exact"
            }
          ],
          "match_score": 0.33,
          "hierarchy_trees": [],
          "standalone_skills": [
            {
              "id": "sk_014redux",
              "name": "Redux"
            }
          ]
        },
        {
          "category_name": "Tools & Others (Optional)",
          "required_skills": [
            {
              "id": "sk_008git",
              "name": "git"
            },
            {
              "id": "sk_025postman",
              "name": "postman"
            },
            {
              "id": "sk_015jwt",
              "name": "jwt"
            },
            {
              "id": "sk_026bcrypt",
              "name": "bcrypt"
            },
            {
              "id": "sk_027cors",
              "name": "cors"
            }
          ],
          "matched_skills": [
            {
              "id": "sk_008git",
              "name": "Git",
              "original_input": "Git",
              "match_type": "exact"
            },
            {
              "id": "sk_015jwt",
              "name": "JWT",
              "original_input": "JWT",
              "match_type": "exact"
            }
          ],
          "match_score": 0.4,
          "hierarchy_trees": [
            {
              "skill_id": "sk_root_dev",
              "skill_name": "Software Development",
              "children": [
                {
                  "skill_id": "sk_vcs",
                  "skill_name": "Version Control Systems",
                  "children": [
                    {
                      "skill_id": "sk_008git",
                      "skill_name": "Git",
                      "is_matched": True,
                      "children": []
                    }
                  ],
                  "is_matched": False
                }
              ],
              "is_matched": False
            }
          ],
          "standalone_skills": [
            {
              "id": "sk_015jwt",
              "name": "JWT"
            }
          ]
        }
      ],
      "missing_skills": [
        {
          "id": "sk_022mongoose",
          "name": "Mongoose",
          "category": "Database"
        },
        {
          "id": "sk_023context",
          "name": "Context API",
          "category": "State Management"
        },
        {
          "id": "sk_025postman",
          "name": "Postman",
          "category": "Tools & Others (Optional)"
        }
      ],
      "category_requirements": {
        "Frontend": {
          "mandatory": True,
          "min_match": 3,
          "met": True,
          "matched_count": 4
        },
        "Backend": {
          "mandatory": True,
          "min_match": 2,
          "met": True,
          "matched_count": 2
        },
        "Database": {
          "mandatory": True,
          "min_match": 1,
          "met": True,
          "matched_count": 1
        },
        "State Management": {
          "mandatory": True,
          "min_match": 1,
          "met": True,
          "matched_count": 1
        },
        "Tools & Others (Optional)": {
          "mandatory": False,
          "min_match": 1,
          "met": True,
          "matched_count": 2
        }
      },
      "overall_reasoning": "Good match (Score: 0.783). Strengths: strong frontend and backend skills, relevant educational background. Areas for improvement: state management ecosystem knowledge, database ORMs."
    },
    "data_science_intern": {
      "job_title": "Data Science Intern",
      "overall_score": 0.52,
      "stack_pass": False,
      "stack_coverage_score": 0.4,
      "education_match_score": 1.0,
      "scoring_breakdown": {
        "technical_weight": 0.6,
        "education_weight": 0.4,
        "calculation": "(0.4 * 0.6) + (1.0 * 0.4) = 0.52"
      },
      "matched_skills_by_category": [
        {
          "category_name": "Programming Language",
          "required_skills": [
            {
              "id": "sk_001python",
              "name": "python"
            }
          ],
          "matched_skills": [
            {
              "id": "sk_001python",
              "name": "Python",
              "original_input": "Python",
              "match_type": "exact"
            }
          ],
          "match_score": 1.0,
          "hierarchy_trees": [
            {
              "skill_id": "sk_root_prog_lang",
              "skill_name": "Programming Languages",
              "children": [
                {
                  "skill_id": "sk_001python",
                  "skill_name": "Python",
                  "is_matched": True,
                  "children": []
                }
              ],
              "is_matched": False
            }
          ],
          "standalone_skills": []
        },
        {
          "category_name": "Database & API",
          "required_skills": [
            {
              "id": "sk_sql_general",
              "name": "sql"
            },
            {
              "id": "sk_nosql_general",
              "name": "nosql"
            }
          ],
          "matched_skills": [
            {
              "id": "sk_016mysql",
              "name": "MySQL",
              "original_input": "MySQL",
              "match_type": "satisfies_sql"
            },
            {
              "id": "sk_005postgres",
              "name": "PostgreSQL",
              "original_input": "PostgreSQL",
              "match_type": "satisfies_sql"
            },
            {
              "id": "sk_004mongo",
              "name": "MongoDB",
              "original_input": "MongoDB",
              "match_type": "satisfies_nosql"
            }
          ],
          "match_score": 1.0,
          "hierarchy_trees": [
            {
              "skill_id": "sk_root_db",
              "skill_name": "Database Systems",
              "children": [
                {
                  "skill_id": "sk_sql_db",
                  "skill_name": "SQL Databases",
                  "children": [
                    {
                      "skill_id": "sk_016mysql",
                      "skill_name": "MySQL",
                      "is_matched": True,
                      "children": []
                    },
                    {
                      "skill_id": "sk_005postgres",
                      "skill_name": "PostgreSQL",
                      "is_matched": True,
                      "children": []
                    }
                  ],
                  "is_matched": False
                },
                {
                  "skill_id": "sk_nosql_db",
                  "skill_name": "NoSQL Databases",
                  "children": [
                    {
                      "skill_id": "sk_004mongo",
                      "skill_name": "MongoDB",
                      "is_matched": True,
                      "children": []
                    }
                  ],
                  "is_matched": False
                }
              ],
              "is_matched": False
            }
          ],
          "standalone_skills": []
        }
      ],
      "missing_skills": [
        {
          "id": "sk_028tensorflow",
          "name": "TensorFlow",
          "category": "Libraries & Frameworks"
        },
        {
          "id": "sk_029pytorch",
          "name": "PyTorch",
          "category": "Libraries & Frameworks"
        },
        {
          "id": "sk_030pandas",
          "name": "Pandas",
          "category": "Data Libraries"
        },
        {
          "id": "sk_031numpy",
          "name": "NumPy",
          "category": "Data Libraries"
        },
        {
          "id": "sk_032jupyter",
          "name": "Jupyter Notebook",
          "category": "Software & Platform"
        }
      ],
      "category_requirements": {
        "Programming Language": {
          "mandatory": True,
          "min_match": 1,
          "met": True,
          "matched_count": 1
        },
        "Libraries & Frameworks": {
          "mandatory": True,
          "min_match": 1,
          "met": False,
          "matched_count": 0
        },
        "Data Libraries": {
          "mandatory": False,
          "min_match": 1,
          "met": False,
          "matched_count": 0
        },
        "Software & Platform": {
          "mandatory": False,
          "min_match": 1,
          "met": False,
          "matched_count": 0
        },
        "Database & API": {
          "mandatory": True,
          "min_match": 1,
          "met": True,
          "matched_count": 2
        }
      },
      "overall_reasoning": "Limited match (Score: 0.52). Failed mandatory technical requirements in: Libraries & Frameworks. Student has strong programming and database foundation but lacks data science specific libraries and frameworks."
    }
  },
}

import json
with open("sample_response_match.json", "w") as f:
    json.dump(sample_response, f, indent=4)