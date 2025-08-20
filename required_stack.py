required_jobs = [
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
            "ORM/ODM": {
                "options": [
                    "mongoose",
                    "prisma",
                    "sequelize",
                    "typeorm"
                ],
                "mandatory": True,
                "min_match": 1
            },
            "Authentication": {
                "options": [
                    "jwt",
                    "bcrypt",
                    "passport",
                    "oauth"
                ],
                "mandatory": True,
                "min_match": 2
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
import json

with open("required_job_stack.json", 'w') as f:
    json.dump(required_jobs, f, indent=2)