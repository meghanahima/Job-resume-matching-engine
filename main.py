from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pickle
import numpy as np
import re
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
import redis
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="ATS Score Prediction API")

import requests
import pickle

try:
    # Replace this with your actual Google Drive direct download URL
    url = "https://drive.google.com/file/d/1T1Hfcuacl4zq1fdEp-aF-8czDCT_F2f6/view?usp=sharing"
    
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad status codes

    # Deserialize the downloaded binary content
    model_data = pickle.loads(response.content)
    xgb_model = model_data['xgb_model']
    sentence_model = model_data['sentence_model']

except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")


# MongoDB client
mongo_client = MongoClient(os.getenv("DB_CONN_STRING"))
db = mongo_client[os.getenv("DB_NAME")]
jobs_collection = db[os.getenv("DB_COLLECTION_NAME")]

# Redis client
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    password=os.getenv("REDIS_PASSWORD"),
    ssl=True
)

# ------------------------ Input Models ------------------------ #
class Skill(BaseModel):
    skill: str
    level: Optional[str] = None

class EmploymentHistory(BaseModel):
    jobTitle: str
    company: str
    city: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    description: str
    presentWorking: bool

class Education(BaseModel):
    degree: str
    institution: str
    year: str

class CandidateInput(BaseModel):
    resume_id: str
    resume_text: str
    candidate_skills: List[Skill]
    candidate_employment_history: List[EmploymentHistory]
    candidate_education: List[Education]

# ------------------------ Helper Functions ------------------------ #
def normalize_skill(skill: str) -> str:
    return skill.lower().strip()

def calculate_skills_match(candidate_skills, job_skills):
    candidate_list = [normalize_skill(skill.skill) for skill in candidate_skills]
    job_list = [skill.strip().lower() for skill in job_skills.split(',')] if isinstance(job_skills, str) else []
    matched = len(set(candidate_list).intersection(set(job_list)))
    return (matched / len(job_list)) * 100 if job_list else 0.0

def calculate_experience_match(employment_history, job_role):
    if not employment_history:
        return 0.0
    experience_years = 0
    for job in employment_history:
        if job_role.lower() in job.jobTitle.lower() or job_role.lower() in job.description.lower():
            try:
                start = datetime.strptime(job.startDate, '%m/%Y') if job.startDate else None
                end = datetime.now() if job.presentWorking or not job.endDate else datetime.strptime(job.endDate, '%m/%Y') if job.endDate else None
                
                if start:
                    end = end or datetime.now()
                    experience_years += (end - start).days / 365.25
            except Exception:
                continue
    if experience_years >= 6:
        return 1.0
    elif experience_years >= 3:
        return 0.75
    elif experience_years > 0:
        return 0.5
    return 0.0

def calculate_education_match(candidate_education, job_description):
    match = re.search(r'Education: (.*?)(?:\n|$)', job_description)
    required = match.group(1).strip().lower() if match else 'none'
    if required == 'none':
        return 1
    return 1 if any(required in edu.degree.lower() for edu in candidate_education) else 0

def calculate_text_similarity(resume_text, job_description):
    summary = re.search(r'Summary\n(.*?)(?:\nEducation|\nSkills|\nExperience|$)', resume_text, re.DOTALL)
    summary = summary.group(1).strip() if summary else resume_text
    embeddings = sentence_model.encode([summary, job_description])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0] * 100

def convert_float32_to_float(data: any) -> any:
    """Recursively convert any np.float32 objects in data to native Python float."""
    if isinstance(data, dict):
        return {key: convert_float32_to_float(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_float32_to_float(item) for item in data]
    elif isinstance(data, np.float32):
        return float(data)
    return data

# ------------------------ Main Endpoint ------------------------ #
@app.post("/predict_top_jobs")
async def predict_top_jobs(candidate: CandidateInput):
    try:
        cache_key = f"resume:{candidate.resume_id}"
        cached = redis_client.get(cache_key)
        if cached:
            return {"jobs cnt": len(json.loads(cached)), "jobs": json.loads(cached)}

        resume_skills = [normalize_skill(skill.skill) for skill in candidate.candidate_skills if normalize_skill(skill.skill)]

        pipeline = [
            {
                "$addFields": {
                    "processedSkills": {
                        "$cond": {
                            "if": {"$eq": [{"$type": "$skills"}, "string"]},
                            "then": {
                                "$map": {
                                    "input": {"$split": [{"$toLower": "$skills"}, ","]},
                                    "as": "skill",
                                    "in": {"$trim": {"input": "$$skill"}}
                                }
                            },
                            "else": {
                                "$map": {
                                    "input": {"$ifNull": ["$skills", []]},
                                    "as": "skill",
                                    "in": {"$trim": {"input": {"$toLower": "$$skill"}}}
                                }
                            }
                        }
                    }
                }
            },
            {
                "$addFields": {
                    "matchingSkills": {
                        "$filter": {
                            "input": "$processedSkills",
                            "as": "skill",
                            "cond": {"$in": ["$$skill", resume_skills]}
                        }
                    }
                }
            },
            {
                "$addFields": {
                    "skillMatchScore": {
                        "$multiply": [
                            {
                                "$divide": [
                                    {"$size": "$matchingSkills"},
                                    {
                                        "$cond": {
                                            "if": {"$gt": [{"$size": "$processedSkills"}, 0]},
                                            "then": {"$size": "$processedSkills"},
                                            "else": 1
                                        }
                                    }
                                ]
                            },
                            100
                        ]
                    }
                }
            },
            {"$match": {"skillMatchScore": {"$gt": 0}}},
            {"$sort": {"skillMatchScore": -1}},
            {"$limit": 1000},
            {
                "$project": {
                    "_id": 1,
                    "uniq_id": 1,
                    "jobtitle": 1,
                    "jobdescription": 1,
                    "skills": 1,
                    "joblocation_address": 1,
                    "company": 1,
                    "advertiserurl": 1,
                    "employmenttype_jobstatus": 1,
                    "skillMatchScore": 1
                }
            }
        ]

        jobs = list(jobs_collection.aggregate(pipeline))
        if not jobs:
            raise HTTPException(status_code=404, detail="No matching jobs found.")

        job_scores = []
        for job in jobs:
            job_id = str(job.get('_id', ''))
            job_title = job.get('jobtitle', 'Unknown')
            job_description = job.get('jobdescription', '')
            skills = job.get('skills', '')
            job_url = job.get('advertiserurl', 'url not available')
            company = job.get('company', '')
            location = job.get('joblocation_address', 'Unknown')

            skill_match = calculate_skills_match(candidate.candidate_skills, skills)
            experience_match = calculate_experience_match(candidate.candidate_employment_history, job_title)
            education_match = calculate_education_match(candidate.candidate_education, job_description)
            similarity = calculate_text_similarity(candidate.resume_text, job_description)

            features = np.array([[skill_match, experience_match, education_match, similarity]])
            ats_score = xgb_model.predict(features)[0]
            ats_score = max(0, min(100, ats_score))
            if education_match == 0:
                ats_score = 0.0

            job_scores.append({
                "job_id": job_id,
                "jobTitle": job_title,
                "company": company,
                "jobDescription": job_description,
                "skills": skills,
                "jobLocation": location,
                "jobURL": job_url,
                "skillMatchScore": round(skill_match, 2),
                "ats_score": round(ats_score, 2)
            })

        top_jobs = sorted(job_scores, key=lambda x: x['ats_score'], reverse=True)[:50]
        top_jobs = convert_float32_to_float(top_jobs)  # ✅ convert before serialization
        redis_client.setex(cache_key, 86400, json.dumps(top_jobs))  # ✅ now safe to cache
        return {"jobs cnt": len(top_jobs), "jobs": top_jobs}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/")
async def root():
    return {"message": "ATS Score Prediction API"}
