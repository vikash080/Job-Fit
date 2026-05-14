import os
import json
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Load once at startup — reused across all requests
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def compute_fit_score(resume_text: str, job_description: str) -> int:
    """Cosine similarity between resume and JD embeddings → 0-100 score."""
    vecs = embedder.encode([resume_text, job_description])
    cosine = np.dot(vecs[0], vecs[1]) / (
        np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1])
    )
    return round(float(cosine) * 100)

def analyze_gap(resume_text: str, job_description: str) -> dict:
    """Ask Groq/Llama to extract skills gap and rewrite tips."""
    prompt = f"""
You are a professional resume coach. Compare this resume and job description.

RESUME:
{resume_text[:3000]}

JOB DESCRIPTION:
{job_description[:2000]}

Respond ONLY with a valid JSON object in this exact format:
{{
  "fit_score": score/100,
  "matching_skills": ["skill1", "skill2"],
  "missing_skills": ["skill1", "skill2"],
  "resume_rewrite_tips": ["tip1", "tip2", "tip3"],
  "verdict": "One sentence summary of the overall fit."
}}
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    raw = response.choices[0].message.content
    return json.loads(raw)

def run_analysis(resume_text: str, job_description: str) -> dict:
    score = compute_fit_score(resume_text, job_description)
    gap = analyze_gap(resume_text, job_description)
    return {"fit_score": score, **gap}