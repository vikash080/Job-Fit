from pydantic import BaseModel

class AnalyzeRequest(BaseModel):
    job_description: str

class AnalyzeResponse(BaseModel):
    fit_score: int                  # 0–100
    matching_skills: list[str]
    missing_skills: list[str]
    resume_rewrite_tips: list[str]
    verdict: str