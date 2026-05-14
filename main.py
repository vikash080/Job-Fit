from fastapi import FastAPI, UploadFile, File, Form
from schemas import AnalyzeResponse
from parser import extract_text_from_pdf
from analyzer import run_analysis

app = FastAPI(title="Job Fit Analyzer")

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    resume_text = extract_text_from_pdf(await resume.read())
    result = run_analysis(resume_text, job_description)
    return result

@app.get("/health")
async def health():
    return {"status": "ok"}