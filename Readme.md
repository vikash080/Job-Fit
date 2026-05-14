# Job Fit Analyzer

A simple FastAPI application that evaluates how well a candidate’s resume matches a target job description.

## How it works
1. **Resume PDF → text**
   - The API accepts a **PDF file** upload and extracts text using `pypdf`.
2. **Fit score (0–100)**
   - Computes a cosine similarity between embeddings of:
     - the extracted resume text
     - the provided job description
   - Uses `SentenceTransformer("all-MiniLM-L6-v2")` and scales cosine similarity to **0–100**.
3. **Skills gap + coaching tips**
   - Sends the (truncated) resume text and job description to Groq Llama **`llama-3.1-8b-instant`**.
   - The prompt instructs the model to return **only valid JSON** with:
     - `matching_skills`
     - `missing_skills`
     - `resume_rewrite_tips`
     - `verdict`

## Features
- `fit_score`: integer from **0 to 100**
- `matching_skills`: list of skills found in both resume and job description
- `missing_skills`: list of skills missing from the resume
- `resume_rewrite_tips`: actionable rewrite tips
- `verdict`: one-sentence summary

## Tech stack
- **FastAPI** (API server)
- **Uvicorn** (dev server)
- **pypdf** (PDF text extraction)
- **sentence-transformers** + **numpy** (embedding similarity)
- **Groq** (LLM for gaps + coaching)
- **python-dotenv** (load `GROQ_API_KEY` from `.env`)

## API
### Health check
`GET /health`

Response:
```json
{"status": "ok"}
```

### Analyze resume vs job description
`POST /analyze`

**Form fields**:
- `resume`: **PDF file** (`multipart/form-data`)
- `job_description`: string (`application/x-www-form-urlencoded` form field)

Example response shape:
```json
{
  "fit_score": 87,
  "matching_skills": ["..."],
  "missing_skills": ["..."],
  "resume_rewrite_tips": ["..."],
  "verdict": "..."
}
```

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Create a `.env` file in `JobFit/Job-Fit/`:
```env
GROQ_API_KEY=your_groq_api_key_here
```

## Run
From `JobFit/Job-Fit/`:
```bash
uvicorn main:app --reload
```

Server will start on the default FastAPI port (typically `http://127.0.0.1:8000`).

## Usage examples (curl)
### Analyze
```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "resume=@./resume.pdf" \
  -F "job_description=Paste the job description here"
```

### Health check
```bash
curl "http://127.0.0.1:8000/health"
```

## Notes
- The LLM prompt includes **truncated** text:
  - resume: first **3000** characters
  - job description: first **2000** characters
- The response is parsed with `json.loads(...)`, so the model must output valid JSON (as instructed by the prompt).

