# Resume Optimization Agent

A focused, practical agent that analyzes and rewrites resumes to optimize their match to specific job descriptions (JDs) and improve job relevance. Built as a compact Streamlit app and a small LLM-driven core, it demonstrates prompt engineering, parsing, and deterministic post-processing suitable for production-grade resume-to-JD optimization tooling.

---

## Executive summary
- Purpose: Automatically rewrite and optimize resumes to maximize alignment with specific job descriptions (JDs) and recommend best-fit roles.  
- Audience: Engineers and hiring teams; suitable for technical evaluation.  
- Tech stack: Python 3.11+, Streamlit UI, pytest, and a prompt-driven LLM layer.  
- Quick start (Windows PowerShell):

```powershell
# Activate virtual environment
& .\venv\Scripts\Activate.ps1
# Install dependencies
python -m pip install -r requirements.txt
# Run the app
python -m streamlit run app.py
```

---

## Why this project 💡
- Real-world problem: Candidates often miss opportunities because resumes and job descriptions are misaligned; this agent applies semantic and structural edits to improve resume-to-JD relevance while preserving fidelity.  
- Engineering focus: Clear separation of concerns (UI, prompts, parsing, rendering) with deterministic outputs and unit tests to lock behavior.  
- Auditability: Prompts and normalization logic are captured in `core/prompts.py` and `core/structure.py` so reviewers can trace behavior.

---

## Features ✨
- Clean Streamlit UI for interactive feedback and downloads (DOCX + optional PDF).  
- Resume parsing rules that enforce strict SKILLS / EXPERIENCE formatting.  
- Evaluate / score function returns a structured JSON (for testing and integration).  
- Job recommendation pipeline with company normalization and score curation.  

---

## Architecture & Key Files 🔧
- `app.py` — Streamlit UI + preview / export.  
- `core/agent.py` — LLM client wrapper (generate/evaluate/rewrite helpers).  
- `core/prompts.py` — Canonical prompts used by agents.  
- `core/structure.py` — Parsing/splitting helper functions and business rules.  
- `core/render.py` — DOCX generation for polished downloads.  
- `core/job_seeker_agent.py` — Role matching and company whitelist logic.  

Design notes: Keep prompts and output shape stable (JSON schema) so consumers can rely on structured output.

---

## Quick Development Guide 🧑‍💻
- Setup (Windows PowerShell):
  - `& .\venv\Scripts\Activate.ps1`  
  - `python -m pip install -r requirements.txt`  
  - Set `OPENAI_API_KEY` (or your LLM provider key) in environment variables.  
- Run the app: `python -m streamlit run app.py`  
- Tests: `python -m pytest -q` (we use `pytest` — keep tests fast and deterministic)  

---

## Testing & Quality ✅
- Unit tests cover parsing, bold/format conversions, and evaluation result shapes.  
- Keep tests small and repeatable; CI should run `pytest` on PRs.  

---

## Examples & Output
- The app produces a Markdown preview (for review), JD-optimized text, and a DOCX export suitable for recruiters and hiring managers.
- SKILLS lines are normalized into consistent categories and formatting to improve keyword matching.

---

## Contribution & Style
- Keep changes minimal and test-first.  
- If you update prompt formats or output JSON shapes, add tests and update `core/structure.py` consumer code.  
- Preserve canonical SKILLS/EXPERIENCE formats (see `core/structure.py` comments).

---

## License & Contact
- [MIT License](LICENSE) (change as required).  
- Maintainer: Junho. For questions or review, open an issue or contact the repo owner.

---

Thank you — this repository showcases focused product thinking and engineering rigor: prompt engineering, deterministic parsing, and end-to-end validation for real-world JD matching and resume optimization problems.