# Resume Optimization Agent

A focused, practical agent that analyzes and rewrites resumes to optimize their match to specific job descriptions (JDs) and improve job relevance. Built as a compact Streamlit app and a small LLM-driven core, it demonstrates prompt engineering, parsing, and deterministic post-processing suitable for production-grade resume-to-JD optimization tooling.

---
## Workflow

1. Input Job Description and Resume

Enter the job description and your resume to generate a tailored analysis.

<img width="1336" height="1452" alt="image" src="https://github.com/user-attachments/assets/022c3df9-e31c-4be1-9fd2-126d67e1ca55" />


2. Review Results and Export

Review the generated optimization results and export the updated resume.

<img width="1591" height="1876" alt="image" src="https://github.com/user-attachments/assets/c638ea8d-f38b-4a69-9dde-7ed8f048558b" />

3. Review Generated Feedback

View detailed feedback and recommendations based on the job description.

<img width="607" height="1848" alt="image" src="https://github.com/user-attachments/assets/da9eae0d-1c8b-4d50-b7dc-9b57acbcbbb4" />


4. Generated Opimized Reumse

Review the Optimized Resume

<img width="1272" height="1588" alt="image" src="https://github.com/user-attachments/assets/d04de9bc-0e5e-4f7f-a5d5-682fe08e3f18" />


---

## Executive summary
- Purpose: Automatically rewrite and optimize resumes to maximize alignment with specific job descriptions (JDs) and recommend best-fit roles.  
- Audience: Engineers and hiring teams; suitable for technical evaluation.  
- Tech stack: Python 3.11+, Streamlit UI, pytest, LangChain (prescreen + header extraction), and a prompt-driven LLM layer.  
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
- Job descriptions can be pasted or uploaded as PDF, DOCX, or TXT files.
- Resume parsing rules that enforce strict SKILLS / EXPERIENCE formatting.  
- Evaluate / score function returns a structured JSON (for testing and integration).  
- LangChain-based pre-screening to stop low-signal runs early.  
- Robust header extraction (name/email/GitHub/LinkedIn) to stabilize parsing with messy inputs.  
- Cache safety so results are tied to the current JD + Master Resume.  

---

## Why LangChain here ✅
We use LangChain for two targeted steps where structure and reliability matter most:

1) **Pre-screening gate (`core/prescreen.py`)**
  - **What:** A lightweight LLM step that estimates skill match and decides if optimization is viable.
  - **Why needed:** Prevents costly rewrite loops when JD ↔ resume alignment is too weak.
  - **Benefit:** Cuts wasted tokens and gives clear feedback early.

2) **Header extraction (`core/header_extract.py`)**
  - **What:** Extracts name/email/phone/LinkedIn/GitHub/location into a normalized `HEADER` block.
  - **Why needed:** Real resumes vary wildly; stable header parsing reduces downstream format breakage.
  - **Benefit:** More consistent parsing and fewer UI/rendering errors with messy inputs.

---

## Architecture & Key Files 🔧
- `app.py` — Streamlit UI + preview / export.  
- `core/agent.py` — LLM client wrapper (generate/evaluate/rewrite helpers).  
- `core/prompts.py` — Canonical prompts used by agents.  
- `core/structure.py` — Parsing/splitting helper functions and business rules.  
- `core/prescreen.py` — LangChain prescreening gate before optimization loops.  
- `core/header_extract.py` — LangChain header extraction for noisy inputs.  
- `core/render.py` — DOCX generation for polished downloads.  
- `core/job_seeker_agent.py` — Role matching and company whitelist logic (legacy).  

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
- [MIT License](LICENSE)
- Maintainer: Junho. For questions or review, open an issue or contact the repo owner.

---

Thank you — this repository showcases focused product thinking and engineering rigor: prompt engineering, deterministic parsing, and end-to-end validation for real-world JD matching and resume optimization problems.
