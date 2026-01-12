## Purpose

Quick, actionable instructions for AI coding agents working on the ATS Resume Optimization Agent.
Focus on what you must preserve and where to make changes so edits are fast, safe, and compatible with the app.

## High-level architecture

- Entry point: `app.py` — Streamlit UI that wires the agent to the browser. Small, easy to modify but keep input/output contracts.
- LLM layer: `core/agent.py` — single place that calls the OpenAI client. Functions used by UI: `generate`, `evaluate`, `rewrite`.
- Prompt definitions: `core/prompts.py` — authoritative prompts and rules. Edits here change model behavior; keep formats strict.
- Parsing & rendering: `core/structure.py` (parsing, splitting roles, merging skills) and `core/render.py` (DOCX generation using python-docx).
- Job-recommendation: `core/job_seeker_agent.py` — uses a JSON schema + post-processing (company whitelist + normalization).

## Developer workflows (how to run & debug)

- Windows quick run: open Powershell in repo root and execute `run_resume_agent.bat`. It:
  - creates/activates `venv` if missing
  - installs `-r requirements.txt`
  - runs `streamlit run app.py`
- Manual run: create/activate venv, `pip install -r requirements.txt`, then `python -m streamlit run app.py`.
- Required env: `OPENAI_API_KEY` must be set (the OpenAI client in `core/agent.py` reads it). Consider using `python-dotenv` if you add a `.env`.
- PDF conversion is optional: `docx2pdf` is tried in `app.py`; missing it is OK (PDF download will be disabled).

## Project-specific conventions & patterns

- Strict resume format enforced by prompts — do not change section headers or SKILLS formatting:
  - Example SKILLS line (must): `Languages: Python, JavaScript, C#`
  - Example EXPERIENCE header line: `Acme Corp | Senior Engineer | 2019–2023 | Full-time`
  - Bullets in EXPERIENCE must start with `-` (used by `split_experience`).
- `evaluate()` expects the model to return a JSON blob matching EVAL_PROMPT and is parsed with `json.loads`. Keep the fields `ats_score`, `verbal_feedback`, `strengths`, `weaknesses` unchanged.
- Job matching agent (`core/job_seeker_agent.py`) returns objects with `title`, `company`, `url`, `match_score`, `reason`. The module enforces:
  - Company whitelist (`ALLOWED_COMPANIES`) — unknown companies are filtered out.
  - Company aliases normalized via `COMPANY_ALIASES`.
  - `match_score` post-processed to be an integer 0–100.

## Where to safely change behavior

- UI tweaks (layout, button text, sliders) — edit `app.py`.
- Prompt tuning — edit `core/prompts.py` but keep header/format rules (if you relax formats, update `structure.py` accordingly).
- LLM client config (model name, temperature) — edit `core/agent.py`.
- Post-processing & validation (e.g., adjust company whitelist or scoring rules) — edit `core/job_seeker_agent.py` or `core/recommend.py`.

## Quick guidance for PRs from AI agents

- Preserve backwards-compatible outputs. If you change a prompt or output shape, update parsing code that consumes it (likely `core/structure.py` or callers in `app.py`).
- Unit tests are minimal in this repo; include a short test or a small example input+expected output when changing parsing or model output formats.

## Files to inspect for examples

- `core/prompts.py` — canonical prompts and rules (most important source of truth).
- `core/structure.py` — how resume text is parsed and required formats.
- `core/render.py` — how resume sections are turned into a DOCX.
- `core/job_seeker_agent.py` — role-matching + schema + whitelist examples.
- `app.py` — shows how the pieces are wired together and how outputs are used.

## If something is ambiguous

- Ask for clarification about changing prompt formats or the SKILLS/EXPERIENCE structure — many downstream utilities assume those exact formats.

---

If you'd like, I can iterate on wording, add a short checklist for reviewers, or include example unit tests that validate `parse_resume` / `split_experience` behaviors.
