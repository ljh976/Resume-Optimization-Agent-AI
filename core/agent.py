
import os, json
from openai import OpenAI
from .prompts import SYSTEM_BASE, DRAFT_PROMPT, EVAL_PROMPT, REWRITE_PROMPT
from .trimming_prompt import TRIM_PROMPT

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def llm(messages):
    # Allow model override via environment var `OPENAI_MODEL`.
    # Default to a cost-effective 4-series model; can be overridden with OPENAI_MODEL env var
    model_name = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    r = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.2
    )
    return r.choices[0].message.content

def generate(jd, master):
    return llm([
        {"role":"system","content":SYSTEM_BASE},
        {"role":"user","content":DRAFT_PROMPT + "\nJD:\n"+jd+"\nMASTER:\n"+master}
    ])

def _heuristic_ats_score(jd: str, resume: str, fb: dict = None):
    """Compute a heuristic ATS score based on keyword overlap, skills, and bullet scores.
    Returns (score_int, breakdown_dict).
    """
    try:
        from .structure import parse_resume
    except Exception:
        parse_resume = None

    jd_lc = (jd or "").lower()
    resume_lc = (resume or "").lower()

    # simple JD keyword set (words length>3)
    import re
    jd_tokens = set([t for t in re.findall(r"\w{4,}", jd_lc)])
    resume_tokens = set([t for t in re.findall(r"\w{4,}", resume_lc)])
    if jd_tokens:
        kw_match = len(jd_tokens & resume_tokens) / float(len(jd_tokens))
    else:
        kw_match = 0.0

    # skills match: parse SKILLS section if possible
    skill_match = 0.0
    total_skills = 0
    matched_skills = 0
    try:
        if parse_resume:
            secs = parse_resume(resume)
            skills_lines = secs.get("SKILLS", [])
            skills = []
            for l in skills_lines:
                if ":" in l:
                    _, rest = l.split(":", 1)
                    skills.extend([s.strip().lower() for s in rest.split(",") if s.strip()])
            skills = [s for s in skills if s]
            total_skills = len(skills)
            for s in skills:
                if s and s in jd_lc:
                    matched_skills += 1
            if total_skills:
                skill_match = matched_skills / float(total_skills)
    except Exception:
        skill_match = 0.0

    # bullet score average (if provided in fb)
    bullet_avg = None
    try:
        if fb and fb.get("bullet_scores"):
            vals = [int(b.get("score", 0)) for b in fb.get("bullet_scores") if b.get("score") is not None]
            if vals:
                bullet_avg = sum(vals) / float(len(vals)) / 100.0
    except Exception:
        bullet_avg = None

    # QUALITY METRICS: avg words per bullet, presence of numeric metrics, presence of action verbs
    try:
        # extract bullets via parse_resume if available, else simple regex
        bullets = []
        if parse_resume:
            secs = parse_resume(resume)
            exp = secs.get("EXPERIENCE", []) or secs.get("EXPERIENCES", []) or []
            # flatten lines starting with '-'
            for line in exp:
                if isinstance(line, str) and line.strip().startswith("-"):
                    bullets.append(line.strip().lstrip('-').strip())
        if not bullets:
            import re
            bullets = [m.group(0).lstrip('-').strip() for m in re.finditer(r"^-\s.*$", resume, re.M)]
    except Exception:
        bullets = []

    avg_words = 0.0
    metric_frac = 0.0
    verb_frac = 0.0
    try:
        if bullets:
            words_counts = [len(b.split()) for b in bullets]
            avg_words = sum(words_counts) / float(len(words_counts))
            import re
            metric_frac = sum(1 for b in bullets if re.search(r"\d|%|AUC|x\.|x\s|k\b", b, re.I)) / float(len(bullets))
            verbs = ["led","improved","reduced","increased","built","implemented","optimized","designed","shipped","automated","mentored","owned","managed"]
            verb_frac = sum(1 for b in bullets if any(v in b.lower() for v in verbs)) / float(len(bullets))
    except Exception:
        avg_words = 0.0
        metric_frac = 0.0
        verb_frac = 0.0

    # normalize avg_words around 15 words ideal
    avg_words_norm = min(avg_words / 15.0, 1.0)
    # quality score combines metric presence, avg length, and action verbs
    quality_score = 0.5 * metric_frac + 0.3 * avg_words_norm + 0.2 * verb_frac

    # normalize components to 0-1
    kw_score = max(0.0, min(1.0, kw_match))
    sk_score = max(0.0, min(1.0, skill_match))
    bs_score = bullet_avg if bullet_avg is not None else 0.5

    qs_score = max(0.0, min(1.0, quality_score))

    # weighted heuristic: skills 35%, bullets 25%, keyword 15%, quality 25%
    heuristic = 0.35 * sk_score + 0.25 * bs_score + 0.15 * kw_score + 0.25 * qs_score
    breakdown = {
        "keyword_coverage": round(kw_score * 100, 1),
        "skill_coverage": round(sk_score * 100, 1),
        "bullet_relevance_avg": round(bs_score * 100, 1),
        "quality_score": round(qs_score * 100, 1),
        "heuristic_score": round(heuristic * 100, 1)
    }
    return int(round(heuristic * 100)), breakdown


def evaluate(jd, resume, fb: dict = None):
    """Evaluate resume vs JD. Returns a dict with ats_score, verbal_feedback,
    strengths, weaknesses and an ats_breakdown. Combines LLM evaluation (when
    available) with a deterministic heuristic for more stable ATS scores.
    """
    raw = llm([
        {"role": "system", "content": SYSTEM_BASE},
        {"role": "user", "content": EVAL_PROMPT + "\nJD:\n" + jd + "\nRESUME:\n" + resume}
    ])

    llm_parsed = None
    llm_score = None
    verbal = ""
    strengths = []
    weaknesses = []
    try:
        llm_parsed = json.loads(raw)
    except Exception:
        # attempt to extract a JSON blob
        def _extract_json_blob(s: str):
            for start_ch in ('{', '['):
                start_idx = s.find(start_ch)
                if start_idx == -1:
                    continue
                stack = []
                pairs = {'{': '}', '[': ']'}
                open_ch = start_ch
                close_ch = pairs[open_ch]
                for k in range(start_idx, len(s)):
                    if s[k] == open_ch:
                        stack.append(open_ch)
                    elif s[k] == close_ch:
                        stack.pop()
                        if not stack:
                            candidate = s[start_idx:k+1]
                            try:
                                return json.loads(candidate)
                            except Exception:
                                break
            return None

        try:
            llm_parsed = _extract_json_blob(raw)
        except Exception:
            llm_parsed = None

    if isinstance(llm_parsed, dict):
        verbal = llm_parsed.get("verbal_feedback", "")
        strengths = llm_parsed.get("strengths", []) or []
        weaknesses = llm_parsed.get("weaknesses", []) or []
        try:
            llm_score = int(round(float(llm_parsed.get("ats_score", None)))) if llm_parsed.get("ats_score") is not None else None
        except Exception:
            llm_score = None
    else:
        # If LLM didn't produce a usable JSON, set verbal to raw text so UI can show it
        verbal = raw

    # Heuristic score and breakdown
    heuristic_score, breakdown = _heuristic_ats_score(jd, resume, fb)

    # Combine scores: prefer heuristic more for stability (60% heuristic, 40% LLM)
    if llm_score is not None:
        combined = int(round(0.6 * heuristic_score + 0.4 * llm_score))
    else:
        combined = heuristic_score

    result = {
        "ats_score": max(0, min(100, int(combined))),
        "verbal_feedback": verbal,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "ats_breakdown": breakdown
    }

    return result

def rewrite(jd, master, resume, fb):
    return llm([
        {"role":"system","content":SYSTEM_BASE},
        {"role":"user","content":REWRITE_PROMPT +
         "\nJD:\n"+jd+
         "\nMASTER:\n"+master+
         "\nCURRENT:\n"+resume+
         "\nFEEDBACK:\n"+json.dumps(fb)}
    ])

def trimmer(jd, master, resume, fb):
    """Perform a targeted trimming rewrite using the TRIM_PROMPT.
    Returns the raw model output (JSON blob + trimmed resume text).
    """
    return llm([
        {"role":"system", "content": SYSTEM_BASE},
        {"role":"user", "content": TRIM_PROMPT + "\nJD:\n" + jd + "\nMASTER:\n" + master + "\nCURRENT:\n" + resume + "\nFEEDBACK:\n" + json.dumps(fb)}
    ])
