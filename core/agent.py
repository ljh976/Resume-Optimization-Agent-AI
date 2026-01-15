
import os, json
from openai import OpenAI
from .prompts import SYSTEM_BASE, DRAFT_PROMPT, EVAL_PROMPT, REWRITE_PROMPT
from .trimming_prompt import TRIM_PROMPT

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def llm(messages):
    """Wrapper around the OpenAI client with a configurable model name."""
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.2
    )
    return response.choices[0].message.content


def generate(jd, master):
    return llm([
        {"role": "system", "content": SYSTEM_BASE},
        {"role": "user", "content": DRAFT_PROMPT + "\nJD:\n" + jd + "\nMASTER:\n" + master}
    ])


def _heuristic_ats_score(jd: str, resume: str, fb: dict = None):
    """Compute a heuristic ATS score based on keyword overlap, skills, bullets, and quality."""
    try:
        from .structure import parse_resume
    except Exception:
        parse_resume = None

    jd_lc = (jd or "").lower()
    resume_lc = (resume or "").lower()

    import re

    jd_tokens = set(re.findall(r"\w{4,}", jd_lc))
    resume_tokens = set(re.findall(r"\w{4,}", resume_lc))
    kw_match = len(jd_tokens & resume_tokens) / float(len(jd_tokens)) if jd_tokens else 0.0

    skill_match = 0.0
    try:
        if parse_resume:
            secs = parse_resume(resume)
            skills_lines = secs.get("SKILLS", [])
            skills = []
            for line in skills_lines:
                if ":" in line:
                    _, rest = line.split(":", 1)
                    skills.extend([s.strip().lower() for s in rest.split(",") if s.strip()])
            skills = [s for s in skills if s]
            total_skills = len(skills)
            if total_skills:
                matched = sum(1 for s in skills if s and s in jd_lc)
                skill_match = matched / float(total_skills)
    except Exception:
        skill_match = 0.0

    bullet_avg = None
    try:
        if fb and fb.get("bullet_scores"):
            vals = [int(b.get("score", 0)) for b in fb.get("bullet_scores") if b.get("score") is not None]
            if vals:
                bullet_avg = sum(vals) / float(len(vals)) / 100.0
    except Exception:
        bullet_avg = None

    try:
        bullets = []
        if parse_resume:
            secs = parse_resume(resume)
            exp_lines = secs.get("EXPERIENCE", []) or secs.get("PROFESSIONAL EXPERIENCE", []) or []
            for line in exp_lines:
                if isinstance(line, str) and line.strip().startswith("-"):
                    bullets.append(line.strip().lstrip("-").strip())
        if not bullets:
            bullets = [m.group(0).lstrip("-").strip() for m in re.finditer(r"^-\s.*$", resume, re.M)]
    except Exception:
        bullets = []

    avg_words = 0.0
    metric_frac = 0.0
    verb_frac = 0.0
    try:
        if bullets:
            word_counts = [len(b.split()) for b in bullets]
            avg_words = sum(word_counts) / float(len(word_counts))
            metric_frac = sum(1 for b in bullets if re.search(r"\d|%|AUC|x\.|x\s|k\b", b, re.I)) / float(len(bullets))
            verbs = [
                "led", "improved", "reduced", "increased", "built", "implemented",
                "optimized", "designed", "shipped", "automated", "mentored", "owned", "managed"
            ]
            verb_frac = sum(1 for b in bullets if any(v in b.lower() for v in verbs)) / float(len(bullets))
    except Exception:
        avg_words = 0.0
        metric_frac = 0.0
        verb_frac = 0.0

    avg_words_norm = min(avg_words / 15.0, 1.0)
    quality_score = 0.5 * metric_frac + 0.3 * avg_words_norm + 0.2 * verb_frac

    kw_score = max(0.0, min(1.0, kw_match))
    sk_score = max(0.0, min(1.0, skill_match))
    bs_score = bullet_avg if bullet_avg is not None else 0.5
    qs_score = max(0.0, min(1.0, quality_score))

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
    """Evaluate resume vs JD using the LLM plus heuristic scoring."""
    raw = llm([
        {"role": "system", "content": SYSTEM_BASE},
        {"role": "user", "content": EVAL_PROMPT + "\nJD:\n" + jd + "\nRESUME:\n" + resume}
    ])

    llm_parsed = None
    llm_score = None
    verbal = ""
    strengths = []
    weaknesses = []
    bullet_scores = []

    def _normalize_bullet_scores(entries):
        normalized = []
        if not entries:
            return normalized
        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            raw_score = entry.get("score")
            if raw_score is None:
                continue
            try:
                score_val = int(round(float(raw_score)))
            except Exception:
                continue
            score_val = max(0, min(100, score_val))
            normalized_entry = {
                "id": entry.get("id") or f"b{idx}",
                "bullet": entry.get("bullet", ""),
                "score": score_val
            }
            if entry.get("reason"):
                normalized_entry["reason"] = entry.get("reason")
            normalized.append(normalized_entry)
        return normalized

    try:
        llm_parsed = json.loads(raw)
    except Exception:
        def _extract_json_blob(s: str):
            for start_ch in ("{", "["):
                start_idx = s.find(start_ch)
                if start_idx == -1:
                    continue
                stack = []
                pairs = {"{": "}", "[": "]"}
                open_ch = start_ch
                close_ch = pairs[open_ch]
                for idx in range(start_idx, len(s)):
                    if s[idx] == open_ch:
                        stack.append(open_ch)
                    elif s[idx] == close_ch:
                        stack.pop()
                        if not stack:
                            candidate = s[start_idx:idx + 1]
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
        bullet_scores = _normalize_bullet_scores(llm_parsed.get("bullet_scores") or [])
        try:
            llm_score = llm_parsed.get("ats_score")
            llm_score = int(round(float(llm_score))) if llm_score is not None else None
        except Exception:
            llm_score = None
    else:
        verbal = raw

    if not bullet_scores and fb and isinstance(fb, dict):
        bullet_scores = _normalize_bullet_scores(fb.get("bullet_scores"))

    heuristic_input = dict(fb) if fb else None
    if bullet_scores:
        heuristic_input = heuristic_input or {}
        heuristic_input["bullet_scores"] = bullet_scores
    heuristic_score, breakdown = _heuristic_ats_score(jd, resume, heuristic_input)

    combined = int(round(0.6 * heuristic_score + 0.4 * llm_score)) if llm_score is not None else heuristic_score

    return {
        "ats_score": max(0, min(100, combined)),
        "verbal_feedback": verbal,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "ats_breakdown": breakdown,
        "bullet_scores": bullet_scores
    }


def rewrite(jd, master, resume, fb):
    return llm([
        {"role": "system", "content": SYSTEM_BASE},
        {"role": "user", "content": REWRITE_PROMPT +
         "\nJD:\n" + jd +
         "\nMASTER:\n" + master +
         "\nCURRENT:\n" + resume +
         "\nFEEDBACK:\n" + json.dumps(fb)}
    ])


def trimmer(jd, master, resume, fb):
    """Run the trimming prompt to get a tighter resume when needed."""
    return llm([
        {"role": "system", "content": SYSTEM_BASE},
        {"role": "user", "content": TRIM_PROMPT +
         "\nJD:\n" + jd +
         "\nMASTER:\n" + master +
         "\nCURRENT:\n" + resume +
         "\nFEEDBACK:\n" + json.dumps(fb)}
    ])
