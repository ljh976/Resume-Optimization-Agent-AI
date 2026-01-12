import json
from typing import List, Dict

from .prompts import SYSTEM_BASE
from .agent import llm


def _build_bullets_payload(roles: List[Dict]) -> List[Dict]:
    payload = []
    for ri, role in enumerate(roles):
        company = role.get("company", "")
        meta = role.get("meta", "")
        for bi, b in enumerate(role.get("bullets", [])):
            payload.append({
                "id": f"r{ri}_b{bi}",
                "role": f"{company} | {meta}",
                "bullet": b
            })
    return payload


def score_bullets_llm(jd: str, roles: List[Dict]) -> List[Dict]:
    """Ask the LLM to score each bullet's relevance to the JD.

    Returns a list of dicts: {id, bullet, score, reason}
    """
    items = _build_bullets_payload(roles)

    if not items:
        return []

    # Build a compact JSON array of bullets to send in the prompt
    bullets_json = json.dumps(items, ensure_ascii=False)

    # Simple in-memory cache to avoid re-scoring identical inputs during a run
    try:
        import hashlib
        cache_key = hashlib.sha256((jd + "::" + bullets_json).encode("utf-8")).hexdigest()
    except Exception:
        cache_key = None

    if cache_key is not None:
        global _SCORE_CACHE
        try:
            _SCORE_CACHE
        except NameError:
            _SCORE_CACHE = {}
        if cache_key in _SCORE_CACHE:
            return _SCORE_CACHE[cache_key]

    prompt = (
        "You are a senior technical recruiter.\n"
        "Given a Job Description (JD) and a list of resume bullets, score how relevant "
        "each bullet is to the JD on a scale from 0 (not relevant) to 100 (highly relevant).\n"
        "Return ONLY a JSON array of objects with the keys: id, bullet, score, reason.\n"
        "Maintain numeric scores as integers. Keep reasons concise (1 sentence).\n\n"
        "JD:\n" + jd + "\n\n"
        "BULLETS (JSON array):\n" + bullets_json + "\n"
    )

    try:
        raw = llm([
            {"role": "system", "content": SYSTEM_BASE},
            {"role": "user", "content": prompt}
        ])

        parsed = json.loads(raw)

        # Validate and normalize
        result = []
        for p in parsed:
            try:
                score = int(round(float(p.get("score", 0))))
            except Exception:
                score = 0
            result.append({
                "id": p.get("id"),
                "bullet": p.get("bullet"),
                "score": max(0, min(100, score)),
                "reason": p.get("reason", "")
            })

        return result
    except Exception:
        # Fallback: assign neutral scores (50) so downstream logic can proceed
        fallback = []
        for it in items:
            fallback.append({
                "id": it["id"],
                "bullet": it["bullet"],
                "score": 50,
                "reason": "fallback: could not parse LLM response"
            })
        return fallback
    finally:
        # store in cache if available
        try:
            if cache_key is not None and 'result' in locals():
                _SCORE_CACHE[cache_key] = result
        except Exception:
            pass
