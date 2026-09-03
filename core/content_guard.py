"""Guardrails that keep optimized resumes useful and visually page-filling."""

import re

from .scoring import meaningful_tokens
from .structure import merge_skills_a1, parse_resume, split_experience


def _wrap_lines(text: str, threshold: int = 92) -> int:
    value = (text or "").strip()
    if not value:
        return 0
    return max(1, (len(value) + threshold - 1) // threshold)


def estimate_resume_lines(resume_text: str) -> int:
    """Estimate occupied DOCX lines, including section and paragraph spacing."""
    try:
        sections = parse_resume(resume_text or "")
        roles = split_experience(
            sections.get("EXPERIENCE", [])
            or sections.get("PROFESSIONAL EXPERIENCE", [])
            or []
        )
    except Exception:
        return len([line for line in (resume_text or "").splitlines() if line.strip()])

    lines = 0
    header = sections.get("HEADER", []) or []
    lines += min(len(header), 2)

    summary = sections.get("SUMMARY", []) or sections.get("PROFESSIONAL SUMMARY", []) or []
    if summary:
        lines += 1 + sum(_wrap_lines(line) for line in summary)

    if roles:
        lines += 1
        for role in roles:
            lines += 1
            lines += sum(_wrap_lines(bullet) for bullet in role.get("bullets") or [])
    else:
        experience = sections.get("EXPERIENCE", []) or sections.get("PROFESSIONAL EXPERIENCE", []) or []
        if experience:
            lines += 1 + sum(_wrap_lines(line.lstrip("- ")) for line in experience)

    skills = merge_skills_a1(sections.get("SKILLS", []) or [])
    if skills:
        lines += 1 + sum(_wrap_lines(line) for line in skills)

    education = sections.get("EDUCATION", []) or []
    if education:
        lines += 1 + sum(_wrap_lines(line) for line in education)

    # Approximate section spacing and the slightly larger name line.
    return lines + 3


def count_experience_bullets(resume_text: str) -> int:
    try:
        sections = parse_resume(resume_text or "")
        roles = split_experience(
            sections.get("EXPERIENCE", [])
            or sections.get("PROFESSIONAL EXPERIENCE", [])
            or []
        )
        return sum(len(role.get("bullets") or []) for role in roles)
    except Exception:
        return 0


def _normalized_company(value: str) -> str:
    words = re.findall(r"[a-z0-9]+", (value or "").lower())
    ignored = {"inc", "llc", "ltd", "corp", "corporation", "company", "co"}
    return " ".join(word for word in words if word not in ignored)


def _same_company(left: str, right: str) -> bool:
    left_norm = _normalized_company(left)
    right_norm = _normalized_company(right)
    if not left_norm or not right_norm:
        return False
    return left_norm == right_norm or left_norm in right_norm or right_norm in left_norm


def _is_probable_duplicate(candidate: str, existing: list[str]) -> bool:
    candidate_tokens = meaningful_tokens(candidate)
    candidate_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", candidate or ""))
    for bullet in existing:
        existing_tokens = meaningful_tokens(bullet)
        union = candidate_tokens | existing_tokens
        overlap = len(candidate_tokens & existing_tokens) / float(len(union)) if union else 0.0
        existing_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", bullet or ""))
        # Similar wording can still describe distinct achievements when the
        # underlying metrics differ; preserve both in that case.
        metrics_conflict = candidate_numbers and existing_numbers and candidate_numbers != existing_numbers
        if overlap >= 0.48 and not metrics_conflict:
            return True
        if candidate_numbers and candidate_numbers == existing_numbers and overlap >= 0.22:
            return True
    return False


def _build_resume(sections: dict, roles: list[dict]) -> str:
    parts = ["HEADER"]
    parts.extend(sections.get("HEADER", []) or [])
    parts.append("SUMMARY")
    parts.extend(sections.get("SUMMARY", []) or sections.get("PROFESSIONAL SUMMARY", []) or [])
    parts.append("EXPERIENCE")
    for role in roles:
        header = " | ".join(
            value for value in [role.get("company") or "", role.get("meta") or ""] if value
        )
        if header:
            parts.append(header)
        parts.extend("- " + bullet.strip() for bullet in role.get("bullets") or [] if bullet.strip())
    parts.append("SKILLS")
    parts.extend(sections.get("SKILLS", []) or [])
    parts.append("EDUCATION")
    parts.extend(sections.get("EDUCATION", []) or [])
    return "\n".join(str(part) for part in parts if str(part).strip()).strip()


def restore_omitted_master_bullets(
    draft: str,
    master: str,
    jd: str,
    target_lines: int = 56,
    max_lines: int = 60,
    max_chars: int = 3600,
) -> tuple[str, int]:
    """Restore omitted factual master bullets until the draft fills a page.

    Existing draft text is never removed. Candidates are copied verbatim from the
    master and ranked by JD keyword overlap, metrics, and recency.
    """
    if estimate_resume_lines(draft) >= target_lines:
        return draft, 0

    try:
        draft_sections = parse_resume(draft or "")
        master_sections = parse_resume(master or "")
        draft_roles = split_experience(
            draft_sections.get("EXPERIENCE", [])
            or draft_sections.get("PROFESSIONAL EXPERIENCE", [])
            or []
        )
        master_roles = split_experience(
            master_sections.get("EXPERIENCE", [])
            or master_sections.get("PROFESSIONAL EXPERIENCE", [])
            or []
        )
    except Exception:
        return draft, 0
    if not draft_roles or not master_roles:
        return draft, 0

    jd_tokens = meaningful_tokens(jd)
    existing_bullets = [bullet for role in draft_roles for bullet in role.get("bullets") or []]
    candidates = []
    for role_index, master_role in enumerate(master_roles):
        for bullet_index, bullet in enumerate(master_role.get("bullets") or []):
            if not bullet.strip() or _is_probable_duplicate(bullet, existing_bullets):
                continue
            bullet_tokens = meaningful_tokens(bullet)
            jd_overlap = len(bullet_tokens & jd_tokens)
            metric_bonus = 2 if re.search(r"\d|%", bullet) else 0
            recency_bonus = max(0, 3 - role_index)
            candidates.append(
                (jd_overlap * 4 + metric_bonus + recency_bonus, -role_index, -bullet_index, master_role, bullet)
            )
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)

    restored = 0
    for _, _, _, source_role, bullet in candidates:
        target_role = next(
            (role for role in draft_roles if _same_company(role.get("company"), source_role.get("company"))),
            None,
        )
        added_role = False
        if target_role is None:
            target_role = {
                "company": source_role.get("company") or "",
                "meta": source_role.get("meta") or "",
                "bullets": [],
            }
            draft_roles.append(target_role)
            added_role = True

        target_role.setdefault("bullets", []).append(bullet.strip())
        tentative = _build_resume(draft_sections, draft_roles)
        tentative_lines = estimate_resume_lines(tentative)
        exceeds_bounds = tentative_lines > max_lines or len(tentative) > max_chars
        if exceeds_bounds:
            target_role["bullets"].pop()
            if added_role:
                draft_roles.pop()
            continue

        draft = tentative
        existing_bullets.append(bullet)
        restored += 1
        if tentative_lines >= target_lines:
            break

    return draft, restored
