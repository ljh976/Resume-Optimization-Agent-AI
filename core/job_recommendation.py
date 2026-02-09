import os
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus

from pydantic import BaseModel, Field, ValidationError

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

try:
    from langchain.output_parsers import OutputFixingParser
except Exception:
    OutputFixingParser = None

from .career_pages import ALL_COMPANIES
from .structure import parse_resume


class ResumeProfile(BaseModel):
    target_roles: List[str] = Field(default_factory=list)
    skills_canonical: List[str] = Field(default_factory=list)
    domains: List[str] = Field(default_factory=list)
    seniority: str = ""
    location_pref: str = ""


class SearchPlan(BaseModel):
    sources: List[str] = Field(default_factory=list)
    role_queries: List[str] = Field(default_factory=list)
    skill_queries: List[str] = Field(default_factory=list)
    domain_queries: List[str] = Field(default_factory=list)
    max_search_calls: int = 3
    target_results: int = 80


class JDProfile(BaseModel):
    must_have_skills: List[str] = Field(default_factory=list)
    nice_to_have_skills: List[str] = Field(default_factory=list)
    seniority: str = ""
    responsibilities: List[str] = Field(default_factory=list)
    domain_keywords: List[str] = Field(default_factory=list)
    work_mode: str = ""
    location: str = ""
    salary_range: str = ""


class RerankItem(BaseModel):
    source_id: str
    fit_score: int
    top_reasons: List[str]
    risks: List[str]
    resume_angle: List[str]


class Telemetry(BaseModel):
    llm_calls: int = 0
    tokens: int = 0
    errors: List[str] = Field(default_factory=list)
    timings_ms: Dict[str, int] = Field(default_factory=dict)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _default_preferences() -> Dict:
    return {
        "preferred_roles": [],
        "work_mode": "Any",
        "location": "",
        "min_salary": None,
        "seniority": "",
        "visa": "",
        "exclude_industries": [],
        "exclude_companies": [],
        "max_results": 3,
    }


def _create_llm() -> ChatOpenAI:
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model_name, temperature=0.2)


def _call_chain(prompt: ChatPromptTemplate, parser: PydanticOutputParser, llm: ChatOpenAI, variables: Dict, telemetry: Telemetry, stage: str) -> Optional[BaseModel]:
    start = _now_ms()
    chain = prompt | llm
    try:
        telemetry.llm_calls += 1
        raw = chain.invoke(variables)
        text = raw.content if hasattr(raw, "content") else str(raw)
        try:
            return parser.parse(text)
        except Exception:
            if OutputFixingParser is None:
                raise
            fixing = OutputFixingParser.from_llm(parser=parser, llm=llm)
            telemetry.llm_calls += 1
            return fixing.parse(text)
    except Exception as exc:
        telemetry.errors.append(f"{stage}: {exc}")
        return None
    finally:
        telemetry.timings_ms[stage] = _now_ms() - start


def _resume_profile_fallback(resume_text: str, preferred_roles: List[str]) -> ResumeProfile:
    skills = []
    domains = []
    seniority = ""
    try:
        sections = parse_resume(resume_text)
        skills_lines = sections.get("SKILLS", []) or []
        for line in skills_lines:
            if ":" in line:
                _, rest = line.split(":", 1)
                skills.extend([s.strip() for s in rest.split(",") if s.strip()])
            else:
                skills.extend([s.strip() for s in line.split(",") if s.strip()])
    except Exception:
        skills = []

    if preferred_roles:
        roles = preferred_roles
    else:
        roles = ["Software Engineer"]

    return ResumeProfile(
        target_roles=roles,
        skills_canonical=skills,
        domains=domains,
        seniority=seniority,
        location_pref="",
    )


def _search_plan_fallback(profile: ResumeProfile, constraints: Dict) -> SearchPlan:
    role_queries = profile.target_roles[:3] if profile.target_roles else ["Software Engineer"]
    skill_queries = profile.skills_canonical[:3] if profile.skills_canonical else []
    domain_queries = profile.domains[:2] if profile.domains else []
    return SearchPlan(
        sources=["career_pages"],
        role_queries=role_queries,
        skill_queries=skill_queries,
        domain_queries=domain_queries,
        max_search_calls=3,
        target_results=80,
    )


def _normalize_company(name: str) -> str:
    return name.strip()


SEARCH_URL_TEMPLATES = {
    "google": "https://careers.google.com/jobs/results/?q={q}",
    "amazon": "https://www.amazon.jobs/en/search?base_query={q}",
    "microsoft": "https://careers.microsoft.com/us/en/search-results?keywords={q}",
    "meta": "https://www.metacareers.com/jobs?keywords={q}",
    "netflix": "https://jobs.netflix.com/search?q={q}",
    "apple": "https://jobs.apple.com/en-us/search?search={q}",
}


def _company_key(name: str) -> str:
    key = name.lower().strip()
    for ch in ["/", "(", ")", ".", ","]:
        key = key.replace(ch, " ")
    key = " ".join(key.split())
    aliases = {
        "google alphabet": "google",
        "meta facebook": "meta",
    }
    return aliases.get(key, key)


def _build_search_url(company: str, base_url: str, role_query: str) -> str:
    key = _company_key(company)
    template = None
    for prefix, value in SEARCH_URL_TEMPLATES.items():
        if key.startswith(prefix):
            template = value
            break
    if not template:
        return base_url
    q = quote_plus(role_query)
    return template.format(q=q)


def _build_candidates(plan: SearchPlan, constraints: Dict) -> List[Dict]:
    candidates = []
    work_mode = constraints.get("work_mode") or "Any"
    location = constraints.get("location") or ""

    role_queries = plan.role_queries or ["Software Engineer"]
    for entry in ALL_COMPANIES:
        company = _normalize_company(entry.get("name", ""))
        url = entry.get("url") or ""
        for role in role_queries[:2]:
            search_url = _build_search_url(company, url, role)
            candidate = {
                "title": role,
                "company": company,
                "location": location or "Various",
                "work_mode": work_mode,
                "url": search_url,
                "posted_date": None,
                "snippet": f"Search {company} careers for {role}",
                "source_id": f"career:{company}:{role}",
                "source": "career_pages",
            }
            candidates.append(candidate)
    return candidates


def _dedupe_candidates(candidates: List[Dict]) -> List[Dict]:
    seen = set()
    deduped = []
    for c in candidates:
        key = (c.get("company"), c.get("title"), c.get("location"), c.get("url"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    return deduped


def _passes_constraints(candidate: Dict, constraints: Dict) -> bool:
    work_mode = constraints.get("work_mode") or "Any"
    if work_mode and work_mode != "Any":
        cand_mode = (candidate.get("work_mode") or "Any").lower()
        if work_mode.lower() not in cand_mode:
            return False

    preferred_roles = constraints.get("preferred_roles") or []
    if preferred_roles:
        title = (candidate.get("title") or "").lower()
        if not any(r.lower() in title for r in preferred_roles):
            return False

    exclude_companies = constraints.get("exclude_companies") or []
    if exclude_companies:
        comp = (candidate.get("company") or "").lower()
        if any(ec.lower() in comp for ec in exclude_companies):
            return False

    return True


def _evaluate_sufficiency(candidates: List[Dict], constraints: Dict) -> bool:
    filtered = [c for c in candidates if _passes_constraints(c, constraints)]
    return len(filtered) >= 20


def _expand_queries(plan: SearchPlan, profile: ResumeProfile) -> SearchPlan:
    synonyms = {
        "backend": ["Server Engineer", "Platform Engineer", "API Engineer"],
        "data": ["Data Engineer", "Analytics Engineer"],
        "ai": ["ML Engineer", "AI Engineer"],
        "frontend": ["Frontend Engineer", "UI Engineer"],
    }
    expanded = list(plan.role_queries)
    for role in plan.role_queries:
        role_lc = role.lower()
        for key, vals in synonyms.items():
            if key in role_lc:
                for v in vals:
                    if v not in expanded:
                        expanded.append(v)
    if profile.target_roles:
        for r in profile.target_roles:
            if r not in expanded:
                expanded.append(r)
    return SearchPlan(
        sources=plan.sources,
        role_queries=expanded[:6],
        skill_queries=plan.skill_queries,
        domain_queries=plan.domain_queries,
        max_search_calls=plan.max_search_calls,
        target_results=plan.target_results,
    )


def _infer_jd_profile(candidate: Dict, profile: ResumeProfile) -> JDProfile:
    skills = []
    for s in profile.skills_canonical[:6]:
        skills.append(s)
    return JDProfile(
        must_have_skills=skills,
        nice_to_have_skills=profile.skills_canonical[6:10],
        seniority=profile.seniority,
        responsibilities=[f"Deliver impact as a {candidate.get('title')}"],
        domain_keywords=profile.domains[:4],
        work_mode=candidate.get("work_mode") or "",
        location=candidate.get("location") or "",
        salary_range="",
    )


def _score_candidate(candidate: Dict, profile: ResumeProfile, constraints: Dict, jd_profile: JDProfile) -> Tuple[int, List[str]]:
    score = 50
    reasons = []
    title = (candidate.get("title") or "").lower()
    for role in profile.target_roles:
        if role.lower() in title:
            score += 12
            reasons.append(f"Role alignment with {role}")
            break

    overlap = 0
    skills = [s.lower() for s in profile.skills_canonical]
    must = [s.lower() for s in jd_profile.must_have_skills]
    for s in must:
        if s in skills:
            overlap += 1
    if must:
        score += min(20, overlap * 4)
        if overlap:
            reasons.append("Skill overlap with must-haves")

    work_mode = constraints.get("work_mode") or "Any"
    if work_mode != "Any":
        score += 8
        reasons.append(f"Work mode matches {work_mode}")

    score = max(0, min(100, score))
    return score, reasons


def _rerank_with_llm(profile: ResumeProfile, candidates: List[Dict], llm: ChatOpenAI, telemetry: Telemetry) -> List[RerankItem]:
    parser = PydanticOutputParser(pydantic_object=RerankItem)
    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a concise recruiter. Rerank a single job candidate vs resume profile."),
        ("human", """
Resume profile:
{profile}

Candidate:
{candidate}

Return JSON only. {format_instructions}
""")
    ])

    results = []
    for c in candidates:
        model_vars = {
            "profile": profile.model_dump(),
            "candidate": c,
            "format_instructions": format_instructions,
        }
        parsed = _call_chain(prompt, parser, llm, model_vars, telemetry, "rerank")
        if parsed:
            results.append(parsed)
    return results


def recommend_jobs(resume_text: str, preferences: Optional[Dict] = None, offline_mode: bool = False) -> Tuple[List[Dict], Dict]:
    prefs = _default_preferences()
    if preferences:
        prefs.update(preferences)

    constraints = {
        "preferred_roles": prefs.get("preferred_roles") or [],
        "work_mode": prefs.get("work_mode") or "Any",
        "location": prefs.get("location") or "",
        "min_salary": prefs.get("min_salary"),
        "seniority": prefs.get("seniority") or "",
        "visa": prefs.get("visa") or "",
        "exclude_industries": prefs.get("exclude_industries") or [],
        "exclude_companies": prefs.get("exclude_companies") or [],
    }

    telemetry = Telemetry()
    state = {
        "profile": None,
        "constraints": constraints,
        "search_plan": None,
        "candidates": [],
        "shortlist": [],
        "telemetry": telemetry.model_dump(),
    }

    llm = None
    if not offline_mode:
        llm = _create_llm()

    if llm:
        profile_parser = PydanticOutputParser(pydantic_object=ResumeProfile)
        profile_prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract a normalized resume profile for job matching."),
            ("human", """
Resume:
{resume_text}

Preferences:
{preferences}

Return JSON only. {format_instructions}
""")
        ])
        profile = _call_chain(
            profile_prompt,
            profile_parser,
            llm,
            {
                "resume_text": resume_text,
                "preferences": prefs,
                "format_instructions": profile_parser.get_format_instructions(),
            },
            telemetry,
            "profile",
        )
    else:
        profile = None

    if not profile:
        profile = _resume_profile_fallback(resume_text, prefs.get("preferred_roles") or [])

    state["profile"] = profile.model_dump()

    if llm:
        plan_parser = PydanticOutputParser(pydantic_object=SearchPlan)
        plan_prompt = ChatPromptTemplate.from_messages([
            ("system", "Create a bounded search plan for job recommendations."),
            ("human", """
Resume profile:
{profile}

Constraints:
{constraints}

Return JSON only. {format_instructions}
""")
        ])
        plan = _call_chain(
            plan_prompt,
            plan_parser,
            llm,
            {
                "profile": profile.model_dump(),
                "constraints": constraints,
                "format_instructions": plan_parser.get_format_instructions(),
            },
            telemetry,
            "search_plan",
        )
    else:
        plan = None

    if not plan:
        plan = _search_plan_fallback(profile, constraints)

    state["search_plan"] = plan.model_dump()

    candidates = _build_candidates(plan, constraints)
    candidates = _dedupe_candidates(candidates)

    if not _evaluate_sufficiency(candidates, constraints):
        plan = _expand_queries(plan, profile)
        candidates = _dedupe_candidates(_build_candidates(plan, constraints))
        state["search_plan"] = plan.model_dump()

    state["candidates"] = candidates

    scored = []
    for c in candidates:
        if not _passes_constraints(c, constraints):
            continue
        jd_profile = _infer_jd_profile(c, profile)
        score, reasons = _score_candidate(c, profile, constraints, jd_profile)
        c_copy = dict(c)
        c_copy["fit_score"] = score
        c_copy["reasons"] = reasons
        c_copy["jd_profile"] = jd_profile.model_dump()
        scored.append(c_copy)

    scored.sort(key=lambda x: x.get("fit_score", 0), reverse=True)
    top_k = scored[:30]

    if llm and top_k:
        reranked = _rerank_with_llm(profile, top_k[:10], llm, telemetry)
        rerank_map = {r.source_id: r for r in reranked}
        for item in top_k:
            rr = rerank_map.get(item.get("source_id"))
            if rr:
                item["fit_score"] = rr.fit_score
                item["top_reasons"] = rr.top_reasons
                item["risks"] = rr.risks
                item["resume_angle"] = rr.resume_angle

    max_results = int(prefs.get("max_results") or 3)
    max_results = max(1, min(3, max_results))
    shortlist = top_k[:max_results]
    state["shortlist"] = shortlist
    state["telemetry"] = telemetry.model_dump()

    recommendations = []
    for item in shortlist:
        recommendations.append({
            "title": item.get("title"),
            "company": item.get("company"),
            "location": item.get("location"),
            "work_mode": item.get("work_mode"),
            "url": item.get("url"),
            "fit_score": item.get("fit_score"),
            "top_reasons": item.get("top_reasons") or item.get("reasons") or [],
            "risks": item.get("risks") or [],
            "resume_angle": item.get("resume_angle") or [],
            "source": item.get("source"),
        })

    return recommendations, state


def default_job_rec_preferences() -> Dict:
    return _default_preferences()
