import os
from typing import Dict, List

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .structure import parse_resume


class PrescreenResult(BaseModel):
    viable: bool = Field(description="Whether the resume is worth optimizing for this JD")
    skill_match_pct: int = Field(description="Estimated skill match percentage 0-100")
    reasons: List[str] = Field(default_factory=list)
    tips: List[str] = Field(default_factory=list)


def _simple_skill_match(jd: str, resume: str) -> int:
    jd_lc = (jd or "").lower()
    resume_lc = (resume or "").lower()

    skills = []
    try:
        sections = parse_resume(resume)
        skills_lines = sections.get("SKILLS", []) or []
        for line in skills_lines:
            if ":" in line:
                _, rest = line.split(":", 1)
                skills.extend([s.strip().lower() for s in rest.split(",") if s.strip()])
            else:
                skills.extend([s.strip().lower() for s in line.split(",") if s.strip()])
    except Exception:
        skills = []

    if not skills:
        return 0

    matched = sum(1 for s in skills if s and s in jd_lc)
    return int(round((matched / float(len(skills))) * 100))


def prescreen_resume(jd: str, resume: str, use_llm: bool = True) -> Dict:
    if not use_llm:
        skill_match_pct = _simple_skill_match(jd, resume)
        viable = skill_match_pct >= 20
        tips = []
        if not viable:
            tips = [
                "Add 5-8 JD keywords to SKILLS and EXPERIENCE bullets.",
                "Align job titles or role focus with the JD's primary role.",
            ]
        return {
            "viable": viable,
            "skill_match_pct": skill_match_pct,
            "reasons": ["Heuristic skill match"],
            "tips": tips,
        }

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0.0)

    parser = PydanticOutputParser(pydantic_object=PrescreenResult)
    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict recruiter doing a quick prescreen."),
        ("human", """
JD:
{jd}

Resume:
{resume}

Decide if the resume is worth optimizing for this JD. Focus on skill alignment.
Return JSON only. {format_instructions}
""")
    ])

    chain = prompt | llm
    raw = chain.invoke({"jd": jd, "resume": resume, "format_instructions": format_instructions})
    text = raw.content if hasattr(raw, "content") else str(raw)
    parsed = parser.parse(text)

    return parsed.model_dump()
