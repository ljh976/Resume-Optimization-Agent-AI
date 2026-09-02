import os
from typing import Dict, List

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .scoring import technical_skill_coverage


class PrescreenResult(BaseModel):
    viable: bool = Field(description="Whether the resume is worth optimizing for this JD")
    skill_match_pct: int = Field(description="Estimated skill match percentage 0-100")
    reasons: List[str] = Field(default_factory=list)
    tips: List[str] = Field(default_factory=list)


def _simple_skill_match(jd: str, resume: str) -> int:
    coverage, _, _ = technical_skill_coverage(jd, resume)
    return int(round(coverage * 100))


def prescreen_resume(jd: str, resume: str, use_llm: bool = True) -> Dict:
    if not use_llm:
        skill_match_pct = _simple_skill_match(jd, resume)
        viable = skill_match_pct >= 15
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
        ("system", "You are a practical recruiter doing a quick prescreen."),
        ("human", """
JD:
{jd}

Resume:
{resume}

        Decide if the resume is worth optimizing for this JD. Focus on core skill and domain alignment.
        A partial match is viable when the candidate has transferable experience; mark it non-viable only
        when the core job family or essential skills are clearly unrelated.
Return JSON only. {format_instructions}
""")
    ])

    chain = prompt | llm
    raw = chain.invoke({"jd": jd, "resume": resume, "format_instructions": format_instructions})
    text = raw.content if hasattr(raw, "content") else str(raw)
    parsed = parser.parse(text)

    return parsed.model_dump()
