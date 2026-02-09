import os
import re
from typing import Dict, List

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class HeaderInfo(BaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    linkedin: str = ""
    github: str = ""
    location: str = ""


def _regex_extract(resume_text: str) -> Dict:
    text = resume_text or ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    email = ""
    phone = ""
    linkedin = ""
    github = ""
    location = ""
    name = ""

    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    if email_match:
        email = email_match.group(0)

    phone_match = re.search(r"\+?\d[\d\s().-]{8,}\d", text)
    if phone_match:
        phone = phone_match.group(0).strip()

    linkedin_match = re.search(r"(https?://)?(www\.)?linkedin\.com/[^\s|]+", text, re.I)
    if linkedin_match:
        linkedin = linkedin_match.group(0)

    github_match = re.search(r"(https?://)?(www\.)?github\.com/[^\s|]+", text, re.I)
    if github_match:
        github = github_match.group(0)

    if lines:
        name = lines[0]
        if len(lines) > 1 and any(ch.isdigit() for ch in lines[1]):
            location = lines[1]

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "linkedin": linkedin,
        "github": github,
        "location": location,
    }


def extract_header_info(resume_text: str, use_llm: bool = True) -> Dict:
    if not use_llm:
        return _regex_extract(resume_text)

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0.0)

    parser = PydanticOutputParser(pydantic_object=HeaderInfo)
    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract header contact info from the resume text."),
        ("human", """
Resume:
{resume}

Return JSON only. {format_instructions}
""")
    ])

    chain = prompt | llm
    raw = chain.invoke({"resume": resume_text, "format_instructions": format_instructions})
    text = raw.content if hasattr(raw, "content") else str(raw)
    parsed = parser.parse(text)
    return parsed.model_dump()


def build_header_lines(info: Dict) -> List[str]:
    parts = []
    location = (info.get("location") or "").strip()
    email = (info.get("email") or "").strip()
    phone = (info.get("phone") or "").strip()
    linkedin = (info.get("linkedin") or "").strip()
    github = (info.get("github") or "").strip()

    for item in [location, email, phone, linkedin, github]:
        if item:
            parts.append(item)

    contact_line = " | ".join(parts)
    name = (info.get("name") or "").strip()

    lines = ["HEADER"]
    if name:
        lines.append(name)
    if contact_line:
        lines.append(contact_line)
    return lines
