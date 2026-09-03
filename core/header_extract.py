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

    canonical_headings = {
        "HEADER", "SUMMARY", "PROFESSIONAL SUMMARY", "EXPERIENCE",
        "PROFESSIONAL EXPERIENCE", "SKILLS", "EDUCATION",
    }
    header_lines = []
    for line in lines:
        if line.upper() == "HEADER":
            continue
        if line.upper() in canonical_headings:
            if header_lines:
                break
            continue
        header_lines.append(line)
        if len(header_lines) >= 4:
            break
    header_text = "\n".join(header_lines)

    email = ""
    phone = ""
    linkedin = ""
    github = ""
    location = ""
    name = ""

    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", header_text)
    if email_match:
        email = email_match.group(0)

    phone_match = re.search(
        r"(?<!\w)(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s.-]\d{3,4}[\s.-]\d{4}(?!\w)",
        header_text,
    )
    if phone_match:
        phone = phone_match.group(0).strip()

    linkedin_match = re.search(r"(https?://)?(www\.)?linkedin\.com/[^\s|]+", header_text, re.I)
    if linkedin_match:
        linkedin = linkedin_match.group(0)

    github_match = re.search(r"(https?://)?(www\.)?github\.com/[^\s|]+", header_text, re.I)
    if github_match:
        github = github_match.group(0)

    if header_lines:
        name = header_lines[0].split("|", 1)[0].strip()
        for line in header_lines:
            for part in [item.strip() for item in line.split("|")]:
                lowered = part.lower()
                if part == name:
                    continue
                if not part or "@" in part or "linkedin" in lowered or "github" in lowered or "http" in lowered:
                    continue
                if phone and phone in part:
                    continue
                if re.search(r"[A-Za-z]", part) and ("," in part or re.search(r"\b[A-Z]{2}\b", part)):
                    location = part
                    break
            if location:
                break

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "linkedin": linkedin,
        "github": github,
        "location": location,
    }


def extract_header_info(resume_text: str, use_llm: bool = True) -> Dict:
    regex_info = _regex_extract(resume_text)
    if not use_llm:
        return regex_info

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
    parsed = parser.parse(text).model_dump()
    # Deterministic matches from the actual header take precedence. This keeps
    # an LLM from duplicating the entire contact row into the location/name field.
    for field, value in regex_info.items():
        if value:
            parsed[field] = value
    return parsed


def build_header_lines(info: Dict) -> List[str]:
    parts = []
    location = (info.get("location") or "").strip()
    email = (info.get("email") or "").strip()
    phone = (info.get("phone") or "").strip()
    linkedin = (info.get("linkedin") or "").strip()
    github = (info.get("github") or "").strip()

    seen = set()
    for item in [location, email, phone, linkedin, github]:
        normalized = item.casefold().strip()
        if item and normalized not in seen:
            parts.append(item)
            seen.add(normalized)

    contact_line = " | ".join(parts)
    name = (info.get("name") or "").split("|", 1)[0].strip()

    lines = ["HEADER"]
    if name:
        lines.append(name)
    if contact_line:
        lines.append(contact_line)
    return lines
