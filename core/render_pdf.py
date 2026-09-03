"""PDF rendering for optimized resumes."""

from html import escape
from io import BytesIO
import re

from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from .structure import merge_skills_a1, parse_resume, split_experience


def _pdf_markup(value: str) -> str:
    """Escape user text while preserving the app's explicit bold markers."""
    text = (value or "").replace("\u00a0", " ")
    text = text.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    marked = escape(text)
    marked = marked.replace("{BOLD_START}", "<b>").replace("{BOLD_END}", "</b>")
    marked = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", marked)
    return marked


def render_pdf_bytes(resume_text: str) -> bytes:
    """Render a structured resume as a polished, downloadable PDF."""
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        rightMargin=0.62 * inch,
        leftMargin=0.62 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="Optimized Resume",
        author="Resume Optimization Agent",
    )

    sample = getSampleStyleSheet()
    name_style = ParagraphStyle(
        "ResumeName", parent=sample["Normal"], fontName="Helvetica-Bold",
        fontSize=17, leading=19, alignment=TA_CENTER, spaceAfter=2,
    )
    contact_style = ParagraphStyle(
        "ResumeContact", parent=sample["Normal"], fontName="Helvetica",
        fontSize=8.5, leading=10, alignment=TA_CENTER, spaceAfter=6,
    )
    section_style = ParagraphStyle(
        "ResumeSection", parent=sample["Normal"], fontName="Helvetica-Bold",
        fontSize=10, leading=12, spaceBefore=5, spaceAfter=2,
        borderWidth=0, borderPadding=0, keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "ResumeBody", parent=sample["Normal"], fontName="Helvetica",
        fontSize=9, leading=11, spaceAfter=1,
    )
    role_style = ParagraphStyle(
        "ResumeRole", parent=body_style, fontName="Helvetica",
        fontSize=9.3, leading=11, spaceBefore=2, spaceAfter=1, keepWithNext=True,
    )
    bullet_style = ParagraphStyle(
        "ResumeBullet", parent=body_style, leftIndent=12, firstLineIndent=-8,
        bulletIndent=0, spaceAfter=1,
    )

    sections = parse_resume(resume_text or "")
    story = []
    header = sections.get("HEADER", []) or []
    if header:
        story.append(Paragraph(_pdf_markup(header[0]), name_style))
        if len(header) > 1:
            story.append(Paragraph(_pdf_markup(header[1]), contact_style))

    summary = sections.get("SUMMARY", []) or sections.get("PROFESSIONAL SUMMARY", []) or []
    if summary:
        story.append(Paragraph("SUMMARY", section_style))
        for line in summary:
            story.append(Paragraph(_pdf_markup(line), body_style))

    experience = sections.get("EXPERIENCE", []) or sections.get("PROFESSIONAL EXPERIENCE", []) or []
    if experience:
        story.append(Paragraph("EXPERIENCE", section_style))
        roles = split_experience(experience)
        if roles:
            for role in roles:
                company = _pdf_markup(role.get("company") or "")
                meta = _pdf_markup(role.get("meta") or "")
                role_header = f"<b>{company}</b>"
                if meta:
                    role_header += f" | {meta}"
                story.append(Paragraph(role_header, role_style))
                for bullet in role.get("bullets") or []:
                    story.append(Paragraph(_pdf_markup(bullet), bullet_style, bulletText="-"))
        else:
            for line in experience:
                stripped = (line or "").strip()
                if stripped.startswith("-"):
                    story.append(Paragraph(_pdf_markup(stripped[1:].strip()), bullet_style, bulletText="-"))
                elif stripped:
                    story.append(Paragraph(_pdf_markup(stripped), role_style))

    skills = merge_skills_a1(sections.get("SKILLS", []) or [])
    if skills:
        story.append(Paragraph("SKILLS", section_style))
        for line in skills:
            if ":" in line:
                category, values = line.split(":", 1)
                rendered = f"<b>{_pdf_markup(category)}:</b> {_pdf_markup(values.strip())}"
            else:
                rendered = _pdf_markup(line)
            story.append(Paragraph(rendered, body_style))

    education = sections.get("EDUCATION", []) or []
    if education:
        story.append(Paragraph("EDUCATION", section_style))
        for line in education:
            story.append(Paragraph(_pdf_markup(line), body_style))

    if not story:
        for line in (resume_text or "").splitlines():
            if line.strip():
                story.append(Paragraph(_pdf_markup(line), body_style))
        if not story:
            story.append(Paragraph("Resume", body_style))

    story.append(Spacer(1, 1))
    document.build(story)
    return buffer.getvalue()
