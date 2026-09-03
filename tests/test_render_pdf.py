from io import BytesIO

from pypdf import PdfReader

from core.render_pdf import render_pdf_bytes


def test_render_pdf_contains_resume_sections_and_content():
    resume = """HEADER
Jane Doe
jane@example.com | linkedin.com/in/janedoe
SUMMARY
Backend engineer building reliable services.
EXPERIENCE
Acme Corp | Senior Engineer | 2021-Present
- Built Python APIs that reduced latency by 35%.
SKILLS
Languages: Python, SQL
EDUCATION
State University | B.S. Computer Science
"""

    payload = render_pdf_bytes(resume)
    reader = PdfReader(BytesIO(payload))
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)

    assert payload.startswith(b"%PDF")
    assert len(reader.pages) >= 1
    assert "Jane Doe" in extracted
    assert "EXPERIENCE" in extracted
    assert "Built Python APIs" in extracted
