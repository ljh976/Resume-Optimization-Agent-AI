from docx import Document

from core.header_extract import _regex_extract, build_header_lines
from core.render import render_docx
from core.structure import merge_skills_a1, parse_resume, split_experience


RESUME = """HEADER
Jane Doe
Chicago, IL | jane@example.com | (312) 555-0199 | linkedin.com/in/janedoe | github.com/janedoe
SUMMARY
Senior backend engineer.
EXPERIENCE
Acme | Engineer | 2021-Present
- Built reliable APIs.
SKILLS
Languages: Python
EDUCATION
State University | B.S. Computer Science
"""


def test_parse_resume_keeps_complete_contact_row():
    header = parse_resume(RESUME)["HEADER"]

    assert header == [
        "Jane Doe",
        "Chicago, IL | jane@example.com | (312) 555-0199 | linkedin.com/in/janedoe | github.com/janedoe",
    ]


def test_regex_header_extraction_builds_exactly_two_rows_without_duplicates():
    info = _regex_extract(RESUME)
    rows = build_header_lines(info)

    assert rows[0] == "HEADER"
    assert rows[1] == "Jane Doe"
    assert rows[2].count("jane@example.com") == 1
    assert rows[2].count("Chicago, IL") == 1
    assert "(312) 555-0199" in rows[2]
    assert "linkedin.com/in/janedoe" in rows[2]


def test_single_line_header_is_split_into_name_and_contact_row():
    one_line = RESUME.replace(
        "Jane Doe\nChicago, IL | jane@example.com | (312) 555-0199 | linkedin.com/in/janedoe | github.com/janedoe",
        "Jane Doe | Chicago, IL | jane@example.com | (312) 555-0199 | linkedin.com/in/janedoe | github.com/janedoe",
    )

    parsed_header = parse_resume(one_line)["HEADER"]
    rebuilt_header = build_header_lines(_regex_extract(one_line))

    assert parsed_header == [
        "Jane Doe",
        "Chicago, IL | jane@example.com | (312) 555-0199 | linkedin.com/in/janedoe | github.com/janedoe",
    ]
    assert rebuilt_header[1] == "Jane Doe"
    assert rebuilt_header[2].startswith("Chicago, IL | jane@example.com")


def test_docx_renders_name_and_contacts_as_separate_paragraphs(tmp_path):
    sections = parse_resume(RESUME)
    roles = split_experience(sections.get("EXPERIENCE", []))
    skills = merge_skills_a1(sections.get("SKILLS", []))
    output = tmp_path / "resume.docx"

    render_docx(str(output), sections, roles, skills)
    paragraphs = [paragraph.text for paragraph in Document(output).paragraphs if paragraph.text.strip()]

    assert paragraphs[0] == "Jane Doe"
    assert "jane@example.com" in paragraphs[1]
    assert "Chicago, IL" in paragraphs[1]
    assert "Jane Doe" not in paragraphs[1]
    assert "LinkedIn" in paragraphs[1]
    assert "GitHub" in paragraphs[1]
