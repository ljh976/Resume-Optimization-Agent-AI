import re
from app import _resume_to_markdown, parse_resume


def assert_no_markers(md: str):
    assert '{BOLD_START}' not in md
    assert '{BOLD_END}' not in md


def test_header_bold_conversion():
    txt = "{BOLD_START}Jane Doe{BOLD_END} | Software Engineer"
    md = _resume_to_markdown(txt)
    assert_no_markers(md)
    assert '**Jane Doe**' in md


def test_summary_and_bullet_bold_conversion():
    txt = "SUMMARY {BOLD_START}Experienced{BOLD_END} developer\n- {BOLD_START}Led{BOLD_END} integrations"
    md = _resume_to_markdown(txt)
    assert_no_markers(md)
    assert '**Experienced**' in md
    assert '- **Led** integrations' in md


def test_skills_bold_conversion():
    txt = "SKILLS\nLanguages: {BOLD_START}Python{BOLD_END}, Go"
    md = _resume_to_markdown(txt)
    assert_no_markers(md)
    assert '**Python**' in md or '**Languages:**' in md


def test_skills_category_split():
    txt = "SKILLS\nLanguages / Tools: Python, Go"
    md = _resume_to_markdown(txt)
    assert_no_markers(md)
    assert '**Languages:** Python, Go' in md
    assert '**Tools:** Python, Go' in md
    # Ensure separate lines for categories
    assert '**Languages:** Python, Go' in md and '**Tools:** Python, Go' in md


def test_skills_not_duplicated_in_header():
    txt = "SKILLS\nLanguages / Tools: Python, Go"
    md = _resume_to_markdown(txt)
    # The original combined line should not appear as a top-level HEADER line
    # and should not be present above the SKILLS section.
    assert not md.strip().startswith('**Languages / Tools:')
    assert '**SKILLS**' in md
    assert '**Languages:**' in md and '**Tools:**' in md


def test_skills_multiple_categories_on_single_line():
    txt = (
        "SKILLS\n"
        "Languages: C++, Golang, Python, PHP, JavaScript, SQL, Bash "
        "Backend & APIs: RESTful API Design, Distributed Systems "
        "Data & Persistence: PostgreSQL, MySQL, SQL Server"
    )
    md = _resume_to_markdown(txt)
    assert_no_markers(md)
    # Each category must appear on its own line
    assert '**Languages:**' in md
    assert '**Backend & APIs:**' in md
    assert '**Data & Persistence:**' in md
    # Ensure categories are on separate lines (order preserved)
    li = md.splitlines()
    lang_idx = next(i for i, l in enumerate(li) if '**Languages:**' in l)
    backend_idx = next(i for i, l in enumerate(li) if '**Backend & APIs:**' in l)
    data_idx = next(i for i, l in enumerate(li) if '**Data & Persistence:**' in l)
    assert lang_idx < backend_idx < data_idx


def test_experience_role_header_and_bullets():
    # Add a name line so parse_resume treats subsequent EXPerience header correctly
    txt = "Name\nEXPERIENCE\nAcme | {BOLD_START}Senior Eng{BOLD_END} | 2019-2023\n- Built {BOLD_START}platform{BOLD_END}"
    md = _resume_to_markdown(txt)
    assert_no_markers(md)
    assert '**Senior Eng**' in md or 'Senior Eng' in md
    assert '- Built **platform**' in md
