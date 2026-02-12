from core.structure import merge_skills_a1


def test_merge_skills_preserves_nonstandard_lines():
    lines = [
        "Python, Go, SQL",  # no category delimiter
        "Cloud：AWS, Azure",  # full-width colon
        "Tools:"  # category-only line
    ]
    merged = merge_skills_a1(lines)

    # Must preserve content rather than dropping SKILLS entirely.
    assert any("Python" in line for line in merged)
    # Category lines may be merged as "Cloud / Tools: ..."
    assert any("Cloud" in line and "AWS" in line for line in merged)
    assert any("Tools" in line for line in merged)


def test_merge_skills_fallback_when_no_colon_lines():
    lines = [
        "Python, Go, SQL",
        "Kubernetes / Docker / Terraform",
    ]
    merged = merge_skills_a1(lines)

    # If no structured category lines can be built, original lines should survive.
    assert merged == lines
