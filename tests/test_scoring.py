from core.scoring import keyword_coverage, technical_skill_coverage


def test_skill_coverage_uses_jd_requirements_not_all_resume_skills():
    jd = "We need Python, SQL, AWS, and Kubernetes experience."
    focused_resume = "SKILLS\nLanguages: Python, SQL\nCloud: AWS, Kubernetes"
    broad_resume = focused_resume + ", Docker, Java, React, Redis, Tableau"

    focused_score, required, focused_matches = technical_skill_coverage(jd, focused_resume)
    broad_score, _, broad_matches = technical_skill_coverage(jd, broad_resume)

    assert required == {"python", "sql", "aws", "kubernetes"}
    assert focused_matches == required
    assert broad_matches == required
    assert focused_score == broad_score == 1.0


def test_partial_skill_match_is_calibrated_but_not_perfect():
    score, required, matched = technical_skill_coverage(
        "Python, SQL, AWS, and Kubernetes are required.",
        "Built Python and SQL data services.",
    )

    assert len(required) == 4
    assert matched == {"python", "sql"}
    assert 0.6 < score < 0.8


def test_keyword_coverage_ignores_generic_jd_boilerplate():
    jd = "You will work with our team. Required experience building reliable payment services."
    resume = "Built reliable payment services for customers."

    assert keyword_coverage(jd, resume) >= 0.9
