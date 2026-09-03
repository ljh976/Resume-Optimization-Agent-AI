from core.content_guard import (
    count_experience_bullets,
    estimate_resume_lines,
    restore_omitted_master_bullets,
)


def _resume(bullets):
    bullet_text = "\n".join(f"- {bullet}" for bullet in bullets)
    return f"""HEADER
Jane Doe
jane@example.com | Chicago, IL
SUMMARY
Senior backend engineer.
Cloud platform specialist.
Reliable systems builder.
EXPERIENCE
Acme Corp | Senior Engineer | 2021-Present
{bullet_text}
SKILLS
Languages: Python, SQL
Cloud: AWS, Kubernetes
EDUCATION
State University | B.S. Computer Science
"""


def test_estimated_lines_increase_with_wrapped_bullets():
    short = _resume(["Built APIs."])
    long = _resume(["Built APIs " + "with reliable distributed processing " * 8])

    assert estimate_resume_lines(long) > estimate_resume_lines(short)


def test_restore_omitted_master_bullets_fills_without_removing_draft_content():
    kept = "Built Python APIs for payment processing."
    draft = _resume([kept])
    master = _resume([
        kept,
        "Reduced API latency by 35% through Redis caching and SQL query optimization.",
        "Led an AWS and Kubernetes migration for six production services.",
        "Automated CI/CD checks and reduced deployment failures by 40%.",
        "Designed Kafka event pipelines processing two million messages daily.",
        "Mentored four engineers and standardized incident response practices.",
        "Improved PostgreSQL reliability through monitoring and capacity planning.",
        "Developed service dashboards that shortened incident diagnosis by 30%.",
    ])

    restored, count = restore_omitted_master_bullets(
        draft,
        master,
        "Python AWS Kubernetes APIs and reliable distributed services",
        target_lines=22,
        max_lines=24,
    )

    assert count > 0
    assert kept in restored
    assert count_experience_bullets(restored) > count_experience_bullets(draft)
    assert estimate_resume_lines(restored) >= 22


def test_full_draft_is_not_modified():
    draft = _resume(["Built service number %d with measurable impact." % i for i in range(12)])

    restored, count = restore_omitted_master_bullets(
        draft,
        draft,
        "backend services",
        target_lines=20,
        max_lines=24,
    )

    assert restored == draft
    assert count == 0


def test_long_master_restores_short_draft_to_full_page_range():
    draft = _resume(["Built Python services for core customer workflows."])
    master_bullets = [
        "Built Python service %d with AWS observability, improved processing reliability by %d%%, and sustained on-call quality."
        % (index, 10 + index)
        for index in range(1, 31)
    ]
    master = _resume(master_bullets)

    restored, count = restore_omitted_master_bullets(
        draft,
        master,
        "Python AWS reliable backend services",
        target_lines=56,
        max_lines=60,
        max_chars=3600,
    )

    estimated_lines = estimate_resume_lines(restored)
    assert count >= 8
    assert 52 <= estimated_lines <= 60
    assert len(restored) <= 3600
    assert count_experience_bullets(restored) > count_experience_bullets(draft)
