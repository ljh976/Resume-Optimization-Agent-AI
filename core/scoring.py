"""Deterministic matching primitives used by ATS and pre-screen scoring."""

import re


_STOPWORDS = {
    "about", "after", "also", "among", "and", "are", "been", "being", "both",
    "but", "can", "company", "description", "each", "for", "from", "have", "into",
    "job", "more", "must", "our", "preferred", "qualifications", "required", "requirements",
    "role", "should", "that", "the", "their", "these", "they", "this", "through", "using",
    "what", "when", "where", "which", "will", "with", "work", "working", "years", "you",
    "your", "experience", "responsibilities", "including", "skills", "ability", "team",
}

# The list is intentionally broad but finite: unlike all JD words, these terms are
# credible hard-skill requirements and produce a meaningful denominator.
_TECH_SKILLS = (
    "a/b testing", "active directory", "agile", "airflow", "angular", "ansible", "apache spark",
    "aws", "azure", "bash", "bigquery", "c#", "c++", "cassandra", "ci/cd", "cloudformation",
    "computer vision", "css", "databricks", "data modeling", "data warehouse", "django", "docker",
    "dynamodb", "elasticsearch", "etl", "fastapi", "flask", "gcp", "git", "github actions", "go",
    "golang", "graphql", "hadoop", "html", "java", "javascript", "jenkins", "jira", "kafka",
    "keras", "kotlin", "kubernetes", "langchain", "linux", "llm", "machine learning", "matlab",
    "mongodb", "mysql", "natural language processing", "next.js", "node.js", "nosql", "numpy",
    "oauth", "openai", "oracle", "pandas", "perl", "php", "postgresql", "power bi", "pytorch",
    "python", "react", "redis", "rest api", "ruby", "rust", "salesforce", "scala", "scikit-learn", "scrum",
    "snowflake", "spark", "spring", "sql", "sqlite", "swift", "tableau", "tensorflow", "terraform",
    "typescript", "unix", "vue", ".net",
)


def _contains_term(text: str, term: str) -> bool:
    pattern = r"(?<![\w])" + re.escape(term).replace(r"\ ", r"[\s\-/]+") + r"(?![\w])"
    return bool(re.search(pattern, text, re.IGNORECASE))


def extract_technical_skills(text: str) -> set[str]:
    return {skill for skill in _TECH_SKILLS if _contains_term(text or "", skill)}


def meaningful_tokens(text: str) -> set[str]:
    # Keep dots only when they join token segments (for example node.js), rather
    # than accidentally treating sentence-final punctuation as part of a word.
    tokens = set(
        re.findall(r"[a-zA-Z][a-zA-Z0-9+#]*(?:\.[a-zA-Z0-9+#]+)*", (text or "").lower())
    )
    return {
        token for token in tokens
        if len(token) >= 2 and token not in _STOPWORDS and not token.isdigit()
    }


def keyword_coverage(jd: str, resume: str) -> float:
    """Return calibrated JD keyword recall in the 0..1 range.

    Natural-language JDs contain many one-off words that should not all be required
    in a resume. Matching roughly 55% of meaningful terms therefore represents full
    practical coverage rather than demanding a keyword-stuffed copy of the JD.
    """
    jd_tokens = meaningful_tokens(jd)
    if not jd_tokens:
        return 0.0
    resume_tokens = meaningful_tokens(resume)
    raw_recall = len(jd_tokens & resume_tokens) / float(len(jd_tokens))
    return min(raw_recall / 0.55, 1.0)


def technical_skill_coverage(jd: str, resume: str) -> tuple[float, set[str], set[str]]:
    required = extract_technical_skills(jd)
    if not required:
        fallback = keyword_coverage(jd, resume)
        return fallback, set(), set()
    matched = {skill for skill in required if _contains_term(resume or "", skill)}
    raw_recall = len(matched) / float(len(required))
    # A resume need not contain every preferred technology to be a strong match.
    calibrated = min(raw_recall / 0.75, 1.0)
    return calibrated, required, matched
