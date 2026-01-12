import json
from .structure import parse_resume

JOB_MATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "company": {"type": "string"},
                    "url": {"type": "string"},
                    "match_score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 100
                    },
                    "reason": {"type": "string"}
                },
                "required": [
                    "title",
                    "company",
                    "url",
                    "match_score",
                    "reason"
                ]
            }
        }
    },
    "required": ["results"]
}

# ✅ 허용 회사 리스트 (단일 리스트)
ALLOWED_COMPANIES = {
    "Google",
    "Amazon",
    "Apple",
    "Microsoft",
    "Meta",
    "Netflix",
    "Salesforce",
    "Adobe",
    "Oracle",
    "IBM",
    "Cisco",
    "ServiceNow",
    "Workday",
    "NVIDIA",
    "Intel",
    "AMD",
    "Snowflake",
    "Databricks",
    "PayPal",
    "Stripe",
    "Block",
    "Uber",
    "Airbnb",
    "Tesla",
}

# ✅ 회사명 alias 정규화
COMPANY_ALIASES = {
    "Alphabet": "Google",
    "Google LLC": "Google",
    "Meta Platforms": "Meta",
    "Facebook": "Meta",
    "Amazon.com": "Amazon",
    "Amazon.com, Inc.": "Amazon",
    "Square": "Block",
}

def normalize_company(name: str) -> str:
    name = name.strip()
    return COMPANY_ALIASES.get(name, name)


def find_matching_roles(resume_text: str, max_results: int = 8, allowed_companies: list = None, use_llm: bool = False):
    # Small mapping of allowed companies -> canonical careers/search page
    COMPANY_CAREERS = {
        "Google": "https://careers.google.com/jobs/results/",
        "Amazon": "https://www.amazon.jobs/en/search",
        "Apple": "https://jobs.apple.com/en-us/search",
        "Microsoft": "https://careers.microsoft.com/us/en/search-results",
        "Meta": "https://www.metacareers.com/jobs",
        "Netflix": "https://jobs.netflix.com/search",
        "Salesforce": "https://www.salesforce.com/company/careers/",
        "Adobe": "https://www.adobe.com/careers.html",
        "Oracle": "https://www.oracle.com/corporate/careers/",
        "IBM": "https://www.ibm.com/employment/",
        "Cisco": "https://jobs.cisco.com/jobs/SearchJobs",
        "ServiceNow": "https://www.servicenow.com/careers.html",
        "Workday": "https://www.workday.com/en-us/company/careers.html",
        "NVIDIA": "https://www.nvidia.com/en-us/about-nvidia/careers/",
        "Intel": "https://jobs.intel.com/",
        "AMD": "https://www.amd.com/en/corporate/careers",
        "Snowflake": "https://www.snowflake.com/careers/",
        "Databricks": "https://databricks.com/company/careers",
        "PayPal": "https://www.paypal.com/us/webapps/mpp/jobs",
        "Stripe": "https://stripe.com/jobs",
        "Block": "https://careers.block.xyz/",
        "Uber": "https://www.uber.com/global/en/careers/",
        "Airbnb": "https://careers.airbnb.com/",
        "Tesla": "https://www.tesla.com/careers",
    }

    # Parse resume to extract SKILLS section
    skills = []
    try:
        secs = parse_resume(resume_text)
        skills_lines = secs.get("SKILLS", []) or []
        for l in skills_lines:
            if ":" in l:
                _, rest = l.split(":", 1)
                skills.extend([s.strip() for s in rest.split(",") if s.strip()])
            else:
                # comma separated or single line
                skills.extend([s.strip() for s in l.split(",") if s.strip()])
    except Exception:
        skills = []

    # Fallback: simple regex search for a SKILLS line if parser didn't find any
    if not skills:
        try:
            import re
            m = re.search(r"^SKILLS:\s*(.+)$", resume_text, re.I | re.M)
            if m:
                skills = [s.strip() for s in m.group(1).split(',') if s.strip()]
        except Exception:
            pass

    skills_norm = [s.lower() for s in skills]

    # If caller provided an allowed_companies override, normalize it
    if allowed_companies:
        allowed = set([c.strip() for c in allowed_companies if c and c.strip()])
    else:
        allowed = ALLOWED_COMPANIES

    # Mapping: heuristic skill -> likely role title
    SKILL_TO_ROLE = {
        "python": "Backend Engineer",
        "java": "Backend Engineer",
        "c#": "Backend Engineer",
        "aws": "Site Reliability Engineer",
        "docker": "Platform Engineer",
        "kubernetes": "Platform Engineer",
        "react": "Frontend Engineer",
        "typescript": "Frontend Engineer",
        "sql": "Data Engineer",
        "pytorch": "Machine Learning Engineer",
        "tensorflow": "Machine Learning Engineer",
        "ml": "Data Scientist",
        "machine learning": "Data Scientist",
        "data": "Data Scientist",
        "spark": "Data Engineer",
        "devops": "DevOps Engineer",
        "ci/cd": "DevOps Engineer",
        "android": "Mobile Engineer",
        "ios": "Mobile Engineer",
    }

    def pick_role_for_skills(skills_list):
        votes = {}
        for s in skills_list:
            for k, title in SKILL_TO_ROLE.items():
                if k in s:
                    votes[title] = votes.get(title, 0) + 1
        if votes:
            return sorted(votes.items(), key=lambda x: x[1], reverse=True)[0][0]
        return "Software Engineer"

    # Deterministic scoring across allowed companies
    results = []
    for company in sorted(allowed):
        comp = normalize_company(company)
        if comp not in ALLOWED_COMPANIES:
            continue
        base = 30 + min(65, 15 * len(skills_norm))
        if any(comp.lower() in s for s in skills_norm):
            base = min(100, base + 5)
        title = pick_role_for_skills(skills_norm)
        reason = f"Suggested role: {title}; matches on skills: " + (", ".join(skills[:6]) if skills else "general tech skills")
        url = COMPANY_CAREERS.get(comp, f"https://{comp.lower()}.com/careers")
        results.append({
            "title": title,
            "company": comp,
            "url": url,
            "match_score": int(base),
            "reason": reason
        })

    # Optionally augment with an LLM to propose more specific titles (best-effort)
    if use_llm:
        try:
            from openai import OpenAI
            import os
            client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
            prompt = (
                "You are a concise recruiter assistant. Based only on the resume below, return a JSON list of"
                " suggested job titles and a one-line reason for each. Use existing company names only if you are"
                " certain. Output: [{\"company\":..., \"title\":..., \"reason\":...}]\nResume:\n" + resume_text
            )
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            raw = resp.choices[0].message.content
            try:
                parsed = json.loads(raw)
                for item in parsed:
                    comp = normalize_company(item.get('company','')) if item.get('company') else None
                    for r in results:
                        if comp and r['company'] == comp:
                            r['match_score'] = min(100, r['match_score'] + 8)
                            if item.get('title'):
                                r['title'] = item.get('title')
            except Exception:
                pass
        except Exception:
            pass

    # sort by score descending and return up to max_results
    results.sort(key=lambda r: r["match_score"], reverse=True)
    return results[:max_results]
