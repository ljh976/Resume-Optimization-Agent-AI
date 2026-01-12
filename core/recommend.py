
def score_job(job, resume_text):
    score = 0
    title = job["title"].lower()
    resume = resume_text.lower()
    for kw in ["backend", "software", "engineer", "data", "platform"]:
        if kw in title and kw in resume:
            score += 1
    return score


def recommend_jobs(jobs, resume_text, top_n=5):
    scored = []
    for j in jobs:
        s = score_job(j, resume_text)
        scored.append((s, j))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [j for s, j in scored[:top_n]]
