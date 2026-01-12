
import functools


@functools.lru_cache(maxsize=256)
def parse_resume(text):
    # cached by full resume text; helps avoid re-parsing during iterative workflows
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    sections = {}

    # If the model outputs an explicit HEADER label first, skip it
    if lines and lines[0].strip().upper() == "HEADER":
        lines = lines[1:]

    # Known canonical headings we want to detect even if not fully uppercase
    canonical_headings = set([
        'HEADER', 'SUMMARY', 'EXPERIENCE', 'PROFESSIONAL EXPERIENCE',
        'EDUCATION', 'SKILLS', 'PROJECTS', 'CERTIFICATIONS', 'ACHIEVEMENTS'
    ])

    sections["HEADER"] = []
    # After optional skip, next two lines are Name and Contact if present.
    # Only treat them as HEADER content if they are not themselves section headings.
    if len(lines) >= 1 and lines[0].strip().upper() not in canonical_headings and not lines[0].isupper():
        sections["HEADER"].append(lines[0])
    # Only treat the second line as header content if the first line is not itself
    # a canonical section heading (e.g., when document starts with "SKILLS/EXPERIENCE").
    if len(lines) >= 2 and lines[0].strip().upper() not in canonical_headings and lines[1].strip().upper() not in canonical_headings and not lines[1].isupper():
        sections["HEADER"].append(lines[1])

    # If the name/contact line(s) include a pipe-separated header (e.g.,
    # "John Doe | Software Engineer | City"), move the trailing pipe part
    # into its own line so it doesn't get attached to the previous content.
    try:
        if sections["HEADER"]:
            first = sections["HEADER"][0]
            if '|' in first and not first.lstrip().startswith('-'):
                name, rest = first.split('|', 1)
                sections["HEADER"][0] = _clean_line(name)
                rest = _clean_line(rest)
                if rest:
                    sections["HEADER"].insert(1, rest)
            if len(sections["HEADER"]) > 1:
                second = sections["HEADER"][1]
                if '|' in second and not second.lstrip().startswith('-'):
                    left, rest = second.split('|', 1)
                    sections["HEADER"][1] = _clean_line(left)
                    rest = _clean_line(rest)
                    if rest:
                        sections["HEADER"].insert(2, rest)
    except Exception:
        pass

    # Accept some common alias headings and normalize them to canonical names.
    # Also treat common headings case-insensitively so titles like
    # 'Professional Experience' are recognized as section headings.
    alias_map = {
        'PROFESSIONAL EXPERIENCE': 'EXPERIENCE'
    }

    def _clean_line(s: str) -> str:
        # Remove accidental JSON tokens (e.g., '{}', '[]') that the model may append,
        # and also remove any surrounding commas/spaces so we don't leave orphan commas.
        import re
        if not s:
            return s
        out = s
        # Remove any occurrence of {} or [] with optional surrounding commas and whitespace
        out = re.sub(r"\s*(?:,)?\s*(?:\{\s*\}|\[\s*\])\s*(?:,)?\s*", " ", out)
        # Collapse multiple spaces and normalize commas spacing
        out = re.sub(r"\s+", " ", out).strip()
        # Remove accidental leading/trailing commas and duplicate commas
        out = re.sub(r",\s*,+", ",", out)
        out = out.strip(' ,')
        return out

    current = None
    # Allow a canonical heading to appear in the first or second line.
    # If the first line is a heading (e.g., document starts with "EXPERIENCE"),
    # initialize that section so subsequent lines are captured as content.
    start_idx = 2
    if lines and lines[0].strip().upper() in canonical_headings:
        key = alias_map.get(lines[0].strip().upper(), lines[0].strip().upper())
        if key not in sections:
            sections[key] = []
        current = key
        start_idx = 1
    elif len(lines) >= 2 and lines[1].strip().upper() in canonical_headings:
        key = alias_map.get(lines[1].strip().upper(), lines[1].strip().upper())
        if key not in sections:
            sections[key] = []
        current = key

    for line in lines[start_idx:]:
        # Never redefine HEADER; treat it as a no-op marker
        if line.strip().upper() == "HEADER":
            continue
        candidate = line.strip()
        cand_upper = candidate.upper()

        # If a line **starts with** a known section heading but contains
        # additional content on the same line (e.g., "EXPERIENCE Paycom ..."),
        # split it so the heading becomes a section marker and the trailing
        # content is treated as the first line in that section.
        import re
        m = re.match(r'^(HEADER|SUMMARY|EXPERIENCE|PROFESSIONAL EXPERIENCE|EDUCATION|SKILLS|PROJECTS|CERTIFICATIONS|ACHIEVEMENTS)\b(.*)', cand_upper)
        if m:
            norm = m.group(1)
            key = alias_map.get(norm, norm)
            if key not in sections:
                sections[key] = []
            current = key
            rest = candidate[len(m.group(1)):].lstrip(': -|').strip()
            if rest:
                cleaned = _clean_line(rest)
                if cleaned:
                    sections[current].append(cleaned)
            continue

        # Consider this line a section heading if it's ALL CAPS or matches a
        # known canonical heading case-insensitively.
        if line.isupper() or cand_upper in canonical_headings:
            norm = cand_upper
            key = alias_map.get(norm, norm)
            if key not in sections:
                sections[key] = []
            current = key
        else:
            if current:
                cleaned = _clean_line(line)
                if cleaned:
                    sections[current].append(cleaned)

    return sections


def split_experience(lines):
    import re
    roles = []
    current = None
    for l in lines:
        # Handle lines that include both a bullet and a pipe-separated header
        # Example problematic input:
        #   - Did X, improved Y. Nth Technologies | Software Engineer | 11/2020 – 02/2024
        # We want to extract the trailing company name as the header and preserve the bullet text.
        if "|" in l:
            parts = [p.strip() for p in l.split("|")]
            if len(parts) >= 3:
                # If the first segment looks like a bullet (starts with '-') then try
                # to split the left segment into bullet_text and a trailing Company name.
                left0 = parts[0]
                if left0.lstrip().startswith("-"):
                    left_clean = left0.lstrip('-').strip()
                    # Heuristic: company names often appear as Trailing Capitalized Words
                    m = re.search(r"([A-Z][\w&\.-]*(?:\s+[A-Z][\w&\.-]*)+)$", left_clean)
                    if m:
                        company_name = m.group(1).strip()
                        bullet_text = left_clean[:m.start()].strip()
                    else:
                        # fallback: assume entire left_clean is company (no bullet_text)
                        company_name = left_clean
                        bullet_text = ""

                    # If we have a preceding role, attach bullet_text to it; otherwise,
                    # keep it as the first bullet of the new role later.
                    if current and bullet_text:
                        current["bullets"].append(bullet_text)

                    if current:
                        roles.append(current)
                    current = {
                        "company": company_name,
                        "meta": " | ".join(parts[1:]),
                        "bullets": []
                    }
                    # If there was a bullet_text but no previous role, attach it to this role
                    if not roles and bullet_text:
                        current["bullets"].append(bullet_text)
                    continue
                else:
                    if current:
                        roles.append(current)
                    current = {
                        "company": parts[0],
                        "meta": " | ".join(parts[1:]),
                        "bullets": []
                    }
                    continue
        elif l.startswith("-") and current:
            current["bullets"].append(l[1:].strip())
    if current:
        roles.append(current)
    return roles


def merge_skills_a1(lines):
    buckets, order = {}, []
    for l in lines:
        if ":" in l:
            cat, rest = l.split(":", 1)
            cat = cat.strip()
            items = [i.strip() for i in rest.split(",") if i.strip()]
            if cat not in buckets:
                buckets[cat] = []
                order.append(cat)
            buckets[cat].extend(items)

    merged, i = [], 0
    while i < len(order):
        cat = order[i]
        items = buckets[cat]
        if i + 1 < len(order):
            nxt = order[i + 1]
            merged.append(f"{cat} / {nxt}: " + ", ".join(items + buckets[nxt]))
            i += 2
        else:
            merged.append(f"{cat}: " + ", ".join(items))
            i += 1
    return merged
 
