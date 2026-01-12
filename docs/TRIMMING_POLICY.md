# Trimming & Inference Policy

A concise summary and operational guidelines. This document explains the trimming behavior in `app.py` and the guidelines sent to the model via `REWRITE_PROMPT`.

- **Preview-First**: Trimming suggestions are generated in preview mode by default. Users must explicitly approve each suggestion before it is applied.
- **Conservative removals**: The number of bullets removed in a single automated run should be limited by `max_removals_per_run`. Recommended default: 3.
- **Bullet preservation guard**: At least 50% of bullets in each role should be preserved (with an absolute minimum of 1 bullet kept).
- **Score-based exceptions**: Candidate bullets with a `feedback_score` at or above `score_threshold` should be excluded from removal suggestions. Recommended default: 60.
- **Repopulation**: If the model returns `repopulate_candidates` JSON, the app will deterministically insert candidate bullets into the given role. Repopulation may be auto-accepted when one of the following conditions is met: minimum document length reached, +1 bullet added, or +50 characters added.
- **Controlled inference**:
  - Only allow inferred facts when the ATS score is very low (e.g., < 30) or the resume is excessively short.
  - If the model generates new (inferred) bullets, it must include `inferred: true` and a `reason`; the UI must clearly indicate these bullets.
  - Newly inferred facts (e.g., proficiency percentages, team size) are not permitted unless there is clear evidence.
- **Enforced output format**: The `REWRITE_PROMPT` must return a `change_log` and an optional `repopulate_candidates` JSON first, followed by the rewritten resume text. The app parses the JSON first.

Operational tips:

- Expose `score_threshold`, `max_removals_per_run`, and `allow_factual_inference` toggles in the sidebar for experimental tuning.
- If `repopulate_candidates` is frequently empty, strengthen `REWRITE_PROMPT` examples to encourage more structured outputs.

Questions / contributions: Coordinate any prompt or behavior changes between `core/prompts.py` and `app.py`.
