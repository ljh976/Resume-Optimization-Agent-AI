# Trimming & Inference Policy

간단한 요약 및 운영 지침입니다. 이 문서는 `app.py`의 트리밍 동작과 `REWRITE_PROMPT`로 모델에 전달되는 가이드라인을 설명합니다.

- **Preview-First**: 트리밍 제안은 기본적으로 미리보기 모드로 생성됩니다. 사용자는 각 제안을 검토하여 명시적으로 승인해야만 적용됩니다.
- **보수적 삭제**: 한 번의 자동 실행에서 삭제되는 불릿 수는 `max_removals_per_run`로 제한되어야 합니다. 기본 권장값: 3개.
- **불릿 보존 가드**: 각 역할(role)에서 최소 50%의 불릿은 보존됩니다(절대값으로 1개 이상 유지).
- **점수 기반 예외**: 후보 불릿의 `feedback_score`가 `score_threshold` 이상이면 삭제 제안에서 제외됩니다. 기본 권장값: 60.
- **재삽입(repopulate)**: 모델이 `repopulate_candidates` JSON을 반환하면, 앱이 구조적으로 해당 역할에 후보 불릿을 deterministic하게 삽입합니다. 재삽입은 다음 중 하나를 만족하면 자동 수용될 수 있습니다: 문서 길이 최소값 도달, 또는 +1 불릿 추가, 또는 +50자 증가.
- **사실 추론(Controlled inference)**:
  - ATS 점수가 매우 낮거나(예: < 30) 이력서가 지나치게 짧을 때만 허용.
  - 모델이 새로운(추정) 불릿을 생성할 경우 `inferred: true`와 `reason`을 반드시 포함해야 하며, UI에서 명확히 표시됩니다.
  - 새로 생성된 사실(예: 기술 숙련도 수치, 규모)은 명확한 근거가 없는 경우 허용되지 않습니다.
- **출력 형식 강제**: `REWRITE_PROMPT`는 `change_log`와 선택적 `repopulate_candidates` JSON을 먼저 반환하고, 이어서 수정된 이력서 텍스트를 반환하도록 요구합니다. 앱은 JSON을 우선 파싱합니다.

운영 팁:
- 사이드바에서 `score_threshold`, `max_removals_per_run`, `allow_factual_inference` 토글을 노출해 실험적으로 조정하세요.
- `repopulate_candidates`가 자주 비어있다면 `REWRITE_PROMPT` 예시를 보강하여 모델에게 구조화된 샘플을 더 자주 출력하도록 유도하세요.

문의/기여: 코드 변경은 `core/prompts.py`와 `app.py`의 동작을 일치시켜야 합니다.
