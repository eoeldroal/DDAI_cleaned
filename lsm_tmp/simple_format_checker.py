import re
import json

def simple_format_checker(data_source, solution_str, ground_truth, extra_info):
    """
    Assistant의 전체 대화 기록(solution_str)을 검사하여,
    모든 턴이 정해진 문법 규칙을 따랐는지 확인하고, 실패 원인을 출력 및 반환합니다.
    Returns:
        score (float): 1.0 (Pass) or 0.0 (Fail)
        reason (str): 실패 원인 설명 (성공 시 None)
    """

    # 1단계: Assistant의 턴(Turn)만 분리하기
    # - Phase 1: action-only trajectory
    # - Phase 2+: trajectory may contain a final "<answer>...</answer>" block (frozen generator)
    assistant_turns = re.findall(r"<\|im_start\|>assistant(.*?)<\|im_end\|>", solution_str, re.DOTALL)

    # Fallback: 일부 파이프라인에서는 <|im_start|>assistant 토큰이 응답에 없을 수 있다.
    # 이 경우 전체 문자열을 하나의 assistant 턴으로 간주해 검사한다.
    if not assistant_turns:
        assistant_turns = [solution_str]

    # 2단계: action 턴 vs answer 턴 분리
    action_turns: list[tuple[int, str]] = []
    answer_only_turns: list[tuple[int, str]] = []

    for i, turn in enumerate(assistant_turns):
        cleaned_turn = turn.strip()
        action_count = cleaned_turn.count('<search>') + cleaned_turn.count('<bbox>') + cleaned_turn.count('<search_complete>')
        has_answer = ('<answer>' in cleaned_turn) or ('</answer>' in cleaned_turn)

        # answer-only 턴은 action 태그가 없고, 내용이 <answer>...</answer>로만 구성된 경우만 허용
        if action_count == 0 and has_answer:
            if cleaned_turn.startswith('<answer>') and cleaned_turn.endswith('</answer>'):
                answer_only_turns.append((i, cleaned_turn))
                continue
            # answer 태그가 섞인 이상한 케이스는 실패 처리
            return 0.0, f"Turn {i} contains malformed/mixed <answer> block"

        # action 태그가 없는 턴은 허용하지 않음 (answer-only 제외)
        if action_count == 0:
            return 0.0, f"Turn {i} missing action tag"

        # action 태그가 있는데 answer가 섞이면 strict fail (answer는 frozen generator 전용)
        if has_answer:
            return 0.0, f"Turn {i} contains <answer> inside action turn"

        action_turns.append((i, cleaned_turn))

    if not action_turns:
        return 0.0, "No action turns found"

    # 3단계: action 턴 공통 문법 검사
    for i, cleaned_turn in action_turns:
        # 3-A: <think> 태그 검사
        if not cleaned_turn.startswith('<think>'):
            return 0.0, f"Turn {i} missing <think> start tag. {cleaned_turn[:50]}..."

        if cleaned_turn.count('<think>') != 1 or cleaned_turn.count('</think>') != 1:
            return 0.0, f"Turn {i} incorrect <think> tag count. {cleaned_turn[:50]}..."

        # 3-B: 행동(Action) 태그 개수 검사 (정확히 1개)
        action_count = cleaned_turn.count('<search>') + cleaned_turn.count('<bbox>') + cleaned_turn.count('<search_complete>')
        if action_count != 1:
            return 0.0, f"Turn {i} invalid action count ({action_count})"

        # 3-C: 행동(Action) 태그 내용 검사 (세부 규칙)
        if '<search>' in cleaned_turn:
            match = re.search(r"<search>(.*?)</search>", cleaned_turn, re.DOTALL)
            if not match or not match.group(1).strip():
                return 0.0, f"Turn {i} empty/malformed <search>"

        elif '<bbox>' in cleaned_turn:
            match = re.search(r"<bbox>(.*?)</bbox>", cleaned_turn, re.DOTALL)
            if not match:
                return 0.0, f"Turn {i} malformed <bbox>"
            try:
                bbox_content = json.loads(match.group(1).strip())
                if not isinstance(bbox_content, list) or len(bbox_content) != 4:
                    return 0.0, f"Turn {i} bbox format error (not length 4)"
                if not all(isinstance(coord, (int, float)) for coord in bbox_content):
                    return 0.0, f"Turn {i} bbox non-number values"
            except json.JSONDecodeError:
                return 0.0, f"Turn {i} bbox JSON decode error"

        elif '<search_complete>' in cleaned_turn:
            if '<search_complete>true</search_complete>' not in cleaned_turn.replace(" ", ""):
                return 0.0, f"Turn {i} <search_complete> value error"

    # 4단계: 마지막 action 턴은 반드시 <search_complete> 여야 함
    last_action_idx, last_action_turn = action_turns[-1]
    if '<search_complete>' not in last_action_turn:
        return 0.0, "Last action turn missing <search_complete>"

    # 5단계: answer-only 턴은 마지막 action 턴 이후에만 허용
    for idx, _turn in answer_only_turns:
        if idx < last_action_idx:
            return 0.0, "Answer block appears before search_complete"

    # 6단계: 최종 합격 판정
    return 1.0, None
