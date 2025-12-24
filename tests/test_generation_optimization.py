"""
generation.py 최적화 전후 동작 검증 테스트

이 테스트는 최적화 전후에 함수의 입출력이 동일함을 보장합니다.
정규식 패턴 매칭, 태그 추출 등 핵심 로직을 검증합니다.
"""

import pytest
import re
from typing import List, Tuple, Any


# =============================================================================
# 테스트 대상 함수들 (독립 실행을 위해 여기에 복사)
# =============================================================================

# 원본 정규식 패턴 (최적화 전)
ORIGINAL_TAG_PATTERN = r"<(search|think|bbox|search_complete)>(.*?)</\1>"
ORIGINAL_ACTION_PATTERN = r'<(search|bbox|search_complete)>(.*?)</\1>'
ORIGINAL_UID_PATTERN = r'(\d+)$'

# 사전 컴파일된 정규식 패턴 (최적화 후)
COMPILED_TAG_PATTERN = re.compile(r"<(search|think|bbox|search_complete)>(.*?)</\1>", re.DOTALL)
COMPILED_ACTION_PATTERN = re.compile(r'<(search|bbox|search_complete)>(.*?)</\1>', re.DOTALL)
COMPILED_UID_PATTERN = re.compile(r'(\d+)$')


# =============================================================================
# 헬퍼 함수
# =============================================================================

def extract_tags_original(text: str) -> str:
    """원본 extract_tags 함수 (매번 정규식 컴파일)"""
    pattern = r"<(search|think|bbox|search_complete)>(.*?)</\1>"
    matches = re.findall(pattern, text, re.DOTALL)
    result = "\n".join([f"<{tag}>{content}</{tag}>" for tag, content in matches])
    return result


def extract_tags_optimized(text: str) -> str:
    """최적화된 extract_tags 함수 (사전 컴파일된 정규식 사용)"""
    matches = COMPILED_TAG_PATTERN.findall(text)
    result = "\n".join([f"<{tag}>{content}</{tag}>" for tag, content in matches])
    return result


def postprocess_prediction_original(prediction: str) -> Tuple[str, str]:
    """원본 postprocess_predictions 로직 (매번 정규식 컴파일)"""
    pattern = r'<(search|bbox|search_complete)>(.*?)</\1>'
    match = re.search(pattern, prediction, re.DOTALL)
    if match:
        content = match.group(2).strip()
        action = match.group(1)
    else:
        content = ''
        action = None
    return action, content


def postprocess_prediction_optimized(prediction: str) -> Tuple[str, str]:
    """최적화된 postprocess_predictions 로직 (사전 컴파일된 정규식 사용)"""
    match = COMPILED_ACTION_PATTERN.search(prediction)
    if match:
        content = match.group(2).strip()
        action = match.group(1)
    else:
        content = ''
        action = None
    return action, content


def extract_uid_original(uid: str) -> int:
    """원본 UID 추출 (매번 정규식 컴파일)"""
    m = re.search(r'(\d+)$', str(uid))
    return int(m.group(1)) if m else -1


def extract_uid_optimized(uid: str) -> int:
    """최적화된 UID 추출 (사전 컴파일된 정규식 사용)"""
    m = COMPILED_UID_PATTERN.search(str(uid))
    return int(m.group(1)) if m else -1


# =============================================================================
# 테스트 케이스
# =============================================================================

class TestExtractTags:
    """extract_tags 함수 테스트"""

    @pytest.mark.parametrize("text,expected_tags", [
        # 기본 케이스
        (
            "<think>reasoning here</think><search>query</search>",
            ["think", "search"]
        ),
        # bbox 포함
        (
            "<think>analysis</think><bbox>[10, 20, 30, 40]</bbox>",
            ["think", "bbox"]
        ),
        # search_complete 포함
        (
            "<think>done</think><search_complete>true</search_complete>",
            ["think", "search_complete"]
        ),
        # 빈 문자열
        ("", []),
        # 태그 없음
        ("just plain text", []),
        # 중첩 태그 (내부 태그만 추출)
        (
            "<think>outer<search>inner</search>end</think>",
            ["think"]  # 외부 태그만 매칭
        ),
        # 여러 개의 동일 태그
        (
            "<search>query1</search><search>query2</search>",
            ["search", "search"]
        ),
        # 멀티라인 콘텐츠
        (
            "<think>\nline1\nline2\n</think>",
            ["think"]
        ),
    ])
    def test_extract_tags_equivalence(self, text, expected_tags):
        """원본과 최적화 버전이 동일한 결과를 반환하는지 검증"""
        original_result = extract_tags_original(text)
        optimized_result = extract_tags_optimized(text)

        # 결과가 동일해야 함
        assert original_result == optimized_result, \
            f"결과 불일치:\n원본: {original_result}\n최적화: {optimized_result}"

        # 추출된 태그 수 검증
        if expected_tags:
            for tag in expected_tags:
                assert f"<{tag}>" in optimized_result


class TestPostprocessPredictions:
    """postprocess_predictions 함수 테스트"""

    @pytest.mark.parametrize("prediction,expected_action,expected_content", [
        # search 액션
        ("<search>what is AI?</search>", "search", "what is AI?"),
        # bbox 액션
        ("<bbox>[10, 20, 30, 40]</bbox>", "bbox", "[10, 20, 30, 40]"),
        # search_complete 액션
        ("<search_complete>true</search_complete>", "search_complete", "true"),
        # 액션 없음
        ("just plain text", None, ""),
        # 빈 문자열
        ("", None, ""),
        # 공백이 있는 콘텐츠
        ("<search>  query with spaces  </search>", "search", "query with spaces"),
        # think 태그 (action에 포함되지 않음)
        ("<think>reasoning</think>", None, ""),
        # 여러 태그 (첫 번째만 추출)
        ("<search>first</search><bbox>[1,2,3,4]</bbox>", "search", "first"),
    ])
    def test_postprocess_equivalence(self, prediction, expected_action, expected_content):
        """원본과 최적화 버전이 동일한 결과를 반환하는지 검증"""
        orig_action, orig_content = postprocess_prediction_original(prediction)
        opt_action, opt_content = postprocess_prediction_optimized(prediction)

        # 결과가 동일해야 함
        assert orig_action == opt_action, f"액션 불일치: {orig_action} vs {opt_action}"
        assert orig_content == opt_content, f"콘텐츠 불일치: {orig_content} vs {opt_content}"

        # 예상값 검증
        assert opt_action == expected_action
        assert opt_content == expected_content


class TestExtractUid:
    """UID 추출 함수 테스트"""

    @pytest.mark.parametrize("uid,expected", [
        # 기본 케이스
        ("sample_123", 123),
        ("batch_0_sample_456", 456),
        # 숫자만
        ("789", 789),
        # 숫자 없음
        ("no_numbers_here", -1),
        # 빈 문자열
        ("", -1),
        # 여러 숫자 (마지막 숫자)
        ("item_1_sub_2_id_3", 3),
    ])
    def test_extract_uid_equivalence(self, uid, expected):
        """원본과 최적화 버전이 동일한 결과를 반환하는지 검증"""
        orig_result = extract_uid_original(uid)
        opt_result = extract_uid_optimized(uid)

        assert orig_result == opt_result, f"결과 불일치: {orig_result} vs {opt_result}"
        assert opt_result == expected


class TestPerformance:
    """성능 비교 테스트 (선택적)"""

    def test_regex_compilation_overhead(self):
        """정규식 컴파일 오버헤드 측정"""
        import time

        test_texts = [
            "<think>reasoning</think><search>query</search>" * 10
            for _ in range(1000)
        ]

        # 원본 (매번 컴파일)
        start = time.time()
        for text in test_texts:
            extract_tags_original(text)
        original_time = time.time() - start

        # 최적화 (사전 컴파일)
        start = time.time()
        for text in test_texts:
            extract_tags_optimized(text)
        optimized_time = time.time() - start

        # 최적화 버전이 더 빠르거나 같아야 함
        print(f"\n원본: {original_time:.4f}s, 최적화: {optimized_time:.4f}s")
        print(f"성능 향상: {((original_time - optimized_time) / original_time) * 100:.1f}%")

        # 최적화 버전이 원본보다 느리면 안 됨 (10% 마진 허용)
        assert optimized_time <= original_time * 1.1, \
            f"최적화 버전이 원본보다 느림: {optimized_time} > {original_time}"


# =============================================================================
# 실행
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
