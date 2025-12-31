"""
rm_phase2.py - Gemini 3 Flash 기반 LLM as Judge Reward Manager

=============================================================================
개요
=============================================================================
이 모듈은 GSPO Phase 2 학습을 위한 Reward Manager입니다.
Gemini 3 Flash를 Judge로 사용하여 Frozen Generator의 <answer> 내용만 평가합니다.

기존 추론 경로 전체 평가 버전은 rm_phase2_trajectory.py를 참조하세요.

설계 철학:
- 이미지 품질 평가: NDCG가 담당 (검색된 이미지 vs 정답 이미지 직접 비교)
- 답변 품질 평가: LLM Judge가 담당 (다양한 자연어 응답의 의미적 타당성)
- 추론 과정(<think>, <search>, <bbox>)은 평가하지 않음 (NDCG로 간접 평가)

=============================================================================
점수 계산 공식
=============================================================================
final_score = 0.8 * judge_score + 0.2 * ndcg_value

- judge_score: Gemini Judge의 연속 점수 (0.0 ~ 1.0)
  - <answer> 내용과 reference_answer의 의미적 일치 정도
  - 연속 점수로 분산 감소 및 dense signal 제공

- ndcg_value: 검색된 이미지와 정답 이미지의 NDCG (0.0 ~ 1.0)

=============================================================================
입/출력 인터페이스
=============================================================================
입력: DataProto
  - batch['prompts']: (batch_size, prompt_length)
  - batch['responses']: (batch_size, response_length)
  - batch['attention_mask']: (batch_size, total_length)
  - non_tensor_batch['extra_info']: dict
  - non_tensor_batch['reward_model']['ground_truth']: str
  - non_tensor_batch['retrievaled_images']: list

출력: (reward_tensor, metrics)
  - reward_tensor: (batch_size, response_length) - 마지막 토큰에만 점수 할당
  - metrics: dict - wandb 로깅용 평균 메트릭

=============================================================================
환경 설정
=============================================================================
필수 환경 변수:
  export GEMINI_API_KEY="your-api-key"

필수 패키지:
  pip install google-generativeai
"""

# =============================================================================
# Imports
# =============================================================================
from verl import DataProto
import torch
import json
import numpy as np
import os
import asyncio
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional

# Gemini SDK
import google.generativeai as genai

from verl.utils.unified_logger import get_unified_logger

_UNIFIED_LOGGER = get_unified_logger()

# =============================================================================
# LLM Judge 프롬프트
# =============================================================================
# 이 프롬프트는 Gemini에게 전달되어 <answer> 내용만 평가합니다.
# 기존 LLM as Judge 방식과 동일하게 이미지 없이 텍스트만 평가합니다.
#
# 평가 기준:
#   - 생성된 답변과 정답의 의미적 일치 여부 (semantic equivalence)
#   - 정확한 문자열 매칭이 아닌, 의미가 동일한지 판단

LLM_JUDGE_PROMPT = """You are an expert evaluation system for a question answering chatbot.

You are given the following information:
- the query
- a generated answer
- a reference answer

Your task is to evaluate the correctness of the generated answer.

## Query
{query}

## Reference Answer
{reference_answer}

## Generated Answer
{generated_answer}

## Evaluation Guidelines
- Evaluate if the generated answer is semantically equivalent to the reference answer
- Consider semantic equivalence, not just exact string match
- If the core meaning is the same, judge it as correct even if the wording differs
- Be lenient with minor formatting differences (e.g., "$4.5B" vs "4.5 billion dollars")

## Scoring
Provide a score from 0.0 to 1.0:
- 1.0: Semantically equivalent (correct)
- 0.5-0.9: Partially correct
- 0.0: Incorrect or unrelated
"""

# =============================================================================
# LLM Judge 응답 스키마 (Gemini Structured Output)
# =============================================================================
# Gemini SDK의 response_schema 기능을 사용하여 JSON 응답을 강제합니다.
# 참고: https://ai.google.dev/gemini-api/docs/structured-output

LLM_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "number",
            "description": "Semantic correctness score from 0.0 (completely wrong) to 1.0 (perfectly correct)"
        }
    },
    "required": ["score"]
}


# =============================================================================
# 헬퍼 함수: NDCG 계산
# =============================================================================
_NDCG_DEBUG_ENABLED = os.getenv("NDCG_DEBUG", "0").lower() in ("1", "true", "yes", "y")
_NDCG_DEBUG_LOG_ALL = os.getenv("NDCG_DEBUG_LOG_ALL", "0").lower() in ("1", "true", "yes", "y")
_NDCG_DEBUG_MAX_LINES = int(os.getenv("NDCG_DEBUG_MAX_LINES", "5000"))
_NDCG_DEBUG_PATH = os.getenv("NDCG_DEBUG_PATH", "./logs/ndcg_debug_{pid}.jsonl").format(pid=os.getpid())
_ndcg_debug_lock = threading.Lock()
_ndcg_debug_lines = 0

_FLASH_RM_LOG_ENABLED = os.getenv("FLASH_RM_LOG", "0").lower() in ("1", "true", "yes", "y")
_FLASH_RM_LOG_MAX_LINES = int(os.getenv("FLASH_RM_LOG_MAX_LINES", "500000"))
_FLASH_RM_LOG_PATH = os.getenv("FLASH_RM_LOG_PATH", "./logs/flash_rm_detail.jsonl")
_flash_rm_log_lock = threading.Lock()
_flash_rm_log_lines = 0


def _ndcg_debug_write(payload: dict) -> None:
    """Write one JSONL line for NDCG debugging (best-effort, bounded)."""
    global _ndcg_debug_lines
    if not _NDCG_DEBUG_ENABLED:
        return
    # Unified log (single-file)
    try:
        _UNIFIED_LOGGER.log_event({**payload, "event_type": "rm.ndcg.debug"})
    except Exception:
        pass
    # (legacy) JSONL 파일에 저장: unified 로깅 사용 시 중복되므로 기본 비활성화
    if _UNIFIED_LOGGER.enabled:
        return
    try:
        with _ndcg_debug_lock:
            if _ndcg_debug_lines >= _NDCG_DEBUG_MAX_LINES:
                return
            log_dir = os.path.dirname(_NDCG_DEBUG_PATH)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            payload = dict(payload)
            payload.setdefault("ts", time.time())
            payload.setdefault("pid", os.getpid())
            with open(_NDCG_DEBUG_PATH, "a") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            _ndcg_debug_lines += 1
    except Exception:
        # 로깅 실패는 학습을 멈추지 않게 한다.
        return


def _flash_rm_log_write(payload: dict) -> None:
    """Write one JSONL line for Flash RM (Gemini Judge) debugging (best-effort, bounded)."""
    global _flash_rm_log_lines
    if not _FLASH_RM_LOG_ENABLED:
        return
    # Unified log (single-file)
    try:
        _UNIFIED_LOGGER.log_event({**payload, "event_type": "rm.flash.detail"})
    except Exception:
        pass

    # (legacy) JSONL 파일에 저장: unified 로깅 사용 시 중복되므로 기본 비활성화
    if _UNIFIED_LOGGER.enabled:
        return

    try:
        with _flash_rm_log_lock:
            if _flash_rm_log_lines >= _FLASH_RM_LOG_MAX_LINES:
                return
            log_dir = os.path.dirname(_FLASH_RM_LOG_PATH)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            payload = dict(payload)
            payload.setdefault("timestamp", time.time())
            payload.setdefault("pid", os.getpid())
            with open(_FLASH_RM_LOG_PATH, "a") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            _flash_rm_log_lines += 1
    except Exception:
        return


def dcg(relevance_scores):
    """
    DCG (Discounted Cumulative Gain) 계산

    검색 결과의 품질을 측정하는 지표입니다.
    순위가 높을수록 (앞에 있을수록) 더 높은 가중치를 부여합니다.

    Args:
        relevance_scores: 각 문서의 관련성 점수 리스트 [1, 0, 1, 0, ...]

    Returns:
        DCG 값 (float)
    """
    dcg_value = 0.0
    for i, relevance in enumerate(relevance_scores, start=1):
        # log2(i+1)로 나누어 순위가 낮을수록 할인(discount) 적용
        dcg_value += (2 ** relevance - 1) / np.log2(i + 1)
    return dcg_value


def ndcg(sorted_docs, golden_answer_list):
    """
    NDCG (Normalized Discounted Cumulative Gain) 계산

    DCG를 이상적인 DCG(IDCG)로 정규화하여 0~1 사이 값으로 변환합니다.
    1에 가까울수록 검색 품질이 좋음을 의미합니다.

    Args:
        sorted_docs: 검색된 문서 리스트 (basename)
        golden_answer_list: 정답 문서 리스트 (basename)

    Returns:
        NDCG 값 (0.0 ~ 1.0)
    """
    if not golden_answer_list:
        _ndcg_debug_write({
            "event": "ndcg",
            "kind": "missing_golden",
            "sorted_docs_len": len(sorted_docs) if sorted_docs is not None else None,
            "golden_len": 0,
            "sorted_docs_head": (list(sorted_docs)[:10] if sorted_docs is not None else None),
        })
        return 0.0

    # Set 기반으로 히트 여부를 빠르게 계산 (로그에도 사용)
    try:
        golden_set = set(golden_answer_list)
        hits = sum(1 for doc in sorted_docs if doc in golden_set)
    except Exception:
        hits = None

    # 각 문서가 정답에 포함되면 1, 아니면 0
    relevance_scores = [1 if doc in golden_answer_list else 0 for doc in sorted_docs]

    # 실제 DCG 계산
    dcg_value = dcg(relevance_scores)

    # 이상적인 DCG (모든 정답이 앞에 배치된 경우)
    ideal_relevance_scores = [1] * len(golden_answer_list) + [0] * (len(sorted_docs) - len(golden_answer_list))
    idcg_value = dcg(ideal_relevance_scores)

    # 분모가 0인 경우 처리
    if idcg_value == 0:
        _ndcg_debug_write({
            "event": "ndcg",
            "kind": "idcg_zero",
            "sorted_docs_len": len(sorted_docs) if sorted_docs is not None else None,
            "golden_len": len(golden_answer_list) if golden_answer_list is not None else None,
            "hits": hits,
            "sorted_docs_head": (list(sorted_docs)[:10] if sorted_docs is not None else None),
            "golden_head": (list(golden_answer_list)[:10] if golden_answer_list is not None else None),
        })
        return 0.0

    ndcg_value = dcg_value / idcg_value

    # 디버그 로깅: ndcg==0인데 입력은 존재하는 케이스(스킴 불일치/매칭 실패)를 추적
    if _NDCG_DEBUG_LOG_ALL or (ndcg_value == 0.0 and sorted_docs and golden_answer_list):
        _ndcg_debug_write({
            "event": "ndcg",
            "kind": "computed",
            "sorted_docs_len": len(sorted_docs) if sorted_docs is not None else None,
            "golden_len": len(golden_answer_list) if golden_answer_list is not None else None,
            "hits": hits,
            "dcg": float(dcg_value),
            "idcg": float(idcg_value),
            "ndcg": float(ndcg_value),
            "sorted_docs_head": (list(sorted_docs)[:10] if sorted_docs is not None else None),
            "golden_head": (list(golden_answer_list)[:10] if golden_answer_list is not None else None),
        })

    return ndcg_value


def get_answer_from_predict_str(text):
    """
    모델 응답에서 <answer>...</answer> 태그 내용 추출

    Args:
        text: 모델의 전체 응답 문자열

    Returns:
        추출된 답변 문자열 또는 None
    """
    end_tag = '</answer>'
    start_tag = '<answer>'

    # 마지막 </answer> 태그 위치 찾기
    end_pos = text.rfind(end_tag)
    if end_pos == -1:
        return None

    # 해당 </answer> 앞의 <answer> 태그 찾기
    start_pos = text.rfind(start_tag, 0, end_pos)
    if start_pos == -1:
        return None

    # 태그 사이의 내용 추출
    start_pos += len(start_tag)
    return text[start_pos:end_pos]


# =============================================================================
# 스트리밍 Reward 데이터 클래스
# =============================================================================
@dataclass
class PromptRewardRequest:
    """프롬프트 단위 Reward 요청"""
    uid: str                        # 프롬프트 고유 ID
    sample_indices: List[int]       # 배치 내 인덱스 (n_agent개)
    samples_data: List[Dict]        # 각 샘플의 전처리된 데이터


@dataclass
class PromptRewardResult:
    """프롬프트 단위 Reward 결과"""
    uid: str                        # 프롬프트 고유 ID
    sample_indices: List[int]       # 배치 내 인덱스
    reward_scores: List[float]      # 각 샘플의 최종 점수
    vlm_results: List[Dict]         # 각 샘플의 VLM 평가 결과
    ndcg_values: List[float]        # 각 샘플의 NDCG 값


# =============================================================================
# RMManager 클래스
# =============================================================================
class RMManager:
    """
    Gemini 3 Flash 기반 LLM as Judge Reward Manager

    GSPO Phase 2 학습에서 Frozen Generator의 <answer> 내용을 평가하여 보상을 계산합니다.
    이미지 품질은 NDCG로 별도 평가하며, LLM Judge는 텍스트 답변의 의미적 정확성만 평가합니다.

    주요 기능:
    1. 모델 응답에서 <answer> 내용 추출
    2. Gemini를 통한 텍스트 기반 답변 정확성 평가 (True/False)
    3. NDCG 기반 검색 품질 평가
    4. 최종 보상 점수 계산: 0.8 * judge + 0.2 * ndcg

    Attributes:
        tokenizer: 토크나이저 (응답 디코딩용)
        log_path: JSONL 로그 파일 경로
        gemini: Gemini 모델 인스턴스
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int = 0,
        compute_score=None,
        log_path: str = "./logs/grpo_log.jsonl",
        gemini_model: str = "gemini-3-flash-preview",
        image_base_path: str = "./data/images",
        max_concurrent_requests: int = 50,
    ):
        """
        RMManager 초기화

        Args:
            tokenizer: 토크나이저 인스턴스
            num_examine: (미사용, 호환성 유지)
            compute_score: (미사용, 호환성 유지)
            log_path: 학습 로그 저장 경로 (JSONL 형식)
            gemini_model: Gemini 모델 이름
            image_base_path: 이미지 파일 기본 경로 (NDCG 계산용)
            max_concurrent_requests: 동시 API 호출 수 제한 (Gemini rate limit 대응)
        """
        self.tokenizer = tokenizer
        self.log_path = log_path
        self.image_base_path = image_base_path
        # Optional: rule-based format reward (custom_reward_function)
        # When enabled (RM_FORMAT_COEF>0), format is treated as a hard gate:
        #   if format_score == 0 -> final_score = 0
        #   else final_score = format_coef*format + judge_coef*judge + ndcg_coef*ndcg
        self.compute_score = compute_score

        # =========================================================================
        # [NEW] Gemini Judge 상세 로깅 설정
        # =========================================================================
        self.gemini_detail_log_path = log_path.replace('.jsonl', '_gemini_detail.jsonl')
        os.makedirs(os.path.dirname(self.gemini_detail_log_path) or '.', exist_ok=True)
        print(f"[RMManager] Gemini Judge 상세 로그: {self.gemini_detail_log_path}")

        # =========================================================================
        # 비동기 배치 처리 설정
        # =========================================================================
        # Gemini API rate limit 대응:
        # - 분당 60 RPM (기본)
        # - 동시 요청: ~10개 권장
        # 세마포어로 동시 요청 수를 제한하여 rate limit 초과 방지
        self.max_concurrent_requests = max_concurrent_requests

        # =========================================================================
        # [NEW] Reward Coefficients 설정 (환경 변수)
        # =========================================================================
        self.judge_coef = float(os.getenv("RM_JUDGE_COEF", "0.8"))
        self.ndcg_coef = float(os.getenv("RM_NDCG_COEF", "0.2"))
        self.format_coef = float(os.getenv("RM_FORMAT_COEF", "0.0"))
        print(
            f"[RMManager] Reward Coefficients: Judge={self.judge_coef}, "
            f"NDCG={self.ndcg_coef}, Format={self.format_coef}"
        )
        if self.format_coef > 0.0 and self.compute_score is None:
            raise ValueError(
                "RM_FORMAT_COEF > 0 requires a custom_reward_function (format checker), "
                "but compute_score is None. Check custom_reward_function.path/name."
            )

        # =========================================================================
        # Gemini SDK 초기화 (Structured Output 설정)
        # =========================================================================
        # 환경 변수에서 API 키 로드
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")

        genai.configure(api_key=api_key)

        # Structured Output 설정 (True/False 이진 판정)
        # response_mime_type: JSON 응답 강제
        # response_schema: is_correct (boolean) 반환
        self.generation_config = {
            "response_mime_type": "application/json",
            "response_schema": LLM_RESPONSE_SCHEMA
        }

        self.gemini = genai.GenerativeModel(
            model_name=gemini_model,
            generation_config=self.generation_config
        )
        # 워커별 이벤트 루프에서 안전하게 새 인스턴스를 만들기 위한 팩토리
        self._gemini_factory = lambda: genai.GenerativeModel(
            model_name=gemini_model,
            generation_config=self.generation_config
        )

        print(f"[RMManager] Gemini LLM Judge 초기화 완료: {gemini_model}")
        print(f"[RMManager] 평가 방식: <answer> 텍스트만 평가 (이미지 없음)")
        print(f"[RMManager] 점수 형식: 연속 점수 (0.0 ~ 1.0)")
        print(f"[RMManager] 비동기 배치 처리: 최대 {max_concurrent_requests}개 동시 요청")
        print(f"[RMManager] 로그 경로: {log_path}")
        # 콘솔에 Gemini 프롬프트/응답을 모두 출력할지 여부 (디버그용)
        # 0: quiet, 1: 요약(길이/스코어), 2: 전체 프롬프트/RAW 응답
        _gemini_verbose = os.environ.get("GEMINI_DEBUG_VERBOSE", "0")
        try:
            self.verbose_gemini_level = int(_gemini_verbose)
        except ValueError:
            self.verbose_gemini_level = 1 if str(_gemini_verbose).lower() in ("1", "true", "t", "yes") else 0

        # =========================================================================
        # 스트리밍 모드 설정
        # =========================================================================
        # 스트리밍 모드: 프롬프트 완료 시 즉시 Reward 계산 시작
        # 배치 모드: 모든 Generation 완료 후 일괄 Reward 계산 (기존 방식)
        self._streaming_mode: bool = False
        self._request_queue: Optional[queue.Queue] = None
        self._result_dict: Dict[str, PromptRewardResult] = {}
        self._result_lock: threading.Lock = threading.Lock()
        self._workers: List[threading.Thread] = []
        self._shutdown_event: threading.Event = threading.Event()
        self._num_worker_threads: int = 4

    def __call__(self, data: DataProto):
        """
        보상 점수 계산 메인 함수

        전체 흐름:
        1. 단일 루프로 전처리 (디코딩, 메타데이터 추출)
        2. Gemini VLM Judge 호출
        3. 점수 계산 및 텐서 할당
        4. JSONL 로그 저장

        Args:
            data: DataProto 객체 (배치 데이터)

        Returns:
            tuple: (reward_tensor, metrics)
                - reward_tensor: (batch_size, response_length) 형태의 보상 텐서
                - metrics: wandb 로깅용 평균 메트릭 딕셔너리
        """
        # =========================================================================
        # 1. reward_tensor 초기화
        # =========================================================================
        # response와 동일한 shape의 zero 텐서 생성
        # 마지막 유효 토큰 위치에만 점수가 할당됨
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # =========================================================================
        # 2. 단일 루프: 전처리 (최적화 포인트 - 중복 순회 제거)
        # =========================================================================
        # 기존 구현은 2번 순회했지만, 여기서는 1번만 순회하며 모든 정보 추출
        preprocessed = []

        def _compute_format_score(
            data_source: str,
            solution_str: str,
            ground_truth: str,
            extra_info: dict,
        ) -> tuple[float, str | None]:
            if self.format_coef <= 0.0:
                return 1.0, None
            if self.compute_score is None:
                return 1.0, None
            try:
                out = self.compute_score(
                    data_source=data_source,
                    solution_str=solution_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
                # Backward-compat: some compute_score returns float, others return (score, reason)
                if isinstance(out, tuple) and len(out) >= 1:
                    score = float(out[0]) if out[0] is not None else 0.0
                    reason = out[1] if len(out) >= 2 else None
                    return score, reason
                return float(out) if out is not None else 0.0, None
            except Exception as e:
                # Fail-closed: if format checker errors, treat as fail (gate to 0)
                return 0.0, f"format_checker_error: {e}"

        for i in range(len(data)):
            data_item = data[i]

            # -----------------------------------------------------------------
            # 2.1 텐서 데이터 추출
            # -----------------------------------------------------------------
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch['responses']

            # attention_mask로 유효한 토큰 길이 계산 (1회만 계산)
            valid_response_length = int(
                data_item.batch['attention_mask'][prompt_length:].sum()
            )

            # -----------------------------------------------------------------
            # 2.2 응답 디코딩 (1회만 수행 - 최적화 포인트)
            # -----------------------------------------------------------------
            valid_response_ids = response_ids[:valid_response_length]
            response_str = self.tokenizer.decode(valid_response_ids)

            # -----------------------------------------------------------------
            # 2.3 메타데이터 추출
            # -----------------------------------------------------------------
            extra_info = data_item.non_tensor_batch.get('extra_info', {})
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch.get('data_source', 'unknown')
            uid_val = data_item.non_tensor_batch.get('uid', None)
            if uid_val is None:
                uid_val = data_item.non_tensor_batch.get('id', None)

            # -----------------------------------------------------------------
            # 2.4 이미지 경로 추출
            # -----------------------------------------------------------------
            # 검색된 이미지 (모델이 검색한 이미지)
            retrieved_images = data_item.non_tensor_batch.get('retrievaled_images', [])

            # NDCG 계산용 basename 추출
            retrieved_basenames = [
                os.path.basename(item.rstrip('/')).split(".jpg")[0]
                for item in retrieved_images
            ]

            # 정답 이미지 basename 구성
            reference_basenames = []
            if 'file_name' in extra_info and 'reference_page' in extra_info:
                reference_basenames = [
                    f'{extra_info["file_name"].split(".pdf")[0]}_{page}'
                    for page in extra_info["reference_page"].tolist()
                ]

            # -----------------------------------------------------------------
            # 2.5 <answer> 내용 추출 (Frozen Generator 출력)
            # -----------------------------------------------------------------
            generated_answer = get_answer_from_predict_str(response_str)
            if generated_answer is None:
                generated_answer = "Please Judge False"  # 기존 방식과 동일

            # -----------------------------------------------------------------
            # 2.6 format reward (optional, hard gate)
            # -----------------------------------------------------------------
            format_score, format_fail_reason = _compute_format_score(
                data_source=str(data_source),
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            format_pass = float(format_score) > 0.0

            # -----------------------------------------------------------------
            # 2.7 전처리 결과 저장
            # -----------------------------------------------------------------
            preprocessed.append({
                'index': i,
                'uid': str(uid_val) if uid_val is not None else None,
                'data_source': str(data_source),
                'response_str': response_str,           # 전체 추론 경로 (로깅용)
                'generated_answer': generated_answer,   # <answer> 내용만 (평가용)
                'valid_response_length': valid_response_length,
                'query': extra_info.get('question', ''),
                'reference_answer': ground_truth,
                'retrieved_basenames': retrieved_basenames,      # NDCG용
                'reference_basenames': reference_basenames,      # NDCG용
                'format_score': float(format_score),
                'format_pass': bool(format_pass),
                'format_fail_reason': format_fail_reason,
            })

        # =========================================================================
        # 3. Gemini LLM Judge 비동기 배치 호출
        # =========================================================================
        # 성능 비교:
        # - 순차 처리: 32개 샘플 × 2초 = 64초
        # - 비동기 처리 (10개 동시): 32 / 10 × 2초 ≈ 7초 (~9배 향상)
        # If format gating is enabled, skip Gemini calls for format-fail samples.
        if self.format_coef > 0.0:
            eval_items = [item for item in preprocessed if item.get('format_pass', False)]
            eval_results = self._run_async_batch_evaluate(eval_items) if eval_items else []
            judge_results = [self._default_result("skipped_due_to_format_fail")] * len(preprocessed)
            for item, res in zip(eval_items, eval_results):
                idx = int(item.get('index', -1))
                if 0 <= idx < len(judge_results):
                    judge_results[idx] = res
        else:
            judge_results = self._run_async_batch_evaluate(preprocessed)

        # =========================================================================
        # 4. 점수 계산 및 텐서 할당
        # =========================================================================
        metrics = {
            'ndcg': [],
            'judge_score': [],      # True=1.0, False=0.0
            'format_score': [],
            'format_pass': [],
            'final_score': [],
        }
        log_entries = []

        for i, item in enumerate(preprocessed):
            judge_result = judge_results[i]

            # -----------------------------------------------------------------
            # 4.1 NDCG 계산
            # -----------------------------------------------------------------
            ndcg_value = ndcg(item['retrieved_basenames'], item['reference_basenames'])

            # -----------------------------------------------------------------
            # 4.2 Judge 점수: 연속 점수 (0.0 ~ 1.0) 직접 사용
            # -----------------------------------------------------------------
            judge_score = judge_result.get('score', 0.0)

            # -----------------------------------------------------------------
            # 4.3 최종 점수 계산: 0.8 * Judge + 0.2 * NDCG
            # -----------------------------------------------------------------
            # Final score:
            # - default: judge_coef*judge + ndcg_coef*ndcg
            # - if RM_FORMAT_COEF>0: hard gate on format_score
            try:
                format_score = float(item.get('format_score', 1.0))
                format_pass = bool(item.get('format_pass', True))
                if self.format_coef > 0.0 and not format_pass:
                    final_score = 0.0
                else:
                    final_score = (
                        (self.format_coef * float(format_score) if self.format_coef > 0.0 else 0.0)
                        + self.judge_coef * float(judge_score)
                        + self.ndcg_coef * float(ndcg_value)
                    )
            except Exception:
                final_score = 0.0

            # -----------------------------------------------------------------
            # 4.4 reward_tensor에 할당
            # -----------------------------------------------------------------
            # 마지막 유효 토큰 위치에만 점수 할당 (verl 프레임워크 요구사항)
            reward_tensor[i, item['valid_response_length'] - 1] = final_score

            # -----------------------------------------------------------------
            # 4.5 메트릭 수집
            # -----------------------------------------------------------------
            metrics['ndcg'].append(ndcg_value)
            metrics['judge_score'].append(judge_score)
            metrics['format_score'].append(float(item.get('format_score', 1.0)))
            metrics['format_pass'].append(1.0 if item.get('format_pass', True) else 0.0)
            metrics['final_score'].append(final_score)

            # -----------------------------------------------------------------
            # 4.6 로그 엔트리 생성
            # -----------------------------------------------------------------
            log_entries.append({
                'uid': item.get('uid'),
                'data_source': item.get('data_source'),
                'query': item['query'],
                'generated_answer': item['generated_answer'],
                'reference_answer': item['reference_answer'],
                'format_score': float(item.get('format_score', 1.0)),
                'format_pass': bool(item.get('format_pass', True)),
                'format_fail_reason': item.get('format_fail_reason'),
                'judge_score': judge_score,
                'ndcg': ndcg_value,
                'final_score': final_score,
            })

        # =========================================================================
        # 5. JSONL 로그 저장 (append 모드 - 최적화 포인트)
        # =========================================================================
        # 기존: 전체 파일 읽기 → 수정 → 전체 쓰기 (O(n))
        # 개선: append 모드로 추가만 (O(1))
        # (legacy) JSONL 로그 저장: unified 로깅 사용 시 중복되므로 기본 비활성화
        if not _UNIFIED_LOGGER.enabled:
            with open(self.log_path, 'a') as f:
                for entry in log_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        else:
            # Unified log: batch-mode RM summary (per-sample)
            for entry in log_entries:
                try:
                    _UNIFIED_LOGGER.log_event({**entry, "event_type": "rm.batch.summary"})
                except Exception:
                    pass

        # =========================================================================
        # 6. 평균 메트릭 계산 및 반환
        # =========================================================================
        def safe_mean(lst):
            return sum(lst) / len(lst) if lst else 0.0

        avg_metrics = {
            'reward/ndcg_mean': safe_mean(metrics['ndcg']),
            'reward/judge_score_mean': safe_mean(metrics['judge_score']),
            'reward/format_score_mean': safe_mean(metrics['format_score']),
            'reward/format_pass_rate': safe_mean(metrics['format_pass']),
            'reward/final_score_mean': safe_mean(metrics['final_score']),
        }

        print(f"[RMManager] 배치 처리 완료: {len(preprocessed)}개 샘플")
        print(f"[RMManager] 평균 점수 - NDCG: {avg_metrics['reward/ndcg_mean']:.4f}, "
              f"Judge: {avg_metrics['reward/judge_score_mean']:.4f}, "
              f"Final: {avg_metrics['reward/final_score_mean']:.4f}")

        return reward_tensor, avg_metrics

    # =========================================================================
    # 비동기 배치 처리 메서드들
    # =========================================================================
    # GRPO 학습 특성상 n_generation 개수만큼 한꺼번에 보상 계산이 필요
    # 순차 처리 시 병목 발생 → 비동기 병렬 처리로 ~9배 성능 향상
    #
    # 구조:
    # _run_async_batch_evaluate (동기 진입점)
    #   └─ _async_batch_vlm_evaluate (비동기 배치 처리)
    #       └─ _async_evaluate_single (개별 샘플 평가, 세마포어 제어)
    #           └─ _prepare_vlm_input (입력 준비)

    def _run_async_batch_evaluate(self, preprocessed: list) -> list:
        """
        동기 컨텍스트에서 비동기 배치 평가 실행

        GRPO 학습 루프는 동기 컨텍스트이므로, 비동기 코드를 실행하기 위한
        진입점 역할을 합니다.

        Args:
            preprocessed: 전처리된 데이터 리스트

        Returns:
            각 샘플의 VLM 평가 결과 리스트
        """
        try:
            # 이미 이벤트 루프가 실행 중인지 확인
            loop = asyncio.get_running_loop()
            # 이미 async 컨텍스트인 경우 (드문 케이스)
            # ThreadPoolExecutor로 새 스레드에서 실행
            with ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self._async_batch_vlm_evaluate(preprocessed)
                )
                return future.result()
        except RuntimeError:
            # 일반적인 동기 컨텍스트 (대부분의 경우)
            return asyncio.run(self._async_batch_vlm_evaluate(preprocessed))

    async def _async_batch_vlm_evaluate(self, preprocessed: list) -> list:
        """
        비동기 배치 VLM 평가

        asyncio.gather를 사용하여 모든 샘플을 병렬로 평가합니다.
        세마포어로 동시 요청 수를 제한하여 rate limit을 준수합니다.

        Args:
            preprocessed: 전처리된 데이터 리스트

        Returns:
            각 샘플의 VLM 평가 결과 리스트 (입력 순서 유지)
        """
        # 세마포어 생성 (동시 요청 수 제한)
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        # 모든 샘플에 대해 비동기 태스크 생성
        tasks = [
            self._async_evaluate_single(item, idx, semaphore)
            for idx, item in enumerate(preprocessed)
        ]

        # 병렬 실행 (return_exceptions=True로 개별 실패가 전체를 중단시키지 않음)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Exception 결과를 기본값으로 변환
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"[RMManager] VLM 평가 실패 (샘플 {i}): {result}")
                processed_results.append(self._default_result(str(result)))
            else:
                processed_results.append(result)

        return processed_results

    async def _async_evaluate_single(
        self,
        item: dict,
        idx: int,
        semaphore: asyncio.Semaphore,
        max_retries: int = 3,
        base_delay: float = 1.0
    ) -> dict:
        """
        단일 샘플 비동기 평가 (세마포어로 동시성 제어 + 재시도 로직)

        세마포어를 사용하여 동시 API 호출 수를 제한합니다.
        이는 Gemini API rate limit (분당 60 RPM)을 준수하기 위함입니다.

        재시도 로직:
        - 최대 max_retries번 재시도 (기본 3번)
        - 지수 백오프: 1초 → 2초 → 4초 (rate limit 대응)
        - 모든 재시도 실패 시 기본값 반환

        Args:
            item: 전처리된 샘플 데이터
            idx: 샘플 인덱스 (로깅용)
            semaphore: 동시성 제어용 세마포어
            max_retries: 최대 재시도 횟수 (기본 3)
            base_delay: 재시도 기본 대기 시간 (초)

        Returns:
            LLM Judge 평가 결과 딕셔너리 (is_correct: bool)
        """
        async with semaphore:
            # 1. LLM 입력 준비 (텍스트 프롬프트만)
            prompt = self._prepare_llm_input(item)

            # 2. Gemini API 호출 (재시도 대상)
            last_error = None
            raw_response_text = None

            for attempt in range(max_retries):
                try:
                    # 텍스트만 전송 (이미지 없음)
                    response = await self.gemini.generate_content_async(prompt)
                    raw_response_text = response.text
                    parsed_result = self._parse_llm_response(raw_response_text)

                    # =========================================================
                    # [NEW] Gemini Judge 상세 로깅 (성공 시)
                    # =========================================================
                    self._log_gemini_detail(
                        idx=idx,
                        query=item.get('query', ''),
                        generated_answer=item.get('generated_answer', ''),
                        reference_answer=item.get('reference_answer', ''),
                        gemini_prompt=prompt,
                        gemini_response=raw_response_text,
                        parsed_score=parsed_result.get('score', 0.0),
                        error=None
                    )

                    return parsed_result

                except Exception as e:
                    last_error = e
                    # 마지막 시도가 아니면 재시도
                    if attempt < max_retries - 1:
                        # 지수 백오프: 1초 → 2초 → 4초
                        delay = base_delay * (2 ** attempt)
                        print(f"[RMManager] 샘플 {idx} 재시도 {attempt + 1}/{max_retries} "
                              f"({delay:.1f}초 후): {e}")
                        await asyncio.sleep(delay)
                    else:
                        # 모든 재시도 실패
                        print(f"[RMManager] 샘플 {idx} 최종 실패 "
                              f"({max_retries}번 시도): {e}")

            # 모든 재시도 실패 시 상세 로깅 후 기본값 반환
            self._log_gemini_detail(
                idx=idx,
                query=item.get('query', ''),
                generated_answer=item.get('generated_answer', ''),
                reference_answer=item.get('reference_answer', ''),
                gemini_prompt=prompt,
                gemini_response=raw_response_text,
                parsed_score=0.0,
                error=str(last_error)
            )
            return self._default_result(f"max_retries_exceeded: {last_error}")

    def _log_gemini_detail(
        self,
        idx: int,
        query: str,
        generated_answer: str,
        reference_answer: str,
        gemini_prompt: str,
        gemini_response: Optional[str],
        parsed_score: float,
        error: Optional[str]
    ):
        """
        [NEW] Gemini Judge 상세 로깅

        입력 프롬프트와 출력 응답을 JSONL 파일과 콘솔에 기록합니다.
        """
        import datetime

        log_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'sample_idx': idx,
            'query': query,
            'generated_answer': generated_answer,
            'reference_answer': reference_answer,
            'gemini_prompt': gemini_prompt,
            'gemini_response': gemini_response,
            'parsed_score': parsed_score,
            'error': error
        }

        # Unified log (single-file)
        try:
            _UNIFIED_LOGGER.log_event({**log_entry, "event_type": "rm.gemini.detail"})
        except Exception:
            pass

        # 콘솔 출력 (간략 버전)
        status = "SUCCESS" if error is None else f"ERROR: {error}"
        print(f"\n{'='*60}")
        print(f"[Gemini Judge] Sample {idx} | Score: {parsed_score:.2f} | {status}")
        print(f"  Query: {query[:100]}{'...' if len(query) > 100 else ''}")
        print(f"  Generated: {generated_answer[:100]}{'...' if len(generated_answer) > 100 else ''}")
        print(f"  Reference: {reference_answer[:100]}{'...' if len(reference_answer) > 100 else ''}")
        print(f"  Gemini Response: {gemini_response[:200] if gemini_response else 'None'}{'...' if gemini_response and len(gemini_response) > 200 else ''}")
        print(f"{'='*60}\n")

        # (legacy) JSONL 파일에 저장: unified 로깅 사용 시 중복되므로 기본 비활성화
        if not _UNIFIED_LOGGER.enabled:
            try:
                with open(self.gemini_detail_log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"[RMManager] Gemini 상세 로그 저장 실패: {e}")

    def _prepare_llm_input(self, item: dict) -> str:
        """
        LLM Judge 입력 준비 (텍스트 프롬프트만)

        이미지 없이 텍스트만 사용하여 <answer> 내용의 정확성을 평가합니다.
        기존 LLM as Judge 방식과 동일합니다.

        Args:
            item: 전처리된 샘플 데이터

        Returns:
            프롬프트 문자열
        """
        prompt = LLM_JUDGE_PROMPT.format(
            query=item['query'],
            generated_answer=item['generated_answer'],  # <answer> 내용만
            reference_answer=item['reference_answer'],
        )

        return prompt

    def _default_result(self, error: str) -> dict:
        """
        기본 결과 (에러 시 반환)

        API 호출 실패 등의 경우에 반환되는 기본값입니다.
        학습 안정성을 위해 0.0점을 부여합니다.

        Args:
            error: 에러 메시지

        Returns:
            기본값이 설정된 결과 딕셔너리
        """
        return {
            'score': 0.0,
            'error': error
        }

    def _parse_llm_response(self, text: str) -> dict:
        """
        LLM Judge 응답 JSON 파싱

        Gemini Structured Output을 사용하므로 응답이 이미 JSON 형식입니다.
        response_mime_type="application/json" 설정으로 JSON 응답이 보장됩니다.

        Args:
            text: Gemini 응답 텍스트 (JSON 형식)

        Returns:
            파싱된 JSON 딕셔너리 또는 기본값
            - score: float (0.0 ~ 1.0)
        """
        default_response = {
            'score': 0.0,
            'parse_error': True
        }

        try:
            # Structured Output 사용 시 응답이 이미 JSON 형식
            result = json.loads(text)

            # score 추출 및 범위 클리핑 (0.0 ~ 1.0)
            raw_score = result.get('score', 0.0)
            score = max(0.0, min(1.0, float(raw_score)))

            return {
                'score': score,
            }

        except json.JSONDecodeError as e:
            print(f"[RMManager] JSON 파싱 실패: {e}")
            return default_response
        except Exception as e:
            print(f"[RMManager] LLM 응답 파싱 실패: {e}")
            return default_response

    def _build_reference_image_paths(self, file_name: str, reference_pages) -> list:
        """
        정답 이미지 경로 구성

        PDF 파일명과 페이지 번호를 바탕으로 이미지 파일 경로를 생성합니다.

        Args:
            file_name: PDF 파일명 (예: "document.pdf")
            reference_pages: 정답 페이지 번호 리스트 또는 numpy 배열

        Returns:
            정답 이미지 파일 경로 리스트
        """
        if not file_name:
            return []

        # numpy 배열인 경우 리스트로 변환
        if hasattr(reference_pages, 'tolist'):
            pages = reference_pages.tolist()
        elif isinstance(reference_pages, list):
            pages = reference_pages
        else:
            return []

        base_name = file_name.split(".pdf")[0]
        paths = []

        for page in pages:
            # 이미지 경로 패턴: {base_path}/{base_name}_{page}.jpg
            img_path = os.path.join(self.image_base_path, f"{base_name}_{page}.jpg")
            paths.append(img_path)

        return paths

    # =========================================================================
    # 스트리밍 Reward 계산 메서드들
    # =========================================================================
    # Producer-Consumer 패턴으로 Generation과 Reward 계산을 병렬화
    #
    # 흐름:
    # 1. start_streaming_mode(): Worker 스레드 시작
    # 2. submit_prompt(): 프롬프트 완료 시 Queue에 요청 추가
    # 3. _streaming_worker_loop(): Worker가 Queue에서 요청 처리
    # 4. wait_and_get_streaming_rewards(): 모든 결과 수집
    # 5. stop_streaming_mode(): Worker 종료

    def start_streaming_mode(self, num_worker_threads: int = 1):
        """
        스트리밍 모드 시작 - Generation 전에 호출

        Worker 스레드를 시작하여 프롬프트 완료 시 즉시 Reward 계산을 시작합니다.
        각 Worker는 자체 이벤트 루프를 가지고 비동기 API 호출을 처리합니다.

        Args:
            num_worker_threads: Worker 스레드 수 (기본 4개)
        """
        if self._streaming_mode:
            print("[RMManager] 스트리밍 모드가 이미 활성화되어 있습니다.")
            return

        if self.format_coef > 0.0:
            raise RuntimeError(
                "Format gating (RM_FORMAT_COEF>0) is not supported in streaming reward mode yet. "
                "Disable reward_model.streaming_reward.enable."
            )

        self._streaming_mode = True
        # Event loop 단일화: 워커 수를 1로 고정하고, 동시성은 세마포어로 제어
        self._num_worker_threads = 1
        self._shutdown_event.clear()
        self._request_queue = queue.Queue()
        self._result_dict.clear()
        self._workers.clear()

        # Worker 스레드 시작
        for i in range(self._num_worker_threads):
            worker = threading.Thread(
                target=self._streaming_worker_loop,
                name=f"StreamingRewardWorker-{i}",
                daemon=True
            )
            worker.start()
            self._workers.append(worker)

        print(f"[RMManager] 스트리밍 모드 시작: {num_worker_threads}개 워커")

    def submit_prompt(
        self,
        uid: str,
        sample_indices: List[int],
        samples_data: List[Dict]
    ):
        """
        프롬프트 완료 시 Reward 계산 요청 제출

        Generation에서 프롬프트의 모든 샘플(n_agent개)이 완료되면
        이 메서드를 호출하여 Reward 계산을 요청합니다.

        Args:
            uid: 프롬프트 고유 ID
            sample_indices: 배치 내 샘플 인덱스들 (n_agent개)
            samples_data: 각 샘플의 전처리된 데이터
        """
        if not self._streaming_mode:
            raise RuntimeError(
                "스트리밍 모드가 활성화되지 않았습니다. "
                "start_streaming_mode()를 먼저 호출하세요."
            )

        request = PromptRewardRequest(
            uid=uid,
            sample_indices=sample_indices,
            samples_data=samples_data
        )
        assert self._request_queue is not None, "Queue not initialized"

        # [DEBUG] 제출되는 요청 상세 로깅
        print(f"\n[DEBUG-RM] submit_prompt 호출됨: uid={uid}, indices={sample_indices}")
        for i, data in enumerate(samples_data[:2]):  # 처음 2개만
            print(f"  sample[{i}]: query={data.get('query', '')[:50]}...")
            print(f"    generated_answer={data.get('generated_answer', 'MISSING')[:50] if data.get('generated_answer') else 'EMPTY'}...")
            print(f"    reference_answer={data.get('reference_answer', '')[:50]}...")

        self._request_queue.put(request)

    def wait_and_get_streaming_rewards(
        self,
        total_prompts: int | None = None,
        timeout: float = 600.0
    ) -> Dict[str, PromptRewardResult]:
        """
        모든 Reward 완료 대기 후 결과 반환

        Generation 완료 후 호출하여 아직 처리 중인 Reward 계산을 기다립니다.
        대부분의 Reward는 Generation 중에 이미 완료되어 있을 것입니다.

        Args:
            total_prompts: 예상 총 프롬프트 수 (검증용)
            timeout: 최대 대기 시간 (초, 기본 10분)

        Returns:
            uid -> PromptRewardResult 딕셔너리
        """
        if not self._streaming_mode:
            raise RuntimeError("스트리밍 모드가 활성화되지 않았습니다.")

        # Queue가 빌 때까지 대기 (모든 요청 처리 완료)
        # NOTE: queue.Queue.join()은 timeout을 지원하지 않아, 무한 대기 방지를 위해
        # unfinished_tasks를 폴링하며 timeout을 구현한다.
        assert self._request_queue is not None, "Queue not initialized"
        q = self._request_queue
        start_t = time.monotonic()
        warned = False

        while True:
            unfinished = getattr(q, "unfinished_tasks", 0)
            if unfinished == 0:
                break

            elapsed = time.monotonic() - start_t
            if elapsed >= timeout:
                # 무한 대기 대신 현재까지의 결과라도 반환해 학습이 멈추지 않게 한다.
                alive_workers = [w.name for w in self._workers if w.is_alive()]
                with self._result_lock:
                    completed_so_far = len(self._result_dict)
                print(
                    "[RMManager] 스트리밍 Reward 대기 timeout: "
                    f"elapsed={elapsed:.1f}s, unfinished={unfinished}, "
                    f"qsize~={q.qsize()}, completed={completed_so_far}, "
                    f"alive_workers={alive_workers}"
                )
                break

            # 너무 자주 출력하지 않도록 10초마다 1번만 경고
            if not warned and elapsed >= 10.0:
                alive_workers = [w.name for w in self._workers if w.is_alive()]
                with self._result_lock:
                    completed_so_far = len(self._result_dict)
                print(
                    "[RMManager] 스트리밍 Reward 대기 중... "
                    f"elapsed={elapsed:.1f}s, unfinished={unfinished}, "
                    f"qsize~={q.qsize()}, completed={completed_so_far}, "
                    f"alive_workers={alive_workers}"
                )
                warned = True

            time.sleep(0.1)

        with self._result_lock:
            completed = len(self._result_dict)
            if total_prompts is None:
                print(f"[RMManager] 스트리밍 완료: {completed} 프롬프트")
            else:
                print(f"[RMManager] 스트리밍 완료: {completed}/{total_prompts} 프롬프트")
                if completed < total_prompts:
                    print(f"[RMManager] 경고: 일부 프롬프트 결과 누락 "
                          f"({total_prompts - completed}개)")

            return dict(self._result_dict)

    def stop_streaming_mode(self):
        """
        스트리밍 모드 종료 - Worker 스레드 정리

        모든 Worker 스레드에 종료 신호를 보내고, 정리가 완료될 때까지 대기합니다.
        """
        # 이벤트 루프/워크 플리핑으로 인한 gRPC 루프 충돌을 방지하기 위해
        # 워커와 이벤트 루프는 유지하고, 큐/결과만 정리한다.
        if not self._streaming_mode:
            return

        # 큐와 결과를 비워 다음 스텝을 위한 깨끗한 상태만 유지
        self._request_queue = queue.Queue()
        with self._result_lock:
            self._result_dict.clear()

        print("[RMManager] 스트리밍 모드 유지: 워커는 계속 실행, 큐/결과만 초기화")

    def _streaming_worker_loop(self):
        """
        Worker 스레드 메인 루프

        Queue에서 요청을 가져와 비동기로 VLM 평가를 수행합니다.
        각 Worker는 자체 asyncio 이벤트 루프를 가집니다.
        """
        # 각 워커가 자체 이벤트 루프 생성 (스레드별로 별도 루프 필요)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # 이벤트 루프마다 안전하게 생성된 Gemini 클라이언트 사용
        gemini_client = self._gemini_factory()

        worker_name = threading.current_thread().name

        try:
            while not self._shutdown_event.is_set():
                # Queue가 초기화되지 않은 경우 스킵
                if self._request_queue is None:
                    time.sleep(0.05)
                    continue

                try:
                    # 0.1초 타임아웃으로 종료 신호 확인 가능하게
                    request_queue = self._request_queue
                    request = request_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                try:
                    # 비동기 평가 실행
                    result = loop.run_until_complete(
                        self._evaluate_prompt_streaming(request, gemini_client)
                    )

                    with self._result_lock:
                        self._result_dict[request.uid] = result

                except Exception as e:
                    print(f"[{worker_name}] Error for uid={request.uid}: {e}")
                    # 에러 시 기본값 저장
                    with self._result_lock:
                        self._result_dict[request.uid] = \
                            self._create_default_streaming_result(request)

                finally:
                    # task_done() 호출로 join()이 완료 감지 가능
                    try:
                        request_queue.task_done()
                    except Exception as e:
                        # task_done 실패 시 worker 스레드가 죽어 join()이 영원히 블로킹되는 것을 방지
                        print(f"[{worker_name}] task_done 실패: {e}")

        finally:
            loop.close()

    async def _evaluate_prompt_streaming(
        self,
        request: PromptRewardRequest,
        gemini_client
    ) -> PromptRewardResult:
        """
        프롬프트의 모든 샘플을 비동기 평가 (LLM as Judge 방식)

        기존 _async_evaluate_single 로직을 재사용하되,
        프롬프트 단위로 결과를 묶어서 반환합니다.

        Args:
            request: 프롬프트 Reward 요청

        Returns:
            프롬프트 단위 Reward 결과
        """
        # 단일 이벤트 루프에서 최대 동시 요청 수를 직접 세마포어로 제어
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        async def evaluate_single(sample_data: Dict, local_idx: int):
            """단일 샘플 평가 (클로저로 semaphore 공유)"""
            async with semaphore:
                # LLM 입력 준비 (텍스트만, 이미지 없음)
                prompt = self._prepare_llm_input(sample_data)

                # [DEBUG] 입력 데이터 확인
                gen_ans = sample_data.get('generated_answer', '')
                ref_ans = sample_data.get('reference_answer', '')
                print(f"[DEBUG-RM] idx={local_idx} | gen_ans={gen_ans[:50] if gen_ans else 'EMPTY'}... | ref_ans={ref_ans[:50] if ref_ans else 'EMPTY'}...")
                if self.verbose_gemini_level >= 1:
                    print(f"[DEBUG-RM][GEMINI PROMPT idx={local_idx}] len={len(prompt)} chars")
                if self.verbose_gemini_level >= 2:
                    print(f"[DEBUG-RM][GEMINI PROMPT FULL idx={local_idx}]\\n{prompt}")

                response_text = None
                error_str = None
                t0 = time.perf_counter()
                try:
                    # 텍스트만 전송 (이미지 없음)
                    response = await gemini_client.generate_content_async(prompt)
                    response_text = response.text
                    judge_result = self._parse_llm_response(response_text)
                    print(f"[DEBUG-RM] idx={local_idx} | Gemini 응답: score={judge_result.get('score', 'N/A')}")
                    if self.verbose_gemini_level >= 2:
                        print(f"[DEBUG-RM][GEMINI RESPONSE RAW idx={local_idx}] raw={response_text}")
                except Exception as e:
                    print(f"[DEBUG-RM] idx={local_idx} | Gemini 오류: {e}")
                    error_str = str(e)
                    judge_result = self._default_result(error_str)
                latency_s = time.perf_counter() - t0

                # NDCG 계산
                ndcg_value = ndcg(
                    sample_data.get('retrieved_basenames', []),
                    sample_data.get('reference_basenames', [])
                )

                # NDCG 디버그: 호출 컨텍스트(uid/sample_idx)를 함께 남긴다 (조건부)
                if _NDCG_DEBUG_ENABLED:
                    try:
                        sample_idx = None
                        if 0 <= local_idx < len(request.sample_indices):
                            sample_idx = request.sample_indices[local_idx]
                        _ndcg_debug_write({
                            "event": "ndcg_context",
                            "uid": request.uid,
                            "local_idx": local_idx,
                            "sample_idx": sample_idx,
                            "retrieved_len": len(sample_data.get("retrieved_basenames", []) or []),
                            "golden_len": len(sample_data.get("reference_basenames", []) or []),
                            "ndcg": float(ndcg_value),
                            "retrieved_head": list(sample_data.get("retrieved_basenames", []) or [])[:10],
                            "golden_head": list(sample_data.get("reference_basenames", []) or [])[:10],
                        })
                    except Exception:
                        pass

                # Judge 점수: 연속 점수 (0.0 ~ 1.0) 직접 사용
                judge_score = judge_result.get('score', 0.0)

                # 최종 점수: 0.8 * Judge + 0.2 * NDCG
                # Final score: coef * judge + coef * ndcg (env: RM_JUDGE_COEF / RM_NDCG_COEF)
                try:
                    final_score = self.judge_coef * float(judge_score) + self.ndcg_coef * float(ndcg_value)
                except Exception:
                    final_score = 0.0
                
                # wandb 집계를 위해 final_score 키도 함께 저장
                judge_result['final_score_combined'] = final_score
                judge_result['final_score'] = final_score
                judge_result['judge_score'] = judge_score

                # Flash RM 상세 로깅 (Frozen generator detail과 유사한 JSONL)
                if _FLASH_RM_LOG_ENABLED:
                    try:
                        sample_idx = None
                        if 0 <= local_idx < len(request.sample_indices):
                            sample_idx = request.sample_indices[local_idx]
                        _flash_rm_log_write({
                            "uid": request.uid,
                            "local_idx": local_idx,
                            "sample_idx": sample_idx,
                            "query": sample_data.get("query", ""),
                            "generated_answer": sample_data.get("generated_answer", ""),
                            "reference_answer": sample_data.get("reference_answer", ""),
                            "retrieved_basenames": list(sample_data.get("retrieved_basenames", []) or []),
                            "reference_basenames": list(sample_data.get("reference_basenames", []) or []),
                            "judge_score": float(judge_score) if judge_score is not None else None,
                            "ndcg": float(ndcg_value),
                            "final_score": float(final_score),
                            "latency_s": float(latency_s),
                            "gemini_raw": (response_text if response_text is not None else None),
                            "error": error_str,
                        })
                    except Exception:
                        pass

                return local_idx, judge_result, ndcg_value

        # 모든 샘플에 대해 비동기 태스크 생성
        tasks = [
            evaluate_single(data, i)
            for i, data in enumerate(request.samples_data)
        ]

        # 병렬 실행
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 정리
        n = len(request.samples_data)
        reward_scores = [0.0] * n
        judge_results = [{}] * n
        ndcg_values = [0.0] * n

        for result in results:
            if isinstance(result, Exception):
                print(f"[RMManager] 샘플 평가 실패: {result}")
                continue
            if isinstance(result, tuple) and len(result) == 3:
                local_idx, judge, ndcg_val = result
                # final_score_combined는 0.8*Judge + 0.2*NDCG
                reward_scores[local_idx] = judge.get('final_score_combined', 0.0)
                judge_results[local_idx] = judge
                ndcg_values[local_idx] = ndcg_val

        return PromptRewardResult(
            uid=request.uid,
            sample_indices=request.sample_indices,
            reward_scores=reward_scores,
            vlm_results=judge_results,  # 필드명은 호환성을 위해 유지
            ndcg_values=ndcg_values
        )

    def _create_default_streaming_result(
        self,
        request: PromptRewardRequest
    ) -> PromptRewardResult:
        """
        에러 시 기본값 반환

        Worker에서 예외 발생 시 학습을 중단하지 않고
        0점 기본값을 반환하여 안정성을 유지합니다.

        Args:
            request: 원본 요청

        Returns:
            기본값이 설정된 결과
        """
        n = len(request.samples_data)
        return PromptRewardResult(
            uid=request.uid,
            sample_indices=request.sample_indices,
            reward_scores=[0.0] * n,
            vlm_results=[{'error': 'worker_failed'}] * n,
            ndcg_values=[0.0] * n
        )
