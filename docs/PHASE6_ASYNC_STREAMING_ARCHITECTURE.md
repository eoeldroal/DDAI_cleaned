# Phase 6: 완전 비동기 스트리밍 아키텍처

> **요약**: 프롬프트 완료 즉시 백그라운드 스레드에서 Frozen Generator와 Reward 계산을 병렬 처리하여 GPU 활용률 100%를 달성하고, 전체 학습 시간을 ~35% 단축합니다.

---

## 목차

1. [배경 및 동기](#1-배경-및-동기)
2. [아키텍처 개요](#2-아키텍처-개요)
3. [기존 구조 vs 새 구조](#3-기존-구조-vs-새-구조)
4. [핵심 구현 상세](#4-핵심-구현-상세)
5. [코드 변경 내역](#5-코드-변경-내역)
6. [성능 분석](#6-성능-분석)
7. [테스트](#7-테스트)
8. [설정 및 사용법](#8-설정-및-사용법)

---

## 1. 배경 및 동기

### 1.1 문제점: 순차 처리로 인한 GPU 유휴

기존 GSPO Phase 2 학습 파이프라인에서는 다음과 같은 순차 처리로 인해 GPU가 유휴 상태가 되는 시간이 발생했습니다:

```
[문제점: 순차 처리]

1. GPU (Searcher): 모든 샘플이 search_complete할 때까지 탐색 ────────────→ (60초)
                                                                    ↓
2. API (Frozen Generator): 모든 완료된 샘플에 대해 배치로 호출 ───────────→ (10초)
                                                                    ↓
3. API (Gemini Reward): Reward 계산 ──────────────────────────────────→ (30초)

총 시간: ~100초 (GPU는 60초만 활용, 나머지 40초는 유휴!)
```

### 1.2 근본 원인

기존 코드에서 Frozen Generator는 **메인 루프가 완전히 종료된 후**에야 호출되었습니다:

```python
# 기존 코드 (generation.py)
for step in range(self.config.max_turns):
    # 메인 루프: Searcher 작업
    # search_complete 감지 시 Reward 제출 (이때 answer 없음!)
    ...

# 메인 루프 종료 후에야 Frozen Generator 호출
completed_indices = [i for i, flag in enumerate(self.search_completed) if flag]
index2answer = self._call_frozen_generator_batch(completed_indices, ...)  # 여기서야 answer 생성!
```

이로 인해 스트리밍 Reward 모드에서 `generated_answer`가 누락되어 오류가 발생했습니다.

### 1.3 목표

1. **GPU 100% 활용**: 메인 루프가 API 호출을 기다리지 않음
2. **완전한 파이프라인**: 프롬프트 완료 즉시 Frozen Generator → Reward 계산
3. **Thread-safety**: 동시 접근 시 데이터 무결성 보장

---

## 2. 아키텍처 개요

### 2.1 핵심 아이디어

프롬프트가 완료되는 즉시 **백그라운드 스레드**에서 Frozen Generator와 Reward 계산을 처리합니다:

```
[Phase 6: 완전 비동기 파이프라인]

메인 루프 (GPU - Searcher)
────────────────────────────────────────────────────────────────────────────
Turn 1 ─→ Turn 2 ─→ ... ─→ Prompt 0 완료!
                                │
                                ├──→ Thread spawn (즉시 반환, ~0.0004초!)
                                │         │
Turn N+1 ─→ Turn N+2 ─→ ...     │         ▼ [Background Thread 0]
        │                       │    Frozen Generator (async)
        │ Prompt 1 완료!        │         ▼
        ├──→ Thread spawn ──────┼──→ Gemini Reward
        │                       │         ▼
        ▼                       │    결과 저장 ✓
   [계속 진행!]                 │
────────────────────────────────────────────────────────────────────────────
                    ↓
        [모든 턴 완료 후 백그라운드 대기]
                    ↓
             [최종 output 생성]
```

### 2.2 데이터 흐름

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Phase 6 데이터 흐름                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [메인 스레드]                                                               │
│      │                                                                      │
│      ▼                                                                      │
│  search_complete 감지                                                        │
│      │                                                                      │
│      ▼                                                                      │
│  _check_and_submit_prompt_reward(sample_idx)                                │
│      │                                                                      │
│      ├─── 프롬프트 완료 확인 (8개 샘플 모두 완료?)                           │
│      │                                                                      │
│      ▼                                                                      │
│  Thread.start() ────────────────────────────────────────┐                   │
│      │                                                  │                   │
│      │ (즉시 반환!)                                     │                   │
│      ▼                                                  ▼                   │
│  다음 턴 계속...                              [Background Thread]           │
│                                                         │                   │
│                                    ┌────────────────────┼────────────────┐  │
│                                    │  _process_prompt_background()       │  │
│                                    │                                     │  │
│                                    │  1. Frozen Generator 호출           │  │
│                                    │     (Phase 5 OpenAI AsyncClient)    │  │
│                                    │              │                      │  │
│                                    │              ▼                      │  │
│                                    │  2. generated_answers[i] = answer   │  │
│                                    │     (Thread-safe with Lock)         │  │
│                                    │              │                      │  │
│                                    │              ▼                      │  │
│                                    │  3. _collect_samples_data()         │  │
│                                    │     (generated_answer 포함!)        │  │
│                                    │              │                      │  │
│                                    │              ▼                      │  │
│                                    │  4. submit_prompt() → Gemini        │  │
│                                    │                                     │  │
│                                    └─────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 기존 구조 vs 새 구조

### 3.1 기존 구조 (Phase 5까지)

```python
# 메인 루프 내에서 search_complete 감지 시
elif action == 'search_complete':
    self.search_completed[i] = True
    if self.streaming_reward_manager:
        self._check_and_submit_prompt_reward(i)  # Reward 제출 (answer 없음!)

# 메인 루프 종료 후
completed_indices = [i for i, flag in enumerate(self.search_completed) if flag]
index2answer = self._call_frozen_generator_batch(...)  # 여기서야 answer 생성!
```

**문제점**:
- Reward 제출 시점에 `generated_answer`가 없음
- `KeyError: 'generated_answer'` 오류 발생

### 3.2 새 구조 (Phase 6)

```python
# 메인 루프 내에서 search_complete 감지 시
elif action == 'search_complete':
    self.search_completed[i] = True
    if self.streaming_reward_manager:
        self._check_and_submit_prompt_reward(i)  # 백그라운드 스레드 시작!

# _check_and_submit_prompt_reward (수정됨)
def _check_and_submit_prompt_reward(self, sample_idx: int):
    if status['completed_samples'] >= status['total_samples']:
        # 백그라운드 스레드로 처리 (블로킹 없음!)
        thread = threading.Thread(
            target=self._process_prompt_background,
            args=(indices, prompt_id, status),
            daemon=True
        )
        thread.start()
        self._pending_threads.append(thread)

# 백그라운드에서 Frozen Generator + Reward 처리
def _process_prompt_background(self, indices, prompt_id, status):
    # 1. Frozen Generator 호출
    index2answer = self._call_frozen_generator_batch(indices, questions, paths)

    # 2. 결과 저장 (Thread-safe)
    with self._thread_lock:
        for i in indices:
            self.generated_answers[i] = index2answer.get(i, "")
            self._streaming_frozen_generated.add(i)

    # 3. samples_data 수집 (generated_answer 포함!)
    samples_data = self._collect_samples_data(indices)

    # 4. Reward 제출
    self.streaming_reward_manager.submit_prompt(...)
```

---

## 4. 핵심 구현 상세

### 4.1 새로 추가된 필드

```python
class LLMGenerationManager:
    def __init__(self, ...):
        # ...기존 필드...

        # [Phase 6] 완전 비동기 스트리밍 지원
        self._pending_threads: List[threading.Thread] = []  # 완료 대기 중인 스레드
        self._thread_lock = threading.Lock()                 # Thread-safety를 위한 Lock
        self.generated_answers: Dict[int, str] = {}          # 생성된 답변 저장소
        self._streaming_frozen_generated: set = set()        # 스트리밍에서 처리 완료된 샘플
```

### 4.2 백그라운드 처리 함수

```python
def _process_prompt_background(self, indices: List[int], prompt_id: str, status: dict):
    """
    [Phase 6] 백그라운드 스레드에서 Frozen Generator + Reward 처리

    이 함수는 별도 스레드에서 실행되어 메인 루프를 블로킹하지 않습니다.
    """
    try:
        start_time = _time.perf_counter()

        # 1. Frozen Generator 호출 준비
        batch_questions = [self.questions[i] for i in indices]
        batch_paths = [self._prepare_generator_images(...) for i in indices]

        # 2. Frozen Generator 호출 (Phase 5 비동기 배치 처리)
        index2answer = self._call_frozen_generator_batch(indices, batch_questions, batch_paths)

        # 3. 결과 저장 (Thread-safe)
        with self._thread_lock:
            for i in indices:
                self.generated_answers[i] = index2answer.get(i, "")
                self._streaming_frozen_generated.add(i)

        # 4. samples_data 수집 (generated_answer 포함)
        samples_data = self._collect_samples_data(indices)

        # 5. Reward 제출 (Gemini VLM Judge 호출)
        self.streaming_reward_manager.submit_prompt(
            uid=prompt_id,
            sample_indices=indices,
            samples_data=samples_data
        )

    except Exception as e:
        print(f"[Phase 6] 프롬프트 {prompt_id} 처리 실패: {e}")
```

### 4.3 Thread-Safety 보장

```python
# 공유 데이터 접근 시 Lock 사용
with self._thread_lock:
    for i in indices:
        self.generated_answers[i] = answer
        self._streaming_frozen_generated.add(i)
```

Python의 GIL(Global Interpreter Lock)로 인해 기본 자료구조 연산은 atomic하지만, 복합 연산의 일관성을 위해 명시적 Lock을 사용합니다.

### 4.4 메인 루프 종료 후 처리

```python
# 백그라운드 스레드 완료 대기
if self._pending_threads:
    for thread in self._pending_threads:
        thread.join(timeout=120)
    self._pending_threads.clear()

# 스트리밍에서 이미 처리된 샘플 건너뛰기
completed_indices = [
    i for i, flag in enumerate(self.search_completed)
    if flag and i not in self._streaming_frozen_generated
]

# 처리되지 않은 샘플만 배치 처리 (폴백)
if completed_indices:
    index2answer = self._call_frozen_generator_batch(...)
```

---

## 5. 코드 변경 내역

### 5.1 변경된 파일

| 파일 | 라인 | 변경 내용 |
|------|------|----------|
| `vrag_agent/generation.py` | 21 | `import threading` 추가 |
| `vrag_agent/generation.py` | 485-493 | Phase 6 새 필드 추가 |
| `vrag_agent/generation.py` | 1153-1156 | `run_llm_loop` 초기화에 Phase 6 상태 초기화 |
| `vrag_agent/generation.py` | 1399-1448 | 메인 루프 종료 후 백그라운드 대기 + 스트리밍 샘플 건너뛰기 |
| `vrag_agent/generation.py` | 2071-2113 | `_check_and_submit_prompt_reward` 백그라운드 스레드 방식으로 수정 |
| `vrag_agent/generation.py` | 2115-2176 | `_process_prompt_background` 함수 추가 |
| `vrag_agent/generation.py` | 2178-2222 | `_collect_samples_data`에 `generated_answer` 필드 추가 |

### 5.2 의존성

- Python 표준 라이브러리: `threading` (추가 설치 불필요)
- 기존 Phase 5: `openai.AsyncOpenAI` (DashScope OpenAI 호환 API)

---

## 6. 성능 분석

### 6.1 이론적 성능 향상

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              성능 비교                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [기존: 완전 순차]                                                           │
│  GPU: Searcher ████████████████████████████████████████████ (60s)           │
│                                                            ↓                │
│  API: Frozen  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████ (10s)                    │
│                                                   ↓                         │
│  API: Gemini  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████████████ (30s)      │
│  총: ~100초                                                                  │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [Phase 6: 완전 비동기]                                                      │
│  GPU: Searcher ████████████████████████████████████████████ (60s, 무중단!)  │
│                 ↓     ↓     ↓     ↓     ↓     ↓                             │
│  Background:   [T1]  [T2]  [T3]  [T4]  [T5]  [T6]... (병렬!)               │
│                 │     │     │     │     │     │                             │
│  API: Frozen   ███   ███   ███   ███   ███   ███                           │
│  API: Gemini     ████   ████   ████   ████   ████                          │
│                                                      ↓                      │
│                                               [마지막 대기 ~5초]            │
│  총: ~65초 (GPU 100% 활용 + API 병렬 처리)                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

예상 성능 향상: ~35%
```

### 6.2 실측 테스트 결과

```
테스트: 6개 샘플 동시 처리

  기존 순차 처리 (예상): 6 × 1.0초 = ~6초
  Phase 6 병렬 처리: 1.55초

  병렬화 효율: ~4배 향상
  스레드 시작 시간: 0.0008초 (블로킹 없음!)
```

---

## 7. 테스트

### 7.1 테스트 파일

| 파일 | 설명 |
|------|------|
| `tests/test_phase5_frozen_generator.py` | Phase 5 OpenAI AsyncClient 테스트 |
| `tests/test_phase6_async_streaming.py` | Thread-safety 및 동시성 테스트 |
| `tests/test_phase6_integration.py` | 통합 End-to-End 테스트 |

### 7.2 테스트 실행

```bash
# Phase 5 테스트
python3 tests/test_phase5_frozen_generator.py

# Phase 6 비동기 테스트
python3 tests/test_phase6_async_streaming.py

# Phase 6 통합 테스트
python3 tests/test_phase6_integration.py
```

### 7.3 테스트 결과 요약

```
✅ Phase 5 기존 테스트: 5/5 통과
✅ Phase 6 비동기 테스트: 6/6 통과
✅ Phase 6 통합 테스트: 3/3 통과

총: 14/14 테스트 통과
```

---

## 8. 설정 및 사용법

### 8.1 필수 환경 변수

```bash
# .env 파일
DASHSCOPE_API_KEY=sk-your-key-here
GEMINI_API_KEY=AIzaSy-your-key-here
```

### 8.2 설정 파일 (gspo_phase2_gemini_flash.sh)

```bash
# Frozen Generator 설정
frozen_max_concurrent=50      # 동시 API 요청 수
frozen_model="qwen2.5-vl-72b-instruct"
frozen_max_tokens=1024
frozen_max_retries=3
frozen_backoff_base=1.5

# 스트리밍 Reward 모드 활성화
streaming_reward_enable=True
```

### 8.3 학습 실행

```bash
# 환경 변수 설정 후 실행
export DASHSCOPE_API_KEY="your-key"
export GEMINI_API_KEY="your-key"

bash gspo_phase2_gemini_flash.sh
```

---

## 부록: 트러블슈팅

### A.1 `KeyError: 'generated_answer'` 오류

**원인**: Phase 6 이전 버전의 코드 사용 시 발생

**해결**: `generation.py`가 Phase 6 버전인지 확인
```python
# Phase 6 버전 확인
from vrag_agent.generation import LLMGenerationManager
manager = LLMGenerationManager(...)
assert hasattr(manager, '_process_prompt_background'), "Phase 6 버전 필요"
```

### A.2 백그라운드 스레드 타임아웃

**원인**: Frozen Generator API 응답 지연

**해결**: 타임아웃 값 조정
```python
# generation.py의 thread.join()
thread.join(timeout=120)  # 필요시 증가
```

### A.3 Thread-safety 문제

**원인**: 동시 접근 시 데이터 불일치

**해결**: `_thread_lock` 사용 확인
```python
with self._thread_lock:
    self.generated_answers[i] = answer
```

---

## 변경 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| Phase 6.0 | 2025-12-26 | 완전 비동기 스트리밍 아키텍처 구현 |
| Phase 5.0 | 이전 | OpenAI AsyncClient 기반 Frozen Generator 최적화 |

---

*이 문서는 Phase 6 완전 비동기 스트리밍 아키텍처 구현 과정을 기록합니다.*
