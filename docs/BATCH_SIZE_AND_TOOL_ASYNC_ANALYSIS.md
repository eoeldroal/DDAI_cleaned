# VERL 배치 크기 구조와 Tool 비동기화 분석

> **핵심 발견**: `ppo_micro_batch_size_per_gpu` 설정은 Generation(Rollout)과 무관하며, Tool 비동기화는 별도로 구현해야 합니다.

---

## 1. 배경: 왜 이 분석이 필요했는가?

Phase 6 완전 비동기 스트리밍을 구현한 후, Tool 호출(search, bbox 등)이 여전히 **동기적 블로킹**을 일으킨다는 것을 발견했습니다.

**의문**: `ppo_micro_batch_size_per_gpu=2`를 1로 바꾸면 샘플별 독립 진행이 가능할까?

**결론**: **아니요!** 이 설정은 Generation과 완전히 별개입니다.

---

## 2. VERL 파이프라인의 두 가지 단계

VERL(Visual Reinforcement Learning) 프레임워크는 크게 두 단계로 동작합니다:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VERL 파이프라인 구조                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [단계 1] Rollout (Generation) - 응답 생성 + Tool 호출                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  배치 크기 = train_batch_size × n_agent                             │   │
│  │            = 64 × 8 = 512개 샘플                                    │   │
│  │                                                                     │   │
│  │  처리 내용:                                                         │   │
│  │    1. Actor 모델이 응답 생성 (search, bbox, search_complete 등)     │   │
│  │    2. Tool 호출 및 결과 처리                                        │   │
│  │    3. 다음 턴 입력 준비                                             │   │
│  │                                                                     │   │
│  │  ⚠️ 모든 512개 샘플이 한꺼번에 처리됨!                               │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│  [단계 2] PPO Update - Gradient 계산 및 모델 업데이트                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  ppo_mini_batch_size = 16                                           │   │
│  │  ppo_micro_batch_size_per_gpu = 2                                   │   │
│  │                                                                     │   │
│  │  처리 내용:                                                         │   │
│  │    1. 생성된 응답으로 policy gradient 계산                          │   │
│  │    2. Actor 모델 파라미터 업데이트                                  │   │
│  │                                                                     │   │
│  │  ✅ Generation과는 완전히 별개!                                      │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 설정값 상세 분석

### 3.1 Rollout (Generation) 관련 설정

| 설정 | 값 | 설명 |
|------|-----|------|
| `train_batch_size` | 64 | 원본 프롬프트 수 |
| `n_agent` | 8 | 프롬프트당 생성할 응답 수 |
| **실제 배치 크기** | **512** | 64 × 8 = 512개 샘플 동시 처리 |

### 3.2 PPO Update 관련 설정

| 설정 | 값 | 설명 |
|------|-----|------|
| `ppo_mini_batch_size` | 16 | PPO 미니배치 크기 |
| `ppo_micro_batch_size_per_gpu` | 2 | GPU당 마이크로배치 크기 |

### 3.3 핵심 코드

```python
# verl/trainer/ppo/ray_trainer.py Line 910
# Generation 전에 배치를 n_agent 배로 복제!
gen_batch = batch.repeat_deepcopy(
    repeat_times=self.config.actor_rollout_ref.rollout.n_agent,  # 8
    interleave=True
)
# 결과: 64 프롬프트 × 8 = 512개 샘플

# verl/workers/rollout/sglang_rollout/sglang_rollout.py Line 200-248
def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
    batch_size = idx.size(0)  # 512

    # 512개 샘플을 한꺼번에 SGLang 엔진에 전달!
    output = self.inference_engine.generate(
        input_ids=idx_list,  # 512개 전체!
        ...
    )
```

---

## 4. `ppo_micro_batch_size_per_gpu` 변경 시 영향

### 4.1 변경 시 (2 → 1)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              ppo_micro_batch_size_per_gpu = 2 → 1 변경 효과                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [변하는 것]                                                                │
│    ✓ PPO Update 시 GPU당 마이크로배치: 2 → 1                               │
│    ✓ PPO Update 시 메모리 사용량: ~50% 감소                                │
│    ✓ PPO Update 시 속도: 약간 느려질 수 있음                               │
│                                                                             │
│  [변하지 않는 것]                                                           │
│    ✗ Generation 배치 크기: 여전히 512                                      │
│    ✗ Tool 호출 방식: 여전히 512개 동시 처리                                │
│    ✗ 샘플별 독립 진행: 여전히 불가능                                       │
│    ✗ Search API 블로킹: 여전히 발생                                        │
│                                                                             │
│  ⚠️ 결론: Tool 비동기화와 완전히 무관!                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 왜 무관한가?

```
시간 흐름:

[단계 1: Rollout]
  Turn 0:
    [Generate] 512개 샘플 동시 생성 (배치)
    [Tool]     512개 샘플의 Tool 호출 (search 블로킹!)  ◀── 여기가 문제!
    [Update]   512개 샘플 상태 업데이트
  Turn 1:
    ...

[단계 2: PPO Update]  ◀── ppo_micro_batch_size_per_gpu는 여기서만 영향!
  [Gradient] 미니배치 단위로 gradient 계산
  [Update]   모델 파라미터 업데이트
```

**핵심**: PPO Update는 Rollout이 **완전히 끝난 후**에 실행됩니다!

---

## 5. 진짜 문제: Tool 호출 동기 블로킹

### 5.1 현재 구조

```python
# vrag_agent/generation.py의 execute_predictions()
def execute_predictions(self, predictions, uids, pad_token, active_mask, do_search=True):
    # 1. 모든 샘플의 action 파싱
    cur_actions, contents = self.postprocess_predictions(predictions)

    # 2. search 요청 수집
    search_requests = []
    for i, (action, content) in enumerate(zip(cur_actions, contents)):
        if action == 'search':
            search_requests.append({...})

    # 3. search API 호출 + 결과 대기 ◀── 블로킹!
    results_map = self._async_search_batches(search_requests)  # 모든 결과 대기!

    # 4. 그제서야 bbox, search_complete 처리
    for i, action in enumerate(cur_actions):
        if action == 'bbox': ...
        elif action == 'search_complete': ...
```

### 5.2 문제점

```
샘플 0: action = search     → API 호출 시작 ─┐
샘플 1: action = bbox       → 대기 중...     │ 모든 search 끝날 때까지
샘플 2: action = search     → API 호출 시작 ─┤ bbox도 처리 못함!
샘플 3: action = complete   → 대기 중...     │
...                                          │
샘플 511: action = search   → API 호출 시작 ─┘
                                              ↓
                            [모든 search 완료] (3-5초 후)
                                              ↓
                            그제서야 bbox, complete 처리
```

---

## 6. 해결 방안: Tool 비동기화

### 6.1 Generation 배치는 유지해야 하는 이유

```
배치 크기와 GPU 효율:

배치 512 (현재):
  GPU 사용률: 95%+
  512개 처리 시간: ~12초
  샘플당 시간: 0.02초

배치 1 (샘플별 독립):
  GPU 사용률: ~10%
  1개 처리 시간: ~2초
  512개 순차 처리: 1024초! (85배 느림!)

결론: Generation 배치 처리는 필수!
```

### 6.2 Tool 호출만 비동기화

```python
# 제안: Phase 7 Tool 비동기화
def execute_predictions(self, ...):
    # 1. search 비동기 시작 (대기 안 함!)
    search_futures = []
    for req in search_requests:
        future = self._search_executor.submit(self._search_single, req)
        search_futures.append((req['request_idx'], future))

    # 2. bbox, search_complete 즉시 처리 (search와 병렬!)
    for i, action in enumerate(cur_actions):
        if action == 'bbox':
            next_obs[i] = self._process_bbox(...)  # 즉시!
        elif action == 'search_complete':
            next_obs[i] = ''  # 즉시!

    # 3. search 결과 대기 (마지막에!)
    for idx, future in search_futures:
        next_obs[idx] = future.result()
```

### 6.3 기대 효과

```
현재 (순차):
  [search 시작] → [search 대기 4초] → [bbox 처리 0.1초]
  총: 4.1초

비동기화 후:
  [search 시작] → [bbox 처리 0.1초] → [search 대기 3.9초]
                       ↑
                 이 시간 동안 search API 진행 중!
  총: 4.0초 (bbox 시간 절약)

Turn 간 파이프라인 적용 시:
  Turn N: [Generate 12초] → [search 시작] → [bbox 처리]
  Turn N+1: [search 결과 수신*] → [Generate 12초] → ...
            * Generate 시간 >> Search 시간이므로 이미 완료!

  효과: Search 대기 시간이 Generate와 중첩! (~25% 개선)
```

---

## 7. 요약

| 항목 | 설명 |
|------|------|
| **문제 인식** | Tool 호출 시 모든 search 결과를 동기적으로 대기 |
| **잘못된 해결책** | `ppo_micro_batch_size_per_gpu` 변경 (PPO Update 전용, Generation 무관) |
| **올바른 해결책** | Tool 호출 비동기화 (search 즉시 시작, 결과 나중에 대기) |
| **구현 난이도** | 낮음 (execute_predictions 함수 수정) |
| **기대 효과** | GPU 활용률 55% → 72%, 전체 시간 ~25% 단축 |

---

## 8. 관련 문서

- [Phase 6 비동기 스트리밍 아키텍처](./PHASE6_ASYNC_STREAMING_ARCHITECTURE.md) - Frozen Generator + RM 비동기화
- [시스템 아키텍처](./ARCHITECTURE_SEARCHER_GENERATOR.md) - 전체 시스템 구조

---

## 9. 참고: 설정 파일 위치

```bash
# Generation 배치 설정
gspo_phase2_gemini_flash.sh:
  train_batch_size=64
  n_agent=8
  # 실제 배치 = 64 × 8 = 512

# PPO Update 배치 설정 (Generation과 무관!)
gspo_phase2_gemini_flash.sh:
  ppo_mini_batch_size=16
  ppo_micro_batch_size_per_gpu=2
```

---

*최종 업데이트: 2025-12-26*
*작성자: Claude Code (Phase 7 Tool 비동기화 분석)*
