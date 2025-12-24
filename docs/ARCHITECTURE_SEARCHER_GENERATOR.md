# DDAI 시스템 아키텍처: Searcher & Frozen Generator

> **목적**: 이 문서는 Searcher와 Frozen Generator의 역할, 기존/새 Reward 계산 방식의 차이를 명확히 정리합니다.
>
> **작성일**: 2024-12-24

---

## 목차

1. [개요](#1-개요)
2. [핵심 모델 역할](#2-핵심-모델-역할)
3. [전체 파이프라인 흐름](#3-전체-파이프라인-흐름)
4. [코드 맵](#4-코드-맵)
5. [기존 방식: LLM as Judge](#5-기존-방식-llm-as-judge)
6. [새 방식: VLM as Judge](#6-새-방식-vlm-as-judge)
7. [핵심 차이점 비교](#7-핵심-차이점-비교)
8. [설계 결정 및 주의사항](#8-설계-결정-및-주의사항)
9. [n_agent 응답 생성 및 Reward 계산 흐름](#9-n_agent-응답-생성-및-reward-계산-흐름)

---

## 1. 개요

DDAI 시스템은 **Visual RAG (Retrieval-Augmented Generation)** 시스템으로, 두 개의 분리된 모델이 협력하여 작동합니다:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DDAI Visual RAG Pipeline                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   [User Query]                                                          │
│        │                                                                │
│        ▼                                                                │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │  SEARCHER (학습 대상)                                         │     │
│   │  - Qwen2.5-VL-7B                                              │     │
│   │  - 역할: 질문에 답하기 위한 이미지 검색                         │     │
│   │  - 출력: <think>, <search>, <bbox>, <search_complete>         │     │
│   └──────────────────────────────────────────────────────────────┘     │
│        │                                                                │
│        │ 검색된 이미지                                                  │
│        ▼                                                                │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │  FROZEN GENERATOR (고정 모델)                                 │     │
│   │  - Qwen2.5-VL-72B-Instruct (DashScope API)                   │     │
│   │  - 역할: 검색된 이미지를 바탕으로 최종 답변 생성                 │     │
│   │  - 출력: <answer>...</answer>                                 │     │
│   └──────────────────────────────────────────────────────────────┘     │
│        │                                                                │
│        ▼                                                                │
│   [Final Response]                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**핵심 포인트**:
- **Searcher만 학습됨** (RL로 정책 업데이트)
- **Frozen Generator는 고정됨** (파라미터 업데이트 없음)
- Searcher의 검색 품질이 Frozen Generator의 답변 품질에 영향

---

## 2. 핵심 모델 역할

### 2.1 Searcher (학습 대상)

| 항목 | 내용 |
|------|------|
| **모델** | Qwen2.5-VL-7B |
| **역할** | 질문에 답하기 위한 관련 이미지 검색 |
| **학습 여부** | ✅ 학습됨 (GRPO로 정책 업데이트) |
| **출력 태그** | `<think>`, `<search>`, `<bbox>`, `<search_complete>` |

**Searcher가 생성하는 trajectory 예시**:
```
<think>사용자가 2023년 매출을 묻고 있다. 재무 보고서를 찾아야 한다.</think>
<search>2023 annual revenue financial report</search>
[Search Engine이 이미지 반환]
<bbox>[[0.1, 0.2, 0.8, 0.9]]</bbox>
<think>이 이미지에서 매출 데이터를 찾았다.</think>
<search_complete>
```

### 2.2 Frozen Generator (고정 모델)

| 항목 | 내용 |
|------|------|
| **모델** | Qwen2.5-VL-72B-Instruct (코드 기본값) |
| **호출 방식** | DashScope API (외부 API) |
| **역할** | Searcher가 찾은 이미지를 바탕으로 최종 답변 생성 |
| **학습 여부** | ❌ 고정됨 (파라미터 업데이트 없음) |
| **출력 태그** | `<answer>...</answer>` |
| **설정 위치** | `generation.py:160` - `frozen_model: str = "qwen2.5-vl-72b-instruct"` |

> **참고**: 코드 기본값은 `qwen2.5-vl-72b-instruct`이며, `config.frozen_model`로 변경 가능합니다.

**Frozen Generator가 생성하는 답변 예시**:
```
<answer>2023년 연간 매출은 $4.5B입니다.</answer>
```

### 2.3 왜 분리되어 있는가?

```
┌──────────────────────────────────────────────────────────────────────┐
│ 학습 목표: Searcher가 "좋은 이미지"를 찾도록 만들기                    │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  문제: "좋은 이미지"를 어떻게 정의할 것인가?                           │
│                                                                      │
│  해결책:                                                              │
│  1. NDCG: 검색된 이미지 vs 정답 이미지 직접 비교 (직접 평가)           │
│  2. Judge: Frozen Generator의 답변 품질 평가 (간접 평가)              │
│                                                                      │
│  논리:                                                                │
│  - Searcher가 좋은 이미지를 찾으면                                    │
│  - → Frozen Generator가 좋은 답변을 생성하고                          │
│  - → Judge가 높은 점수를 줌                                           │
│  - → Searcher가 보상을 받음                                           │
│                                                                      │
│  즉, Judge 점수는 간접적으로 Searcher의 검색 품질을 평가               │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. 전체 파이프라인 흐름

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          Generation Pipeline                               │
│                      (vrag_agent/generation.py)                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Step 1: Searcher 실행                                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  actor_rollout_wg.generate_sequences(active_batch)                   │ │
│  │  → Searcher 모델이 <think>, <search>, <bbox> 등 생성                 │ │
│  │  → 코드 위치: generation.py:617-618                                  │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                              │                                             │
│                              ▼                                             │
│  Step 2: Search Engine 호출                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  execute_predictions(predictions, ...)                               │ │
│  │  → <search> 쿼리를 Search Engine API로 전송                          │ │
│  │  → 이미지 검색 결과 수신                                              │ │
│  │  → 코드 위치: generation.py:1139-1177                                │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                              │                                             │
│                              ▼                                             │
│  Step 3: Frozen Generator 호출                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  _call_frozen_generator_batch(indices, questions, images)            │ │
│  │  → DashScope API로 Qwen2.5-VL-72B-Instruct 호출                      │ │
│  │  → <answer>...</answer> 생성                                         │ │
│  │  → 코드 위치: generation.py:955-991, 1339-1380                       │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                              │                                             │
│                              ▼                                             │
│  Step 4: 최종 Response 조합                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  response = Searcher trajectory + Frozen Generator <answer>          │ │
│  │                                                                      │ │
│  │  예시:                                                                │ │
│  │  <think>...</think>                     ← Searcher                   │ │
│  │  <search>...</search>                   ← Searcher                   │ │
│  │  <bbox>...</bbox>                       ← Searcher                   │ │
│  │  <search_complete>                      ← Searcher                   │ │
│  │  <answer>...</answer>                   ← Frozen Generator           │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                          Reward Calculation                                │
│                   (verl/workers/reward_manager/)                           │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌─────────────────────────┐     ┌─────────────────────────┐              │
│  │  기존 방식               │     │  새 방식                 │              │
│  │  rm_phase2_deprecated.py│     │  rm_phase2.py           │              │
│  │                         │     │                         │              │
│  │  • <answer>만 추출       │     │  • 전체 response 사용    │              │
│  │  • LLM Judge 평가       │     │  • VLM Judge 평가       │              │
│  │  • 0.2*judge + 0.8*ndcg │     │  • 0.8*vlm + 0.2*ndcg   │              │
│  └─────────────────────────┘     └─────────────────────────┘              │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 코드 맵

### 4.1 Searcher 실행

```python
# 파일: vrag_agent/generation.py
# 위치: 609-618 라인

def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
    """
    Wrapper for generation that handles multi-GPU padding requirements.
    """
    num_gpus = self.config.num_gpus
    if num_gpus <= 1:
        return self.actor_rollout_wg.generate_sequences(active_batch)  # ← Searcher 모델 호출!
```

### 4.2 Search Engine 호출

```python
# 파일: vrag_agent/generation.py
# 위치: 1139-1177 라인

def execute_predictions(self, predictions: List[str], uids: np.ndarray, ...):
    cur_actions, contents = self.postprocess_predictions(predictions)

    search_requests = []
    for i, (action, content) in enumerate(zip(cur_actions, contents)):
        if action == 'search':
            search_requests.append({
                "query": content,  # ← Searcher가 생성한 검색 쿼리
                "id": str(search_id),
                "request_idx": i
            })

    if do_search and len(search_requests) > 0:
        if getattr(self.config, 'async_search', True):
            results_map = self._async_search_batches(search_requests)  # ← Search Engine API 호출
```

### 4.3 Frozen Generator 호출

```python
# 파일: vrag_agent/generation.py
# 위치: 955-991 라인 (호출부)

# ===== generator added =====
gen_to_tokenize = [""] * len(self.retrievaled_images)

completed_indices = [i for i, flag in enumerate(self.search_completed) if flag]

if completed_indices:
    batch_questions = []
    batch_paths = []

    for i in completed_indices:
        q = self.questions[i]
        paths = self._prepare_generator_images(
            self.retrievaled_images[i],  # ← Searcher가 찾은 이미지
            self.cropped_images[i]
        )
        batch_questions.append(q)
        batch_paths.append(paths)

    frozen_monitor.start()
    index2answer = self._call_frozen_generator_batch(  # ← Frozen Generator 호출!
        completed_indices, batch_questions, batch_paths
    )
    frozen_monitor.stop()

    for i in completed_indices:
        ans = index2answer.get(i, "")
        if ans:
            gen_to_tokenize[i] = f"<answer>{ans}</answer>{self.tokenizer.eos_token}"  # ← <answer> 태그로 감싸기
```

```python
# 파일: vrag_agent/generation.py
# 위치: 1289-1336 라인 (실제 API 호출)

def _call_frozen_generator_single(self, question: str, image_paths: List[str]) -> Tuple[int, str]:
    if not _HAS_DASHSCOPE:
        return (0, "")

    sys_prompt = (
        "You are a visual QA generator. "
        "Use only the provided images and the user question. "
        "Return ONLY the final answer text without extra explanations."
    )

    # 이미지 파트 구성
    user_content = []
    if image_paths:
        for p in image_paths:
            part = _to_image_part(p)
            if part:
                user_content.append(part)
    user_content.append({"text": f"Question: {qtext}"})

    # DashScope API 호출 (Qwen2.5-VL-72B-Instruct)
    resp = _dashscope_call_with_fallback(
        model=self.config.frozen_model,  # ← "qwen2.5-vl-72b-instruct"
        messages=messages,
        max_tokens=int(getattr(self.config, "frozen_max_tokens", 256)),
    )
```

### 4.4 Frozen Generator에 전달되는 이미지

Searcher는 `<search>`와 `<bbox>` 두 가지 액션으로 이미지를 수집하며, **두 종류 모두** Frozen Generator에 전달됩니다.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    Searcher의 이미지 수집 과정                              │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Step 1: <search> 액션                                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  Searcher: <search>2023 revenue report</search>                      │ │
│  │                         ↓                                            │ │
│  │  Search Engine API 호출                                               │ │
│  │                         ↓                                            │ │
│  │  이미지 반환 → retrievaled_images[idx].append(image_path)            │ │
│  │  (generation.py:414-415)                                             │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  Step 2: <bbox> 액션 (선택적)                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  Searcher: <bbox>[[0.1, 0.2, 0.8, 0.9]]</bbox>                       │ │
│  │                         ↓                                            │ │
│  │  최근 검색 이미지를 bbox 좌표로 crop                                   │ │
│  │  (generation.py:373-380)                                             │ │
│  │                         ↓                                            │ │
│  │  crop_path = {crops_dir}/{uuid}.jpg                                  │ │
│  │  cropped_images[idx].append(crop_path)                               │ │
│  │  (generation.py:384-386)                                             │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  [반복] Searcher가 <search_complete>를 출력할 때까지 반복                   │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

#### 이미지 준비 함수 (generation.py:1275-1285)

```python
def _prepare_generator_images(self, originals: List[str], crops: List[str]) -> List[str]:
    # 존재하는 파일만, 중복 제거, 최대 장수 제한
    seen = set()
    out = []
    for p in (originals + crops):  # ← 원본 + crop 이미지 모두!
        if p and (p not in seen) and os.path.exists(p):
            seen.add(p)
            out.append(p)
        if len(out) >= self.config.generator_max_images:
            break
    return out
```

#### 예시: Searcher가 2번 검색 + 1번 crop한 경우

```
retrievaled_images = [
    "/data/images/financial_report_page1.jpg",   # 첫 번째 검색
    "/data/images/financial_report_page5.jpg",   # 두 번째 검색
]

cropped_images = [
    "/tmp/crops/abc123.jpg",   # 첫 번째 이미지의 bbox crop
]

↓ _prepare_generator_images 호출

Frozen Generator에 전달되는 이미지:
[
    "/data/images/financial_report_page1.jpg",   # 원본 1
    "/data/images/financial_report_page5.jpg",   # 원본 2
    "/tmp/crops/abc123.jpg",                     # crop 1
]
```

#### 이미지 종류 정리

| 이미지 종류 | 저장 위치 | 생성 시점 | Frozen Generator 전달 |
|------------|----------|----------|---------------------|
| **검색 원본** | `retrievaled_images` | `<search>` 액션 후 | ✅ 전달 |
| **Crop 이미지** | `cropped_images` | `<bbox>` 액션 후 | ✅ 전달 |

---

## 5. 기존 방식: LLM as Judge

### 5.1 개요

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    기존 방식 (rm_phase2_deprecated.py)                      │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  전체 Response:                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │ <think>...</think>                     ← Searcher (무시됨)           │ │
│  │ <search>...</search>                   ← Searcher (무시됨)           │ │
│  │ <bbox>...</bbox>                       ← Searcher (무시됨)           │ │
│  │ <search_complete>                      ← Searcher (무시됨)           │ │
│  │ ┌────────────────────────────────────────────────────────────────┐  │ │
│  │ │ <answer>2023년 매출은 $4.5B입니다.</answer>  ← 이것만 추출!     │  │ │
│  │ └────────────────────────────────────────────────────────────────┘  │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                              │                                             │
│                              ▼                                             │
│  LLM Judge (Qwen2.5-72B-Instruct via API)                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  입력:                                                               │ │
│  │  - Query: "2023년 매출이 얼마인가요?"                                 │ │
│  │  - Generated Answer: "2023년 매출은 $4.5B입니다." (Frozen Gen 출력)  │ │
│  │  - Reference Answer: "$4.5 billion"                                  │ │
│  │                                                                      │ │
│  │  출력: True / False (이진 판정)                                      │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                              │                                             │
│                              ▼                                             │
│  최종 점수: 0.2 * judge_score + 0.8 * ndcg_value                           │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 핵심 코드

#### 5.2.1 `<answer>` 내용만 추출

```python
# 파일: verl/workers/reward_manager/rm_phase2_deprecated.py
# 위치: 77-90 라인

def get_answer_from_predict_str(text):
    """
    전체 response에서 <answer>...</answer> 내용만 추출

    예시:
    입력: "<think>...</think><search>...</search><answer>$4.5B</answer>"
    출력: "$4.5B"
    """
    end_tag = '</answer>'
    start_tag = '<answer>'

    end_pos = text.rfind(end_tag)
    if end_pos == -1:
        return None  # </answer> 없으면 None

    start_pos = text.rfind(start_tag, 0, end_pos)
    if start_pos == -1:
        return None  # <answer> 없으면 None

    start_pos += len(start_tag)  # <answer> 태그 건너뛰기
    return text[start_pos:end_pos]  # ← 내용만 반환!
```

#### 5.2.2 LLM Judge용 데이터 준비

```python
# 파일: verl/workers/reward_manager/rm_phase2_deprecated.py
# 위치: 205-212 라인

for i in range(len(data)):
    data_item = data[i]
    # ...

    # 전체 response를 디코딩
    generated_answer = get_answer_from_predict_str(
        self.tokenizer.decode(valid_response_ids)
    )  # ← <answer> 내용만 추출!

    if generated_answer is None:
        generated_answer = 'Please Judge False'

    data_eval.append(dict(
        query = extra_info['question'],
        generated_answer = generated_answer,  # ← Frozen Generator가 생성한 답변!
        reference_answer = data_item.non_tensor_batch['reward_model']['ground_truth']
    ))
```

#### 5.2.3 LLM Judge API 호출

```python
# 파일: verl/workers/reward_manager/rm_phase2_deprecated.py
# 위치: 226-229 라인

print("=====================eval model start=====================")
response = requests.post(self.rm_url, json=request_data_to_be_eval)  # POST to http://0.0.0.0:8003/eval
eval_results = response.json()  # True → 1.0, False → 0.0
print("=====================eval model end=====================")
```

#### 5.2.4 LLM Judge 프롬프트

```python
# 파일: model_eval/model_eval.py
# 위치: 4-28 라인

DEFAULT_SYSTEM_TEMPLATE = """You are an expert evaluation system for a question answering chatbot.

You are given the following information:
- the query
- a generated answer        ← Frozen Generator의 <answer> 내용
- a reference answer

Your task is to evaluate the correctness of the generated answer.

## Query
{query}

## Reference Answer
{reference_answer}

## Generated Answer
{generated_answer}

Your response should be formatted as following:
<judge>True or False</judge>

If the generated answer is correct, please set "judge" to True. Otherwise, please set "judge" to False.
"""
```

#### 5.2.5 최종 점수 계산

```python
# 파일: verl/workers/reward_manager/rm_phase2_deprecated.py
# 위치: 294-299 라인

# NDCG 계산 (Searcher의 검색 품질 직접 평가)
retrievaled_images_basename_list = [
    os.path.basename(item.rstrip('/')).split(".jpg")[0]
    for item in data_item.non_tensor_batch['retrievaled_images']
]
reference_images_basename_list = [
    f'{extra_info["file_name"].split(".pdf")[0]}_{page}'
    for page in extra_info["reference_page"].tolist()
]
ndcg_value = ndcg(retrievaled_images_basename_list, reference_images_basename_list)

# LLM Judge 결과 (Frozen Generator의 답변 품질 평가)
model_eval_score = eval_results.pop(0)  # 1.0 or 0.0

# 최종 점수
final_score = 0.2 * model_eval_score + 0.8 * ndcg_value  # ← 기존 공식
```

---

## 6. 새 방식: VLM as Judge

### 6.1 개요

```
┌────────────────────────────────────────────────────────────────────────────┐
│                       새 방식 (rm_phase2.py)                               │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  전체 Response (전부 사용):                                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │ <think>...</think>                     ← Searcher (평가됨!)          │ │
│  │ <search>...</search>                   ← Searcher (평가됨!)          │ │
│  │ <bbox>...</bbox>                       ← Searcher (평가됨!)          │ │
│  │ <search_complete>                      ← Searcher (평가됨!)          │ │
│  │ <answer>...</answer>                   ← Frozen Generator (평가됨!)  │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                              │                                             │
│                              ▼                                             │
│  VLM Judge (Gemini 3 Flash)                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  입력:                                                               │ │
│  │  - Images: 검색된 이미지 + 정답 이미지                                │ │
│  │  - Query: "2023년 매출이 얼마인가요?"                                 │ │
│  │  - Agent's Response: 전체 response (Searcher + Frozen Generator)     │ │
│  │  - Reference Answer: "$4.5 billion"                                  │ │
│  │                                                                      │ │
│  │  출력: 0.0 ~ 1.0 연속 점수 (Structured Output)                       │ │
│  │  - answer_accuracy: 0.0 ~ 1.0                                        │ │
│  │  - visual_grounding: 0.0 ~ 1.0                                       │ │
│  │  - reasoning_consistency: 0.0 ~ 1.0                                  │ │
│  │  - final_score: 0.0 ~ 1.0                                            │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                              │                                             │
│                              ▼                                             │
│  최종 점수: 0.8 * vlm_score + 0.2 * ndcg_value                             │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 핵심 코드

#### 6.2.1 전체 Response 사용

```python
# 파일: verl/workers/reward_manager/rm_phase2.py
# 위치: 404-405 라인

# 응답 디코딩 (1회만 수행)
valid_response_ids = response_ids[:valid_response_length]
response_str = self.tokenizer.decode(valid_response_ids)  # ← 전체 response!
```

#### 6.2.2 VLM Judge용 데이터 준비

```python
# 파일: verl/workers/reward_manager/rm_phase2.py
# 위치: 442-452 라인

preprocessed.append({
    'index': i,
    'response_str': response_str,           # ← 전체 추론 경로 (think + search + bbox + answer)
    'valid_response_length': valid_response_length,
    'query': extra_info.get('question', ''),
    'reference_answer': ground_truth,
    'retrieved_images': retrieved_images,   # ← VLM에 전달할 검색 이미지
    'reference_image_paths': reference_image_paths,  # ← VLM에 전달할 정답 이미지
    'retrieved_basenames': retrieved_basenames,      # NDCG용
    'reference_basenames': reference_basenames,      # NDCG용
})
```

#### 6.2.3 VLM Judge 프롬프트

```python
# 파일: verl/workers/reward_manager/rm_phase2.py
# 위치: 88-120 라인

VLM_JUDGE_PROMPT = """You are an expert evaluator for a visual question answering agent.
Evaluate the agent's response based on the provided images and reference answer.

## Input
- Images: {image_description}
- Query: {query}
- Agent's Response:
{full_response}                    # ← 전체 response가 여기 들어감!
- Reference Answer: {reference_answer}

## Evaluation Criteria (in order of importance)

### 1. Answer Accuracy (Most Important)
Does the final answer correctly address the query?
- Compare with the reference answer
- Consider semantic equivalence, not just exact match
- Partial credit for partially correct answers

### 2. Visual Grounding
Are the claims in the response supported by the actual image content?
- Check if visual descriptions match what's in the images
- Penalize fabricated or hallucinated details

### 3. Reasoning Consistency
Is the reasoning process logically coherent?
- Does the conclusion follow from the observations?
- Are there any contradictions in the reasoning?

## Scoring Guidelines
- Prioritize answer correctness above all else
- A correct answer with minor reasoning flaws should score higher than incorrect answer with good reasoning
- Give partial credit when appropriate
"""
```

#### 6.2.4 최종 점수 계산

```python
# 파일: verl/workers/reward_manager/rm_phase2.py
# 위치: 476-488 라인

for i, item in enumerate(preprocessed):
    vlm_result = vlm_scores[i]

    # NDCG 계산 (기존과 동일)
    ndcg_value = ndcg(item['retrieved_basenames'], item['reference_basenames'])

    # VLM Judge 결과
    vlm_score = vlm_result.get('final_score', 0.0)  # 0.0 ~ 1.0 연속 점수

    # 최종 점수
    final_score = 0.8 * vlm_score + 0.2 * ndcg_value  # ← 새 공식 (비중 역전!)
```

---

## 7. 핵심 차이점 비교

### 7.1 비교 테이블

| 항목 | 기존 방식 (deprecated) | 새 방식 (rm_phase2.py) |
|------|------------------------|------------------------|
| **파일** | `rm_phase2_deprecated.py` | `rm_phase2.py` |
| **Judge 모델** | Qwen2.5-72B-Instruct (LLM) | Gemini 3 Flash (VLM) |
| **평가 입력** | `<answer>` 내용만 | 전체 response |
| **이미지 입력** | ❌ 없음 (텍스트만) | ✅ 검색 이미지 + 정답 이미지 |
| **Frozen Generator** | ✅ 명시적으로 `<answer>` 분리 추출 | ⚠️ 전체 response에 포함 |
| **점수 형식** | 이진 (True: 1.0 / False: 0.0) | 연속 (0.0 ~ 1.0) |
| **세부 점수** | ❌ 없음 | ✅ answer_accuracy, visual_grounding, reasoning_consistency |
| **점수 공식** | `0.2 * judge + 0.8 * ndcg` | `0.8 * vlm + 0.2 * ndcg` |
| **NDCG 비중** | 80% | 20% |
| **Judge 비중** | 20% | 80% |

### 7.2 점수 공식 비교

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           점수 공식 비교                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  기존 방식:                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  final_score = 0.2 × LLM_Judge + 0.8 × NDCG                         │   │
│  │                   │                   │                              │   │
│  │                   │                   └─ Searcher 직접 평가 (80%)    │   │
│  │                   └─ Frozen Generator 답변 평가 (20%)                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  새 방식:                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  final_score = 0.8 × VLM_Judge + 0.2 × NDCG                         │   │
│  │                   │                   │                              │   │
│  │                   │                   └─ Searcher 직접 평가 (20%)    │   │
│  │                   └─ 전체 response 평가 (80%)                        │   │
│  │                      (Searcher trajectory + Frozen Generator answer) │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.3 평가 범위 비교

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           평가 범위 비교                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  전체 Response 구조:                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  <think>사용자가 매출을 묻고 있다...</think>      ← Searcher         │   │
│  │  <search>2023 revenue report</search>            ← Searcher         │   │
│  │  <bbox>[[0.1, 0.2, 0.8, 0.9]]</bbox>             ← Searcher         │   │
│  │  <search_complete>                               ← Searcher         │   │
│  │  <answer>2023년 매출은 $4.5B입니다.</answer>     ← Frozen Generator │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  기존 방식 평가 범위:                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  <think>...</think>                              ← ❌ 무시           │   │
│  │  <search>...</search>                            ← ❌ 무시           │   │
│  │  <bbox>...</bbox>                                ← ❌ 무시           │   │
│  │  <search_complete>                               ← ❌ 무시           │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │ <answer>2023년 매출은 $4.5B입니다.</answer>   ← ✅ 평가됨     │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  새 방식 평가 범위:                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │ <think>...</think>                            ← ✅ 평가됨     │  │   │
│  │  │ <search>...</search>                          ← ✅ 평가됨     │  │   │
│  │  │ <bbox>...</bbox>                              ← ✅ 평가됨     │  │   │
│  │  │ <search_complete>                             ← ✅ 평가됨     │  │   │
│  │  │ <answer>...</answer>                          ← ✅ 평가됨     │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. 설계 결정 및 주의사항

### 8.1 새 방식의 설계 의도

새 방식(rm_phase2.py)은 다음을 목표로 설계되었습니다:

1. **추론 과정 전체 평가**: Searcher의 `<think>`, `<search>`, `<bbox>` 과정도 평가
2. **시각적 근거 확인**: VLM이 실제 이미지를 보고 추론이 이미지와 일치하는지 확인
3. **연속 점수**: 이진 판정(True/False)이 아닌 세분화된 연속 점수

### 8.2 주의사항: Frozen Generator 출력의 포함

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ⚠️ 주의사항                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  새 방식에서 response_str에 Frozen Generator의 <answer>가 포함됨            │
│                                                                             │
│  이것이 의미하는 바:                                                         │
│  - VLM Judge의 "answer_accuracy" 점수는 Frozen Generator의 <answer>를 평가  │
│  - 이 점수가 최종 보상에 80% 비중으로 영향                                   │
│  - Frozen Generator는 학습되지 않는 모델                                     │
│                                                                             │
│  잠재적 문제:                                                                │
│  - Searcher가 좋은 이미지를 찾아도 Frozen Generator가 잘못된 답변을 하면     │
│    → Searcher가 낮은 보상을 받음 (불공정할 수 있음)                          │
│                                                                             │
│  그러나 이것이 의도된 설계일 수도 있음:                                       │
│  - Searcher가 찾은 이미지가 충분히 좋으면                                    │
│    → Frozen Generator도 좋은 답변을 할 가능성이 높음                         │
│  - 결국 "좋은 이미지를 찾는 것"이 학습 목표이므로                            │
│    → 간접적인 평가가 타당할 수 있음                                          │
│                                                                             │
│  기존 방식과의 차이:                                                         │
│  - 기존: <answer>만 평가 (Frozen Generator 출력 명시적 분리)                 │
│  - 새로: 전체 response 평가 (Searcher traj + Frozen Generator answer 함께)  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.3 NDCG의 역할

NDCG는 **두 방식 모두에서 동일하게 작동**합니다:

```python
# Searcher가 찾은 이미지 vs 정답 이미지 직접 비교
ndcg_value = ndcg(retrieved_basenames, reference_basenames)
```

- **기존 방식**: NDCG가 80% 비중 → Searcher의 검색 품질이 주요 평가 기준
- **새 방식**: NDCG가 20% 비중 → VLM의 전체 평가가 주요 평가 기준

---

## 9. n_agent 응답 생성 및 Reward 계산 흐름

### 9.1 개요

GRPO/GSPO 학습에서는 각 프롬프트에 대해 **n_agent개의 응답**을 생성하여 그룹 내에서 상대적인 품질을 비교합니다.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        n_agent 응답 생성 흐름                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  원본 배치: [P1, P2, ..., P64]  (train_batch_size=64)                       │
│                    │                                                        │
│                    ▼  batch.repeat_deepcopy(repeat_times=8, interleave=True)│
│                                                                             │
│  복제된 배치 (512개):                                                        │
│  [P1_1, P2_1, ..., P64_1,  ← 첫 번째 응답 그룹                               │
│   P1_2, P2_2, ..., P64_2,  ← 두 번째 응답 그룹                               │
│   ...                                                                       │
│   P1_8, P2_8, ..., P64_8]  ← 여덟 번째 응답 그룹                             │
│                                                                             │
│  → 총 64 × 8 = 512개 샘플이 동시에 처리됨                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 전체 코드 흐름

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ray_trainer.py (Driver)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 1. 배치 복제 (L902)                                                     │ │
│  │    gen_batch = batch.repeat_deepcopy(                                  │ │
│  │        repeat_times=n_agent,  # 예: 8                                  │ │
│  │        interleave=True                                                 │ │
│  │    )                                                                   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│      │                                                                       │
│      ▼                                                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 2. Generation 단계 (L938-941)                                          │ │
│  │    final_gen_batch_output = generation_manager.run_llm_loop(gen_batch) │ │
│  │    → 512개 샘플 모두에 대해 Searcher + Frozen Generator 실행           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│      │                                                                       │
│      ▼                                                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 3. Log Probability 계산 (L977-978)                                     │ │
│  │    old_log_prob = actor_rollout_wg.compute_log_prob(batch)             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│      │                                                                       │
│      ▼                                                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 4. Reward 계산 (L1001-1027)                                            │ │
│  │    - Streaming Mode: 프롬프트 완료 시 바로 reward 계산 시작             │ │
│  │    - Batch Mode: 모든 Generation 완료 후 일괄 계산                     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│      │                                                                       │
│      ▼                                                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 5. Advantage 계산 (L1040-1044)                                         │ │
│  │    compute_advantage(batch, num_repeat=n_agent)                        │ │
│  │    → 같은 프롬프트의 8개 응답 간 상대 비교                              │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│      │                                                                       │
│      ▼                                                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 6. Actor 업데이트 (L1066)                                              │ │
│  │    actor_output = actor_rollout_wg.update_actor(batch)                 │ │
│  │    → Searcher 모델 학습                                                │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.3 Generation 내부 흐름 (generation.py)

**중요**: 개별 프롬프트가 완료되어도 **즉시 Frozen Generator로 넘기지 않습니다**.
배치 전체의 main loop가 종료된 후, 완료된 샘플들에 대해 일괄적으로 Frozen Generator를 호출합니다.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     generation.py: run_llm_loop()                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  입력: 512개 샘플 (64 프롬프트 × 8 응답)                                     │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Main Loop (L768-940)                                                  │  │
│  │                                                                       │  │
│  │  for step in range(max_turns):  # 예: 7턴                             │  │
│  │      │                                                                │  │
│  │      ├─► active_mask 체크: 완료되지 않은 샘플만 처리                  │  │
│  │      │   (L770: if not active_mask.sum(): break)                     │  │
│  │      │                                                                │  │
│  │      ├─► Searcher 추론 (L814-816)                                    │  │
│  │      │   gen_output = _generate_with_gpu_padding(rollings_active)    │  │
│  │      │   → 512개 샘플 중 active인 것만 동시 처리                     │  │
│  │      │                                                                │  │
│  │      ├─► 환경 실행 (L860)                                            │  │
│  │      │   next_obs, dones = execute_predictions(responses_str, ...)   │  │
│  │      │   → <search> 결과 반영, 이미지 수집                           │  │
│  │      │   → <search_complete> 생성 시 done=True                       │  │
│  │      │                                                                │  │
│  │      └─► active_mask 업데이트 (L867-868)                             │  │
│  │          curr_active_mask = [not done for done in dones]             │  │
│  │          active_mask = active_mask * curr_active_mask                │  │
│  │                                                                       │  │
│  │  [루프 종료]                                                          │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                              │                                               │
│                              ▼                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Frozen Generator 호출 (L958-974)                                      │  │
│  │                                                                       │  │
│  │  # 완료된 모든 샘플 수집                                               │  │
│  │  completed_indices = [i for i, flag in enumerate(search_completed)   │  │
│  │                       if flag]                                        │  │
│  │                                                                       │  │
│  │  if completed_indices:                                                │  │
│  │      # 완료된 샘플들의 질문과 이미지 수집                              │  │
│  │      for i in completed_indices:                                     │  │
│  │          q = self.questions[i]                                       │  │
│  │          paths = _prepare_generator_images(                          │  │
│  │              retrievaled_images[i], cropped_images[i]                │  │
│  │          )                                                           │  │
│  │          batch_questions.append(q)                                   │  │
│  │          batch_paths.append(paths)                                   │  │
│  │                                                                       │  │
│  │      # 72B Frozen Generator 일괄 호출                                 │  │
│  │      index2answer = _call_frozen_generator_batch(                    │  │
│  │          completed_indices, batch_questions, batch_paths             │  │
│  │      )                                                               │  │
│  │                                                                       │  │
│  │      # <answer>...</answer> 태그로 감싸서 response에 추가             │  │
│  │      for i in completed_indices:                                     │  │
│  │          ans = index2answer.get(i, "")                               │  │
│  │          gen_to_tokenize[i] = f"<answer>{ans}</answer>{eos_token}"   │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                              │                                               │
│                              ▼                                               │
│  [최종 출력 반환: 512개 샘플의 complete response]                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.4 Streaming Reward Mode (선택적)

Generation과 Reward 계산을 파이프라인으로 병렬 처리하여 약 13% 성능 향상을 달성합니다.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Streaming Reward Mode 동작 방식                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  일반 배치 모드:                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  [Generation] ──────────────────────────────────► [Reward 계산]     │   │
│  │  (모든 512개 샘플 완료)                           (512개 일괄 처리)  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  스트리밍 모드 (streaming_reward_enable=True):                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  [Generation]                                                       │   │
│  │      P1 완료 ──┐                                                    │   │
│  │      P2 완료 ──┼──► [Reward 워커 스레드]                            │   │
│  │      ...      ──┤    ├─ P1 Reward 계산 (병렬)                       │   │
│  │      P64 완료 ──┘    ├─ P2 Reward 계산 (병렬)                       │   │
│  │                      └─ ...                                         │   │
│  │                                                                     │   │
│  │  → 프롬프트 완료 시점에 바로 Reward 계산 시작                        │   │
│  │  → Generation이 진행되는 동안 Reward도 병렬로 계산                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 코드 위치

```python
# 파일: verl/trainer/ppo/ray_trainer.py
# 위치: L933-934 (스트리밍 모드 시작)

if self.streaming_reward_enabled:
    self.reward_fn.start_streaming_mode(num_worker_threads=4)

# 위치: L1006-1022 (스트리밍 결과 수집)

if self.streaming_reward_enabled:
    # 프롬프트 완료 시점에 이미 reward 계산이 시작됨
    reward_results = self.reward_fn.wait_and_get_streaming_rewards(
        total_prompts=num_prompts
    )
    # 결과를 reward_tensor로 변환
    reward_tensor, reward_metrics = self._convert_streaming_rewards_to_tensor(
        batch, reward_results
    )
else:
    # 기존 배치 모드: 여기서 일괄 계산
    reward_tensor, reward_metrics = self.reward_fn(batch)
```

### 9.5 핵심 정리

| 단계 | 처리 단위 | 설명 |
|------|----------|------|
| **배치 복제** | 배치 전체 | `repeat_deepcopy(n_agent=8)` → 64×8=512개 |
| **Main Loop** | 512개 동시 | active_mask로 미완료 샘플만 처리 |
| **Frozen Generator** | 완료된 샘플 일괄 | main loop 종료 후 한 번에 호출 |
| **Reward 계산** | 512개 일괄 (또는 스트리밍) | 배치 모드 또는 스트리밍 모드 |
| **Advantage** | 그룹 단위 | 같은 프롬프트의 8개 응답 비교 |
| **Actor 업데이트** | 배치 전체 | Searcher 모델 학습 |

---

## 부록: 파일 경로 요약

| 역할 | 파일 경로 | 핵심 함수/클래스 |
|------|-----------|-----------------|
| **Searcher 실행** | `vrag_agent/generation.py` | `_generate_with_gpu_padding()` (L609) |
| **Search Engine 호출** | `vrag_agent/generation.py` | `execute_predictions()` (L1139) |
| **Frozen Generator 호출** | `vrag_agent/generation.py` | `_call_frozen_generator_batch()` (L1339) |
| **기존 Reward 계산** | `verl/workers/reward_manager/rm_phase2_deprecated.py` | `RMManager.__call__()` |
| **새 Reward 계산** | `verl/workers/reward_manager/rm_phase2.py` | `RMManager.__call__()` |
| **LLM Judge API 서버** | `model_eval/model_eval_api.py` | `eval()` |
| **LLM Judge 프롬프트** | `model_eval/model_eval.py` | `DEFAULT_SYSTEM_TEMPLATE` |

---

*마지막 업데이트: 2024-12-24*
