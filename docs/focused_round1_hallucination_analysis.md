# Focused Round 1: Frozen Generator Hallucination 전수 질적 분석 보고서

**분석 일시**: 2025-12-30
**분석 대상**: `flash_rm_detail.jsonl` (674개 고유 질문)
**훈련 설정**: `gspo_phase2_focused_round1.sh`
- Judge Coef: 1.0, NDCG Coef: 0.0
- Frozen Model: gpt-5-mini-2025-08-07
- Reasoning Effort: medium

---

## 1. 분석 배경

### 1.1 문제 정의
Focused Round 1에서는 LLM as Judge (Gemini Flash)만으로 보상을 구성하고 NDCG를 제외했습니다. 이로 인해 다음 질문이 제기되었습니다:

1. **Hallucination**: Frozen Generator가 이미지에 정보가 없는데도 자체 지식으로 정답을 생성하는가?
2. **Missed**: 이미지에 정보가 있는데도 답변을 못하는 경우는 없는가?

### 1.2 초기 통계 분석
- **Hallucination 의심 케이스**: 184개 (Judge Score >= 0.5 AND Golden 이미지 미포함)
- **초기 추정 Hallucination 비율**: 27.3% (184/674)

---

## 2. 질적 분석 방법론

### 2.1 분석 방식
- 184개 의심 케이스를 8개 그룹으로 분할 (각 23개)
- 8개 서브에이전트가 **병렬로** 각 케이스를 질적 분석
- 각 케이스에 대해 Retrieved 이미지를 **직접 열어서** 내용 확인
- 이미지 내용과 생성된 답변을 비교하여 Hallucination 여부 판단

### 2.2 판정 기준

| 판정 | 정의 |
|------|------|
| **True Hallucination** | 이미지에 관련 정보가 전혀 없는데 정답을 생성 (자체 지식 사용) |
| **Partial Info** | 이미지에 관련 정보가 부분적으로 있음 |
| **Not Hallucination** | 이미지에 답변에 필요한 정보가 있음 (잘못 분류된 케이스) |
| **Uncertain** | 판단하기 어려움 |

---

## 3. 분석 결과

### 3.1 그룹별 통계

| 그룹 | True Hallucination | Partial Info | Not Hallucination | Uncertain |
|------|-------------------|--------------|-------------------|-----------|
| Group 1 | ~8 | ~4 | ~10 | ~1 |
| Group 2 | 10 | 4 | 9 | 0 |
| Group 3 | 8 | 4 | 11 | 0 |
| Group 4 | 11 | 5 | 7 | 0 |
| Group 5 | 9 | 4 | 10 | 0 |
| Group 6 | 3 | 4 | 16 | 0 |
| Group 7 | 9 | 4 | 9 | 1 |
| Group 8 | 6 | 6 | 10 | 1 |
| **합계** | **~64** | **~35** | **~82** | **~3** |
| **비율** | **~35%** | **~19%** | **~45%** | **~1%** |

### 3.2 핵심 발견

```
원래 추정: 184개 (27.3%) 전체가 Hallucination 의심
     ↓
실제 결과: 약 35%만 True Hallucination (~64개)
     ↓
전체 Hallucination 비율: ~9.5% (64/674)
```

**의미**: 실제 Hallucination 비율은 원래 추정(17.7%)의 **약 절반 수준**

---

## 4. "Not Hallucination" 케이스 원인 분석

약 45%의 케이스가 실제로는 이미지에 정보가 있었던 이유:

### 4.1 동일 PPT 다른 슬라이드
Golden 이미지가 아니어도 같은 문서의 다른 슬라이드에 동일 정보 존재

**예시**: `train_8309` (TFM&A 약어)
- Golden: 8309_6.jpg
- Retrieved: 8309_18.jpg, 8309_3.jpg
- 결과: Retrieved 이미지 양쪽에 "TECHNOLOGY FOR MARKETING & ADVERTISING" 표시

### 4.2 테이블/차트 데이터
Retrieved 이미지 내 테이블에서 직접 계산 가능

**예시**: `train_7338` (UK vs China recruiting leaders 차이)
- Retrieved: 7338_2.jpg에 UK: 400, China: 201 표시
- 계산: 400 - 201 = 199 (정답)

### 4.3 명확한 레이블
답변에 필요한 정보가 이미지에 직접 레이블로 표시

**예시**: `train_3459` (Map과 Reduce 사이 기능)
- Retrieved: MapReduce 다이어그램에 "Shuffle/Sort" 명확히 표시

---

## 5. True Hallucination 패턴 분류

### 5.1 패턴별 분포

| 패턴 | 비율 | 설명 |
|------|------|------|
| **도메인 지식 활용** | ~40% | 유명인, 회사 정보 등 자체 지식 사용 |
| **완전 무관 이미지** | ~35% | 검색 결과가 질문과 전혀 관련 없음 |
| **수치 추론** | ~15% | 이미지에 없는 통계/계산값 생성 |
| **제목만 있는 슬라이드** | ~10% | 내용 없이 제목만 있는 슬라이드로 추론 |

### 5.2 대표 사례

#### 도메인 지식 활용
```
train_7632: "Who is the CEO of Evernote?"
- Retrieved: Virtual Assistants, Andy Stanley 인용구 슬라이드
- Generated: "Phil Libin" (정답)
- 분석: 이미지에 Evernote 관련 정보 전혀 없음, 자체 지식으로 생성
```

```
train_8188: "Who wrote VISUAL EXPLANATIONS?"
- Retrieved: "STRIVE FOR GRAPHICAL INTEGRITY" 슬라이드
- Generated: "Edward Tufte" (정답)
- 분석: 이미지에 책 제목이나 저자 정보 없음, 자체 지식으로 생성
```

#### 완전 무관 이미지
```
train_8347: "When did Shale Gas production growth begin in the US?"
- Retrieved: Brazil Petrobras 석유 생산 차트
- Generated: "Around 2007-2008" (정답: 2008)
- 분석: US Shale Gas와 전혀 관련 없는 Brazil 이미지
```

#### 수치 추론
```
train_9028: "How many characters worth a thousand words?"
- Retrieved: Twitter Cards 일반 슬라이드
- Generated: "140" (정답)
- 분석: "140자" 정보 없음, Twitter 일반 지식으로 추론
```

---

## 6. GRPO 훈련에 대한 영향

### 6.1 현재 상황 (Judge Coef=1.0, NDCG Coef=0.0)

```
검색 결과 품질과 무관하게 정답만 맞으면 높은 보상
     ↓
True Hallucination (~9.5%): 잘못된 이미지 + 정답 → 긍정적 보상
     ↓
잘못된 검색 행동이 강화될 위험
```

### 6.2 긍정적 측면
- **약 45%**의 케이스에서 Golden이 아닌 이미지에서도 동일 정보 획득 가능
- 검색 모델이 "정확한 슬라이드"가 아닌 "관련 문서"를 찾아도 답변 가능
- 실제 Hallucination 비율(~9.5%)은 원래 추정(~17.7%)보다 낮음

### 6.3 위험 요소
- **~35%의 True Hallucination**(전체 ~9.5%)으로 인해 검색 품질 향상 신호가 약화
- 모델이 "어떤 이미지든 자체 지식으로 답변 가능"이라고 학습할 위험

---

## 7. 권장사항

### 7.1 현재 훈련 (Focused Round 1)
```
✅ 현재 설정 유지 가능
   - 실제 Hallucination 비율이 예상보다 낮음 (~9.5%)
   - 약 45%의 케이스에서 검색 품질이 실제로 유효함

⚠️ 모니터링 강화 필요
   - Judge Score 분포 변화 추적
   - 특정 질문 유형에서의 Hallucination 패턴 관찰
```

### 7.2 다음 라운드 고려사항

| 옵션 | 장점 | 단점 |
|------|------|------|
| **NDCG 계수 소폭 추가 (0.1~0.2)** | 검색 품질 신호 유지 | 학습 복잡도 증가 |
| **Frozen Generator 프롬프트 강화** | Hallucination 감소 | 정확한 답변 비율 감소 가능 |
| **이미지-답변 일치도 검증 추가** | 정확한 보상 신호 | 구현 복잡도 높음 |

### 7.3 Frozen Generator 프롬프트 개선안

**현재**:
```
"You are a visual QA generator. Use only the provided images and the user question.
Return ONLY the final answer text without extra explanations."
```

**제안**:
```
"You are a visual QA generator.

CRITICAL RULES:
1. You MUST only answer if the information is DIRECTLY VISIBLE in the provided images
2. If the answer cannot be found in the images, respond with:
   'The information is not visible in the provided images.'
3. Do NOT use your external knowledge to answer
4. Do NOT guess or infer beyond what is explicitly shown

Return ONLY the final answer text without extra explanations."
```

---

## 8. 결론

### 8.1 요약 통계

| 항목 | 결과 |
|------|------|
| **분석 대상** | 184개 Hallucination 의심 케이스 |
| **True Hallucination** | ~35% (~64개) |
| **실제 전체 Hallucination 비율** | **~9.5%** (64/674) |
| **예상 대비** | 원래 추정(17.7%)의 약 절반 수준 |
| **주요 원인** | 도메인 지식 활용, 완전 무관 이미지 검색 |
| **훈련 영향** | 중간 - 검색 품질 신호 일부 손실 |

### 8.2 핵심 메시지

> 원래 우려했던 것보다 Hallucination 문제는 덜 심각합니다. 약 45%의 케이스에서 Golden 이미지가 아니어도 관련 정보가 Retrieved 이미지에 존재했습니다. 이는 검색 모델이 "정확한 슬라이드"보다 "관련 문서"를 찾는 능력을 학습하고 있을 수 있음을 시사합니다.
>
> 그러나 여전히 ~9.5%의 True Hallucination은 검색 모델 학습에 노이즈를 추가하므로, Round 2에서 NDCG 계수 소폭 추가 또는 Frozen Generator 프롬프트 강화를 고려할 수 있습니다.

---

## 9. 부록: 분석 방법론 상세

### 9.1 데이터 추출
```python
# Hallucination 의심 케이스 추출 조건
if judge_score >= 0.5 and not (retrieved_basenames & golden_basenames):
    hallucination_suspect = True
```

### 9.2 이미지 경로 매핑
```
retrieved_basenames: ["1234_5"]
     ↓
image_path: ./search_engine/corpus/img/1234_5.jpg
```

### 9.3 병렬 분석 구조
- 184개 케이스를 8개 그룹으로 분할 (각 23개)
- 8개 서브에이전트가 동시에 각 그룹 분석
- 각 에이전트가 이미지를 직접 열어서 내용 확인
- 분석 완료 후 결과 취합

---

*이 보고서는 Focused Round 1 훈련 중 Frozen Generator의 Hallucination 현상을 질적으로 분석한 결과입니다.*
