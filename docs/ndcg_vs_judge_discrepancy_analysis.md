# NDCG vs LLM as Judge 점수 차이 질적 분석 보고서

**분석 일시**: 2025-12-30
**분석 대상**: `flash_rm_detail.jsonl` (731개 고유 질문)
**분석 방법**: 7개 서브에이전트 병렬 질적 분석

---

## 1. 분석 배경

### 1.1 목적
NDCG(Normalized Discounted Cumulative Gain)와 LLM as Judge 점수 간의 차이가 큰 케이스들을 분석하여, 기존 NDCG 방식의 한계를 파악합니다.

### 1.2 분석 대상 케이스
두 가지 유형의 케이스를 추출:

| 유형 | 조건 | 케이스 수 | 의미 |
|------|------|----------|------|
| **High Judge, Low NDCG** | Judge >= 0.5, NDCG == 0 | 133개 | Golden 이미지 없이 정답 |
| **Low Judge, High NDCG** | Judge == 0, NDCG == 1.0 | 17개 | 완벽 검색 but 오답 |

---

## 2. Category 1: High Judge, Low NDCG (133개)

### 2.1 정의
- **Judge Score >= 0.5**: 정답으로 판정됨
- **NDCG == 0**: Golden 이미지를 전혀 검색하지 못함

이 케이스들은 **"NDCG가 0인데도 정답을 맞춘 경우"**로, NDCG 평가 방식의 한계를 보여줍니다.

### 2.2 분류별 통계

| 분류 | 개수 | 비율 | 설명 |
|------|------|------|------|
| **Same Document** | ~91 | ~69% | 같은 문서의 다른 슬라이드에서 동일 정보 획득 |
| **Domain Knowledge** | ~25 | ~19% | 모델의 자체 지식으로 답변 (Hallucination) |
| **Table/Chart Data** | ~11 | ~8% | 다른 형태의 차트/테이블에서 동일 데이터 |
| **Partial Info** | ~5 | ~4% | 부분적 단서로 추론 |

### 2.3 핵심 발견: Same Document 패턴 (69%)

**가장 빈번한 원인**은 같은 PPT/문서의 다른 슬라이드에 동일한 정보가 존재하는 경우입니다.

#### 패턴 유형

1. **애니메이션 분할 저장**
   - PPT 애니메이션이 슬라이드별로 저장되어 같은 표/차트가 여러 이미지로 분할
   - 예: train_8269 - "Social + Emotional Intelligence" 표가 단계별로 여러 슬라이드에 존재

2. **정보 반복**
   - 표지, 요약, 상세 슬라이드에 동일 키워드/정보 반복
   - 예: train_8635 - "DHMS" 약어가 표지와 내용 슬라이드 모두에 등장

3. **동일 데이터의 다른 표현**
   - 같은 데이터가 텍스트와 차트 두 형태로 존재
   - 예: train_10603 - "2009년 8% vacancy rate"가 텍스트 슬라이드와 차트 슬라이드에 모두 존재

### 2.4 대표 사례

#### Case 1: train_10033 (Same Document)
```
질문: "Who is a player?"
정답: "Person playing your game"
Retrieved: 10033_14 (Gamification 101 표)
Golden: 10033_16

분석: Retrieved 이미지에 "Player: the person playing your game"이 명확히 표시됨
     → NDCG=0이지만 실제로 정답 정보가 Retrieved 이미지에 있음!
```

#### Case 2: train_8188 (Domain Knowledge)
```
질문: "Who wrote VISUAL EXPLANATIONS?"
정답: "Edward Tufte"
Retrieved: 8188_3 (STRIVE FOR GRAPHICAL INTEGRITY 슬라이드)
Golden: 8188_1

분석: Retrieved 이미지에 책 제목/저자 정보 없음
     → 모델의 사전 지식으로 정답 생성 (Edward Tufte는 유명 저자)
```

#### Case 3: train_10603 (Table/Chart Data)
```
질문: "What percent of apartment units were vacant in 2009?"
정답: "8%"
Retrieved: 10603_15 (Apartment Vacancy Rate Since 2004 차트)
Golden: 10603_14 (텍스트 설명)

분석: Retrieved 차트에 2009년 8.0% 데이터 포인트가 명확히 표시
     → 같은 데이터가 차트와 텍스트 두 형태로 존재
```

---

## 3. Category 2: Low Judge, High NDCG (17개)

### 3.1 정의
- **Judge Score == 0**: 완전히 오답으로 판정됨
- **NDCG == 1.0**: Golden 이미지를 완벽하게 검색함

이 케이스들은 **"완벽한 검색 ≠ 정확한 답변"**을 보여줍니다.

### 3.2 분류별 통계

| 분류 | 개수 | 비율 | 설명 |
|------|------|------|------|
| **Calculation Error** | 5 | 29.4% | 숫자 계산 과정에서 오류 |
| **Misreading** | 5 | 29.4% | 텍스트/데이터를 잘못 읽음 |
| **Complex Reasoning** | 3 | 17.6% | 복잡한 시각적 추론 실패 |
| **Interpretation Error** | 2 | 11.8% | 질문 의도 잘못 해석 |
| **Ambiguous Question** | 1 | 5.9% | 질문 자체가 모호 |
| **Image Quality** | 1 | 5.9% | 이미지 품질 문제 |

### 3.3 핵심 발견

**검색 성공이 답변 성공을 보장하지 않음**

올바른 이미지를 검색했음에도 불구하고:
- 숫자를 읽고 계산하는 것은 별개의 어려운 작업
- 유사한 텍스트/값을 혼동하는 경우 다수
- 객체 세기, 성별 구분 등 복잡한 시각적 추론 필요

### 3.4 대표 사례

#### Case 1: train_7033 (Calculation Error)
```
질문: "How many more pre-paid subscribers were there in 2010 than post-paid subscribers?"
생성 답변: "57,213,354"
정답: "51850427"
NDCG: 1.0 (완벽 검색)

분석: 올바른 이미지(7033_5)를 검색했지만 숫자 계산 과정에서 오류 발생
```

#### Case 2: train_1871 (Complex Reasoning)
```
질문: "How many humans are visible in total?"
생성 답변: "Four fewer females than males"
정답: "6"
NDCG: 1.0 (완벽 검색)

분석: 질문은 총 인원수를 물었지만, 모델은 성별 비교로 잘못 해석
```

#### Case 3: train_3173 (Misreading)
```
질문: "Which element has the lowest abundance?"
생성 답변: "Osmium (Os)"
정답: "Ir" (Iridium)
NDCG: 1.0 (완벽 검색)

분석: 테이블에서 가장 낮은 값을 찾는 과정에서 유사한 값들 혼동
```

---

## 4. NDCG 방식의 한계

### 4.1 구조적 한계

```
NDCG의 가정: "특정 Golden 이미지만이 정답을 담고 있다"
     ↓
현실: 같은 정보가 여러 슬라이드에 분산되어 있음
     ↓
결과: 유효한 검색 결과가 NDCG=0으로 평가됨
```

### 4.2 한계점 요약

| 한계 | 설명 | 영향 |
|------|------|------|
| **단일 Golden 가정** | 하나의 이미지만 정답으로 지정 | 정보 중복성 무시 |
| **의미적 동등성 미고려** | 같은 정보의 다른 표현 인식 못함 | False Negative 증가 |
| **문서 구조 무시** | PPT 슬라이드 간 관계 고려 안함 | 관련 슬라이드 패널티 |
| **검색-생성 분리 불가** | 검색 품질과 답변 품질 혼재 | 정확한 진단 어려움 |

### 4.3 수치적 증거

```
전체 731개 질문 중:
├── High Judge, Low NDCG: 133개 (18.2%)
│   └── 그 중 ~69%는 실제로 유효한 정보 검색
│       → NDCG가 약 12%의 케이스를 부당하게 0점 처리
│
└── Low Judge, High NDCG: 17개 (2.3%)
    └── 완벽한 검색에도 답변 실패
        → NDCG 1.0이 답변 품질을 보장하지 않음
```

---

## 5. LLM as Judge vs NDCG 비교

| 측면 | NDCG | LLM as Judge |
|------|------|--------------|
| **평가 대상** | 검색된 이미지의 ID | 최종 답변의 의미적 정확성 |
| **정보 중복** | 고려 안함 | 자동으로 고려 |
| **도메인 지식** | 패널티 (hallucination) | 정확하면 인정 |
| **계산 오류** | 감지 못함 | 감지함 |
| **실용성** | 검색 시스템 평가에 적합 | 최종 사용자 경험 평가에 적합 |

---

## 6. 권장사항

### 6.1 평가 메트릭 개선

1. **다중 Golden 이미지 설정**
   - 같은 정보를 담은 모든 슬라이드를 Golden으로 지정
   - 정보 동등성 기반 자동 라벨링 검토

2. **문서 단위 평가 (Document-level NDCG)**
   - 슬라이드가 아닌 문서/프레젠테이션 단위로 검색 성공 여부 판단
   - 같은 문서의 다른 슬라이드는 부분 점수 부여

3. **하이브리드 메트릭**
   ```
   Final Score = α × NDCG + (1-α) × Judge Score

   권장: α = 0.0~0.2 (Judge 중심)
   ```

### 6.2 훈련 전략 조정

현재 설정 (Focused Round 1):
```
RM_JUDGE_COEF=1.0
RM_NDCG_COEF=0.0
```

**권장사항**: 현재 설정 유지
- NDCG의 구조적 한계가 명확히 확인됨
- Judge Score가 실제 답변 품질을 더 정확히 반영
- 단, Domain Knowledge (hallucination) 모니터링 필요

### 6.3 데이터셋 개선

1. **Golden 라벨링 검토**
   - 같은 정보를 담은 슬라이드들 추가 라벨링
   - 정보 동등성 검증 파이프라인 구축

2. **질문 품질 검토**
   - 모호한 질문들 재검토 (Category 2의 Ambiguous Question)
   - Reference answer 정확성 검증

---

## 7. 결론

### 7.1 핵심 발견

> **NDCG의 가장 큰 한계**: 같은 문서 내 정보 중복성을 고려하지 못함
>
> 133개의 "High Judge, Low NDCG" 케이스 중 **약 69%**가 같은 문서의 다른 슬라이드에서 유효한 정보를 검색한 경우였습니다. 이는 NDCG가 약 **12%의 전체 케이스**를 부당하게 0점 처리하고 있음을 의미합니다.

### 7.2 실무적 시사점

1. **Visual RAG 평가에서 NDCG만으로는 불충분**
   - LLM as Judge와 조합 필요
   - 또는 Document-level NDCG로 대체

2. **검색 성공 ≠ 답변 성공**
   - 17개 케이스에서 완벽한 검색에도 답변 실패
   - 검색과 생성 능력을 별도로 평가/개선 필요

3. **현재 훈련 설정 (Judge Coef=1.0) 적합**
   - NDCG의 한계를 고려할 때 Judge 중심 보상이 적절
   - Domain Knowledge (hallucination)에 대한 추가 모니터링 권장

---

## 8. 부록: 분석 방법론

### 8.1 케이스 추출
```python
# High Judge, Low NDCG
if judge_score >= 0.5 and ndcg == 0.0:
    high_judge_low_ndcg.append(case)

# Low Judge, High NDCG
if judge_score == 0.0 and ndcg == 1.0:
    low_judge_high_ndcg.append(case)
```

### 8.2 병렬 분석 구조
- High Judge, Low NDCG: 6개 그룹 (각 22개)
- Low Judge, High NDCG: 1개 그룹 (17개)
- 총 7개 서브에이전트가 병렬로 질적 분석 수행

### 8.3 이미지 경로
```
retrieved_basenames: ["1234_5"]
     ↓
image_path: ./search_engine/corpus/img/1234_5.jpg
```

---

*이 보고서는 NDCG와 LLM as Judge 평가 방식 간의 차이를 질적으로 분석한 결과입니다.*
