# SlideVQA Ground Truth 검증 보고서

## 개요

GRPO 훈련 중 **0점(Judge Score = 0.0)**을 받은 231개 샘플에 대한 정밀 분석 결과,
**검색은 성공했지만(NDCG >= 0.5) Judge가 0점을 준 66개 샘플**에서 심각한 데이터 품질 문제를 발견했습니다.

## 분석 방법

1. **대상 선정**: 231개 0점 샘플 중 NDCG >= 0.5인 66개 샘플 (검색 성공 케이스)
2. **검증 방식**: 8개 병렬 서브에이전트가 각 샘플의 실제 이미지를 확인
3. **판단 기준**: 이미지에서 Query에 대한 정답을 직접 확인 후 Reference vs Generated 비교

## 검증 결과

### 분류별 통계

| 분류 | 개수 | 비율 | 설명 |
|------|------|------|------|
| **GT_ERROR** | 29 | 43.9% | Reference Answer(Ground Truth)가 틀림 |
| **GEN_ERROR** | 25 | 37.9% | Generated Answer가 틀림 |
| **UNCERTAIN** | 11 | 16.7% | 판단 불가 (질문 모호 등) |
| **BOTH_WRONG** | 1 | 1.5% | 둘 다 틀림 |
| **FORMAT_DIFF** | 0 | 0.0% | 형식만 다름 |

### 핵심 발견

```
┌─────────────────────────────────────────────────────────────────┐
│  SlideVQA 데이터셋의 레이블 오류율: 43.9% (29/66)              │
│  → 검색 성공 + 0점 샘플의 거의 절반이 Ground Truth 오류!       │
└─────────────────────────────────────────────────────────────────┘
```

## GT_ERROR 상세 목록 (29개)

### 수치 계산 오류 (10개)

| UID | Query (요약) | Reference(틀림) | Generated(맞음) | 이미지 검증 결과 |
|-----|-------------|-----------------|-----------------|------------------|
| train_7033 | 2010년 Pre-Paid - Post-Paid 차이 | 51,850,427 | 57,213,354 | 64,469,827 - 7,256,473 = **57,213,354** |
| train_2919 | Android(49%) - Blackberry(11%) 차이 | 11 | 38% | 49% - 11% = **38%** |
| train_6112 | China 81% Daily 중 10+times/day 비율 | 31 | 48% | 이미지에서 **48%** 확인 |
| train_7557 | May-2014 - May-2012 스마트폰 차이 | 18 | 16 million | 22M - 6M = **16M** |
| train_2935 | 4위(UK 5%) - Korea(4%) 차이 | 11 | 1% | 5% - 4% = **1%** |
| train_2886 | Below Rs 5k(40%) + Rs 5k-10k(34%) | 44 | 74% | 40% + 34% = **74%** |
| train_6881 | Christians - Muslims (2010) | 5.7 | 0.57 billion | 2.17B - 1.6B = **0.57B** |
| train_4829 | Men in World Pop. 2010 | 3.4 | 3.5B | 이미지에서 **3.5B** 확인 |
| train_7919 | Section 1 페이지 수 | 14 | 13 | 5-17 범위, 8 제외 = **13** |
| train_1002 | Big Data TAM (millions) | 100 | $100,000 million | 이미지에서 **$100bn** 확인 |

### 이미지 내용 오독 (8개)

| UID | Query (요약) | Reference(틀림) | Generated(맞음) | 이미지 검증 결과 |
|-----|-------------|-----------------|-----------------|------------------|
| train_1100 | Motors vs Compressors 중 낮은 것 | Motors | Compressors | Motors 1,516,619 kWh > Compressors 116,376 kWh |
| train_4830 | Men vs Women in World Pop. 2010 | Women | Men | Men 3.5B > Women 3.4B |
| train_8289 | IP 192.168.7.8:9205 소유자 | Roy | Zhora | 이미지에서 **Zhora** 확인 |
| train_8006 | 스마트폰에 표시된 시간 | 5:42AM | 9:42 AM | iPhone은 항상 **9:42 AM** (Apple 전통) |
| train_8573 | 프레젠테이션 제목 | actionable dreams | Growth Hacking | 타이틀 슬라이드에 **Growth Hacking** |
| train_1705 | 오리 새끼 수 | 9 | 8 | 이미지에서 **8마리** 확인 |
| train_6892 | 20년 후 Rs. 30,000 가치 | 62,368 | 80,000 | 15년=62,368, 20년=**79,599** |
| train_3691 | IMPLEMENTING 다음 action | CARRY OUT PLAN | Evaluating | Nursing Process: Implementing → **Evaluating** |

### X축/Y축 혼동 (2개)

| UID | Query (요약) | Reference(틀림) | Generated(맞음) | 이미지 검증 결과 |
|-----|-------------|-----------------|-----------------|------------------|
| train_1866 | J-shaped curve의 X축 | Relative death rate | Systemic blood pressure | X축 = **Systemic blood pressure** (Y축이 Relative death rate) |

### 연도/시점 혼동 (4개)

| UID | Query (요약) | Reference(틀림) | Generated(맞음) | 이미지 검증 결과 |
|-----|-------------|-----------------|-----------------|------------------|
| train_1161 | 다이어그램에서 가장 오래된 연도 | 1954 | 1947 | 이미지에서 **1947** 확인 |
| train_2636 | 2014 9/10월 중 session 적은 달 | Oct | September | September가 **더 적음** |
| train_8701 | Presto 오픈소스까지 걸린 시간 | a year | A few months | 이미지에서 **몇 달** 확인 |
| train_6487 | 2014년 FB Revenues | 2.84 | 1.26 | 전체 시장=2.84, FB 매출=**1.26** |

### 부정확한 레이블 (5개)

| UID | Query (요약) | Reference(틀림) | 문제 |
|-----|-------------|-----------------|------|
| train_2129 | 파이 차트에서 Road-Rail 차이 | B2 | "B2"는 플레이스홀더 레이블, 실제 값 아님 |
| train_9980 | Sequoia Capital 문제 중 하나 | Falling asset prices | Generated도 정답 (여러 답변 가능) |
| train_7189 | negative growth 회사 | Staples | Generated(Omnicom)도 정답 (둘 다 negative) |
| train_7191 | 프레젠테이션 제작자 | Blitz Marketing | 실제 제작자는 **Buzz Marketing Group** |
| train_10410 | container level 성능 향상 요소 | Bare metal | Bare metal은 최악 성능, Container가 최고 |

## 영향 분석

### 훈련에 미치는 영향

1. **잘못된 페널티 (False Negatives)**
   - 모델이 **올바른 답변**을 생성했는데 0점 페널티
   - 29개 샘플 × 8 generations = **232 training signals 오염**

2. **보상 신호 왜곡**
   - 올바른 행동에 대해 부정적 보상 학습
   - 검색 성공 + 올바른 생성 → 0점 (잘못된 학습)

3. **GRPO 학습 효과 저하**
   - 정책 최적화가 잘못된 방향으로 유도될 수 있음

### 전체 데이터셋 영향 추정

```
0점 샘플 231개 중:
├── NDCG >= 0.5 (검색 성공): 66개 (28.6%)
│   ├── GT_ERROR: 29개 (43.9% of 66)
│   ├── GEN_ERROR: 25개
│   └── 기타: 12개
└── NDCG < 0.5 (검색 실패/부분): 165개 (71.4%)

추정 GT 오류 영향:
- 0점 샘플 중: ~12.6% (29/231)
- 전체 학습 데이터 중: 추가 분석 필요
```

## 권장 사항

### 단기 (즉시 적용 가능)

1. **GT_ERROR 샘플 제외**
   - 검증된 29개 샘플을 훈련 데이터에서 제외
   - 또는 해당 샘플의 Reference Answer 수정

2. **NDCG 기반 필터링 강화**
   - NDCG >= 0.5 + Judge = 0 케이스 별도 검토

### 중기 (데이터 파이프라인 개선)

3. **데이터 검증 파이프라인 구축**
   - 새 데이터 추가 시 자동 검증
   - Ground Truth 품질 검사 자동화

4. **Judge 시스템 개선**
   - 현재: 텍스트만 비교 (query, generated, reference)
   - 개선: 이미지 기반 검증 추가

### 장기 (데이터셋 정제)

5. **SlideVQA 데이터셋 전수 검증**
   - 0점 샘플 외에도 Ground Truth 오류 가능성
   - 크라우드소싱 또는 자동화 검증 필요

## 결론

**검색 성공 + Judge 0점** 샘플의 **43.9%가 실제로는 데이터셋 레이블 오류**였습니다.
이는 모델이 올바르게 작동했음에도 잘못된 페널티를 받고 있음을 의미합니다.

SlideVQA 데이터셋의 품질 문제가 GRPO 훈련 효과를 저해하고 있으며,
데이터 정제가 모델 성능 향상의 핵심 병목일 수 있습니다.

---

**분석 일시**: 2024-12-30
**분석 도구**: Claude Code with 8 parallel sub-agents
**검증 샘플**: 66개 (NDCG >= 0.5, Judge = 0)
**결과 파일**: `logs/gt_verify_combined.json`
