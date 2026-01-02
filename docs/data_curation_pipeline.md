# 데이터 큐레이션 파이프라인

## 개요

본 문서는 SlideVQA 강화학습을 위한 데이터 큐레이션 파이프라인을 설명한다. 핵심 전략은 **점진적 데이터 정제**를 통해 어려운 샘플에 컴퓨팅 자원을 집중 투자하는 것이다.

```
Original Dataset (6,667 samples)
    ↓ Phase 1 학습 + NDCG 기반 버켓팅
Curriculum Buckets (A: 1,560, B: 2,791, 0: 637)
    ↓ 성능 기반 필터링 + 수동 품질 검수
Focused Round 1 (854 samples)
    ↓ Round 1 학습 + GT 검증 (3단계)
Focused Round 2 (366 samples)
```

---

## Stage 1: Curriculum Bucket 생성

### 목적

Phase 1 학습 결과를 바탕으로 샘플을 난이도별로 분류하여, 학습 효율이 높은 영역에 집중한다.

### 버켓 분류 기준

| 버킷 | NDCG 범위 | 샘플 수 | 설명 |
|------|-----------|---------|------|
| **B** (Mastered) | > 0.7 | 2,791 | 모델이 잘 수행 → 추가 학습 제외 |
| **A** (Edge-of-Competence) | 0.1 - 0.7 | 1,560 | 학습 최적 영역 |
| **0** (Unsolvable) | < 0.1 | 637 | 완전 실패 → 품질 조사 필요 |

### 구현

**스크립트**: `scripts/generate_curriculum_data.py`

```python
# 버켓 분류 로직
for uid, score in uid_scores.items():
    if score > 0.7:
        buckets['B'].append((uid, score))
    elif 0.1 <= score <= 0.7:
        buckets['A'].append((uid, score))
    else:
        buckets['0'].append((uid, score))
```

**입력**:
- `logs/gspo_phase1.json` - Phase 1 NDCG 점수
- `data/rag/slidevqa_train_6667.parquet` - 원본 데이터셋

**출력**:
- `data/curriculum_bucket_a.parquet` (1,560 samples)
- `data/curriculum_bucket_b.parquet` (2,791 samples)
- `data/curriculum_bucket_0.parquet` (637 samples)

### Bucket 0 품질 검수

Bucket 0은 NDCG < 0.1로 모델이 완전히 실패한 샘플들이다. 실패 원인이 **진정한 난이도** 때문인지 **데이터 품질 문제** 때문인지 구분이 필요하다.

**검수 방법**:
- 637개 샘플 전수 조사
- 8개 병렬 에이전트로 원본 이미지 직접 확인
- 질문-답변-이미지 일치 여부 검증

**검수 결과**:

| 문제 유형 | 샘플 수 | 비율 |
|----------|---------|------|
| 질문-이미지 불일치 | 32 | 58.2% |
| 정답 오류 | 19 | 34.5% |
| 질문-답변 유형 불일치 | 4 | 7.3% |
| **총 문제 샘플** | **55** | **8.6%** |

**출력**:
- `data/curriculum_bucket_0_filtered.parquet` (582 samples) - 정상 샘플
- `data/curriculum_bucket_0_excluded.parquet` (55 samples) - 문제 샘플

**상세 문서**: `docs/bucket_0_data_quality_analysis.md`

---

## Stage 2: Focused Round 1 생성

### 목적

Bucket A에서 여전히 어려운 샘플과 품질 검증된 Bucket 0을 병합하여 집중 학습 데이터셋을 구성한다.

### 구성 로직

**스크립트**: `scripts/prepare_focused_input.py`

```python
# 1. Bucket A에서 Hard samples 추출 (score <= 0.2)
df_hard_a = load_hard_samples_from_logs(
    log_path="logs/flash_rm_detail.jsonl",
    original_data_path="data/curriculum_bucket_a.parquet",
    threshold=0.2
)

# 2. Bucket 0 전체 (품질 검증 완료)
df_bucket_0 = load_bucket_0("data/curriculum_bucket_0_filtered.parquet")

# 3. 병합
df_merged = pd.concat([df_hard_a, df_bucket_0], ignore_index=True)
```

### 데이터셋 구성

| 소스 | 샘플 수 | 선택 기준 |
|------|---------|----------|
| Bucket A (hard) | 272 | Judge score <= 0.2 |
| Bucket 0 (filtered) | 582 | 품질 검수 통과 |
| **합계** | **854** | |

**출력**: `data/focused_round1.parquet` (854 samples)

---

## Stage 3: Focused Round 2 생성

### 목적

Round 1 학습 후에도 성능이 낮은 샘플 중 GT 오류가 없는 샘플만 선별하여 최종 집중 학습 데이터셋을 구성한다.

### Round 1 학습 결과 분석

**로그**: `logs/all_samples_reward_stats.jsonl`

| 점수 구간 | 샘플 수 | 비율 |
|----------|---------|------|
| Score = 0 | 231 | 29.1% |
| 0 < Score ≤ 0.1 | 15 | 1.9% |
| 0.1 < Score ≤ 0.2 | 112 | 14.1% |
| 0.2 < Score ≤ 0.3 | 78 | 9.8% |
| Score > 0.3 | 359 | 45.2% |

### GT 검증 (3단계)

낮은 점수의 원인이 **모델 능력 부족**인지 **GT 오류**인지 구분하기 위해 3단계 GT 검증을 수행했다.

| Phase | 대상 조건 | 조사 샘플 | GT_ERROR | 검증 방법 |
|-------|----------|----------|----------|----------|
| 1 | score=0, NDCG≥0.5 | 66 | 29 | 이미지 찾았는데 score=0 → GT 의심 |
| 2 | 0<score≤0.3, NDCG≥0.5 | 111 | 7 | 이미지 찾았는데 낮은 score → GT 의심 |
| 3 | NDCG<0.5 | 352+64 | 29+5 | 전수조사 + 샘플링 |

**검증 결과 요약**:

| 분류 | 샘플 수 | 설명 |
|------|---------|------|
| GT_ERROR | 70 | Reference Answer가 틀림 |
| UNCERTAIN | 11 | 판단 불가 |
| BOTH_WRONG | 1 | GT와 생성 답변 모두 틀림 |
| **필터 대상 합계** | **82** | |

**상세 결과**: `logs/samples_to_filter.json`

### 필터링 과정

```python
# 1. Score <= 0.3 샘플 추출
low_score_uids = {d['uid'] for d in log_data if d['avg_final_score'] <= 0.3}
# → 436 samples

# 2. GT 오류 등 필터 대상 로드
filter_uids = set(filter_data['filter_uids']['all'])
# → 82 samples

# 3. 교집합 제거
final_uids = low_score_uids - filter_uids
# → 436 - 70 = 366 samples

# 4. focused_round1에서 해당 샘플 추출
df_round2 = df_round1[df_round1['id'].isin(final_uids)]
```

**출력**: `data/focused_round2.parquet` (366 samples)

---

## 데이터 감소 추이

```
Stage           샘플 수    누적 감소율    큐레이션 방법
─────────────────────────────────────────────────────────
Original        6,667      -            원본 데이터셋
Curriculum A+0  2,142      67.9% ↓      NDCG 자동 버켓팅
Focused R1        854      60.1% ↓      Score 필터 + 수동 품질 검수
Focused R2        366      57.1% ↓      Score 필터 + LLM GT 검증
─────────────────────────────────────────────────────────
최종             366       94.5% ↓      원본 대비 5.5% 유지
```

---

## 컴퓨팅 자원 투자 전략

### 핵심 아이디어

동일한 총 컴퓨팅 예산으로 어려운 샘플에 더 많은 자원을 투자한다.

| Stage | 샘플 수 | 샘플당 상대 컴퓨트 | 의미 |
|-------|---------|------------------|------|
| Phase 1 (전체) | 6,667 | 1x | 기준선 |
| Focused R1 | 854 | ~8x | 8배 집중 |
| Focused R2 | 366 | ~18x | 18배 집중 |

### 기대 효과

1. **학습 효율 극대화**: 이미 잘하는 샘플(Bucket B) 제외
2. **노이즈 제거**: GT 오류 샘플 필터링으로 보상 신호 정화
3. **난이도 적응**: 점진적으로 어려운 샘플에 집중

---

## 파일 구조

### 데이터 파일

```
data/
├── rag/
│   └── slidevqa_train_6667.parquet    # 원본 (6,667)
├── curriculum_bucket_a.parquet         # Stage 1: NDCG 0.1-0.7 (1,560)
├── curriculum_bucket_b.parquet         # Stage 1: NDCG >0.7 (2,791)
├── curriculum_bucket_0.parquet         # Stage 1: NDCG <0.1 (637)
├── curriculum_bucket_0_filtered.parquet # Stage 1: 품질 검수 통과 (582)
├── curriculum_bucket_0_excluded.parquet # Stage 1: 품질 문제 (55)
├── focused_round1.parquet              # Stage 2 (854)
└── focused_round2.parquet              # Stage 3 (366)
```

### 로그 파일

```
logs/
├── gspo_phase1.json                    # Phase 1 NDCG 점수
├── flash_rm_detail.jsonl               # Phase 1/2 Judge 상세 로그
├── all_samples_reward_stats.jsonl      # Round 1 통계 (UID별 평균)
├── samples_to_filter.json              # GT 검증 결과 (필터 목록)
└── full_analysis/                      # NDCG<0.5 전수조사 결과
    ├── group_1.json ... group_22.json
    └── result_1.json ... result_22.json
```

### 스크립트

```
scripts/
├── generate_curriculum_data.py         # Stage 1: Bucket 생성
└── prepare_focused_input.py            # Stage 2: Focused R1 생성
```

### 문서

```
docs/
├── bucket_0_data_quality_analysis.md   # Bucket 0 품질 검수 보고서
└── data_curation_pipeline.md           # 본 문서
```

---

## 요약

| 단계 | 입력 | 출력 | 핵심 작업 |
|------|------|------|----------|
| Stage 1 | Original (6,667) | Buckets A/B/0 | NDCG 기반 자동 분류 + Bucket 0 수동 검수 |
| Stage 2 | Bucket A + 0 | Focused R1 (854) | Score 기반 hard sample 추출 |
| Stage 3 | Focused R1 | Focused R2 (366) | Score 필터 + GT 검증 |

**핵심 원칙**:
1. **자동 + 수동 병행**: 자동 필터링으로 대량 처리, 수동 검수로 품질 보장
2. **점진적 정제**: 각 단계에서 다른 기준으로 필터링
3. **컴퓨팅 집중**: 데이터 감소 → 샘플당 자원 증가

---

*문서 생성일: 2024-12-30*
*최종 업데이트: Focused Round 2 생성 완료*
