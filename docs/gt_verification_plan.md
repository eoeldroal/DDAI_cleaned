# Ground Truth 검증 계획

## 현재 진행 상황

### Phase 1: 0점 샘플 검증 (완료)
- **대상**: 231개 0점 샘플 중 NDCG >= 0.5인 66개
- **방법**: 8개 병렬 서브에이전트로 이미지 직접 확인
- **결과**:
  - GT_ERROR: 29개 (43.9%) - Reference Answer 오류
  - GEN_ERROR: 25개 (37.9%) - Generated Answer 오류
  - UNCERTAIN: 11개 (16.7%) - 판단 불가
  - BOTH_WRONG: 1개 (1.5%) - 둘 다 오류

### Phase 2: 낮은 점수 샘플 검증 (계획)
- **대상**: 0 < score <= 0.3, NDCG >= 0.5인 111개
- **방법**: 8개 병렬 서브에이전트로 이미지 직접 확인
- **목표**: 추가 GT_ERROR 발견

## 필터링 대상 목록

### 확정된 필터링 대상 (41개)
저장 위치: `logs/samples_to_filter.json`

| 분류 | 개수 | UIDs |
|------|------|------|
| GT_ERROR | 29 | train_7033, train_1100, train_2919, ... |
| UNCERTAIN | 11 | train_3820, train_3229, train_5878, ... |
| BOTH_WRONG | 1 | train_6893 |

### 추가 검증 대상 (111개)
저장 위치: `logs/extended_gt_check_candidates.json`

## 전체 데이터 분포

```
795개 샘플
├── 0점 (231개, 29.1%)
│   ├── NDCG >= 0.5: 66개 (검증 완료)
│   │   ├── GT_ERROR: 29개 → 필터링 대상
│   │   ├── GEN_ERROR: 25개 → 유지
│   │   └── 기타: 12개 → 필터링 대상
│   └── NDCG < 0.5: 165개 (검색 실패 → 유지)
│
├── 0.0-0.3점 (205개, 25.8%)
│   ├── NDCG >= 0.5: 111개 ← Phase 2 검증 대상
│   └── NDCG < 0.5: 94개
│
├── 0.3-0.5점 (73개, 9.2%)
├── 0.5-0.7점 (88개, 11.1%)
├── 0.7-0.9점 (85개, 10.7%)
├── 0.9-1.0점 (11개, 1.4%)
└── 1.0 만점 (102개, 12.8%)
```

## 다음 단계

### Phase 2 실행 계획
1. 111개 샘플을 8개 그룹으로 분배 (~14개씩)
2. 8개 병렬 서브에이전트 실행
3. 결과 종합 및 필터링 목록 업데이트

### 예상 추가 GT_ERROR
- Phase 1에서 43.9% GT_ERROR 발견
- Phase 2에서도 유사한 비율 예상 → 약 40-50개 추가 발견 가능

## 파일 목록

| 파일 | 설명 |
|------|------|
| `logs/samples_to_filter.json` | 확정 필터링 대상 41개 |
| `logs/gt_verify_combined.json` | Phase 1 상세 결과 |
| `logs/extended_gt_check_candidates.json` | Phase 2 검증 대상 111개 |
| `logs/high_ndcg_zero_score.json` | Phase 1 원본 데이터 |

---
최종 업데이트: 2024-12-30
