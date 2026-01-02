# SFT “성공 샘플만” 큐레이션 스펙 (Focused2/3/3v2)

목표: **기존 traj(도구 사용 정책)를 최대한 보존**한 채로, “근거 있는 성공” 샘플만 모아 SFT에 사용한다.  
원칙: **리라이팅은 하지 않는다(0%)**. 대신 *규칙 기반 하드 게이트* + *GPT 최종 검수(pass/fail)*로 품질을 보장한다.

대상 로그:
- `logs/focused2/unified_trajectory.jsonl`
- `logs/focused3/unified_trajectory.jsonl`
- `logs/focused3_v2/unified_trajectory.jsonl`

---

## 1) 용어

- **샘플(sample/rollout)**: 하나의 `uid` (예: `train_2747__s10__2c451b12`)
- **그룹(prompt-group)**: 동일 프롬프트에서 생성된 `n_agent`개의 샘플 묶음(예: `train_2747__*`의 `__s{idx}__`만 다른 16개)
- **성공(judge=1)**: 최종 보상이 `final_reward==1`(현재 설정에서 judge=1과 동치)
- **근거 있는 성공(evidence-success)**: `judge==1` AND `ndcg>0`

---

## 2) 규칙 기반(Deterministic) 1차 큐레이션

### 2.1 그룹 단위 채택 조건 (Hard-only 강화)

그룹 내 `judge==1` 샘플 수를 `n_success`라 할 때:
- `n_success > 8` 이면 **그룹 전체 폐기**
- `n_success <= 8` 이면 **그룹 채택(다음 단계로 진행)**

의도: “너무 쉬운 프롬프트(성공 다수)”는 제외하고, **어려운데 맞힌 케이스**를 강화한다.

### 2.2 샘플 단위 후보 조건 (SFT 입력 안전성 + 근거성)

그룹이 채택된 경우, 샘플 후보는 아래를 모두 만족해야 한다:
- `search_complete==true` (파싱/종료 성공)
- **시스템 에러 없음**
  - 예: `"[System Error: BBox crop failed ...]"`가 traj에 존재하면 즉시 제외
- `judge==1` (== `final_reward==1`)
- `ndcg > 0` (judge=1인데 ndcg=0인 케이스는 “근거 없는 성공”으로 제외)

주의:
- 본 큐레이션은 Focused2/3/3v2 로그만 사용한다는 전제이므로, `ndcg`는 항상 존재한다고 가정한다.

### 2.3 그룹 내 최대 4개 제한 (과적합 완화)

한 그룹에서 최종 유지하는 샘플은 **최대 4개**다. (4 미만이어도 그대로 유지)

4개로 줄일 때 정렬/타이브레이커(효율 우선):
1) `ndcg` 내림차순
2) `#search` 오름차순
3) `#bbox` 오름차순
4) traj 토큰 길이(또는 메시지 길이) 오름차순

---

## 3) GPT 최종 검수 (Pass/Fail)

### 3.1 적용 범위

규칙 기반 1차를 통과한 **모든 후보 샘플에 대해 GPT 검수**를 수행한다.  
GPT가 `pass=false`를 반환하면 **무조건 제외**한다(예외 없음).

### 3.2 검수 목적(리라이팅 금지)

GPT는 “다음과 같은 붕괴/저품질 신호”를 잡아내는 용도이며, **traj를 수정하지 않는다**.

검수 대상 실패 유형(전반적으로 모두 탐지):
- **쿼리 붕괴**: 의미 없는 토큰 나열/다국어 폭주/질문과 무관한 query
- **반복 행동**: 동일 query 연속 반복, 무의미한 bbox 남발/반복
- **근거-추론 불일치(명백한 수준)**: 전혀 관련 없는 이미지를 보고 있다고 우기는 등
- **형식 붕괴**: `<think>...</think>` 및 action tag 규격 위반(남아있을 경우)

권장 출력 형식(머신 후처리 용이):
- GPT 출력은 JSON만:
  - `pass: true|false`
  - `reasons: string[]`
  - `flags: { query_collapse: bool, repetitive: bool, gibberish: bool, evidence_mismatch: bool, format_violation: bool }`
  - `severity: 0..3`

---

## 4) 산출물(권장)

큐레이션 결과는 목적별로 최소 2종을 만든다:

1) **SFT 학습용**
   - `judge==1 AND ndcg>0 AND no_system_error AND search_complete`
   - GPT 검수 `pass=true`
   - 그룹당 최대 4개

2) **분석/디버깅용(옵션)**
   - (a) `judge==1` 전체(그룹 필터 적용 후) / (b) `judge==1 AND ndcg>0`만
   - GPT `pass=false`로 떨어진 샘플도 별도 보관하면, “붕괴 원인” 분석에 유용함

---

## 5) 구현 시 주의점(로그/런 섞임)

unified JSONL이 “단일 파일 append” 구조인 경우:
- 반드시 **stage 경로 기준(focused2/3/3v2)**으로 입력 로그를 분리하거나,
- 파일 단일인 경우 **run_id / experiment_name / timestamp** 기반 필터를 적용해 런이 섞이지 않게 한다.

---

## 6) 다음 단계(코드 변경 계획 요약)

코드 수정 시, 아래 흐름으로 구성한다:
1) `scripts/extract_trajectories.py`에서 Focused2/3/3v2 unified log를 읽고, judge/ndcg 기반으로 후보를 Parquet으로 export
2) `scripts/vet_sft_rollouts_gpt.py`로 후보 전체를 GPT pass/fail 최종 검수
3) `scripts/finalize_success_only_sft_dataset.py`로 prompt-group당 최대 4개 캡(효율 우선) 후 train1-style Parquet 생성

권장 실행 예시:

```bash
# (옵션) 한 번에 돌리는 래퍼:
#   bash scripts/pipeline_success_only_sft.sh --stage focused3_v2 --out-prefix after_focus_success_only --model gpt-5.2 --concurrency 200
#
# 1) 후보 추출 (그룹 난이도 필터: judge==1 성공률 <= 0.5, rollout 필터: judge==1 & ndcg>0)
python scripts/extract_trajectories.py \
  --input logs/focused2/unified_trajectory.jsonl \
  --output logs/focused2/success_candidates.parquet \
  --export-train1-parquet \
  --export-extra-metrics \
  --drop-system-errors \
  --min-judge-score 1.0 \
  --min-ndcg 1e-9 \
  --group-success-metric judge \
  --group-success-threshold 1.0 \
  --max-success-rate 0.5

# 2) GPT 최종 검수 (pass==true만 남김)
python scripts/vet_sft_rollouts_gpt.py \
  --input logs/focused2/success_candidates.parquet \
  --output logs/focused2/success_candidates.vetted.parquet \
  --report logs/focused2/vet_report.jsonl \
  --cache logs/focused2/vet_cache.jsonl \
  --model gpt-5.2 --concurrency 200 --workers 200

# 3) 그룹당 최대 4개 캡 + train1 schema로 정리
python scripts/finalize_success_only_sft_dataset.py \
  --input logs/focused2/success_candidates.vetted.parquet \
  --output data/sft_success_only.train1.parquet \
  --max-per-group 4
```
