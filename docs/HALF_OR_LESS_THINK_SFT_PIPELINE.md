# Half-Or-Less 성공률 SFT 큐레이션 (Think-Only 리라이팅)

이 문서는 RL 로그에서 SFT 데이터를 **재현 가능하게** 큐레이션하는 파이프라인을 설명합니다.

핵심 아이디어는 다음과 같습니다.

1) 한 프롬프트에 대해 여러 번 생성(rollout)했을 때 **정답률(성공률)이 절반 이하인 “불안정/어려운” 프롬프트**를 고르고,  
2) 그 안에서 **성공한 rollout만** SFT 타깃으로 남기며(필요 시 top-k),  
3) (선택) **GPT 기반 필터링**으로 위험/저품질 샘플을 제거하고,  
4) **GPT 기반 think-only 편집(리라이팅)**으로 `<think>...</think>`만 문장/논리 수준에서 정돈한 뒤,  
5) 마지막으로 **안전 후처리(safety post-pass)**로 포맷/정책 오염을 방지합니다.

목표는 “강화학습 중 운 좋게 맞춘(rare success) 궤적”을 바탕으로 **에이전트의 행동 정책(search/bbox/stop 시퀀스)은 그대로 유지**하면서, 사람이 읽어도 깔끔한 SFT 예제로 만드는 것입니다.

---

## 철학(Philosophy)

### 왜 “성공률 ≤ 0.5” 프롬프트인가?

`focused2` 설정에서는 한 프롬프트에 대해 16번 rollout(생성)을 수행합니다. 성공률 ≤ 0.5 프롬프트는 “불안정/어려운 케이스”로,

- 몇 번은 맞지만, 여러 번은 틀리는 “아슬아슬한 성공”을 포함하고
- 그 “성공 궤적”을 SFT로 증류하면
- 쉬운 데이터에 과적합하지 않으면서 해당 케이스의 **일관성(consistency)**을 끌어올리기 좋습니다.

### 왜 성공 rollout만 남기는가?

실패 rollout까지 SFT에 넣으면 다음과 같은 나쁜 정책을 학습할 위험이 있습니다.

- 무효 `bbox` 반복(bbox 실패 패턴 고착)
- 검색 루프
- 잘못된 종료/중단

따라서 이 파이프라인은 “프롬프트 선택은 낮은 성공률로” 하되, **학습에 쓰는 rollout은 성공한 것만** 남깁니다.

### 왜 `<think>...</think>`만 리라이팅하는가?

리라이팅은 “정답을 더 맞히기”가 아니라 **편집(editing)** 작업으로 취급합니다.

- 액션 태그(` <search>`, `<bbox>`, `<search_complete>`) 및 순서를 **절대 변경하지 않음**
- 새로운 사실(특히 이미지 관측 사실)을 **절대 추가하지 않음**
- `<think>...</think>` 내부의 문장/논리/가독성만 개선

---

## 입력/출력

### 입력

- RL 로그: `logs/focused2/unified_trajectory.jsonl`

### 중간 산출물 / 최종 산출물(기본값)

- 로그→큐레이션(train1 parquet): `/tmp/focused2_curated_half_or_less_train1.parquet`
- GPT 필터링 결과: `logs/focused2/sft_candidates_half_or_less_filtered.parquet`
- think-only 리라이팅 결과: `data/after_focus_half_or_less_sft.v2.rewritten.parquet`
- 최종 “safe” 데이터셋(SFT 권장): `data/after_focus_half_or_less_sft.v2.rewritten.safe.parquet`

모든 parquet은 **train1 호환 스키마**입니다.

- `uid`: string
- `format_version`: `"v1"`
- `messages`: JSON string (`[{role, content}, ...]`)

---

## 빠른 실행(권장)

아래 스크립트는 1~4단계를 한 번에 수행하고, 최종 산출물 경로(`${SAFE_OUT}`)를 출력합니다.

- `bash scripts/pipeline_half_or_less_think_sft.sh`

주의: 이 스크립트는 **SFT 학습을 실행하지 않습니다.**  
데이터셋을 검토한 뒤, 별도의 학습 스크립트로 진행하세요.

---

## 단계별 실행(무엇을 하는지)

### 1) RL 로그에서 큐레이션(성공률 ≤ 0.5)

성공(success)은 `score >= min_score`로 정의됩니다.  
성공률 ≤ 0.5인 “프롬프트”만 남기고, 그 안에서 성공 rollout만 export 하며, 프롬프트당 top-k를 적용할 수 있습니다.

- `python scripts/extract_trajectories.py --input logs/focused2/unified_trajectory.jsonl --output /tmp/focused2_curated_half_or_less_train1.parquet --export-train1-parquet --min-score 1.0 --min-success-rate 1e-7 --max-success-rate 0.5 --top-k 4`

메모:
- `focused2`에서는 프롬프트당 rollout이 16개(코드에서 `--stats`로 확인 가능)
- `--min-score 1.0`: 완전 정답만 성공으로 간주
- `--min-success-rate 1e-7`: “성공이 0개인 프롬프트”는 제외(즉, 최소 1개 성공은 있어야 함)
- `--top-k 4`: 프롬프트당 성공 rollout 중 상위 4개까지 유지

### 2) GPT로 쿼리/샘플 필터링(저품질/위험 샘플 제거)

- `python scripts/filter_sft_queries_gpt.py --input /tmp/focused2_curated_half_or_less_train1.parquet --output logs/focused2/sft_candidates_half_or_less_filtered.parquet --report logs/focused2/query_filter_report_half_or_less.jsonl --limit 0 --sample random --seed 0 --concurrency 100`

### 3) GPT로 `<think>`만 리라이팅(편집 작업)

이 단계는 반드시 아래를 보존해야 합니다.

- 액션 태그 및 순서 불변
- 메시지 내 think 블록 개수 불변
- `<think>...</think>` 바깥 텍스트 불변

추천 옵션:
- `--revert-new-image-claims`: 리라이팅이 새 “이미지 관측 단정”을 추가하면 해당 think를 원문으로 롤백
- `--revert-action-tags-in-think`: think 안에 `<bbox>` 같은 태그 텍스트를 넣으면 해당 think를 원문으로 롤백

- `python scripts/rewrite_sft_think_gpt.py --input logs/focused2/sft_candidates_half_or_less_filtered.parquet --output data/after_focus_half_or_less_sft.v2.rewritten.parquet --report logs/focused2/think_rewrite_report_half_or_less_v2.jsonl --cache logs/focused2/think_rewrite_cache_half_or_less_v2.jsonl --model gpt-5.2-2025-12-11 --reasoning-effort high --max-output-tokens 6000 --concurrency 200 --max-retries 5 --max-units-per-call 3 --request-timeout-seconds 600 --on-error drop --revert-new-image-claims --revert-action-tags-in-think`

### 4) 안전 후처리(safety post-pass, 권장)

이 단계는 다음을 수행합니다.

- **원문부터** `<think>` 안에 액션 태그 텍스트가 들어있던 row를 드랍(포맷 오염 방지)
- 리라이팅이 새로 추가한 이미지-단정 문구는 원문 think로 롤백

- `python scripts/clean_rewritten_sft_dataset.py --original logs/focused2/sft_candidates_half_or_less_filtered.parquet --rewritten data/after_focus_half_or_less_sft.v2.rewritten.parquet --output data/after_focus_half_or_less_sft.v2.rewritten.safe.parquet --drop-if-action-tag-in-think --revert-new-image-claims`

---

## 환경/크레덴셜

GPT 단계(2, 3)는 OpenAI 크레덴셜이 필요합니다(키를 스크립트에 하드코딩하지 마세요).

- `.env`에 최소:
  - `OPENAI_API_KEY=...`
  - `OPENAI_BASE_URL=https://api.openai.com/v1`

`Connection error` / `Temporary failure in name resolution`가 나오면 DNS부터 확인하세요:

- `getent hosts api.openai.com`

---

## SFT 전 점검(Sanity Checks)

### 데이터셋 크기/스키마 확인

- `python - <<'PY'\nimport pandas as pd\np='data/after_focus_half_or_less_sft.v2.rewritten.safe.parquet'\ndf=pd.read_parquet(p)\nprint('rows',len(df))\nprint(df.columns)\nprint(df['format_version'].unique())\nPY`

### “non-think 텍스트가 바뀌지 않았는지” 확인(리라이팅 정확성)

- `python - <<'PY'\nimport json,re,pandas as pd\norig=pd.read_parquet('logs/focused2/sft_candidates_half_or_less_filtered.parquet')\nnew=pd.read_parquet('data/after_focus_half_or_less_sft.v2.rewritten.safe.parquet')\nnewm={r.uid:r.messages for r in new.itertuples()}\nTH=re.compile(r\"<think>(.*?)</think>\", re.S|re.I)\n\ndef strip_think(s):\n  return TH.sub('<think></think>', s)\n\nbad=0\nfor r in orig.itertuples():\n  uid=r.uid\n  if uid not in newm: continue\n  m0=json.loads(r.messages); m1=json.loads(newm[uid])\n  for a,b in zip(m0,m1):\n    if a.get('role')=='assistant' and isinstance(a.get('content'),str) and isinstance(b.get('content'),str):\n      if strip_think(a['content'])!=strip_think(b['content']):\n        bad+=1; break\nprint('nonthink_changed_rows',bad)\nPY`

---

## 학습(별도 수행)

최종 safe parquet를 SFT trainer에 넘기면 됩니다(예시):

- `data.train_files="data/after_focus_half_or_less_sft.v2.rewritten.safe.parquet"`
- `data.val_files="data/after_focus_half_or_less_sft.v2.rewritten.safe.parquet"` (의도적으로 val split을 쓰지 않을 때)

Hydra override에서는 파일명에 `.`가 있으면 파싱 이슈가 날 수 있으니, **따옴표로 감싸는 것**을 권장합니다.

