# DDAI 강화학습의 특징과 독창성 분석 (현재까지의 궤적 기준)

> 구현 디테일(코드/최적화/로그 포맷)보다 **“무엇을 학습하고, 왜 이 설계를 택했고, 무엇이 독창적인가”**에 초점을 둔 개념 문서다.  
> (실행/구조/파일 경로 중심의 핸드오프는 루트 `AGENTS.md` 참고)

---

## 0. 한 문장 정의

DDAI의 Phase 2 / Focused RL은 **“정답을 직접 생성하는 능력”이 아니라 “정답을 찾을 수 있게 만드는 검색·도구사용 정책(Searcher)”**을, 고비용 외부 모델(Generator/Judge)을 활용해 **데이터·보상·컴퓨트 커리큘럼까지 함께 설계하며** 강화학습으로 최적화하는 파이프라인이다.

---

## 1. 문제 설정 자체가 만드는 RL의 성격

### 1.1 우리가 푸는 문제는 “문서 기반 Visual RAG”의 *정책 학습*

일반적인 VQA/멀티모달 QA는 “이미지+질문 → 답변”의 단발 생성에 가깝다.  
하지만 DDAI의 환경은 다음 3가지가 결합된다:

1) **문서(슬라이드/페이지) 코퍼스**에서 증거를 *찾아야* 한다.  
2) 증거를 찾기 위해 **행동(action)** 을 해야 한다 (`<search>`, `<bbox>`, `<search_complete>`).  
3) 최종 답변은 “이미지에 보이는 정보”에 기반해야 하며, 그렇지 않으면 보상 신호가 왜곡된다.

따라서 이 RL은 본질적으로 “답변 생성”보다 **탐색/수집/확인/종료**로 구성되는 *행동 정책* 문제에 더 가깝다.

### 1.2 관측(observation)과 행동(action)이 ‘텍스트 안의 구조’로 정의됨

정책의 행동 공간이 자연어 전체가 아니라, **태그 기반의 구조화된 행동**으로 제한된다:

- `<search>query</search>`: 정보 획득(문서/페이지 이미지 검색)
- `<bbox>[x1,y1,x2,y2]</bbox>`: 최근 증거를 “더 잘 보이게” 가공(확대/크롭)
- `<search_complete>true</search_complete>`: 종료(더 이상 도구가 필요 없음을 선언)

이 구조화는 RL 관점에서 매우 중요하다:
- “무엇이 행동인가?”가 명확해져 **크레딧 할당(credit assignment)** 이 쉬워진다.
- 포맷 위반/무효 행동을 강하게 제어할 수 있어 **학습 안정성**이 올라간다.
- 후속 분석(로그 기반 진단/큐레이션)이 가능해져 **데이터-중심 RL**로 확장된다.

---

## 2. 핵심 독창성 1: 두 모델 분업(Trainable Searcher + Frozen Generator)

### 2.1 “정답을 말하는 모델”이 아니라 “정답을 찾는 모델”을 학습한다

DDAI는 Searcher(학습 대상)와 Frozen Generator(고정 모델)를 분리한다.

- Searcher: 멀티턴으로 검색/크롭/종료 결정을 내리는 정책
- Frozen Generator: Searcher가 모은 이미지를 보고 최종 `<answer>` 생성 (학습하지 않음)

이 분업이 주는 RL적 이점:

1) **학습 목표의 분해(Decoupling)**  
   - “언어 유창성/상식/추론” 같은 거대 능력은 Frozen Generator가 맡고,  
   - Searcher는 “어떤 증거를 가져오면 답을 낼 수 있는가”에 집중한다.

2) **학습 신호의 순도(Purity) 상승**  
   - 보상이 나빠도 그 원인이 “검색 실패”인지 “생성 실패”인지 구분 가능해진다(완벽하진 않지만 훨씬 명료해짐).

3) **도구사용 정책의 일반화 가능성**  
   - 특정 답변 스타일/문장 습관이 아니라 “증거 수집 행동”이 강화되므로, Generator를 바꿔도 정책이 재사용될 가능성이 생긴다.

### 2.2 State masking: “답변 토큰을 학습하지 않는” RL

Phase 2의 응답 문자열은 Searcher의 궤적 + Frozen Generator의 `<answer>`가 합쳐진다.  
하지만 학습은 Searcher 토큰에만 걸리도록 마스킹한다(문서/가이드에서 중요 포인트로 반복 강조됨).

이 설계는 RLHF/일반 PPO와 다른 성격을 만든다:
- 최적화 대상이 “정답 문장”이 아니라 **도구사용/탐색 전략**에 수렴한다.
- Frozen Generator가 우연히 잘 맞춘(혹은 환각으로 맞춘) 케이스가 있더라도, 그때 Searcher 행동이 무엇이었는지로 학습 신호가 귀속된다.

---

## 3. 핵심 독창성 2: 보상 설계가 “검색 품질 vs 답변 품질”을 분해한다

### 3.1 두 축의 보상 신호

DDAI Phase 2 보상은 크게 두 성분으로 구성된다:

1) **답변 품질(semantic correctness)**: Gemini Judge가 `<answer>`와 정답(reference)을 비교해 0~1 연속 점수를 산출  
2) **검색 품질(retrieval quality)**: NDCG로 “검색된 이미지(베이스네임)”과 “정답 이미지(베이스네임)”의 일치 정도 평가

이 혼합은 단순 가중합처럼 보이지만, 실제로는 다음을 동시에 해결하려는 설계다:

- Judge는 “정답만 맞으면 됨”이라서 **근거 없는 정답(환각/상식 베팅)** 을 보상할 위험이 있다.
- NDCG는 “Golden 이미지” 가정이 강해 **같은 문서의 다른 슬라이드**에서 동일 정보가 있는 경우를 과소평가할 수 있다.

즉, 두 메트릭은 서로의 구조적 약점을 보완한다(완벽한 해결은 아니며 trade-off 설계).

### 3.2 “NDCG의 한계”를 정면으로 다루는 RL

분석 문서들에서 반복 등장하는 핵심 발견:
- NDCG는 “지정된 golden 이미지”와 다르면 0점이 될 수 있지만,
- 실제로는 같은 문서 내 다른 페이지에 동일 정보가 존재하는 경우가 많고,
- 사용자 관점에서는 그것도 충분히 성공이다.

따라서 DDAI는 NDCG만으로 Searcher를 학습하지 않고,
**Judge(답변의 의미적 정확성)** 를 결합해 “사용자 체감 성공”을 더 직접적으로 반영하려 한다.  
동시에 Judge만 쓰면 “근거 없는 정답”을 강화할 수 있으므로, NDCG를 보조 신호로 남겨 **검색 행동의 학습 압력**을 유지한다.

### 3.3 계수(coef)와 게이트(gate): 보상은 ‘실험적으로 조절되는 레버’다

Focused RL에서 보상 계수는 단순 파라미터가 아니라, **학습이 어떤 실패모드를 줄이도록 유도하는 정책 레버**다.

- `Judge 중심(예: RM_JUDGE_COEF↑, RM_NDCG_COEF↓)`:
  - 목표: “정답 자체”를 맞추는 경험을 최대화하고, 희소한 성공을 늘려 GRPO 비교학습의 신호를 확보
  - 위험: 근거 없는 정답(상식/환각)이 정책을 오염시킬 가능성
- `NDCG 중심(예: RM_NDCG_COEF↑)`:
  - 목표: “Golden을 찾는 정책”을 더 직접적으로 압박
  - 위험: Golden 단일 가정 때문에 실제로는 유효한 검색을 0점 처리하는 False Negative 증가

또한 포맷/행동 제약을 강화하기 위한 “게이트형 보상(형식 점수 0이면 전체 0)”이 존재한다.  
이는 모델이 태그를 깨고 자연어로 도망가거나, 잘못된 액션을 반복해 환경을 망가뜨리는 것을 막는 **안전장치형 RL 설계**로 볼 수 있다.

### 3.4 Judge를 왜 외부 LLM으로 두는가: “평가자”를 정책에서 분리

이 프로젝트는 평가자를 Searcher와 분리한다:
- 평가자가 학습 중인 정책과 같은 모델이면, 평가 기준이 함께 흔들릴 수 있다.
- 외부 LLM Judge는 “비용”은 크지만, 상대적으로 **평가 기준이 안정적**이며 “의미적 정답”을 더 직접적으로 측정한다.

이건 RLHF에서 흔한 “고정 RM” 철학과 유사하지만, DDAI는 여기에 **검색 메트릭(NDCG)** 까지 섞어 “근거 기반 정답”으로 가는 압력을 유지하려는 점이 다르다.

---

## 4. 핵심 독창성 3: GRPO + n_agent가 만드는 ‘그룹 경쟁’ 기반 탐색

### 4.1 GRPO의 의미: “절대 점수”보다 “같은 문제에서 누가 더 잘했나”

GRPO는 Critic을 쓰지 않고, 같은 프롬프트에서 나온 여러 샘플(n_agent)을 비교해 advantage를 만든다.  
이 방식은 다음 성격을 강화한다:

- **탐색(exploration)의 구조화**: 같은 질문을 여러 번 시도해 서로 다른 검색 쿼리/크롭/종료 타이밍을 비교한다.
- **분산 감소**: 외부 API(Judge/Generator)의 노이즈가 있더라도 “같은 문제에서의 상대 비교”는 더 안정적일 수 있다.
- **샘플 효율**: 비싼 환경에서 “한 번의 질문”으로 여러 정책 변형을 동시에 관찰한다.

### 4.2 n_agent는 단순 배치 확장이 아니라 “탐색 예산(Exploration Budget)”

`n_agent`는 “프롬프트당 rollout 개수”이므로, 사실상 다음을 결정한다:
- 한 질문에 대해 정책이 얼마나 다양한 경로를 시도하는가?
- GRPO가 비교할 후보가 얼마나 많은가?

특히 Focused RL에서는 **“될성부른 문제에서 희소한 성공(rare success)을 확보”**하는 것이 중요해진다.  
이때 n_agent는 곧 **성공 궤적을 수확할 확률을 올리는 레버**다.

### 4.3 DDAI-47 관점: compute curriculum의 중심축은 사실상 n_agent × max_turns

Focused RL 설계 초안(DDAI-47)은 “Small→Medium→Large”의 탐색 예산을 단계적으로 늘린다.  
이를 RL 언어로 바꾸면:

- `max_turns` 증가: 한 episode에서 더 오래 탐색(더 많은 도구 호출/관측 업데이트)
- `n_agent` 증가: 같은 문제를 더 많이 병렬 시도(상대 비교 기반 학습 신호 강화)
- (선택) temperature 조정: 정책의 다양성(탐색성)을 조절

즉, DDAI-47의 독창성은 “온라인 승격” 자체보다, **고비용 환경에서 탐색 예산을 데이터/성과에 맞춰 재배분**하는 발상에 있다.

---

## 5. 핵심 독창성 4: 고비용 외부 API 환경을 RL의 일부로 취급한다

### 5.1 DDAI의 RL은 ‘환경 비용’이 매우 큰 특수 조건을 가진다

Phase 2는 외부 API 의존도가 높다:
- Search API (대규모 코퍼스 검색)
- Frozen Generator API (멀티모달 답변 생성)
- Gemini Judge API (정답 평가)

이 구조는 단순 “엔지니어링 비용” 문제가 아니라, RL 자체의 성격을 바꾼다:
- **reward delay(보상 지연)** 이 커지고,
- 환경 step의 wall-clock 변동성이 커져,
- “학습이 불안정해 보이는 원인”이 알고리즘 외부(지연/타임아웃/레이트리밋)에 의해 결정될 수 있다.

따라서 DDAI는 “학습 알고리즘”과 별개로, **학습이 가능해지는 시스템 조건(throughput, tail latency)** 을 함께 설계해야 했다.

### 5.2 Streaming reward는 ‘성능 최적화’이면서 동시에 ‘학습 구조의 변화’

스트리밍 reward(프롬프트 단위 병렬 처리)는 흔히 성능 최적화로만 보이지만, RL 관점에서는 다음 효과가 있다:

- Generation(Searcher)과 Reward(Judge)가 파이프라인으로 겹쳐져 **학습 step당 벽시계 시간**이 줄어든다.
- 긴 지연을 가진 외부 호출이 전체 학습을 멈추지 않게 해 **학습의 진행 가능성**을 높인다.
- 결과적으로 같은 예산에서 더 많은 rollout을 수행할 수 있어, **탐색 예산을 늘린 것과 유사한 효과**가 난다.

즉, “비싼 환경에서의 RL”은 알고리즘만으로 성립하지 않고, **지연을 흡수하는 실행 구조**까지 포함해 하나의 방법론이 된다.

---

## 6. 핵심 독창성 5: 데이터-중심 RL(큐레이션·검증·라운드 기반 집중)

### 6.1 DDAI는 ‘전체 데이터셋을 끝까지 RL’하지 않는다

외부 API 기반 환경에서 전체 데이터셋(수천~수만)을 동일하게 학습하는 것은 비효율적이다.  
그래서 DDAI는 “학습 신호가 잘 나오는 구간”에 컴퓨트를 집중한다.

대표적인 전략:
- Phase 1 결과(NDCG 등)로 **난이도 버킷(A/B/0)** 을 만들고,
- 버킷 중 “solvability gap(될 가능성이 있는 어려운 문제)”에 집중하며,
- Focused Round를 반복하며 데이터셋을 점점 더 작고 더 비싼 문제들로 압축한다.

이 과정은 단순 “데이터 선택”이 아니라, RL을 성공시키기 위한 핵심 설계(환경 비용을 고려한 샘플 효율화)다.

### 6.2 Ground Truth 품질 검증은 ‘보상 신호 정화’ 작업이다

분석 문서들에서 드러난 중요한 사실:
- 일부 구간에서는 GT 오류가 매우 높은 비율로 존재할 수 있으며,
- 이런 샘플은 RL에 들어오면 “정답 행동을 벌주는” 최악의 노이즈가 된다.

따라서 GT 검증/필터링은:
- 단순 데이터 청소가 아니라,
- **보상 모델의 일관성(Reward Consistency)** 을 회복하는 작업이며,
- GRPO 같은 상대 비교 기반 학습에서도 “그룹 내 비교가 무의미해지는” 상황을 줄인다.

### 6.3 Focused RL의 산출물은 ‘성능’만이 아니라 ‘Golden Trajectory’

DDAI-47 설계의 핵심 산출물은 “맞췄다”가 아니라:
- 어떤 검색 쿼리를 냈고,
- 어떤 이미지/크롭을 선택했고,
- 언제 멈췄으며,
- 그 결과 어떤 답을 얻었는지

즉, **trajectory(행동 경로)** 자체가 자산이 된다.

이 관점은 다음 파생을 낳는다:
- 성공 궤적을 모아 SFT 데이터로 재활용(rare success distillation)
- 실패 패턴을 유형화하여 환경/보상/검색엔진 개선으로 연결

---

## 7. 핵심 독창성 6: “로그 기반 과학”과 RL↔SFT 폐루프

고비용 RL에서 가장 위험한 것은 “무엇이 왜 좋아졌는지 모르는 상태로 비용만 쓰는 것”이다.  
그래서 DDAI는 일찍부터 다음을 강조한다:

1) **통합 로그(unified trajectory)** 로 episode/turn/도구호출/보상까지 한 흐름으로 남긴다.  
2) 그 로그를 기반으로:
   - 성공률/분포 분석
   - NDCG↔Judge 불일치 케이스 분석
   - Hallucination(근거 없는 정답) vs same-document(유효 근거) 분해
   - GT 오류 검증 및 필터링
3) 나아가 “성공 궤적”을 **SFT 데이터로 변환**하는 파이프라인(think-only rewrite 등)까지 연결한다.

이건 RL을 “최종 목적”이 아니라, **고품질 행동 데이터를 생성하는 엔진**으로도 취급하는 접근이다.

### 7.1 Focused RL → (rare success) SFT 증류: “성공률이 낮은 프롬프트에서 성공 궤적만 증류”

Focused Phase 2에서는 동일 프롬프트에 대해 다수 rollout을 생성한다(예: `focused2` 로그 기준 프롬프트당 rollout=16).  
이때 “대부분 실패하지만 가끔 성공하는” 프롬프트는 다음 두 이유로 SFT 증류 대상으로 유의미하다:

1) RL이 만들어낸 *희소한 성공 궤적(rare success)* 자체가 “학습 가능한 행동 경로”로서 가치가 크다.  
2) 반대로 실패 궤적을 그대로 SFT로 학습시키면 검색 루프/무효 행동이 고착될 수 있으므로, **성공 궤적만** 추출해 “재현성”을 강화하는 편이 안전하다.

따라서 우리는 다음과 같은 큐레이션 규칙을 사용해 SFT 후보를 만든다:
- 프롬프트 단위로 성공률(success rate)이 낮은 샘플(예: `success_rate ≤ 0.5`)을 선택
- 그 샘플 내 rollout 중 **정답 판정(LLM-as-Judge score==1)** 인 성공 rollout만 유지
- 프롬프트당 성공 궤적은 top-k로 제한(예: `top-k=4`)하여 데이터 규모를 통제

이 접근은 “어려운 문제에서 우연히 맞은 행동 경로”를 SFT로 증류해 **정책의 일관성을 높이는** 역할을 하며, 이후 RL 라운드의 초기화(bootstrap)에도 활용할 수 있다.

### 7.2 Think-only 리라이팅 + Safety post-pass: “행동 정책 불변”을 강제하는 데이터 정제

큐레이션된 성공 궤적을 바로 SFT로 학습시키면, 로그/환경 오류/포맷 오염이 모델 행동 정책을 흐릴 수 있다.  
특히 본 프로젝트의 행동 공간은 태그 기반이므로, 데이터 정제는 단순한 텍스트 정리가 아니라 **정책 공간(policy space)의 오염 방지**에 해당한다.

이를 위해 SFT 데이터 구성에서 아래 원칙을 강제한다:

- 리라이팅은 `<think>...</think>`에만 적용(편집 작업)하고, `<search>/<bbox>/<search_complete>` 및 순서는 변경하지 않는다.
- “이미지에서 보인다/I can see/the image shows” 같은 관측 단정은 새로 추가하지 않는다(원문에 없던 관측 주장 주입 방지).
- `<think>` 안에 `<bbox>...</bbox>` 같은 액션 태그를 “텍스트로” 포함한 샘플은 포맷 오염 위험이 크므로 제거하거나 롤백한다.

결과적으로 SFT는 “정답 문장 스타일”이 아니라 **성공 궤적의 행동 정책을 안정적으로 재현**하도록 작동하게 된다.

---

## 8. 이 접근이 갖는 독창성 요약 (체크리스트)

다음 항목이 동시에 결합된 점이 DDAI RL의 독창성이다:

1) **도구사용 정책(Searcher)만 학습**하고 답변은 고정 Generator에 위임  
2) **태그 기반 행동 공간**으로 관측/행동을 구조화하여 안정성과 분석 가능성을 확보  
3) **보상 분해(Judge + NDCG)** 로 사용자 체감 성공과 근거 기반 성공을 동시에 압박  
4) **GRPO(n_agent 그룹 비교)** 로 고비용 환경에서도 탐색 다양성과 학습 신호를 확보  
5) **스트리밍 reward/파이프라인 병렬화**로 reward delay를 줄여 RL 진행 가능성과 샘플 효율을 올림  
6) **데이터 큐레이션/검증/라운드 기반 집중**을 RL 설계의 중심으로 둠(“어디에 compute를 쓰는가”)  
7) 로그 분석과 SFT 증류까지 포함한 **폐루프(Closed-loop) 학습 전략**

---

## 9. 강화학습 훈련 절차(논문용 Method/Training Details)

이 섹션은 “무엇을 어떻게 학습했는가?”를 논문에 그대로 옮길 수 있도록, **훈련 절차를 단계/루프/설정/데이터 흐름** 중심으로 기술한다.  
(수치/데이터셋 크기/라운드 구성은 `docs/data_curation_pipeline.md`, `docs/focused_round*_analysis.md`에서도 근거를 찾을 수 있다.)

### 9.1 전체 훈련 파이프라인: Phase 분해

DDAI는 한 번의 RL로 모든 것을 해결하기보다, **학습 신호의 성격이 다른 구간을 Phase로 분해**해 순차적으로 진행했다.

1) **(선행) SFT**  
   - 목적: 태그 기반 형식/기본 지시 수행 능력, 멀티모달 입력 처리 안정화.  
   - 결과: Searcher가 최소한 “환경과 대화 가능한” 정책을 갖게 함.

2) **Phase 1: Format + NDCG 게이트 기반 RL (도구사용/검색 부트스트랩)**  
   - 목적: Searcher가 `<search>`, `<bbox>`, `<search_complete>` 등 “도구사용 언어”를 안정적으로 사용하고, 검색을 통해 golden에 가까운 이미지를 찾도록 유도.  
   - 특징:
     - Frozen Generator/LLM Judge를 쓰지 않고(또는 최소화하고), **형식 준수 + NDCG** 로 “근거 수집 정책”을 먼저 만든다.
     - 보상이 gate 형태(형식 실패 시 0)라서 초기 붕괴를 줄이고, “규칙을 지키는 정책”을 강화한다.

3) **Phase 2: Frozen Generator + LLM-as-Judge + (선택) NDCG 기반 RL (정답 경험 수확 + Focused RL)**  
   - 목적: Searcher가 “정답을 낼 수 있는 증거”를 더 잘 수집하도록 학습 신호를 **답변 정확성**까지 확장.  
   - 특징:
     - Searcher는 여전히 “도구사용 정책”만 학습하며, 최종 답변은 고정된 Frozen Generator가 생산한다.
     - Gemini Judge(텍스트 기반)가 `<answer>`를 reference와 비교해 0~1 연속 점수를 제공해, 희소 보상을 완화한다.
     - 데이터 큐레이션/라운드 기반 Focused RL로 컴퓨트를 집중 투입한다.

#### 9.1.1 데이터 구성(커리큘럼→Focused)과 라운드 전환(실제 진행)

이 프로젝트에서 “어떻게 RL을 진행했는가?”는 **알고리즘(Policy Gradient)만**이 아니라,  
**어떤 데이터를 어떤 순서로, 어떤 탐색 예산으로 학습했는가**를 포함한다.

핵심은 다음의 “funnel”이다(상세 근거/숫자는 `docs/data_curation_pipeline.md` 참고):

1) **Original SlideVQA train**: 6,667 samples  
2) **Curriculum bucketing (Phase 1 결과 기반)**  
   - Bucket B (Mastered): NDCG > 0.7 (2,791 samples) → 추가 학습 효용이 낮아 제외  
   - Bucket A (Edge-of-Competence): 0.1 ≤ NDCG ≤ 0.7 (1,560 samples) → RL의 주 타깃  
   - Bucket 0 (Unsolvable-for-now): NDCG < 0.1 (637 samples) → 데이터 품질/환경 문제 가능성이 높아 별도 처리  
3) **Bucket 0 품질 검수/필터링**  
   - Bucket 0는 “진짜 어려움”과 “라벨/참조 오류”가 섞여 있을 수 있어, 수동 검수로 문제 샘플을 제외  
   - 필터링 결과: 55개(8.6%) 문제 샘플 제외 → 582개 정상 샘플 확보  
4) **Focused Round 1 데이터 생성**  
   - “Hard A”(Bucket A 중 낮은 성능 구간) + “품질 검수된 Bucket 0”를 합쳐, **집중 RL**에 쓸 작은 데이터셋을 구성  
   - 결과: 854 samples (Hard A 272 + Bucket 0 filtered 582)  
5) **Round 1 학습 결과 분석 → Focused Round 2 데이터 생성**  
   - 낮은 점수의 원인을 GT 오류/검색 실패/FG 실패로 분해  
   - GT 오류/불확실 샘플을 제거해 reward를 정화  
   - Round 1에서 score≤0.3 구간을 중심으로 추출(436 samples) 후, GT 검증 기반 필터(82 samples) 제거 → 366 samples  

정리하면, DDAI의 RL 진행은:
- “모든 샘플을 똑같이 RL”이 아니라  
- **(1) 데이터 품질을 확인하고, (2) 학습 가능성이 큰 구간을 고르고, (3) 점점 비싼 탐색 예산을 배정하는 방식**으로 전개되었다.

#### 9.1.2 (추가) Focused RL → SFT → Focused RL: “증류된 성공 궤적으로 다음 라운드 초기화”

Focused RL이 축적한 통합 로그(unified trajectory)는 다음 라운드의 성능 향상을 위해 SFT로 재활용될 수 있다.

- Focused RL 로그에서 “성공률이 낮지만(난제) 성공 rollout이 존재하는 프롬프트”를 우선 선택한다.
- 그 프롬프트의 **score==1** 성공 rollout을 top-k로 추출해 train1 호환 SFT parquet를 만든다.
- Think-only 리라이팅/안전 후처리로 “행동 정책 불변” 조건을 만족하는 데이터셋을 만든다.
- 이 데이터셋으로 짧은 SFT를 수행해 체크포인트를 만든 뒤, 이를 다음 Focused RL 라운드의 초기화로 사용한다.

예시(실험적 구성):
- SFT를 통해 생성된 체크포인트 `checkpoints/sft_qwen2_5_sft_7b_after_focus/global_step_64`를 다음 라운드 초기화에 사용

이 폐루프는 RL을 “정책 자체 최적화”뿐 아니라 **고품질 trajectory 생성→증류→재부트스트랩**의 엔진으로 활용한다는 점에서 실용적이다.

### 9.2 환경(Visual RAG) 정의: 무엇이 RL의 episode인가

하나의 **episode**는 “하나의 질문(프롬프트)에 대해, 답을 찾기 위해 여러 번 도구를 호출하고, 최종 답을 생성하는 과정”이다.

- **초기 상태**: 질문 텍스트(+ 데이터셋 메타데이터: file_name, reference_page, ground_truth 등).  
- **행동(action)**: 태그 기반 문자열 중 하나를 출력
  - `<search>query</search>`: 검색 도구 호출
  - `<bbox>[x1,y1,x2,y2]</bbox>`: 관측된 이미지 일부를 선택해 크롭
  - `<search_complete>true</search_complete>`: 탐색 종료
- **전이(transition)**:
  - `<search>` → 검색 엔진이 이미지 후보를 반환하고, 그 결과가 다음 관측에 반영된다.
  - `<bbox>` → 최근 검색 이미지에 대한 크롭 결과가 다음 관측에 반영된다.
  - `<search_complete>` 또는 `max_turns` 도달 → episode 종료.
- **종료 후 처리(terminal augmentation)**:
  - Searcher가 수집한 이미지(원본+크롭)를 입력으로 **Frozen Generator**가 최종 `<answer>`를 생성한다.
  - 이 `<answer>`는 “정책이 직접 생성한 텍스트”가 아니라, **정책이 모은 증거로부터 나온 결과물**로 취급한다.

이 정의가 중요한 이유:
- 정책은 “답 문장 생성”이 아니라 **정보 획득·검증·종료**라는 일련의 행동을 최적화한다.
- reward는 episode 종료 시점에 주로 결정되므로, 실질적으로 **terminal reward가 강한 환경**이다.

#### 9.2.1 Episode budget(탐색 예산)과 종료 규칙

episode는 “검색을 얼마나 할 것인가?”라는 예산 제약을 갖는다. 본 프로젝트에서는 이를 `max_turns`로 직접 제어했다.

- `max_turns`가 작을수록:
  - 비용이 줄고 학습이 빠르지만, golden rank가 깊은 문제(여러 번 검색해야 하는 문제)는 구조적으로 풀기 어렵다.
- `max_turns`가 클수록:
  - 더 많은 검색/크롭 시도를 허용하여 “찾을 때까지 시도”할 확률이 늘어나지만, step당 비용(도구 호출)이 증가한다.

또한 실제 실행에서는 다음이 중요했다:
- 모델이 끝까지 `<search_complete>`를 내지 않으면, `max_turns`에서 강제 종료되는 상황이 발생할 수 있다.
- 이런 강제 종료는 “종료 정책” 자체를 학습하게 만드는 압력으로 작용한다(무한 검색 루프 억제).

#### 9.2.2 Evidence set(증거 집합) 구성: 검색 이미지 + 크롭

Searcher는 episode 동안 증거를 누적한다:
- `<search>`로 얻은 **retrieved images**
- `<bbox>`로 만든 **cropped images**

Frozen Generator는 이 누적 증거를 입력으로 받아 최종 `<answer>`를 생성한다.  
중요한 개념은 “답변의 품질”이 **증거 집합의 품질**에 의해 크게 좌우된다는 점이다.

논문 서술 관점에서 DDAI의 학습 목표는 다음으로 요약된다:
- Searcher 정책 \(\pi_\theta\)는 “정답을 말하는” 것이 아니라,  
  **Frozen Generator가 정답을 말할 수 있게 만드는 증거 집합을 구성**하도록 학습된다.

#### 9.2.3 Invalid action 처리는 ‘학습 안정성’ 장치다

도구사용 정책에서 흔한 실패는 “행동 포맷 붕괴”다(태그 누락, bbox 포맷 오류 등).  
이때 환경이 완전히 깨지면 reward가 노이즈로 변하고, RL이 불안정해진다.

따라서 DDAI는 다음 철학을 따른다:
- invalid action은 “학습을 멈추게 하는 예외”가 아니라,
- **환경이 모델에게 교정 신호를 주는 형태의 관측**으로 바꿔, 다음 턴에서 수습할 기회를 준다.

### 9.3 Rollout 수집: `n_agent`는 “그룹 비교를 위한 동시 탐색”

훈련은 매 스텝마다 다음 구조를 가진다:

- 한 번에 `train_batch_size = B` 개의 프롬프트를 샘플링
- 각 프롬프트를 `n_agent = N` 번 복제하여 총 `B×N` 개의 rollout trajectory를 생성
- 동일 프롬프트의 N개 trajectory는 GRPO에서 “한 그룹”으로 취급됨

이를 논문식으로 쓰면:
- 프롬프트 \(x\) 에 대해 \(N\)개의 trajectory \(\{\tau_i\}_{i=1}^N\) 를 샘플링하고,
- 그룹 내 상대 비교로 advantage를 계산해 업데이트를 수행한다.

#### 9.3.1 탐색 다양성은 “샘플링 + 그룹 크기”의 곱으로 만들어진다

이 프로젝트는 탐색을 다음 두 축으로 설계했다:
1) **샘플링 기반 다양성**: rollout은 기본적으로 sampling(do_sample)을 사용한다.  
2) **그룹 기반 다양성**: `n_agent`로 같은 문제를 여러 번 시도한다.

또한 `rollout.n`(단일 forward에서 n개 샘플 생성) 대신 `n_agent`를 핵심 레버로 쓴 이유는,
에이전트 환경에서는 중간에 tool 호출이 필요해 **“한 번의 forward로 n개 완성 답변 생성”** 구조와 맞지 않기 때문이다.

### 9.4 보상 설계: Judge 점수와 NDCG의 결합, 그리고 Focused 설정

#### 9.4.1 기본 구성

각 trajectory \(i\)에 대해:
- Frozen Generator가 답변 \(a_i\) 를 생성
- Gemini Judge가 \((q, a_i, y)\) 를 비교하여 \(s_i \in [0,1]\) 점수 산출
- 검색 품질은 NDCG로 \(n_i \in [0,1]\) 산출

최종 reward는 선형 결합(혹은 gate 포함) 형태로 구성한다:

\[
r_i =
\begin{cases}
0, & \text{(format gate fail)} \\\\
w_j \cdot s_i + w_n \cdot n_i \; (+ w_f \cdot f_i), & \text{otherwise}
\end{cases}
\]

여기서 \(w_j, w_n\)은 환경변수로 조절되는 계수(`RM_JUDGE_COEF`, `RM_NDCG_COEF`)이며,  
필요 시 형식 점수/게이트(`RM_FORMAT_COEF`)를 포함할 수 있다.

#### 9.4.2 Focused Round에서의 선택(실제 실행)

Focused Round 1~3에서는 다음을 선택했다:
- `RM_JUDGE_COEF=1.0`, `RM_NDCG_COEF=0.0` (Judge-only)

의도(문서/분석 기반):
- NDCG의 single-golden 한계(같은 문서 다른 슬라이드)로 인해 “실제 성공”이 0점 처리되는 문제가 있었고,
- Focused RL의 목표는 **희소한 성공 trajectory를 더 많이 수확**하여 “정답 경험”을 확장하는 것이었기 때문에,
- 우선 답변 정확성 신호에 집중해 그룹 비교 학습의 밀도를 높였다.

대신, Judge-only가 유발할 수 있는 “근거 없는 정답” 리스크는 별도 분석(환각/불일치 분석, GT 검증)로 모니터링했다.

#### 9.4.3 Judge는 ‘텍스트 기반 의미 채점기’로 사용했다(현 실행 기준)

Focused RL 스크립트들은 `reward_model.reward_manager='rm'`을 사용한다.  
이는 현재 코드에서 `rm_phase2.py`(Gemini **텍스트 기반** LLM-as-Judge)로 연결된다:
- 입력: query / generated_answer(`<answer>`) / reference_answer
- 출력: `{"score": float}` (0.0~1.0 연속값; Structured Output)

즉, “VLM as Judge(이미지 포함)”가 아니라, **답변 텍스트의 의미적 정확성**을 중심으로 학습 신호를 구성했다.  
이 선택은 NDCG의 구조적 한계(단일 golden)와 결합할 때 trade-off가 있으며, 그 분석이 focused round 문서들에 축적되어 있다.

### 9.5 최적화: GRPO advantage + GSPO(보수적 PPO 변형) 업데이트

#### 9.5.1 GRPO(advantage estimator)

동일 프롬프트 그룹에서 reward를 상대화한다. 전형적으로 다음 형태를 취한다:

\[
A_i = \frac{r_i - \mu(r)}{\sigma(r) + \epsilon}
\]

여기서 \(\mu, \sigma\)는 동일 프롬프트에서 생성된 \(N\)개 rollout reward의 평균/표준편차이다.

GRPO가 중요한 이유:
- 외부 API 기반 환경에서 reward 노이즈가 있어도 “같은 질문 내 상대 비교”는 더 안정적일 수 있다.
- Critic 없이도 advantage를 만들 수 있어 복잡도를 줄인다.

#### 9.5.2 GSPO(policy loss)

업데이트는 PPO 계열의 clipped objective를 기반으로 하되, 이 프로젝트는 매우 보수적인 클리핑을 사용한다.

Focused RL 실행 스크립트에서 사용한 대표 설정:
- `actor_rollout_ref.actor.policy_loss_mode="gspo"`
- `clip_ratio_low=3e-4`, `clip_ratio_high=4e-4` (극도로 작은 비대칭 범위)
- `lr=1e-6`, `entropy_coeff=0`, `kl_coef=0`

이 조합의 의미:
- 정책이 한 번에 크게 변하면 tool-use 포맷 붕괴/탐색 루프/보상 해킹이 발생할 수 있다.
- 외부 평가/생성 모델이 포함된 고비용 환경에서는 “큰 폭의 업데이트로 빠르게 바꾸기”보다,  
  **아주 작은 폭으로 안정적으로 누적 개선**하는 것이 더 실용적일 수 있다.

#### 9.5.3 State masking(학습 대상 토큰의 분리)

응답에는 Searcher의 행동 토큰과 Frozen Generator의 `<answer>` 토큰이 함께 존재하지만,
- 학습은 Searcher의 행동/추론 토큰에만 적용하고,
- `<answer>` 영역은 gradient에서 제외한다.

이는 “정답 문장 스타일”이 아니라 “정답을 찾는 행동”을 학습 대상으로 고정시키는 핵심 장치다.

#### 9.5.4 학습 안정성을 위한 ‘보수적 업데이트’ 설계(실행 설정 기반)

Focused Round 실행 설정은 PPO/GRPO 계열 중에서도 특히 보수적이다:
- 작은 learning rate (`1e-6`)
- 아주 작은 clip 범위 (`3e-4`, `4e-4`)
- 엔트로피 보너스 0
- KL 페널티 0

이를 논문에서 해석하면:
- 외부 API 기반 환경은 (1) reward delay가 크고, (2) reward variance가 높으며, (3) tail latency가 길어,
  공격적인 업데이트가 곧바로 “행동 포맷 붕괴/검색 루프/학습 불안정”으로 이어질 수 있다.
- 따라서 DDAI는 “한 번에 크게 개선”보다 **작은 변화의 누적**을 우선시하는 설계를 취했다.

### 9.6 한 스텝의 훈련 루프(논문용 의사코드)

아래는 DDAI Phase 2의 한 training step을 논문에서 설명하기 위한 수준의 의사코드다.

```
for step in training_steps:
  # 1) batch sampling
  prompts = sample_batch(D, size=B)                       # B=train_batch_size
  group_prompts = repeat(prompts, times=N, interleave=True) # N=n_agent

  # 2) rollout in tool-use environment (Searcher policy)
  trajectories = []
  for each prompt instance in group_prompts:
    state = init(prompt)
    for t in 1..max_turns:
      action_text ~ πθ(.|state)  # <search>/<bbox>/<search_complete>
      state = env_step(state, action_text)  # search/crop results become obs
      if done(action_text): break
    trajectories.append(state.history)

  # 3) terminal augmentation: Frozen Generator answers
  for τ in trajectories:
    images = τ.retrieved_images + τ.crops
    τ.answer = FrozenGenerator(q, images)

  # 4) reward computation (Gemini Judge + optional NDCG)
  for τ in trajectories:
    s = GeminiJudge(q, τ.answer, ground_truth)
    n = NDCG(τ.retrieved_basenames, reference_basenames)
    τ.reward = wj*s + wn*n  (plus gating if used)

  # 5) GRPO advantage within each prompt-group
  advantages = group_normalize_rewards({τ.reward} grouped by prompt)

  # 6) GSPO/PPO-style policy update with state masking (exclude <answer>)
  θ ← optimizer_step(θ, objective(πθ, trajectories, advantages))
```

### 9.7 실제 실행(Focused Round 1~3) 설정 요약

아래 표는 “논문 재현성” 관점에서 중요한 실제 실행 설정을 요약한 것이다(스크립트 기반).

| Stage | Script | Train data | Init checkpoint (script) | B | N | B×N | max_turns | Reward weights | Judge model | Frozen model | Frozen max tokens | Streaming reward |
|---|---|---|---|---:|---:|---:|---:|---|---|---|---:|---|
| Phase 1 | `gspo_phase1_gemini_flash.sh` | `data/curriculum_bucket_0.parquet` | `RL_results/gspo_phase1` | 32 | 4 | 128 | 7 | if format pass: 0.1 + 0.9·NDCG else 0 | (none) | (none) | (n/a) | off |
| Focused R1 | `gspo_phase2_focused_round1.sh` | `data/focused_round1.parquet` | `checkpoints/gspo_phase2_gemini_flash_curriculum/global_step_48/actor/huggingface` | 16 | 8 | 128 | 7 | Judge=1.0, NDCG=0.0 | `gemini-3-flash-preview` | `gpt-5-mini-2025-08-07` | 3072 | on |
| Focused R2 | `gspo_phase2_focused_round2.sh` | `data/focused_round2.parquet` | `checkpoints/gspo_phase2_gemini_flash_curriculum_focused_round_1/merged_model` | 8 | 16 | 128 | 15 | Judge=1.0, NDCG=0.0 | `gemini-3-flash-preview` | `gpt-5-mini-2025-08-07` | 3072 | on |
| Focused R3 | `gspo_phase2_focused_round3.sh` | `data/focused_round2.parquet` | `checkpoints/sft_qwen2_5_sft_7b_after_focus/global_step_64` | 32 | 16 | 512 | 15 | Judge=1.0, NDCG=0.0 | `gemini-3-flash-preview` | `gpt-5-mini-2025-08-07` | 5120 | on |

추가로 공통/핵심 하이퍼파라미터(스크립트 기반):
- `algorithm.adv_estimator=grpo`
- `policy_loss_mode=gspo`, `clip_ratio_low=3e-4`, `clip_ratio_high=4e-4`
- `lr=1e-6`, `entropy_coeff=0`, `kl_coef=0`
- `ppo_mini_batch_size=8`, `ppo_micro_batch_size_per_gpu=1`
- `data.max_prompt_length=256`, `data.max_response_length=2048`
- `trainer.total_epochs=1`, `trainer.save_freq=30`, `trainer.resume_mode=auto`

Focused Round의 compute 관점 해석:
- Round 1은 “짧은 episode(7 turns) + 적당한 탐색(8 agents)”로 빠르게 성공 경험을 수확하려는 probing 성격.
- Round 2/3은 “긴 episode(15 turns) + 더 큰 그룹(16 agents)”로, golden rank가 깊거나 FG 실패가 섞인 난제를 더 강하게 탐색하는 성격.

#### 9.7.1 외부 API 기반 환경에서의 실행 파라미터(비용/지연 제어)

논문에서 “실제로 어떻게 돌렸는가”를 재현 가능하게 하려면, 아래 항목이 중요하다:

- **Frozen Generator 설정(예시: Focused R1/R2/R3)**  
  - 모델: `gpt-5-mini-2025-08-07`  
  - reasoning effort: `medium`  
  - `max_output_tokens`: R1/R2=3072, R3=5120  
  - `max_concurrent`: R1/R2=64, R3=512  
  - `total_timeout`: 600s (prompt 단위 제한)  

- **Gemini Judge 설정(예시)**  
  - 모델: `gemini-3-flash-preview`  
  - `max_concurrent_requests`: R1/R2=64, R3=512  
  - streaming reward timeout: 90s (스크립트 값; 시스템 상태에 따라 조정 가능)  

이 수치들은 “알고리즘 파라미터”라기보다, 고비용 환경에서 **학습을 진행 가능하게 만드는 실행 예산**에 가깝다.

### 9.8 Round-to-Round로 ‘어떻게’ 다음 훈련을 설계했는가 (과정 기록)

Focused RL은 단순히 설정만 올리는 것이 아니라, **로그/분석 결과로 다음 라운드의 데이터와 예산을 결정**했다.

대표적인 의사결정 흐름(문서 기반):
1) Round 1 실행 로그/통계에서 “낮은 점수 구간”을 식별  
2) 낮은 점수의 원인이
   - 검색 실패(NDCG 낮음)인지,
   - Generator 병목(FG 실패)인지,
   - Judge/GT 오류인지
   를 분해 분석
3) GT 오류/불확실 샘플을 제거해 보상 신호를 정화
4) 남은 “학습 가능성이 있는 난제”로 `focused_round2.parquet`을 구성
5) 더 큰 탐색 예산(`max_turns`, `n_agent`)을 배정해 재학습

이 과정은 “알고리즘을 바꿔서 성능을 올린다”기보다,  
**고비용 RL 환경에서 ‘학습 가능한 문제’에 컴퓨트를 몰아주는 방법론**을 형성한다.

실제로는 아래 산출물/문서들이 “라운드 전환의 근거”로 사용되었다:
- `docs/data_curation_pipeline.md`: 전체 funnel과 데이터셋 생성 규칙
- `docs/ground_truth_verification_report.md`: GT 오류가 reward를 오염시키는 사례와 필터링 근거
- `docs/ndcg_vs_judge_discrepancy_analysis.md`: NDCG 단일 golden 한계(동일 문서 다른 슬라이드) 분석
- `docs/focused_round1_hallucination_analysis.md`: Judge-only에서의 “근거 없는 정답” 위험 모니터링
- `docs/focused_round2_analysis.md`: “검색 실패 vs FG 병목” 분해를 통한 compute 예산 설계

---

## 10. 남는 질문(리서치/실험 방향)

이 설계는 강력하지만, 다음이 계속 중요한 연구·실험 축이다:

- **근거 없는 정답을 어떻게 억제할 것인가?**  
  - Judge-only 설정은 환각을 강화할 수 있다. “근거 일치도”를 더 직접적으로 측정하는 보상(예: VLM-as-Judge, evidence check)이 필요할 수 있다.

- **NDCG의 single-golden 한계를 어떻게 완화할 것인가?**  
  - 문서 단위 평가, multi-golden 라벨링, semantic-equivalent slide 자동 확장 등.

- **Generator 병목(FG 오류) 구간에서 Searcher RL은 어디까지 의미가 있는가?**  
  - 검색이 성공해도 답변 모델이 실패하면 보상 신호가 막힌다. 이 구간은 데이터/Generator/평가자의 공동 문제로 다뤄야 한다.

- **compute curriculum을 자동화할 수 있는가?**  
  - 라운드 기반(offline) 필터링을 넘어, “어떤 샘플에 어떤 예산을 줄지”를 온라인으로 결정하는 메타-정책(하지만 복잡도/불안정성도 증가).

---

## 11. 참고(더 깊이 읽을 문서)

- 전체 파이프라인/인자 설명: `docs/GSPO_PHASE2_GEMINI_FLASH_GUIDE.md`
- 스트리밍 reward의 발상과 구조: `docs/PHASE6_ASYNC_STREAMING_ARCHITECTURE.md`
- 배치/업데이트 관점 이해: `docs/RL_BATCH_AND_UPDATE_MECHANISM.md`
- Tool async가 왜 별개인지: `docs/BATCH_SIZE_AND_TOOL_ASYNC_ANALYSIS.md`
- 데이터 큐레이션/검증/포커싱: `docs/data_curation_pipeline.md`, `docs/ground_truth_verification_report.md`
- NDCG vs Judge 불일치: `docs/ndcg_vs_judge_discrepancy_analysis.md`
- Focused round 분석: `docs/focused_round1_hallucination_analysis.md`, `docs/focused_round2_analysis.md`
