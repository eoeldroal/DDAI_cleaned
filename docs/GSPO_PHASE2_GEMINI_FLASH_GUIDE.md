# GSPO Phase 2 Gemini Flash 스크립트 완벽 가이드

> **목적**: `gspo_phase2_gemini_flash.sh` 스크립트의 모든 인자와 동작 원리를 상세히 설명합니다.
>
> **작성일**: 2024-12-25

---

## 목차

1. [개요](#1-개요)
2. [환경 설정](#2-환경-설정)
3. [알고리즘 관련 인자](#3-알고리즘-관련-인자)
4. [데이터 관련 인자](#4-데이터-관련-인자)
5. [Actor/Rollout/Ref 인자](#5-actorrolloutref-인자)
6. [GSPO 알고리즘 상세](#6-gspo-알고리즘-상세)
7. [Reward Model 인자](#7-reward-model-인자)
8. [Trainer 인자](#8-trainer-인자)
9. [Retriever 및 기타 인자](#9-retriever-및-기타-인자)
10. [n_agent vs rollout.n 차이점](#10-n_agent-vs-rolloutn-차이점)
11. [GPU 메모리 설정 가이드](#11-gpu-메모리-설정-가이드)
12. [전체 학습 파이프라인](#12-전체-학습-파이프라인)
13. [트러블슈팅](#13-트러블슈팅)

---

## 1. 개요

### 1.1 스크립트 목적

`gspo_phase2_gemini_flash.sh`는 DDAI Visual RAG 시스템의 **Phase 2 강화학습** 스크립트입니다.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GSPO Phase 2 학습 개요                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 1 (SFT)                    Phase 2 (RL) ← 현재 스크립트               │
│  ├─ Supervised Fine-Tuning        ├─ GSPO 강화학습                          │
│  ├─ 기본 검색 능력 학습             ├─ VLM as Judge로 정밀 평가               │
│  └─ 출력: merged_gspo_phase1      └─ 검색 품질 최적화                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 기존 방식과의 차이점

| 항목 | 기존 (gspo_phase2.slurm) | 새 방식 (gemini_flash) |
|------|-------------------------|------------------------|
| **Judge 모델** | Qwen2.5-72B (LLM) | Gemini 3 Flash (VLM) |
| **평가 입력** | `<answer>` 텍스트만 | 전체 response + 이미지 |
| **점수 형식** | 이진 (True/False) | 연속 (0.0~1.0) |
| **점수 공식** | 0.2×Judge + 0.8×NDCG | 0.8×VLM + 0.2×NDCG |
| **서버 의존성** | FastAPI 서버 필요 | Gemini SDK 직접 호출 |
| **성능** | 기준 | ~9배 향상 (비동기 배치) |

### 1.3 필수 조건

```bash
# 1. Gemini API 키 설정
export GEMINI_API_KEY='your-api-key'

# 2. Phase 1 모델 다운로드
hf download SOGANG-ISDS/DDAI_RL_PHASE1_GSPO \
    --local-dir ./RL_results/merged_gspo_phase1

# 3. 데이터셋 준비
# ./data/rag/slidevqa_train_crop.parquet
# ./data/rag/overall_test_crop.parquet

# 4. 의존성 설치
pip install google-generativeai
```

---

## 2. 환경 설정

### 2.1 GPU 설정

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7  # GPU 4~7 사용 (4개)
```

### 2.2 Ray 설정

```bash
mkdir -p ~/eoeldroal/ray_tmp
export TMPDIR=~/eoeldroal/ray_tmp
export RAY_TMPDIR=~/eoeldroal/ray_tmp
export RAY_memory_usage_threshold=0.995  # 메모리 99.5%까지 사용
```

### 2.3 WandB 설정

```bash
export WANDB_API_KEY='your-wandb-key'
export WANDB_PROJECT='gspo_phase2_gemini'
```

> ⚠️ **보안 주의**: API 키를 스크립트에 하드코딩하지 마세요. 환경 변수로 관리하세요.

---

## 3. 알고리즘 관련 인자

### 3.1 `algorithm.adv_estimator=grpo`

**역할**: Advantage Estimator 선택

**코드 위치**: `verl/trainer/ppo/core_algos.py:119-162`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Advantage Estimator 비교                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  GAE (Generalized Advantage Estimation):                                    │
│  ├─ Critic 모델 필요 (Value Function 학습)                                   │
│  ├─ 토큰별 Advantage 계산: A_t = δ_t + γλδ_{t+1} + ...                       │
│  └─ 메모리 사용량 높음                                                        │
│                                                                             │
│  GRPO (Group Relative Policy Optimization): ← 선택됨                        │
│  ├─ Critic 불필요 (메모리 절약)                                               │
│  ├─ 그룹 내 상대 비교: A_i = (R_i - mean(R)) / std(R)                        │
│  └─ n_agent개 응답을 그룹으로 묶어 비교                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**GRPO Advantage 계산 코드**:

```python
# verl/trainer/ppo/core_algos.py:119-162
def compute_grpo_outcome_advantage(token_level_rewards, eos_mask, index, epsilon=1e-6):
    """
    같은 프롬프트(index)의 응답들을 그룹화하여 상대적 품질 점수 계산
    """
    scores = token_level_rewards.sum(dim=-1)  # 시퀀스별 총 보상

    # 그룹별 평균/표준편차 계산
    for idx in id2score:
        id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
        id2std[idx] = torch.std(torch.tensor([id2score[idx]]))

    # 정규화: (점수 - 그룹평균) / 그룹표준편차
    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
```

### 3.2 `algorithm.kl_ctrl.kl_coef=0.0`

**역할**: KL Divergence 페널티 비활성화

**의미**: GSPO의 클리핑 메커니즘이 정책 divergence를 제어하므로 추가 KL 페널티 불필요

---

## 4. 데이터 관련 인자

### 4.1 데이터 파일

| 인자 | 값 | 설명 |
|------|-----|------|
| `data.train_files` | `./data/rag/slidevqa_train_crop.parquet` | 학습 데이터 |
| `data.val_files` | `./data/rag/overall_test_crop.parquet` | 검증 데이터 |
| `data.image_key` | `images` | Parquet 내 이미지 컬럼명 |

### 4.2 배치 크기

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           배치 크기 계산                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  train_batch_size = 64      (원본 프롬프트 수)                                │
│  n_agent = 8                (프롬프트당 응답 수)                               │
│  ─────────────────────────────────────────────                               │
│  실제 배치 크기 = 64 × 8 = 512 샘플                                           │
│                                                                             │
│  PPO 업데이트:                                                                │
│  ppo_mini_batch_size = 16   → 512 / 16 = 32 mini-batch                      │
│  ppo_micro_batch_size_per_gpu = 4                                            │
│  → 16 / (4 GPU × 4) = 1 accumulation step                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 시퀀스 길이

| 인자 | 값 | 설명 |
|------|-----|------|
| `data.max_prompt_length` | 4096 | 입력 프롬프트 최대 토큰 |
| `data.max_response_length` | 2048 | 생성 응답 최대 토큰 |

---

## 5. Actor/Rollout/Ref 인자

### 5.1 모델 설정

| 인자 | 값 | 설명 |
|------|-----|------|
| `actor_rollout_ref.model.path` | `./RL_results/merged_gspo_phase1` | Searcher 모델 경로 |
| `actor_rollout_ref.model.use_remove_padding` | `True` | 패딩 제거 최적화 |
| `actor_rollout_ref.model.enable_gradient_checkpointing` | `True` | 메모리 절약 (속도↓) |

### 5.2 Optimizer 설정

```bash
actor_rollout_ref.actor.optim.lr=1e-6                    # 학습률 (매우 작음)
actor_rollout_ref.actor.optim.lr_warmup_steps=12         # 워밍업 스텝
actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1  # 워밍업 비율
+actor_rollout_ref.actor.optim.name='AdamW'              # Optimizer
```

> **왜 1e-6인가?**: Phase 1에서 이미 SFT된 모델을 미세 조정하므로, 기존 지식을 보존하면서 점진적으로 개선합니다.

### 5.3 GSPO 핵심 설정

```bash
actor_rollout_ref.actor.policy_loss_mode="gspo"   # GSPO 손실 함수 활성화
actor_rollout_ref.actor.clip_ratio_low=3e-4       # 하한 클리핑 (0.9997)
actor_rollout_ref.actor.clip_ratio_high=4e-4      # 상한 클리핑 (1.0004)
```

**클리핑 범위 의미**:
```
[1.0 - 0.0003, 1.0 + 0.0004] = [0.9997, 1.0004]

→ 매우 좁은 범위로 정책 업데이트를 극히 보수적으로 제한
→ Phase 1의 지식을 최대한 보존하면서 미세 조정
```

### 5.4 State Masking

```bash
actor_rollout_ref.actor.state_masking=True
```

**역할**: Frozen Generator의 `<answer>` 출력을 gradient 계산에서 제외

**코드 위치**: `verl/trainer/ppo/ray_trainer.py:1104-1192`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        State Masking 동작                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  전체 Response:                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ <think>...</think>        ← Searcher (학습됨 ✓)                      │   │
│  │ <search>...</search>      ← Searcher (학습됨 ✓)                      │   │
│  │ <bbox>...</bbox>          ← Searcher (학습됨 ✓)                      │   │
│  │ <search_complete>         ← Searcher (학습됨 ✓)                      │   │
│  │ <answer>...</answer>      ← Frozen Generator (마스킹됨 ✗)            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  loss_mask가 <answer> 영역을 0으로 설정하여 gradient 제외                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.5 KL Loss 설정

```bash
actor_rollout_ref.actor.use_kl_loss=False    # KL Loss 비활성화
actor_rollout_ref.actor.kl_loss_coef=0.0     # 계수 0
actor_rollout_ref.actor.kl_loss_type=clipping
actor_rollout_ref.actor.entropy_coeff=0      # Entropy 보너스 비활성화
```

### 5.6 FSDP 설정

```bash
actor_rollout_ref.actor.fsdp_config.param_offload=false      # 파라미터 CPU 오프로드 안 함
actor_rollout_ref.actor.fsdp_config.optimizer_offload=false  # Optimizer CPU 오프로드 안 함
actor_rollout_ref.ref.fsdp_config.param_offload=True         # Ref Policy는 오프로드
```

> **왜 Actor는 오프로드 안 하나?**: A100-80GB에서 7B 모델은 충분히 GPU에 적재 가능. 오프로드는 오히려 성능 저하.

### 5.7 Rollout (vLLM) 설정

```bash
actor_rollout_ref.rollout.name=vllm                          # vLLM 추론 엔진
actor_rollout_ref.rollout.tensor_model_parallel_size=1       # TP 비활성화 (7B는 단일 GPU)
actor_rollout_ref.rollout.gpu_memory_utilization=0.5         # GPU 메모리 50%
actor_rollout_ref.rollout.free_cache_engine=True             # 생성 후 캐시 해제
actor_rollout_ref.rollout.enable_chunked_prefill=False       # Chunked prefill 비활성화
actor_rollout_ref.rollout.enforce_eager=False                # CUDA graph 사용
actor_rollout_ref.rollout.n=1                                # vLLM 레벨 샘플링 수
actor_rollout_ref.rollout.n_agent=8                          # 배치 복제 수
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8
```

---

## 6. GSPO 알고리즘 상세

### 6.1 GSPO vs Vanilla PPO

**코드 위치**: `verl/trainer/ppo/core_algos.py:375-445`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GSPO vs Vanilla PPO 비교                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Importance Ratio 계산                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Vanilla PPO: 토큰별                                                   │   │
│  │   ratio[t] = exp(log_prob[t] - old_log_prob[t])                      │   │
│  │                                                                       │   │
│  │ GSPO: 시퀀스 수준 평균 후 broadcast                                    │   │
│  │   avg_kl_seq = mean(log_prob - old_log_prob)  # 시퀀스 평균           │   │
│  │   ratio[t] = exp(log_prob[t] - old_log_prob[t] + avg_kl_seq)         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  2. 클리핑                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Vanilla PPO: 대칭                                                     │   │
│  │   clamp(ratio, 1-ε, 1+ε)  예: [0.8, 1.2]                             │   │
│  │                                                                       │   │
│  │ GSPO: 비대칭                                                          │   │
│  │   clamp(ratio, 1-ε_low, 1+ε_high)  예: [0.9997, 1.0004]              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  3. 손실 집계                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Vanilla PPO: token-mean                                               │   │
│  │   loss = mean(all_token_losses)                                       │   │
│  │                                                                       │   │
│  │ GSPO: seq-mean-token-mean                                             │   │
│  │   seq_loss[s] = mean(token_losses[s])  # 시퀀스별 평균                 │   │
│  │   loss = mean(seq_losses)               # 시퀀스 평균                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 GSPO 핵심 코드

```python
# verl/trainer/ppo/core_algos.py:396-418
def compute_policy_loss_gspo(...):
    # 1. 토큰별 log-ratio
    negative_approx_kl = log_prob - old_log_prob  # (B, T)

    # 2. 시퀀스 수준 평균 log-ratio
    seq_lengths = torch.sum(eos_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * eos_mask, dim=-1) / seq_lengths

    # 3. 시퀀스 수준 ratio를 토큰에 broadcast
    log_seq_importance_ratio = (
        log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    )
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    # 4. 비대칭 클리핑
    pg_losses2 = -advantages * torch.clamp(
        seq_importance_ratio,
        1.0 - clip_ratio_low,   # 0.9997
        1.0 + clip_ratio_high,  # 1.0004
    )
```

### 6.3 왜 시퀀스 수준 Ratio인가?

긴 시퀀스에서 개별 토큰의 importance ratio가 크게 다르면 gradient가 불안정해집니다. 시퀀스 수준에서 평균을 취함으로써:

1. **분산 감소**: 토큰별 ratio 변동을 완화
2. **일관성 유지**: 전체 시퀀스에 동일한 가중치 적용
3. **학습 안정성**: 극단적인 ratio로 인한 불안정 방지

---

## 7. Reward Model 인자

### 7.1 기본 설정

```bash
reward_model.reward_manager='rm'                              # RMManager 사용
+reward_model.gemini_model=gemini-3-flash-preview            # Gemini 모델
+reward_model.log_path=./logs/gspo_gemini_output.jsonl       # 로그 경로
+reward_model.image_base_path=./data/images                  # 이미지 경로
+reward_model.max_concurrent_requests=50                     # 동시 API 요청 수
```

### 7.2 Streaming Reward Mode

```bash
+reward_model.streaming_reward.enable=True
```

**코드 위치**: `verl/workers/reward_manager/rm_phase2.py:775-876`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Streaming Reward Mode                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  일반 배치 모드:                                                              │
│  [Generation 512개] ──────────────► [Reward 512개 일괄 계산]                 │
│       끝날 때까지 대기                    그 후에 시작                         │
│                                                                             │
│  스트리밍 모드 (약 13% 성능 향상):                                             │
│  [Generation]                                                               │
│      P1 완료 ──┐                                                             │
│      P2 완료 ──┼──► [Worker 스레드 4개]                                      │
│      ...      ──┤    ├─ P1 Reward 계산 (병렬)                                │
│      P64 완료 ──┘    ├─ P2 Reward 계산 (병렬)                                │
│                      └─ ...                                                 │
│                                                                             │
│  → 프롬프트 완료 시점에 즉시 Reward 계산 시작                                   │
│  → Generation과 Reward 계산이 파이프라인으로 병렬 처리                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.3 점수 계산 공식

```python
# verl/workers/reward_manager/rm_phase2.py:476-488
final_score = 0.8 * vlm_score + 0.2 * ndcg_value
```

| 구성 요소 | 비중 | 설명 |
|----------|------|------|
| `vlm_score` | 80% | Gemini VLM Judge 평가 (0.0~1.0) |
| `ndcg_value` | 20% | 검색 이미지 vs 정답 이미지 직접 비교 |

### 7.4 Custom Reward Function

```bash
custom_reward_function.path=./lsm_tmp/simple_format_checker.py
custom_reward_function.name=simple_format_checker
```

**역할**: 형식 검사 (올바른 태그 사용 여부 등)

---

## 8. Trainer 인자

### 8.1 GPU 및 노드 설정

```bash
trainer.n_gpus_per_node=4     # 노드당 GPU 수
trainer.nnodes=1              # 총 노드 수
# 총 GPU 수 = 4 × 1 = 4개
```

### 8.2 로깅 설정

```bash
trainer.logger=['wandb','console']
trainer.project_name=gspo_phase2_gemini
trainer.experiment_name=gspo_phase2_gemini_flash
```

### 8.3 체크포인트 및 검증

```bash
trainer.save_freq=30          # 30 step마다 저장
trainer.test_freq=1000000     # 검증 빈도 (사실상 비활성화)
trainer.resume_mode=auto      # 자동 재개
trainer.total_epochs=1        # 총 에폭 수
trainer.critic_warmup=0       # Critic 워밍업 (GRPO는 Critic 없음)
```

---

## 9. Retriever 및 기타 인자

### 9.1 Search Engine 설정

```bash
retriever.url=http://163.239.28.21:5002/search
```

**Search API 엔드포인트**: Searcher가 `<search>` 쿼리를 생성하면 이 URL로 POST 요청

### 9.2 Max Turns

```bash
max_turns=7
```

**의미**: Searcher가 최대 7턴까지 `<search>` 액션 수행 가능

```python
# vrag_agent/generation.py:1005-1009
for step in range(self.config.max_turns):  # 0~6
    if not active_mask.sum():
        break

    is_last_turn = step == last_turn_idx  # step==6이면 강제 종료
```

---

## 10. n_agent vs rollout.n 차이점

### 10.1 핵심 차이

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       n_agent vs rollout.n                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  rollout.n (vLLM 레벨):                                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  프롬프트 1 ──► vLLM ──► [응답1, 응답2, ..., 응답n] (단일 forward)      │ │
│  │                                                                        │ │
│  │  ✓ 장점: 효율적 (KV 캐시 공유)                                          │ │
│  │  ✗ 문제: 중간에 Search Engine 호출 불가                                 │ │
│  │         → Agent 시스템과 호환 안 됨!                                    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  n_agent (배치 복제 레벨):                                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  배치 [P1, P2, ..., P64]                                               │ │
│  │       ↓ repeat_deepcopy(n_agent=8)                                     │ │
│  │  확장 배치 [P1_1, P1_2, ..., P1_8, P2_1, ..., P64_8]  (512개)          │ │
│  │       ↓                                                                │ │
│  │  각 샘플이 독립적인 Agent Loop 실행:                                    │ │
│  │  ├─ P1_1: <search>query1</search> → 이미지A → <bbox>... → 완료         │ │
│  │  ├─ P1_2: <search>query2</search> → 이미지B → <search>... → 완료       │ │
│  │  └─ ...각자 다른 검색 경로!                                             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.2 rollout.n > 1이 동작하지 않는 이유

**기술적 원인**: `generation.py`의 상태 배열이 입력 배치 크기로 초기화됨

```python
# generation.py:1002-1003
batch_size = gen_batch.batch['input_ids'].shape[0]  # 64
self.retrievaled_images = [[] for _ in range(batch_size)]  # 64개!
self.cropped_images = [[] for _ in range(batch_size)]      # 64개!

# rollout.n=8이면 vLLM이 512개 출력 → 상태 배열 크기 불일치!
# self.retrievaled_images[idx] 접근 시 idx >= 64면 IndexError
```

**n_agent가 동작하는 이유**: 배치 복제가 `generation.py` **외부**에서 일어나므로, 입력 시점에 이미 512개

---

## 11. GPU 메모리 설정 가이드

### 11.1 현재 설정 분석

```bash
# 현재 설정
gpu_memory_utilization=0.5   # 40GB (80GB의 50%)
param_offload=false
optimizer_offload=false
```

### 11.2 다른 스크립트와 비교

| 스크립트 | gpu_util | param_offload | optimizer_offload |
|---------|----------|---------------|-------------------|
| train_gspo_qwen2_5_vl_7b.sh | 0.7 | True | True |
| train_grpo_qwen2_5_vl_7b.sh | 0.7 | True | True |
| **gspo_phase2_gemini_flash.sh** | **0.5** | **false** | **false** |

### 11.3 권장 설정

```bash
# A100-80GB × 4 기준 권장 설정
gpu_memory_utilization=0.65   # 52GB로 상향

# 메모리 계산:
# vLLM KV 캐시: 80 × 0.65 = 52GB
# 모델 (FSDP 분산): 14GB / 4 = 3.5GB/GPU
# Gradients: 3.5GB/GPU
# Optimizer: 7GB/GPU
# Activations: ~10GB/GPU
# 여유: ~4GB/GPU
```

> **참고**: `free_cache_engine=True`로 Generation과 Training이 시간적으로 분리되므로 0.65~0.7도 안전합니다.

---

## 12. 전체 학습 파이프라인

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GSPO Phase 2 학습 파이프라인                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. 데이터 로딩                                                               │
│     └─ slidevqa_train_crop.parquet에서 64개 프롬프트 로드                     │
│                                                                             │
│  2. 배치 복제 (ray_trainer.py:902)                                           │
│     └─ repeat_deepcopy(n_agent=8) → 64×8=512개 샘플                          │
│                                                                             │
│  3. Generation (vrag_agent/generation.py)                                   │
│     ├─ Searcher (7B): <think>, <search>, <bbox> 생성 (max 7턴)               │
│     ├─ Search Engine API 호출하여 이미지 검색                                  │
│     └─ Frozen Generator (72B, DashScope): <answer> 생성                      │
│                                                                             │
│  4. Reward 계산 (rm_phase2.py)                                               │
│     ├─ Gemini 3 Flash VLM Judge                                              │
│     │   ├─ 입력: 전체 response + 검색 이미지 + 정답 이미지                      │
│     │   └─ 출력: answer_accuracy, visual_grounding, reasoning_consistency   │
│     ├─ NDCG: 검색 이미지 vs 정답 이미지 직접 비교                              │
│     └─ 최종: 0.8 × VLM + 0.2 × NDCG                                          │
│                                                                             │
│  5. GRPO Advantage 계산 (core_algos.py:119-162)                              │
│     └─ 같은 프롬프트의 8개 응답을 비교하여 상대적 품질 점수화                    │
│        A_i = (R_i - mean(R)) / std(R)                                        │
│                                                                             │
│  6. GSPO Policy Update (core_algos.py:375-445)                               │
│     ├─ 시퀀스 수준 importance ratio 계산                                      │
│     ├─ 비대칭 클리핑 [0.9997, 1.0004]                                         │
│     ├─ State Masking으로 Searcher 토큰만 학습                                  │
│     └─ Gradient 업데이트 (lr=1e-6)                                            │
│                                                                             │
│  7. 반복                                                                      │
│     └─ 전체 데이터셋 1 에폭 반복, 30 step마다 체크포인트 저장                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 13. 트러블슈팅

### 13.1 OOM (Out of Memory)

```bash
# 해결 방법 1: gpu_memory_utilization 낮추기
gpu_memory_utilization=0.4

# 해결 방법 2: CPU 오프로드 활성화
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True

# 해결 방법 3: 배치 크기 줄이기
train_batch_size=32
ppo_mini_batch_size=8
```

### 13.2 Gemini API Rate Limit

```bash
# 동시 요청 수 줄이기
+reward_model.max_concurrent_requests=30

# 또는 rm_phase2.py에서 백오프 설정 조정
```

### 13.3 Search Engine 연결 실패

```bash
# Search Engine 상태 확인
curl http://163.239.28.21:5002/search -X POST -H "Content-Type: application/json" \
    -d '[{"query": "test", "id": "1"}]'
```

### 13.4 Phase 1 모델 없음

```bash
# 다운로드
hf download SOGANG-ISDS/DDAI_RL_PHASE1_GSPO \
    --local-dir ./RL_results/merged_gspo_phase1
```

---

## 부록: 파일 경로 요약

| 역할 | 파일 경로 |
|------|-----------|
| **메인 스크립트** | `gspo_phase2_gemini_flash.sh` |
| **PPO Trainer** | `verl/trainer/ppo/ray_trainer.py` |
| **Core Algorithms** | `verl/trainer/ppo/core_algos.py` |
| **Generation Manager** | `vrag_agent/generation.py` |
| **Reward Manager (VLM)** | `verl/workers/reward_manager/rm_phase2.py` |
| **vLLM Rollout** | `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` |
| **FSDP Workers** | `verl/workers/fsdp_workers.py` |
| **Config Template** | `verl/trainer/config/ppo_trainer.yaml` |

---

*마지막 업데이트: 2024-12-25*
