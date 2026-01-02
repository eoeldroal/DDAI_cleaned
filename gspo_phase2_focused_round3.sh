#!/bin/bash

set -x  # 디버그 모드 (실행 명령어 출력)

# =============================================================================
# 1. 환경 설정
# =============================================================================
# .env 파일 로드 (루트 폴더)
if [ -f .env ]; then
    echo ">>> .env 파일 로드 중..."
    export $(grep -v '^#' .env | xargs)
fi

# OpenAI API 키 확인 (Frozen Generator용)
if [ -z "$OPENAI_API_KEY" ]; then
    echo "=========================================="
    echo "ERROR: OPENAI_API_KEY 환경변수가 설정되지 않았습니다!"
    echo ""
    echo ".env에 OPENAI_API_KEY를 추가하거나 아래와 같이 설정하세요:"
    echo "  export OPENAI_API_KEY='your-openai-api-key'"
    echo "=========================================="
    exit 1
fi
echo ">>> OpenAI API Key 확인 완료"

export PYTHONNOUSERSITE=1
export PYTHONASYNCIODEBUG=0

# export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync,max_split_size_mb:256 -> RuntimeError: cudaMallocAsync does not yet support shareIpcHandle. If you need it, please file an issue describing your use case.

export GEMINI_DEBUG_VERBOSE=0
export FROZEN_DEBUG_VERBOSE=0
export NDCG_DEBUG=1  # <--- 추가됨
export NDCG_DEBUG_PATH=./logs/ndcg_debug.jsonl
# API 입/출력(detail) 로그: unified log에만 남기고 콘솔 출력은 끈다
export FLASH_RM_LOG=1
export GEMINI_DETAIL_LOG=1
export FROZEN_DETAIL_LOG=1
# 콘솔에는 출력하지 않고, unified log에만 남긴다
export GEMINI_DETAIL_CONSOLE=0
export FROZEN_DETAIL_CONSOLE=0
export NDCG_DEBUG_LOG_ALL=1
export SEARCH_DEBUG=1
export VERL_PPO_LOGGING_LEVEL=DEBUG

# =============================================================================
# API latency stats logging (Gemini / Frozen(OpenAI-compatible))
# =============================================================================
# - Summary stats are emitted to unified JSONL and (optionally) console.
# - Per-call events can be enabled if you need raw distributions.
export API_LATENCY_ENABLE=1
export API_LATENCY_CONSOLE=1
export API_LATENCY_PER_CALL=0
export API_LATENCY_SUMMARY_EVERY_N=200
export API_LATENCY_SUMMARY_EVERY_S=30
export API_LATENCY_SAMPLE_SIZE=5000

# =============================================================================
# Unified trajectory logging (single JSONL)
# =============================================================================
export UNIFIED_LOG_ENABLE=1
export UNIFIED_LOG_PATH=./logs/focused3/unified_trajectory.jsonl
export UNIFIED_LOG_CLIENT_BATCH_SIZE=200
export UNIFIED_LOG_CLIENT_FLUSH_INTERVAL_S=1.0
export UNIFIED_LOG_WRITER_FLUSH_EVERY_N=2000
export UNIFIED_LOG_WRITER_FLUSH_INTERVAL_S=2.0

# =============================================================================
# Reward Coefficients (Focused RL 설정)
# =============================================================================
# Round 1: Judge Score 100%, NDCG 0% (정답 자체에 집중)
export RM_JUDGE_COEF=1.0
export RM_NDCG_COEF=0.0

# =============================================================================
# Frozen Generator Reasoning Effort
# =============================================================================
# Options: minimal, low, medium, high (default: minimal)
# Focused Round 1에서는 더 깊은 추론을 위해 "medium" 권장
export FROZEN_REASONING_EFFORT="medium"

export SEARCH_BATCH_SIZE=32
export SEARCH_MAX_WORKERS=4
export SEARCH_TIMEOUT=120

export UVLOOP_AUTO=0

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Ray 임시 디렉토리 설정 (AF_UNIX 경로 길이 제한 107바이트 대응)
mkdir -p /tmp/ray_$USER
export TMPDIR=/tmp/ray_$USER
export RAY_TMPDIR=/tmp/ray_$USER

# WandB 설정
export WANDB_API_KEY='8d955a8fe09693b7a2e983616a79aae912307d79'
export WANDB_PROJECT='gspo_phase2_gemini'

# =============================================================================
# Flash Attention 비활성화 설정
# =============================================================================
# vLLM 사용 시 PyTorch 내장 SDPA 사용 (flash_attn 패키지 불필요)
# export VLLM_ATTENTION_BACKEND=TORCH_SDPA
# HuggingFace transformers에서 flash_attn 사용 비활성화
# export ATTN_BACKEND=native

export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True

# =============================================================================
# 2. Gemini API 키 설정 (필수!)
# =============================================================================
# 방법 1: 직접 설정 (보안상 비추천, 테스트용)
# export GEMINI_API_KEY="your-api-key-here"

# 방법 2: 환경변수에서 로드 (추천)
if [ -z "$GEMINI_API_KEY" ]; then
    echo "=========================================="
    echo "ERROR: GEMINI_API_KEY 환경변수가 설정되지 않았습니다!"
    echo ""
    echo "설정 방법:"
    echo "  export GEMINI_API_KEY='your-api-key'"
    echo ""
    echo "API 키 발급: https://aistudio.google.com/app/apikey"
    echo "=========================================="
    exit 1
fi

echo ">>> Gemini API Key 확인 완료"

# =============================================================================
# 2-1. DashScope API 키 설정 (Frozen Generator용)
# =============================================================================
# 터미널에서 이미 설정된 경우 그대로 사용
if [ -z "$DASHSCOPE_API_KEY" ]; then
    echo "WARNING: DASHSCOPE_API_KEY가 설정되지 않았습니다."
    echo "Frozen Generator (Qwen2.5-VL-72B) 사용 불가"
else
    echo ">>> DashScope API Key 확인 완료"
fi
export DASHSCOPE_BASE_URL="${DASHSCOPE_BASE_URL:-https://dashscope-intl.aliyuncs.com/api/v1}"

# =============================================================================
# 3. 모델 및 학습 설정
# =============================================================================
# 첫 번째 인자를 엔진으로 사용할 때만 소비(shift)합니다.
# key=value 형태의 Hydra override가 첫 인자인 경우에는 엔진 기본값(sglang)을 유지합니다.
ENGINE=sglang
if [ $# -ge 1 ] && [[ "$1" != *=* ]]; then
    ENGINE="$1"
    shift
fi

# 모델 경로
model_path=./checkpoints/sft_qwen2_5_sft_7b_after_focus/global_step_64

# GPU 설정
n_gpus=8

# 배치 크기 설정
# - train_batch_size: 원본 프롬프트 수
# - n_agent: 프롬프트당 생성할 응답 수
# - 실제 배치 크기 = train_batch_size × n_agent
train_batch_size=16
ppo_mini_batch_size=8
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=1
n_agent=16

# 기타 설정
tensor_model_parallel_size=1
val_before_train=False
max_turns=15

# =============================================================================
# 4. Gemini VLM Judge 설정
# =============================================================================
# 로그 경로 (JSONL 형식)
log_path="./logs/focused3/gspo_gemini_output.jsonl"

# 이미지 기본 경로 (검색된 이미지 및 정답 이미지)
image_base_path="./data/images"

# Gemini 모델 설정
gemini_model="gemini-3-flash-preview"

# 동시 API 요청 수 (rate limit 대응)
max_concurrent_requests=512

# 스트리밍 Reward 모드 (프롬프트 완료 시 즉시 Reward 계산 시작)
# - True: Generation과 Reward를 파이프라인으로 병렬 처리 (약 13% 성능 향상)
# - False: 기존 배치 모드 (모든 Generation 완료 후 Reward 계산)
streaming_reward_enable=True
streaming_reward_timeout=90

# =============================================================================
# 4-1. Frozen Generator 설정 (Qwen2.5-VL-72B-Instruct via DashScope)
# =============================================================================
# [Phase 5] OpenAI 호환 비동기 API 사용으로 ~10x 성능 향상
# - 동시 API 요청 수 (rate limit 대응, 기본값: 50)
frozen_max_concurrent=512

# Frozen Generator 모델명 (DashScope에서 제공하는 모델)
# frozen_model="qwen2.5-vl-72b-instruct"
frozen_model="gpt-5-mini-2025-08-07"

# 최대 토큰 수 (답변 길이)
frozen_max_tokens=5120

# 재시도 설정
frozen_max_retries=8
frozen_backoff_base=2
frozen_total_timeout=600
frozen_async_wrapper_timeout=600

# =============================================================================
# 5. Retriever 설정
# =============================================================================
search_url="http://163.239.28.21:5002/search"

# =============================================================================
# 6. 로그 디렉토리 생성
# =============================================================================
mkdir -p ./logs/focused3
echo ">>> 로그 경로: $log_path"

# =============================================================================
# 7. Ray 메모리 설정
# =============================================================================
export RAY_memory_usage_threshold=0.995

# =============================================================================
# 8. 훈련 실행
# =============================================================================
echo "=========================================="
echo "GSPO Phase 2 Training - Gemini Flash VLM Judge"
echo "=========================================="
echo "모델: $model_path"
echo "배치 크기: $train_batch_size × $n_agent = $((train_batch_size * n_agent))"
echo "Gemini 모델: $gemini_model"
echo "동시 요청 수: $max_concurrent_requests"
echo "스트리밍 Reward: $streaming_reward_enable"
echo "스트리밍 Reward timeout(s): $streaming_reward_timeout"
echo "Reward Coefs: Judge=$RM_JUDGE_COEF, NDCG=$RM_NDCG_COEF"
echo "Frozen Reasoning Effort: $FROZEN_REASONING_EFFORT"
echo "----------------------------------------"
echo "[Phase 5] Frozen Generator (OpenAI Async)"
echo "  모델: $frozen_model"
echo "  동시 요청 수: $frozen_max_concurrent"
echo "  최대 토큰: $frozen_max_tokens"
echo "  총 timeout(s): $frozen_total_timeout"
echo "  async wrapper timeout(s): $frozen_async_wrapper_timeout"
echo "=========================================="

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=./data/focused_round2.parquet \
    data.val_files=./data/rag/overall_test_crop.parquet \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=256 \
    data.max_response_length=2048 \
    data.image_key=images \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    +actor_rollout_ref.actor.optim.name='AdamW' \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.enable_activation_offload=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.strategy="fsdp" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=clipping \
    actor_rollout_ref.actor.clip_ratio_low=3e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    actor_rollout_ref.actor.policy_loss_mode="gspo" \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.state_masking=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=bf16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype=bf16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    +actor_rollout_ref.model.attn_implementation=sdpa \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.n_agent=$n_agent \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager='rm' \
    +reward_model.log_path=$log_path \
    +reward_model.gemini_model=$gemini_model \
    +reward_model.image_base_path=$image_base_path \
    +reward_model.max_concurrent_requests=$max_concurrent_requests \
    +reward_model.streaming_reward.enable=$streaming_reward_enable \
    +reward_model.streaming_reward.timeout=$streaming_reward_timeout \
    +frozen_generator.model=$frozen_model \
    +frozen_generator.max_tokens=$frozen_max_tokens \
    +frozen_generator.max_concurrent=$frozen_max_concurrent \
    +frozen_generator.max_retries=$frozen_max_retries \
    +frozen_generator.backoff_base=$frozen_backoff_base \
    +frozen_generator.total_timeout=$frozen_total_timeout \
    +frozen_generator.async_wrapper_timeout=$frozen_async_wrapper_timeout \
    custom_reward_function.path=./lsm_tmp/simple_format_checker.py \
    custom_reward_function.name=simple_format_checker \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb','console'] \
    trainer.project_name=gspo_phase2_gemini \
    trainer.experiment_name=gspo_phase2_gemini_flash_curriculum_focused_round_3 \
    trainer.n_gpus_per_node=$n_gpus \
    trainer.nnodes=1 \
    trainer.save_freq=4 \
    trainer.test_freq=1000000 \
    trainer.total_epochs=1 \
    trainer.resume_mode=auto \
    retriever.url=$search_url \
    max_turns=$max_turns "$@"

exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "=========================================="
    echo "Training Failed! (exit code: $exit_code)"
    echo "=========================================="
    exit $exit_code
fi
# =============================================================================
# 9. 완료
# =============================================================================
echo "=========================================="
echo "Training Completed!"
echo "로그 파일: $log_path"
echo "=========================================="
