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
export PYTHONASYNCIODEBUG=1

export GEMINI_DEBUG_VERBOSE=2
export FROZEN_DEBUG_VERBOSE=2
export NDCG_DEBUG=1
export NDCG_DEBUG_PATH=./logs/ndcg_debug.jsonl
export FLASH_RM_LOG=1
export NDCG_DEBUG_LOG_ALL=1
export SEARCH_DEBUG=1
export SEARCH_DEBUG_LOG_ALL=1
export SEARCH_DEBUG_MAX_LINES=1000000000

# =============================================================================
# Unified trajectory logging (single JSONL; append)
# =============================================================================
export UNIFIED_LOG_ENABLE=1
export UNIFIED_LOG_PATH=./logs/unified_trajectory.jsonl
export UNIFIED_LOG_CLIENT_BATCH_SIZE=200
export UNIFIED_LOG_CLIENT_FLUSH_INTERVAL_S=1.0
export UNIFIED_LOG_WRITER_FLUSH_EVERY_N=2000
export UNIFIED_LOG_WRITER_FLUSH_INTERVAL_S=2.0

# =============================================================================
# Reward Coefficients (Unified pre-focused)
# - Hard gate: if format fails -> final score = 0
# =============================================================================
export RM_FORMAT_COEF=0.1
export RM_NDCG_COEF=0.5
export RM_JUDGE_COEF=0.4

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
export WANDB_PROJECT='gspo_phase12_unified'

# Flash Attention 비활성화 설정
export ATTN_BACKEND=native

export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True

# =============================================================================
# 2. Gemini API 키 설정 (필수!)
# =============================================================================
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
if [ -z "$DASHSCOPE_API_KEY" ]; then
    echo "WARNING: DASHSCOPE_API_KEY가 설정되지 않았습니다."
    echo "Frozen Generator (DashScope) 사용 불가"
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
model_path=./RL_results/gspo_phase1

# GPU 설정
n_gpus=8

# 배치 크기 설정
train_batch_size=8
ppo_mini_batch_size=8
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=1
n_agent=16

# 기타 설정
tensor_model_parallel_size=1
max_turns=7

# =============================================================================
# 4. Gemini VLM Judge 설정
# =============================================================================
log_path="./logs/gspo_phase12_unified_gemini_output.jsonl"
image_base_path="./data/images"
gemini_model="gemini-3-flash-preview"
max_concurrent_requests=64

# =============================================================================
# 4-1. Frozen Generator 설정
# =============================================================================
frozen_max_concurrent=64
frozen_model="gpt-5-mini-2025-08-07"
frozen_max_tokens=2048
frozen_max_retries=10
frozen_backoff_base=1.5
frozen_total_timeout=120
frozen_async_wrapper_timeout=120

# =============================================================================
# 5. Retriever 설정
# =============================================================================
search_url="http://163.239.28.21:5002/search"

# =============================================================================
# 6. 로그 디렉토리 생성
# =============================================================================
mkdir -p ./logs
echo ">>> 로그 경로: $log_path"

# =============================================================================
# 7. Ray 메모리 설정
# =============================================================================
export RAY_memory_usage_threshold=0.995

# =============================================================================
# 8. 훈련 실행
# =============================================================================
echo "=========================================="
echo "GSPO Unified Phase (pre-focused) - Gemini Judge + NDCG + Format(Gate)"
echo "=========================================="
echo "모델: $model_path"
echo "배치 크기: $train_batch_size × $n_agent = $((train_batch_size * n_agent))"
echo "Reward Coefs (Gate on format fail): Format=$RM_FORMAT_COEF, NDCG=$RM_NDCG_COEF, Judge=$RM_JUDGE_COEF"
echo "Epochs: 2"
echo "----------------------------------------"
echo "[Frozen Generator]"
echo "  모델: $frozen_model"
echo "  동시 요청 수: $frozen_max_concurrent"
echo "  최대 토큰: $frozen_max_tokens"
echo "=========================================="

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=./data/curriculum_bucket_a.parquet \
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
    +actor_rollout_ref.model.enable_activation_offload=True \
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
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.actor.state_masking=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.n_agent=$n_agent \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager='rm' \
    +reward_model.log_path=$log_path \
    +reward_model.gemini_model=$gemini_model \
    +reward_model.image_base_path=$image_base_path \
    +reward_model.max_concurrent_requests=$max_concurrent_requests \
    +reward_model.streaming_reward.enable=False \
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
    trainer.project_name=gspo_phase12_unified \
    trainer.experiment_name=gspo_phase12_unified \
    trainer.n_gpus_per_node=$n_gpus \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=1000000 \
    trainer.total_epochs=2 \
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

echo "=========================================="
echo "Training Completed!"
echo "Unified log: $UNIFIED_LOG_PATH"
echo "=========================================="

