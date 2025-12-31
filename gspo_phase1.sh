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

export PYTHONNOUSERSITE=1
export PYTHONASYNCIODEBUG=1

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

export UVLOOP_AUTO=0

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Ray 임시 디렉토리 설정 (AF_UNIX 경로 길이 제한 107바이트 대응)
mkdir -p /tmp/ray_$USER
export TMPDIR=/tmp/ray_$USER
export RAY_TMPDIR=/tmp/ray_$USER

# WandB 설정
export WANDB_API_KEY='8d955a8fe09693b7a2e983616a79aae912307d79'
export WANDB_PROJECT='gspo_phase1'

export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True

# =============================================================================
# 2. 모델 및 학습 설정
# =============================================================================
# 첫 번째 인자를 엔진으로 사용할 때만 소비(shift)합니다.
# key=value 형태의 Hydra override가 첫 인자인 경우에는 엔진 기본값(sglang)을 유지합니다.
ENGINE=sglang
if [ $# -ge 1 ] && [[ "$1" != *=* ]]; then
    ENGINE="$1"
    shift
fi

# 모델 경로 (필요 시 환경변수/override로 변경)
model_path=./RL_results/gspo_phase1

# GPU 설정
n_gpus=8

# 배치 크기 설정
# - train_batch_size: 원본 프롬프트 수
# - n_agent: 프롬프트당 생성할 응답 수
train_batch_size=32
ppo_mini_batch_size=8
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=1
n_agent=4

# 기타 설정
tensor_model_parallel_size=1
max_turns=7

# Retriever 설정
search_url="http://163.239.28.21:5002/search"

# =============================================================================
# 3. 로그 디렉토리 생성
# =============================================================================
mkdir -p ./logs

# =============================================================================
# 4. Ray 메모리 설정
# =============================================================================
export RAY_memory_usage_threshold=0.995

# =============================================================================
# 5. 훈련 실행 (Phase 1: gate reward = 0.1*format + 0.9*ndcg)
# =============================================================================
echo "=========================================="
echo "GSPO Phase 1 Training - Format + NDCG (Gate)"
echo "=========================================="
echo "모델: $model_path"
echo "배치 크기: $train_batch_size × $n_agent = $((train_batch_size * n_agent))"
echo "----------------------------------------"
echo "Reward: if format pass -> 0.1 + 0.9*NDCG else 0"
echo "Generation: phase1 (no frozen generator)"
echo "Unified log: $UNIFIED_LOG_PATH"
echo "=========================================="

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=./data/curriculum_bucket_0.parquet \
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
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
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
    reward_model.reward_manager='rm_phase1' \
    custom_reward_function.path=./lsm_tmp/simple_format_checker.py \
    custom_reward_function.name=simple_format_checker \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb','console'] \
    trainer.project_name=gspo_phase1 \
    trainer.experiment_name=gspo_phase1 \
    trainer.n_gpus_per_node=$n_gpus \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=1000000 \
    trainer.total_epochs=1 \
    trainer.resume_mode=auto \
    trainer.val_before_train=false \
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

