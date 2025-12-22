#!/bin/bash
# =============================================================================
# GSPO Phase 2 - Gemini Flash VLM as Judge
# =============================================================================
#
# 기존 gspo_phase2.slurm과의 차이점:
# 1. FastAPI 서버 불필요 (Gemini SDK 직접 호출)
# 2. GEMINI_API_KEY 환경변수 필요
# 3. 비동기 배치 처리로 ~9배 성능 향상
# 4. 점수 공식: 0.8 * VLM + 0.2 * NDCG
#
# 사용법:
#   bash gspo_phase2_gemini_flash.sh
#
# 필수 환경변수:
#   GEMINI_API_KEY: Gemini API 키 (https://aistudio.google.com/app/apikey)
#
# =============================================================================

set -x  # 디버그 모드 (실행 명령어 출력)

# =============================================================================
# 1. 환경 설정
# =============================================================================
export PYTHONNOUSERSITE=1

# Ray 임시 디렉토리 설정
mkdir -p ~/sangmin/ray_tmp
export TMPDIR=~/sangmin/ray_tmp
export RAY_TMPDIR=~/sangmin/ray_tmp

# WandB 설정
export WANDB_API_KEY='8d955a8fe09693b7a2e983616a79aae912307d79'
export WANDB_PROJECT='gspo_phase2_gemini'

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
# 3. 모델 및 학습 설정
# =============================================================================
ENGINE=${1:-vllm}

# 모델 경로
model_path=./RL_results/merged_gspo_phase1

# GPU 설정
n_gpus=4

# 배치 크기 설정
# - train_batch_size: 원본 프롬프트 수
# - n_agent: 프롬프트당 생성할 응답 수
# - 실제 배치 크기 = train_batch_size × n_agent = 64 × 8 = 512
train_batch_size=64
ppo_mini_batch_size=16
ppo_micro_batch_size_per_gpu=4
log_prob_micro_batch_size_per_gpu=8
n_agent=8

# 기타 설정
tensor_model_parallel_size=1
val_before_train=False
max_turns=7

# =============================================================================
# 4. Gemini VLM Judge 설정
# =============================================================================
# 로그 경로 (JSONL 형식)
log_path="./logs/gspo_gemini_output.jsonl"

# 이미지 기본 경로 (검색된 이미지 및 정답 이미지)
image_base_path="./data/images"

# Gemini 모델 설정
gemini_model="gemini-3-flash-preview"

# 동시 API 요청 수 (rate limit 대응)
max_concurrent_requests=50

# 스트리밍 Reward 모드 (프롬프트 완료 시 즉시 Reward 계산 시작)
# - True: Generation과 Reward를 파이프라인으로 병렬 처리 (약 13% 성능 향상)
# - False: 기존 배치 모드 (모든 Generation 완료 후 Reward 계산)
streaming_reward_enable=True

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
echo "GSPO Phase 2 Training - Gemini Flash VLM Judge"
echo "=========================================="
echo "모델: $model_path"
echo "배치 크기: $train_batch_size × $n_agent = $((train_batch_size * n_agent))"
echo "Gemini 모델: $gemini_model"
echo "동시 요청 수: $max_concurrent_requests"
echo "스트리밍 Reward: $streaming_reward_enable"
echo "=========================================="

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=./data/rag/slidevqa_train_crop.parquet \
    data.val_files=./data/rag/overall_test_crop.parquet \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.image_key=images \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    +actor_rollout_ref.actor.optim.name='adamw_8bit' \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
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
    +reward_model.streaming_reward.enable=$streaming_reward_enable \
    custom_reward_function.path=./lsm_tmp/simple_format_checker.py \
    custom_reward_function.name=simple_format_checker \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb','console'] \
    trainer.project_name=gspo_phase2_gemini \
    trainer.experiment_name=gspo_phase2_gemini_flash \
    trainer.n_gpus_per_node=$n_gpus \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=1000000 \
    trainer.total_epochs=1 \
    trainer.resume_mode=auto \
    retriever.url=$search_url \
    max_turns=$max_turns $@

# =============================================================================
# 9. 완료
# =============================================================================
echo "=========================================="
echo "Training Completed!"
echo "로그 파일: $log_path"
echo "=========================================="
