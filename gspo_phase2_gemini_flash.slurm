#!/bin/bash
# =============================================================================
# GSPO Phase 2 - Gemini Flash VLM as Judge (SLURM Version)
# =============================================================================
#
# gspo_phase2_gemini_flash.sh를 SLURM으로 실행하기 위한 스크립트
#
# 사용법:
#   sbatch gspo_phase2_gemini_flash.slurm
#
# 필수 환경변수 (sbatch 실행 전에 설정 필요):
#   export GEMINI_API_KEY='your-api-key'
#   export DASHSCOPE_API_KEY='your-api-key'  # Frozen Generator용
#
# =============================================================================

# --- Slurm Resource Configuration ---
#SBATCH --job-name=gspo_gemini
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=48:00:00
#SBATCH --output=./logs/sbatch_log/slurm_%j.out
#SBATCH --error=./logs/sbatch_log/slurm_%j.err
#SBATCH --partition=debug

# =============================================================================
# 1. 환경 설정
# =============================================================================
cd $SLURM_SUBMIT_DIR

# Conda 환경 활성화
source /home/isdslab/miniconda3/etc/profile.d/conda.sh
conda activate /home/isdslab/miniconda3/envs/LNS

export PYTHONNOUSERSITE=1
export PYTHONASYNCIODEBUG=1
export PATH=/home/isdslab/miniconda3/envs/LNS/bin:$PATH

# Ray 정리 및 임시 디렉토리 설정 (AF_UNIX 경로 길이 제한 107바이트 대응)
ray stop
export RAY_ADDRESS=''
mkdir -p /tmp/ray_$USER
export TMPDIR=/tmp/ray_$USER
export RAY_TMPDIR=/tmp/ray_$USER
export RAY_memory_usage_threshold=0.995

# WandB 설정
export WANDB_API_KEY='8d955a8fe09693b7a2e983616a79aae912307d79'
export WANDB_PROJECT='gspo_phase2_gemini'

# Attention Backend 설정
export ATTN_BACKEND=native
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True

# =============================================================================
# 2. API 키 확인
# =============================================================================
if [ -z "$GEMINI_API_KEY" ]; then
    echo "=========================================="
    echo "ERROR: GEMINI_API_KEY 환경변수가 설정되지 않았습니다!"
    echo ""
    echo "설정 방법:"
    echo "  export GEMINI_API_KEY='your-api-key'"
    echo "  sbatch gspo_phase2_gemini_flash.slurm"
    echo ""
    echo "API 키 발급: https://aistudio.google.com/app/apikey"
    echo "=========================================="
    exit 1
fi
echo ">>> Gemini API Key 확인 완료"

if [ -z "$DASHSCOPE_API_KEY" ]; then
    echo "WARNING: DASHSCOPE_API_KEY가 설정되지 않았습니다."
    echo "Frozen Generator (Qwen2.5-VL-72B) 사용 불가"
else
    echo ">>> DashScope API Key 확인 완료"
fi
export DASHSCOPE_BASE_URL="${DASHSCOPE_BASE_URL:-https://dashscope-intl.aliyuncs.com/api/v1}"

# =============================================================================
# 3. 디버깅 정보 출력
# =============================================================================
echo "=========================================="
echo "GSPO Phase 2 - Gemini Flash (SLURM)"
echo "=========================================="
echo "Node: $(hostname)"
echo "User: $(whoami)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="

# =============================================================================
# 4. 모델 및 학습 설정
# =============================================================================
ENGINE=${1:-sglang}

# 모델 경로
model_path=/home/eoeldroal/WorkPlace/DDAI_cleaned/RL_results/gspo_phase1

# GPU 설정
n_gpus=4

# 배치 크기 설정
train_batch_size=4
ppo_mini_batch_size=4
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=1
n_agent=2

# 기타 설정
tensor_model_parallel_size=1
val_before_train=False
max_turns=7

# =============================================================================
# 5. Gemini VLM Judge 설정
# =============================================================================
log_path="./logs/gspo_gemini_output.jsonl"
image_base_path="./data/images"
gemini_model="gemini-3-flash-preview"
max_concurrent_requests=50
streaming_reward_enable=True

# =============================================================================
# 6. Frozen Generator 설정
# =============================================================================
frozen_max_concurrent=50
frozen_model="qwen2.5-vl-72b-instruct"
frozen_max_tokens=1024
frozen_max_retries=5
frozen_backoff_base=1.5

# =============================================================================
# 7. Retriever 설정
# =============================================================================
search_url="http://163.239.28.21:5002/search"

# =============================================================================
# 8. 로그 디렉토리 생성
# =============================================================================
mkdir -p ./logs
mkdir -p ./logs/sbatch_log
echo ">>> 로그 경로: $log_path"

# =============================================================================
# 9. 설정 정보 출력
# =============================================================================
echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo "모델: $model_path"
echo "배치 크기: $train_batch_size × $n_agent = $((train_batch_size * n_agent))"
echo "Gemini 모델: $gemini_model"
echo "동시 요청 수: $max_concurrent_requests"
echo "스트리밍 Reward: $streaming_reward_enable"
echo "----------------------------------------"
echo "[Phase 5] Frozen Generator (OpenAI Async)"
echo "  모델: $frozen_model"
echo "  동시 요청 수: $frozen_max_concurrent"
echo "  최대 토큰: $frozen_max_tokens"
echo "=========================================="

# =============================================================================
# 10. 훈련 실행
# =============================================================================
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
    actor_rollout_ref.actor.optim.lr_warmup_steps=12 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    +actor_rollout_ref.actor.optim.name='AdamW' \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.enable_activation_offload=True \
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
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.actor.state_masking=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.n_agent=$n_agent \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    reward_model.reward_manager='rm' \
    +reward_model.log_path=$log_path \
    +reward_model.gemini_model=$gemini_model \
    +reward_model.image_base_path=$image_base_path \
    +reward_model.max_concurrent_requests=$max_concurrent_requests \
    +reward_model.streaming_reward.enable=$streaming_reward_enable \
    +frozen_generator.model=$frozen_model \
    +frozen_generator.max_tokens=$frozen_max_tokens \
    +frozen_generator.max_concurrent=$frozen_max_concurrent \
    +frozen_generator.max_retries=$frozen_max_retries \
    +frozen_generator.backoff_base=$frozen_backoff_base \
    custom_reward_function.path=./lsm_tmp/simple_format_checker.py \
    custom_reward_function.name=simple_format_checker \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb','console'] \
    trainer.project_name=gspo_phase2_gemini \
    trainer.experiment_name=gspo_phase2_gemini_flash \
    trainer.n_gpus_per_node=$n_gpus \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=1000000 \
    trainer.total_epochs=1 \
    trainer.resume_mode=auto \
    retriever.url=$search_url \
    max_turns=$max_turns $@

# =============================================================================
# 11. 완료
# =============================================================================
echo "=========================================="
echo "Training Completed!"
echo "End Time: $(date)"
echo "로그 파일: $log_path"
echo "=========================================="
