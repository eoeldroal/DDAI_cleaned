'''
=== GSPO 기본 설정 ===
actor_rollout_ref.actor.use_kl_loss=False \ -> 현재 상태는 kl을 사용 x
actor_rollout_ref.actor.kl_loss_coef=0.0 \
actor_rollout_ref.actor.kl_loss_type=clipping \
actor_rollout_ref.actor.clip_ratio_low=3e-4 \
actor_rollout_ref.actor.clip_ratio_high=4e-4 \
actor.policy_loss_mode="gspo"
actor_rollout_ref.actor.entropy_coeff=0 \

needed adding
actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
algorithm.use_kl_in_reward=False \

'''
export PYTHONNOUSERSITE=1
set -x
ENGINE=${1:-vllm}
mkdir -p ~/sangmin/ray_tmp
export TMPDIR=~/sangmin/ray_tmp
export RAY_TMPDIR=~/sangmin/ray_tmp
export WANDB_API_KEY='8d955a8fe09693b7a2e983616a79aae912307d79'


model_path=./RL_results/merged_gspo_phase1
n_gpus=4

train_batch_size=64
ppo_mini_batch_size=16 #수정( 4*4)
ppo_micro_batch_size_per_gpu=4
log_prob_micro_batch_size_per_gpu=8
n_agent=8 #수정

tensor_model_parallel_size=1
val_before_train=False
search_url="http://163.239.28.21:5002/search"
rm_url="http://0.0.0.0:8003/eval"
max_turns=7
#project_name="vrag"
#experiment_name="SFT_w_crop_${n_gpus}_gpus_${max_turns}_maxturns_${n_agent}_ngroups_qwen2_5_vl_7b"


log_path="./logs/gspo_output.json"

export RAY_memory_usage_threshold=0.995
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=./data/rag/slidevqa_train_crop.parquet \
    data.val_files=./data/rag/overall_test_crop.parquet \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=4096 \
    data.max_response_length=2048  \
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
    reward_model.rm_url=$rm_url \
    +reward_model.log_path=$log_path \
    custom_reward_function.path=./lsm_tmp/simple_format_checker.py \
    custom_reward_function.name=simple_format_checker \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb','console'] \
    trainer.project_name=vrag_test \
    trainer.experiment_name=gspo_phase2 \
    trainer.n_gpus_per_node=$n_gpus \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=1000000 \
    trainer.total_epochs=1 \
    trainer.resume_mode=auto \
    trainer.resume_from_path=/data/daedong/gspo_phase2/global_step_70 \
    trainer.val_before_train=$val_before_train \
    retriever.url=$search_url \
    max_turns=$max_turns $@