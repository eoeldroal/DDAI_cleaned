set -x
#export VLLM_ATTENTION_BACKEND=XFORMERS
ENGINE=${1:-vllm}
mkdir -p ~/sangmin/ray_tmp
export TMPDIR=~/sangmin/ray_tmp
export RAY_TMPDIR=~/sangmin/ray_tmp
export WANDB_API_KEY='8d955a8fe09693b7a2e983616a79aae912307d79'


model_path=./merged_model_grpo_step50
#model_path=Qwen/Qwen2.5-VL-7B-Instruct
#n_gpus=$(nvidia-smi -L | wc -l)
#for test
n_gpus=4

#resume rl
# Resume configuration (override by exporting RESUME_MODE/RESUME_FROM_PATH before running).
# RESUME_MODE=disable
# RESUME_FROM_PATH=""
# extra_resume_args=()
# if [ -n "$RESUME_FROM_PATH" ]; then
#     extra_resume_args+=("trainer.resume_from_path=$RESUME_FROM_PATH")
# fi
#//

train_batch_size=64
#ppo_mini_batch_size=$((4 * n_gpus)) #4= gpu에 올릴 데이터 개수
ppo_mini_batch_size=16 #수정( 4*4)
ppo_micro_batch_size_per_gpu=4
log_prob_micro_batch_size_per_gpu=8
#n_agent=5
n_agent=8 #수정

#oom 해결 위해
#actor_rollout_ref.rollout.free_cache_engine=False \ -> True로 수정
#actor_rollout_ref.actor.optim.name='adamw_8bit' \ 추가
#verl/workers/fsdp_workers.py 수정

tensor_model_parallel_size=1
val_before_train=False
#search_url="http://0.0.0.0:8002/search"
search_url="http://163.239.28.21:5002/search"
# search_url='http://127.0.0.1:5000/search'
rm_url="http://0.0.0.0:8003/eval"
max_turns=10
project_name="vrag"
experiment_name="SFT_w_crop_${n_gpus}_gpus_${max_turns}_maxturns_${n_agent}_ngroups_qwen2_5_vl_7b"

#trainer.save_freq, trainer.test_freq: 25 -> 2로 수정
#trainer.total_epochs=1 \ -> 2로 수정

#문제상황: oom
#수정 사항
#data.max_prompt_length=8192 \  -> data.max_prompt_length=2048
#data.max_response_length=2048 \ -> data.max_response_length=512
#actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \ -> 0.4 : VLLM 엔진이 GPU 메모리의 60%를 미리 점유하도록 설정되어 있습니다. 이 비율을 낮춰서 다른 학습 과정이 사용할 공간을 더 확보


#9/11 수정사항
#verl/trainer/ppo/core_algos.py 여기에 clipping 추가 
#actor_rollout_ref.actor.kl_loss_type=clipping \ 수정
#actor_rollout_ref.actor.kl_clip_coef = 0.2 \ 추가 
#nsys profile -o report 추가 -> nsight
log_path="./logs/grpo_output.json"
# custom_reward_function.path=/path/to/your/my_reward_functions.py \ #추가: reward에서 score담당
# custom_reward_function.name=simple_format_checker \#추가: reward에서 score담당

# #reward_model.log_path="'./logs/my_first_experiment.json'" \ 추가 
# export SCRIPT_DIR=$(pwd)
# NSYS_TEMP_DIR="${SCRIPT_DIR}/tmp"
# TMP_REPORT_PATH="${SCRIPT_DIR}/logs/nsight/temp_report" 
# mkdir -p ${NSYS_TEMP_DIR}
##
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
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=clipping \
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
    trainer.experiment_name=my_run \
    trainer.n_gpus_per_node=$n_gpus \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=1000000 \
    trainer.total_epochs=1 \
    trainer.resume_mode=auto \
    trainer.resume_from_path=./checkpoints/vrag_test/my_run/global_step_10 \
    trainer.val_before_train=$val_before_train \
    retriever.url=$search_url \
    max_turns=$max_turns $@