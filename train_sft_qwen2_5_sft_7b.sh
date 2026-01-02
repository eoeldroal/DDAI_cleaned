
#SFT 데이터셋 개수 약 3000개
#batch size: 64
#epoch:44
#-> 총 188 step

export WANDB_API_KEY='8d955a8fe09693b7a2e983616a79aae912307d79'
export WANDB_PROJECT='after_focus_sft'

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_name="sft_qwen2_5_sft_7b_after_focus"
ckpt_dir="./checkpoints/${run_name}"
log_file="./logs/${run_name}.log"
mkdir -p "$(dirname "$log_file")" "$ckpt_dir"


torchrun --nproc_per_node=8 -m verl.trainer.fsdp_sft_trainer \
  data.train_files="data/after_focus_half_or_less_sft.v2.rewritten.safe.parquet" \
  data.val_files="data/after_focus_half_or_less_sft.v2.rewritten.safe.parquet" \
  model.partial_pretrain=/opt/dlami/nvme/isdslab/HyunBin/DDAI_cleaned/checkpoints/gspo_phase2_gemini_flash_curriculum_focused_round_2/global_step_30/actor/huggingface \
  trainer.project_name=after_focus_sft \
  trainer.experiment_name=${run_name} \
  trainer.default_local_dir=${ckpt_dir} \
  trainer.default_hdfs_dir=null \
  model.trust_remote_code=True \
  model.attention_impl=sdpa \
  data.micro_batch_size_per_gpu=1 \
  data.train_batch_size=8 \
  data.max_length=16384 \
  model.enable_gradient_checkpointing=True \
  trainer.total_epochs=2 \
  optim.lr=5e-6 \
  optim.weight_decay=0.05 | tee "${log_file}"