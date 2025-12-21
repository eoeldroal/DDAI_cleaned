#!/bin/bash
#SBATCH --job-name=VRAG-SFT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --gpus-per-node=3
#SBATCH --partition=debug
#SBATCH -o logs/sft_output_%j.log

source /home/isdslab/miniconda3/etc/profile.d/conda.sh
conda activate /home/isdslab/miniconda3/envs/LNS

export PYTHONNOUSERSITE=1
export PATH=/home/isdslab/miniconda3/envs/vrag_test/bin:$PATH

export WANDB_API_KEY='8d955a8fe09693b7a2e983616a79aae912307d79'
export WANDB_PROJECT='vrag_sft'      


torchrun --nproc_per_node=$SLURM_GPUS_PER_NODE -m verl.trainer.fsdp_sft_trainer \
    data.train_files=lsm_tmp/results/sft_dataset/train1.parquet \
    data.val_files=lsm_tmp/results/sft_dataset/val1.parquet \
    model.partial_pretrain=Qwen/Qwen2.5-VL-7B-Instruct \
    trainer.default_local_dir=outputs/sft-test-filtered_test \
    model.enable_gradient_checkpointing=true \
    model.trust_remote_code=true \
    data.micro_batch_size_per_gpu=2 \
    data.train_batch_size=24 \
    data.max_length=1024 \
    data.use_remove_padding=true \
    model.enable_gradient_checkpointing=true \
    trainer.total_epochs=6 \
    optim.warmup_steps_ratio=0.12 \
    optim.lr=1e-5 \
    optim.weight_decay=0.05 | tee logs/sft_test_500.log
