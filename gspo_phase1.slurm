#!/bin/bash

# --- Slurm Resource Configuration ---
#SBATCH --job-name=vrag_grpo          # 작업 이름 (원하는 대로 수정)
#SBATCH --nodes=1                     # 노드 수 (srun --nodes=1)
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32            # CPU 코어 수 (srun --cpus-per-task=32)
#SBATCH --mem=480G                    # 메모리 (srun --mem=480G)
#SBATCH --time=48:00:00               # 제한 시간 (srun --time=4:00:00)
#SBATCH --output=./logs/sbatch_log/slurm_%j.out  # 표준 출력 로그 저장 위치 (%j는 Job ID)
#SBATCH --error=./logs/sbatch_log/slurm_%j.err   # 에러 로그 저장 위치
#SBATCH --partition=debug               # (중요) 사용하는 파티션 이름이 있다면 주석 해제 후 수정 (예: gpu, batch 등)

# --- Environment Setup ---
# 1. 작업 디렉토리로 이동 (필요하다면 경로 수정)
# Slurm 스크립트를 제출하는 위치를 기준으로 하려면 아래 줄은 주석 처리해도 됩니다.
cd $SLURM_SUBMIT_DIR 

# 2. Conda 환경 활성화
# (이전 로그에서 'LNS' 환경을 사용하는 것으로 보였습니다. 경로가 다르다면 수정해주세요)
source /home/isdslab/miniconda3/etc/profile.d/conda.sh
conda activate /home/isdslab/miniconda3/envs/LNS

export PYTHONNOUSERSITE=1
export PATH=/home/isdslab/miniconda3/envs/LNS/bin:$PATH

# API Key는 이전에 공유주신 파일의 키를 사용했습니다.
export WANDB_API_KEY='8d955a8fe09693b7a2e983616a79aae912307d79'
export WANDB_PROJECT='vrag_test'      # 프로젝트 이름 (원하는 대로 수정 가능)

# 3. Ray 관련 포트 충돌 방지 및 정리 (Ray 사용 시 권장)
ray stop
export RAY_ADDRESS=''

# --- Debugging Info ---
echo "Node: $(hostname)"
echo "User: $(whoami)"
echo "Start Time: $(date)"

# --- Execution ---
# 기존에 주신 sh 파일을 실행합니다.
# 실행 권한이 없으면 미리 'chmod +x train_grpo_qwen2_5_vl_7b.sh'를 해주세요.
bash train_gspo_qwen2_5_vl_7b.sh

echo "End Time: $(date)"