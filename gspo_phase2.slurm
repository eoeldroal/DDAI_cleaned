#!/bin/bash

# --- Slurm Resource Configuration ---
#SBATCH --job-name=vrag_gspo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=32:00:00
#SBATCH --output=./logs/sbatch_log/slurm_%j.out
#SBATCH --error=./logs/sbatch_log/slurm_%j.err
#SBATCH --partition=debug

# --- Environment Setup ---
source /home/isdslab/miniconda3/etc/profile.d/conda.sh
conda activate /home/isdslab/miniconda3/envs/LNS

export PYTHONNOUSERSITE=1
export PATH=/home/isdslab/miniconda3/envs/LNS/bin:$PATH
export WANDB_API_KEY='8d955a8fe09693b7a2e983616a79aae912307d79'
export WANDB_PROJECT='gspo_phase2'

# Ray 정리
ray stop
export RAY_ADDRESS=''

# ==========================================
# 1. 경로 설정 (여기를 잘 봐주세요!)
# ==========================================
# 서버를 실행할 위치 (부모 폴더)
SERVER_ROOT="/home/isdslab/sangmin/VRAG_test/"

# 훈련 스크립트가 있는 위치 (코드 폴더)
# 만약 .sh 파일이 VRAG_lsm 안에 있다면 아래 경로를 쓰세요.
TRAIN_DIR="/home/isdslab/sangmin/VRAG_test/VRAG_lsm/"


# ==========================================
# 2. API 서버 실행 (위치 이동: VRAG_test)
# ==========================================
echo ">>> [Step 1] Moving to SERVER ROOT: $SERVER_ROOT"
cd $SERVER_ROOT

# 혹시 모를 경로 에러 방지용 PYTHONPATH 추가
export PYTHONPATH=$SERVER_ROOT:$PYTHONPATH

SERVER_LOG="./logs/sbatch_log/server_debug_${SLURM_JOB_ID}.log"
mkdir -p ./logs/sbatch_log 

echo ">>> Starting Uvicorn Server..."
# 이제 여기서 실행하므로 'VRAG_lsm' 패키지를 찾을 수 있음
python -m uvicorn VRAG_lsm.model_eval.model_eval_api:app --host 0.0.0.0 --port 8003 > $SERVER_LOG 2>&1 &
SERVER_PID=$!
echo ">>> Server PID: $SERVER_PID"


# ==========================================
# 3. 서버 대기 (포트 8003 열릴 때까지)
# ==========================================
echo ">>> [Step 2] Waiting for Server to be ready..."
MAX_RETRIES=300
COUNT=0
IS_READY=0

while [ $COUNT -lt $MAX_RETRIES ]; do
    if ! kill -0 $SERVER_PID > /dev/null 2>&1; then
        echo "!!! CRITICAL: Server process died !!!"
        tail -n 10 $SERVER_LOG
        exit 1
    fi

    if curl -s http://127.0.0.1:8003/docs > /dev/null; then
        echo ">>> Port 8003 is OPEN! Server is ready."
        IS_READY=1
        break
    fi

    sleep 2
    COUNT=$((COUNT+2))
done

if [ $IS_READY -eq 0 ]; then
    echo "!!! Timeout !!!"
    kill $SERVER_PID
    exit 1
fi


# ==========================================
# 4. 훈련 스크립트 실행 (위치 이동: VRAG_lsm)
# ==========================================
echo ">>> [Step 3] Moving to TRAIN DIR: $TRAIN_DIR"
cd $TRAIN_DIR

echo ">>> Starting Training Script..."
# 이제 훈련 코드가 있는 폴더로 왔으니 스크립트 이름만 부르면 됩니다.
# (만약 스크립트 파일이 다른 곳에 있다면 경로를 수정해주세요)
bash train_gspo_qwen2_5_vl_7b.sh


# ==========================================
# 5. 종료
# ==========================================
echo ">>> Training Finished. Killing Server..."
kill $SERVER_PID