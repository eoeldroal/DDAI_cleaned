#!/usr/bin/env bash
set -euo pipefail

# Success-only SFT curation pipeline (Focused2/3/3v2 unified logs)
#
# This pipeline:
#  1) Extracts success-only candidates from unified_trajectory.jsonl
#     - group filter: judge==1 success_rate <= 0.5 (i.e., <=8/16)
#     - rollout filter: judge==1 AND ndcg>0
#  2) Runs GPT pass/fail vetting on all candidates
#  3) Caps to max 4 per prompt-group and outputs train1-style parquet
#
# Usage:
#   bash scripts/pipeline_success_only_sft.sh \
#     --stage focused3_v2 \
#     --out-prefix after_focus_success_only \
#     --model gpt-5.2 --concurrency 200

STAGE=""
OUT_PREFIX="sft_success_only"
MODEL="gpt-5.2"
CONCURRENCY="200"
WORKERS=""
MAX_PER_GROUP="4"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --out-prefix) OUT_PREFIX="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --max-per-group) MAX_PER_GROUP="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${STAGE}" ]]; then
  echo "Missing --stage (focused2|focused3|focused3_v2)" >&2
  exit 1
fi

LOG_DIR="logs/${STAGE}"
IN_LOG="${LOG_DIR}/unified_trajectory.jsonl"
if [[ ! -f "${IN_LOG}" ]]; then
  echo "Missing unified log: ${IN_LOG}" >&2
  exit 1
fi

OUT_DIR="${LOG_DIR}/sft_success_only"
mkdir -p "${OUT_DIR}"

CAND="${OUT_DIR}/${OUT_PREFIX}.candidates.parquet"
VETTED="${OUT_DIR}/${OUT_PREFIX}.vetted.parquet"
REPORT="${OUT_DIR}/${OUT_PREFIX}.vet_report.jsonl"
CACHE="${OUT_DIR}/${OUT_PREFIX}.vet_cache.jsonl"
FINAL="data/${OUT_PREFIX}.train1.parquet"

echo "[1/3] Extract candidates -> ${CAND}"
python scripts/extract_trajectories.py \
  --input "${IN_LOG}" \
  --output "${CAND}" \
  --export-train1-parquet \
  --export-extra-metrics \
  --drop-system-errors \
  --min-judge-score 1.0 \
  --min-ndcg 1e-9 \
  --group-success-metric judge \
  --group-success-threshold 1.0 \
  --max-success-rate 0.5

echo "[2/3] GPT vetting -> ${VETTED}"
WORKERS_FLAG=()
if [[ -n "${WORKERS}" ]]; then
  WORKERS_FLAG=(--workers "${WORKERS}")
fi

python scripts/vet_sft_rollouts_gpt.py \
  --input "${CAND}" \
  --output "${VETTED}" \
  --report "${REPORT}" \
  --cache "${CACHE}" \
  --model "${MODEL}" \
  --concurrency "${CONCURRENCY}" \
  "${WORKERS_FLAG[@]}"

echo "[3/3] Finalize (cap per group=${MAX_PER_GROUP}) -> ${FINAL}"
python scripts/finalize_success_only_sft_dataset.py \
  --input "${VETTED}" \
  --output "${FINAL}" \
  --max-per-group "${MAX_PER_GROUP}"

echo "Done."
echo "- candidates: ${CAND}"
echo "- vetted:     ${VETTED}"
echo "- final:      ${FINAL}"
