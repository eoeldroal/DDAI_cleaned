#!/usr/bin/env bash
set -euo pipefail

# Pipeline: (1) curate from unified_trajectory.jsonl -> (2) GPT filter -> (3) GPT rewrite <think> only -> (4) safety post-pass
#
# Notes
# - Step (3) is editing-only: it must not change tool/action sequence; it only rewrites <think>...</think>.
# - Step (4) drops rows where original <think> contains action tags and reverts newly introduced image-claim language.
# - This script does NOT run SFT training; run training separately after reviewing ${SAFE_OUT}.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

UNIFIED_LOG="${UNIFIED_LOG:-logs/focused2/unified_trajectory.jsonl}"
CURATED_TRAIN1="${CURATED_TRAIN1:-/tmp/focused2_curated_half_or_less_train1.parquet}"

FILTERED_PARQUET="${FILTERED_PARQUET:-logs/focused2/sft_candidates_half_or_less_filtered.parquet}"
FILTER_REPORT="${FILTER_REPORT:-logs/focused2/query_filter_report_half_or_less.jsonl}"

REWRITE_OUT="${REWRITE_OUT:-data/after_focus_half_or_less_sft.v2.rewritten.parquet}"
REWRITE_REPORT="${REWRITE_REPORT:-logs/focused2/think_rewrite_report_half_or_less_v2.jsonl}"
REWRITE_CACHE="${REWRITE_CACHE:-logs/focused2/think_rewrite_cache_half_or_less_v2.jsonl}"

SAFE_OUT="${SAFE_OUT:-data/after_focus_half_or_less_sft.v2.rewritten.safe.parquet}"

# Curation knobs (16 rollouts per prompt in focused2; "half or less correct" => success_rate <= 0.5)
MIN_SCORE="${MIN_SCORE:-1.0}"
MIN_SUCCESS_RATE="${MIN_SUCCESS_RATE:-1e-7}"
MAX_SUCCESS_RATE="${MAX_SUCCESS_RATE:-0.5}"
TOP_K="${TOP_K:-4}"

# GPT filter knobs
FILTER_CONCURRENCY="${FILTER_CONCURRENCY:-100}"
FILTER_SEED="${FILTER_SEED:-0}"

# GPT rewrite knobs
REWRITE_MODEL="${REWRITE_MODEL:-gpt-5.2-2025-12-11}"
REWRITE_REASONING_EFFORT="${REWRITE_REASONING_EFFORT:-high}"
REWRITE_CONCURRENCY="${REWRITE_CONCURRENCY:-200}"
REWRITE_MAX_RETRIES="${REWRITE_MAX_RETRIES:-5}"
REWRITE_MAX_UNITS_PER_CALL="${REWRITE_MAX_UNITS_PER_CALL:-3}"
REWRITE_TIMEOUT_SECONDS="${REWRITE_TIMEOUT_SECONDS:-600}"
REWRITE_MAX_OUTPUT_TOKENS="${REWRITE_MAX_OUTPUT_TOKENS:-6000}"

echo "[1/5] Curate from logs -> train1 parquet"
python scripts/extract_trajectories.py \
  --input "${UNIFIED_LOG}" \
  --output "${CURATED_TRAIN1}" \
  --export-train1-parquet \
  --min-score "${MIN_SCORE}" \
  --min-success-rate "${MIN_SUCCESS_RATE}" \
  --max-success-rate "${MAX_SUCCESS_RATE}" \
  --top-k "${TOP_K}"

echo "[2/5] GPT filter queries/samples"
python scripts/filter_sft_queries_gpt.py \
  --input "${CURATED_TRAIN1}" \
  --output "${FILTERED_PARQUET}" \
  --report "${FILTER_REPORT}" \
  --limit 0 \
  --sample random \
  --seed "${FILTER_SEED}" \
  --concurrency "${FILTER_CONCURRENCY}"

echo "[3/5] GPT rewrite (<think> only, editing task)"
python scripts/rewrite_sft_think_gpt.py \
  --input "${FILTERED_PARQUET}" \
  --output "${REWRITE_OUT}" \
  --report "${REWRITE_REPORT}" \
  --cache "${REWRITE_CACHE}" \
  --model "${REWRITE_MODEL}" \
  --reasoning-effort "${REWRITE_REASONING_EFFORT}" \
  --max-output-tokens "${REWRITE_MAX_OUTPUT_TOKENS}" \
  --concurrency "${REWRITE_CONCURRENCY}" \
  --max-retries "${REWRITE_MAX_RETRIES}" \
  --max-units-per-call "${REWRITE_MAX_UNITS_PER_CALL}" \
  --request-timeout-seconds "${REWRITE_TIMEOUT_SECONDS}" \
  --on-error drop \
  --revert-new-image-claims \
  --revert-action-tags-in-think

echo "[4/5] Safety post-pass (drop original action-tag-in-think rows; revert newly introduced image-claim language)"
python scripts/clean_rewritten_sft_dataset.py \
  --original "${FILTERED_PARQUET}" \
  --rewritten "${REWRITE_OUT}" \
  --output "${SAFE_OUT}" \
  --drop-if-action-tag-in-think \
  --revert-new-image-claims

echo "Done. Final dataset:"
echo "  ${SAFE_OUT}"
