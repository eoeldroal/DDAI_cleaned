#!/usr/bin/env bash
set -euo pipefail

# Run success-only SFT curation for multiple stages in one shot, optionally merging outputs.
#
# Default stages: focused2 focused3 focused3_v2
#
# Examples:
#   bash scripts/run_success_only_sft_all.sh --model gpt-5.2 --concurrency 200
#   bash scripts/run_success_only_sft_all.sh --stages focused2,focused3_v2 --merge --out-prefix after_focus_success_only
#
# Outputs:
#   data/${OUT_PREFIX}_${stage}.train1.parquet  (per-stage)
#   data/${OUT_PREFIX}.train1.parquet          (merged, if --merge)
#
# NOTE: This will make network/API calls during GPT vetting.

STAGES_CSV="focused2,focused3,focused3_v2"
OUT_PREFIX="sft_success_only"
MODEL="gpt-5.2"
CONCURRENCY="200"
WORKERS=""
MAX_PER_GROUP="4"
MERGE="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stages) STAGES_CSV="$2"; shift 2 ;;
    --out-prefix) OUT_PREFIX="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --max-per-group) MAX_PER_GROUP="$2"; shift 2 ;;
    --merge) MERGE="1"; shift 1 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

IFS=',' read -r -a STAGES <<< "${STAGES_CSV}"

per_stage_outputs=()

for stage in "${STAGES[@]}"; do
  stage="$(echo "${stage}" | xargs)"
  if [[ -z "${stage}" ]]; then
    continue
  fi
  echo "=================================================================="
  echo "[RUN] stage=${stage}"
  echo "=================================================================="

  stage_prefix="${OUT_PREFIX}_${stage}"
  cmd=(bash scripts/pipeline_success_only_sft.sh
    --stage "${stage}"
    --out-prefix "${stage_prefix}"
    --model "${MODEL}"
    --concurrency "${CONCURRENCY}"
    --max-per-group "${MAX_PER_GROUP}"
  )
  if [[ -n "${WORKERS}" ]]; then
    cmd+=(--workers "${WORKERS}")
  fi
  "${cmd[@]}"

  per_stage_outputs+=("data/${stage_prefix}.train1.parquet")
done

echo "=================================================================="
echo "[DONE] per-stage outputs:"
for p in "${per_stage_outputs[@]}"; do
  echo "- ${p}"
done

if [[ "${MERGE}" != "1" ]]; then
  exit 0
fi

echo "=================================================================="
echo "[MERGE] -> data/${OUT_PREFIX}.train1.parquet"
OUT_PREFIX_ENV="${OUT_PREFIX}" PER_STAGE_OUTPUTS="$(printf "%s\n" "${per_stage_outputs[@]}")" python - <<'PY'
import pandas as pd
from pathlib import Path
import os

out_prefix = os.environ.get("OUT_PREFIX_ENV", "sft_success_only")
paths = [p.strip() for p in os.environ.get("PER_STAGE_OUTPUTS", "").splitlines() if p.strip()]
dfs = []
for p in paths:
    pp = Path(p)
    if not pp.exists():
        raise SystemExit(f"Missing parquet: {pp}")
    dfs.append(pd.read_parquet(pp))

df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["uid","format_version","messages"])
out = Path("data") / f"{out_prefix}.train1.parquet"
out.parent.mkdir(parents=True, exist_ok=True)
df[["uid","format_version","messages"]].to_parquet(out, engine="pyarrow", index=False)
print(f"merged_rows={len(df)} saved={out}")
PY
