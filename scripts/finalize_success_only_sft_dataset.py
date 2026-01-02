#!/usr/bin/env python3
"""
Finalize a vetted success-only candidate parquet into train1-style SFT parquet.

Rules (per docs/SFT_SUCCESS_ONLY_CURATION_SPEC.md):
  - Group by prompt_uid
  - Keep up to K rows per group (default K=4)
  - Prefer efficiency:
      ndcg desc, num_search asc, num_bbox asc, approx_length asc
  - Output ONLY uid/format_version/messages columns (train1-compatible)

Input parquet is expected to contain:
  - uid, format_version, messages
and preferably helper columns:
  - prompt_uid, ndcg
If helper columns are missing, prompt_uid is derived from uid prefix before "__s".
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


_RE_SEARCH = re.compile(r"<search>(.*?)</search>", re.DOTALL | re.IGNORECASE)
_RE_BBOX = re.compile(r"<bbox>\\s*\\[(.*?)\\]\\s*</bbox>", re.DOTALL | re.IGNORECASE)


def _parse_messages(messages: Any) -> List[Dict[str, Any]]:
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except Exception:
            return []
    if not isinstance(messages, list):
        return []
    return [m for m in messages if isinstance(m, dict)]


def _count_actions(messages: List[Dict[str, Any]]) -> Tuple[int, int]:
    n_search = 0
    n_bbox = 0
    for m in messages:
        if m.get("role") != "assistant":
            continue
        c = m.get("content")
        if not isinstance(c, str):
            continue
        n_search += len([q for q in _RE_SEARCH.findall(c) if (q or "").strip()])
        n_bbox += len([b for b in _RE_BBOX.findall(c) if (b or "").strip()])
    return n_search, n_bbox


def _approx_len(messages: List[Dict[str, Any]]) -> int:
    # Cheap proxy: character length of all assistant messages.
    total = 0
    for m in messages:
        if m.get("role") != "assistant":
            continue
        c = m.get("content")
        if isinstance(c, str):
            total += len(c)
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True, help="Output train1 parquet path")
    ap.add_argument("--max-per-group", type=int, default=4)
    args = ap.parse_args()

    import pandas as pd

    df = pd.read_parquet(args.input)
    required = {"uid", "format_version", "messages"}
    if not required.issubset(set(df.columns)):
        raise SystemExit(f"Unexpected schema; need {sorted(required)}")

    # Ensure prompt_uid exists
    if "prompt_uid" not in df.columns:
        df = df.copy()
        df["prompt_uid"] = df["uid"].astype(str).apply(lambda u: u.split("__s", 1)[0] if "__s" in u else u)

    # Ensure ndcg exists (best-effort, but spec expects it for focused logs)
    if "ndcg" not in df.columns:
        df = df.copy()
        df["ndcg"] = 0.0

    # Compute efficiency metrics from messages (robust even if not provided by extractor)
    n_search_list = []
    n_bbox_list = []
    approx_len_list = []
    for _, r in df.iterrows():
        msgs = _parse_messages(r.get("messages"))
        ns, nb = _count_actions(msgs)
        n_search_list.append(ns)
        n_bbox_list.append(nb)
        approx_len_list.append(_approx_len(msgs))
    df = df.copy()
    df["n_search"] = n_search_list
    df["n_bbox"] = n_bbox_list
    df["approx_len"] = approx_len_list

    # Rank within each group
    kept_rows = []
    max_k = max(1, int(args.max_per_group))
    for prompt_uid, g in df.groupby("prompt_uid", sort=False):
        gg = g.sort_values(
            by=["ndcg", "n_search", "n_bbox", "approx_len"],
            ascending=[False, True, True, True],
            kind="mergesort",
        )
        kept_rows.append(gg.head(max_k))

    out = pd.concat(kept_rows, ignore_index=True) if kept_rows else df.head(0).copy()
    out_final = out[["uid", "format_version", "messages"]].copy()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_final.to_parquet(out_path, engine="pyarrow", index=False)

    print(f"input_rows={len(df)} output_rows={len(out_final)} groups={df['prompt_uid'].nunique()} max_per_group={max_k}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

