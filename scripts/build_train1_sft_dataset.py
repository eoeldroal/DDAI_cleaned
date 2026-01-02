#!/usr/bin/env python3
"""
Build train1-style SFT datasets (train1.parquet/val1.parquet) from an existing
train1-style parquet (uid/format_version/messages).

Pipeline:
  1) Load input parquet
  2) Filter by number of System Error messages (<= max_system_errors)
  3) Split train/val
  4) Save to outdir as train1.parquet / val1.parquet

Notes:
  - This script does NOT rewrite messages; it only filters/splits.
  - "System Error" is counted from user messages containing "[System Error:".
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _parse_messages(messages: Any) -> List[Dict[str, Any]]:
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except Exception:
            return []
    if not isinstance(messages, list):
        return []
    return [m for m in messages if isinstance(m, dict)]


def _count_system_errors(messages: List[Dict[str, Any]]) -> int:
    n = 0
    for m in messages:
        if (m.get("role") or "") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str) and "[System Error:" in c:
            n += 1
    return n


@dataclass(frozen=True)
class Row:
    uid: str
    format_version: str
    messages_json: str
    system_errors: int


def _split_rows(rows: List[Row], val_ratio: float, seed: int) -> Tuple[List[Row], List[Row]]:
    if float(val_ratio) <= 0.0:
        return rows, []
    idxs = list(range(len(rows)))
    random.seed(int(seed))
    random.shuffle(idxs)
    v = int(round(len(rows) * float(val_ratio)))
    v = max(1, v) if len(rows) > 1 else 0
    val_set = set(idxs[:v])
    train, val = [], []
    for i, r in enumerate(rows):
        (val if i in val_set else train).append(r)
    return train, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input train1-style parquet (uid/format_version/messages)")
    ap.add_argument("--outdir", required=True, help="Output dir (will create train1.parquet/val1.parquet)")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--max-system-errors",
        type=int,
        default=1,
        help="Keep rows with System Error count <= this value (default: 1). Set 0 to keep only clean rows.",
    )
    args = ap.parse_args()

    import pandas as pd

    df = pd.read_parquet(args.input)
    required = {"uid", "format_version", "messages"}
    if not required.issubset(set(df.columns)):
        raise SystemExit(f"Unexpected schema. Need {sorted(required)}, got: {list(df.columns)}")

    rows: List[Row] = []
    dropped = 0
    for _, r in df.iterrows():
        uid = str(r.get("uid", ""))
        fmt = str(r.get("format_version", "v1"))
        msgs_raw = r.get("messages")
        msgs = _parse_messages(msgs_raw)
        se = _count_system_errors(msgs)
        if se > int(args.max_system_errors):
            dropped += 1
            continue
        msgs_json = msgs_raw if isinstance(msgs_raw, str) else json.dumps(msgs_raw, ensure_ascii=False, separators=(",", ":"))
        rows.append(Row(uid=uid, format_version=fmt, messages_json=msgs_json, system_errors=se))

    train, val = _split_rows(rows, val_ratio=args.val_ratio, seed=args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    train_path = outdir / "train1.parquet"
    val_path = outdir / "val1.parquet"

    df_train = pd.DataFrame([{"uid": r.uid, "format_version": r.format_version, "messages": r.messages_json} for r in train])
    df_val = pd.DataFrame([{"uid": r.uid, "format_version": r.format_version, "messages": r.messages_json} for r in val])
    df_train.to_parquet(train_path, engine="pyarrow", index=False)
    if len(df_val) > 0:
        df_val.to_parquet(val_path, engine="pyarrow", index=False)

    kept = len(rows)
    total = len(df)
    print(f"input_rows={total} kept={kept} dropped={dropped} max_system_errors={int(args.max_system_errors)}")
    print(f"split train={len(df_train)} val={len(df_val)} val_ratio={args.val_ratio}")
    print(f"saved: {train_path}")
    if len(df_val) > 0:
        print(f"saved: {val_path}")


if __name__ == "__main__":
    main()
