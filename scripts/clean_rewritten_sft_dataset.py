#!/usr/bin/env python3
"""
Clean a rewritten train1-style SFT parquet by enforcing strict invariants:
  - Do NOT change anything outside <think>...</think>.
  - If a rewritten <think> contains tool/action tags as *literal text*
    (e.g., "<bbox>", "<search_complete>"), revert that think block to the original.

This is a safety post-pass to prevent format contamination during SFT.

Input schema (train1.parquet compatible):
  - uid: string
  - format_version: string ("v1")
  - messages: string (JSON-encoded list of {role, content})

Example:
  python scripts/clean_rewritten_sft_dataset.py \
    --original data/after_focus_RL_sft_train.parquet \
    --rewritten data/after_focus_RL_sft_train.rewritten.parquet \
    --output data/after_focus_RL_sft_train.rewritten.cleaned.parquet
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


_RE_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_RE_ACTION_TAG_IN_THINK = re.compile(r"</?\s*(search|bbox|search_complete|answer)\b", re.IGNORECASE)
_IMAGE_CLAIM_PATTERNS = [
    re.compile(r"\b(the image shows|in the image|visible in the image|i can see|it clearly shows)\b", re.IGNORECASE),
    re.compile(r"(이미지(에서|에)|사진(에서|에)).*(보이|보인다|확인)", re.IGNORECASE),
]


def _has_image_claim(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return any(p.search(text) for p in _IMAGE_CLAIM_PATTERNS)


def _parse_messages(messages: Any) -> List[Dict[str, Any]]:
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except Exception:
            return []
    if not isinstance(messages, list):
        return []
    return [m for m in messages if isinstance(m, dict)]


def _iter_assistant_thinks(messages: List[Dict[str, Any]]) -> List[Tuple[int, str, List[str]]]:
    out: List[Tuple[int, str, List[str]]] = []
    for idx, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        c = m.get("content")
        if not isinstance(c, str):
            continue
        thinks = [t for t in _RE_THINK.findall(c)]
        if thinks:
            out.append((idx, c, thinks))
    return out


def _rewrite_thinks_in_text(assistant_content: str, new_thinks: List[str]) -> str:
    i = 0

    def repl(m: re.Match) -> str:
        nonlocal i
        if i >= len(new_thinks):
            return m.group(0)
        body = new_thinks[i]
        i += 1
        return f"<think>{body}</think>"

    return _RE_THINK.sub(repl, assistant_content)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True)
    ap.add_argument("--rewritten", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--drop-uids",
        default="",
        help="Comma-separated uid list to drop entirely (optional).",
    )
    ap.add_argument(
        "--drop-if-action-tag-in-think",
        action="store_true",
        help="Drop rows where any <think> contains action/tool tags as literal text.",
    )
    ap.add_argument(
        "--revert-new-image-claims",
        action="store_true",
        help="If a rewritten think introduces a new image-observation claim not present in the original think, revert that think to the original.",
    )
    args = ap.parse_args()

    import pandas as pd

    orig = pd.read_parquet(args.original)
    rew = pd.read_parquet(args.rewritten)
    required = {"uid", "format_version", "messages"}
    if not required.issubset(set(orig.columns)) or not required.issubset(set(rew.columns)):
        raise SystemExit("Unexpected schema; need uid/format_version/messages")

    orig_map = {str(r.uid): str(r.messages) for r in orig.itertuples()}
    drop = {u.strip() for u in str(args.drop_uids).split(",") if u.strip()}

    kept_rows = []
    n_rows = 0
    n_dropped = 0
    n_reverted_thinks = 0
    n_flagged_thinks = 0
    n_reverted_image_claims = 0

    for r in rew.itertuples():
        uid = str(r.uid)
        if uid in drop:
            n_dropped += 1
            continue
        n_rows += 1
        o_str = orig_map.get(uid)
        if o_str is None:
            raise SystemExit(f"uid missing in original: {uid}")

        o_msgs = _parse_messages(o_str)
        w_msgs = _parse_messages(r.messages)
        if len(o_msgs) != len(w_msgs):
            raise SystemExit(f"messages length mismatch for uid={uid}")

        o_units = _iter_assistant_thinks(o_msgs)
        w_units = _iter_assistant_thinks(w_msgs)
        if [i for i, _, _ in o_units] != [i for i, _, _ in w_units]:
            raise SystemExit(f"assistant indices mismatch for uid={uid}")

        if bool(args.drop_if_action_tag_in_think):
            if any(_RE_ACTION_TAG_IN_THINK.search(t) for _, _, ts in o_units for t in ts):
                n_dropped += 1
                continue

        for (ai, w_content, w_thinks), (_, _o_content, o_thinks) in zip(w_units, o_units):
            if len(w_thinks) != len(o_thinks):
                raise SystemExit(f"think count mismatch for uid={uid} assistant_index={ai}")
            new_thinks = list(w_thinks)
            for j, t in enumerate(w_thinks):
                if _RE_ACTION_TAG_IN_THINK.search(t):
                    n_flagged_thinks += 1
                    new_thinks[j] = o_thinks[j]
                    n_reverted_thinks += 1
                    continue
                if bool(args.revert_new_image_claims) and _has_image_claim(t) and not _has_image_claim(o_thinks[j]):
                    new_thinks[j] = o_thinks[j]
                    n_reverted_thinks += 1
                    n_reverted_image_claims += 1
            if new_thinks != w_thinks:
                w_msgs[ai]["content"] = _rewrite_thinks_in_text(w_content, new_thinks)

        kept_rows.append(
            {
                "uid": uid,
                "format_version": "v1",
                "messages": json.dumps(w_msgs, ensure_ascii=False, separators=(",", ":")),
            }
        )

    out = pd.DataFrame(kept_rows, columns=["uid", "format_version", "messages"])
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, engine="pyarrow", index=False)

    print(
        f"input_rows={len(rew)} kept={len(out)} dropped={n_dropped} "
        f"flagged_thinks={n_flagged_thinks} reverted_thinks={n_reverted_thinks} reverted_image_claims={n_reverted_image_claims}"
    )
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
