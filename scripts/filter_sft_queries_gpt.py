#!/usr/bin/env python3
"""
Filter train1-style SFT Parquet by search-query corruption using GPT (default: gpt-5.2).

Goal:
  - Keep the original trajectories/messages intact.
  - Drop an entire rollout if it contains ANY severely corrupted/gibberish search query.
  - Allow some semantic drift (keep but label as drift).

Input schema (train1.parquet compatible):
  - uid: string
  - format_version: string ("v1")
  - messages: string (JSON-encoded list of {role, content})

Example:
  python scripts/filter_sft_queries_gpt.py \
    --input logs/focused2/sft_train1_like.parquet \
    --output logs/focused2/sft_train1_like.filtered.parquet \
    --report logs/focused2/query_filter_report.jsonl \
    --cache logs/focused2/query_filter_cache.jsonl \
    --limit 10 --sample random --seed 0
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


_RE_SEARCH = re.compile(r"<search>(.*?)</search>", re.DOTALL | re.IGNORECASE)


def _load_repo_dotenv() -> None:
    if load_dotenv is None:
        return
    repo_root = Path(__file__).resolve().parents[1]
    dotenv_path = repo_root / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
    else:
        load_dotenv()


def _get_openai_async_client():
    from openai import AsyncOpenAI

    primary_key = os.getenv("OPENAI_API_KEY")
    primary_base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("FROZEN_OPENAI_BASE_URL")

    api_key = primary_key or fallback_key
    base_url = primary_base_url or (fallback_base_url if fallback_key else None) or "https://api.openai.com/v1"

    if not api_key:
        raise RuntimeError(
            "Missing API key. Set OPENAI_API_KEY in .env (or DASHSCOPE_API_KEY as fallback)."
        )

    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=1000.0,
        max_retries=10,
    )


def _stable_hash(obj: Any) -> str:
    blob = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _extract_json(text: str) -> Optional[dict]:
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except Exception:
        return None


def _parse_messages(messages_str: str) -> List[Dict[str, Any]]:
    try:
        msgs = json.loads(messages_str)
    except Exception:
        return []
    if not isinstance(msgs, list):
        return []
    return [m for m in msgs if isinstance(m, dict)]


def _extract_question_and_queries(messages: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    question = ""
    queries: List[str] = []

    for m in messages:
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            # 첫 user 텍스트를 질문으로 사용
            question = m["content"].strip()
            break

    for m in messages:
        if m.get("role") != "assistant":
            continue
        content = m.get("content")
        if not isinstance(content, str):
            continue
        for q in _RE_SEARCH.findall(content):
            q = q.strip()
            if q:
                queries.append(q)

    return question, queries


@dataclass(frozen=True)
class JudgeInput:
    question: str
    queries: List[str]


_DEV_PROMPT = """You are a strict data-quality judge for search-agent training logs.

Task:
- You will receive a QUESTION and a list of SEARCH QUERIES the agent issued across turns.
- You must label each query as one of: ok, drift, gibberish, code.

Definitions:
- ok: a coherent search query in any language that could plausibly help answer the QUESTION.
- drift: a coherent search query but likely unrelated to the QUESTION. Drift is allowed (do NOT reject solely for drift).
- gibberish: severely corrupted/uninterpretable text (random mixed scripts/noise, broken fragments, nonsense) such that it is not a reasonable search query.
- code: code, stack traces, config dumps, URLs with encoded parameters, or other non-query artifacts.

Decision rule:
- Output keep=true if and ONLY if there is NO query labeled gibberish or code.
- Drift is allowed: keep can still be true even if some queries are drift.

Output:
- Return ONLY valid JSON (no markdown, no prose).
- JSON schema:
  {
    "keep": boolean,
    "queries": [
      {"index": int, "label": "ok"|"drift"|"gibberish"|"code", "severity": 0|1|2|3, "reason": string}
    ]
  }
Where severity meaning:
  0=ok, 1=minor drift, 2=major drift, 3=gibberish/code.

Constraints:
- Do NOT copy the full query text into the output JSON (use only the provided index).
- Keep each reason short (<= 15 words, no newlines).
"""

_PROMPT_ID = hashlib.sha256(_DEV_PROMPT.encode("utf-8")).hexdigest()


async def _judge_one(
    client,
    model: str,
    effort: Optional[str],
    max_output_tokens: int,
    item: JudgeInput,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    user_payload = {
        "question": item.question,
        "queries": [{"index": i, "query": q} for i, q in enumerate(item.queries)],
    }

    inputs = [
        {"role": "developer", "content": _DEV_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]

    kwargs: Dict[str, Any] = {
        "model": model,
        "input": inputs,
        "max_output_tokens": max_output_tokens,
    }
    if effort:
        kwargs["reasoning"] = {"effort": effort}

    async with semaphore:
        response = await client.responses.create(**kwargs)
        print(response)

    text = getattr(response, "output_text", None) or ""
    parsed = _extract_json(text)
    if parsed is None:
        raise ValueError(f"Failed to parse JSON from model output: {text[:200]}")
    return parsed


def _load_cache(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    cache: Dict[str, Dict[str, Any]] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            k = obj.get("cache_key")
            v = obj.get("result")
            if isinstance(k, str) and isinstance(v, dict):
                cache[k] = v
    return cache


def _append_cache(path: Optional[str], cache_key: str, result: Dict[str, Any]) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"cache_key": cache_key, "result": result}, ensure_ascii=False) + "\n")


def _append_report(path: Optional[str], obj: Dict[str, Any]) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _decide_keep(result: Dict[str, Any]) -> bool:
    queries = result.get("queries")
    if not isinstance(queries, list):
        return bool(result.get("keep", True))
    for q in queries:
        if not isinstance(q, dict):
            continue
        label = (q.get("label") or "").strip().lower()
        if label in ("gibberish", "code"):
            return False
    return True


async def main_async() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input Parquet (train1 schema)")
    ap.add_argument("--output", required=True, help="Output Parquet (filtered)")
    ap.add_argument("--report", default=None, help="JSONL report path (optional)")
    ap.add_argument("--cache", default=None, help="JSONL cache path (optional)")
    ap.add_argument("--model", default=os.getenv("FILTER_MODEL", "gpt-5.2-2025-12-11"))
    ap.add_argument("--reasoning-effort", default=os.getenv("FILTER_REASONING_EFFORT", os.getenv("FROZEN_REASONING_EFFORT", "medium")))
    ap.add_argument("--max-output-tokens", type=int, default=60000)
    ap.add_argument("--limit", type=int, default=10, help="How many rows to evaluate (default: 10)")
    ap.add_argument("--sample", choices=["first", "random"], default="random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--on-error", choices=["keep", "drop"], default="keep")
    args = ap.parse_args()

    _load_repo_dotenv()

    import pandas as pd

    df = pd.read_parquet(args.input)
    if not {"uid", "format_version", "messages"}.issubset(set(df.columns)):
        raise SystemExit(f"Unexpected schema. Need uid/format_version/messages, got: {list(df.columns)}")

    idxs = list(range(len(df)))
    if args.sample == "random":
        random.seed(args.seed)
        random.shuffle(idxs)
    if int(args.limit) > 0:
        idxs = idxs[: int(args.limit)]
    df_sel = df.iloc[idxs].copy()

    client = _get_openai_async_client()
    semaphore = asyncio.Semaphore(max(1, int(args.concurrency)))

    cache = _load_cache(args.cache)

    kept_rows = []
    n_ok = 0
    n_drop = 0
    n_error = 0

    async def process_row(row: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        uid = str(row.get("uid"))
        messages = _parse_messages(str(row.get("messages")))
        question, queries = _extract_question_and_queries(messages)
        judge_in = JudgeInput(question=question, queries=queries)

        cache_key = _stable_hash(
            {
                "prompt_id": _PROMPT_ID,
                "model": args.model,
                "question": judge_in.question,
                "queries": judge_in.queries,
            }
        )

        if cache_key in cache:
            cached = cache[cache_key]
            if isinstance(cached, dict) and "error" not in cached:
                result = cached
            else:
                result = None
        else:
            result = None

        if result is None:
            try:
                result = await _judge_one(
                    client,
                    model=args.model,
                    effort=(args.reasoning_effort or None),
                    max_output_tokens=args.max_output_tokens,
                    item=judge_in,
                    semaphore=semaphore,
                )
            except Exception as e:
                result = {
                    "keep": (args.on_error == "keep"),
                    "queries": [],
                    "error": str(e),
                }
            cache[cache_key] = result
            _append_cache(args.cache, cache_key, result)

        keep = _decide_keep(result) if "error" not in result else bool(result.get("keep", True))

        report_obj = {
            "uid": uid,
            "keep": bool(keep),
            "question": question,
            "queries": queries,
            "result": result,
        }
        _append_report(args.report, report_obj)
        return keep, report_obj

    tasks = [process_row(r) for r in df_sel.to_dict(orient="records")]
    results = await asyncio.gather(*tasks)

    for (keep, rep), (_, row) in zip(results, df_sel.iterrows()):
        if "error" in rep.get("result", {}):
            n_error += 1
        if keep:
            kept_rows.append(row)
            n_ok += 1
        else:
            n_drop += 1

    if kept_rows:
        df_out = df_sel.loc[[r.name for r in kept_rows]].copy()
    else:
        df_out = df_sel.iloc[0:0].copy()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path, engine="pyarrow", index=False)

    print(f"evaluated={len(df_sel)} kept={n_ok} dropped={n_drop} errors={n_error}")
    print(f"saved: {out_path}")
    if args.report:
        print(f"report: {args.report}")
    if args.cache:
        print(f"cache: {args.cache}")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
