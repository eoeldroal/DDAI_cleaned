#!/usr/bin/env python3
"""
GPT pass/fail final vetting for train1-style SFT trajectories.

This script is intentionally stricter than query-only filters:
  - It can reject rollouts with severe query collapse, repetitive/degenerate actions,
    obvious gibberish, or clear format violations.
  - It does NOT rewrite anything; it only filters.

Input:
  A parquet containing at least: uid, format_version, messages
  (Extra columns like prompt_uid/sample_idx/ndcg/judge_score are preserved.)

Output:
  - A filtered parquet (pass==true only)
  - A JSONL report (one per uid)
  - A JSONL cache to allow resume

Example:
  python scripts/vet_sft_rollouts_gpt.py \
    --input logs/focused2/success_candidates.parquet \
    --output logs/focused2/success_candidates.vetted.parquet \
    --report logs/focused2/vet_report.jsonl \
    --cache logs/focused2/vet_cache.jsonl \
    --model gpt-5.2 --concurrency 200
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
_RE_BBOX = re.compile(r"<bbox>\\s*\\[(.*?)\\]\\s*</bbox>", re.DOTALL | re.IGNORECASE)
_RE_SC = re.compile(r"<search_complete>\\s*true\\s*</search_complete>", re.IGNORECASE)
_RE_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _load_repo_dotenv() -> None:
    if load_dotenv is None:
        return
    repo_root = Path(__file__).resolve().parents[1]
    dotenv_path = repo_root / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
    else:
        load_dotenv()


def _get_openai_async_client(max_retries: int):
    from openai import AsyncOpenAI

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("FROZEN_OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("FROZEN_OPENAI_BASE_URL") or "https://api.openai.com/v1"
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY (or FROZEN_OPENAI_API_KEY).")
    return AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=1000.0, max_retries=int(max_retries))


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


def _parse_messages(messages_str: Any) -> List[Dict[str, Any]]:
    if isinstance(messages_str, str):
        try:
            messages_str = json.loads(messages_str)
        except Exception:
            return []
    if not isinstance(messages_str, list):
        return []
    return [m for m in messages_str if isinstance(m, dict)]


def _first_user_question(messages: List[Dict[str, Any]]) -> str:
    for m in messages:
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            return m["content"].strip()
    return ""


def _assistant_texts(messages: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        c = m.get("content")
        if isinstance(c, str) and c.strip():
            out.append(c)
    return out


def _summarize_rollout(messages: List[Dict[str, Any]], max_think_chars: int) -> Dict[str, Any]:
    question = _first_user_question(messages)
    assistants = _assistant_texts(messages)

    queries: List[str] = []
    bboxes: List[str] = []
    thinks: List[str] = []
    has_sc = False
    system_errors = 0

    for m in messages:
        if (m.get("role") or "") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str) and "[System Error:" in c:
            system_errors += 1

    for a in assistants:
        for q in _RE_SEARCH.findall(a):
            q = (q or "").strip()
            if q:
                queries.append(q)
        for b in _RE_BBOX.findall(a):
            b = (b or "").strip()
            if b:
                bboxes.append(b)
        if _RE_SC.search(a):
            has_sc = True
        for t in _RE_THINK.findall(a):
            t = (t or "").strip()
            if not t:
                continue
            if len(t) > int(max_think_chars):
                t = t[: int(max_think_chars)] + "…"
            thinks.append(t)

    # repetition heuristic (cheap signal; GPT decides final)
    repeats = 0
    for i in range(1, len(queries)):
        if queries[i].strip().lower() == queries[i - 1].strip().lower():
            repeats += 1

    return {
        "question": question,
        "has_search_complete_true": has_sc,
        "num_system_errors": system_errors,
        "num_search": len(queries),
        "num_bbox": len(bboxes),
        "num_assistant_messages": len(assistants),
        "repeated_adjacent_queries": repeats,
        "queries": [{"index": i, "query": q} for i, q in enumerate(queries)],
        "bboxes": [{"index": i, "bbox": b} for i, b in enumerate(bboxes[:50])],  # cap size
        "think_snippets": [{"index": i, "think": t} for i, t in enumerate(thinks[:50])],  # cap size
    }


@dataclass(frozen=True)
class VetItem:
    uid: str
    payload: Dict[str, Any]
    row_index: int


_DEV_PROMPT = """You are a strict final-vetting judge for search-agent training trajectories.

You will receive a JSON payload describing ONE rollout:
- question
- ordered search queries
- bbox strings
- snippets of think text
- some simple counters

Goal:
- Decide whether this rollout is acceptable for SFT, WITHOUT rewriting anything.

Reject if any of the following are severe:
- query_collapse: search queries are gibberish/noise or obviously corrupted (mixed scripts, nonsense, fragments)
- repetitive_actions: many repeated queries or meaningless bbox spam
- gibberish: the think text is largely nonsensical / random mixed-language garbage
- evidence_mismatch: the text strongly claims specific image content not supported by the provided context (obvious hallucination)
- format_violation: not following the required tool-tag protocol (missing think, missing action tag, etc.)

Important:
- Drift (a coherent but off-topic query) is allowed unless it is severe/cascading collapse.
- Do NOT be lenient: we prefer precision over recall for this dataset.

Output:
- Return ONLY valid JSON (no markdown, no prose).
- Schema:
  {
    "pass": boolean,
    "severity": 0|1|2|3,
    "reasons": [string, ...],
    "flags": {
      "query_collapse": boolean,
      "repetitive_actions": boolean,
      "gibberish": boolean,
      "evidence_mismatch": boolean,
      "format_violation": boolean
    }
  }

Keep reasons short (<= 15 words each), no newlines, do not quote full texts.
"""

_PROMPT_ID = hashlib.sha256(_DEV_PROMPT.encode("utf-8")).hexdigest()


async def _vet_one(
    client,
    model: str,
    effort: Optional[str],
    max_output_tokens: int,
    item: VetItem,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    inputs = [
        {"role": "developer", "content": _DEV_PROMPT},
        {"role": "user", "content": json.dumps(item.payload, ensure_ascii=False)},
    ]

    req: Dict[str, Any] = {
        "model": model,
        "input": inputs,
        "max_output_tokens": int(max_output_tokens),
    }
    if effort:
        req["reasoning"] = {"effort": effort}

    async with semaphore:
        resp = await client.responses.create(**req)

    # Extract text output (Responses API)
    out_text = ""
    try:
        out_text = resp.output_text  # type: ignore[attr-defined]
    except Exception:
        pass
    if not out_text:
        try:
            parts = []
            for o in getattr(resp, "output", []) or []:
                for c in getattr(o, "content", []) or []:
                    if getattr(c, "type", "") == "output_text":
                        parts.append(getattr(c, "text", ""))
            out_text = "\n".join(parts).strip()
        except Exception:
            out_text = ""

    parsed = _extract_json(out_text)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Failed to parse JSON from model output: {out_text[:200]}")
    return parsed


def _load_cache(cache_path: Path) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if not cache_path.exists():
        return cache
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            key = obj.get("key")
            val = obj.get("result")
            if isinstance(key, str) and isinstance(val, dict):
                cache[key] = val
    return cache


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


async def _run_async(args, df) -> Tuple[List[bool], int, int]:
    client = _get_openai_async_client(max_retries=int(args.max_retries))
    semaphore = asyncio.Semaphore(int(args.concurrency))

    cache_path = Path(args.cache) if args.cache else None
    cache: Dict[str, Dict[str, Any]] = _load_cache(cache_path) if cache_path else {}

    keep_flags = [False] * len(df)
    n_cached = 0
    n_api = 0

    queue: asyncio.Queue[Optional[VetItem]] = asyncio.Queue(maxsize=int(args.queue_size))

    async def worker() -> None:
        nonlocal n_cached, n_api
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                return
            key_obj = {"prompt_id": _PROMPT_ID, "model": args.model, "payload": item.payload}
            key = _stable_hash(key_obj)
            if key in cache:
                result = cache[key]
                n_cached += 1
            else:
                result = await _vet_one(
                    client=client,
                    model=args.model,
                    effort=args.effort,
                    max_output_tokens=args.max_output_tokens,
                    item=item,
                    semaphore=semaphore,
                )
                cache[key] = result
                n_api += 1
                if cache_path:
                    _append_jsonl(cache_path, {"key": key, "result": result})

            passed = bool(result.get("pass", False))
            keep_flags[item.row_index] = passed

            if args.report:
                _append_jsonl(
                    Path(args.report),
                    {
                        "uid": item.uid,
                        "pass": passed,
                        "severity": result.get("severity"),
                        "flags": result.get("flags"),
                        "reasons": result.get("reasons"),
                    },
                )
            queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(int(args.workers))]

    # Build work list
    indices = list(range(len(df)))
    if args.limit is not None:
        indices = indices[: int(args.limit)]
    if args.sample == "random" and args.limit is not None:
        random.seed(int(args.seed))
        indices = list(range(len(df)))
        random.shuffle(indices)
        indices = indices[: int(args.limit)]

    for row_i in indices:
        uid = str(df.iloc[row_i]["uid"])
        msgs = df.iloc[row_i]["messages"]
        messages = _parse_messages(msgs)
        payload = _summarize_rollout(messages, max_think_chars=int(args.max_think_chars))
        payload["uid"] = uid  # include for model context, but model must not echo it
        await queue.put(VetItem(uid=uid, payload=payload, row_index=row_i))

    for _ in workers:
        await queue.put(None)

    await queue.join()
    for w in workers:
        await w

    try:
        await client.close()
    except Exception:
        pass

    return keep_flags, n_cached, n_api


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", default="")
    ap.add_argument("--cache", default="")
    ap.add_argument("--model", default="gpt-5.2")
    ap.add_argument("--effort", default=None, help="reasoning.effort for Responses API (optional)")
    ap.add_argument("--max-output-tokens", type=int, default=400)
    ap.add_argument("--concurrency", type=int, default=200)
    ap.add_argument("--workers", type=int, default=200, help="Number of worker tasks (can be >= concurrency).")
    ap.add_argument("--queue-size", type=int, default=5000)
    ap.add_argument("--max-think-chars", type=int, default=480)
    ap.add_argument("--max-retries", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample", choices=["head", "random"], default="head")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _load_repo_dotenv()

    import pandas as pd

    df = pd.read_parquet(args.input)
    required = {"uid", "format_version", "messages"}
    if not required.issubset(set(df.columns)):
        raise SystemExit(f"Unexpected schema; need {sorted(required)}")

    if int(args.workers) <= 0:
        args.workers = max(1, int(args.concurrency))

    keep_flags, n_cached, n_api = asyncio.run(_run_async(args, df))

    kept = df[keep_flags].copy()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(out_path, engine="pyarrow", index=False)

    print(
        f"input_rows={len(df)} kept={len(kept)} dropped={len(df)-len(kept)} "
        f"cached={n_cached} api_calls={n_api} model={args.model}"
    )
    print(f"saved: {out_path}")
    if args.report:
        print(f"report: {Path(args.report)}")
    if args.cache:
        print(f"cache: {Path(args.cache)}")


if __name__ == "__main__":
    main()
