#!/usr/bin/env python3
"""
Rewrite ONLY <think>...</think> spans in train1-style SFT parquet using GPT (OpenAI Responses API).

Requirements / Guarantees:
  - Do NOT change anything outside <think>...</think>.
  - Keep the same number of <think> blocks per assistant message.
  - Do NOT add new facts; do NOT pretend to have seen image contents.
  - Preserve tone/language as much as possible while making the reasoning clearer.

Input schema (train1.parquet compatible):
  - uid: string
  - format_version: string ("v1")
  - messages: string (JSON-encoded list of {role, content})

Example:
  python scripts/rewrite_sft_think_gpt.py \
    --input data/after_focus_RL_sft_train.parquet \
    --output data/after_focus_RL_sft_train.rewritten.parquet \
    --report logs/focused2/think_rewrite_report.jsonl \
    --cache logs/focused2/think_rewrite_cache.jsonl \
    --concurrency 20
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


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


def _parse_messages(messages_str: str) -> List[Dict[str, Any]]:
    try:
        msgs = json.loads(messages_str)
    except Exception:
        return []
    if not isinstance(msgs, list):
        return []
    return [m for m in msgs if isinstance(m, dict)]


def _first_user_question(messages: List[Dict[str, Any]]) -> str:
    for m in messages:
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            return m["content"].strip()
    return ""


def _assistant_actions_hint(assistant_content: str) -> str:
    s = assistant_content
    # keep only action tags (search/bbox/search_complete/answer) to provide minimal context
    tags = []
    for t in ("search", "bbox", "search_complete", "answer"):
        m = re.search(rf"<{t}>(.*?)</{t}>", s, flags=re.DOTALL | re.IGNORECASE)
        if m:
            inner = (m.group(1) or "").strip()
            tags.append(f"<{t}>{inner}</{t}>")
    return "\n".join(tags)


def _strip_thinks_keep_rest(assistant_content: str) -> str:
    # Replace think bodies with a fixed token to compare structure safely.
    def repl(m: re.Match) -> str:
        open_tag = "<think>"
        close_tag = "</think>"
        # preserve original tag casing? keep canonical
        return open_tag + "__THINK__" + close_tag
    return _RE_THINK.sub(repl, assistant_content)


def _rewrite_thinks_in_text(assistant_content: str, rewritten_thinks: List[str]) -> str:
    idx = 0

    def repl(m: re.Match) -> str:
        nonlocal idx
        if idx >= len(rewritten_thinks):
            return m.group(0)
        body = rewritten_thinks[idx]
        idx += 1
        return f"<think>{body}</think>"

    return _RE_THINK.sub(repl, assistant_content)


def _validate_no_external_change(before: str, after: str) -> Optional[str]:
    if _strip_thinks_keep_rest(before) != _strip_thinks_keep_rest(after):
        return "Non-think text changed"
    # Ensure think tags count matches
    if len(_RE_THINK.findall(before)) != len(_RE_THINK.findall(after)):
        return "Think block count changed"
    return None


@dataclass(frozen=True)
class RewriteUnit:
    uid: str
    assistant_index: int
    assistant_content: str
    thinks: List[str]


_DEV_PROMPT = """You are a careful editor for search-agent training logs.

Task:
- You will receive a QUESTION and a list of ASSISTANT_UNITS.
- Each ASSISTANT_UNIT contains: assistant_index, action_hint, and an ordered list of THINK blocks extracted from that assistant message.
- Rewrite each THINK block to be more coherent, logical, and fluent while preserving the original meaning and style.
- This is an EDITING task, not a problem-solving task. Do not try to improve correctness by adding new evidence.

Hard constraints:
- Output EXACTLY the same number of ASSISTANT_UNITS as input.
- For each unit, output EXACTLY the same number of THINK blocks as input.
- Do NOT include <think> or </think> tags in your output.
- Do NOT change any tool actions or add new actions.
- Do NOT add new facts. Do NOT claim you saw image contents. If the original thinks guessed based on an image, keep uncertainty and avoid inventing details.
- Do NOT newly introduce image-observation claims. If the original THINK did NOT contain phrases like "the image shows", "I can see", "in the image", "visible", or equivalents, you MUST NOT add them. Prefer intention/plan language (e.g., "I need to crop/inspect the relevant area") or keep it abstract.
- If the original THINK already contained an image-observation claim, you may keep it but MUST NOT add any new observed details (no new numbers, names, locations, or specific descriptions) and MUST NOT increase certainty (avoid upgrading "seems/might" to "is/clearly").
- Do NOT include tool/action tags as literal text inside THINK (e.g., do not write "<bbox>...</bbox>" or "<search_complete>true</search_complete>" inside the THINK body).
- Keep the language and tone consistent with the original THINK (e.g., if it is English, keep English; if it mixes languages, reduce corruption but do not fabricate content).
- Keep length roughly similar (do not expand excessively).

Output:
- Return ONLY valid JSON (no markdown, no prose).
- JSON schema:
  {
    "units": [
      {"assistant_index": int, "rewritten_thinks": ["...", "...", ...]},
      ...
    ]
  }
"""

_PROMPT_ID = hashlib.sha256(_DEV_PROMPT.encode("utf-8")).hexdigest()

_IMAGE_CLAIM_PATTERNS = [
    # English (best-effort)
    re.compile(r"\\b(the image shows|in the image|visible in the image|i can see|it clearly shows)\\b", re.IGNORECASE),
    # Korean (best-effort)
    re.compile(r"(이미지(에서|에)|사진(에서|에)).*(보이|보인다|확인)", re.IGNORECASE),
]


def _has_image_claim(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return any(p.search(text) for p in _IMAGE_CLAIM_PATTERNS)


_RE_ACTION_TAG_TEXT = re.compile(r"</?\\s*(search|bbox|search_complete|answer)\\b", re.IGNORECASE)


async def _rewrite_one(
    client,
    model: str,
    effort: str,
    max_output_tokens: int,
    question: str,
    units_payload: List[Dict[str, Any]],
    semaphore: asyncio.Semaphore,
    request_timeout_seconds: float,
) -> Dict[str, Any]:
    user_payload = {"question": question, "assistant_units": units_payload}
    # Use Responses API in a way that aligns with the official docs:
    # - Put long-form policy in `instructions` (developer/system).
    # - Provide the payload as user `input_text`.
    # - Request Structured Outputs (JSON schema) to reduce parse failures.
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "units": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "assistant_index": {"type": "integer"},
                        "rewritten_thinks": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["assistant_index", "rewritten_thinks"],
                },
            }
        },
        "required": ["units"],
    }

    kwargs: Dict[str, Any] = {
        "model": model,
        "instructions": _DEV_PROMPT,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": json.dumps(user_payload, ensure_ascii=False)},
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "think_rewrite",
                "schema": schema,
                "strict": True,
            }
        },
        "max_output_tokens": int(max_output_tokens),
        "reasoning": {"effort": effort},
    }

    async with semaphore:
        response = await asyncio.wait_for(
            client.responses.create(**kwargs),
            timeout=float(request_timeout_seconds),
        )

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


async def main_async() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input Parquet (train1 schema)")
    ap.add_argument("--output", required=True, help="Output Parquet (rewritten)")
    ap.add_argument("--report", default=None, help="JSONL report path (optional)")
    ap.add_argument("--cache", default=None, help="JSONL cache path (optional)")
    ap.add_argument("--model", default=os.getenv("REWRITE_MODEL", os.getenv("FILTER_MODEL", "gpt-5.2-2025-12-11")))
    ap.add_argument("--reasoning-effort", default=os.getenv("REWRITE_REASONING_EFFORT", "high"))
    ap.add_argument("--max-output-tokens", type=int, default=6000)
    ap.add_argument("--concurrency", type=int, default=500)
    ap.add_argument("--max-retries", type=int, default=5, help="OpenAI client max_retries (default: 5).")
    ap.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=900.0,
        help="Hard timeout per API call in seconds (default: 900).",
    )
    ap.add_argument("--max-units-per-call", type=int, default=3, help="How many assistant messages to rewrite per API call (default: 3).")
    ap.add_argument(
        "--revert-new-image-claims",
        action="store_true",
        help="Revert a rewritten think if it introduces a new image-observation claim not present in the original think.",
    )
    ap.add_argument(
        "--revert-action-tags-in-think",
        action="store_true",
        help="Revert a rewritten think if it contains action/tool tags as literal text (e.g., <bbox>).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0 means all rows")
    ap.add_argument("--sample", choices=["all", "random"], default="all")
    ap.add_argument("--on-error", choices=["keep", "drop"], default="keep")
    args = ap.parse_args()

    _load_repo_dotenv()

    import pandas as pd

    df = pd.read_parquet(args.input)
    if not {"uid", "format_version", "messages"}.issubset(set(df.columns)):
        raise SystemExit(f"Unexpected schema. Need uid/format_version/messages, got: {list(df.columns)}")

    idxs = list(range(len(df)))
    if args.sample == "random":
        random.seed(int(args.seed))
        random.shuffle(idxs)
    if int(args.limit) > 0:
        idxs = idxs[: int(args.limit)]
    df_sel = df.iloc[idxs].copy()

    client = _get_openai_async_client(max_retries=int(args.max_retries))
    semaphore = asyncio.Semaphore(max(1, int(args.concurrency)))
    cache = _load_cache(args.cache)

    n_ok = 0
    n_drop = 0
    n_error = 0
    n_changed = 0

    async def process_row(row: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        uid = str(row.get("uid"))
        messages = _parse_messages(str(row.get("messages")))
        question = _first_user_question(messages)

        # build rewrite units for assistant messages (per-row, single request)
        units: List[RewriteUnit] = []
        units_payload: List[Dict[str, Any]] = []
        for idx, m in enumerate(messages):
            if m.get("role") != "assistant":
                continue
            content = m.get("content")
            if not isinstance(content, str):
                continue
            thinks = [t.strip() for t in _RE_THINK.findall(content)]
            if not thinks:
                continue
            unit = RewriteUnit(
                uid=uid,
                assistant_index=idx,
                assistant_content=content,
                thinks=thinks,
            )
            units.append(unit)
            units_payload.append(
                {
                    "assistant_index": idx,
                    "action_hint": _assistant_actions_hint(content),
                    "thinks": [{"index": i, "think": t} for i, t in enumerate(thinks)],
                }
            )

        if not units:
            # No thinks to rewrite
            return True, {"uid": uid, "changed": False, "n_units": 0, "n_thinks": 0}, row.get("messages")

        changed_any = False
        errors: List[str] = []

        # Chunk assistant units per API call to keep prompts small and latency reasonable.
        chunk_size = max(1, int(args.max_units_per_call))
        for start in range(0, len(units), chunk_size):
            chunk_units = units[start : start + chunk_size]
            chunk_payload = units_payload[start : start + chunk_size]

            cache_key = _stable_hash(
                {
                    "prompt_id": _PROMPT_ID,
                    "model": args.model,
                    "effort": args.reasoning_effort,
                    "question": question,
                    "units": chunk_payload,
                }
            )

            if cache_key in cache and isinstance(cache[cache_key], dict) and "error" not in cache[cache_key]:
                result = cache[cache_key]
            else:
                try:
                    print(
                        f"uid={uid} call_units={len(chunk_units)} assistant_index={chunk_units[0].assistant_index}..",
                        flush=True,
                    )
                    result = await _rewrite_one(
                        client,
                        model=args.model,
                        effort=str(args.reasoning_effort),
                        max_output_tokens=int(args.max_output_tokens),
                        question=question,
                        units_payload=chunk_payload,
                        semaphore=semaphore,
                        request_timeout_seconds=float(args.request_timeout_seconds),
                    )
                except Exception as e:
                    result = {"error": str(e)}
                cache[cache_key] = result
                _append_cache(args.cache, cache_key, result)

            if "error" in result:
                errors.append(result["error"])
                continue

            out_units = result.get("units")
            if not isinstance(out_units, list) or len(out_units) != len(chunk_units):
                errors.append("bad_units_shape")
                continue

            out_map: Dict[int, List[str]] = {}
            for ou in out_units:
                if not isinstance(ou, dict):
                    errors.append("bad_unit_type")
                    continue
                ai = ou.get("assistant_index")
                rts = ou.get("rewritten_thinks")
                if not isinstance(ai, int) or not isinstance(rts, list):
                    errors.append("bad_unit_fields")
                    continue
                out_map[ai] = rts

            for unit in chunk_units:
                rewritten_thinks = out_map.get(unit.assistant_index)
                if not isinstance(rewritten_thinks, list) or len(rewritten_thinks) != len(unit.thinks):
                    errors.append("bad_output_shape")
                    continue

                cleaned: List[str] = []
                bad = False
                for old_t, t in zip(unit.thinks, rewritten_thinks):
                    if not isinstance(t, str):
                        bad = True
                        break
                    t2 = t.strip()
                    if "<think" in t2.lower() or "</think" in t2.lower():
                        bad = True
                        break
                    if bool(args.revert_action_tags_in_think) and _RE_ACTION_TAG_TEXT.search(t2):
                        t2 = old_t.strip()
                    if bool(args.revert_new_image_claims) and _has_image_claim(t2) and not _has_image_claim(old_t):
                        t2 = old_t.strip()
                    cleaned.append(t2)
                if bad:
                    errors.append("bad_output_content")
                    continue

                new_content = _rewrite_thinks_in_text(unit.assistant_content, cleaned)
                err = _validate_no_external_change(unit.assistant_content, new_content)
                if err:
                    errors.append(err)
                    continue

                if new_content != unit.assistant_content:
                    changed_any = True
                messages[unit.assistant_index]["content"] = new_content

        if errors:
            keep = (args.on_error == "keep")
            rep = {"uid": uid, "changed": changed_any, "n_units": len(units), "errors": errors}
            return keep, rep, (json.dumps(messages, ensure_ascii=False, separators=(",", ":")) if keep else None)

        rep = {"uid": uid, "changed": changed_any, "n_units": len(units), "errors": []}
        return True, rep, json.dumps(messages, ensure_ascii=False, separators=(",", ":"))

    tasks = [asyncio.create_task(process_row(r)) for r in df_sel.to_dict(orient="records")]
    kept_rows = []
    done = 0
    for fut in asyncio.as_completed(tasks):
        keep, rep, new_messages_str = await fut
        done += 1
        if rep.get("errors"):
            n_error += 1
        if keep and isinstance(new_messages_str, str):
            kept_rows.append({"uid": rep.get("uid", ""), "format_version": "v1", "messages": new_messages_str})
            n_ok += 1
            if rep.get("changed"):
                n_changed += 1
        else:
            n_drop += 1

        _append_report(args.report, rep)
        if done % 10 == 0 or done == len(df_sel):
            print(f"progress {done}/{len(df_sel)} kept={n_ok} errors={n_error} changed_rows={n_changed}", flush=True)

    import pandas as pd

    df_out = pd.DataFrame(kept_rows, columns=["uid", "format_version", "messages"])
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path, engine="pyarrow", index=False)

    # Post-check: verify invariants across the saved parquet.
    bad_invariants = 0
    for s in df_out["messages"].tolist():
        msgs = _parse_messages(str(s))
        for m in msgs:
            if m.get("role") != "assistant":
                continue
            c = m.get("content")
            if not isinstance(c, str):
                continue
            # Ensure no placeholder dict remains
            if isinstance(c, dict):
                bad_invariants += 1
                break
            # Ensure think tags are balanced syntactically (count match)
            if c.lower().count("<think>") != c.lower().count("</think>"):
                bad_invariants += 1
                break

    print(f"evaluated={len(df_sel)} kept={n_ok} dropped={n_drop} errors={n_error} changed_rows={n_changed} bad_invariants={bad_invariants}")
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
