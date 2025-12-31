#!/usr/bin/env python3
"""
Trajectory Extractor for Visual RAG Training Logs

unified_trajectory.jsonl에서 모델의 trajectory를 깔끔하게 추출합니다.

Usage:
    python scripts/extract_trajectories.py --input logs/focused2/unified_trajectory.jsonl --output logs/focused2/trajectories.json

    # 필터링 옵션
    python scripts/extract_trajectories.py --input logs/focused2/unified_trajectory.jsonl --output logs/focused2/sft_data.json \
        --min-score 1.0 --min-ndcg 0.5 --best-only

    # 샘플당 상위 K개 rollout만 유지 (best_only=K=1의 축약)
    python scripts/extract_trajectories.py --input logs/focused2/unified_trajectory.jsonl --output logs/focused2/sft_data.json \
        --min-score 1.0 --min-ndcg 0.5 --top-k 3

    # train1.parquet(=uid/format_version/messages) 호환 SFT 데이터셋 생성
    python scripts/extract_trajectories.py --input logs/focused2/unified_trajectory.jsonl --output logs/focused2/sft_train1_like.parquet \
        --export-train1-parquet --min-score 1.0 --min-ndcg 0.5 --best-only
"""

import json
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re


def load_log_file(log_path: str) -> List[Dict]:
    """JSONL 로그 파일 로드"""
    lines = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


def extract_trajectories(lines: List[Dict]) -> Dict[str, Any]:
    """
    로그에서 trajectory 추출

    Returns:
        {
            "metadata": {...},
            "samples": {
                "uid": {
                    "query": str,
                    "reference_answer": str,
                    "reference_basenames": [...],
                    "rollouts": [
                        {
                            "sample_idx": int,
                            "score": float,
                            "ndcg": float,
                            "generated_answer": str,
                            "retrieved_basenames": [...],
                            "trajectory": [
                                {
                                    "turn": int,
                                    "think": str,
                                    "search_query": str,
                                    "search_result": str (image basename),
                                    "is_final": bool
                                },
                                ...
                            ]
                        },
                        ...
                    ]
                },
                ...
            }
        }
    """
    # 메타데이터 추출 (run.start 이벤트에서)
    metadata = {}
    for l in lines:
        if l.get('event_type') == 'run.start':
            metadata = {
                'run_id': l.get('run_id'),
                'experiment_name': l.get('experiment_name'),
                'n_agent': l.get('n_agent'),
                'max_turns': l.get('config', {}).get('max_turns'),
                'model_path': l.get('config', {}).get('actor_rollout_ref', {}).get('model', {}).get('path'),
            }
            break

    # UID별 이벤트 그룹화
    uid_events = defaultdict(list)
    for l in lines:
        uid = l.get('uid')
        if uid:
            uid_events[uid].append(l)

    samples = {}

    for uid, events in uid_events.items():
        # rm.flash.detail에서 기본 정보 추출
        rm_events = [e for e in events if e.get('event_type') == 'rm.flash.detail']
        if not rm_events:
            continue

        query = rm_events[0].get('query', '').split('\nassistant')[0].strip()
        reference_answer = rm_events[0].get('reference_answer', '')
        reference_basenames = rm_events[0].get('reference_basenames', [])

        # sample_idx별로 그룹화 (각 rollout)
        sample_idx_set = set(e.get('sample_idx') for e in events if 'sample_idx' in e)

        rollouts = []
        for sample_idx in sorted(sample_idx_set):
            rollout_events = [e for e in events if e.get('sample_idx') == sample_idx]

            # rm.flash.detail 찾기
            rm = next((e for e in rollout_events if e.get('event_type') == 'rm.flash.detail'), None)
            if not rm:
                continue

            # model.plan 이벤트들 (턴별 정렬)
            plans = sorted(
                [e for e in rollout_events if e.get('event_type') == 'model.plan'],
                key=lambda x: x.get('turn_idx', 0)
            )

            # tool.search.detail 이벤트들 (검색 상세 정보)
            # search_detail은 request_idx로 구분됨 (sample_idx와 동일한 값)
            search_details = [e for e in events
                              if e.get('event_type') == 'tool.search.detail'
                              and e.get('request_idx') == sample_idx]

            # trajectory 구성
            trajectory = []
            search_idx = 0  # 검색 순서 인덱스

            for plan in plans:
                turn_idx = plan.get('turn_idx', 0)
                text = plan.get('text', '')

                # think와 search_query 파싱
                think = ''
                search_query = ''
                is_final = False

                if '<think>' in text and '</think>' in text:
                    think = text.split('<think>')[1].split('</think>')[0].strip()

                if '<search>' in text and '</search>' in text:
                    search_query = text.split('<search>')[1].split('</search>')[0].strip()

                if '<search_complete>' in text:
                    is_final = True

                if '<answer>' in text:
                    is_final = True

                # 검색 결과 찾기 (search_detail에서)
                search_result = None
                chosen_rank = None
                golden_in_results = None

                if search_query and search_idx < len(search_details):
                    sd = search_details[search_idx]
                    search_result = sd.get('chosen_basename')
                    chosen_rank = sd.get('chosen_rank')
                    golden_in_results = sd.get('golden_in_results')
                    search_idx += 1

                trajectory.append({
                    'turn': turn_idx,
                    'think': think,
                    'search_query': search_query,
                    'search_result': search_result,
                    'chosen_rank': chosen_rank,
                    'golden_in_results': golden_in_results,
                    'is_final': is_final,
                    'raw_text': text  # 원본 텍스트도 보존
                })

            rollouts.append({
                'sample_idx': sample_idx,
                'score': rm.get('final_score', 0),
                'ndcg': rm.get('ndcg', 0),
                'judge_score': rm.get('judge_score', 0),
                'generated_answer': rm.get('generated_answer', ''),
                'retrieved_basenames': rm.get('retrieved_basenames', []),
                'num_turns': len(plans),
                'trajectory': trajectory
            })

        samples[uid] = {
            'query': query,
            'reference_answer': reference_answer,
            'reference_basenames': reference_basenames,
            'num_rollouts': len(rollouts),
            'rollouts': rollouts
        }

    return {
        'metadata': metadata,
        'samples': samples
    }


DEFAULT_SYSTEM_PROMPT_V1 = (
    "You are a search agent.\n"
    "You must always begin with <think>...</think> showing your reasoning about the question.\n"
    "After reasoning, output exactly one action tag among <search>...</search> or <bbox>[x1, y1, x2, y2]</bbox> or <search_complete>true</search_complete>.\n"
    "Do not write anything before <think>. Keep actions on a new line after </think>.\n"
    "When using <search>, vary or refine the query using evidence from previous steps, and do not repeat the same query twice.\n"
)


_SEARCH_COMPLETE_TRUE_RE = re.compile(r"<search_complete>\s*true\s*</search_complete>", re.IGNORECASE)
_ACTION_TAG_RE = re.compile(r"</think>\s*\n?\s*(<search>|<bbox>|<search_complete>)", re.IGNORECASE)
_IM_END_RE = re.compile(r"<\|im_end\|>\s*$")


def _clean_plan_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    text = _IM_END_RE.sub("", text)
    return text.strip()


def _extract_question(query_field: Any) -> str:
    if not isinstance(query_field, str):
        return ""
    q = query_field.strip()
    m = re.search(r"\n\s*assistant\b", q, flags=re.IGNORECASE)
    if m:
        q = q[: m.start()].strip()
    return q


def _sort_key(event: Dict, fallback_idx: int) -> Tuple[float, int]:
    ts = event.get("ts")
    if ts is None:
        ts = event.get("timestamp")
    if ts is None:
        ts = 0.0
    try:
        ts = float(ts)
    except Exception:
        ts = 0.0
    return ts, fallback_idx


def _group_events_by_uid_sample(lines: List[Dict]) -> Dict[Tuple[str, int], List[Dict]]:
    grouped: Dict[Tuple[str, int], List[Dict]] = defaultdict(list)
    for idx, e in enumerate(lines):
        uid = e.get("uid")
        sample_idx = e.get("sample_idx")
        if not uid or sample_idx is None:
            continue
        try:
            sample_idx = int(sample_idx)
        except Exception:
            continue
        grouped[(uid, sample_idx)].append({**e, "_line_idx": idx})
    for k, evs in grouped.items():
        evs.sort(key=lambda x: _sort_key(x, x.get("_line_idx", 0)))
    return grouped


def _messages_from_events_v1(
    events: List[Dict],
    system_prompt: str,
    require_search_complete_true: bool = True,
    strict_action_tag: bool = True,
    synthesize_missing_tool_outputs: bool = True,
) -> Optional[List[Dict[str, Any]]]:
    rm = next((e for e in events if e.get("event_type") in ("rm.flash.detail", "rm.phase1.detail")), None)
    if not rm:
        return None

    question = _extract_question(rm.get("query"))
    if not question:
        return None

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    # generation.py의 search 이미지 선택 정책을 최대한 모사:
    # - search 결과 리스트에서 "이 rollout에서 아직 안 보여준 첫 이미지"를 선택
    # - 전부 이미 본 이미지라면 fallback으로 0번 사용
    seen_search_images: set[str] = set()

    pending_search = 0
    pending_bbox = 0
    saw_search_complete_true = False

    for e in events:
        et = e.get("event_type")
        if et == "model.plan":
            # If the previous turn requested a tool output but none arrived, synthesize a user message.
            # This avoids dropping most samples due to logging gaps (e.g., bbox failures not logged).
            if synthesize_missing_tool_outputs:
                while pending_search > 0:
                    messages.append(
                        {
                            "role": "user",
                            "content": "[System Error: Search did not return an image. Please try a different query.]",
                        }
                    )
                    pending_search -= 1
                while pending_bbox > 0:
                    messages.append(
                        {
                            "role": "user",
                            "content": "[System Error: BBox crop failed. Please adjust the bbox or search again.]",
                        }
                    )
                    pending_bbox -= 1

            text = _clean_plan_text(e.get("text"))
            if not text:
                continue

            # Strict validation: environment가 실제로 실행했을 가능성이 가장 높은
            # "첫 액션 태그"를 기준으로 tool 결과와 1:1로 맞추는 것이 목적.
            # (모델이 추가 태그를 더 출력했더라도, execution은 보통 첫 액션을 기준으로 진행됨)
            # - strict_action_tag=True일 때: 첫 액션을 식별/파싱할 수 없으면 rollout drop
            # - allow-invalid-actions를 쓰면 이 검증이 완화됨(상위에서 strict_action_tag=False로 전달)
            after_think = text
            think_end = text.lower().find("</think>")
            if think_end != -1:
                after_think = text[think_end + len("</think>"):]

            candidates = []
            for tag in ("<search>", "<bbox>", "<search_complete>"):
                pos = after_think.lower().find(tag)
                if pos != -1:
                    candidates.append((pos, tag))
            if not candidates:
                # fallback: 전체 텍스트에서 탐색
                for tag in ("<search>", "<bbox>", "<search_complete>"):
                    pos = text.lower().find(tag)
                    if pos != -1:
                        candidates.append((pos, tag))
            candidates.sort(key=lambda x: x[0])

            first_action = candidates[0][1] if candidates else None

            has_search = False
            has_bbox = False
            has_sc = False

            if first_action == "<search>":
                m = re.search(r"<search>(.*?)</search>", text, flags=re.DOTALL | re.IGNORECASE)
                has_search = bool(m and isinstance(m.group(1), str) and m.group(1).strip())
            elif first_action == "<bbox>":
                m = re.search(r"<bbox>\\s*\\[(.*?)\\]\\s*</bbox>", text, flags=re.DOTALL | re.IGNORECASE)
                if m and isinstance(m.group(1), str):
                    parts = [p.strip() for p in m.group(1).split(",")]
                    if len(parts) == 4:
                        try:
                            _ = [int(round(float(v))) for v in parts]
                            has_bbox = True
                        except Exception:
                            has_bbox = False
            elif first_action == "<search_complete>":
                m = re.search(r"<search_complete>(.*?)</search_complete>", text, flags=re.DOTALL | re.IGNORECASE)
                has_sc = bool(m)

            if strict_action_tag:
                if not _ACTION_TAG_RE.search(text):
                    return None
                if not (has_search or has_bbox or has_sc):
                    return None

            if has_search:
                pending_search += 1
            if has_bbox:
                pending_bbox += 1
            if _SEARCH_COMPLETE_TRUE_RE.search(text):
                saw_search_complete_true = True
            messages.append({"role": "assistant", "content": text})

        elif et == "tool.search.response":
            if pending_search <= 0:
                continue
            results = e.get("results")
            if not isinstance(results, list) or not results:
                return None

            image_file = None
            for item in results:
                if not isinstance(item, dict):
                    continue
                cand = item.get("image_file")
                if not isinstance(cand, str) or not cand:
                    continue
                if cand not in seen_search_images:
                    image_file = cand
                    break

            if image_file is None:
                first = results[0] if isinstance(results[0], dict) else None
                image_file = first.get("image_file") if first else None
            if not isinstance(image_file, str) or not image_file:
                return None
            seen_search_images.add(image_file)
            messages.append({"role": "user", "content": [{"image": image_file}]})
            pending_search -= 1

        elif et == "tool.bbox.result":
            if pending_bbox <= 0:
                continue
            crop_path = e.get("crop_path")
            if not isinstance(crop_path, str) or not crop_path:
                return None
            messages.append({"role": "user", "content": [{"image": crop_path}]})
            pending_bbox -= 1

    if synthesize_missing_tool_outputs:
        while pending_search > 0:
            messages.append(
                {
                    "role": "user",
                    "content": "[System Error: Search did not return an image. Please try a different query.]",
                }
            )
            pending_search -= 1
        while pending_bbox > 0:
            messages.append(
                {
                    "role": "user",
                    "content": "[System Error: BBox crop failed. Please adjust the bbox or search again.]",
                }
            )
            pending_bbox -= 1

    if pending_search != 0 or pending_bbox != 0:
        return None

    if require_search_complete_true and not saw_search_complete_true:
        return None

    return messages


def to_train1_parquet_rows(
    filtered_data: Dict[str, Any],
    events_by_uid_sample: Dict[Tuple[str, int], List[Dict]],
    system_prompt: str,
    strict_action_tag: bool = True,
    synthesize_missing_tool_outputs: bool = True,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    for uid, sample in filtered_data["samples"].items():
        for rollout in sample["rollouts"]:
            sample_idx = rollout["sample_idx"]
            events = events_by_uid_sample.get((uid, sample_idx))
            if not events:
                continue

            messages = _messages_from_events_v1(
                events,
                system_prompt=system_prompt,
                require_search_complete_true=True,
                strict_action_tag=strict_action_tag,
                synthesize_missing_tool_outputs=synthesize_missing_tool_outputs,
            )
            if not messages:
                continue

            rm = next((e for e in events if e.get("event_type") in ("rm.flash.detail", "rm.phase1.detail")), {})
            run_id = rm.get("run_id") or ""
            suffix = ""
            if isinstance(run_id, str) and run_id:
                suffix = run_id.split("_")[-1]
            out_uid = f"{uid}__s{sample_idx}"
            if suffix:
                out_uid = f"{out_uid}__{suffix}"

            rows.append(
                {
                    "uid": out_uid,
                    "format_version": "v1",
                    "messages": json.dumps(messages, ensure_ascii=False, separators=(",", ":")),
                }
            )

    return rows


def filter_trajectories(
    data: Dict[str, Any],
    min_score: float = 0.0,
    min_ndcg: float = 0.0,
    max_turns: Optional[int] = None,
    best_only: bool = False,
    min_success_rate: float = 0.0,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Trajectory 필터링

    Args:
        min_score: 최소 score (default: 0.0)
        min_ndcg: 최소 NDCG (default: 0.0)
        max_turns: 최대 턴 수 (default: None = 제한 없음)
        best_only: 샘플당 best rollout만 선택 (default: False)
        min_success_rate: 최소 성공률 (default: 0.0)
        top_k: 샘플당 상위 K개 rollout만 유지 (default: None=제한 없음)
    """
    filtered_samples = {}

    for uid, sample in data['samples'].items():
        # 성공률 계산
        total = len(sample['rollouts'])
        success = sum(1 for r in sample['rollouts'] if r['score'] >= min_score)
        success_rate = success / total if total > 0 else 0

        if success_rate < min_success_rate:
            continue

        # rollout 필터링
        filtered_rollouts = []
        for rollout in sample['rollouts']:
            if rollout['score'] < min_score:
                continue
            if rollout['ndcg'] < min_ndcg:
                continue
            if max_turns and rollout['num_turns'] > max_turns:
                continue
            filtered_rollouts.append(rollout)

        if not filtered_rollouts:
            continue

        # best_only / top_k 모드 (점수 우선, NDCG 차선)
        if best_only:
            top_k = 1
        if top_k is not None:
            try:
                k = int(top_k)
            except Exception:
                k = 0
            if k > 0:
                filtered_rollouts = sorted(filtered_rollouts, key=lambda r: (r['score'], r['ndcg']), reverse=True)[:k]

        filtered_samples[uid] = {
            **sample,
            'rollouts': filtered_rollouts,
            'num_rollouts': len(filtered_rollouts),
            'success_rate': success_rate
        }

    return {
        'metadata': data['metadata'],
        'filter_config': {
            'min_score': min_score,
            'min_ndcg': min_ndcg,
            'max_turns': max_turns,
            'best_only': best_only,
            'min_success_rate': min_success_rate,
            'top_k': top_k,
        },
        'samples': filtered_samples
    }


def to_sft_format(data: Dict[str, Any], include_raw: bool = False) -> List[Dict]:
    """
    SFT 학습용 포맷으로 변환

    Returns:
        [
            {
                "uid": str,
                "query": str,
                "reference_answer": str,
                "generated_answer": str,
                "score": float,
                "ndcg": float,
                "conversation": [
                    {"role": "user", "content": "..."},
                    {"role": "assistant", "content": "..."},
                    ...
                ]
            },
            ...
        ]
    """
    sft_data = []

    for uid, sample in data['samples'].items():
        for rollout in sample['rollouts']:
            conversation = []

            # 초기 질문
            conversation.append({
                "role": "user",
                "content": sample['query']
            })

            # 각 턴의 대화
            for step in rollout['trajectory']:
                # assistant의 think + search
                assistant_content = ""
                if step['think']:
                    assistant_content += f"<think>{step['think']}</think>\n"
                if step['search_query']:
                    assistant_content += f"<search>{step['search_query']}</search>"
                if step['is_final']:
                    assistant_content += f"\n<answer>{rollout['generated_answer']}</answer>"

                if assistant_content:
                    conversation.append({
                        "role": "assistant",
                        "content": assistant_content.strip()
                    })

                # 검색 결과 (user role로)
                if step['search_result']:
                    conversation.append({
                        "role": "user",
                        "content": f"[Image: {step['search_result']}]"
                    })

            entry = {
                "uid": uid,
                "sample_idx": rollout['sample_idx'],
                "query": sample['query'],
                "reference_answer": sample['reference_answer'],
                "generated_answer": rollout['generated_answer'],
                "score": rollout['score'],
                "ndcg": rollout['ndcg'],
                "num_turns": rollout['num_turns'],
                "conversation": conversation
            }

            if include_raw:
                entry["raw_trajectory"] = rollout['trajectory']

            sft_data.append(entry)

    return sft_data


def print_statistics(data: Dict[str, Any]):
    """통계 출력"""
    samples = data['samples']

    total_samples = len(samples)
    total_rollouts = sum(s['num_rollouts'] for s in samples.values())

    all_rollouts = []
    for s in samples.values():
        all_rollouts.extend(s['rollouts'])

    if not all_rollouts:
        print("No rollouts found.")
        return

    scores = [r['score'] for r in all_rollouts]
    ndcgs = [r['ndcg'] for r in all_rollouts]
    turns = [r['num_turns'] for r in all_rollouts]

    print("=" * 60)
    print("Trajectory Statistics")
    print("=" * 60)
    print(f"Total samples: {total_samples}")
    print(f"Total rollouts: {total_rollouts}")
    print(f"Score: mean={sum(scores)/len(scores):.3f}, min={min(scores):.2f}, max={max(scores):.2f}")
    print(f"NDCG:  mean={sum(ndcgs)/len(ndcgs):.3f}, min={min(ndcgs):.2f}, max={max(ndcgs):.2f}")
    print(f"Turns: mean={sum(turns)/len(turns):.1f}, min={min(turns)}, max={max(turns)}")

    # Score 분포
    score_1 = sum(1 for s in scores if s == 1.0)
    score_09 = sum(1 for s in scores if s >= 0.9)
    print(f"\nScore distribution:")
    print(f"  score == 1.0: {score_1} ({score_1/len(scores)*100:.1f}%)")
    print(f"  score >= 0.9: {score_09} ({score_09/len(scores)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Extract trajectories from training logs')
    parser.add_argument('--input', '-i', required=True, help='Input log file (jsonl)')
    parser.add_argument('--output', '-o', required=True, help='Output file (json)')
    parser.add_argument('--min-score', type=float, default=0.0, help='Minimum score filter')
    parser.add_argument('--min-ndcg', type=float, default=0.0, help='Minimum NDCG filter')
    parser.add_argument('--max-turns', type=int, default=None, help='Maximum turns filter')
    parser.add_argument('--best-only', action='store_true', help='Keep only best rollout per sample')
    parser.add_argument('--top-k', type=int, default=None,
                        help='Keep only top-K rollouts per sample (sorted by score, then ndcg). Ignored if <=0.')
    parser.add_argument('--min-success-rate', type=float, default=0.0, help='Minimum success rate filter')
    parser.add_argument('--sft-format', action='store_true', help='Output in SFT format')
    parser.add_argument('--include-raw', action='store_true', help='Include raw trajectory in SFT format')
    parser.add_argument('--stats', action='store_true', help='Print statistics')
    parser.add_argument('--export-train1-parquet', action='store_true',
                        help='Export SFT dataset in train1.parquet-compatible schema (uid/format_version/messages)')
    parser.add_argument('--system-prompt', type=str, default=DEFAULT_SYSTEM_PROMPT_V1,
                        help='System prompt to use for train1-compatible messages')
    parser.add_argument('--allow-invalid-actions', action='store_true',
                        help='Do not drop samples with invalid action-tag formatting (train1 export only)')
    parser.add_argument('--no-synthesize-missing-tools', action='store_true',
                        help='Do not synthesize missing tool outputs; drop rollouts when tool outputs are missing (train1 export only)')

    args = parser.parse_args()

    print(f"Loading {args.input}...")
    lines = load_log_file(args.input)
    print(f"Loaded {len(lines)} log entries")

    print("Extracting trajectories...")
    data = extract_trajectories(lines)
    print(f"Extracted {len(data['samples'])} samples")

    # 필터링
    if args.min_score > 0 or args.min_ndcg > 0 or args.max_turns or args.best_only or args.top_k or args.min_success_rate > 0:
        print(f"Filtering (score>={args.min_score}, ndcg>={args.min_ndcg}, turns<={args.max_turns}, best_only={args.best_only}, top_k={args.top_k})...")
        data = filter_trajectories(
            data,
            min_score=args.min_score,
            min_ndcg=args.min_ndcg,
            max_turns=args.max_turns,
            best_only=args.best_only,
            min_success_rate=args.min_success_rate,
            top_k=args.top_k,
        )
        print(f"After filtering: {len(data['samples'])} samples")

    # 통계 출력
    if args.stats:
        print_statistics(data)

    # 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.export_train1_parquet or output_path.suffix.lower() == ".parquet":
        # train1.parquet 스키마 호환 Parquet 저장
        events_by_uid_sample = _group_events_by_uid_sample(lines)
        rows = to_train1_parquet_rows(
            data,
            events_by_uid_sample=events_by_uid_sample,
            system_prompt=args.system_prompt,
            strict_action_tag=not args.allow_invalid_actions,
            synthesize_missing_tool_outputs=not args.no_synthesize_missing_tools,
        )
        print(f"Exporting train1-compatible rows: {len(rows)}")
        if not rows:
            raise SystemExit("No rows to export (filters too strict, or trajectories missing tool results).")

        import pandas as pd
        df = pd.DataFrame(rows, columns=["uid", "format_version", "messages"])
        df.to_parquet(output_path, engine="pyarrow", index=False)
        print(f"Saved to {output_path}")
        return

    if args.sft_format:
        output_data = to_sft_format(data, include_raw=args.include_raw)
        print(f"Converted to SFT format: {len(output_data)} entries")
    else:
        output_data = data

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Saved to {output_path}")


if __name__ == '__main__':
    main()
