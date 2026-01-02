# DDAI / GSPO codebase & runtime guide (agent handoff)

This repository trains a **Searcher** (a vision-language “tool-use” policy) with **GRPO/GSPO** so it can reliably:
1) generate `<search>` queries, 2) optionally crop images with `<bbox>`, and 3) stop with `<search_complete>`,
then a separate **Frozen Generator** (external model via API) produces the final `<answer>`.
Reward is computed with **Gemini (LLM-as-Judge)** on the `<answer>` plus **NDCG** for retrieval quality.

This document is written to help a future coding agent quickly understand **what runs**, **where it lives**, and **how to debug it**.

---

## Quick links (start here)

- Phase 2 training entrypoint (Focused Round 3): `gspo_phase2_focused_round3.sh`
- Trainer entrypoint: `verl/trainer/main_ppo.py` → `verl/trainer/ppo/ray_trainer.py`
- Generation environment (Searcher + tools + Frozen Generator): `vrag_agent/generation.py`
- Reward manager (Gemini Judge + NDCG + streaming): `verl/workers/reward_manager/rm_phase2.py`
- Search server (FastAPI): `search_engine/search_engine_api.py`
- Unified log writer: `verl/utils/unified_logger.py`
- API latency stats utility: `verl/utils/api_latency.py`

Related docs (under `docs/`):
- `docs/GSPO_PHASE2_GEMINI_FLASH_GUIDE.md` (flags + pipeline explanation)
- `docs/PHASE6_ASYNC_STREAMING_ARCHITECTURE.md` (prompt-level streaming reward architecture)
- `docs/RL_BATCH_AND_UPDATE_MECHANISM.md` (train_batch_size / n_agent / mini/micro batch)
- `docs/BATCH_SIZE_AND_TOOL_ASYNC_ANALYSIS.md` (why tool async is separate from PPO micro-batch)
- `docs/data_curation_pipeline.md` + focused round analyses

---

## docs/ index (what each document is for)

If you need to rebuild mental context quickly, read in this order:
1) `docs/ARCHITECTURE_SEARCHER_GENERATOR.md` → 2) `docs/GSPO_PHASE2_GEMINI_FLASH_GUIDE.md`
→ 3) `docs/PHASE6_ASYNC_STREAMING_ARCHITECTURE.md` → 4) batch/analysis docs.

Per-file summaries:

- `docs/ARCHITECTURE_SEARCHER_GENERATOR.md`
  - Defines the Searcher vs Frozen Generator split and the tag-based tool-use protocol.
  - Contains a code map and diagrams of the end-to-end trajectory structure.
  - Mentions both “answer-only judge” and “full-trajectory VLM judge” approaches; verify against the actual wired RM (`rm_phase2.py` vs `rm_phase2_trajectory.py`).

- `docs/GSPO_PHASE2_GEMINI_FLASH_GUIDE.md`
  - Walkthrough of Phase 2 training script arguments (Hydra overrides), with emphasis on GRPO, state masking, and the reward formula.
  - Explains `n_agent` vs `rollout.n`, GPU memory knobs, and troubleshooting (OOM/rate limits/search failures).

- `docs/PHASE6_ASYNC_STREAMING_ARCHITECTURE.md`
  - Explains why sequential “generation then frozen then reward” underutilizes GPU.
  - Describes prompt-level streaming reward: spawn background threads after prompt completion to overlap API work with ongoing Searcher turns.

- `docs/RL_BATCH_AND_UPDATE_MECHANISM.md`
  - Clear explanation of how `train_batch_size × n_agent` defines rollout collection volume, and how PPO updates are split into mini/micro batches.
  - Useful when diagnosing “why is this step taking so long?” vs “why am I OOM?”.

- `docs/BATCH_SIZE_AND_TOOL_ASYNC_ANALYSIS.md`
  - Key point: PPO `ppo_micro_batch_size_per_gpu` affects *update* memory only; it does not make tool calls “per-sample”.
  - Motivates explicit tool-async work in `vrag_agent/generation.py`.

- `docs/data_curation_pipeline.md`
  - Describes the funnel from the full SlideVQA set → curriculum buckets (A/B/0) → Focused Round datasets.
  - Explains why filtering GT errors and focusing compute improves RL sample-efficiency in a high-cost API loop.

- `docs/bucket_0_data_quality_analysis.md`
  - Manual audit of Bucket 0 (NDCG<0.1) samples; quantifies dataset quality issues and produces filtered subsets.

- `docs/ndcg_vs_judge_discrepancy_analysis.md`
  - Qualitative analysis of “Judge high, NDCG low” vs “Judge low, NDCG high” cases; highlights NDCG’s single-golden limitation.

- `docs/ground_truth_verification_report.md` + `docs/gt_verification_plan.md`
  - Finds substantial GT error rates in “NDCG high but judge=0” cases; documents filtering plans and impact on RL signal quality.

- `docs/focused_round1_hallucination_analysis.md`
  - Investigates when Frozen Generator answers correctly without golden images; separates true hallucination vs “same-document slide” effects.

- `docs/focused_round2_analysis.md`
  - Analyzes Focused Round 2 dataset composition (success history vs complete failures) and implications for where RL compute will pay off.

- `docs/HALF_OR_LESS_THINK_SFT_PIPELINE.md`
  - Pipeline to distill rare-success RL trajectories into SFT data by rewriting `<think>` only (keeping actions unchanged).

- `docs/DDAI-46.txt` / `docs/DDAI-47.txt`
  - Early design drafts for curriculum learning and focused RL “compute curriculum” (round-based approach).
  - Useful as intent/strategy context; validate against current scripts and configs.

---

## Glossary (project-specific)

- **Searcher**: the trainable VLM policy (typically Qwen2.5-VL-7B) generating tool-use tags:
  - `<think>...</think>`, `<search>...</search>`, `<bbox>[x1,y1,x2,y2]</bbox>`, `<search_complete>true</search_complete>`
  - Implemented as the PPO “actor rollout” model in `verl/`, invoked from `vrag_agent/generation.py`.
- **Retriever / Search Engine**: external service returning candidate images for a query.
  - Called from `vrag_agent/generation.py` via HTTP; server code exists in `search_engine/`.
- **Frozen Generator**: a *fixed* model queried via API to produce `<answer>...</answer>`.
  - Implemented in `vrag_agent/generation.py` (OpenAI-compatible async client preferred; DashScope sync fallback).
- **Reward Manager (Phase 2)**: computes per-sample reward using:
  - Gemini Judge on `<answer>` only (text; no images) + NDCG for retrieved-image quality.
  - Implemented in `verl/workers/reward_manager/rm_phase2.py`.
  - Note: an “evaluate full trajectory + images” variant exists in `verl/workers/reward_manager/rm_phase2_trajectory.py` but is not wired by default.
- **NDCG**: retrieval metric comparing retrieved image basenames vs reference (gold) basenames.
- **GRPO**: group-relative advantage estimator; uses `n_agent` samples per prompt as the group.
- **Streaming reward (Phase 6)**: when enabled, reward for a prompt starts as soon as all `n_agent` samples for that prompt finish (no need to wait for the whole batch).

### Resuming after a crash (Focused Round 3)

- `gspo_phase2_focused_round3.sh` sets `trainer.save_freq=4`, so a crash at step ~19 typically resumes from `global_step_16` (steps 17–19 will re-run).
- The script sets `trainer.resume_mode=auto`, so rerunning the same script (from repo root) should print `Load from checkpoint folder: .../global_step_16` and continue.
- If auto-resume fails (wrong CWD / different `trainer.experiment_name`), force it with a path override:
  - `trainer.default_local_dir=./checkpoints/gspo_phase2_gemini_flash_curriculum_focused_round_3`
  - `trainer.resume_mode=./checkpoints/gspo_phase2_gemini_flash_curriculum_focused_round_3/global_step_16`

---

## Repo layout (what lives where)

- `verl/`: RL trainer + worker framework (Ray-based). Key pieces:
  - `verl/trainer/main_ppo.py`: Hydra entrypoint; creates tokenizer/processor; selects reward manager; builds `RayPPOTrainer`.
  - `verl/trainer/ppo/ray_trainer.py`: main training loop; calls generation; computes reward; runs GRPO/GSPO update.
  - `verl/workers/reward_manager/`: reward managers (Phase 1 / Phase 2 variants).
  - `verl/utils/unified_logger.py`: single JSONL “unified log” sink (Ray actor or local writer).
- `vrag_agent/`: the “environment” around the actor:
  - `vrag_agent/generation.py`: Phase 2+ environment (tools + Frozen Generator + streaming hooks).
  - `vrag_agent/generation_phase1.py`: Phase 1 environment (no Frozen Generator; faster format/tool training).
- `search_engine/`: retrieval server and optimized GPU search engine:
  - `search_engine/search_engine_api.py`: FastAPI `/search` endpoint, multi-GPU round-robin.
  - `search_engine/search_engine.py`: ColQwen-based embedding search; aggressive GPU caching.
- `scripts/`: data curation, trajectory extraction, SFT dataset utilities.
- `docs/`: design notes, analysis reports, and training guides (some may describe variants not currently wired).

---

## End-to-end runtime (Phase 2 training)

### 0) Shell script config (Focused Round 3)

`gspo_phase2_focused_round3.sh` does three important things:
1) Exports **API keys** and **debug/log flags** (Gemini/OpenAI, unified logging, latency stats).
2) Chooses batch sizes and “compute budget” (`train_batch_size`, `n_agent`, `max_turns`, timeouts).
3) Runs `python3 -m verl.trainer.main_ppo` with Hydra overrides:
   - `algorithm.adv_estimator=grpo`
   - `reward_model.reward_manager='rm'` (Phase 2 reward manager)
   - `retriever.url=...` (search server)
   - `+frozen_generator.*=...` (Frozen Generator settings passed into `vrag_agent/generation.py`)

### 1) Python entrypoint → Ray bootstrap

- `verl/trainer/main_ppo.py` starts a local Ray runtime (if not already running) and launches a `TaskRunner` Ray actor.
- `TaskRunner.run()` loads the model checkpoint locally, creates tokenizer/processor, instantiates worker groups, and constructs `RayPPOTrainer`.

Reward manager wiring:
- If `reward_model.reward_manager == "rm_phase1"` → uses Phase 1 generation manager (`vrag_agent/generation_phase1.py`).
- Else (Phase 2+) → uses `vrag_agent/generation.py` and `verl/workers/reward_manager/rm_phase2.py`.

### 2) Training loop (RayPPOTrainer.fit)

In `verl/trainer/ppo/ray_trainer.py` (high-level per-step):
1) Load a batch of prompts (`train_batch_size`).
2) **Repeat** the batch `n_agent` times (interleaved) to create a rollout batch of size `train_batch_size × n_agent`.
3) Call `generation_manager.run_llm_loop(...)` to produce the multi-turn tool-use trajectory and final `<answer>`.
4) Compute old logprobs / ref logprobs / values (depending on config).
5) Compute rewards:
   - Batch mode: `reward_fn(data)` (`RMManager.__call__`)
   - Streaming mode: `reward_fn.start_streaming_mode()` before generation, then `reward_fn.wait_and_get_streaming_rewards()` after generation
6) Compute GRPO advantage (grouped by prompt) and apply GSPO policy update.

### 3) Generation loop (Searcher + tools + Frozen Generator)

`vrag_agent/generation.py::LLMGenerationManager.run_llm_loop()` implements the environment:

Per-turn loop (up to `max_turns`):
1) Call the **Searcher** model to produce an action string (tags).
2) Parse action tags:
   - `<search>query</search>` → call Search API to get images
   - `<bbox>[x1,y1,x2,y2]</bbox>` → crop last retrieved image and attach as additional image evidence
   - `<search_complete>true</search_complete>` → mark sample done
3) Convert tool outputs into the next observation (new image tokens, error messages, etc.) and continue.

Tool concurrency:
- `_execute_async_pipeline(...)` overlaps **async HTTP search** with **CPU bbox parsing** to reduce head-of-line blocking.
- Search uses `aiohttp` per pipeline execution (`_async_search_batches_aio`).

Search API contract (what `vrag_agent/generation.py` expects):
- Request: `POST $SEARCH_URL` with JSON list of objects like:
  - `{"query": "...", "id": "123", "request_idx": 7}`
- Response: JSON list of objects that include:
  - `request_idx` (int) and `results` (list)
  - each `results` item is expected to contain an `image_file` path (used for downstream loading/cropping)

⚠️ The included server in `search_engine/search_engine_api.py` currently exposes a different interface (`GET /search` with `queries=[...]`), so if you run the local server code you may need an adapter or update the endpoint to match the contract above.

Frozen Generator:
- After search completion, the pipeline appends `<answer>...</answer>` to the response:
  - If streaming reward is enabled:
    - full prompt-group completion → background thread triggers Frozen + RM submit immediately.
    - partial completion (some samples never reach `search_complete`) → end-of-loop “partial submit” evaluates only successful samples and leaves failures at 0.
  - Otherwise: Frozen Generator is called in batch after the main loop (no streaming overlap).

OpenAI-compatible call shape (Responses API):
- Implemented in `_call_frozen_generator_async_single_impl(...)` using `AsyncOpenAI().responses.create(...)`.
- Inputs:
  - `instructions=sys_prompt`
  - `input=[{"role":"user","content":[{"type":"input_image","image_url":"data:..."}, {"type":"input_text","text":"Question: ..."}]}]`
  - `reasoning={"effort": $FROZEN_REASONING_EFFORT}`
  - `max_output_tokens=frozen_max_tokens`

Thread-safety note (important for Phase 6):
- **AsyncOpenAI client is not thread-safe across event loops.**
- The implementation creates a **new client per async batch execution** (`_make_openai_async_client()` inside `_call_frozen_generator_batch_async`), then closes it best-effort.

DashScope fallback:
- If OpenAI-compatible async client is unavailable, falls back to DashScope `MultiModalConversation.call(...)`.

### 4) Reward computation (Gemini Judge + NDCG)

Default Phase 2 reward manager: `verl/workers/reward_manager/rm_phase2.py::RMManager`

What it evaluates:
- Extracts `<answer>...</answer>` from the final response string.
- Calls Gemini **text-only** judge prompt (no images) with Structured Output schema:
  - Returns `{"score": <float 0..1>}`
- Computes NDCG from retrieved basenames vs reference basenames.
- Combines into final score:
  - `final_score = RM_JUDGE_COEF * judge_score + RM_NDCG_COEF * ndcg_value`
  - (Optional) format gating via `RM_FORMAT_COEF` + `custom_reward_function.*` (not supported in streaming mode).

Streaming reward mode:
- `start_streaming_mode()` starts a worker thread with its own asyncio event loop.
- `submit_prompt(uid, sample_indices, samples_data)` is called from `vrag_agent/generation.py`.
  - **Full submit**: when *all* `n_agent` samples in a prompt-group emit `<search_complete>true</search_complete>` (old Phase 6 behavior).
  - **Partial submit**: at `run_llm_loop()` end, for prompt-groups that were not fully submitted but have some `search_complete` samples,
    submit only those successful sample indices; the rest stay **0 reward** with no FG/RM call.
- `wait_and_get_streaming_rewards()` blocks until all queued prompts are processed (or timeout).

Variant: “VLM as Judge” (images + full trajectory)
- Exists as `verl/workers/reward_manager/rm_phase2_trajectory.py`.
- Not selected by `reward_model.reward_manager='rm'` today; switching requires wiring changes.
- Some docs describe this variant; always verify which RM file you are actually using.

---

## Logging & observability

### Unified log (single JSONL)

`verl/utils/unified_logger.py` provides a best-effort centralized JSONL sink:
- Enable: `UNIFIED_LOG_ENABLE=1`
- Path: `UNIFIED_LOG_PATH=./logs/.../unified_trajectory.jsonl`
- Writer:
  - If Ray is initialized → uses a Ray actor writer (batched writes).
  - Else → uses a local buffered writer.

Common event types (non-exhaustive):
- Trainer: `train.step.start`, `train.generation.start`, `train.generation.end`
- Tool search: `tool.search.request`, `tool.search.response`
- Actor plan text: `model.plan`
- Reward summaries: `rm.batch.summary` (batch mode), plus streaming equivalents
- Detail I/O logs (if enabled): `rm.gemini.detail`, `frozen.response`
- API latency stats: `api.gemini.*`, `api.frozen.*`

W&B streaming reward metrics (PPO):
- `reward/streaming_final_score_mean`, `reward/streaming_vlm_score_mean`: mean over **scored samples only** (excludes implicit zeros).
- `reward/streaming_samples_scored`: number of samples that received a score in `reward_tensor`.
- `reward/streaming_sample_coverage`: `samples_scored / (train_batch_size*n_agent)`.
- `reward/streaming_prompts_processed`: number of submitted prompt-groups (may be `< train_batch_size` if no sample completed).

### “Detail” I/O logs: keep unified log, hide console

There are two different categories of “verbosity”:

1) **Legacy console dumps** (raw prompts/responses printed to stdout)
   - `GEMINI_DEBUG_VERBOSE` (rm_phase2): 0/1/2
   - `FROZEN_DEBUG_VERBOSE` (generation): 0/1/2

2) **Per-call input/output detail logs** (recorded to unified JSONL; optional console mirror)
   - Gemini Judge:
     - `GEMINI_DETAIL_LOG=1` enables unified event `rm.gemini.detail`
     - `GEMINI_DETAIL_CONSOLE=0` disables console prints while keeping unified log
   - Frozen Generator:
     - `FROZEN_DETAIL_LOG=1` enables unified event `frozen.response`
     - `FROZEN_DETAIL_CONSOLE=0` disables console prints while keeping unified log

This split is intentional: you can keep rich unified JSONL traces without flooding stdout.

### API latency stats (statistical timing, includes max)

`verl/utils/api_latency.py` emits:
- per-call events (optional) and/or
- summary stats events containing distribution metrics including **max**.

Config (global or per-component prefix):
- Enable: `API_LATENCY_ENABLE=1` (or `GEMINI_API_LATENCY_ENABLE=1`, `FROZEN_API_LATENCY_ENABLE=1`)
- Console mirror: `API_LATENCY_CONSOLE=0|1` (unified log is independent)
- Per-call raw events: `API_LATENCY_PER_CALL=0|1`
- Summary cadence:
  - `API_LATENCY_SUMMARY_EVERY_N=200`
  - `API_LATENCY_SUMMARY_EVERY_S=30`
- Reservoir sample size for quantiles: `API_LATENCY_SAMPLE_SIZE=5000`

Measured “spans” currently instrumented:
- Frozen(OpenAI-compatible):
  - `frozen.openai.semaphore_wait`
  - `frozen.image_base64_encode`
  - `openai.responses.create`
- Gemini:
  - `gemini.semaphore_wait`
  - `gemini.generate_content_async`

Interpretation tips:
- `p95/p99/max` capture long-tail stalls (rate limit, transient network, backend saturation).
- Compare `semaphore_wait` vs request latency to see if you are queueing locally or stalling remotely.

---

## Common failure modes (and what to check)

### 1) “protocol.resume_writing() failed … Non-thread-safe operation … other event loop”

Symptom (seen in logs):
- uvloop/anyio stack trace complaining about non-thread-safe event-loop operation.
- Training appears to “hang” around Phase 6 background processing.

Root cause:
- Sharing a single async HTTP client across multiple threads/event loops.

Fix in current code:
- `vrag_agent/generation.py` uses **per-batch/per-loop** `AsyncOpenAI` clients.

Regression test:
- `tests/test_phase5_openai_thread_safety.py` asserts the async client is not shared across threads and validates Responses API call shape.

### 2) Search API head-of-line blocking / timeouts

If many samples issue `<search>` at once, synchronous waiting can stall everything.
Current mitigation:
- Async pipeline overlaps search with bbox processing.
Check:
- `SEARCH_TIMEOUT`, `SEARCH_BATCH_SIZE`, `SEARCH_MAX_WORKERS`
- Search server health endpoint if you run your own: `search_engine/search_engine_api.py:/health`

### 3) Gemini rate limits / backoff effects

Symptoms:
- Reward time dominates step wall time; p99/max spikes in `api.gemini.request.stats`.
Controls:
- `reward_model.max_concurrent_requests` (Hydra) and internal semaphores.

### 4) Frozen Generator long-tail latency / token length

If `frozen_max_tokens` is large, requests can legitimately take longer.
Controls:
- `frozen_generator.max_tokens`, `frozen_generator.total_timeout`, `frozen_generator.async_wrapper_timeout`
- Client-level timeout is currently set in `_make_openai_async_client(timeout=60.0)`; align it if you expect longer generations.

---

## Tests & local validation

- Thread-safety + payload-shape tests (unittest): `tests/test_phase5_openai_thread_safety.py`
  - Run: `python -m unittest -q tests.test_phase5_openai_thread_safety`

---

## Git history highlights (recent, high-signal commits)

Run `git log --oneline -n 80` for full context. Notable milestones:

- `88685a1` feat(logging): unified trajectory logger
- `e75c044` feat(training): Phase 1/2 separation + unified logging
- `22131f8` feat(reward): unified logging + configurable coefficients
- `ac69b16` feat(generation): unified logging + configurable reasoning effort
- `d9f1699` feat: Phase 5-7 async optimizations (generation pipeline)
- `fafaac5` adjust Frozen Generator OpenAI call for reasoning models
- `e8862c2` feat(search-engine): multi-GPU round-robin load balancing
- `5bde48b` feat(reward-manager): Gemini detail logging + thread-safe factory pattern
- `0118826` feat(generation): timeout controls + NDCG golden basenames mapping fix
- `c6ab71d` chore(config): focused round training configs

---

## Where docs may diverge from current wiring

Some docs (e.g. `docs/ARCHITECTURE_SEARCHER_GENERATOR.md`) describe a “VLM as Judge” setup where Gemini receives images and the *full* response (think/search/bbox/answer).
In current default training (`reward_model.reward_manager='rm'`), the wired reward manager is:
- `verl/workers/reward_manager/rm_phase2.py` (LLM-as-Judge, text-only `<answer>` evaluation + NDCG)

If you want the VLM trajectory+images judge, you must re-wire to use:
- `verl/workers/reward_manager/rm_phase2_trajectory.py` (and update selection in `verl/trainer/main_ppo.py` / `verl/workers/reward_manager/__init__.py`)
