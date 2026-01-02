# Focused Round 3 v2: Behavior Shift & “Collapse” Signals (Log-Based)

This note summarizes findings from qualitative + aggregate analysis of unified logs:

- **Focused Round 2 (pre-SFT)**: `logs/focused2/unified_trajectory.jsonl`
  - Primary run analyzed: `gspo_phase2_gemini_flash_curriculum_focused_round_2_20251230_193750_2c451b12`
- **Focused Round 3 v2 (post-SFT)**: `logs/focused3_v2/unified_trajectory.jsonl`
  - Primary run analyzed: `gspo_phase2_gemini_flash_curriculum_focused_round_3_v2_20260102_130750_ce431b83`

Important: **These runs differ in both checkpoint/training stage and config** (notably dataset + `max_turns`), so “SFT caused X” is not provable without a controlled A/B run. We treat these as *observed deltas*.

---

## 1) Config Context (why comparisons are confounded)

### Focused Round 2 (pre-SFT)
- Dataset: `data/focused_round2.parquet`
- `max_turns=15`
- Reward weights: `RM_JUDGE_COEF=1.0`, `RM_NDCG_COEF=0.0` (Judge-only; NDCG logged but not rewarded)
- Streaming reward enabled

### Focused Round 3 v2 (post-SFT checkpoint used)
- Dataset: `data/focused_round1.parquet`
- `max_turns=10` (forced completion at final turn)
- Reward weights: `RM_JUDGE_COEF=1.0`, `RM_NDCG_COEF=0.0` (Judge-only; NDCG logged but not rewarded)
- Streaming reward enabled

---

## 2) Aggregate Behavior Changes (observed)

### Action length

From run-level aggregates:

- **Focused2** (run `..._2c451b12`):
  - mean `#<search>` ≈ 5.93 (p50=6, p90=11)
  - mean `#<bbox>` ≈ 2.96 (p50=2)
  - `#search == 0` rollouts: 0%
  - `#search <= 5` rollouts: ≈ 49.2%

- **Focused3 v2** (run `..._ce431b83`):
  - mean `#<search>` ≈ 2.61 (p50=2)
  - mean `#<bbox>` ≈ 3.35 (p50=3)
  - `#search == 0` rollouts: ≈ 10.2% (52/510)
  - `#search <= 5` rollouts: ≈ 95.9%

Interpretation:
- v2 learned/shifted toward **fewer searches** and **more bbox usage**.
- The new **non-trivial 0-search behavior** is **RAG-misaligned** and a strong “collapse” indicator.

### Query repetition / degeneration

- **Focused2** showed extremely high query repetition:
  - duplicate query rollouts ≈ 59%
  - consecutive repeated query rollouts ≈ 57%
- **Focused3 v2** repetition dropped sharply:
  - duplicate query rollouts ≈ 2.4%
  - consecutive repeats ≈ 2.2%

Interpretation:
- Repetition decreased, but some samples exhibit **format/behavior collapse** via “empty plan” and “searchless bbox loops”.

### RM / retrieval signal (logged, not rewarded)

Note: both runs are Judge-only (`RM_NDCG_COEF=0`), but NDCG is logged.

- **Focused2**:
  - `judge_mean≈0.274`, `ndcg_mean≈0.192`
  - `ndcg>0` rate ≈ 39.7%
- **Focused3 v2**:
  - `judge_mean≈0.206`, `ndcg_mean≈0.090`
  - `ndcg>0` rate ≈ 17.6%

Interpretation:
- Under v2, retrieval success (ndcg>0) appears to **drop substantially**, consistent with “less search” + “more bbox” but not necessarily “better retrieval”.

---

## 3) “Collapse” Signals to Watch (log-level)

### A) 0-search rollouts (new in v2)

Example (Focused3 v2): `train_3639__s13`
- Observed: `#search=0`, `#bbox=9`, `retrieved_basenames=[]`
- RM judged correct (`judge=1`) with answer `$2,000` despite no retrieval evidence.

Risk:
- This can “reward” policies that do not retrieve evidence (bad for ViDoSeek / RAG).
- Could also reflect missing tool logging (verify `tool.*` events for the rollout).

### B) “Empty” plan outputs / weak action tagging

We measured the fraction of `model.plan` events without an action tag and the fraction that are literally `<|im_end|>`:

- **Focused2**:
  - `no_action_rate≈33.6%`
  - `only_im_end_rate≈4.6%`
- **Focused3 v2**:
  - `no_action_rate≈31.8%`
  - `only_im_end_rate≈8.8%` (worse)

Interpretation:
- While “no_action” is partly expected (some events may be truncated/formatting), a higher “only `<|im_end|>`” fraction is a clear quality regression.

### C) GT error (reward contamination) surfaced via manual audit

Example (Focused3 v2): `train_8289__s331`
- Retrieval contained the golden page (`ndcg≈0.631`, retrieved included `8289_10`).
- Visual inspection shows the entity mapped to the IP/port is **“Zhora”**.
- Reference answer in the dataset is **“Roy”**; judge returns 0.

Risk:
- GT errors directly poison RL reward, especially in hard retrieval settings.

---

## 4) Qualitative Comparison: Pre-SFT vs Post-SFT behavior shape

High-level differences visible in trajectories:

- **Pre-SFT (Focused2)**:
  - Frequent “search loop” behavior: repeated or near-duplicate queries.
  - Longer search horizons (p50 search=6).
  - Less bbox-heavy.

- **Post-SFT / v2 run**:
  - Much shorter search horizons (p50 search=2).
  - Increased bbox usage.
  - A notable minority of rollouts execute **bbox without any search**, which is likely invalid and/or log-misaligned.
  - Retrieval quality metrics (ndcg>0 rate) appear lower.

---

## 5) Practical Recommendations (non-code, operational)

To align training with ViDoSeek-style retrieval benchmarks (multi-doc, layout-heavy):

1) **Exclude 0-search rollouts from SFT curation** (and consider filtering them out for RL-based curation).
2) **Prefer trajectories with non-empty retrieved evidence** (`retrieved_basenames` non-empty, ndcg>0) when building SFT datasets.
3) For future comparisons, keep runs isolated by log path (avoid mixing multiple run_ids in one JSONL when possible).
4) If action editing is attempted during SFT rewriting, keep a strict safety policy:
   - “think-only” as default, and only remove actions under narrow, verifiable failure patterns.

