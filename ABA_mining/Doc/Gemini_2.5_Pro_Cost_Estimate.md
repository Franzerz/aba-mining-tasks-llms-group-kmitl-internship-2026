# Gemini 2.5 Pro — Cost & Time Estimate for Task 1, 2, 3

**Date:** 2026-08-21
**Purpose:** Estimate token usage, cost, and runtime if Task 1, Task 2, and Task 3 were run
against Gemini 2.5 Pro (via the professor's API key) instead of the local llama3.2 model,
before actually spending API budget on it.

---

## Bottom line

| | Calls | Input tokens | Output tokens | **Cost (USD)** | Time* |
|---|---|---|---|---|---|
| Task 1 | ~12,900 | ~43.5M | ~1.35M | **~$67.87** | ~1.4 hr |
| Task 2 | ~38,100 | ~25.0M | ~0.67M | **~$37.98** | ~4.2 hr |
| Task 3 | ~29,400 | ~8.8M | ~0.51M | **~$16.17** | ~3.3 hr |
| **Total** | **~80,400** | **~77.3M** | **~2.5M** | **≈ $122** | **≈ 8.9 hr** |

\* Time assumes a Tier 1 paid-tier rate limit of 150 requests/minute (see [Methodology](#methodology--sources) — **verify your professor's actual tier limits** at [aistudio.google.com/rate-limit](https://aistudio.google.com/rate-limit), since this is not published per-account and could be higher or lower).

**This is an estimate, not a quote** — see [Accuracy caveats](#accuracy-caveats) below for what could make the real number differ.

---

## How the scope was defined

Rather than guessing typical prompt/output sizes, this reuses the **exact experimental scope
already completed with llama3.2** in this project — same review/instance counts, same number
of prompt versions, same number of repeated runs — so the estimate reflects real, already-
validated work rather than a hypothetical:

- **Task 1:** the "combined" experiment (all 7 annotation rules, full topic/span/sentiment
  output schema) — 784 reviews × 3 sub-runs per review (`runs_per_id`) × 3 independent outer
  runs, matching the 3 completed `run1/run2/run3.jsonl` files.
- **Task 2:** all 6 prompt-engineering versions (`generator_v1.txt` … `v6.txt`) × 2,078
  ground-truth instances × 3 runs each, matching the completed sweep.
- **Task 3:** all 8 prompt versions (`zero_shot`, `one_shot`, `contrary_v1`–`v6`) × the 4
  topics with real gold data (check-in, check-out, price, staff — 1,215 pairs/version/run) ×
  3 runs, matching the completed sweep (see `Task3_Implementation_Report.md`).

If you only want a subset (e.g. just Task 3, or just the winning `contrary_v5` prompt instead
of all 8), the cost scales down roughly linearly with however many fewer calls that scope
needs — see the per-version breakdown tables below.

---

## Task 1 detail

| | Value |
|---|---|
| Reviews | 784 |
| Sub-runs per review (`runs_per_id`) | 3 |
| Outer runs | 3 |
| Base generation calls | 7,056 |
| Avg. generation prompt size | **5,541 tokens** (7 rule blocks + review text — this is the biggest single driver of cost) |
| Avg. generation output size | 105 tokens |
| Avg. retries per call (cascading JSON validator) | 0.824 — measured directly from real retry counts in the completed llama3.2 run |
| Extra validator calls (from retries) | ~5,811 |
| Avg. validator retry prompt size | **751 tokens** (smaller — review + candidate JSON + error list, no rule blocks) |
| **Total calls (generation + validator retries)** | **~12,867** |

The 82% average retry rate reflects llama3.2 struggling with the complex 11-topic JSON schema
— Gemini 2.5 Pro may need fewer retries (lower cost) or a similar/higher rate (similar cost);
this can't be known until it's actually run. Using llama3.2's real retry rate is a reasonable
starting assumption, not a guarantee.

## Task 2 detail (per prompt version)

| Version | Avg. input tokens | Avg. output tokens | Calls | Notes |
|---|---|---|---|---|
| generator_v1 | 1,145 | 14 | 6,234 | Longest prompt (full rule list + 5-shot) |
| generator_v2 | 566 | 13 | 6,234 | |
| generator_v3 | 607 | 11 | 6,234 | |
| generator_v4 | 430 | 15 | 6,234 | Shortest prompt |
| generator_v5 | 606 | 40 | 6,234 | Highest retry rate (10.5%) and output size |
| generator_v6 | 585 | 11 | 6,234 | |

(6,234 = 2,078 instances × 3 runs per version)

## Task 3 detail (per prompt version)

| Version | Avg. input tokens | Avg. output tokens | Calls |
|---|---|---|---|
| zero_shot | 55 | 1 | 3,645 |
| one_shot | 83 | 1 | 3,645 |
| contrary_v1 | 623 | 1 | 3,645 |
| contrary_v2 | 251 | 23 | 3,645 |
| contrary_v3 | 317 | 43 | 3,645 |
| contrary_v4 | 332 | 2 | 3,645 |
| contrary_v5 | 327 | 1 | 3,645 |
| contrary_v6 | 410 | 64 | 3,917 |

`contrary_v5` — the prompt version that scored best on F1 in the llama3.2 evaluation — is also
one of the cheapest per call (short prompt, short "Yes"/"No" output). `contrary_v3` and `v6`
(chain-of-thought / recipe style) cost more per call because the model has to write out
reasoning before the answer, not just because of prompt length.

---

## Methodology & sources

- **Pricing** — Gemini 2.5 Pro standard tier, confirmed directly against Google's official
  pricing page: **$1.25 / 1M input tokens, $10.00 / 1M output tokens** (for prompts ≤200k
  tokens — everything here is far under that; output pricing already includes "thinking"
  tokens, billed at the same rate, not separately).
  Source: [ai.google.dev/gemini-api/docs/pricing](https://ai.google.dev/gemini-api/docs/pricing)
- **Token counts** — measured with `tiktoken`'s `cl100k_base` encoding as a standard
  cross-model proxy, applied to:
  - **Real rendered prompts** — built using this project's actual prompt-construction code
    (`build_modular_prompt`, `render_prompt`) with real review/instance data, not hand-typed
    examples.
  - **Real generated outputs** — sampled directly from the completed llama3.2 run files
    (`outputs/task1/**/*.jsonl`, `outputs/task2/**/*.jsonl`, `outputs/task3/**/log/*.csv`).
  - Gemini's own tokenizer will count somewhat differently than `cl100k_base` — expect this
    estimate to be within roughly **±15–20%** of Gemini's actual billed token count, not exact.
- **Rate limit for the time estimate** — 150 requests/minute is the commonly-cited Gemini 2.5
  Pro **Tier 1** (paid) limit; Google does not publish per-model RPM/TPM figures on a public
  page — your professor's actual limit depends on their specific account tier and is only
  visible in their own [AI Studio rate-limit dashboard](https://aistudio.google.com/rate-limit).
  At 80,417 total calls, the **request-count limit (150 RPM) is the binding constraint**, not
  the token-throughput limit (2M tokens/minute would only take ~40 minutes for this volume) —
  so the ~8.9 hour estimate is driven by number of calls, not by data volume.

## Accuracy caveats

- **Output length may differ from llama3.2.** Gemini 2.5 Pro is a "thinking" model by default
  and may produce longer (or shorter) responses than llama3.2 for the same prompt, especially
  on the chain-of-thought Task 3 versions (`contrary_v3`, `v6`) — this directly affects output
  cost, which is 8× more expensive per token than input.
- **Retry rates may differ.** Task 1's cost is dominated by validator retries measured from
  llama3.2's real (fairly high, 82%) failure rate on the complex JSON schema — a stronger model
  may need meaningfully fewer retries, which would lower Task 1's cost substantially.
- **Time estimate assumes sustained rate-limit-saturating throughput** with enough parallel
  requests in flight — actual wall-clock time could be longer if per-request latency (Gemini's
  thinking time) limits how many concurrent requests are practical to run, or shorter if the
  actual account tier has a higher RPM limit than the Tier 1 figure used here.
- **Batch API discount not applied.** Google offers a 50% discount ($0.625/$5 per 1M
  input/output tokens) via the Batch API for non-time-sensitive workloads — since none of this
  needs a live/interactive response, running it through the Batch API instead of the standard
  API could cut the total cost to **roughly $61** instead of $122, at the cost of slower
  turnaround (batch jobs aren't guaranteed to complete within a fixed time window). Worth
  asking your professor whether this is an option for their key.

---

## Files

Raw measurement data (per-call token samples, retry rates, all figures behind the tables
above) is in `estimate_gemini_cost.py` — rerun it any time (e.g. after adding `llama4:scout` or
after adjusting scope) to get updated numbers from the current state of the project's data.
