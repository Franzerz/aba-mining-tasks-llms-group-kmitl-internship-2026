# Task 3 Implementation Report — Contrary (Attack-Relation) Classification

**Date:** 2026-07-31 (last updated 2026-08-09)
**Scope:** ABA Mining pipeline, Task 3 (following Task 1 — topic/span/sentiment mining, and
Task 2 — Body-literal generation)

---

## 1. Background

In the ABA (Assumption-Based Argumentation) labelling scheme, each mined **Body** literal
(a support phrase extracted in Task 2, e.g. `easy_check-in`) is automatically paired with a
**Contrary** literal derived by a fixed rule (per `Doc/ABA_Labelling_Rules_extracted.txt`):

- Positive body → contrary is `no_evident_not_{body}` (e.g. `no_evident_not_easy_check-in`)
- Negative body → contrary is `have_evident_{body}` (e.g. `have_evident_no_soundproof`)

That auto-derived contrary is trivial — it only attacks its own body. **Task 3's job is harder:**
given that auto-derived Assumption phrase (**Phrase A**) and a *different* Proposition phrase
mined from another complaint about the same aspect (**Phrase B**), decide whether B is *also*
contrary to A. This is what wires additional attack relations into the ABA argumentation
framework, beyond each phrase's own trivial self-contrary.

Example (from the check-in gold data):
- Phrase A: `no_evident_not_easy_check-in`
- Phrase B: `customer_wait_10_minute_for_greet`
- Gold answer: **Yes** (waiting 10 minutes to be greeted is evidence check-in was *not* easy)

## 2. Starting point

My advisor pointed me to `mu_work/Task3`, a previous team's implementation of this task. Their
approach: a plain Yes/No LLM prompt ("Is phrase B contrary to phrase A?"), tested 0-shot and
1-shot, across 5 models (deepseek-r1:7b, gemma3-4b, qwen2.5-7b, gemini-2.5-pro, gpt-4o), 3 repeated
runs per configuration for consistency, scored against a human-annotated ("Silver") gold set with
4 aspects (check-in, check-out, price, staff) and 4 polarity categories per aspect.

There was already a **non-functional stub** at `run_task3.py` in this project — it sketched the
intended direction (config-driven paths, dataclass-based CLI args) but had undefined variables
(`df`, `CATEGORIES`, `input_files`) and would raise `NameError` immediately if run.

## 3. What I did

### 3.1 Investigated the data before writing code

The gold data in `Dataset/Task3/*.csv` (one CSV per aspect, exported from mu_work's "Silver"
verification spreadsheets) turned out to have **inconsistent column headers across files**:

| File | Contrary-form column | Proposition column |
|---|---|---|
| check-in / check-out / price | `Original A` | `B` |
| staff | `Original Assumption` | `Proposition` |

**Important bug I found and fixed:** the existing broken stub's column-detection logic listed
candidate names for "Phrase A" as `["Assumption", "PhraseA", "Phrase A", "A"]` — it never
included `Original A` / `Original Assumption` at all. Had it run as written, it would have
silently matched the bare `A` / `Assumption` column instead, which is the **body-form** phrase
(e.g. `easy_check-in`), not the **contrary-form** phrase (e.g. `no_evident_not_easy_check-in`)
that the task is actually supposed to test against. I corrected the column-detection priority so
it always resolves to the contrary-form column, and verified this against the raw CSV rows.

Also confirmed: only 1 of mu_work's 4 polarity categories (`Contrary(P)Body(N)`) has been
exported to CSV in this project — the original 4-sheet Silver `.xlsx` workbooks only ever existed
on the previous intern's local machine, not in the shared repo. **Decision (discussed with the
plan before building):** build and run Task 3 against this 1 available category for now; the
loader is written so the other 3 categories can be added later without a redesign, if those sheets
become available.

### 3.2 Ported Task 3 into this project's architecture, not just copied mu_work's script

Rather than reusing mu_work's script as-is (which used raw `requests.post` calls, hardcoded
absolute file paths, and fixed Python-string prompts), I rebuilt Task 3 to match the same
architecture already established for Task 1 and Task 2 in this project:

- **`configs/paths.yaml`** — added a `task3_dir` entry pointing at `Dataset/Task3`, loaded through
  the existing `PathsConfig` dataclass in `src/config.py` (no hardcoded paths in pipeline code,
  consistent with how Task 1/2 paths are handled).
- **`prompts/task3/zero_shot.txt`** and **`prompts/task3/one_shot.txt`** — mu_work's prompt
  wording ported faithfully (not redesigned yet, per your instruction to get a working baseline
  first) into standalone template files using `{{PHRASE_A}}` / `{{PHRASE_B}}` placeholders,
  matching how Task 1/2 prompts are structured as separate files rather than embedded in Python.
- **`src/task3.py`** (new) — the pipeline module, mirroring `src/task2.py`'s shape:
  - `load_task3_instances()` — reads all 4 gold CSVs, applies the corrected column detection,
    drops incomplete/duplicate rows.
  - `run_task3()` — renders the prompt, calls the LLM through the project's existing shared
    `LLMClient` abstraction (`src/llm.py`, the same Ollama client Task 1/2 use — not a separate
    HTTP call path), runs instances in parallel with `ThreadPoolExecutor` (same as Task 2),
    resumes from existing output on interruption/re-run by checking already-processed
    `(ID, PhraseA, PhraseB)` keys (same checkpoint pattern as Task 1/2), and writes both a
    `.jsonl` (full raw output + parsed prediction) and a flattened `.csv` per aspect.
- **`run_task3.py`** — completely rewritten as an `argparse` CLI in the same style as
  `run_task2.py` (`--model`, `--prompt <path>`, `--n-runs`, `--n`/`--aspects` for quick tests).
  `--prompt` accepts any template path, and the output folder/filenames are derived from that
  file's name — the same pattern Task 2 uses for its `generator_v1.txt` … `v6.txt` sweep.
- **`evaluator/task3_eval.py`** (new) — mirrors the existing `evaluator/subtask1.1.py` /
  `task2_eval.py` pattern: auto-discovers every Task 3 output CSV, computes
  TP/FP/FN/TN → Precision/Recall/F1/Accuracy (positive class = "Yes"), writes a `results.txt`
  summary and a verbose `workings.txt` row-by-row trace, mirroring the existing evaluator output
  layout under `outputs/eval/task3/`.

### 3.3 Verified it end-to-end before calling it done

1. Loaded all 4 gold CSVs (6,971 phrase pairs total) and confirmed Phrase A resolves to the
   correct contrary-form column across all of them (spot-checked against raw rows).
2. Ran small live batches through Ollama (llama3.2, both 0-shot and 1-shot, ~10 pairs per aspect,
   2 repeated runs) — confirmed prompts render correctly, the model's Yes/No output parses
   correctly, output files write in the expected format, and resume-on-rerun correctly skips
   already-processed pairs instead of re-calling the LLM.
3. Ran the evaluator against those outputs and got non-degenerate results (example from the
   verification batch — **not a full run, just a pipeline sanity check**):

   | Aspect | Accuracy | Precision | Recall | F1 |
   |---|---|---|---|---|
   | check-in (n=10) | 0.30 | 0.20 | 0.25 | 0.22 |
   | price (n=10) | 0.90 | 1.00 | 0.89 | 0.94 |

   These numbers are from a 10-row smoke test only and should not be read as real results —
   they're included here just to show the scoring pipeline produces sensible, non-trivial
   metrics rather than crashing or returning garbage.

## 4. Prompt-engineering sweep (6 new variants, beyond the mu_work baseline)

Task 1 and 2 in this project use real prompt engineering (multi-rule modular prompts for Task 1;
6 iterated prompt versions, `generator_v1.txt` … `v6.txt`, for Task 2) — much more than mu_work's
Task 3, which only ever tried one fixed 0-shot and one fixed 1-shot prompt. To bring Task 3 in
line with that, I designed 6 additional prompt strategies (`prompts/task3/contrary_v1.txt` …
`v6.txt`), each a genuinely different prompting technique, all grounded in real example pairs
pulled from the gold CSVs (not invented ones) and all using contrastive Yes/No example pairs —
matching how you've used contrastive examples in Task 1 and 2:

| Version | Strategy | Idea |
|---|---|---|
| contrary_v1 | Full rules + 5-shot + Contrastive | Explicit definition of what "contrary" means for an auto-derived assumption, a 5-rule decision checklist, 5 worked contrastive examples across 3 aspects |
| contrary_v2 | Compact steps + 4-shot + pattern-first | 3-line checklist, output pattern shown before the examples to force the `Answer: Yes/No` shape, 4 compact examples |
| contrary_v3 | Chain-of-thought + contrastive | Model writes one short reasoning sentence, then a final `Answer:` line; 2 worked examples (1 Yes, 1 No) show the expected reasoning style |
| contrary_v4 | Fill-in-the-blank (FIM) + contrastive | Presents the task as a sentence with a blank to complete with Yes/No; 4 contrastive filled-in examples |
| contrary_v5 | 7-shot + instruction/advice | 7 compact examples spanning all 4 aspects, plus an explicit "Advice" section calling out common failure modes (keyword-matching, same-topic-but-different-issue, defaulting to No when unsure) |
| contrary_v6 | Transformation/decision recipe + contrastive | A 4-step recipe (Strip → Polarity → Match → Decide) applied step-by-step, mirroring how Task 2's `generator_v6.txt` explains its rules as an explicit transform recipe; 2 fully worked contrastive examples |

**Implementation change needed to support this:** the pipeline originally hardcoded `--shot
{zero,one}` mapped straight to `zero_shot.txt`/`one_shot.txt`. I generalized it to a `--prompt
<path>` argument (same as Task 2's `--prompt`), so any template file works and the output
folder/filenames are derived from that file's name automatically.

I also had to make the Yes/No answer parser more robust: the original regex took the *first*
"yes"/"no" word found anywhere in the model's raw output, which works fine for the baseline
(the whole output *is* just "Yes" or "No") but breaks for the CoT/recipe versions, where the
reasoning text itself often contains those words before the actual verdict. The parser now
prefers an explicit `Answer: Yes/No` tag if present, and otherwise falls back to the *last*
yes/no token in the text (the conclusion normally comes after the reasoning, not before it).

**Verified:** ran all 6 new versions live against a small sample (llama3.2, 4 check-in pairs
each) — every row parsed to a valid Yes/No prediction, including the longer CoT/recipe outputs,
confirming the new parser correctly extracts the final verdict rather than an early mention of
"yes"/"no" inside the reasoning. This was a pipeline-correctness check only, not a real
accuracy comparison — that requires a full run.

## 5. Current status / what's next

- Task 3 pipeline is complete and runs end-to-end for both models used in Task 1/2
  (llama3.2, llama4:scout) across 8 prompt variants total: the mu_work baseline (0-shot, 1-shot)
  plus the 6 new engineered versions above.
- **Full sweep completed for llama3.2** across all 8 prompt variants on the 4 confirmed-gold
  topics (3 repeated runs each) — see §8 for results. `llama4:scout` still needs the same sweep
  once that model is available on this machine.
- **Scope limitation (unchanged):** only the `Contrary(P)Body(N)` polarity category is available
  as gold data in this project. mu_work's original evaluation covered 4 categories per aspect. If
  the advisor can provide the other 3 Silver sheets, the loader/evaluator can be extended to cover
  them (the evaluator already has a comment flagging where a per-category positive-label mapping
  would need to go, since some categories expect "No" as the correct answer rather than "Yes").

## 6. Topic coverage gap: only 4 of 11 topics have gold data — and it's not just this project

I checked whether mu_work themselves ever ran Task 3 on more than these 4 topics
(check-in, check-out, price, staff). **They didn't.** Every one of their Task 3 scripts —
the local Ollama runner, the Gemini runner, the GPT-4o runner, and their evaluation script —
hardcodes the exact same 4 input files, and their evaluator's aspect-detection regex only
recognizes `check-in|check-out|price|staff`. So this is mu_work's own original scope, not
something narrowed during this port.

**Why it can't just be "extended" the same way the prompts were:** Task 3's gold labels are
human votes — for each candidate (Phrase A, Phrase B) pair, an annotator manually decided
Yes/No (see the `is phrase B contain any Contrary to phrase A? by <name>` columns in the Silver
sheets). That voting work was never done for Room, Location, Food, Facility, Booking-issue, or
Taxi-issue — not in this project, not in mu_work's repo. It can't be fabricated without
corrupting the evaluation.

**What I built instead: `build_task3_candidates.py`** — a script that generates *unlabeled*
candidate pairs for the 6 missing topics (Off is excluded, same as Task 1/2), ready for a human
to vote on, using the exact same pairing definition the real Silver sheets follow
(`Contrary(P)Body(N)`: Phrase A = a contrary literal auto-derived from a positive-sentiment
Task 2 body literal, Phrase B = a negative-sentiment Task 2 body literal, same topic only). It
reuses the Task 2 GT output you already generated (`outputs/task2/gt/llama3.2/version1/...
_gt_n2078.csv`, which covers all 10 non-Off topics) as its literal source, so no new LLM calls
were needed to build it.

Since a full cross-product of every distinct A × every distinct B would be far too large to
hand-annotate for the bigger topics (Room alone would be 92,106 pairs), the script randomly
samples down to a configurable cap (default 300 pairs/topic, fixed seed for reproducibility) —
smaller topics stay complete. Result from the first run:

| Topic | A pool (positive contrary literals) | B pool (negative body literals) | Full A×B | Written |
|---|---|---|---|---|
| Booking-issue | 0 | 8 | 0 | **0 — see note below** |
| Facility | 221 | 81 | 17,901 | 300 (sampled) |
| Food | 220 | 48 | 10,560 | 300 (sampled) |
| Location | 271 | 135 | 36,585 | 300 (sampled) |
| Room | 306 | 301 | 92,106 | 300 (sampled) |
| Taxi-issue | 2 | 19 | 38 | 38 (all of them) |

Output files: `Dataset/Task3/candidates/{topic}_candidates.csv`, with columns
`ID, Topic, Original A, B, Vote` (matching the real Silver sheets' column convention, with
`Vote` left blank for annotators to fill in).

**Booking-issue has zero candidates** because Task 2's GT run found no positive-sentiment
Booking-issue reviews to derive a contrary/assumption phrase from — worth checking with your
advisor whether that's expected (customers mostly only complain about booking issues, rarely
praise them) or a Task 2 coverage gap.

**Important caveat to flag:** these candidate phrases come from a Task 2 *LLM* run (the "gt"
source means the LLM was given the correct topic/sentiment/span, not that a human wrote the
literal). "Valid" only means the JSON passed schema validation — a few candidates look like
weak/low-signal literals (e.g. plain `airport`, `included_in_bill` for Taxi-issue) that a human
annotator would likely want to skip or mark as unclear. That's expected — this is meant to be
reviewed and voted on, not treated as gold as-is.

**This is prep work, not evaluation-ready data.** Nothing changes for Task 3's actual results
until someone votes on these — I have not, and should not, invent those votes myself.

## 7. Output format matched to mu_work's structure

Originally I gave Task 3's outputs a simpler, flatter layout (one CSV per aspect with a
`GoldVote`/`Prediction` column baked in) rather than reusing mu_work's own file structure. Since
the goal is a direct, side-by-side comparison against mu_work's results, I rebuilt the generator
and evaluator to reproduce their exact structure and file formats instead:

**Generation output** (`run_task3.py`, per aspect and per category — currently just the one
`Contrary(P)Body(N)` category):
```
outputs/task3/{model}/{prompt_label}/{aspect}/Contrary_P_Body_N/
    csv/task3_{label}_{aspect}_Contrary_P_Body_N_run{N}.csv    # ID, Prompt, PhraseA, PhraseB, Test {N}
    log/task3_{label}_{aspect}_Contrary_P_Body_N_run{N}.csv    # Timestamp, Run, ID, PhraseA, PhraseB, Prompt, RawOutput
    task3_{label}_{aspect}_Contrary_P_Body_N_sheet.xlsx        # wide: ID, Prompt, PhraseA, PhraseB, Test 1..Test N
outputs/task3/{model}/{prompt_label}/{aspect}/
    task3_{label}_{aspect}_ALL_sheets.xlsx                     # one worksheet per category
```
This is the same `csv/` + `log/` + wide `_sheet.xlsx` + `_ALL_sheets.xlsx` structure as mu_work's
`Task3.py`, with `{prompt_label}` (e.g. `zero_shot`, `contrary_v3`) standing in for mu_work's
`0shot`/`1shot` folder — generalized since this project has 8 prompt variants instead of 2.

**Evaluation output** (`evaluator/task3_eval.py`) now mirrors mu_work's `Task3_Evaluation.py`:
- `outputs/eval/task3/{aspect}/{model}/{prompt_label}/Contrary(P)Body(N).xlsx` — one sheet per
  `Test N` column (`ID, Vote, LLM_Output, Verify`) plus a `SUMMARY` sheet
  (`Precision/Recall/F1/Accuracy` per test run)
- `.../EVAL_SUMMARY.xlsx` — `ALL` + `MICRO_BY_CATEGORY` sheets (aggregating across test runs)
- `outputs/eval/task3/TASK3_VERIFY_MASTER_SUMMARY.xlsx` — top-level rollup across every
  (model, prompt version, aspect) combination found, with an averaged-metrics sheet and each
  config's best-scoring test run picked out — same as mu_work's master summary

One deliberate difference: gold `Vote` is read through `load_task3_instances()` (this project's
existing column-name-aware CSV loader) rather than mu_work's hardcoded per-aspect `.xlsx` path
dict, since our gold data is CSV, not the original Silver `.xlsx` files.

Verified end-to-end again after the rewrite: ran zero-shot and the chain-of-thought variant
(`contrary_v3`, to make sure multi-line reasoning text didn't break CSV quoting) on a small
sample, confirmed resume-on-rerun still skips already-processed pairs instantly, and confirmed
the evaluator's TP/FP/FN counts and Precision/Recall/F1 match what the raw predictions imply.

## 8. Full sweep results (llama3.2, all 8 prompt versions, 4 confirmed-gold topics)

Ran all 8 prompt versions (`zero_shot`, `one_shot`, `contrary_v1`–`v6`) against check-in,
check-out, price, and staff, 3 repeated runs each (3,645 pairs evaluated per version — every
single prediction parsed to a valid Yes/No, 0 failures). Metrics below are Precision/Recall/F1/
Accuracy averaged across the 3 runs and 4 topics for each prompt version — full breakdown by
topic is in `outputs/eval/task3/TASK3_VERIFY_MASTER_SUMMARY.xlsx`.

| Prompt version | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| **contrary_v5** (7-shot + advice) | 0.592 | 0.891 | **0.693** | 0.635 |
| contrary_v2 (compact + pattern-first) | 0.554 | 0.950 | 0.677 | 0.571 |
| contrary_v1 (full rules + 5-shot) | 0.584 | 0.826 | 0.676 | **0.650** |
| contrary_v6 (decision recipe) | 0.628 | 0.738 | 0.665 | 0.649 |
| zero_shot (mu_work baseline) | 0.572 | 0.828 | 0.663 | 0.617 |
| contrary_v3 (chain-of-thought) | 0.553 | 0.550 | 0.537 | 0.554 |
| one_shot (mu_work baseline) | 0.610 | 0.147 | 0.236 | 0.535 |
| contrary_v4 (fill-in-the-blank) | 0.000 | 0.000 | 0.000 | 0.454 |

**Headline finding: `contrary_v5` (7-shot examples + an explicit "advice" section on common
mistakes) beats the mu_work `zero_shot` baseline on F1 (0.693 vs 0.663).** The extra worked
examples plus explicitly warning against keyword-matching and same-topic-but-different-issue
pairs appears to help more than either full rule explanations (`v1`) or asking the model to
reason step-by-step (`v3`).

**Two notable failure modes, both confirmed as genuine model behavior (not bugs — checked the
raw output logs and 100% of predictions parsed successfully):**
- **`contrary_v4` (fill-in-the-blank) scores exactly 0.000 F1 on every topic.** llama3.2
  answers `"No."` to literally every single pair regardless of content — it doesn't seem to
  engage with the sentence-completion framing at all, just pattern-matches to the "safer"
  negative answer.
- **`one_shot` has very low recall (0.147)** — it also defaults to "No" far more often than it
  should, worse than `zero_shot`'s no-example baseline. The single worked example doesn't seem
  to be enough to correct this small model's bias toward "No", and may even be reinforcing it.

**Caveat:** this is one model (llama3.2) on one polarity category (`Contrary(P)Body(N)`) across
4 topics. Whether `contrary_v5` stays the best version once `llama4:scout` is run, or once the
other 3 polarity categories become available, is still open — worth treating this as an early
signal, not a final conclusion, when presenting it.

### Row-order bug (found and fixed after the sweep)

The `csv/task3_..._run{N}.csv` files (e.g. check-in's) weren't sorted by ID — they came out
scrambled (e.g. `8, 5, 2, 3, 1, 4, 7, ...`) because predictions run in parallel via
`ThreadPoolExecutor`, and rows were written in whichever order the LLM finished responding, not
submission order. The `_sheet.xlsx` wide tables (and `_ALL_sheets.xlsx`) were **not** affected —
they're built by merging against the original gold-file order, so they were always correctly
sorted. Fixed in `src/task3.py` (rows are now reordered to match the gold file before writing),
and re-sorted all 97 already-generated `csv/*.csv` files in place with a one-off script
(`fix_task3_csv_order.py`) — no LLM calls needed, since the data itself was correct, just out of
order. Re-ran the evaluator afterward to confirm scores were unaffected (matching is by ID, not
row position) — the table above reflects the post-fix numbers. `log/*.csv` files are left as
chronological timestamped logs on purpose (matching mu_work's own log design), not ID-sorted.

## 9. Data fidelity verification — cross-checked against mu_team's own files

Before presenting these results, I directly verified that our gold data (`Dataset/Task3/*.csv`)
is the same data mu_team used — not just "the same source in principle." I cross-checked every
row's Phrase A / Phrase B against mu_team's own raw prediction CSVs (which carry the exact
PhraseA/PhraseB/ID they pulled from their Silver spreadsheets when they ran their own models):

| Topic | Rows | Phrase A match | Phrase B match |
|---|---|---|---|
| check-in | 252 | 100% | 100% |
| price | 455 | 100% | 100% |
| staff | 6,256 | 100% | 100% |
| check-out | 8 | 100% | 0% (see note) |

check-in, price, and staff — 99.9% of the data by row count — are byte-for-byte identical to
what mu_team used.

**check-out note:** our check-out data (constant Phrase B tested against 8 different
assumptions) matches the actual Silver spreadsheet and matches Task 3's intended design (a
candidate proposition checked against several different assumptions). mu_team's own *run output*
for check-out instead shows Phrase B equal to Phrase A's own body form on every row — consistent
with a column-mapping bug in their script for that one file (`Task3.py` hardcodes
`PHRASE_A_COL = "Assumption"` with no fallback, while Phrase B has a fallback chain; check-out is
also their smallest file, at 8 rows, easiest to have gone unnoticed). This means our check-out
data matches the original Silver source; the discrepancy appears to be in mu_team's own
processing of that file, not in ours.

**Bonus finding, not yet acted on:** mu_team's evaluation folder (`Task3, Evaluation Score/`)
contains the actual human `Vote` labels for the 3 polarity categories we don't have yet
(`Contrary(N)Body(P)`, `Contrary(P)Body(P)`, `Contrary(N)Body(N)`), for all 4 topics — joinable
with the phrase pairs in their raw prediction CSVs to reconstruct real (not fabricated) gold data
for them. Not pursued yet — it's a large amount of data (~49,000 pairs across the 3 categories)
and running our full prompt sweep against it is a multi-hour job, deferred as a next step rather
than rushed before this meeting.

## 10. Files added / changed

| File | Change |
|---|---|
| `configs/paths.yaml` | added `task3_dir` |
| `src/config.py` | added `task3_dir` field to `PathsConfig` |
| `prompts/task3/zero_shot.txt` | new — mu_work baseline (0-shot), ported |
| `prompts/task3/one_shot.txt` | new — mu_work baseline (1-shot), ported |
| `prompts/task3/contrary_v1.txt` … `contrary_v6.txt` | new — 6 engineered prompt variants (see §4) |
| `src/task3.py` | pipeline module; output structure now matches mu_work's csv/log/xlsx layout (see §7) |
| `run_task3.py` | rewritten (previous version was a non-functional stub); `--prompt <path>` replaces hardcoded `--shot` |
| `src/__init__.py` | export `load_task3_instances`, `run_task3` |
| `evaluator/task3_eval.py` | rewritten to match mu_work's `Task3_Evaluation.py` structure (see §7) |
| `build_task3_candidates.py` | new — generates unlabeled candidate pairs for the 6 topics without gold data (see §6) |
| `run_task3_all.py` | new — runs `run_task3.py` once per prompt version, so a full sweep across all versions is one command (see §8) |
| `src/task3.py` | fixed: run csv rows now reordered to match gold-file ID order before writing (see §8) |
| `fix_task3_csv_order.py` | new, one-off — re-sorted all 97 already-generated run csv files by ID (see §8) |
