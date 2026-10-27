"""
Estimates token usage, time, and USD cost for running Task 1, Task 2, and Task 3 with
Gemini 2.5 Pro, using REAL prompts (rendered via the project's own prompt-building code)
and REAL generated outputs already on disk from the llama3.2 runs, not guesses.

Gemini 2.5 Pro pricing (standard tier, <=200k context, confirmed against
https://ai.google.dev/gemini-api/docs/pricing on 2026-08-21):
  Input:  $1.25 / 1M tokens
  Output: $10.00 / 1M tokens (includes "thinking" tokens)

Token counts are approximated with tiktoken's cl100k_base encoding (a standard proxy widely
used for cross-model estimation; Gemini's own tokenizer will differ somewhat, expect this
estimate to be within +/-15-20% of the real Gemini token count).
"""
import json
import random
import sys
from pathlib import Path

import pandas as pd
import tiktoken

REPO_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = REPO_ROOT.parent.parent
for p in (REPO_ROOT, WORKSPACE_ROOT):
    sys.path.insert(0, str(p))

from internship.ABA_mining.src.config import load_paths_config, load_topics_config
from internship.ABA_mining.src.prompts import build_modular_prompt, load_prompt, render_prompt
from internship.ABA_mining.src.task1 import load_task1_instances_from_input
from internship.ABA_mining.src.task2 import load_task2_instances_gt

enc = tiktoken.get_encoding("cl100k_base")


def ntok(text: str) -> int:
    return len(enc.encode(text or "", disallowed_special=()))


IN_PRICE = 1.25 / 1_000_000   # $ per input token
OUT_PRICE = 10.00 / 1_000_000  # $ per output token

results = {}

# =====================================================================================
# TASK 1 — "combined" experiment (all 7 rules, full output schema), matches the actual
# completed llama3.2 scope: 784 reviews x runs_per_id=3 x 3 outer runs.
# =====================================================================================
print("=== TASK 1 ===")
paths_cfg = load_paths_config(REPO_ROOT)
topics_cfg = load_topics_config(REPO_ROOT)
instances = load_task1_instances_from_input(paths_cfg)
n_reviews = len(instances)
print(f"reviews: {n_reviews}")

# Sample real reviews, build the REAL combined prompt (rules 1-7, full schema) via the
# project's own build_modular_prompt(), same as run_task1.py does.
random.seed(42)
sample = random.sample(instances, min(60, len(instances)))
prompt_toks = []
for inst in sample:
    header = build_modular_prompt(REPO_ROOT, rules=[1, 2, 3, 4, 5, 6, 7], output_schema="full")
    full_prompt = header.replace("{{REVIEW_TEXT}}", inst.review_text) if "{{REVIEW_TEXT}}" in header else header + "\n\nReview:\n" + inst.review_text
    prompt_toks.append(ntok(full_prompt))
avg_prompt_tok_t1 = sum(prompt_toks) / len(prompt_toks)
print(f"avg input tokens/call (sampled {len(sample)} reviews): {avg_prompt_tok_t1:.0f}")

# Real output sizes from actual llama3.2 raw_output (same JSON schema Gemini would produce)
out_toks = []
jf = REPO_ROOT / "outputs/task1/llama3.2/modular/combined/task1_llama3.2_extended11_combined_run1.jsonl"
with jf.open(encoding="utf-8") as f:
    recs = [json.loads(l) for l in f]
sample_out = random.sample(recs, min(200, len(recs)))
for r in sample_out:
    out_toks.append(ntok(r.get("raw_output") or ""))
avg_out_tok_t1 = sum(out_toks) / len(out_toks)
avg_retries_t1 = sum(r.get("retries", 0) for r in recs) / len(recs)
print(f"avg output tokens/call (sampled {len(sample_out)} real outputs): {avg_out_tok_t1:.0f}")
print(f"avg retries/call (cascading validator, up to max_retries=2): {avg_retries_t1:.3f}")

# Validator retry prompt is a DIFFERENT, smaller template (review + candidate JSON + errors,
# no 7 rule blocks) — measured directly from real retried records, not assumed equal to the
# main generation prompt.
val_template = load_prompt(REPO_ROOT, "prompts/task1/Contrastive/validator_v1.txt")
topics_str = ", ".join(topics_cfg.topics)
inst_by_id = {i.review_id: i for i in instances}
retried_recs = [r for r in recs if r.get("retries", 0) > 0]
val_sample = random.sample(retried_recs, min(60, len(retried_recs)))
val_prompt_toks = []
for r in val_sample:
    inst = inst_by_id.get(r["review_id"])
    if not inst:
        continue
    vp = render_prompt(val_template, TOPICS=topics_str, REVIEW_TEXT=inst.review_text,
                        CANDIDATE_JSON=r["raw_output"], ERRORS="; ".join(r.get("errors", [])))
    val_prompt_toks.append(ntok(vp))
avg_val_prompt_tok_t1 = sum(val_prompt_toks) / len(val_prompt_toks)
print(f"avg validator retry prompt tokens (measured, {len(val_prompt_toks)} real samples): {avg_val_prompt_tok_t1:.0f}")

runs_per_id = 3   # src/task1.py default
outer_runs = 3    # matches the 3 completed run1/run2/run3.jsonl files
base_calls_t1 = n_reviews * runs_per_id * outer_runs
extra_validator_calls_t1 = base_calls_t1 * avg_retries_t1  # cascading validator, avg 0.82 extra calls/base call
total_calls_t1 = base_calls_t1 + extra_validator_calls_t1

in_tok_t1 = base_calls_t1 * avg_prompt_tok_t1 + extra_validator_calls_t1 * avg_val_prompt_tok_t1
out_tok_t1 = total_calls_t1 * avg_out_tok_t1  # validator output is the same corrected-JSON schema
cost_t1 = in_tok_t1 * IN_PRICE + out_tok_t1 * OUT_PRICE

results["task1"] = dict(
    scope="combined experiment (7 rules, full schema), 784 reviews x runs_per_id=3 x 3 outer runs",
    base_calls=base_calls_t1, extra_validator_calls=extra_validator_calls_t1,
    total_calls=total_calls_t1,
    avg_input_tok=avg_prompt_tok_t1, avg_validator_input_tok=avg_val_prompt_tok_t1,
    avg_output_tok=avg_out_tok_t1,
    total_input_tok=in_tok_t1, total_output_tok=out_tok_t1, cost_usd=cost_t1,
)
print(f"TASK1 total calls: {total_calls_t1:.0f}  cost: ${cost_t1:.2f}\n")

# =====================================================================================
# TASK 2 — 6 prompt versions (generator_v1..v6), matches completed llama3.2 scope:
# 2078 GT instances x 6 versions x 3 runs.
# =====================================================================================
print("=== TASK 2 ===")
t2_instances = load_task2_instances_gt(paths_cfg.gold_csv)
n_t2 = len(t2_instances)
print(f"GT instances: {n_t2}")

sample_t2 = random.sample(t2_instances, min(60, len(t2_instances)))
version_stats = {}
for v in range(1, 7):
    prompt_path = f"prompts/task2/generator_v{v}.txt"
    template = load_prompt(REPO_ROOT, prompt_path)
    ptoks = []
    for inst in sample_t2:
        p = render_prompt(template, TOPIC=inst.topic, SENTIMENT=inst.sentiment, SELECTED_CONTENT=inst.selected_content)
        ptoks.append(ntok(p))
    avg_in = sum(ptoks) / len(ptoks)

    # real output sizes from that version's actual run1 jsonl
    jf = REPO_ROOT / f"outputs/task2/gt/llama3.2/version{v}/task2_llama3.2_llama3.2_generator_v{v}_run1_gt_n2078.jsonl"
    otoks = []
    retries = 0
    if jf.exists():
        with jf.open(encoding="utf-8") as f:
            recs = [json.loads(l) for l in f]
        sample_out = random.sample(recs, min(200, len(recs)))
        otoks = [ntok(r.get("raw_output") or "") for r in sample_out]
        retries = sum(1 for r in recs if r.get("retries", 0) > 0) / len(recs)
    avg_out = sum(otoks) / len(otoks) if otoks else 0
    version_stats[v] = dict(avg_in=avg_in, avg_out=avg_out, retry_rate=retries)
    print(f"  v{v}: avg_in={avg_in:.0f} avg_out={avg_out:.0f} retry_rate={retries:.1%}")

runs_t2 = 3
total_calls_t2 = 0
in_tok_t2 = 0
out_tok_t2 = 0
for v, st in version_stats.items():
    base = n_t2 * runs_t2
    extra = base * st["retry_rate"]
    calls = base + extra
    total_calls_t2 += calls
    in_tok_t2 += calls * st["avg_in"]
    out_tok_t2 += calls * st["avg_out"]

cost_t2 = in_tok_t2 * IN_PRICE + out_tok_t2 * OUT_PRICE
results["task2"] = dict(
    scope="6 prompt versions (generator_v1..v6), 2078 GT instances x 3 runs each",
    total_calls=total_calls_t2, total_input_tok=in_tok_t2, total_output_tok=out_tok_t2,
    cost_usd=cost_t2, version_stats=version_stats,
)
print(f"TASK2 total calls: {total_calls_t2:.0f}  cost: ${cost_t2:.2f}\n")

# =====================================================================================
# TASK 3 — 8 prompt versions, measured DIRECTLY from real log/ CSVs (Prompt + RawOutput
# columns are both stored in full, so this is the most precise measurement of the three).
# Scope: 4 confirmed-gold topics (check-in/check-out/price/staff) x 8 versions x 3 runs
# = 1215 pairs/version/run (already the real completed llama3.2 scope).
# =====================================================================================
print("=== TASK 3 ===")
versions_t3 = ["zero_shot", "one_shot", "contrary_v1", "contrary_v2", "contrary_v3",
               "contrary_v4", "contrary_v5", "contrary_v6"]
topics_t3 = ["check-in", "check-out", "price", "staff"]

total_calls_t3 = 0
in_tok_t3 = 0
out_tok_t3 = 0
t3_version_stats = {}
for v in versions_t3:
    ins, outs, n = [], [], 0
    for topic in topics_t3:
        log_dir = REPO_ROOT / f"outputs/task3/llama3.2/{v}/{topic}/Contrary_P_Body_N/log"
        for run in (1, 2, 3):
            f = log_dir / f"task3_{v}_{topic}_Contrary_P_Body_N_run{run}.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f, dtype=str).fillna("")
            sub = df.sample(min(40, len(df)), random_state=42) if len(df) > 0 else df
            for _, row in sub.iterrows():
                ins.append(ntok(row["Prompt"]))
                outs.append(ntok(row["RawOutput"]))
            n += len(df)
    if not ins:
        continue
    avg_in = sum(ins) / len(ins)
    avg_out = sum(outs) / len(outs)
    t3_version_stats[v] = dict(avg_in=avg_in, avg_out=avg_out, n_calls=n)
    total_calls_t3 += n
    in_tok_t3 += n * avg_in
    out_tok_t3 += n * avg_out
    print(f"  {v}: avg_in={avg_in:.0f} avg_out={avg_out:.0f} calls={n}")

cost_t3 = in_tok_t3 * IN_PRICE + out_tok_t3 * OUT_PRICE
results["task3"] = dict(
    scope="8 prompt versions, 4 topics (check-in/check-out/price/staff) x 3 runs each",
    total_calls=total_calls_t3, total_input_tok=in_tok_t3, total_output_tok=out_tok_t3,
    cost_usd=cost_t3, version_stats=t3_version_stats,
)
print(f"TASK3 total calls: {total_calls_t3:.0f}  cost: ${cost_t3:.2f}\n")

# =====================================================================================
grand_calls = results["task1"]["total_calls"] + results["task2"]["total_calls"] + results["task3"]["total_calls"]
grand_in = results["task1"]["total_input_tok"] + results["task2"]["total_input_tok"] + results["task3"]["total_input_tok"]
grand_out = results["task1"]["total_output_tok"] + results["task2"]["total_output_tok"] + results["task3"]["total_output_tok"]
grand_cost = results["task1"]["cost_usd"] + results["task2"]["cost_usd"] + results["task3"]["cost_usd"]

print("=== GRAND TOTAL ===")
print(f"calls: {grand_calls:.0f}")
print(f"input tokens: {grand_in:,.0f}")
print(f"output tokens: {grand_out:,.0f}")
print(f"cost: ${grand_cost:.2f}")

results["grand_total"] = dict(total_calls=grand_calls, total_input_tok=grand_in,
                               total_output_tok=grand_out, cost_usd=grand_cost)

out_json = REPO_ROOT / "Doc" / "gemini_cost_results.json"
with out_json.open("w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved raw results to {out_json}")
