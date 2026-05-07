import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Plus

# Paths
TASK_DIR = Path(__file__).resolve().parent.parent
GT_CSV   = TASK_DIR / "data" / (
    "Original ABA Dataset for Version 2 "
    "[Oct 23, 2025], Senior Project, MUICT - Sheet2_.csv"
)
LLM_DIR  = TASK_DIR / "outputs" / "task1"
EVAL_DIR = TASK_DIR / "outputs" / "eval" / "sentiment"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


# Rule → subtask mapping
RULE_SUBTASKS: dict = {
    1: {"1.2"},
    2: {"1.1"},
    3: {"1.3"},
    4: {"1.3"},
    5: {"1.1"},
    6: {"1.1", "1.2"},
    7: {"1.2"},
}


def applicable_subtasks(llm_csv: Path, llm_dir: Path) -> set:
    parts = llm_csv.relative_to(llm_dir).parts
    try:
        folder = parts[list(parts).index("modular") + 1]
    except (ValueError, IndexError):
        return {"1.1", "1.2", "1.3"}
    if folder == "combined":
        return {"1.1", "1.2", "1.3"}
    if folder.startswith("subtask"):
        s = set()
        if "1_1" in folder: s.add("1.1")
        if "1_2" in folder: s.add("1.2")
        if "1_3" in folder: s.add("1.3")
        return s or {"1.1", "1.2", "1.3"}
    if "rule" in folder:
        s = set()
        for n in re.findall(r"\d+", folder):
            s |= RULE_SUBTASKS.get(int(n), set())
        return s or {"1.1", "1.2", "1.3"}
    return {"1.1", "1.2", "1.3"}


def normalize(text: str) -> str:
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def tokenize(text: str) -> list:
    return re.findall(r"\w+", text.lower())


def best_bm25_idx(query: str, candidates: list) -> int:
    try:
        bm25   = BM25Plus([tokenize(c) for c in candidates])
        scores = bm25.get_scores(tokenize(query))
        return int(np.argmax(scores))
    except Exception:
        return 0


def majority_sentiment(sentiments: list) -> str:
    """Return the most common sentiment; on tie return the first."""
    from collections import Counter
    counts = Counter(sentiments)
    return counts.most_common(1)[0][0]


# Load ground truth
# GT uses 'Selected Content' for text span and 'Pos/Neg' for sentiment
if not GT_CSV.exists():
    sys.exit(f"[ERROR] Ground truth not found:\n  {GT_CSV}")

gt_raw = pd.read_csv(GT_CSV)
gt_raw = gt_raw.rename(columns={"ID": "Review ID", "Pos/Neg": "Sentiment"})
gt_raw = gt_raw[
    gt_raw["Topic"].notna() &
    gt_raw["Selected Content"].notna() &
    gt_raw["Sentiment"].notna() &
    (gt_raw["Sentiment"].str.strip() != "")
].copy()
gt_raw["Topic"]     = gt_raw["Topic"].str.strip()
gt_raw["Sentiment"] = gt_raw["Sentiment"].str.strip()
gt_raw["norm"]      = gt_raw["Selected Content"].apply(normalize)
gt_raw = gt_raw[gt_raw["norm"] != ""].reset_index(drop=True)

# gt_index: (Review ID, Topic) → list of (norm_content, original_content, sentiment)
gt_index: dict = {}
for _, row in gt_raw.iterrows():
    key = (row["Review ID"], row["Topic"])
    gt_index.setdefault(key, []).append((
        row["norm"], row["Selected Content"], row["Sentiment"]
    ))

ALL_SENTIMENTS = sorted(gt_raw["Sentiment"].unique())

print(f"Ground truth: {len(gt_raw)} rows, {gt_raw['Review ID'].nunique()} unique Review IDs")
print(f"Sentiments: {ALL_SENTIMENTS}\n")


def evaluate(llm_csv: Path) -> None:
    if "1.3" not in applicable_subtasks(llm_csv, LLM_DIR):
        print(f"  [SKIP 1.3] not applicable by rules – {llm_csv.name}")
        return

    t_start = time.perf_counter()

    raw = pd.read_csv(llm_csv)
    if "Review ID" not in raw.columns and "ID" in raw.columns:
        raw = raw.rename(columns={"ID": "Review ID"})

    if "Errors" in raw.columns:
        raw = raw[raw["Errors"].isna() | (raw["Errors"].astype(str).str.strip().isin({"", "nan"}))].copy()

    if "Topic" not in raw.columns or "Sentiment" not in raw.columns:
        print(f"  [SKIP 1.3] missing 'Topic' or 'Sentiment' – {llm_csv.name}")
        return

    has_span = "Text Span" in raw.columns

    _bad = {"parse failed", "topics not found", "topic not found", "(parse failed)"}
    raw = raw[
        raw["Topic"].notna() &
        (raw["Topic"].str.strip() != "") &
        (~raw["Topic"].str.strip().str.lower().isin(_bad)) &
        raw["Sentiment"].notna() &
        (raw["Sentiment"].str.strip() != "")
    ].copy()
    raw["Topic"]     = raw["Topic"].str.strip()
    raw["Sentiment"] = raw["Sentiment"].str.strip()

    if has_span:
        raw["norm"] = raw["Text Span"].apply(normalize)
    else:
        raw["norm"] = ""

    raw = raw.reset_index(drop=True)

    if raw.empty:
        print(f"  [SKIP 1.3] no valid rows after filtering – {llm_csv.name}")
        return

    tp: dict = defaultdict(int)
    fp: dict = defaultdict(int)
    fn: dict = defaultdict(int)
    tn: dict = defaultdict(int)

    n_evaluated = n_skipped = n_correct = 0

    workings: list = []
    W = workings.append
    W("=" * 72)
    W(f"WORKINGS  –  {llm_csv.name}")
    W("=" * 72)
    W(f"\nHas 'Text Span' : {has_span}")
    W(f"Match strategy  : {'BM25 on text span' if has_span else 'majority GT sentiment for (ID, Topic)'}")
    W(f"Total LLM rows  : {len(raw)}")
    W(f"Sentiments      : {ALL_SENTIMENTS}\n")

    for _, row in raw.iterrows():
        rid      = row["Review ID"]
        topic    = row["Topic"]
        llm_sent = row["Sentiment"]
        lnorm    = row["norm"]

        gt_entries = gt_index.get((rid, topic), [])

        W(f"\n{'─' * 60}")
        W(f"ID={rid}  Topic={topic}")
        if has_span:
            W(f"  LLM Span      : {row['Text Span'][:80]}")
        W(f"  LLM Sentiment : {llm_sent}")
        W(f"  GT candidates : {len(gt_entries)}")

        if not gt_entries:
            n_skipped += 1
            W("  [No GT rows for this ID+Topic → skipped]")
            continue

        gt_norms = [e[0] for e in gt_entries]
        gt_conts = [e[1] for e in gt_entries]
        gt_sents = [e[2] for e in gt_entries]

        if has_span and lnorm:
            best_i  = best_bm25_idx(lnorm, gt_norms)
            gt_sent = gt_sents[best_i]
            gt_cont = gt_conts[best_i]
            W(f"  Best GT Span      : {gt_cont[:80]}")
        else:
            gt_sent = majority_sentiment(gt_sents)
            W(f"  Majority GT sent  : {gt_sent}  (from {gt_sents})")

        n_evaluated += 1
        if llm_sent == gt_sent:
            n_correct += 1

        W(f"  Best GT Sentiment : {gt_sent}  → "
          f"{'CORRECT' if llm_sent == gt_sent else 'WRONG'}")

        for sent in ALL_SENTIMENTS:
            llm_has = (llm_sent == sent)
            gt_has  = (gt_sent  == sent)
            if   llm_has and     gt_has: tp[sent] += 1; label = "TP"
            elif llm_has and not gt_has: fp[sent] += 1; label = "FP"
            elif not llm_has and gt_has: fn[sent] += 1; label = "FN"
            else:                        tn[sent] += 1; label = "TN"
            W(f"    {sent:<12}  LLM={'yes' if llm_has else 'no ':<4} "
              f"GT={'yes' if gt_has else 'no ':<4} → {label}")

    # Results
    results: list = []
    R = results.append
    R("=" * 72)
    R(f"EVALUATION RESULTS  –  {llm_csv.name}")
    R("=" * 72)
    R(f"\nTotal LLM rows  : {len(raw)}")
    R(f"Evaluated rows  : {n_evaluated}  (GT candidate found)")
    R(f"Skipped rows    : {n_skipped}  (no GT for ID+Topic)")
    R(f"Has Text Span   : {has_span}")
    R(f"Sentiments      : {ALL_SENTIMENTS}")

    ov_tp  = sum(tp.values()); ov_fp  = sum(fp.values())
    ov_fn  = sum(fn.values()); ov_tn  = sum(tn.values())
    ov_acc   = n_correct / n_evaluated            if n_evaluated      > 0 else 0
    ov_prec  = ov_tp / (ov_tp + ov_fp)           if (ov_tp + ov_fp)  > 0 else 0
    ov_rec   = ov_tp / (ov_tp + ov_fn)           if (ov_tp + ov_fn)  > 0 else 0
    ov_f1    = 2*ov_prec*ov_rec/(ov_prec+ov_rec) if (ov_prec+ov_rec) > 0 else 0

    R(f"\n{'─' * 72}")
    R("OVERALL SUMMARY")
    R(f"{'─' * 72}")
    R(f"  Correct / Evaluated : {n_correct} / {n_evaluated}")
    R(f"  TP={ov_tp}  FP={ov_fp}  FN={ov_fn}  TN={ov_tn}")
    R(f"  Accuracy  : {ov_acc:.4f}")
    R(f"  Precision : {ov_prec:.4f}")
    R(f"  Recall    : {ov_rec:.4f}")
    R(f"  F1        : {ov_f1:.4f}")

    R(f"\n{'─' * 72}")
    R("PER-SENTIMENT METRICS")
    R(f"{'─' * 72}")
    R(f"{'Sentiment':<12} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}"
      f"  {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}")
    R("─" * 72)

    micro_tp = micro_fp = micro_fn = 0
    for sent in ALL_SENTIMENTS:
        t = tp[sent]; f_p = fp[sent]; f_n = fn[sent]; t_n = tn[sent]
        total = t + f_p + f_n + t_n
        acc  = (t + t_n) / total     if total        > 0 else 0
        prec = t / (t + f_p)         if (t + f_p)    > 0 else 0
        rec  = t / (t + f_n)         if (t + f_n)    > 0 else 0
        f1   = 2*prec*rec/(prec+rec) if (prec + rec) > 0 else 0
        micro_tp += t; micro_fp += f_p; micro_fn += f_n
        R(f"{sent:<12} {acc:>8.4f} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f}"
          f"  {t:>5} {f_p:>5} {f_n:>5} {t_n:>5}")
        W(f"\n[Per-sentiment totals: {sent}]")
        W(f"  TP={t}  FP={f_p}  FN={f_n}  TN={t_n}")
        W(f"  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
    R("─" * 72)

    micro_prec = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0
    micro_rec  = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0
    micro_f1   = (2*micro_prec*micro_rec / (micro_prec+micro_rec)
                  if (micro_prec + micro_rec) > 0 else 0)

    R(f"\n{'─' * 72}")
    R("MICRO-AVERAGED METRICS  (aggregate TP/FP/FN across all sentiments)")
    R(f"{'─' * 72}")
    R(f"  Total TP={micro_tp}  FP={micro_fp}  FN={micro_fn}")
    R(f"  Micro Precision : {micro_prec:.4f}")
    R(f"  Micro Recall    : {micro_rec:.4f}")
    R(f"  Micro F1        : {micro_f1:.4f}")

    elapsed = time.perf_counter() - t_start
    R(f"\nRuntime: {elapsed:.2f}s")

    _write(llm_csv, workings, results)
    print("\n".join(results))


def _stem_parts(llm_csv: Path, base_dir: Path) -> tuple:
    rel    = llm_csv.relative_to(base_dir)
    subdir = Path(*rel.parts[:-1]) if len(rel.parts) > 1 else Path(".")
    stem   = llm_csv.stem
    stem   = re.sub(r"^task\d+_", "", stem)
    stem   = re.sub(r"_extended\d+", "", stem)
    stem   = re.sub(r"_generator", "", stem)
    stem   = re.sub(r"_n\d+$", "", stem)
    return subdir, stem


def _write(llm_csv: Path, workings: list, results: list) -> None:
    subdir, stem = _stem_parts(llm_csv, LLM_DIR)
    out_dir = EVAL_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    r_path = out_dir / f"{stem}_results.txt"
    w_path = out_dir / f"{stem}_workings.txt"
    r_path.write_text("\n".join(results),  encoding="utf-8")
    w_path.write_text("\n".join(workings), encoding="utf-8")
    print(f"  Saved: {r_path.relative_to(TASK_DIR)}")
    print(f"  Saved: {w_path.relative_to(TASK_DIR)}")


# Main
llm_csvs = sorted(p for p in LLM_DIR.rglob("*.csv") if p.stem.endswith("_n20"))
if not llm_csvs:
    sys.exit(f"[ERROR] No CSV files found under {LLM_DIR}")

print(f"Found {len(llm_csvs)} LLM output file(s):\n")
for p in llm_csvs:
    print(f"  {p.relative_to(TASK_DIR)}")
print()

total_start = time.perf_counter()

for llm_csv in llm_csvs:
    print(f"\n{'#' * 72}")
    print(f"Processing: {llm_csv.relative_to(TASK_DIR)}")
    print(f"{'#' * 72}")
    evaluate(llm_csv)

total_elapsed = time.perf_counter() - total_start
print(f"\n\nAll done. Results in: {EVAL_DIR.relative_to(TASK_DIR)}/")
print(f"Total runtime: {total_elapsed:.2f}s")
