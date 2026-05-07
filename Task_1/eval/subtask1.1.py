import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

# Paths
TASK_DIR   = Path(__file__).resolve().parent.parent
GT_CSV     = TASK_DIR / "data" / (
    "Original ABA Dataset for Version 2 "
    "[Oct 23, 2025], Senior Project, MUICT - Sheet2_.csv"
)
LLM_DIR    = TASK_DIR / "outputs" / "task1"
EVAL_DIR   = TASK_DIR / "outputs" / "eval" / "topic"
TOPICS_CFG = TASK_DIR / "configs" / "topics.yaml"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Rule → subtask mapping (from rules table)
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


# Global topic list
with open(TOPICS_CFG, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
active        = cfg.get("active_schema", "extended11")
GLOBAL_TOPICS = cfg["schemas"][active]["topics"]

# Load ground truth
if not GT_CSV.exists():
    sys.exit(f"[ERROR] Ground truth not found:\n  {GT_CSV}")

gt_raw = pd.read_csv(GT_CSV)
gt_raw = gt_raw.rename(columns={"ID": "Review ID"})
gt_raw = gt_raw[gt_raw["Topic"].notna() & (gt_raw["Topic"].str.strip() != "")].copy()
gt_raw["Topic"] = gt_raw["Topic"].str.strip()

gt_sets: dict = gt_raw.groupby("Review ID")["Topic"].apply(set).to_dict()

extra      = sorted({t for ts in gt_sets.values() for t in ts} - set(GLOBAL_TOPICS))
ALL_TOPICS = GLOBAL_TOPICS + extra

print(f"Ground truth: {len(gt_raw)} rows across {len(gt_sets)} Review IDs")
print(f"Global topics ({len(ALL_TOPICS)}): {ALL_TOPICS}\n")


_BAD_VALS = {"parse failed", "topics not found", "topic not found", "(parse failed)"}


def _load_topic_sets(raw: pd.DataFrame) -> dict | None:
    """
    Return {Review ID: set of topics} from either format:
      - 'Topics Present' column  (comma-separated list per row)
      - 'Topic' column           (one topic per row)
    Returns None if neither column exists.
    """
    if "Topics Present" in raw.columns:
        result: dict = {}
        valid = raw[raw["Topics Present"].notna()].copy()
        valid = valid[~valid["Topics Present"].str.strip().str.lower().isin(_BAD_VALS)]
        for _, row in valid.iterrows():
            topics = {t.strip() for t in str(row["Topics Present"]).split(",") if t.strip()}
            result.setdefault(row["Review ID"], set()).update(topics)
        return result

    if "Topic" in raw.columns:
        valid = raw[
            raw["Topic"].notna() &
            (raw["Topic"].str.strip() != "") &
            (~raw["Topic"].str.strip().str.lower().isin(_BAD_VALS))
        ].copy()
        valid["Topic"] = valid["Topic"].str.strip()
        return valid.groupby("Review ID")["Topic"].apply(set).to_dict()

    return None


def evaluate(llm_csv: Path) -> None:
    if "1.1" not in applicable_subtasks(llm_csv, LLM_DIR):
        print(f"  [SKIP 1.1] not applicable by rules – {llm_csv.name}")
        return

    raw = pd.read_csv(llm_csv)
    if "Review ID" not in raw.columns and "ID" in raw.columns:
        raw = raw.rename(columns={"ID": "Review ID"})

    if "Errors" in raw.columns:
        raw = raw[raw["Errors"].isna() | (raw["Errors"].astype(str).str.strip().isin({"", "nan"}))].copy()

    fmt = ("Topics Present" if "Topics Present" in raw.columns
           else "Topic"     if "Topic"          in raw.columns
           else None)

    if fmt is None:
        print(f"  [SKIP 1.1] no topic column – {llm_csv.name}")
        return

    llm_sets = _load_topic_sets(raw)
    if not llm_sets:
        print(f"  [SKIP 1.1] no valid topic data – {llm_csv.name}")
        return

    tp: dict = defaultdict(int)
    fp: dict = defaultdict(int)
    fn: dict = defaultdict(int)
    tn: dict = defaultdict(int)

    all_rids = sorted(llm_sets)
    id_stats: dict = {}

    workings: list = []
    W = workings.append
    W("=" * 72)
    W(f"WORKINGS  –  {llm_csv.name}")
    W("=" * 72)
    W(f"\nColumn format   : {fmt}")
    W(f"GT  Review IDs  : {len(gt_sets)}")
    W(f"LLM Review IDs  : {len(llm_sets)}")
    W(f"IDs evaluated   : {len(all_rids)}")
    W(f"Global topics   : {ALL_TOPICS}\n")

    for rid in all_rids:
        gt_t  = gt_sets.get(rid,  set())
        llm_t = llm_sets.get(rid, set())
        id_tp = id_fp = id_fn = id_tn = 0

        W(f"\n{'─' * 60}")
        W(f"Review ID : {rid}")
        W(f"  GT  topics : {sorted(gt_t)  if gt_t  else '(none)'}")
        W(f"  LLM topics : {sorted(llm_t) if llm_t else '(none)'}")
        W("")

        for topic in ALL_TOPICS:
            in_gt  = topic in gt_t
            in_llm = topic in llm_t
            if   in_gt  and     in_llm: tp[topic] += 1; id_tp += 1; label = "TP"
            elif in_gt  and not in_llm: fn[topic] += 1; id_fn += 1; label = "FN"
            elif not in_gt and  in_llm: fp[topic] += 1; id_fp += 1; label = "FP"
            else:                       tn[topic] += 1; id_tn += 1; label = "TN"
            W(f"    {topic:<20}  GT={'yes' if in_gt else 'no ':<4} "
              f"LLM={'yes' if in_llm else 'no ':<4} → {label}")

        id_stats[rid] = {
            "tp": id_tp, "fp": id_fp, "fn": id_fn, "tn": id_tn,
            "gt": gt_t,  "llm": llm_t,
        }

    # Results
    results: list = []
    R = results.append
    R("=" * 72)
    R(f"EVALUATION RESULTS  –  {llm_csv.name}")
    R("=" * 72)
    R(f"\nReview IDs in LLM  : {len(all_rids)}")
    R(f"Review IDs in GT   : {len(gt_sets)}")
    R(f"Topics evaluated   : {len(ALL_TOPICS)}")
    R(f"Total observations : {len(all_rids)} IDs x {len(ALL_TOPICS)} topics"
      f" = {len(all_rids) * len(ALL_TOPICS)}")

    ov_tp = sum(tp.values()); ov_fp = sum(fp.values())
    ov_fn = sum(fn.values()); ov_tn = sum(tn.values())
    ov_total = ov_tp + ov_fp + ov_fn + ov_tn
    ov_acc  = (ov_tp + ov_tn) / ov_total        if ov_total         > 0 else 0
    ov_prec = ov_tp / (ov_tp + ov_fp)           if (ov_tp + ov_fp)  > 0 else 0
    ov_rec  = ov_tp / (ov_tp + ov_fn)           if (ov_tp + ov_fn)  > 0 else 0
    ov_f1   = 2*ov_prec*ov_rec/(ov_prec+ov_rec) if (ov_prec+ov_rec) > 0 else 0

    R(f"\n{'─' * 72}")
    R("OVERALL SUMMARY")
    R(f"{'─' * 72}")
    R(f"  TP={ov_tp}  FP={ov_fp}  FN={ov_fn}  TN={ov_tn}")
    R(f"  Accuracy  : {ov_acc:.4f}")
    R(f"  Precision : {ov_prec:.4f}")
    R(f"  Recall    : {ov_rec:.4f}")
    R(f"  F1        : {ov_f1:.4f}")

    R(f"\n{'─' * 72}")
    R("PER-ID RESULTS")
    R(f"{'─' * 72}")
    R(f"{'ID':>6} {'Prec':>8} {'Rec':>8} {'F1':>8}  {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
    R("─" * 72)
    for rid in all_rids:
        s = id_stats[rid]
        i_tp, i_fp, i_fn, i_tn = s["tp"], s["fp"], s["fn"], s["tn"]
        i_prec = i_tp / (i_tp + i_fp) if (i_tp + i_fp) > 0 else 0
        i_rec  = i_tp / (i_tp + i_fn) if (i_tp + i_fn) > 0 else 0
        i_f1   = 2*i_prec*i_rec / (i_prec+i_rec) if (i_prec+i_rec) > 0 else 0
        R(f"{rid:>6} {i_prec:>8.4f} {i_rec:>8.4f} {i_f1:>8.4f}"
          f"  {i_tp:>4} {i_fp:>4} {i_fn:>4} {i_tn:>4}")
    R("─" * 72)

    R(f"\n{'─' * 72}")
    R("PER-TOPIC METRICS")
    R(f"{'─' * 72}")
    R(f"{'Topic':<20} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}"
      f"  {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}")
    R("─" * 72)

    micro_tp = micro_fp = micro_fn = 0
    for topic in ALL_TOPICS:
        t = tp[topic]; f_p = fp[topic]; f_n = fn[topic]; t_n = tn[topic]
        total = t + f_p + f_n + t_n
        acc  = (t + t_n) / total     if total        > 0 else 0
        prec = t / (t + f_p)         if (t + f_p)    > 0 else 0
        rec  = t / (t + f_n)         if (t + f_n)    > 0 else 0
        f1   = 2*prec*rec/(prec+rec) if (prec + rec) > 0 else 0
        micro_tp += t; micro_fp += f_p; micro_fn += f_n
        R(f"{topic:<20} {acc:>8.4f} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f}"
          f"  {t:>5} {f_p:>5} {f_n:>5} {t_n:>5}")
        W(f"\n[Per-topic totals: {topic}]")
        W(f"  TP={t}  FP={f_p}  FN={f_n}  TN={t_n}")
        W(f"  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
    R("─" * 72)

    micro_prec = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0
    micro_rec  = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0
    micro_f1   = (2*micro_prec*micro_rec / (micro_prec+micro_rec)
                  if (micro_prec + micro_rec) > 0 else 0)

    R(f"\n{'─' * 72}")
    R("MICRO-AVERAGED METRICS  (aggregate TP/FP/FN across all topics)")
    R(f"{'─' * 72}")
    R(f"  Total TP={micro_tp}  FP={micro_fp}  FN={micro_fn}")
    R(f"  Micro Precision : {micro_prec:.4f}")
    R(f"  Micro Recall    : {micro_rec:.4f}")
    R(f"  Micro F1        : {micro_f1:.4f}")

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

for llm_csv in llm_csvs:
    print(f"\n{'#' * 72}")
    print(f"Processing: {llm_csv.relative_to(TASK_DIR)}")
    print(f"{'#' * 72}")
    evaluate(llm_csv)

print(f"\n\nAll done. Results in: {EVAL_DIR.relative_to(TASK_DIR)}/")
