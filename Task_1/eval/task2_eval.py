"""
Task 2 Evaluation — Body Label Prediction.

Computes Precision, Recall, F1 (multi-label) for body and contrastive labels.
Compares source=gt vs source=llm side-by-side.

Usage:
    python Task_1/eval/task2_eval.py
    python Task_1/eval/task2_eval.py --model llama4_scout
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

TASK_DIR = Path(__file__).resolve().parent.parent
GT_CSV   = Path("/Users/Pngoendee/ReaLearn/Dataset") / (
    "Original ABA Dataset for Version 2 "
    "[Oct 23, 2025], Senior Project, MUICT - Sheet2_.csv"
)
TASK2_DIR = TASK_DIR / "outputs" / "task2"
EVAL_DIR  = TASK_DIR / "outputs" / "eval" / "task2"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


# ── Ground truth builder ───────────────────────────────────────────────────────

def build_gt_index(gold_csv: Path) -> dict[tuple, dict[str, set]]:
    """Returns {(review_id, topic): {"body": set, "cont": set}}."""
    gt = pd.read_csv(gold_csv)
    gt = gt.rename(columns={"ID": "Review ID"})
    gt = gt[gt["Topic"].notna()].copy()
    gt["Topic"] = gt["Topic"].str.strip()
    gt["Review ID"] = gt["Review ID"].astype(str)

    body_cols = [f"Body {i}" for i in range(1, 16)]
    cont_cols  = [f"Cont. Body {i}" for i in range(1, 16)]

    index: dict[tuple, dict[str, set]] = defaultdict(lambda: {"body": set(), "cont": set()})

    for _, row in gt.iterrows():
        key = (str(row["Review ID"]), row["Topic"])
        for c in body_cols:
            v = str(row[c]).strip() if pd.notna(row[c]) else ""
            if v:
                index[key]["body"].add(v)
        for c in cont_cols:
            v = str(row[c]).strip() if pd.notna(row[c]) else ""
            if v:
                index[key]["cont"].add(v)

    return dict(index)


# ── Metrics helpers ────────────────────────────────────────────────────────────

def prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


# ── Load Task 2 predictions from CSV ──────────────────────────────────────────

def load_predictions(csv_path: Path) -> dict[tuple, dict[str, set]]:
    """Returns {(review_id, topic, run): {"body": set, "cont": set}}."""
    df = pd.read_csv(csv_path)
    df["Review ID"] = df["Review ID"].astype(str)

    preds: dict[tuple, dict[str, set]] = defaultdict(lambda: {"body": set(), "cont": set()})

    for _, row in df.iterrows():
        rid = str(row["Review ID"])
        topic = str(row["Topic"]).strip()
        run_val = row.get("Run", "")
        run = str(int(run_val)) if pd.notna(run_val) and str(run_val).strip() not in ("", "nan") else "1"
        key = (rid, topic, run)

        label = str(row.get("Label", "")).strip()
        ltype = str(row.get("Label Type", "")).strip()

        if label in ("(no labels)", "(parse failed)", ""):
            continue
        if ltype == "body":
            preds[key]["body"].add(label)
        elif ltype == "cont":
            preds[key]["cont"].add(label)

    return dict(preds)


# ── Core evaluation ────────────────────────────────────────────────────────────

def evaluate_file(
    csv_path: Path,
    gt_index: dict[tuple, dict[str, set]],
    *,
    label_type: str = "both",  # "body", "cont", or "both"
) -> dict:
    """
    Returns a summary dict with keys:
      micro: {prec, rec, f1, tp, fp, fn}
      per_topic: {topic: {prec, rec, f1, tp, fp, fn}}
      per_instance: list of {review_id, topic, run, tp, fp, fn, prec, rec, f1}
    """
    preds = load_predictions(csv_path)
    if not preds:
        return {"micro": {}, "per_topic": {}, "per_instance": []}

    # Collect all (review_id, topic) pairs from predictions
    # key is (rid, topic, run) — group by (rid, topic) when run is irrelevant (gt source)
    micro_tp = micro_fp = micro_fn = 0
    topic_tp: dict[str, int] = defaultdict(int)
    topic_fp: dict[str, int] = defaultdict(int)
    topic_fn: dict[str, int] = defaultdict(int)
    per_instance = []

    for (rid, topic, run), pred_lbls in preds.items():
        gt_entry = gt_index.get((rid, topic), {"body": set(), "cont": set()})

        if label_type == "body":
            pred_set = pred_lbls["body"]
            gt_set   = gt_entry["body"]
        elif label_type == "cont":
            pred_set = pred_lbls["cont"]
            gt_set   = gt_entry["cont"]
        else:
            pred_set = pred_lbls["body"] | pred_lbls["cont"]
            gt_set   = gt_entry["body"]  | gt_entry["cont"]

        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        prec, rec, f1 = prf1(tp, fp, fn)

        micro_tp += tp; micro_fp += fp; micro_fn += fn
        topic_tp[topic] += tp; topic_fp[topic] += fp; topic_fn[topic] += fn

        per_instance.append({
            "review_id": rid, "topic": topic, "run": run,
            "tp": tp, "fp": fp, "fn": fn,
            "prec": prec, "rec": rec, "f1": f1,
            "pred": sorted(pred_set), "gt": sorted(gt_set),
        })

    micro_prec, micro_rec, micro_f1 = prf1(micro_tp, micro_fp, micro_fn)
    micro = {"prec": micro_prec, "rec": micro_rec, "f1": micro_f1,
             "tp": micro_tp, "fp": micro_fp, "fn": micro_fn}

    per_topic = {}
    for topic in sorted(set(topic_tp) | set(topic_fp) | set(topic_fn)):
        tp_ = topic_tp[topic]; fp_ = topic_fp[topic]; fn_ = topic_fn[topic]
        p, r, f = prf1(tp_, fp_, fn_)
        per_topic[topic] = {"prec": p, "rec": r, "f1": f, "tp": tp_, "fp": fp_, "fn": fn_}

    return {"micro": micro, "per_topic": per_topic, "per_instance": per_instance}


# ── Report writer ──────────────────────────────────────────────────────────────

def _fmt_section(title: str, result: dict, label_type: str) -> list[str]:
    lines = []
    L = lines.append
    if not result.get("micro"):
        L(f"  [No data for {title}]")
        return lines

    m = result["micro"]
    L(f"\n{'─' * 72}")
    L(f"  {title}  (label_type={label_type})")
    L(f"{'─' * 72}")
    L(f"  Instances evaluated : {len(result['per_instance'])}")
    L(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")
    L(f"  Micro Precision : {m['prec']:.4f}")
    L(f"  Micro Recall    : {m['rec']:.4f}")
    L(f"  Micro F1        : {m['f1']:.4f}")

    # Per-topic table
    pt = result["per_topic"]
    if pt:
        L(f"\n  {'Topic':<20} {'Prec':>8} {'Rec':>8} {'F1':>8}  {'TP':>5} {'FP':>5} {'FN':>5}")
        L("  " + "─" * 62)
        for topic in sorted(pt):
            t = pt[topic]
            L(f"  {topic:<20} {t['prec']:>8.4f} {t['rec']:>8.4f} {t['f1']:>8.4f}"
              f"  {t['tp']:>5} {t['fp']:>5} {t['fn']:>5}")
        L("  " + "─" * 62)

    return lines


def write_report(
    out_path: Path,
    csv_path: Path,
    gt_index: dict,
) -> None:
    lines: list[str] = []
    L = lines.append

    L("=" * 72)
    L(f"TASK 2 EVALUATION  —  {csv_path.name}")
    L("=" * 72)

    for ltype in ("body", "cont", "both"):
        result = evaluate_file(csv_path, gt_index, label_type=ltype)
        lines.extend(_fmt_section(f"Label type: {ltype.upper()}", result, ltype))

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {out_path.relative_to(TASK_DIR)}")
    print("\n".join(lines[:40]))


# ── Comparison table ───────────────────────────────────────────────────────────

def comparison_table(
    results_by_label: dict[str, dict],
    csv_paths: list[Path],
    out_path: Path,
) -> None:
    lines: list[str] = []
    L = lines.append
    L("=" * 100)
    L("TASK 2  —  SOURCE COMPARISON  (source=gt vs source=llm)")
    L("=" * 100)

    for ltype in ("body", "cont", "both"):
        L(f"\n{'─' * 100}")
        L(f"  Label type: {ltype.upper()}")
        L(f"{'─' * 100}")
        header = f"  {'Source / File':<45} {'Prec':>8} {'Rec':>8} {'F1':>8}  {'TP':>6} {'FP':>6} {'FN':>6}"
        L(header)
        L("  " + "─" * 95)
        for csv_path in csv_paths:
            key = str(csv_path)
            r = results_by_label.get((key, ltype), {}).get("micro", {})
            if r:
                name = csv_path.stem[:45]
                L(f"  {name:<45} {r['prec']:>8.4f} {r['rec']:>8.4f} {r['f1']:>8.4f}"
                  f"  {r['tp']:>6} {r['fp']:>6} {r['fn']:>6}")
        L("  " + "─" * 95)

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nComparison table saved: {out_path.relative_to(TASK_DIR)}")
    print("\n".join(lines))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Task 2 body label predictions")
    parser.add_argument("--model", default=None, help="Filter by model subfolder (e.g. llama4_scout)")
    args = parser.parse_args()

    if not GT_CSV.exists():
        sys.exit(f"[ERROR] GT CSV not found:\n  {GT_CSV}")
    if not TASK2_DIR.exists():
        sys.exit(f"[ERROR] Task 2 output dir not found:\n  {TASK2_DIR}\n  Run run_task2.py first.")

    print(f"Building GT index from: {GT_CSV.name}")
    gt_index = build_gt_index(GT_CSV)
    print(f"  {len(gt_index)} (Review ID, Topic) entries in GT\n")

    csv_paths = sorted(
        p for p in TASK2_DIR.rglob("*.csv")
        if "old_output" not in p.parts
    )
    if args.model:
        csv_paths = [p for p in csv_paths if args.model in str(p)]

    if not csv_paths:
        sys.exit(f"[ERROR] No Task 2 CSV files found under {TASK2_DIR}")

    print(f"Found {len(csv_paths)} Task 2 CSV file(s):")
    for p in csv_paths:
        print(f"  {p.relative_to(TASK_DIR)}")

    results_by_label: dict[tuple, dict] = {}

    for csv_path in csv_paths:
        print(f"\n{'#' * 72}")
        print(f"Evaluating: {csv_path.relative_to(TASK_DIR)}")
        print(f"{'#' * 72}")

        subdir = csv_path.relative_to(TASK2_DIR)
        out_dir = EVAL_DIR / Path(*subdir.parts[:-1]) if len(subdir.parts) > 1 else EVAL_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (csv_path.stem + "_eval.txt")

        for ltype in ("body", "cont", "both"):
            r = evaluate_file(csv_path, gt_index, label_type=ltype)
            results_by_label[(str(csv_path), ltype)] = r

        write_report(out_path, csv_path, gt_index)

    if len(csv_paths) > 1:
        comparison_table(
            results_by_label,
            csv_paths,
            EVAL_DIR / "comparison_table.txt",
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
