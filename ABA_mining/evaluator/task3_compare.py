#### PATCHARAKORN ####

"""
Cross-model comparison for Task 3, built on top of task3_eval.py's master summary.

Why a separate step: task3_eval.py scores every file it finds, but the four models in the
server batch were NOT run on equal footing, and a single "average F1 per model" table would
quietly compare different things. This script only ever compares like with like:

  FAIR set   check-in, check-out, price — every model ran the complete gold set for these.
  FULL set   all five topics — only models that ran every topic on the complete gold set.

Anything that doesn't meet a set's requirements is reported as excluded, with the reason,
rather than dropped silently.

Metrics are MICRO: TP/FP/FN/TN are pooled across the set's topics for one (model, version, run)
and the metrics computed from the pooled counts, so check-out's 8 rows weigh 8 rows and not a
fifth of the result. Each (model, version) is then summarised as mean ± std over its runs.

Only Contrary(P)Body(N) is compared — it is the one category every model was evaluated on.

Usage:
    python evaluator/task3_compare.py --eval-dir outputs/eval/task3_server
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

CATEGORY = "Contrary(P)Body(N)"
FAIR_TOPICS = ["check-in", "check-out", "price"]
ALL_TOPICS = ["check-in", "check-out", "price", "staff", "facility"]
VERSION_ORDER = ["zero_shot", "one_shot", "contrary_v1", "contrary_v2", "contrary_v3",
                 "contrary_v4", "contrary_v5", "contrary_v6"]


def _div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _micro(group: pd.DataFrame) -> pd.Series:
    tp, fp, fn, tn = (group[c].sum() for c in ("TP", "FP", "FN", "TN"))
    p, r = _div(tp, tp + fp), _div(tp, tp + fn)
    return pd.Series({
        "Precision": p,
        "Recall": r,
        "F1": _div(2 * p * r, p + r),
        "Accuracy": _div(tp + tn, tp + tn + fp + fn),
        "N_valid": int(group["N_valid"].sum()),
        "Pred_Yes": int(tp + fp),
    })


def _complete_runs(df: pd.DataFrame, topics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keeps (Model, Shot, Test) runs that scored every topic in `topics` on the full gold set.

    Returns (kept_rows, exclusions) — exclusions explains every run that was left out.
    """
    sub = df[df["Topic"].isin(topics)]
    kept, excluded = [], []
    for (model, shot, test), g in sub.groupby(["Model", "Shot", "Test"]):
        missing = sorted(set(topics) - set(g["Topic"]))
        partial = g[g["N_pred"] < g["N_gold"]]
        if missing:
            excluded.append({"Model": model, "Shot": shot, "Test": test,
                             "Reason": f"no result for {', '.join(missing)}"})
        elif not partial.empty:
            detail = "; ".join(f"{r.Topic} {int(r.N_pred)}/{int(r.N_gold)}" for r in partial.itertuples())
            excluded.append({"Model": model, "Shot": shot, "Test": test,
                             "Reason": f"ran on a subset only ({detail})"})
        else:
            kept.append(g)
    kept_df = pd.concat(kept, ignore_index=True) if kept else sub.iloc[0:0]
    return kept_df, pd.DataFrame(excluded)


def summarise(df: pd.DataFrame, topics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    kept, excluded = _complete_runs(df, topics)
    if kept.empty:
        return pd.DataFrame(), excluded

    per_run = kept.groupby(["Model", "Shot", "Test"]).apply(_micro, include_groups=False).reset_index()

    rows = []
    for (model, shot), g in per_run.groupby(["Model", "Shot"]):
        row = {"Model": model, "Version": shot, "Runs": len(g), "N_valid/run": int(g["N_valid"].mean())}
        for m in ("F1", "Precision", "Recall", "Accuracy"):
            row[m] = g[m].mean()
            row[f"{m}_std"] = g[m].std(ddof=0) if len(g) > 1 else np.nan
        row["Pred_Yes/run"] = g["Pred_Yes"].mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    out["_v"] = out["Version"].map({v: i for i, v in enumerate(VERSION_ORDER)})
    out = out.sort_values(["Model", "_v"]).drop(columns="_v").reset_index(drop=True)
    return out, excluded


def gold_positive_rate(topics: list[str]) -> float:
    """Share of gold 'Yes' among the gold Yes/No votes in the set ('Undetermined' excluded, as
    the evaluator does). A model answering 'Yes' to everything scores F1 = 2r/(1+r) — the floor
    any real model should clear. Read from the gold files themselves, not from a model's output."""
    import sys
    for p in (REPO_ROOT, REPO_ROOT.parent.parent):
        sys.path.insert(0, str(p))
    from internship.ABA_mining.src import load_paths_config
    from internship.ABA_mining.src.task3 import load_task3_instances

    task3_dir = load_paths_config(REPO_ROOT).task3_dir
    votes = [inst.gold_vote for inst in load_task3_instances(
        task3_dir, aspects=topics, categories=[CATEGORY], max_per_group=None)]
    yes = sum(1 for v in votes if str(v).strip().lower() == "yes")
    no = sum(1 for v in votes if str(v).strip().lower() == "no")
    return _div(yes, yes + no)


def fmt(v: float, std: float | None = None) -> str:
    if pd.isna(v):
        return "–"
    s = f"{v:.3f}"
    if std is not None and not pd.isna(std):
        s += f" ± {std:.3f}"
    return s


def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair cross-model comparison for Task 3.")
    parser.add_argument("--eval-dir", default="outputs/eval/task3_server")
    args = parser.parse_args()
    eval_dir = (REPO_ROOT / args.eval_dir).resolve() if not Path(args.eval_dir).is_absolute() else Path(args.eval_dir)

    master = eval_dir / "TASK3_VERIFY_MASTER_SUMMARY_ALL_TOPICS.xlsx"
    df = pd.read_excel(master, sheet_name="ALL_EVAL_SUMMARY")
    df = df[df["Category"] == CATEGORY].copy()

    report: list[str] = ["# Task 3 — cross-model comparison", "",
                         f"Source: `{master.relative_to(REPO_ROOT)}` · category `{CATEGORY}` · micro-averaged, "
                         "mean ± std over runs.", ""]

    for title, topics in (("FAIR set — check-in, check-out, price (all 4 models)", FAIR_TOPICS),
                          ("FULL set — all 5 topics", ALL_TOPICS)):
        summary, excluded = summarise(df, topics)
        tag = "fair" if topics == FAIR_TOPICS else "full"
        if not excluded.empty:
            excluded.to_csv(eval_dir / f"COMPARE_{tag}_EXCLUDED.csv", index=False)

        rate = gold_positive_rate(topics)
        always_yes_f1 = 2 * rate / (1 + rate)
        majority_acc = max(rate, 1 - rate)
        majority = "Yes" if rate >= 0.5 else "No"
        report += [f"## {title}", "",
                   f"Gold 'Yes' rate in this set: **{rate:.1%}**. Trivial baselines to beat:", "",
                   f"- answering **Yes** to everything → F1 = **{always_yes_f1:.3f}**",
                   f"- answering **{majority}** (the majority class) to everything → "
                   f"Accuracy = **{majority_acc:.3f}**", "",
                   "A model is only doing real work if it clears these, so read every score below "
                   "against them.", ""]

        if summary.empty:
            report += ["_No model has complete runs for this set._", ""]
        else:
            summary["F1_vs_allYes"] = summary["F1"] - always_yes_f1
            summary["Acc_vs_majority"] = summary["Accuracy"] - majority_acc
            summary.to_csv(eval_dir / f"COMPARE_{tag}.csv", index=False)

            view = summary.copy()
            view["F1"] = [fmt(a, b) for a, b in zip(summary["F1"], summary["F1_std"])]
            for m in ("Precision", "Recall", "Accuracy"):
                view[m] = [fmt(a) for a in summary[m]]
            view["ΔF1 vs all-Yes"] = [f"{d:+.3f}" for d in summary["F1_vs_allYes"]]
            view["ΔAcc vs majority"] = [f"{d:+.3f}" for d in summary["Acc_vs_majority"]]
            report += [md_table(view, ["Model", "Version", "Runs", "N_valid/run",
                                       "F1", "ΔF1 vs all-Yes", "Precision", "Recall",
                                       "Accuracy", "ΔAcc vs majority"]), ""]

            best = summary.loc[summary.groupby("Model")["F1"].idxmax()].sort_values("F1", ascending=False)
            best_view = best.copy()
            best_view["F1"] = [fmt(a, b) for a, b in zip(best["F1"], best["F1_std"])]
            best_view["Precision"] = [fmt(a) for a in best["Precision"]]
            best_view["Recall"] = [fmt(a) for a in best["Recall"]]
            report += ["**Best version per model:**", "",
                       md_table(best_view, ["Model", "Version", "Runs", "F1", "Precision", "Recall"]), ""]

        if not excluded.empty:
            report += ["<details><summary>Runs excluded from this set "
                       f"({len(excluded)})</summary>", "",
                       md_table(excluded, ["Model", "Shot", "Test", "Reason"]), "", "</details>", ""]

    # per-topic view, all runs that exist (no completeness filter), for context
    per_topic = (df.groupby(["Model", "Shot", "Topic"])
                   .apply(_micro, include_groups=False).reset_index())
    per_topic.to_csv(eval_dir / "COMPARE_per_topic.csv", index=False)

    out_md = eval_dir / "TASK3_MODEL_COMPARISON.md"
    out_md.write_text("\n".join(report), encoding="utf-8")
    # The Windows console here defaults to a codepage (cp874) that can't encode ±, Δ or ·.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"[COMPARISON] {out_md.relative_to(REPO_ROOT)}")
    print("\n".join(report))


if __name__ == "__main__":
    main()
