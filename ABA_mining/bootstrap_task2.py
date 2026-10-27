"""Variability of the Task 2 scores: spread across repeated runs, and a bootstrap over reviews.

Two different questions, and they do not have the same answer.

  run-to-run spread   how much the score moves if the same configuration is run again.
                      Decoding is deterministic (temperature 0), so this is near zero and says
                      almost nothing.

  bootstrap over      how much the score would move if the corpus had contained a different
  reviews             sample of reviews. This is the uncertainty worth reporting.

Both are read out of the evaluator's own output, so neither needs the models re-run:

  * the per-run `*_results.txt` gives the headline micro P/R/F1 (spread across runs);
  * the matching `*_workings.txt` gives one `=> instance TP=.. FP=.. FN=..` line per
    (review, topic), which is exactly what the micro ratios are summed from, so resampling
    reviews and re-summing gives an exact bootstrap.

Reviews, not instances, are the resampling unit: the instances mined from one review share its
text and its annotator, so resampling instances would treat dependent observations as
independent and report an interval that is too narrow.

Usage:
    python bootstrap_task2.py --root outputs/Task2_new_eval/task2/gt
    python bootstrap_task2.py --root outputs/Task2_new_eval/task2/gt --spread-only
"""
from __future__ import annotations

import argparse
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

NUM = r"([01]\.\d+)"
INSTANCE = re.compile(r"^ID=(\S+)\s+Topic=(\S+)", re.M)
COUNTS = re.compile(r"^\s*=> instance TP=(\d+) FP=(\d+) FN=(\d+)", re.M)


def parse_results(path: Path) -> dict:
    """Headline numbers of one run."""
    t = path.read_text(encoding="utf-8", errors="replace")
    rec = {}
    m = re.search(r"Triple-Gate pass rate \(accuracy\)\s*:\s*\d+/\d+ = " + NUM, t)
    if m:
        rec["head"] = float(m.group(1))
    m = re.search(r"Error rate\s*:\s*\d+/\d+ = " + NUM, t)
    if m:
        rec["error_rate"] = float(m.group(1))
    blocks = t.split("--- INCLUDING error instances")
    if len(blocks) > 1:
        b = blocks[1]
        for key, pat in (("P", r"Micro Precision\s*:\s*" + NUM),
                         ("R", r"Micro Recall\s*:\s*" + NUM),
                         ("F1", r"Micro F1\s*:\s*" + NUM)):
            mm = re.search(pat, b)
            if mm:
                rec[key] = float(mm.group(1))
    return rec


def parse_workings(path: Path) -> list[tuple[str, int, int, int]]:
    """(review, TP, FP, FN) per scored instance.

    The file has two sections that both start their blocks with `ID=`; only the body/cont
    section carries an `=> instance TP=` line, so pairing each counts line with the most recent
    `ID=` header picks out exactly the scored instances.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    heads = [(m.start(), m.group(1)) for m in INSTANCE.finditer(text)]
    out: list[tuple[str, int, int, int]] = []
    i = 0
    for m in COUNTS.finditer(text):
        while i + 1 < len(heads) and heads[i + 1][0] < m.start():
            i += 1
        if heads and heads[i][0] < m.start():
            out.append((heads[i][1], int(m.group(1)), int(m.group(2)), int(m.group(3))))
    return out


def micro(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def bootstrap(rows, reps: int, seed: int, level: float):
    by_review: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for review, tp, fp, fn in rows:
        acc = by_review[review]
        acc[0] += tp
        acc[1] += fp
        acc[2] += fn
    arr = np.array(list(by_review.values()), dtype=np.int64)
    n = len(arr)
    if n == 0:
        return None
    point = micro(*arr.sum(axis=0))
    rng = np.random.default_rng(seed)
    draws = np.empty((reps, 3))
    for b in range(reps):
        draws[b] = micro(*arr[rng.integers(0, n, n)].sum(axis=0))
    lo, hi = np.percentile(draws, [(1 - level) / 2 * 100, (1 + level) / 2 * 100], axis=0)
    return {"n": n, "point": point, "lo": lo, "hi": hi, "se": draws.std(axis=0, ddof=1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/Task2_new_eval/task2/gt")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--models", default="gemma4_31b,qwen3.8_27b,llama3.3_70b,llama4_scout,llama3.2",
                    help="comma-separated model folders to include; "
                         "the default omits qwen3.8_27b_old, which duplicates qwen3.8_27b")
    ap.add_argument("--spread-only", action="store_true")
    args = ap.parse_args()
    keep = {m.strip() for m in args.models.split(",") if m.strip()}

    root = Path(args.root)
    runs: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    workings: dict[tuple[str, str], Path] = {}
    for p in sorted(root.rglob("*_results.txt")):
        if "_avg_" in p.name:
            continue
        m = re.search(r"_v(\d)_run(\d)_", p.name)
        if not m:
            continue
        model = p.relative_to(root).parts[0]
        if keep and model not in keep:
            continue
        key = (model, f"v{m.group(1)}")
        runs[key][int(m.group(2))] = parse_results(p)
        if int(m.group(2)) == 1:
            w = p.with_name(p.name.replace("_results.txt", "_workings.txt"))
            if w.exists():
                workings[key] = w

    print("RUN-TO-RUN SPREAD  (mean +/- sample sd over the repeated runs)")
    print(f"{'model':16}{'ver':5}{'runs':>5}{'err':>7}  "
          + "".join(f"{c:>19}" for c in ("head", "P", "R", "F1")))
    print("-" * 94)
    worst = 0.0
    for key, per_run in sorted(runs.items()):
        err = max((r.get("error_rate", 0.0) for r in per_run.values()), default=0.0)
        cells = []
        for c in ("head", "P", "R", "F1"):
            vals = [r[c] for r in per_run.values() if c in r]
            if len(vals) >= 2:
                sd = st.stdev(vals)
                worst = max(worst, sd)
                cells.append(f"{st.mean(vals):>11.4f}±{sd:<7.5f}")
            elif vals:
                cells.append(f"{vals[0]:>14.4f}{'':>5}")
            else:
                cells.append(f"{'—':>19}")
        flag = "FAIL" if err >= 1.0 else f"{err:.3f}"
        print(f"{key[0]:16}{key[1]:5}{len(per_run):>5}{flag:>7}  " + "".join(cells))
    print(f"\nlargest run-to-run standard deviation anywhere: {worst:.5f}")
    print("'err' is the share of CSV rows the evaluator flagged invalid; FAIL = every row.")

    if args.spread_only:
        return

    print(f"\n\nBOOTSTRAP OVER REVIEWS  (run 1, B={args.reps}, {args.level:.0%} percentile interval)")
    print(f"{'model':16}{'ver':5}{'reviews':>8}  "
          + "".join(f"{c:>26}" for c in ("Precision", "Recall", "F1")))
    print("-" * 109)
    ses = []
    for key in sorted(workings):
        rows = parse_workings(workings[key])
        res = bootstrap(rows, args.reps, args.seed, args.level)
        if res is None:
            continue
        cells = []
        for i in range(3):
            cells.append(f"{res['point'][i]:>8.4f} [{res['lo'][i]:.4f},{res['hi'][i]:.4f}]")
            ses.append(res["se"][i])
        print(f"{key[0]:16}{key[1]:5}{res['n']:>8}  " + "".join(f"{c:>26}" for c in cells))
    if ses:
        print(f"\nbootstrap standard error across all cells: "
              f"{min(ses):.4f} to {max(ses):.4f}")


if __name__ == "__main__":
    main()
