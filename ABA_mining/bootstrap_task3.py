"""Variability of the Task 3 scores: spread across the three runs, and a bootstrap over pairs.

The Task 3 evaluator already scores each of the three runs separately -- they are the rows
"Test 1", "Test 2", "Test 3" in every EVAL_SUMMARY.txt -- so the run-to-run spread needs no
re-running.

The resampling unit here is the candidate PAIR, not the review: a Task 3 item is an
(assumption, proposition) pair drawn from a topic's pair set, and the pairs are not grouped by
review the way Task 2's instances are. F1 depends only on the TP/TN/FP/FN counts, so resampling
pairs from a population with that composition is exact.

Usage:
    python bootstrap_task3.py
    python bootstrap_task3.py --spread-only
    python bootstrap_task3.py --model qwen3.8_27b
"""
from __future__ import annotations

import argparse
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("outputs/eval/task3_server")
# Model Shot Topic Category Test Positive_Label N_valid TP TN FP FN P R F1 Acc
ROW = re.compile(
    r"^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(Test \d)\s+(\S+)\s+"
    r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", re.M)


def f1_from(tp: float, fp: float, fn: float) -> float:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def bootstrap_pairs(tp: int, tn: int, fp: int, fn: int, reps: int, seed: int, level: float):
    """Resample the N scored pairs with replacement; F1 is a function of the counts alone."""
    n = tp + tn + fp + fn
    if n == 0:
        return None
    probs = np.array([tp, tn, fp, fn], dtype=float) / n
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    for b in range(reps):
        c = rng.multinomial(n, probs)          # one resample of n pairs
        draws[b] = f1_from(c[0], c[2], c[3])
    lo, hi = np.percentile(draws, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return f1_from(tp, fp, fn), lo, hi, draws.std(ddof=1), n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--model", default=None, help="restrict to one model")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--spread-only", action="store_true")
    args = ap.parse_args()

    # (model, prompt, topic, category) -> {test -> (tp, tn, fp, fn, f1)}
    cells: dict[tuple, dict[str, tuple]] = defaultdict(dict)
    for p in sorted(Path(args.root).rglob("EVAL_SUMMARY.txt")):
        head = p.read_text(encoding="utf-8", errors="replace").split("MICRO_BY_CATEGORY")[0]
        for m in ROW.finditer(head):
            model, shot, topic, cat, test = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
            if args.model and model != args.model:
                continue
            tp, tn, fp, fn = (int(m.group(i)) for i in (8, 9, 10, 11))
            cells[(model, shot, topic, cat)][test] = (tp, tn, fp, fn, float(m.group(14)))

    if not cells:
        print("no EVAL_SUMMARY rows found")
        return

    print("RUN-TO-RUN SPREAD  (F1 over Test 1/2/3)")
    print(f"{'model':13}{'prompt':13}{'topic':11}{'runs':>5}{'pairs':>8}{'mean F1':>10}{'sd':>10}")
    print("-" * 70)
    worst = 0.0
    nonzero = 0
    for key, tests in sorted(cells.items()):
        f1s = [v[4] for v in tests.values()]
        n = max(sum(v[:4]) for v in tests.values())
        sd = st.stdev(f1s) if len(f1s) > 1 else 0.0
        worst = max(worst, sd)
        nonzero += sd > 0
        print(f"{key[0]:13}{key[1]:13}{key[2]:11}{len(f1s):>5}{n:>8}"
              f"{st.mean(f1s):>10.4f}{sd:>10.5f}")
    print(f"\nconfigurations: {len(cells)}   with a non-zero run-to-run sd: {nonzero}")
    print(f"largest run-to-run standard deviation: {worst:.5f}")

    if args.spread_only:
        return

    print(f"\n\nBOOTSTRAP OVER PAIRS  (Test 1, B={args.reps}, {args.level:.0%} interval)")
    print(f"{'model':13}{'prompt':13}{'topic':11}{'pairs':>8}{'F1':>9}{'95% CI':>20}{'SE':>9}")
    print("-" * 83)
    ses = []
    for key, tests in sorted(cells.items()):
        first = tests.get("Test 1")
        if not first:
            continue
        res = bootstrap_pairs(*first[:4], args.reps, args.seed, args.level)
        if res is None:
            continue
        f1, lo, hi, se, n = res
        ses.append(se)
        print(f"{key[0]:13}{key[1]:13}{key[2]:11}{n:>8}{f1:>9.4f}"
              f"{f'[{lo:.4f}, {hi:.4f}]':>20}{se:>9.4f}")
    if ses:
        print(f"\nbootstrap standard error: {min(ses):.4f} to {max(ses):.4f}")


if __name__ == "__main__":
    main()
