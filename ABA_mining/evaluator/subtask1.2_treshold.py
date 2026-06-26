#### RATCHANON ####

import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASK_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = TASK_DIR / "outputs" / "eval" / "content"
OUT_DIR  = TASK_DIR / "outputs" / "eval" / "threshold_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_METHODS = ["Parity", "TF-IDF", "BM25", "Sentence Emb", "Cross-Encoder"]
METHODS     = ["TF-IDF", "Sentence Emb", "Cross-Encoder"]
THRESHOLDS  = [round(t * 0.1, 1) for t in range(1, 11)]  # 0.1 to 1.0

COLORS     = {"TF-IDF": "#ff7f0e", "Sentence Emb": "#9467bd", "Cross-Encoder": "#d62728"}
LINESTYLES = ["-", "--", ":"]
MARKERS    = ["o", "s", "D"]

METHOD_PAT = re.compile(
    r"^\s+(Parity|TF-IDF|BM25|Sentence Emb|Cross-Encoder)\s*:.*?best=(-?\S+)"
)
SEP_PAT = re.compile(r"^[─\-]{50,}$")
RUN_PAT = re.compile(r"^\s*RUN\s+(\d+)\s*$")


def parse_workings(path: Path) -> dict:
    """Returns {run_id: {method: [best_scores]}} for all methods."""
    run_scores = defaultdict(lambda: {m: [] for m in ALL_METHODS})
    current_run = 1
    current = {}

    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            rm = RUN_PAT.match(line)
            if rm:
                current_run = int(rm.group(1))
                current = {}
                continue
            if SEP_PAT.match(line.strip()):
                current = {}
                continue
            if "[No GT rows" in line:
                for m in ALL_METHODS:
                    run_scores[current_run][m].append(0.0)
                current = {}
                continue

            hit = METHOD_PAT.match(line)
            if hit:
                method = hit.group(1)
                try:
                    score = float(hit.group(2))
                except ValueError:
                    score = 0.0
                current[method] = score
                if len(current) == len(ALL_METHODS):
                    for m in ALL_METHODS:
                        run_scores[current_run][m].append(current[m])
                    current = {}

    return dict(run_scores)


def passes(scores: list, threshold: float) -> int:
    return sum(1 for s in scores if s >= threshold)


def print_tables(label: str, run_scores: dict) -> dict:
    results = {}
    for run_id in sorted(run_scores.keys()):
        sc    = run_scores[run_id]
        total = len(sc[METHODS[0]])
        if total == 0:
            continue
        results[run_id] = {"total": total, "thresholds": {}}

        print(f"\n{'=' * 60}")
        print(f"FILE : {label}  |  RUN {run_id}  (rows: {total})")
        print(f"{'=' * 60}")

        for t in THRESHOLDS:
            print(f"\n  threshold = {t:.1f}")
            print(f"  {'Method':<16} {'Passed':>8} {'Total':>7} {'Accuracy':>10}")
            print("  " + "-" * 44)
            results[run_id]["thresholds"][t] = {}
            for m in METHODS:
                p   = passes(sc[m], t)
                acc = p / total
                results[run_id]["thresholds"][t][m] = {"passed": p, "accuracy": acc}
                print(f"  {m:<16} {p:>8} {total:>7} {acc:>10.4f}")

    return results


def _draw_9lines(ax, run_scores: dict, value_fn):
    runs = sorted(run_scores.keys())
    for run_idx, run_id in enumerate(runs):
        sc    = run_scores[run_id]
        total = len(sc[METHODS[0]])
        if total == 0:
            continue
        ls = LINESTYLES[run_idx % len(LINESTYLES)]
        mk = MARKERS[run_idx % len(MARKERS)]
        for m in METHODS:
            ys = [value_fn(sc[m], t) for t in THRESHOLDS]
            ax.plot(THRESHOLDS, ys,
                    color=COLORS[m], linestyle=ls, marker=mk,
                    linewidth=1.6, markersize=5,
                    label=f"{m}  Run {run_id}")
    ax.set_xticks(THRESHOLDS)
    ax.set_xticklabels([f"{t:.1f}" for t in THRESHOLDS])
    ax.set_xlabel("Threshold", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5, ncol=3, loc="upper right")


def plot_all_models(all_run_scores: dict) -> None:
    labels = list(all_run_scores.keys())
    n      = len(labels)
    cols   = 2
    rows   = (n + 1) // cols

    for chart_type in ("accuracy", "count"):
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(13 * cols, 6 * rows),
                                 squeeze=False)
        ax_flat = [axes[r][c] for r in range(rows) for c in range(cols)]

        for ax, label in zip(ax_flat, labels):
            run_scores = all_run_scores[label]

            if chart_type == "accuracy":
                _draw_9lines(ax, run_scores,
                             value_fn=lambda sc, t: passes(sc, t) / len(sc) if sc else 0)
                ax.set_ylabel("Accuracy (Pass Rate)", fontsize=9)
                ax.set_ylim(0, 1.05)
            else:
                _draw_9lines(ax, run_scores,
                             value_fn=lambda sc, t: passes(sc, t))
                for run_id in sorted(run_scores.keys()):
                    total = len(run_scores[run_id][METHODS[0]])
                    if total:
                        ax.axhline(total, color="gray", linewidth=0.8,
                                   linestyle="-.", alpha=0.5,
                                   label=f"Total Run {run_id} ({total})")
                ax.set_ylabel("Rows Passed", fontsize=9)

            ax.set_title(label, fontsize=10, pad=6)

        for ax in ax_flat[n:]:
            ax.set_visible(False)

        title = ("Subtask 1.2 - Accuracy vs Threshold  (all models)"
                 if chart_type == "accuracy"
                 else "Subtask 1.2 - Rows Passed vs Threshold  (all models)")
        fig.suptitle(title, fontsize=13, y=1.01)
        plt.tight_layout()
        out = OUT_DIR / f"threshold_sweep_{chart_type}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out.relative_to(TASK_DIR)}")



workings_files = sorted(
    p for p in EVAL_DIR.rglob("*_workings.txt")
    if "old_eval" not in p.parts
)
if not workings_files:
    sys.exit(f"[ERROR] No workings files found under {EVAL_DIR}")

print(f"Found {len(workings_files)} workings file(s):\n")
for p in workings_files:
    print(f"  {p.relative_to(TASK_DIR)}")

summary_lines:   list = []
all_run_scores: dict = {}

for wf in workings_files:
    label      = wf.stem.replace("_workings", "")
    run_scores = parse_workings(wf)
    all_run_scores[label] = run_scores

    print(f"\n{'#' * 60}")
    print(f"  {label}")
    print(f"{'#' * 60}")

    results = print_tables(label, run_scores)

    for run_id, r in results.items():
        total = r["total"]
        summary_lines += [
            f"\n{'=' * 60}",
            f"FILE : {label}  |  RUN {run_id}  (rows: {total})",
            f"{'=' * 60}",
        ]
        for t in THRESHOLDS:
            summary_lines.append(f"\n  threshold = {t:.1f}")
            summary_lines.append(f"  {'Method':<16} {'Passed':>8} {'Total':>7} {'Accuracy':>10}")
            summary_lines.append("  " + "-" * 44)
            for m in METHODS:
                d = r["thresholds"][t][m]
                summary_lines.append(
                    f"  {m:<16} {d['passed']:>8} {total:>7} {d['accuracy']:>10.4f}")

print()
plot_all_models(all_run_scores)

summary_path = OUT_DIR / "threshold_sweep_summary.txt"
summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
print(f"\nSummary saved: {summary_path.relative_to(TASK_DIR)}")
print("\nAll done.")
