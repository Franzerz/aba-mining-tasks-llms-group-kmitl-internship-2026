import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

TASK_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = TASK_DIR / "outputs" / "eval" / "content"
OUT_DIR  = TASK_DIR / "outputs" / "eval" / "cluster"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_METHODS = ["Parity", "TF-IDF", "BM25", "Sentence Emb", "Cross-Encoder"]
STD_METHODS = ["TF-IDF", "Sentence Emb", "Cross-Encoder"]

METHOD_SLUG = {
    "TF-IDF":        "tfidf",
    "Sentence Emb":  "sentence_emb",
    "Cross-Encoder": "cross_encoder",
}

BAND_EDGES  = [round(t * 0.1, 1) for t in range(11)]          # 0.0 … 1.0
BANDS       = list(zip(BAND_EDGES[:-1], BAND_EDGES[1:]))       # 10 bands
BAND_COLORS = plt.cm.tab10(np.linspace(0, 0.9, len(BANDS)))

METHOD_PAT = re.compile(
    r"^\s+(Parity|TF-IDF|BM25|Sentence Emb|Cross-Encoder)\s*:.*?best=(-?\S+)"
)
SEP_PAT = re.compile(r"^[─\-]{50,}$")
RUN_PAT = re.compile(r"^\s*RUN\s+(\d+)\s*$")

RNG = np.random.default_rng(42)


def parse_workings(path: Path) -> dict:
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


def plot_cluster(label: str, run_scores: dict) -> None:
    runs = sorted(run_scores.keys())
    if not runs:
        print(f"  [SKIP] No data for {label}")
        return

    for method in STD_METHODS:
        n_runs = len(runs)
        fig, axes = plt.subplots(n_runs, 1,
                                 figsize=(14, 4 * n_runs),
                                 squeeze=False)

        for ax, run_id in zip(axes[:, 0], runs):
            scores = np.array(run_scores[run_id][method], dtype=float)
            n      = len(scores)
            if n == 0:
                ax.set_visible(False)
                continue

            x     = np.arange(n)
            alpha = max(0.15, min(0.6, 1500 / n))

            # Dots coloured by threshold band
            for i, (lo, hi) in enumerate(BANDS):
                mask = (scores >= lo) & (scores <= hi) if hi == 1.0 \
                       else (scores >= lo) & (scores < hi)
                if mask.any():
                    ax.scatter(x[mask], scores[mask],
                               color=BAND_COLORS[i], alpha=alpha,
                               s=6, linewidths=0)
            for t in BAND_EDGES[1:]:
                ax.axhline(t, color="gray", linewidth=0.7,
                           linestyle="--", alpha=0.5)
                ax.text(n * 1.002, t, f"{t:.1f}",
                        va="center", fontsize=7, color="gray")

            ax.set_xlim(0, n)
            ax.set_ylim(-0.03, 1.08)
            ax.set_yticks(BAND_EDGES)
            ax.set_yticklabels([f"{e:.1f}" for e in BAND_EDGES], fontsize=8)
            ax.set_xlabel("Row index", fontsize=9)
            ax.set_ylabel("Score", fontsize=9)
            ax.set_title(f"Run {run_id}  ({n} rows)", fontsize=10)
            ax.grid(True, axis="x", alpha=0.12)

        patches = [
            mpatches.Patch(color=BAND_COLORS[i],
                           label=f"{BANDS[i][0]:.1f}-{BANDS[i][1]:.1f}")
            for i in range(len(BANDS))
        ]
        fig.legend(handles=patches, title="Score band", fontsize=7,
                   title_fontsize=8, loc="lower center", ncol=5,
                   bbox_to_anchor=(0.5, -0.02))

        fig.suptitle(
            f"Subtask 1.2 - Row Scores  |  {method}\n{label}",
            fontsize=12, y=1.01,
        )
        plt.tight_layout()

        out_path = OUT_DIR / f"{label}_{METHOD_SLUG[method]}_cluster.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path.relative_to(TASK_DIR)}")

workings_files = sorted(
    p for p in EVAL_DIR.rglob("*_workings.txt")
    if "old_eval" not in p.parts
)
if not workings_files:
    sys.exit(f"[ERROR] No workings files found under {EVAL_DIR}")

print(f"Found {len(workings_files)} workings file(s):\n")
for p in workings_files:
    print(f"  {p.relative_to(TASK_DIR)}")
print()

for wf in workings_files:
    label      = wf.stem.replace("_workings", "")
    run_scores = parse_workings(wf)
    print(f"Processing: {label}  ({sum(len(v[STD_METHODS[0]]) for v in run_scores.values())} rows)")
    plot_cluster(label, run_scores)

print("\nAll done.")
