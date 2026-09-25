#### PATCHARAKORN ####

"""
Generates UNLABELED Task 3 candidate phrase pairs for topics that don't yet have a
human-voted "Silver" gold set (everything except Check-in, Check-out, Price, Staff,
which already have real annotated data under Dataset/Task3/).

This does NOT produce gold data or run any LLM. It only proposes candidate pairs for a
human to vote on, using the same "Contrary(P)Body(N)" definition the existing Silver
sheets use:
  - Phrase A = a contrary literal auto-derived from a POSITIVE-sentiment Task 2 body
    literal (Literal Type == "cont", Sentiment == "Positive")
  - Phrase B = a NEGATIVE-sentiment Task 2 body literal (Literal Type == "body",
    Sentiment == "Negative")
  for the same topic.

Caveat: the source Literal strings come from a Task 2 LLM run (source="gt" means the
LLM was given ground-truth topic/sentiment/span, not that the literal itself was human
-written). "Valid" only means the output passed JSON-schema validation, not that a human
verified the literal's wording — so these candidates should be read as proposals, same as
what an annotator would have reviewed when the original Silver sheets were built.

Usage:
  python build_task3_candidates.py
  python build_task3_candidates.py --max-pairs-per-topic 500
  python build_task3_candidates.py --task2-csv outputs/task2/gt/llama3.2/version1/task2_llama3.2_llama3.2_generator_v1_run1_gt_n2078.csv
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent

# These topics already have a human-voted Silver gold set under Dataset/Task3/ — skip them.
TOPICS_WITH_GOLD = {"Check-in", "Check-out", "Price", "Staff"}

DEFAULT_TASK2_CSV = (
    REPO_ROOT / "outputs" / "task2" / "gt" / "llama3.2" / "version1"
    / "task2_llama3.2_llama3.2_generator_v1_run1_gt_n2078.csv"
)
DEFAULT_OUT_DIR = REPO_ROOT / "Dataset" / "Task3" / "candidates"


def _safe_name(topic: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", topic.strip(), flags=re.UNICODE)
    return re.sub(r"_+", "_", s).strip("_").lower()


def build_candidates(
    task2_csv: Path,
    out_dir: Path,
    max_pairs_per_topic: int | None,
    seed: int,
) -> None:
    if not task2_csv.exists():
        raise FileNotFoundError(
            f"Task 2 output not found: {task2_csv}\n"
            f"Run run_task2.py first, or pass --task2-csv to point at an existing output."
        )

    df = pd.read_csv(task2_csv)
    df = df[df["Valid"] == True].copy()  # noqa: E712

    positive_cont = df[(df["Sentiment"] == "Positive") & (df["Literal Type"] == "cont")]
    negative_body = df[(df["Sentiment"] == "Negative") & (df["Literal Type"] == "body")]

    all_topics = sorted(set(df["Topic"].dropna()) - {"Off"})
    topics = [t for t in all_topics if t not in TOPICS_WITH_GOLD]
    skipped = [t for t in all_topics if t in TOPICS_WITH_GOLD]

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    print(f"Source: {task2_csv.relative_to(REPO_ROOT)}")
    print(f"Skipping (already have gold data): {', '.join(skipped)}\n")
    print(f"{'Topic':<15} {'A pool':>7} {'B pool':>7} {'Full A*B':>9} {'Written':>8}  File")
    print("-" * 80)

    for topic in topics:
        a_pool = sorted(set(positive_cont[positive_cont["Topic"] == topic]["Literal"].dropna()))
        b_pool = sorted(set(negative_body[negative_body["Topic"] == topic]["Literal"].dropna()))

        pairs = [(a, b) for a in a_pool for b in b_pool]
        full_count = len(pairs)

        if not pairs:
            print(f"{topic:<15} {len(a_pool):>7} {len(b_pool):>7} {full_count:>9} {'0':>8}  "
                  f"(skipped: {'no positive contrary literals' if not a_pool else 'no negative body literals'})")
            continue

        if max_pairs_per_topic and full_count > max_pairs_per_topic:
            pairs = rng.sample(pairs, max_pairs_per_topic)
        pairs.sort()

        rows = [
            {"ID": i + 1, "Topic": topic, "Original A": a, "B": b, "Vote": ""}
            for i, (a, b) in enumerate(pairs)
        ]
        out_path = out_dir / f"{_safe_name(topic)}_candidates.csv"
        pd.DataFrame(rows, columns=["ID", "Topic", "Original A", "B", "Vote"]).to_csv(out_path, index=False)

        print(f"{topic:<15} {len(a_pool):>7} {len(b_pool):>7} {full_count:>9} {len(pairs):>8}  "
              f"{out_path.relative_to(REPO_ROOT)}")

    print(f"\nDone. These are UNLABELED candidate pairs under {out_dir.relative_to(REPO_ROOT)}/")
    print("A human needs to fill in the 'Vote' column (Yes/No) before these can be used as gold data.")
    print("Once voted, move/rename a file into Dataset/Task3/ (matching the existing naming style)")
    print("for run_task3.py's loader to pick it up.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task2-csv", type=Path, default=DEFAULT_TASK2_CSV,
                         help="Task 2 GT output CSV to source body/contrary literals from")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                         help="Where to write the per-topic candidate CSVs")
    parser.add_argument("--max-pairs-per-topic", type=int, default=300,
                         help="Cap on pairs per topic (random sample if the full A*B cross-product "
                              "exceeds this; 0 = no cap). Default 300 keeps sheets hand-annotatable.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default 42, for reproducibility)")
    args = parser.parse_args()

    build_candidates(
        task2_csv=args.task2_csv,
        out_dir=args.out_dir,
        max_pairs_per_topic=args.max_pairs_per_topic or None,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
