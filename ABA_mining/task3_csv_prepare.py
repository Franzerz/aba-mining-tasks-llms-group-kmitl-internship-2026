"""
Splits a Task 2 output CSV into per-topic Task 3 input CSVs, two files per topic:

  - Contrary(P)Body(N):
      Original A = contrary literal from a POSITIVE-sentiment row (Literal Type == "cont"),
                   e.g. "no_evident_not_good_check-out_time"
      A          = Original A with the "no_evident_not_" prefix cut ("good_check-out_time")
      B          = body literal from a NEGATIVE-sentiment row (Literal Type == "body")
  - Contrary(N)Body(P):
      Original A = contrary literal from a NEGATIVE-sentiment row (Literal Type == "cont"),
                   e.g. "have_evident_not_clean_room"
      A          = Original A with the "have_evident_" prefix cut ("not_clean_room")
      B          = body literal from a POSITIVE-sentiment row (Literal Type == "body")

Pairs are the full A x B cross product within a topic (unique literals only), grouped by A
like the Silver verification sheets: each A repeats once per B, and B is unique within an A block. Rows that can't fill every column (invalid row,
missing literal / type, wrong prefix, empty after cutting the prefix) are dropped.

Output columns: ID, Original A, A, B

Usage:
  python task3_csv_prepare.py
  python task3_csv_prepare.py --task2-csv outputs/task2/gt/<model>/version1/<file>.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
TASK2_DIR = REPO_ROOT / "outputs" / "task2"
DEFAULT_OUT_DIR = REPO_ROOT / "Dataset" / "Task3"

# name -> (contrary sentiment, contrary prefix to cut, body sentiment)
COMBINATIONS = {
    "Contrary(P)Body(N)": ("Positive", "no_evident_not_", "Negative"),
    "Contrary(N)Body(P)": ("Negative", "have_evident_", "Positive"),
}
COLUMNS = ["ID", "Original A", "A", "B"]


def _safe_name(topic: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", topic.strip(), flags=re.UNICODE)
    return re.sub(r"_+", "_", s).strip("_")


def _unique(values: pd.Series) -> list[str]:
    """Unique values in first-appearance order."""
    return list(dict.fromkeys(values))


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["Valid"].astype(str).str.lower() == "true"].copy()
    for col in ("Topic", "Sentiment", "Literal", "Literal Type"):
        df[col] = df[col].astype("string").str.strip()
    df = df.dropna(subset=["Topic", "Sentiment", "Literal", "Literal Type"])
    return df[(df["Literal"] != "") & (df["Literal"] != "(no literals)")]


def build_pairs(df: pd.DataFrame, cont_sent: str, prefix: str, body_sent: str) -> pd.DataFrame:
    conts = df[(df["Sentiment"] == cont_sent) & (df["Literal Type"] == "cont")]["Literal"]
    conts = conts[conts.str.startswith(prefix) & (conts.str.len() > len(prefix))]
    a_pool = [(orig, orig[len(prefix):]) for orig in _unique(conts)]

    b_pool = _unique(df[(df["Sentiment"] == body_sent) & (df["Literal Type"] == "body")]["Literal"])

    rows = [(orig, a, b) for orig, a in a_pool for b in b_pool]
    out = pd.DataFrame(rows, columns=COLUMNS[1:])
    out.insert(0, "ID", range(1, len(out) + 1))
    return out


def prepare(task2_csv: Path, out_dir: Path) -> None:
    df = _clean(pd.read_csv(task2_csv))

    print(f"Source: {task2_csv.relative_to(REPO_ROOT)}")
    print(f"Output: {out_dir.relative_to(REPO_ROOT)}\n")
    print(f"{'Topic':<15} {'Combination':<20} {'Rows':>8}  File")
    print("-" * 80)

    for topic in sorted(set(df["Topic"]) - {"Off"}):
        topic_df = df[df["Topic"] == topic]
        topic_dir = out_dir / _safe_name(topic)
        for combo, (cont_sent, prefix, body_sent) in COMBINATIONS.items():
            pairs = build_pairs(topic_df, cont_sent, prefix, body_sent)
            if pairs.empty:
                print(f"{topic:<15} {combo:<20} {0:>8}  (skipped: no complete pairs)")
                continue
            topic_dir.mkdir(parents=True, exist_ok=True)
            out_path = topic_dir / f"{_safe_name(topic)} - {combo}.csv"
            pairs.to_csv(out_path, index=False, encoding="utf-8")
            print(f"{topic:<15} {combo:<20} {len(pairs):>8}  {out_path.relative_to(REPO_ROOT)}")


def _default_task2_csv() -> Path:
    csvs = sorted(TASK2_DIR.rglob("*.csv"), key=lambda p: p.stat().st_mtime)
    if not csvs:
        raise FileNotFoundError(f"No Task 2 CSV found under {TASK2_DIR}. Pass --task2-csv.")
    return csvs[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task2-csv", type=Path, default=None,
                        help="Task 2 output CSV (default: newest CSV under outputs/task2/)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="Output folder, one subfolder per topic (default: Dataset/Task3)")
    args = parser.parse_args()

    task2_csv = (args.task2_csv or _default_task2_csv()).resolve()
    prepare(task2_csv, args.out_dir.resolve())


if __name__ == "__main__":
    main()
