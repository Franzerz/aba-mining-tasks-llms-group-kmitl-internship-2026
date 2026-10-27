"""
Adds the model's "Test 1" answer from the outputs/task3 sheet xlsx files as a "Vote"
column onto the gold Dataset/Task3 CSVs, matched by ID per topic and Contrary(N/P) variant.
Each pair is checked so the sheet's PhraseA/PhraseB equal the gold "Original A"/"B".

Output: Dataset/Task3_Vote/<same filename>.csv with columns ID,Original A,A,B,Vote

Usage: python task3_add_vote.py [--run-dir outputs/task3/qwen3.8_27b/contrary_v6]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
DATASET_DIR = REPO_ROOT / "Dataset" / "Task3"
OUT_DIR = REPO_ROOT / "Dataset" / "Task3_Vote"
DEFAULT_RUN_DIR = REPO_ROOT / "outputs" / "task3" / "qwen3.8_27b" / "contrary_v6"

# "Food - Contrary(N)Body(P).csv" -> ("food", "Contrary_N_Body_P")
NAME_RE = re.compile(r"^(?P<topic>.+?) - Contrary\((?P<c>[NP])\)Body\((?P<b>[NP])\)\.csv$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else REPO_ROOT / args.run_dir

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for gold_path in sorted(DATASET_DIR.glob("*.csv")):
        m = NAME_RE.match(gold_path.name)
        if not m:
            print(f"  [SKIP] unrecognised name: {gold_path.name}")
            continue
        topic = m["topic"].lower()
        variant = f"Contrary_{m['c']}_Body_{m['b']}"
        sheets = list((run_dir / topic / variant).glob(f"task3_*_{variant}_sheet.xlsx"))
        sheets = [p for p in sheets if not p.name.startswith("~$")]
        if len(sheets) != 1:
            print(f"  [SKIP] expected 1 sheet for {topic}/{variant}, found {len(sheets)}")
            continue

        gold = pd.read_csv(gold_path, dtype=str).fillna("")
        sheet = pd.read_excel(sheets[0], dtype=str).fillna("")
        sheet = sheet[["ID", "PhraseA", "PhraseB", "Test 1"]]

        merged = gold.merge(sheet, on="ID", how="left", validate="one_to_one")
        missing = merged["Test 1"].isna().sum()
        mismatch = merged["Test 1"].notna() & (
            (merged["PhraseA"] != merged["Original A"]) | (merged["PhraseB"] != merged["B"])
        )
        if mismatch.any():
            raise ValueError(
                f"{gold_path.name}: {mismatch.sum()} rows where sheet phrases differ from gold, "
                f"e.g. ID {merged.loc[mismatch, 'ID'].iloc[0]}"
            )

        out = merged[["ID", "Original A", "A", "B", "Test 1"]].rename(columns={"Test 1": "Vote"})
        out.to_csv(OUT_DIR / gold_path.name, index=False, encoding="utf-8")
        counts = out["Vote"].value_counts(dropna=False).to_dict()
        print(f"  {gold_path.name}: {len(out)} rows, missing={missing}, votes={counts}")

    print(f"Written to {OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
