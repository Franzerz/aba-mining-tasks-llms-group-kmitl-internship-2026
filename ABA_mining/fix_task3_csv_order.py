#### PATCHARAKORN ####

"""
One-off cleanup: re-sorts every already-generated outputs/task3/**/csv/*.csv file by
numeric ID. These files were written in ThreadPoolExecutor completion order (see the fix
in src/task3.py), so existing files from before that fix are scrambled. The wide
*_sheet.xlsx files were never affected (built from the original gold-file order directly)
and are left untouched. The log/*.csv files are intentionally chronological (a timestamped
audit trail, matching mu_work's design) and are also left untouched.

Usage: python fix_task3_csv_order.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
TASK3_OUT = REPO_ROOT / "outputs" / "task3"


def main() -> None:
    csv_files = sorted(TASK3_OUT.rglob("csv/*.csv"))
    if not csv_files:
        print(f"No files found under {TASK3_OUT}/**/csv/")
        return

    fixed = 0
    already_ok = 0
    for path in csv_files:
        df = pd.read_csv(path, dtype=str).fillna("")
        if "ID" not in df.columns:
            print(f"  [SKIP] no ID column: {path.relative_to(REPO_ROOT)}")
            continue

        ids_numeric = pd.to_numeric(df["ID"], errors="coerce")
        was_sorted = ids_numeric.is_monotonic_increasing

        if was_sorted:
            already_ok += 1
            continue

        df = df.assign(_sort=ids_numeric).sort_values("_sort").drop(columns="_sort").reset_index(drop=True)
        df.to_csv(path, index=False)
        fixed += 1
        print(f"  [FIXED] {path.relative_to(REPO_ROOT)}  (now starts at ID {df['ID'].iloc[0]})")

    print(f"\nDone. Fixed {fixed} file(s), {already_ok} were already in order "
          f"(out of {len(csv_files)} total).")


if __name__ == "__main__":
    main()
