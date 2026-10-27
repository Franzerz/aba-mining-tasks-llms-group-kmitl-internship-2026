#### PATCHARAKORN ####

"""
Builds the FULL, unvoted Task 3 pair sets an LLM is run over, so every topic can get contraries
even where no human has voted yet. Nothing here is gold data — the predictions made on these
pairs are LLM output and must be presented as such.

Per topic, one workbook `Dataset/Task3/llm_pairs/<topic>.xlsx` with up to two sheets, laid out
like the voted Facility workbook so `src.task3.load_task3_instances` reads them unchanged:

  Contrary(P)Body(N)   ID | Original A          | A (positive literal) | B (negative literal) | Vote
                       "is the negative literal B evidence against no_evident_not_A?"
                       Written only for topics WITHOUT human votes for this category.
  Contrary(N)Body(P)   ID | Original Assumption | A (negative literal) | B (positive literal) | Vote
                       "is the positive literal B evidence against have_evident_A?"
                       Written for every topic: no topic has votes for it yet.

Contrary(P)Body(P) / Contrary(N)Body(N) are not built: they pair an assumption with a literal of
the SAME polarity, which is the mirror of the two above and not needed for the framework.

Where the literals come from — the same place the voted topics' literals came from:
  - topics with a voted Contrary(P)Body(N) set: the A / B literals in that set, so the new
    Contrary(N)Body(P) pairs line up atom-for-atom with the framework already built from it;
  - every other topic: the human-annotated Body literals of the Task 1/2 gold dataset —
    `gold_csv` in configs/paths.yaml, "Original ABA Dataset for Version 2 [Oct 23, 2025]" —
    split by the span's sentiment.
Assumption names follow the annotation convention (CLAR 2025, Sect. 4.2), which the gold data
follows for 99.5% of its literals: positive -> "no_evident_not_" + literal,
negative -> "have_evident_" + literal.

Pair IDs are STABLE across rebuilds: a pair (assumption, B) that already exists in the previous
build keeps its ID, and only new pairs get new ones (after the highest ID used). run_task3.py
resumes by (ID, PhraseA, PhraseB), so after the literal sets change only the new pairs are sent
to the model; predictions for pairs that no longer exist are simply never read again.

Usage:
  python build_task3_llm_pairs.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src import load_paths_config, load_task3_instances  # noqa: E402

DEFAULT_OUT_DIR = REPO_ROOT / "Dataset" / "Task3" / "llm_pairs"

PN, NP = "Contrary(P)Body(N)", "Contrary(N)Body(P)"
POS_PREFIX, NEG_PREFIX = "no_evident_not_", "have_evident_"


def topic_key(topic: str) -> str:
    """"Check-in" -> "check-in", "Taxi-issue" -> "taxi-issue" — the aspect names the rest of
    the pipeline uses (and what src.task3.aspect_from_filename recovers from the file name)."""
    return re.sub(r"[^\w\-]+", "_", topic.strip()).lower()


def category_token(category: str) -> str:
    """"Contrary(P)Body(N)" -> "Contrary_P_Body_N", the token run_task3.py uses in its paths."""
    return re.sub(r"[^\w\-]+", "_", category).strip("_")


def literals_from_spans(csv_path: Path) -> dict[str, dict[str, set[str]]]:
    """topic -> {"Positive": {...}, "Negative": {...}} from the gold spans' Body columns."""
    df = pd.read_csv(csv_path, dtype=str)
    df["Topic"] = df["Topic"].fillna("").str.strip()
    df["Pos/Neg"] = df["Pos/Neg"].fillna("").str.strip()
    df = df[(df["Topic"] != "") & (df["Topic"].str.lower() != "off")
            & df["Pos/Neg"].isin(["Positive", "Negative"])]
    body_cols = [c for c in df.columns if re.fullmatch(r"Body \d+", str(c))]

    out: dict[str, dict[str, set[str]]] = {}
    for (topic, sentiment), group in df.groupby(["Topic", "Pos/Neg"]):
        values = out.setdefault(topic_key(topic), {"Positive": set(), "Negative": set()})[sentiment]
        for col in body_cols:
            values |= {v.strip() for v in group[col].dropna() if v.strip()}
    return out


def literals_from_voted(task3_dir: Path) -> dict[str, dict[str, set[str]]]:
    """topic -> literal sets, for topics whose Contrary(P)Body(N) set is already voted."""
    out: dict[str, dict[str, set[str]]] = {}
    for inst in load_task3_instances(task3_dir, categories=[PN], max_per_group=None):
        if not inst.phrase_a.startswith(POS_PREFIX):
            raise ValueError(f"{inst.aspect}: unexpected assumption name {inst.phrase_a!r}")
        sets = out.setdefault(inst.aspect, {"Positive": set(), "Negative": set()})
        sets["Positive"].add(inst.phrase_a[len(POS_PREFIX):])
        sets["Negative"].add(inst.phrase_b)
    return out


def category_ids(previous: Path) -> dict[tuple[str, str], int]:
    """(assumption, B) -> ID from the previous build of one sheet, if there is one."""
    if not previous.exists():
        return {}
    old = pd.read_csv(previous, dtype=str, keep_default_na=False)
    asm_col = "Original A" if "Original A" in old.columns else "Original Assumption"
    return {(a, b): int(i) for i, a, b in zip(old["ID"], old[asm_col], old["B"])}


def with_stable_ids(rows: list[tuple[str, str, str]], asm_col: str, previous: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=[asm_col, "A", "B"])
    known = category_ids(previous)
    next_id = max(known.values(), default=0) + 1
    ids = []
    for a, b in zip(df[asm_col], df["B"]):
        if (a, b) in known:
            ids.append(known[(a, b)])
        else:
            ids.append(next_id)
            next_id += 1
    df.insert(0, "ID", ids)
    df["Vote"] = ""
    return df


def pn_sheet(pos: set[str], neg: set[str], previous: Path) -> pd.DataFrame:
    rows = [(POS_PREFIX + a, a, b) for a in sorted(pos) for b in sorted(neg)]
    return with_stable_ids(rows, "Original A", previous)


def np_sheet(pos: set[str], neg: set[str], previous: Path) -> pd.DataFrame:
    rows = [(NEG_PREFIX + a, a, b) for a in sorted(neg) for b in sorted(pos)]
    return with_stable_ids(rows, "Original Assumption", previous)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    paths = load_paths_config(REPO_ROOT)
    voted = literals_from_voted(paths.task3_dir)
    spans = literals_from_spans(paths.gold_csv)
    print(f"Literals from: {paths.gold_csv.name}\n")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'topic':<14} {'source':<14} {'pos':>5} {'neg':>5} {PN + ' pairs':>26} {NP + ' pairs':>26}")
    total_pn = total_np = 0
    for topic in sorted(set(voted) | set(spans)):
        has_votes = topic in voted
        sets = voted[topic] if has_votes else spans[topic]
        pos, neg = sets["Positive"], sets["Negative"]

        previous = lambda cat: args.out_dir / "csv" / f"{topic}__{category_token(cat)}.csv"  # noqa: E731
        sheets: dict[str, pd.DataFrame] = {}
        if not has_votes and pos and neg:
            sheets[PN] = pn_sheet(pos, neg, previous(PN))
        if pos and neg:
            sheets[NP] = np_sheet(pos, neg, previous(NP))

        n_pn = len(sheets.get(PN, ()))
        n_np = len(sheets.get(NP, ()))
        # pairs the previous build did not have (kept pairs keep their IDs)
        new = {cat: int((~df["ID"].isin(set(category_ids(previous(cat)).values()))).sum())
               for cat, df in sheets.items()}
        pn_note = "human-voted" if has_votes else f"{n_pn:,}"
        print(f"{topic:<14} {'voted set' if has_votes else 'gold spans':<14} {len(pos):>5} {len(neg):>5} "
              f"{pn_note:>26} {n_np:>26,}   new pairs: {new}")
        if not sheets:
            continue
        total_pn += n_pn
        total_np += n_np
        with pd.ExcelWriter(args.out_dir / f"{topic}.xlsx", engine="openpyxl") as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name, index=False)
        # Same sheets as plain CSV for readers that can't afford openpyxl (room's workbook takes
        # ~10 s to parse; the web app reads these). A subfolder, so the Task 3 loader's
        # top-level *.csv glob never mistakes them for voted sets.
        (args.out_dir / "csv").mkdir(exist_ok=True)
        for name, df in sheets.items():
            df.to_csv(args.out_dir / "csv" / f"{topic}__{category_token(name)}.csv", index=False)

    shown = args.out_dir.relative_to(REPO_ROOT) if args.out_dir.is_relative_to(REPO_ROOT) else args.out_dir
    print(f"\nIn the sets: {total_pn:,} {PN} pairs, {total_np:,} {NP} pairs -> {shown}/")


if __name__ == "__main__":
    main()
