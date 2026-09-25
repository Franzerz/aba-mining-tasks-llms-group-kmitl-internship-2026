#### PATCHARAKORN ####

from __future__ import annotations

import csv as csv_mod
import random
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .config import ModelConfig
from .llm import LLMClient
from .prompts import load_prompt, render_prompt

ID_CANDIDATES = ["ID", "Id", "id"]

# Per-category column layout. The 4 polarity categories don't share one schema — even within
# the same Facility workbook, Contrary(P)Body(N)/Contrary(N)Body(P) use "Original A (or
# Assumption)" / "B (or Proposition)" / "Vote", while Contrary(P)Body(P) reuses the header
# "Assumption" twice (pandas dedupes the second occurrence to "Assumption.1") and
# Contrary(N)Body(N) uses "Assumption" / "Proposition", both with a "Contrary?" vote column
# instead of "Vote". Phrase A must always resolve to the contrary-form column, never the bare
# body-form one (see Task3_Implementation_Report.md for the bug this fixes).
CATEGORY_COLUMN_SPECS: dict[str, dict[str, list[str]]] = {
    "Contrary(P)Body(N)": {
        "phrase_a": ["Original Assumption", "Original A"],
        "phrase_b": ["Proposition", "B"],
        "vote": ["Vote", "Contrary?"],
    },
    "Contrary(N)Body(P)": {
        "phrase_a": ["Original Assumption", "Original A"],
        "phrase_b": ["Proposition", "B"],
        "vote": ["Vote", "Contrary?"],
    },
    "Contrary(P)Body(P)": {
        "phrase_a": ["Assumption"],
        "phrase_b": ["Assumption.1"],
        "vote": ["Contrary?", "Vote"],
    },
    "Contrary(N)Body(N)": {
        "phrase_a": ["Assumption"],
        "phrase_b": ["Proposition"],
        "vote": ["Contrary?", "Vote"],
    },
}

# Output format mirrors mu_work's Task3.py (csv/ + log/ + wide "_sheet.xlsx" per category,
# "_ALL_sheets.xlsx" per aspect).
#
# All 4 polarity categories are candidates to look for in a source file, but whether a given
# category is actually usable depends on the ASPECT too — the same category name can be
# complete for one aspect and unvoted for another (e.g. Contrary(P)Body(N) is fully voted for
# check-in/check-out/price/staff, but Facility's copy of that same sheet has an empty Vote
# column — only 1 of the expected annotators has gone through it). So usability is tracked
# per (aspect, category) pair below, not per category name alone.
ALL_CATEGORIES = ["Contrary(P)Body(N)", "Contrary(N)Body(P)", "Contrary(P)Body(P)", "Contrary(N)Body(N)"]

# (aspect, category) pairs known to exist in a source file but NOT usable yet because voting
# isn't finished. Remove an entry once its Vote/Contrary? column is fully populated (see
# Task3_Implementation_Report.md).
INCOMPLETE_ASPECT_CATEGORIES: set[tuple[str, str]] = {
    # facility Contrary(P)Body(N) is now fully voted as of the "(Gold)" workbook (12,060/12,060
    # Vote populated, confirmed 2026-09-04) -- no longer listed here.
    ("facility", "Contrary(N)Body(P)"),  # still 0/12060 Vote populated, not started
}

# Default cap on instances per (aspect, category) group when a full run isn't practical
# (e.g. Facility's Contrary(P)Body(P) alone is 32,400 pairs). None/0 = no cap.
DEFAULT_MAX_PER_GROUP = 500


@dataclass(frozen=True)
class Task3Instance:
    row_id: str
    aspect: str
    category: str
    phrase_a: str
    phrase_b: str
    gold_vote: str  # "Yes" / "No" / "" if unavailable


def normalize_yes_no(raw: str | None) -> str:
    """Extract the model's Yes/No verdict.

    Plain zero/one-shot prompts return a single word, so first-match works fine.
    Chain-of-thought / recipe-style prompts may use the words "yes"/"no" inside the
    reasoning before the actual verdict, so: prefer an explicit "Answer: Yes/No" tag
    if present, otherwise fall back to the LAST standalone yes/no token (the
    conclusion normally comes after the reasoning, not before it).
    """
    value = (raw or "").strip()
    if not value:
        return ""

    tagged = re.search(r"answer\s*[:\-]?\s*\**\s*(yes|no)\b", value, flags=re.IGNORECASE)
    if tagged:
        return "Yes" if tagged.group(1).lower() == "yes" else "No"

    matches = re.findall(r"\b(yes|no)\b", value, flags=re.IGNORECASE)
    if not matches:
        return ""
    return "Yes" if matches[-1].lower() == "yes" else "No"


def _detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    mapping = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in mapping:
            return mapping[cand.lower()]
    return None


def aspect_from_filename(path: Path) -> str:
    lower = path.name.lower()
    if "check-out" in lower or "checkout" in lower:
        return "check-out"
    if "check-in" in lower or "checkin" in lower:
        return "check-in"
    if "price" in lower:
        return "price"
    if "staff" in lower:
        return "staff"
    if "facility" in lower:
        return "facility"
    return path.stem[:40]


def safe_file_token(value: str) -> str:
    text = re.sub(r"[^\w\-]+", "_", value.strip(), flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "sheet"


def _extract_rows(
    df: pd.DataFrame,
    category: str,
    aspect: str,
    csv_path_name: str,
) -> list[Task3Instance]:
    spec = CATEGORY_COLUMN_SPECS[category]
    id_col = _detect_column(df, ID_CANDIDATES)
    phrase_a_col = _detect_column(df, spec["phrase_a"])
    phrase_b_col = _detect_column(df, spec["phrase_b"])
    vote_col = _detect_column(df, spec["vote"])

    if not id_col or not phrase_a_col or not phrase_b_col:
        raise ValueError(
            f"Could not find required columns for category '{category}' in "
            f"{csv_path_name}. Found: {list(df.columns)}"
        )

    work = df[[id_col, phrase_a_col, phrase_b_col] + ([vote_col] if vote_col else [])].copy()
    rename_map = {id_col: "ID", phrase_a_col: "PhraseA", phrase_b_col: "PhraseB"}
    if vote_col:
        rename_map[vote_col] = "Vote"
    work = work.rename(columns=rename_map)
    if "Vote" not in work.columns:
        work["Vote"] = ""

    work = work.dropna(subset=["ID", "PhraseA", "PhraseB"])
    work["PhraseA"] = work["PhraseA"].astype(str).str.strip()
    work["PhraseB"] = work["PhraseB"].astype(str).str.strip()
    work = work[(work["PhraseA"] != "") & (work["PhraseB"] != "")]
    work = work.drop_duplicates(subset=["ID", "PhraseA", "PhraseB"])

    return [
        Task3Instance(
            row_id=str(row["ID"]).strip(),
            aspect=aspect,
            category=category,
            phrase_a=row["PhraseA"],
            phrase_b=row["PhraseB"],
            gold_vote=normalize_yes_no(row.get("Vote", "")),
        )
        for _, row in work.iterrows()
    ]


def load_task3_instances(
    task3_dir: Path,
    aspects: list[str] | None = None,
    categories: list[str] | None = None,
    max_per_group: int | None = DEFAULT_MAX_PER_GROUP,
    seed: int = 42,
) -> list[Task3Instance]:
    """Load Phrase A / Phrase B pairs + gold Vote from the Task 3 gold data.

    Supports two source formats found in Dataset/Task3/:
    - CSV: one file per aspect (check-in, check-out, price, staff), each the single
      "Contrary(P)Body(N)" sheet exported from the original Silver workbooks.
    - XLSX: a full Silver workbook (e.g. facility) with up to 4 category sheets
      (Contrary(P)Body(N), Contrary(N)Body(P), Contrary(P)Body(P), Contrary(N)Body(N)).

    Any (aspect, category) pair listed in INCOMPLETE_ASPECT_CATEGORIES is skipped regardless
    of `categories` — those sheets exist but don't have finished human votes yet.

    `max_per_group` randomly samples down any (aspect, category) group larger than this
    (fixed `seed` for reproducibility) — some categories run into the tens of thousands of
    pairs, impractical to run an LLM sweep against in full. Pass None/0 for no cap.
    """
    wanted_categories = categories if categories is not None else ALL_CATEGORIES
    instances: list[Task3Instance] = []

    for csv_path in sorted(task3_dir.glob("*.csv")):
        aspect = aspect_from_filename(csv_path)
        if aspects and aspect not in aspects:
            continue
        if "Contrary(P)Body(N)" not in wanted_categories:
            continue
        if (aspect, "Contrary(P)Body(N)") in INCOMPLETE_ASPECT_CATEGORIES:
            continue

        df = pd.read_csv(csv_path, dtype=str)
        instances.extend(_extract_rows(df, "Contrary(P)Body(N)", aspect, csv_path.name))

    # Multiple xlsx workbooks can match the same aspect (e.g. a "(Gold)" workbook superseding
    # an older "(Silver)" one once voting finishes) -- load exactly ONE per aspect, never both,
    # or every category gets double-counted. Prefer "(Gold)" > "(Silver)" > whatever's left,
    # by filename.
    xlsx_by_aspect: dict[str, list[Path]] = {}
    for xlsx_path in sorted(task3_dir.glob("*.xlsx")):
        aspect = aspect_from_filename(xlsx_path)
        if aspects and aspect not in aspects:
            continue
        xlsx_by_aspect.setdefault(aspect, []).append(xlsx_path)

    def _xlsx_priority(path: Path) -> int:
        name_lower = path.name.lower()
        if "(gold)" in name_lower:
            return 0
        if "(silver)" in name_lower:
            return 1
        return 2

    for aspect, candidates in xlsx_by_aspect.items():
        candidates.sort(key=_xlsx_priority)
        xlsx_path = candidates[0]
        if len(candidates) > 1:
            print(f"[task3] Multiple xlsx files match aspect '{aspect}': "
                  f"{[c.name for c in candidates]} -- using '{xlsx_path.name}' only.")

        xl = pd.ExcelFile(xlsx_path)
        for category in wanted_categories:
            if category not in xl.sheet_names:
                continue
            if (aspect, category) in INCOMPLETE_ASPECT_CATEGORIES:
                continue
            df = pd.read_excel(xl, sheet_name=category, dtype=str)
            instances.extend(_extract_rows(df, category, aspect, f"{xlsx_path.name}::{category}"))

    if max_per_group:
        rng = random.Random(seed)
        grouped: dict[tuple[str, str], list[Task3Instance]] = {}
        for inst in instances:
            grouped.setdefault((inst.aspect, inst.category), []).append(inst)

        capped: list[Task3Instance] = []
        for (aspect, category), group in grouped.items():
            if len(group) > max_per_group:
                group = rng.sample(group, max_per_group)
            capped.extend(group)
        instances = capped

    return instances


def _load_done_keys(run_csv: Path) -> set[str]:
    """Resume support: read (ID, PhraseA, PhraseB) already present in a run's csv/ file."""
    if not run_csv.exists():
        return set()
    try:
        old = pd.read_csv(run_csv, dtype=str).fillna("")
        if not {"ID", "PhraseA", "PhraseB"}.issubset(old.columns):
            return set()
        return {
            f"{rid}||{a}||{b}"
            for rid, a, b in zip(old["ID"], old["PhraseA"], old["PhraseB"])
        }
    except Exception:
        return set()


def _build_master_wide(df_input: pd.DataFrame, csv_dir: Path, run_base: str, n_runs: int) -> pd.DataFrame:
    """Merge every run's csv/ file into one wide table: ID, Prompt, PhraseA, PhraseB, Test 1..Test N."""
    wide = pd.DataFrame({
        "ID": df_input["ID"].astype(str),
        "Prompt": df_input["Prompt"].astype(str),
        "PhraseA": df_input["PhraseA"].astype(str),
        "PhraseB": df_input["PhraseB"].astype(str),
    })

    for run in range(1, n_runs + 1):
        run_csv = csv_dir / f"{run_base}_run{run}.csv"
        test_col = f"Test {run}"
        if not run_csv.exists():
            wide[test_col] = ""
            continue
        rdf = pd.read_csv(run_csv, dtype=str).fillna("")
        for c in ["ID", "PhraseA", "PhraseB", test_col]:
            if c not in rdf.columns:
                rdf[c] = ""
        rdf = rdf[["ID", "PhraseA", "PhraseB", test_col]]
        wide = wide.merge(rdf, on=["ID", "PhraseA", "PhraseB"], how="left")

    cols = ["ID", "Prompt", "PhraseA", "PhraseB"] + [f"Test {i}" for i in range(1, n_runs + 1)]
    for c in cols:
        if c not in wide.columns:
            wide[c] = ""
    wide = wide[cols].drop_duplicates(subset=["ID", "PhraseA", "PhraseB"], keep="last")
    return wide


def run_task3(
    *,
    client: LLMClient,
    model_cfg: ModelConfig,
    repo_root: Path,
    instances: list[Task3Instance],
    prompt_path: str,
    label: str,
    run_idx: int,
    n_runs: int,
    output_dir: Path,
    max_output_tokens: int = 300,
    resume: bool = True,
) -> dict[str, Path]:
    """Run Task 3 Yes/No contrary classification for one (prompt version, run) pair.

    Output layout mirrors mu_work's Task3.py, per aspect and per category:
      {output_dir}/{aspect}/{category}/csv/task3_{label}_{aspect}_{category}_run{N}.csv
          columns: ID, Prompt, PhraseA, PhraseB, Test {N}
      {output_dir}/{aspect}/{category}/log/task3_{label}_{aspect}_{category}_run{N}.csv
          columns: Timestamp, Run, ID, PhraseA, PhraseB, Prompt, RawOutput
      {output_dir}/{aspect}/{category}/task3_{label}_{aspect}_{category}_sheet.xlsx
          wide table merging every run's Test column so far: ID, Prompt, PhraseA, PhraseB, Test 1..Test n_runs
      {output_dir}/{aspect}/task3_{label}_{aspect}_ALL_sheets.xlsx
          one worksheet per category actually present for that aspect (varies — see
          INCOMPLETE_ASPECT_CATEGORIES; check-in/check-out/price/staff only have
          Contrary(P)Body(N), facility also has Contrary(P)Body(P) and Contrary(N)Body(N))

    Checkpointing: each prediction is written to run_csv/run_log the moment it completes
    (thread-safe, one row at a time), not buffered until the whole (aspect, category) batch
    finishes. If the process is killed mid-batch — e.g. a remote server closing partway through
    a large category — everything already completed stays on disk; re-running the exact same
    command later (resume=True, the default) picks up only the remaining un-processed pairs.
    """
    template = load_prompt(repo_root, prompt_path)
    write_lock = threading.Lock()

    def _predict_one(prompt: str) -> dict[str, Any]:
        resp = client.complete(
            model=model_cfg.task1_model,
            prompt=prompt,
            temperature=model_cfg.temperature,
            top_p=model_cfg.top_p,
            max_output_tokens=max_output_tokens,
        )
        prediction = normalize_yes_no(resp.text)
        return {"prediction": prediction, "raw_output": resp.text}

    def _append_row(path: Path, cols: list[str], row_dict: dict[str, Any]) -> None:
        """Appends one CSV row immediately, writing the header first if the file is new.
        Locked so concurrent worker threads can't interleave writes to the same file."""
        with write_lock:
            is_new = not path.exists()
            with path.open("a", newline="", encoding="utf-8") as f:
                writer = csv_mod.writer(f)
                if is_new:
                    writer.writerow(cols)
                writer.writerow([row_dict[c] for c in cols])

    from concurrent.futures import ThreadPoolExecutor, as_completed

    by_aspect_category: dict[tuple[str, str], list[Task3Instance]] = {}
    for inst in instances:
        by_aspect_category.setdefault((inst.aspect, inst.category), []).append(inst)

    aspects = sorted({inst.aspect for inst in instances})
    written: dict[str, Path] = {}

    for aspect in aspects:
        aspect_dir = output_dir / aspect
        aspect_all_sheets = aspect_dir / f"task3_{label}_{aspect}_ALL_sheets.xlsx"
        category_wide: dict[str, pd.DataFrame] = {}

        categories_for_aspect = sorted(c for (a, c) in by_aspect_category if a == aspect)
        for category in categories_for_aspect:
            aspect_instances = by_aspect_category[(aspect, category)]
            category_token = safe_file_token(category)
            sheet_dir = aspect_dir / category_token
            csv_dir = sheet_dir / "csv"
            log_dir = sheet_dir / "log"
            csv_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)

            run_base = f"task3_{label}_{aspect}_{category_token}"
            run_csv = csv_dir / f"{run_base}_run{run_idx}.csv"
            run_log = log_dir / f"{run_base}_run{run_idx}.csv"
            test_col = f"Test {run_idx}"
            run_csv_cols = ["ID", "Prompt", "PhraseA", "PhraseB", test_col]
            log_cols = ["Timestamp", "Run", "ID", "PhraseA", "PhraseB", "Prompt", "RawOutput"]

            df_input = pd.DataFrame([
                {
                    "ID": inst.row_id,
                    "PhraseA": inst.phrase_a,
                    "PhraseB": inst.phrase_b,
                    "Prompt": render_prompt(template, PHRASE_A=inst.phrase_a, PHRASE_B=inst.phrase_b),
                }
                for inst in aspect_instances
            ])

            if not resume:
                # "ignore existing output and re-run everything" — start this run's files clean
                # rather than merging with old data.
                run_csv.unlink(missing_ok=True)
                run_log.unlink(missing_ok=True)

            done_keys = _load_done_keys(run_csv) if resume else set()
            todo_mask = ~(df_input["ID"] + "||" + df_input["PhraseA"] + "||" + df_input["PhraseB"]).isin(done_keys)
            todo = df_input[todo_mask]

            if len(todo):
                with ThreadPoolExecutor(max_workers=model_cfg.num_workers) as ex:
                    futs = {ex.submit(_predict_one, row["Prompt"]): row for _, row in todo.iterrows()}

                    for fut in tqdm(as_completed(futs), total=len(futs),
                                     desc=f"Task3[{aspect}][{category}][{label} run{run_idx}]"):
                        row = futs[fut]
                        result = fut.result()
                        # Checkpoint: written to disk the instant it completes, not batched.
                        _append_row(run_csv, run_csv_cols, {
                            "ID": row["ID"], "Prompt": row["Prompt"],
                            "PhraseA": row["PhraseA"], "PhraseB": row["PhraseB"],
                            test_col: result["prediction"],
                        })
                        _append_row(run_log, log_cols, {
                            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Run": run_idx, "ID": row["ID"],
                            "PhraseA": row["PhraseA"], "PhraseB": row["PhraseB"],
                            "Prompt": row["Prompt"], "RawOutput": result["raw_output"],
                        })

            if run_csv.exists():
                # Tidy pass: dedupe defensively and reorder to match df_input's gold-file order —
                # incremental writes land in whatever order the LLM responded, not gold order
                # (same as the _sheet.xlsx wide table, which is built from df_input directly).
                on_disk = pd.read_csv(run_csv, dtype=str).fillna("")
                on_disk = on_disk.drop_duplicates(subset=["ID", "PhraseA", "PhraseB"], keep="last")
                order = {
                    key: i for i, key in enumerate(
                        df_input["ID"] + "||" + df_input["PhraseA"] + "||" + df_input["PhraseB"]
                    )
                }
                on_disk["_order"] = (
                    on_disk["ID"] + "||" + on_disk["PhraseA"] + "||" + on_disk["PhraseB"]
                ).map(order)
                on_disk = on_disk.sort_values("_order").drop(columns="_order").reset_index(drop=True)
                on_disk.to_csv(run_csv, index=False)

            wide = _build_master_wide(df_input, csv_dir, run_base, n_runs)
            sheet_xlsx = sheet_dir / f"{run_base}_sheet.xlsx"
            wide.to_excel(sheet_xlsx, index=False)
            category_wide[category] = wide

        with pd.ExcelWriter(aspect_all_sheets, engine="openpyxl") as writer:
            for category, wide in category_wide.items():
                wide.to_excel(writer, sheet_name=category[:31], index=False)
        written[aspect] = aspect_all_sheets

    return written
