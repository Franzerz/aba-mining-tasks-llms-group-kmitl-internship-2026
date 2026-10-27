#### PATCHARAKORN ####

"""
Task 3 evaluator, structured to match mu_work's Task3_Evaluation.py:
  - one {category}.xlsx per (model, prompt version, aspect) with one sheet per Test column
    (ID, Vote, LLM_Output, Verify) plus a SUMMARY sheet
  - one EVAL_SUMMARY.xlsx per (model, prompt version, aspect) with ALL + MICRO_BY_CATEGORY sheets
  - one top-level TASK3_VERIFY_MASTER_SUMMARY.xlsx aggregating every discovered prediction file
    (ALL_EVAL_SUMMARY + AVG_BY_MODEL_SHOT_TOPIC_CAT, picking the best-scoring Test per config)
  - two CSVs laid out like the team's Google Sheet, ready to import:
      TASK3_SHEET_PER_TOPIC.csv              one row per model/version/topic/category/test
      TASK3_SUMMARY_FOR_SHEETS[_ALL_TOPICS].csv  one row per model/version/run, macro + micro

Reads predictions from either layout run_task3.py can leave behind, under
{pred_dir}/{model}/{label}/{aspect}/{category}/:
  - csv/..._run{N}.csv   one file per run, each carrying a "Test N" column (the per-prediction
                         checkpoint files — always present, and the only thing a run that was
                         copied back from the server before its tidy pass finished will have)
  - ..._sheet.xlsx       the wide ID + Test 1..N table written after all runs finish
Per-run CSVs win when both exist, since they are the source of truth the sheet is built from.
Gold Vote comes from the Task 3 gold files via src.task3.load_task3_instances (not a separate
hardcoded GT path list, since that loader already handles this project's column-name variants).

Every prediction row is checked against its gold row: the (PhraseA, PhraseB) the model was shown
must be the pair gold has under that ID. IDs only mean something relative to one specific gold
file, so a run made against a different file version would otherwise score silently wrong.

"Shot" in the output columns holds the prompt-version label (e.g. "zero_shot", "contrary_v3"),
generalizing mu_work's 0-shot/1-shot column to this project's 8 prompt variants.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent.parent
for p in (REPO_ROOT, WORKSPACE_ROOT):
    sys.path.insert(0, str(p))

from internship.ABA_mining.src import load_paths_config
from internship.ABA_mining.src.task3 import load_task3_instances

PRED_BASE = REPO_ROOT / "outputs" / "task3"
EVAL_BASE = REPO_ROOT / "outputs" / "eval" / "task3"
MASTER_SUMMARY_XLSX = EVAL_BASE / "TASK3_VERIFY_MASTER_SUMMARY.xlsx"

# Positive label per category (mirrors mu_work's YES_POSITIVE_SHEETS / NO_POSITIVE_SHEETS).
# Only Contrary(P)Body(N) has gold data in this project; the others are placeholders for
# if/when the remaining Silver sheets become available (see Task3_Implementation_Report.md).
POSITIVE_LABEL_BY_CATEGORY = {
    "Contrary(P)Body(N)": "Yes",
    "Contrary(N)Body(P)": "Yes",
    "Contrary(P)Body(P)": "No",
    "Contrary(N)Body(N)": "No",
}

def norm_yn(x) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in {"yes", "y", "true", "1"}:
        return "Yes"
    if s in {"no", "n", "false", "0"}:
        return "No"
    return None


def classify_verify(gt: str | None, pred: str | None, positive_label: str) -> str | None:
    if gt not in {"Yes", "No"} or pred not in {"Yes", "No"}:
        return None
    if gt == positive_label and pred == positive_label:
        return "TP"
    if gt == positive_label and pred != positive_label:
        return "FN"
    if gt != positive_label and pred == positive_label:
        return "FP"
    return "TN"


def safe_div0(a: float, b: float) -> float:
    return (a / b) if b else 0.0


def safe_div_series(num: pd.Series, den: pd.Series) -> pd.Series:
    num = pd.to_numeric(num, errors="coerce").fillna(0.0).astype(float)
    den = pd.to_numeric(den, errors="coerce").fillna(0.0).astype(float)
    return pd.Series(np.where(den != 0, num / den, 0.0), index=num.index, dtype=float)


def compute_metrics(vote: pd.Series, pred: pd.Series, positive_label: str) -> dict:
    y_true = vote.map(norm_yn)
    y_pred = pred.map(norm_yn)
    ok = y_true.notna() & y_pred.notna()
    y_true, y_pred = y_true[ok], y_pred[ok]
    n = int(ok.sum())

    if n == 0:
        return {"N_valid": 0, "TP": 0, "TN": 0, "FP": 0, "FN": 0,
                "Precision": 0.0, "Recall": 0.0, "F1": 0.0, "Accuracy": 0.0}

    negative_label = "No" if positive_label == "Yes" else "Yes"
    tp = int(((y_true == positive_label) & (y_pred == positive_label)).sum())
    tn = int(((y_true == negative_label) & (y_pred == negative_label)).sum())
    fp = int(((y_true == negative_label) & (y_pred == positive_label)).sum())
    fn = int(((y_true == positive_label) & (y_pred == negative_label)).sum())

    precision = round(safe_div0(tp, tp + fp), 6)
    recall = round(safe_div0(tp, tp + fn), 6)
    f1 = round(safe_div0(2 * precision * recall, precision + recall), 6)
    accuracy = round(safe_div0(tp + tn, tp + tn + fp + fn), 6)
    return {"N_valid": n, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "Precision": precision, "Recall": recall, "F1": f1, "Accuracy": accuracy}


def _write_txt_companion(xlsx_path: Path, sheets: list[tuple[str, pd.DataFrame]]) -> None:
    """Writes a plain-text companion next to an .xlsx (same stem, .txt), one section per
    sheet — readable directly in an editor without an Excel/spreadsheet viewer extension."""
    lines: list[str] = []
    for name, df in sheets:
        lines.append("=" * 70)
        lines.append(name)
        lines.append("=" * 70)
        lines.append(df.to_string(index=False) if not df.empty else "(empty)")
        lines.append("")
    xlsx_path.with_suffix(".txt").write_text("\n".join(lines), encoding="utf-8")


_METRIC_COLS = ["Precision", "Recall", "F1", "Accuracy"]


def _round_metrics(df: pd.DataFrame, decimals: int = 6) -> pd.DataFrame:
    """Rounds whichever metric columns are present, so the .xlsx cell itself holds the same
    rounded value shown in the .txt companion — not just a display-level truncation."""
    df = df.copy()
    for c in _METRIC_COLS:
        if c in df.columns:
            df[c] = df[c].round(decimals)
    return df


_RUN_RE = re.compile(r"_run(\d+)\.csv$")
UNPARSED = "<unparsed>"


def _category_from_token(token: str) -> str:
    """Maps the filesystem-safe category token (Contrary_P_Body_N) back to its display name."""
    return next(
        (c for c in POSITIVE_LABEL_BY_CATEGORY if re.sub(r"[^\w\-]+", "_", c).strip("_") == token),
        token,
    )


def _discover_predictions(pred_base: Path) -> dict[tuple[str, str, str, str], dict]:
    """Finds every (model, label, aspect, category) under pred_base, in either layout.

    Returns {key: {"runs": {run_no: csv_path}, "sheet": xlsx_path | None}}.
    """
    found: dict[tuple[str, str, str, str], dict] = {}

    def entry(parts: tuple[str, ...]) -> dict:
        key = (parts[0], parts[1], parts[2], _category_from_token(parts[3]))
        return found.setdefault(key, {"runs": {}, "sheet": None})

    for path in pred_base.rglob("*.csv"):
        parts = path.relative_to(pred_base).parts
        # {model}/{label}/{aspect}/{category}/csv/{file}_runN.csv  (log/ holds raw text; skip)
        if len(parts) != 6 or parts[4] != "csv":
            continue
        m = _RUN_RE.search(path.name)
        if m:
            entry(parts)["runs"][int(m.group(1))] = path

    for path in pred_base.rglob("*_sheet.xlsx"):
        parts = path.relative_to(pred_base).parts
        if len(parts) == 5:
            entry(parts)["sheet"] = path

    return found


def _load_wide_predictions(item: dict) -> pd.DataFrame:
    """ID [+ PhraseA, PhraseB] + one "Test N" column per run.

    Runs are joined on ID with an OUTER join: a run that was cut off partway (the server dying
    mid-run) simply has NaN for the IDs it never reached, and is scored on the rows it did finish
    rather than dragging the complete runs down with it.
    """
    if item["runs"]:
        wide: pd.DataFrame | None = None
        for run_no in sorted(item["runs"]):
            df = pd.read_csv(item["runs"][run_no], dtype=str)
            test_col = f"Test {run_no}"
            if test_col not in df.columns:
                continue
            keep = ["ID", test_col] + [c for c in ("PhraseA", "PhraseB") if c in df.columns]
            df = df[keep].drop_duplicates(subset=["ID"], keep="last")
            # A blank cell here means the row WAS attempted but the answer couldn't be parsed
            # (e.g. a reasoning prompt running past max_tokens before "Answer: Yes/No"). Mark it,
            # so after the outer join a NaN unambiguously means "this run never reached the row".
            # The marker is not Yes/No, so it still counts as invalid rather than as a prediction.
            df[test_col] = df[test_col].fillna(UNPARSED)
            df.loc[df[test_col].astype(str).str.strip() == "", test_col] = UNPARSED
            if wide is None:
                wide = df
            else:
                phrase_cols = [c for c in ("PhraseA", "PhraseB") if c in df.columns and c in wide.columns]
                wide = wide.merge(df.drop(columns=phrase_cols), on="ID", how="outer")
                # fill phrases for IDs only this run reached
                for c in phrase_cols:
                    fill = df.set_index("ID")[c]
                    wide[c] = wide[c].fillna(wide["ID"].map(fill))
        return wide if wide is not None else pd.DataFrame()

    if item["sheet"] is not None:
        return pd.read_excel(item["sheet"], dtype=str)
    return pd.DataFrame()


def _check_alignment(pdf: pd.DataFrame, gold: dict) -> tuple[int, int, list[str]]:
    """Returns (checked, mismatched, examples): does each predicted ID carry the same phrase pair
    as the gold row with that ID?"""
    if "PhraseA" not in pdf.columns or "PhraseB" not in pdf.columns:
        return 0, 0, []
    checked = mismatched = 0
    examples: list[str] = []
    for id_, a, b in zip(pdf["ID"], pdf["PhraseA"], pdf["PhraseB"]):
        g = gold.get(id_)
        if g is None:
            continue
        checked += 1
        if (str(a).strip(), str(b).strip()) != (g.phrase_a.strip(), g.phrase_b.strip()):
            mismatched += 1
            if len(examples) < 3:
                examples.append(f"ID {id_}: pred=({a}, {b}) gold=({g.phrase_a}, {g.phrase_b})")
    return checked, mismatched, examples


_GOLD_CACHE: dict[tuple[str, str], dict] = {}


def _gold_for(task3_dir: Path, aspect: str, category: str) -> dict:
    """ID -> Task3Instance for one (aspect, category); loaded once per process."""
    key = (aspect, category)
    if key not in _GOLD_CACHE:
        instances = load_task3_instances(task3_dir, aspects=[aspect], categories=[category], max_per_group=None)
        _GOLD_CACHE[key] = {inst.row_id: inst for inst in instances}
    return _GOLD_CACHE[key]


ALIGNMENT_ISSUES: list[dict] = []


def evaluate_one(model: str, label: str, aspect: str, category: str, item: dict, task3_dir: Path) -> pd.DataFrame:
    positive_label = POSITIVE_LABEL_BY_CATEGORY.get(category)
    if positive_label is None:
        print(f"  [SKIP] unknown category '{category}' for {model}/{label}/{aspect} — no positive-label mapping")
        return pd.DataFrame()

    # IDs are only unique WITHIN a category (each category sheet restarts its own ID
    # numbering), so the gold lookup must be scoped to this exact (aspect, category) —
    # otherwise votes from a different category sharing the same aspect would collide.
    # max_per_group=None: predictions may have been generated from a random subset: we need
    # the FULL vote lookup so any ID that shows up in the prediction file resolves correctly,
    # not just whichever subset a sampling cap would keep.
    gold = _gold_for(task3_dir, aspect, category)
    gt_map = {row_id: inst.gold_vote for row_id, inst in gold.items()}
    if not gt_map:
        print(f"  [SKIP] no gold data for aspect '{aspect}' / category '{category}'")
        return pd.DataFrame()

    pdf = _load_wide_predictions(item)
    if pdf.empty:
        print(f"  [SKIP] no readable predictions for {model}/{label}/{aspect}/{category}")
        return pd.DataFrame()
    test_cols = sorted(
        (c for c in pdf.columns if re.match(r"^Test \d+$", str(c))),
        key=lambda c: int(c.split()[1]),
    )
    if not test_cols:
        print(f"  [SKIP] no Test N columns for {model}/{label}/{aspect}/{category}")
        return pd.DataFrame()

    checked, mismatched, examples = _check_alignment(pdf, gold)
    if mismatched:
        ALIGNMENT_ISSUES.append({"Model": model, "Shot": label, "Topic": aspect, "Category": category,
                                 "Checked": checked, "Mismatched": mismatched,
                                 "Example": examples[0] if examples else ""})
        print(f"  [WARN] {model}/{label}/{aspect}/{category}: {mismatched}/{checked} prediction rows "
              f"do not match their gold phrase pair — e.g. {examples[0]}")
    pdf = pdf.fillna("")

    out_dir = EVAL_BASE / aspect / model / label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_xlsx = out_dir / f"{category}.xlsx"

    summary_rows = []
    sheets_for_txt: list[tuple[str, pd.DataFrame]] = []
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        for tcol in test_cols:
            tmp = pdf[["ID", tcol]].copy()
            tmp.columns = ["ID", "LLM_Output"]
            # N_pred = rows this run ATTEMPTED (a run cut off mid-way leaves them blank);
            # N_unparsed = attempted but no Yes/No could be read from the answer.
            attempted = tmp["LLM_Output"].astype(str).str.strip() != ""
            n_pred = int(attempted.sum())
            n_unparsed = int((tmp["LLM_Output"] == UNPARSED).sum())
            tmp["LLM_Output"] = tmp["LLM_Output"].map(norm_yn)
            tmp["Vote"] = tmp["ID"].map(gt_map)
            tmp = tmp[tmp["Vote"].notna()].copy()
            tmp["Verify"] = [classify_verify(g, p, positive_label) for g, p in zip(tmp["Vote"], tmp["LLM_Output"])]
            tmp = tmp[["ID", "Vote", "LLM_Output", "Verify"]]
            tmp.to_excel(writer, sheet_name=tcol, index=False)
            sheets_for_txt.append((tcol, tmp))

            met = compute_metrics(tmp["Vote"], tmp["LLM_Output"], positive_label)
            met.update({"Model": model, "Shot": label, "Topic": aspect, "Category": category,
                        "Test": tcol, "Positive_Label": positive_label,
                        "N_pred": n_pred, "N_unparsed": n_unparsed, "N_gold": len(gt_map)})
            summary_rows.append(met)

        summary_df = pd.DataFrame(summary_rows)[
            ["Model", "Shot", "Topic", "Category", "Test", "Positive_Label",
             "N_gold", "N_pred", "N_unparsed", "N_valid", "TP", "TN", "FP", "FN",
             "Precision", "Recall", "F1", "Accuracy"]
        ]
        summary_df.to_excel(writer, sheet_name="SUMMARY", index=False)
        sheets_for_txt.append(("SUMMARY", summary_df))

    _write_txt_companion(out_xlsx, sheets_for_txt)
    print(f"  [OK] {aspect} | {model} | {label} | {category} -> {out_xlsx.relative_to(REPO_ROOT)} (+ .txt)")
    return summary_df


def write_eval_summary(aspect: str, model: str, label: str, combined_df: pd.DataFrame) -> None:
    if combined_df.empty:
        return
    out_dir = EVAL_BASE / aspect / model / label
    out_path = out_dir / "EVAL_SUMMARY.xlsx"
    cols = ["Model", "Shot", "Topic", "Category", "Test", "Positive_Label",
            "N_valid", "TP", "TN", "FP", "FN", "Precision", "Recall", "F1", "Accuracy"]

    all_df = combined_df[cols]
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        all_df.to_excel(writer, sheet_name="ALL", index=False)

        group_cols = ["Model", "Shot", "Topic", "Category", "Positive_Label"]
        tot = combined_df.groupby(group_cols, as_index=False)[["N_valid", "TP", "TN", "FP", "FN"]].sum()
        tot["Precision"] = safe_div_series(tot["TP"], tot["TP"] + tot["FP"])
        tot["Recall"] = safe_div_series(tot["TP"], tot["TP"] + tot["FN"])
        tot["F1"] = safe_div_series(2 * tot["Precision"] * tot["Recall"], tot["Precision"] + tot["Recall"])
        tot["Accuracy"] = safe_div_series(tot["TP"] + tot["TN"], tot["TP"] + tot["TN"] + tot["FP"] + tot["FN"])
        tot = _round_metrics(tot)
        micro_df = tot[group_cols + ["Precision", "Recall", "F1", "Accuracy", "N_valid", "TP", "TN", "FP", "FN"]]
        micro_df.to_excel(writer, sheet_name="MICRO_BY_CATEGORY", index=False)

    _write_txt_companion(out_path, [("ALL", all_df), ("MICRO_BY_CATEGORY", micro_df)])
    print(f"  [EVAL_SUMMARY] {out_path.relative_to(REPO_ROOT)} (+ .txt)")


# mu_team's own Task 3 scope, confirmed by checking every one of their scripts — see
# Task3_Implementation_Report.md §6. Facility (and any other topic) is data this project
# gathered beyond what mu_team ever ran.
MU_WORK_TOPICS = ["check-in", "check-out", "price", "staff"]


def write_master_summary(
    all_rows: list[pd.DataFrame],
    topics: list[str] | None = None,
    out_path: Path | None = None,
) -> None:
    """topics=None writes the full summary (every topic evaluated so far). Pass a topic list
    (e.g. MU_WORK_TOPICS) to write a scoped-down summary instead, at out_path."""
    if not all_rows:
        print("\n[WARN] No evaluated rows — nothing to summarize.")
        return

    master_df = pd.concat(all_rows, ignore_index=True)
    if topics is not None:
        master_df = master_df[master_df["Topic"].isin(topics)]
        if master_df.empty:
            print(f"\n[WARN] No rows left after filtering to topics={topics}")
            return

    out_path = out_path or MASTER_SUMMARY_XLSX
    cols = ["Model", "Shot", "Topic", "Category", "Test", "Positive_Label",
            "N_gold", "N_pred", "N_unparsed", "N_valid", "TP", "TN", "FP", "FN",
            "Precision", "Recall", "F1", "Accuracy"]
    all_eval_summary = master_df[cols].copy()

    avg_metrics = (
        master_df.groupby(["Model", "Shot", "Topic", "Category", "Positive_Label"], as_index=False)
        .agg({"Precision": "mean", "Recall": "mean", "F1": "mean", "Accuracy": "mean"})
    )
    avg_metrics = _round_metrics(avg_metrics)

    # Best-scoring run (of the 3 repeated runs) per (Model, Shot, Topic, Category) — one column,
    # joined on the row's own Shot. (A previous version pivoted "Shot" into separate wide
    # columns and joined only on Model/Topic/Category/Positive_Label, which — since that key
    # doesn't pin down a single Shot — broadcast every shot's best-test pick onto every other
    # shot's row for the same topic. That's what caused the repeated/duplicated-looking values,
    # most visible for facility since it's missing several shots and left them as stray NaNs.)
    best_pick = (
        master_df.sort_values(
            by=["Model", "Shot", "Topic", "Category", "Positive_Label", "F1", "Accuracy", "Recall", "Precision"],
            ascending=[True, True, True, True, True, False, False, False, False],
        )
        .groupby(["Model", "Shot", "Topic", "Category", "Positive_Label"], as_index=False)
        .first()
    )
    best_test = best_pick[["Model", "Shot", "Topic", "Category", "Positive_Label", "Test"]].rename(
        columns={"Test": "Best Test"}
    )

    avg_by_model_shot_topic_cat = avg_metrics.merge(
        best_test, on=["Model", "Shot", "Topic", "Category", "Positive_Label"], how="left"
    ).sort_values(by=["Model", "Shot", "Topic", "Category"]).reset_index(drop=True)

    EVAL_BASE.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        all_eval_summary.to_excel(writer, sheet_name="ALL_EVAL_SUMMARY", index=False)
        avg_by_model_shot_topic_cat.to_excel(writer, sheet_name="AVG_BY_MODEL_SHOT_TOPIC_CAT", index=False)

    _write_txt_companion(out_path, [
        ("ALL_EVAL_SUMMARY", all_eval_summary),
        ("AVG_BY_MODEL_SHOT_TOPIC_CAT", avg_by_model_shot_topic_cat),
    ])
    print(f"\n[MASTER SUMMARY SAVED] {out_path.relative_to(REPO_ROOT)} (+ .txt)")


# Display names used by the Google Sheet's dropdown chips, so the CSVs paste in without
# "invalid value" warnings. Anything not listed here is written as-is.
SHEET_VERSION_NAMES = {"zero_shot": "Zero-shot", "one_shot": "One-shot",
                       **{f"contrary_v{i}": f"Contrary{i}" for i in range(1, 10)}}
SHEET_TOPIC_NAMES = {"check-in": "Check-in", "check-out": "Check-out", "price": "Price",
                     "staff": "Staff", "facility": "Facility"}
SHEET_TOPIC_ORDER = list(SHEET_TOPIC_NAMES)
SHEET_METRICS = ["Precision", "Recall", "F1", "Accuracy"]

# The per-run table pools TP/FP/FN/TN across topics, which is only meaningful when every pooled
# row uses the same positive label — so it covers ONE category. Contrary(P)Body(N) is the one
# every model ran; the other categories stay visible, per topic, in the per-topic table.
SHEET_RUN_CATEGORY = "Contrary(P)Body(N)"


def _sheet_order(df: pd.DataFrame) -> pd.DataFrame:
    version_rank = {v: i for i, v in enumerate(SHEET_VERSION_NAMES)}
    topic_rank = {t: i for i, t in enumerate(SHEET_TOPIC_ORDER)}
    return df.assign(
        _v=df["Shot"].map(version_rank).fillna(len(version_rank)),
        _t=df["Topic"].map(topic_rank).fillna(len(topic_rank)) if "Topic" in df else 0,
        _r=df["Test"].str.extract(r"(\d+)", expand=False).astype(int),
    )


def write_sheet_per_topic(all_rows: list[pd.DataFrame], out_path: Path) -> None:
    """Sheet table 1 — one row per (Model, Version, Topic, Category, Test):
    Model | Version | Topic | Category | Test | Label | Precision | Recall | F1 | Accuracy
    Label is the positive label the metrics were computed against."""
    if not all_rows:
        return
    df = _sheet_order(pd.concat(all_rows, ignore_index=True))
    df = df.sort_values(["Model", "_v", "_t", "Category", "_r"])
    out = pd.DataFrame({
        "Model": df["Model"],
        "Version": df["Shot"].map(lambda v: SHEET_VERSION_NAMES.get(v, v)),
        "Topic": df["Topic"].map(lambda t: SHEET_TOPIC_NAMES.get(t, t)),
        "Category": df["Category"],
        "Test": df["Test"],
        "Label": df["Positive_Label"],
        **{m: df[m].round(4) for m in SHEET_METRICS},
    })
    out.to_csv(out_path, index=False)
    print(f"[SHEET TABLE 1] {out_path.relative_to(REPO_ROOT)}  ({len(out)} rows)")


def write_sheet_summary(
    all_rows: list[pd.DataFrame],
    topics: list[str] | None = None,
    out_path: Path | None = None,
) -> None:
    """Sheet table 2 — one row per (Model, Version, Run), aggregated across topics:
    Model | Version | Topic | Run | Macro P/R/F1/Acc | Micro P/R/F1/Acc
    Macro = plain mean of the per-topic metrics; Micro = metrics from TP/FP/FN/TN pooled across
    topics. "Topic" lists the topics that went into that run's row, so a run that is missing a
    topic is visible rather than silently averaged over fewer.

    topics=None aggregates every topic evaluated so far; pass a topic list (e.g.
    MU_WORK_TOPICS) to scope the aggregation down to just those topics.
    """
    if not all_rows:
        return
    master_df = pd.concat(all_rows, ignore_index=True)
    master_df = master_df[master_df["Category"] == SHEET_RUN_CATEGORY]
    if topics is not None:
        master_df = master_df[master_df["Topic"].isin(topics)]
    if master_df.empty:
        print(f"[WARN] No {SHEET_RUN_CATEGORY} rows left for topics={topics}")
        return
    group_cols = ["Model", "Shot", "Test"]

    macro = master_df.groupby(group_cols, as_index=False)[SHEET_METRICS].mean()
    macro = macro.rename(columns={m: f"Macro {m}" for m in SHEET_METRICS})

    pooled = master_df.groupby(group_cols, as_index=False)[["TP", "TN", "FP", "FN"]].sum()
    pooled["Micro Precision"] = safe_div_series(pooled["TP"], pooled["TP"] + pooled["FP"])
    pooled["Micro Recall"] = safe_div_series(pooled["TP"], pooled["TP"] + pooled["FN"])
    pooled["Micro F1"] = safe_div_series(2 * pooled["Micro Precision"] * pooled["Micro Recall"],
                                          pooled["Micro Precision"] + pooled["Micro Recall"])
    pooled["Micro Accuracy"] = safe_div_series(pooled["TP"] + pooled["TN"],
                                                pooled["TP"] + pooled["TN"] + pooled["FP"] + pooled["FN"])

    topic_rank = {t: i for i, t in enumerate(SHEET_TOPIC_ORDER)}
    topics_covered = (
        master_df.groupby(group_cols)["Topic"]
        .apply(lambda s: ", ".join(SHEET_TOPIC_NAMES.get(t, t)
                                   for t in sorted(set(s), key=lambda t: topic_rank.get(t, 99))))
        .reset_index()
    )

    out = macro.merge(pooled.drop(columns=["TP", "TN", "FP", "FN"]), on=group_cols)
    out = out.merge(topics_covered, on=group_cols)
    out = _sheet_order(out).sort_values(["Model", "_v", "_r"])

    metric_cols = [f"Macro {m}" for m in SHEET_METRICS] + [f"Micro {m}" for m in SHEET_METRICS]
    out = pd.DataFrame({
        "Model": out["Model"],
        "Version": out["Shot"].map(lambda v: SHEET_VERSION_NAMES.get(v, v)),
        "Topic": out["Topic"],
        "Run": out["_r"],
        **{c: out[c].round(4) for c in metric_cols},
    })

    csv_path = out_path or (EVAL_BASE / "TASK3_SUMMARY_FOR_SHEETS.csv")
    out.to_csv(csv_path, index=False)
    print(f"[SHEET TABLE 2] {csv_path.relative_to(REPO_ROOT)}  ({len(out)} rows)")


def main() -> None:
    global PRED_BASE, EVAL_BASE, MASTER_SUMMARY_XLSX

    parser = argparse.ArgumentParser(description="Evaluate Task 3 predictions against gold votes.")
    parser.add_argument("--pred-dir", default=str(PRED_BASE),
                        help="Folder whose children are model folders (default: outputs/task3). "
                             "Point this at a copied-back server folder to evaluate it on its own.")
    parser.add_argument("--eval-dir", default=str(EVAL_BASE),
                        help="Where to write results (default: outputs/eval/task3). Use a separate "
                             "folder per batch so results from different runs never mix.")
    args = parser.parse_args()

    PRED_BASE = Path(args.pred_dir).resolve()
    EVAL_BASE = Path(args.eval_dir).resolve()
    MASTER_SUMMARY_XLSX = EVAL_BASE / "TASK3_VERIFY_MASTER_SUMMARY.xlsx"

    paths_cfg = load_paths_config(REPO_ROOT)

    discovered = _discover_predictions(PRED_BASE)
    if not discovered:
        sys.exit(f"[ERROR] No prediction files found under {PRED_BASE} (run run_task3.py first)")

    n_runs = sum(len(v["runs"]) for v in discovered.values())
    print(f"Found {len(discovered)} (model, version, topic, category) group(s), {n_runs} run file(s) "
          f"under {PRED_BASE}\n")

    # Group by (model, label, aspect) so EVAL_SUMMARY.xlsx covers every category found for that combo.
    grouped: dict[tuple[str, str, str], list[tuple[str, dict]]] = {}
    for (model, label, aspect, category), item in discovered.items():
        grouped.setdefault((model, label, aspect), []).append((category, item))

    all_rows: list[pd.DataFrame] = []
    for (model, label, aspect), items in sorted(grouped.items()):
        combined = []
        for category, item in sorted(items, key=lambda x: x[0]):
            df = evaluate_one(model, label, aspect, category, item, paths_cfg.task3_dir)
            if not df.empty:
                combined.append(df)
        if combined:
            combined_df = pd.concat(combined, ignore_index=True)
            write_eval_summary(aspect, model, label, combined_df)
            all_rows.append(combined_df)

    # Default output is scoped to mu_team's own 4-topic scope (check-in/check-out/price/staff)
    # so it's directly comparable to their results without extra filtering. The full picture
    # (including facility and anything else evaluated) is the explicitly-named ALL_TOPICS file.
    write_master_summary(all_rows, topics=MU_WORK_TOPICS)
    write_master_summary(
        all_rows, out_path=EVAL_BASE / "TASK3_VERIFY_MASTER_SUMMARY_ALL_TOPICS.xlsx",
    )
    write_sheet_per_topic(all_rows, out_path=EVAL_BASE / "TASK3_SHEET_PER_TOPIC.csv")
    write_sheet_summary(all_rows, topics=MU_WORK_TOPICS)
    write_sheet_summary(
        all_rows, out_path=EVAL_BASE / "TASK3_SUMMARY_FOR_SHEETS_ALL_TOPICS.csv",
    )

    report = EVAL_BASE / "ALIGNMENT_CHECK.csv"
    if ALIGNMENT_ISSUES:
        pd.DataFrame(ALIGNMENT_ISSUES).to_csv(report, index=False)
        print()
        print(f"[ALIGNMENT] {len(ALIGNMENT_ISSUES)} group(s) have predictions that do not match "
              f"their gold phrase pair — see {report.relative_to(REPO_ROOT)}")
    else:
        pd.DataFrame([{"Result": "every prediction row matched its gold (PhraseA, PhraseB) pair"}]).to_csv(
            report, index=False)
        print()
        print("[ALIGNMENT] every prediction row matched its gold phrase pair")


if __name__ == "__main__":
    main()
