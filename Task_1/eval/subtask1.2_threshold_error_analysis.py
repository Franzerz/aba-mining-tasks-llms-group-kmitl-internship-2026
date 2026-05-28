"""
Subtask 1.2 – Threshold Error Analysis & Decision Helper
---------------------------------------------------------
Following professor's advice: only TF-IDF + Cross-Encoder.
For each threshold (0.3, 0.4, 0.5) show a breakdown of what kind
of errors exist so you can decide which threshold is acceptable.

Error types (from worst to best):
  PARSE FAILED   – LLM output could not be parsed (JSON error) — always wrong
  HALLUCINATION  – LLM gave text for a topic GT marks null (no GT entry)
  WRONG TOPIC    – GT exists but score = 0 (completely different sentence)
  WRONG CONTENT  – Has GT, 0 < score < 0.3 (very different text)
  PARAPHRASE     – Has GT, 0.3 ≤ score < threshold (similar meaning, diff words)
  PARTIAL EXTRACT– LLM text is a sub-phrase of GT text (acceptable)
  PASS           – score ≥ threshold
"""

import re
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Paths ────────────────────────────────────────────────────────────────────────
TASK_DIR = Path(__file__).resolve().parent.parent
GT_CSV = Path("/Users/Pngoendee/ReaLearn/Dataset") / (
    "Original ABA Dataset for Version 2 "
    "[Oct 23, 2025], Senior Project, MUICT - Sheet2_.csv"
)
LLM_CSVS = [
    TASK_DIR / "outputs/task1/llama4_scout/modular/subtask1_2"
             / "task1_llama4_scout_extended11_subtask1_2.csv",
    TASK_DIR / "outputs/task1/llama3.2/modular/subtask1_2"
             / "task1_llama3.2_extended11_subtask1_2.csv",
]
OUT_DIR = TASK_DIR / "outputs/eval/content"
OUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
HTML_REVIEW_LIMIT = 50  # HTML shows only Review ID 1–50

parser = argparse.ArgumentParser()
parser.add_argument("--skip-ce", action="store_true",
                    help="Skip Cross-Encoder (TF-IDF only, much faster)")
args = parser.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────────────────
def normalize(text) -> str:
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def tfidf_score_all(query: str, candidates: list[str]) -> list[float]:
    if not candidates or not query:
        return [0.0] * len(candidates)
    try:
        vec = TfidfVectorizer(min_df=1, analyzer="word", ngram_range=(1, 2))
        mat = vec.fit_transform([query] + candidates)
        return cosine_similarity(mat[0:1], mat[1:])[0].tolist()
    except Exception:
        return [0.0] * len(candidates)


_cross_model = None


def _get_cross_model():
    global _cross_model
    if _cross_model is None:
        from sentence_transformers import CrossEncoder
        print("  [Loading Cross-Encoder …]")
        _cross_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_model


def is_partial_extract(llm_norm: str, gt_norm: str) -> bool:
    if not llm_norm or not gt_norm:
        return False
    if llm_norm in gt_norm:
        return True
    if gt_norm in llm_norm and len(gt_norm) > 15:
        return True
    return False


def classify(score, hallucinated: bool, partial: bool,
             parse_failed: bool, threshold: float) -> str:
    if parse_failed:
        return "PARSE FAILED"
    if hallucinated:
        return "HALLUCINATION"
    if pd.isna(score):
        return "PARSE FAILED"
    if score >= threshold:
        return "PASS"
    if partial:
        return "PARTIAL EXTRACT"
    if score == 0.0:
        return "WRONG TOPIC"
    if score < 0.3:
        return "WRONG CONTENT"
    return "PARAPHRASE"


# ── Load GT ──────────────────────────────────────────────────────────────────────
if not GT_CSV.exists():
    sys.exit(f"[ERROR] Ground truth not found:\n  {GT_CSV}")

gt_raw = pd.read_csv(GT_CSV)
gt_raw = gt_raw.rename(columns={"ID": "Review ID"})
gt_raw = gt_raw[gt_raw["Topic"].notna() & gt_raw["Selected Content"].notna()].copy()
gt_raw["Topic"] = gt_raw["Topic"].str.strip()
gt_raw["norm"] = gt_raw["Selected Content"].apply(normalize)
gt_raw = gt_raw[gt_raw["norm"] != ""].reset_index(drop=True)

gt_index: dict = {}
for _, row in gt_raw.iterrows():
    key = (row["Review ID"], row["Topic"])
    gt_index.setdefault(key, []).append((row["norm"], row["Selected Content"]))

# All review IDs that exist in GT
gt_review_ids = set(gt_raw["Review ID"].unique())

print(f"Ground truth: {len(gt_raw)} rows, {gt_raw['Review ID'].nunique()} reviews\n")


# ── Score all rows ────────────────────────────────────────────────────────────────
def score_csv(llm_csv: Path) -> pd.DataFrame:
    raw = pd.read_csv(llm_csv)
    if "Review ID" not in raw.columns and "ID" in raw.columns:
        raw = raw.rename(columns={"ID": "Review ID"})

    records = []
    all_pairs, pair_meta = [], []

    _bad_topics = {"parse failed", "topics not found", "topic not found", "(parse failed)"}

    # Group by Review ID and Run to detect parse failures at the review level
    run_col = "Run" if "Run" in raw.columns else None
    groups = list(raw.groupby(["Review ID", "Run"] if run_col else ["Review ID"]))

    for keys, grp in groups:
        rid = keys[0] if isinstance(keys, tuple) else keys
        run = keys[1] if isinstance(keys, tuple) and len(keys) > 1 else 1

        # Check if this run is a parse failure
        has_errors = "Errors" in grp.columns
        all_failed = has_errors and grp["Errors"].notna().all() and (
            grp["Errors"].astype(str).str.strip().str.len() > 0
        ).all()
        has_bad_topic = (
            grp["Topic"].isna().all() or
            grp["Topic"].str.strip().str.lower().isin(_bad_topics).all()
        )

        if all_failed or has_bad_topic:
            # This run fully failed — record once per (review, run) with GT topics
            gt_topics_for_rid = set(
                t for (r, t) in gt_index.keys() if r == rid
            )
            if not gt_topics_for_rid:
                gt_topics_for_rid = {"(unknown)"}
            for topic in gt_topics_for_rid:
                gt_entries = gt_index.get((rid, topic), [])
                gt_text = gt_entries[0][1] if gt_entries else "(no GT entry)"
                records.append({
                    "Review ID": rid,
                    "Run": run,
                    "Topic": topic,
                    "LLM Text": "parse failed: no output",
                    "GT Text": gt_text,
                    "TF-IDF": float("nan"),
                    "CE": float("nan"),
                    "Hallucinated": False,
                    "Partial Extract": False,
                    "Parse Failed": True,
                })
            continue

        # Filter to valid rows with text spans
        valid = grp.copy()
        if has_errors:
            valid = valid[
                valid["Errors"].isna() |
                valid["Errors"].astype(str).str.strip().isin({"", "nan"})
            ]
        valid = valid[
            valid["Topic"].notna() &
            (valid["Topic"].str.strip() != "") &
            (~valid["Topic"].str.strip().str.lower().isin(_bad_topics)) &
            valid["Text Span"].notna() &
            (valid["Text Span"].str.strip() != "")
        ].copy()
        valid["Topic"] = valid["Topic"].str.strip()
        valid["norm"] = valid["Text Span"].apply(normalize)
        valid = valid[valid["norm"] != ""]
        valid = valid.drop_duplicates(subset=["Review ID", "Topic", "norm"])

        for _, row in valid.iterrows():
            key = (row["Review ID"], row["Topic"])
            gt_entries = gt_index.get(key, [])
            gt_norms = [e[0] for e in gt_entries]
            gt_originals = [e[1] for e in gt_entries]

            if not gt_entries:
                records.append({
                    "Review ID": row["Review ID"],
                    "Run": run,
                    "Topic": row["Topic"],
                    "LLM Text": row["Text Span"],
                    "GT Text": "Topic is null in ground truth",
                    "TF-IDF": 0.0,
                    "CE": float("nan"),
                    "Hallucinated": True,
                    "Partial Extract": False,
                    "Parse Failed": False,
                })
                continue

            scores = tfidf_score_all(row["norm"], gt_norms)
            tfidf = max(scores)
            best_idx = int(np.argmax(scores))
            partial = is_partial_extract(row["norm"], gt_norms[best_idx])

            rec = {
                "Review ID": row["Review ID"],
                "Run": run,
                "Topic": row["Topic"],
                "LLM Text": row["Text Span"],
                "GT Text": gt_originals[best_idx],
                "TF-IDF": tfidf,
                "CE": float("nan"),
                "Hallucinated": False,
                "Partial Extract": partial,
                "Parse Failed": False,
                "_llm_norm": row["norm"],
                "_gt_norms": gt_norms,
            }
            records.append(rec)
            if not args.skip_ce:
                for g in gt_norms:
                    all_pairs.append((row["norm"], g))
                    pair_meta.append(len(records) - 1)

    if not args.skip_ce and all_pairs:
        print("  Cross-Encoder (batched) …")
        model = _get_cross_model()
        logits = np.array(
            model.predict(all_pairs, batch_size=256, show_progress_bar=True), dtype=float
        )
        probs = 1.0 / (1.0 + np.exp(-logits))
        best_ce: dict[int, float] = {}
        for k, rec_idx in enumerate(pair_meta):
            best_ce[rec_idx] = max(best_ce.get(rec_idx, 0.0), float(probs[k]))
        for rec_idx, ce_score in best_ce.items():
            records[rec_idx]["CE"] = ce_score

    df = pd.DataFrame(records)
    for col in ["_llm_norm", "_gt_norms"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    n_total = len(df)
    n_parse = int(df["Parse Failed"].sum())
    n_halluc = int(df["Hallucinated"].sum())
    print(f"  {n_total} total rows  ({n_parse} parse-failed, {n_halluc} hallucinated)")
    return df


# ── Decision analysis ──────────────────────────────────────────────────────────
def decision_analysis(df: pd.DataFrame, model_label: str) -> None:
    n_total = len(df)
    n_parse = int(df["Parse Failed"].sum())
    n_halluc = int(df["Hallucinated"].sum())
    real = df[~df["Hallucinated"] & ~df["Parse Failed"]].copy()
    n_real = len(real)

    print(f"\n{'=' * 72}")
    print(f"THRESHOLD DECISION ANALYSIS — {model_label}")
    print(f"{'=' * 72}")
    print(f"Total rows          : {n_total}")
    print(f"PARSE FAILED        : {n_parse}  (LLM couldn't produce valid JSON)")
    print(f"HALLUCINATION       : {n_halluc}  (text for topic GT marks null)")
    print(f"Rows with GT (scoreable): {n_real}\n")

    ORDER = ["PASS", "PARTIAL EXTRACT", "PARAPHRASE", "WRONG CONTENT", "WRONG TOPIC"]

    rows_by_t: dict = {}
    for t in THRESHOLDS:
        tmp = real.copy()
        tmp["Error Type"] = tmp.apply(
            lambda r: classify(r["TF-IDF"], r["Hallucinated"],
                               r["Partial Extract"], r["Parse Failed"], t), axis=1
        )
        rows_by_t[t] = tmp

    meanings = {
        "PASS": "score ≥ threshold → accepted",
        "PARTIAL EXTRACT": "LLM picked sub-phrase of GT sentence → acceptable",
        "PARAPHRASE": "similar meaning, different wording → borderline",
        "WRONG CONTENT": "score 0–0.3, mostly wrong sentence",
        "WRONG TOPIC": "score = 0, completely different content",
    }

    print(f"{'Error Type':<20} {'T=0.3':>8} {'T=0.4':>8} {'T=0.5':>8}  Meaning")
    print("─" * 80)
    for et in ORDER:
        counts = [int((rows_by_t[t]["Error Type"] == et).sum()) for t in THRESHOLDS]
        print(f"  {et:<18} {counts[0]:>8} {counts[1]:>8} {counts[2]:>8}  {meanings[et]}")
    print(f"  {'HALLUCINATION':<18} {n_halluc:>8} {n_halluc:>8} {n_halluc:>8}"
          f"  LLM invented text, topic not in GT")
    print(f"  {'PARSE FAILED':<18} {n_parse:>8} {n_parse:>8} {n_parse:>8}"
          f"  LLM output couldn't be parsed")
    print("─" * 80)
    print(f"  {'TOTAL':<18} {n_total:>8} {n_total:>8} {n_total:>8}")

    print(f"\n{'─' * 72}")
    print("WHAT CHANGES BETWEEN THRESHOLDS (new failures added):")
    print(f"{'─' * 72}")
    for i in range(len(THRESHOLDS) - 1):
        t_lo, t_hi = THRESHOLDS[i], THRESHOLDS[i + 1]
        new_fail = real[(real["TF-IDF"] >= t_lo) & (real["TF-IDF"] < t_hi)].copy()
        n_new = len(new_fail)
        n_partial = int(new_fail["Partial Extract"].sum())
        n_para = n_new - n_partial
        pct_ok = n_partial / n_new * 100 if n_new > 0 else 0
        print(f"\n  T={t_lo} → T={t_hi}: {n_new} newly rejected")
        print(f"    ├── PARTIAL EXTRACT : {n_partial}  (acceptable)")
        print(f"    └── PARAPHRASE      : {n_para}  (borderline)")
        print(f"         → {pct_ok:.0f}% are partial extracts (acceptable)")

    print(f"\n{'─' * 72}")
    print("RECOMMENDATION:")
    print(f"{'─' * 72}")
    new_03_04 = real[(real["TF-IDF"] >= 0.3) & (real["TF-IDF"] < 0.4)]
    pct = new_03_04["Partial Extract"].mean() * 100 if len(new_03_04) > 0 else 0
    if pct >= 50:
        print(f"  {pct:.0f}% of T=0.3→0.4 new rejections are partial extracts → T=0.4 too strict")
        print("  ★  Use T=0.3")
    else:
        print(f"  Only {pct:.0f}% of T=0.3→0.4 rejections are partial extracts")
        print("  ★  Use T=0.4")
    print("  (Check HTML to manually confirm borderline cases)\n")


# ── HTML ───────────────────────────────────────────────────────────────────────
_BG = {
    "PASS":            "#d9ead3",
    "PARTIAL EXTRACT": "#cfe2f3",
    "PARAPHRASE":      "#fff2cc",
    "WRONG CONTENT":   "#fce5cd",
    "WRONG TOPIC":     "#f4cccc",
    "HALLUCINATION":   "#e6b8af",
    "PARSE FAILED":    "#b4a7d6",
}

_LABEL_DESC = {
    "PARTIAL EXTRACT": "LLM selected a sub-phrase of the GT sentence — acceptable",
    "PARAPHRASE":      "Similar meaning but different words — decide if acceptable",
    "WRONG CONTENT":   "Mostly different text (score 0–0.3) — likely wrong",
    "WRONG TOPIC":     "Score = 0, completely different sentence — very wrong",
    "HALLUCINATION":   "LLM gave text for a topic GT marks null — always wrong",
    "PARSE FAILED":    "LLM output could not be parsed — no text extracted at all",
}


def build_html(df: pd.DataFrame, model_label: str, threshold: float) -> str:
    df = df[df["Review ID"] <= HTML_REVIEW_LIMIT].copy()
    n_total = len(df)
    n_parse = int(df["Parse Failed"].sum())
    n_halluc = int(df["Hallucinated"].sum())
    real = df[~df["Hallucinated"] & ~df["Parse Failed"]].copy()
    n_real = len(real)

    real["Error Type"] = real.apply(
        lambda r: classify(r["TF-IDF"], False, r["Partial Extract"], False, threshold),
        axis=1
    )
    n_pass = int((real["Error Type"] == "PASS").sum())

    # Build failures table: non-passing real rows + hallucinated + parse-failed
    parse_df = df[df["Parse Failed"]].copy()
    parse_df["Error Type"] = "PARSE FAILED"
    halluc_df = df[df["Hallucinated"]].copy()
    halluc_df["Error Type"] = "HALLUCINATION"
    fail_real = real[real["Error Type"] != "PASS"].copy()

    failures = pd.concat([fail_real, halluc_df, parse_df]) \
                 .sort_values(["Review ID", "Run", "Topic"]) \
                 .reset_index(drop=True)

    # Summary pills
    et_order = ["PARTIAL EXTRACT", "PARAPHRASE", "WRONG CONTENT",
                "WRONG TOPIC", "HALLUCINATION", "PARSE FAILED"]
    counts = {et: int((real["Error Type"] == et).sum()) for et in et_order}
    counts["HALLUCINATION"] = n_halluc
    counts["PARSE FAILED"] = n_parse

    pills = "".join(
        f'<span class="pill" style="background:{_BG[et]}">{et}: {counts[et]}</span> '
        for et in et_order
    )

    # Filter options per column
    def opts(values):
        uniq = sorted(set(str(v) for v in values if str(v) not in ("nan", "")))
        return "".join(f'<option value="{v}">{v}</option>' for v in uniq)

    topic_opts   = opts(failures["Topic"])
    run_opts     = opts(failures["Run"])
    errtype_opts = opts(failures["Error Type"])

    rows_html = ""
    for _, r in failures.iterrows():
        et = r.get("Error Type", "HALLUCINATION")
        bg = _BG.get(et, "#fff")
        tfidf_str = f"{r['TF-IDF']:.3f}" if not pd.isna(r["TF-IDF"]) else ""
        ce_val = r.get("CE", float("nan"))
        ce_str = f"{ce_val:.3f}" if not pd.isna(ce_val) else ""
        rows_html += f"""
        <tr style="background:{bg};color:#000" data-topic="{r['Topic']}"
            data-run="{int(r['Run'])}" data-errtype="{et}"
            title="{_LABEL_DESC.get(et, '')}">
          <td>{int(r['Review ID'])}</td>
          <td>{int(r['Run'])}</td>
          <td>{r['Topic']}</td>
          <td class="text">{r['LLM Text']}</td>
          <td class="text">{r['GT Text']}</td>
          <td style="text-align:center">{tfidf_str}</td>
          <td style="text-align:center">{ce_str}</td>
          <td><b>{et}</b></td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Subtask 1.2 Errors — {model_label} T={threshold} (ID 1–{HTML_REVIEW_LIMIT})</title>
  <style>
    body {{ font-family: Arial, sans-serif; font-size: 13px; margin: 20px; background: #fafafa; }}
    h1 {{ color: #222; margin-bottom: 4px; }}
    h2 {{ color: #444; margin-top: 2px; margin-bottom: 12px; }}
    .info {{ background: #f0f0f0; padding: 10px 14px; border-radius: 6px;
             margin-bottom: 10px; line-height: 2; }}
    .pill {{ display: inline-block; padding: 2px 10px; border-radius: 10px;
             font-size: 0.82em; font-weight: bold; margin: 2px; border: 1px solid rgba(0,0,0,.1); }}
    .filters {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
                margin-bottom: 10px; }}
    .filters label {{ font-size: 0.85em; color: #555; }}
    .filters select {{ font-size: 0.85em; padding: 3px 6px; border-radius: 4px;
                       border: 1px solid #ccc; }}
    .filters button {{ font-size: 0.82em; padding: 3px 10px; border-radius: 4px;
                       border: 1px solid #aaa; cursor: pointer; background: #fff; }}
    table {{ border-collapse: collapse; width: 100%; background: #fff;
             box-shadow: 0 1px 4px rgba(0,0,0,.1); border-radius: 6px; overflow: hidden; }}
    th {{ background: #2d3748; color: #fff; padding: 9px 10px; text-align: left;
          font-size: 0.88em; white-space: nowrap; }}
    td {{ padding: 7px 10px; border-bottom: 1px solid #e2e8f0; vertical-align: top; }}
    td.text {{ font-size: 0.87em; max-width: 300px; }}
    tr:hover td {{ filter: brightness(0.93); }}
    .hidden {{ display: none; }}
    #count {{ font-size: 0.85em; color: #555; margin-bottom: 6px; }}
  </style>
</head>
<body>
  <h1>Subtask 1.2 — Error Analysis (Failures Only)</h1>
  <h2>{model_label} &nbsp;|&nbsp; Threshold = {threshold} &nbsp;|&nbsp; Review ID 1–{HTML_REVIEW_LIMIT}</h2>

  <div class="info">
    <b>Total rows:</b> {n_total} &nbsp;|&nbsp;
    <b>PASS (score ≥ {threshold}):</b> {n_pass} ({n_pass/n_real*100:.1f}% of scoreable) &nbsp;|&nbsp;
    <b>Failures below:</b> {len(failures)}<br>
    {pills}
  </div>

  <div class="filters">
    <label>Filter by Run:
      <select id="f-run" onchange="applyFilters()">
        <option value="">All</option>{run_opts}
      </select>
    </label>
    <label>Filter by Topic:
      <select id="f-topic" onchange="applyFilters()">
        <option value="">All</option>{topic_opts}
      </select>
    </label>
    <label>Filter by Error Type:
      <select id="f-errtype" onchange="applyFilters()">
        <option value="">All</option>{errtype_opts}
      </select>
    </label>
    <button onclick="resetFilters()">Reset</button>
    <span id="count"></span>
  </div>

  <table id="main-table">
    <thead>
      <tr>
        <th>Review ID</th><th>Run</th><th>Topic</th>
        <th>LLM Text</th><th>GT Text</th>
        <th>TF-IDF</th><th>CE</th><th>Error Type</th>
      </tr>
    </thead>
    <tbody id="tbody">
      {rows_html}
    </tbody>
  </table>

  <script>
    function applyFilters() {{
      const run     = document.getElementById('f-run').value;
      const topic   = document.getElementById('f-topic').value;
      const errtype = document.getElementById('f-errtype').value;
      const rows    = document.querySelectorAll('#tbody tr');
      let visible   = 0;
      rows.forEach(row => {{
        const matchRun  = !run     || row.dataset.run     === run;
        const matchTopic= !topic   || row.dataset.topic   === topic;
        const matchErr  = !errtype || row.dataset.errtype === errtype;
        if (matchRun && matchTopic && matchErr) {{
          row.classList.remove('hidden'); visible++;
        }} else {{
          row.classList.add('hidden');
        }}
      }});
      document.getElementById('count').textContent =
        `Showing ${{visible}} of ${{rows.length}} rows`;
    }}
    function resetFilters() {{
      ['f-run','f-topic','f-errtype'].forEach(id =>
        document.getElementById(id).value = '');
      applyFilters();
    }}
    applyFilters();
  </script>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────────
for llm_csv in LLM_CSVS:
    if not llm_csv.exists():
        print(f"[SKIP] {llm_csv}")
        continue

    model_label = llm_csv.parent.parent.parent.name
    print(f"\n{'#' * 72}")
    print(f"Model: {model_label}")
    print(f"{'#' * 72}")

    df = score_csv(llm_csv)
    decision_analysis(df, model_label)

    for t in THRESHOLDS:
        html = build_html(df, model_label, t)
        t_str = str(t).replace(".", "")
        out = OUT_DIR / f"subtask1.2_error_analysis_{model_label}_t{t_str}.html"
        out.write_text(html, encoding="utf-8")
        print(f"  HTML → {out.relative_to(TASK_DIR)}")

    csv_out = OUT_DIR / f"subtask1.2_scores_{model_label}.csv"
    df.to_csv(csv_out, index=False)
    print(f"  CSV  → {csv_out.relative_to(TASK_DIR)}")

print("\nDone.")
