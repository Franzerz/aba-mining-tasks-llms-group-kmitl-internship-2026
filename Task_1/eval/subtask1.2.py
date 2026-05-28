import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Plus
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Paths & settings
TASK_DIR  = Path(__file__).resolve().parent.parent
GT_CSV    = TASK_DIR / "data" / (
    "Original ABA Dataset for Version 2 "
    "[Oct 23, 2025], Senior Project, MUICT - Sheet2_.csv"
)
LLM_DIR   = TASK_DIR / "outputs" / "task1"
EVAL_DIR  = TASK_DIR / "outputs" / "eval" / "content"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.5
METHODS   = ["Parity", "TF-IDF", "BM25", "Sentence Emb", "Cross-Encoder"]

RULE_SUBTASKS: dict = {
    1: {"1.2"},
    2: {"1.1"},
    3: {"1.3"},
    4: {"1.3"},
    5: {"1.1"},
    6: {"1.1", "1.2"},
    7: {"1.2"},
}


def applicable_subtasks(llm_csv: Path, llm_dir: Path) -> set:
    parts = llm_csv.relative_to(llm_dir).parts
    try:
        folder = parts[list(parts).index("modular") + 1]
    except (ValueError, IndexError):
        return {"1.1", "1.2", "1.3"}
    if folder == "combined":
        return {"1.1", "1.2", "1.3"}
    if folder.startswith("subtask"):
        s = set()
        if "1_1" in folder: s.add("1.1")
        if "1_2" in folder: s.add("1.2")
        if "1_3" in folder: s.add("1.3")
        return s or {"1.1", "1.2", "1.3"}
    if "rule" in folder:
        s = set()
        for n in re.findall(r"\d+", folder):
            s |= RULE_SUBTASKS.get(int(n), set())
        return s or {"1.1", "1.2", "1.3"}
    return {"1.1", "1.2", "1.3"}


_sent_model  = None
_cross_model = None


def _get_sent_model():
    global _sent_model
    if _sent_model is None:
        from sentence_transformers import SentenceTransformer
        print("  [Loading sentence model all-MiniLM-L6-v2 …]")
        _sent_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sent_model


def _get_cross_model():
    global _cross_model
    if _cross_model is None:
        from sentence_transformers import CrossEncoder
        print("  [Loading cross-encoder ms-marco-MiniLM-L-6-v2 …]")
        _cross_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_model


def normalize(text: str) -> str:
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def tokenize(text: str) -> list:
    return re.findall(r"\w+", text.lower())


def tfidf_scores(query: str, candidates: list) -> list:
    try:
        vec = TfidfVectorizer(min_df=1, analyzer="word", ngram_range=(1, 2))
        mat = vec.fit_transform([query] + candidates)
        return cosine_similarity(mat[0:1], mat[1:])[0].tolist()
    except Exception:
        return [0.0] * len(candidates)


def bm25_scores(query: str, candidates: list) -> list:
    try:
        bm25 = BM25Plus([tokenize(c) for c in candidates])
        return bm25.get_scores(tokenize(query)).tolist()
    except Exception:
        return [0.0] * len(candidates)


def sent_emb_scores(query: str, candidates: list) -> list:
    try:
        model = _get_sent_model()
        embs  = model.encode([query] + candidates, convert_to_numpy=True)
        sims  = cosine_similarity(embs[0:1], embs[1:])[0]
        return sims.tolist()
    except Exception:
        return [0.0] * len(candidates)


def cross_enc_scores(query: str, candidates: list) -> list:
    try:
        model  = _get_cross_model()
        pairs  = [(query, c) for c in candidates]
        logits = np.array(model.predict(pairs), dtype=float)
        return (1 / (1 + np.exp(-logits))).tolist()
    except Exception:
        return [0.0] * len(candidates)


def passes_threshold(method: str, scores: list) -> bool:
    """All methods: best score >= 0.5 threshold."""
    if not scores:
        return False
    return max(scores) >= THRESHOLD


# Load ground truth
if not GT_CSV.exists():
    sys.exit(f"[ERROR] Ground truth not found:\n  {GT_CSV}")

gt_raw = pd.read_csv(GT_CSV)
gt_raw = gt_raw.rename(columns={"ID": "Review ID"})
gt_raw = gt_raw[gt_raw["Topic"].notna() & gt_raw["Selected Content"].notna()].copy()
gt_raw["Topic"] = gt_raw["Topic"].str.strip()
gt_raw["norm"]  = gt_raw["Selected Content"].apply(normalize)
gt_raw = gt_raw[gt_raw["norm"] != ""].reset_index(drop=True)

gt_index: dict = {}
for _, row in gt_raw.iterrows():
    key = (row["Review ID"], row["Topic"])
    gt_index.setdefault(key, []).append((row["norm"], row["Selected Content"]))

print(f"Ground truth: {len(gt_raw)} rows, "
      f"{gt_raw['Review ID'].nunique()} unique Review IDs\n")


def _eval_run(run_id, df: pd.DataFrame, workings: list, results: list) -> None:
    W = workings.append
    R = results.append

    _bad = {"parse failed", "topics not found", "topic not found", "(parse failed)"}
    df = df[
        df["Topic"].notna() &
        (df["Topic"].str.strip() != "") &
        (~df["Topic"].str.strip().str.lower().isin(_bad)) &
        df["Text Span"].notna() &
        (df["Text Span"].str.strip() != "")
    ].copy()
    df["Topic"] = df["Topic"].str.strip()
    df["norm"]  = df["Text Span"].apply(normalize)
    df = df[df["norm"] != ""].reset_index(drop=True)

    if df.empty:
        R(f"  [No valid rows for Run {run_id}]")
        return

    passed: dict = {m: [] for m in METHODS}

    W(f"\n  Total rows : {len(df)}")

    for _, row in df.iterrows():
        rid   = row["Review ID"]
        topic = row["Topic"]
        lnorm = row["norm"]
        lcont = row["Text Span"]

        gt_entries = gt_index.get((rid, topic), [])
        gt_norms   = [e[0] for e in gt_entries]
        gt_conts   = [e[1] for e in gt_entries]

        W(f"\n{'─' * 60}")
        W(f"ID={rid}  Topic={topic}")
        W(f"  LLM : {lcont[:100]}")
        W(f"  GT rows for (ID={rid}, Topic={topic}): {len(gt_entries)}")

        if not gt_entries:
            for m in METHODS:
                passed[m].append(False)
            W("  [No GT rows → all methods: NO MATCH]")
            continue

        scores_map = {
            "Parity":        [1.0 if lnorm == gn else 0.0 for gn in gt_norms],
            "TF-IDF":        tfidf_scores(lnorm, gt_norms),
            "BM25":          bm25_scores(lnorm, gt_norms),
            "Sentence Emb":  sent_emb_scores(lnorm, gt_norms),
            "Cross-Encoder": cross_enc_scores(lnorm, gt_norms),
        }

        for m in METHODS:
            passed[m].append(passes_threshold(m, scores_map[m]))

        best_j = int(np.argmax(scores_map["TF-IDF"]))
        W(f"  Best GT : {gt_conts[best_j][:100]}")
        for m in METHODS:
            s   = scores_map[m]
            hit = passes_threshold(m, s)
            W(f"  {m:<14}: {'PASS     ' if hit else 'FAIL     '}  best={max(s):.4f}")

    n = len(df)
    R(f"\n{'─' * 72}")
    R("OVERALL SUMMARY")
    R(f"{'─' * 72}")
    R(f"  Total rows evaluated : {n}")
    R(f"  Threshold            : {THRESHOLD}  (all methods)")
    R(f"\n{'─' * 72}")
    R("PASSING COUNT  (score >= 0.5)")
    R(f"{'─' * 72}")
    R(f"{'Method':<16} {'Passed':>8} {'Total':>7} {'Pass Rate':>10}")
    R("─" * 72)
    for m in METHODS:
        np_ = sum(passed[m])
        rate = np_ / n if n > 0 else 0
        R(f"{m:<16} {np_:>8} {n:>7} {rate:>10.4f}")
    R("─" * 72)


def evaluate(llm_csv: Path) -> None:
    if "1.2" not in applicable_subtasks(llm_csv, LLM_DIR):
        print(f"  [SKIP 1.2] not applicable by rules – {llm_csv.name}")
        return

    raw = pd.read_csv(llm_csv)
    if "Review ID" not in raw.columns and "ID" in raw.columns:
        raw = raw.rename(columns={"ID": "Review ID"})

    if "Errors" in raw.columns:
        raw = raw[raw["Errors"].isna() | (raw["Errors"].astype(str).str.strip().isin({"", "nan"}))].copy()

    if "Topic" not in raw.columns or "Text Span" not in raw.columns:
        print(f"  [SKIP 1.2] missing 'Topic' or 'Text Span' – {llm_csv.name}")
        return

    workings: list = []
    results: list  = []
    W = workings.append
    R = results.append

    W("=" * 72)
    W(f"WORKINGS  –  {llm_csv.name}")
    W("=" * 72)
    W(f"\nThreshold (all methods) : {THRESHOLD}")
    W(f"Total rows in file      : {len(raw)}\n")

    R("=" * 72)
    R(f"EVALUATION RESULTS  –  {llm_csv.name}")
    R("=" * 72)

    run_groups = sorted(raw.groupby("Run")) if "Run" in raw.columns else [(1, raw)]

    for run_id, df in run_groups:
        R(f"\n{'=' * 72}")
        R(f"RUN {run_id}")
        R(f"{'=' * 72}")
        W(f"\n{'=' * 72}")
        W(f"RUN {run_id}")
        W(f"{'=' * 72}")
        _eval_run(run_id, df.reset_index(drop=True), workings, results)

    _write(llm_csv, workings, results)
    print("\n".join(results))


def _stem_parts(llm_csv: Path, base_dir: Path) -> tuple:
    rel    = llm_csv.relative_to(base_dir)
    subdir = Path(*rel.parts[:-1]) if len(rel.parts) > 1 else Path(".")
    stem   = llm_csv.stem
    stem   = re.sub(r"^task\d+_", "", stem)
    stem   = re.sub(r"_extended\d+", "", stem)
    stem   = re.sub(r"_generator", "", stem)
    stem   = re.sub(r"_n\d+$", "", stem)
    return subdir, stem


def _write(llm_csv: Path, workings: list, results: list) -> None:
    subdir, stem = _stem_parts(llm_csv, LLM_DIR)
    out_dir = EVAL_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    r_path = out_dir / f"{stem}_results.txt"
    w_path = out_dir / f"{stem}_workings.txt"
    r_path.write_text("\n".join(results),  encoding="utf-8")
    w_path.write_text("\n".join(workings), encoding="utf-8")
    print(f"  Saved: {r_path.relative_to(TASK_DIR)}")
    print(f"  Saved: {w_path.relative_to(TASK_DIR)}")


# Main – exclude old_output
llm_csvs = sorted(
    p for p in LLM_DIR.rglob("*.csv")
    if "old_output" not in p.parts
)
if not llm_csvs:
    sys.exit(f"[ERROR] No CSV files found under {LLM_DIR}")

print(f"Found {len(llm_csvs)} LLM output file(s):\n")
for p in llm_csvs:
    print(f"  {p.relative_to(TASK_DIR)}")
print()

for llm_csv in llm_csvs:
    print(f"\n{'#' * 72}")
    print(f"Processing: {llm_csv.relative_to(TASK_DIR)}")
    print(f"{'#' * 72}")
    evaluate(llm_csv)

print(f"\n\nAll done. Results in: {EVAL_DIR.relative_to(TASK_DIR)}/")
