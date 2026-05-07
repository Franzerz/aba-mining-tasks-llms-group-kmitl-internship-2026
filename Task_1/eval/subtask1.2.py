import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Plus
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Paths & settings
TASK_DIR = Path(__file__).resolve().parent.parent
GT_CSV   = TASK_DIR / "data" / (
    "Original ABA Dataset for Version 2 "
    "[Oct 23, 2025], Senior Project, MUICT - Sheet2_.csv"
)
LLM_DIR  = TASK_DIR / "outputs" / "task1"
EVAL_DIR = TASK_DIR / "outputs" / "eval" / "content"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.5

METHODS = ["Parity", "TF-IDF", "BM25", "Sentence Emb", "Cross-Encoder"]

# Global topic list (from topics.yaml)
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

# Helpers
def normalize(text: str) -> str:
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def tokenize(text: str) -> list:
    return re.findall(r"\w+", text.lower())


def tfidf_scores(query: str, candidates: list) -> list:
    try:
        vec  = TfidfVectorizer(min_df=1, analyzer="word", ngram_range=(1, 2))
        mat  = vec.fit_transform([query] + candidates)
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
        return (1 / (1 + np.exp(-logits))).tolist()   # sigmoid → [0, 1]
    except Exception:
        return [0.0] * len(candidates)


def is_match(method: str, scores: list) -> bool:
    if not scores:
        return False
    best = max(scores)
    if method == "Parity":
        return best == 1.0
    elif method == "BM25":
        return best > 0
    else:
        return best >= THRESHOLD


# ---------------------------------------------------------------------------
# Load ground truth
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Evaluate one LLM CSV
# ---------------------------------------------------------------------------
def evaluate(llm_csv: Path) -> None:

    raw = pd.read_csv(llm_csv)
    if "Review ID" not in raw.columns and "ID" in raw.columns:
        raw = raw.rename(columns={"ID": "Review ID"})

    for col in ("Review ID", "Topic", "Selected Content"):
        if col not in raw.columns:
            print(f"  [SKIP] missing column '{col}' – {llm_csv.name}")
            return

    raw = raw[
        raw["Topic"].notna() &
        (raw["Topic"].str.strip() != "") &
        (raw["Topic"].str.strip() != "(parse failed)") &
        raw["Selected Content"].notna() &
        (raw["Selected Content"].str.strip() != "")
    ].copy()
    raw["Topic"] = raw["Topic"].str.strip()
    raw["norm"]  = raw["Selected Content"].apply(normalize)
    raw = raw[raw["norm"] != ""].reset_index(drop=True)

    correct: dict = {m: [] for m in METHODS}

    workings = []
    W = workings.append
    W("=" * 72)
    W(f"WORKINGS  –  {llm_csv.name}")
    W("=" * 72)
    W(f"\nThreshold (TF-IDF / Sentence Emb / Cross-Encoder) : {THRESHOLD}")
    W(f"BM25 match : score > 0  (any term overlap)")
    W(f"Total LLM rows : {len(raw)}\n")

    for _, row in raw.iterrows():
        rid    = row["Review ID"]
        topic  = row["Topic"]
        lnorm  = row["norm"]
        lcont  = row["Selected Content"]

        gt_entries = gt_index.get((rid, topic), [])
        gt_norms   = [e[0] for e in gt_entries]
        gt_conts   = [e[1] for e in gt_entries]

        W(f"\n{'─' * 60}")
        W(f"ID={rid}  Topic={topic}")
        W(f"  LLM : {lcont[:100]}")
        W(f"  GT rows for (ID={rid}, Topic={topic}): {len(gt_entries)}")

        if not gt_entries:
            for m in METHODS:
                correct[m].append(False)
            W("  [No GT rows → all methods: NO MATCH]")
            continue

        scores_map = {
            "Parity":       [1.0 if lnorm == gn else 0.0 for gn in gt_norms],
            "TF-IDF":       tfidf_scores(lnorm, gt_norms),
            "BM25":         bm25_scores(lnorm, gt_norms),
            "Sentence Emb": sent_emb_scores(lnorm, gt_norms),
            "Cross-Encoder":cross_enc_scores(lnorm, gt_norms),
        }

        for m in METHODS:
            correct[m].append(is_match(m, scores_map[m]))

        best_j = int(np.argmax(scores_map["TF-IDF"]))
        W(f"  Best GT : {gt_conts[best_j][:100]}")
        for m in METHODS:
            s  = scores_map[m]
            hit = is_match(m, s)
            W(f"  {m:<14}: {'MATCH    ' if hit else 'NO MATCH '}  "
              f"best={max(s):.4f}")

    # -----------------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------------
    n = len(raw)
    results = []
    R = results.append
    R("=" * 72)
    R(f"EVALUATION RESULTS  –  {llm_csv.name}")
    R("=" * 72)
    R(f"\nTotal LLM rows : {n}")
    R(f"Threshold      : {THRESHOLD}  (TF-IDF / Sentence Emb / Cross-Encoder)")
    R(f"BM25 match     : score > 0")
    R(f"\n{'─' * 72}")
    R("OVERALL ACCURACY BY METHOD")
    R(f"{'─' * 72}")
    R(f"{'Method':<16} {'Correct':>8} {'Total':>7} {'Accuracy':>10}")
    R("─" * 72)
    for m in METHODS:
        nc  = sum(correct[m])
        acc = nc / n if n > 0 else 0
        R(f"{m:<16} {nc:>8} {n:>7} {acc:>10.4f}")
    R("─" * 72)

    _write(llm_csv, workings, results)
    print("\n".join(results))


def _short_stem(llm_csv: Path) -> str:
    s = llm_csv.stem
    s = re.sub(r"^task\d+_", "", s)
    s = re.sub(r"_extended\d+", "", s)
    s = re.sub(r"_generator", "", s)
    s = re.sub(r"_n\d+$", "", s)
    return s


def _write(llm_csv: Path, workings: list, results: list) -> None:
    stem   = _short_stem(llm_csv)
    r_path = EVAL_DIR / f"{stem}_results.txt"
    w_path = EVAL_DIR / f"{stem}_workings.txt"
    r_path.write_text("\n".join(results),  encoding="utf-8")
    w_path.write_text("\n".join(workings), encoding="utf-8")
    print(f"  Saved: {r_path.relative_to(TASK_DIR)}")
    print(f"  Saved: {w_path.relative_to(TASK_DIR)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
llm_csvs = sorted(LLM_DIR.rglob("*.csv"))
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
