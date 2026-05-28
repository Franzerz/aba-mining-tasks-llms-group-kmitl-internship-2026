"""
Similarity Score Distribution – Subtask 1.2 (Selected Content / Text Span).

Plots the distribution of best-match similarity scores for TF-IDF,
Sentence Embeddings, and Cross-Encoder to justify threshold selection.
Scoring is fully batched for speed.
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TASK_DIR = Path(__file__).resolve().parent.parent
GT_CSV   = Path("/Users/Pngoendee/ReaLearn/Dataset") / (
    "Original ABA Dataset for Version 2 "
    "[Oct 23, 2025], Senior Project, MUICT - Sheet2_.csv"
)
LLM_CSVS = [
    TASK_DIR / "outputs" / "task1" / "llama4_scout" / "modular" / "subtask1_2" /
    "task1_llama4_scout_extended11_subtask1_2.csv",
    TASK_DIR / "outputs" / "task1" / "llama3.2" / "modular" / "subtask1_2" /
    "task1_llama3.2_extended11_subtask1_2.csv",
]
OUT_DIR  = TASK_DIR / "outputs" / "eval" / "content"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["TF-IDF", "Sentence Emb", "Cross-Encoder"]
COLORS  = {"TF-IDF": "#2196F3", "Sentence Emb": "#4CAF50", "Cross-Encoder": "#FF5722"}

# ── lazy model loaders ──────────────────────────────────────────────────────
_sent_model  = None
_cross_model = None


def _get_sent_model():
    global _sent_model
    if _sent_model is None:
        from sentence_transformers import SentenceTransformer
        print("  [Loading sentence model …]")
        _sent_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sent_model


def _get_cross_model():
    global _cross_model
    if _cross_model is None:
        from sentence_transformers import CrossEncoder
        print("  [Loading cross-encoder …]")
        _cross_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_model


# ── text helpers ─────────────────────────────────────────────────────────────
def normalize(text) -> str:
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def tfidf_best(query: str, candidates: list) -> float:
    try:
        vec = TfidfVectorizer(min_df=1, analyzer="word", ngram_range=(1, 2))
        mat = vec.fit_transform([query] + candidates)
        return float(cosine_similarity(mat[0:1], mat[1:])[0].max())
    except Exception:
        return 0.0


# ── ground truth ─────────────────────────────────────────────────────────────
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
    gt_index.setdefault(key, []).append(row["norm"])


# ── batched scoring ───────────────────────────────────────────────────────────
def collect_scores(llm_csv: Path) -> dict[str, list[float]]:
    raw = pd.read_csv(llm_csv)
    if "Review ID" not in raw.columns and "ID" in raw.columns:
        raw = raw.rename(columns={"ID": "Review ID"})
    if "Errors" in raw.columns:
        raw = raw[raw["Errors"].isna() | (raw["Errors"].astype(str).str.strip().isin({"", "nan"}))].copy()

    _bad = {"parse failed", "topics not found", "topic not found", "(parse failed)"}
    raw = raw[
        raw["Topic"].notna() &
        (raw["Topic"].str.strip() != "") &
        (~raw["Topic"].str.strip().str.lower().isin(_bad)) &
        raw["Text Span"].notna() &
        (raw["Text Span"].str.strip() != "")
    ].copy()
    raw["Topic"] = raw["Topic"].str.strip()
    raw["norm"]  = raw["Text Span"].apply(normalize)
    raw = raw[raw["norm"] != ""].reset_index(drop=True)

    n = len(raw)
    print(f"  {n} valid rows")

    rows_data = []
    for _, row in raw.iterrows():
        gt_norms = gt_index.get((row["Review ID"], row["Topic"]), [])
        rows_data.append((row["norm"], gt_norms))

    best = {m: [0.0] * n for m in METHODS}

    # TF-IDF (fast per-row)
    print("  TF-IDF …")
    for i, (lnorm, gt_norms) in enumerate(rows_data):
        if gt_norms:
            best["TF-IDF"][i] = tfidf_best(lnorm, gt_norms)

    # Sentence Embeddings – encode all unique texts once
    print("  Sentence Embeddings (batched) …")
    unique_texts = list({t for lnorm, gt_norms in rows_data for t in [lnorm] + gt_norms})
    t2i = {t: i for i, t in enumerate(unique_texts)}
    sent_model = _get_sent_model()
    all_embs = sent_model.encode(
        unique_texts, convert_to_numpy=True, batch_size=256, show_progress_bar=True
    )
    for i, (lnorm, gt_norms) in enumerate(rows_data):
        if gt_norms:
            q  = all_embs[t2i[lnorm]].reshape(1, -1)
            cs = np.array([all_embs[t2i[g]] for g in gt_norms])
            best["Sentence Emb"][i] = float(cosine_similarity(q, cs)[0].max())

    # Cross-Encoder – predict all pairs in one batch
    print("  Cross-Encoder (batched) …")
    all_pairs, pair_meta = [], []
    for i, (lnorm, gt_norms) in enumerate(rows_data):
        for g in gt_norms:
            all_pairs.append((lnorm, g))
            pair_meta.append(i)

    if all_pairs:
        ce_model = _get_cross_model()
        logits = np.array(
            ce_model.predict(all_pairs, batch_size=256, show_progress_bar=True), dtype=float
        )
        probs = 1.0 / (1.0 + np.exp(-logits))
        for k, i in enumerate(pair_meta):
            if probs[k] > best["Cross-Encoder"][i]:
                best["Cross-Encoder"][i] = float(probs[k])

    return best


# ── plot ───────────────────────────────────────────────────────────────────────
THRESHOLD = 0.5
BINS      = 40


def plot_score_distribution(
    scores_per_method: dict[str, list[float]],
    label: str,
    out_path: Path,
) -> None:
    n = len(next(iter(scores_per_method.values())))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Similarity Score Distribution – Subtask 1.2\n{label}  (n = {n})",
        fontsize=13,
        fontweight="bold",
    )

    for ax, method in zip(axes, METHODS):
        s = np.array(scores_per_method[method])

        counts, edges, patches = ax.hist(
            s, bins=BINS, range=(0.0, 1.0),
            color=COLORS[method], alpha=0.75, edgecolor="white",
        )

        # KDE overlay
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(s, bw_method=0.1)
            xs  = np.linspace(0, 1, 300)
            ax2 = ax.twinx()
            ax2.plot(xs, kde(xs), color=COLORS[method], linewidth=2, alpha=0.9)
            ax2.set_ylabel("Density", fontsize=9, color="grey")
            ax2.tick_params(axis="y", labelcolor="grey", labelsize=8)
            ax2.set_ylim(bottom=0)
        except ImportError:
            pass

        # Threshold line + annotation
        pct_above = 100 * (s >= THRESHOLD).mean()
        ax.axvline(
            THRESHOLD, color="red", linestyle="--", linewidth=2,
            label=f"Threshold {THRESHOLD}  ({pct_above:.1f}% pass)",
        )

        # Mean & median
        ax.axvline(np.mean(s),   color="black",  linestyle=":",  linewidth=1.2, label=f"Mean {np.mean(s):.2f}")
        ax.axvline(np.median(s), color="purple", linestyle="-.", linewidth=1.2, label=f"Median {np.median(s):.2f}")

        ax.set_title(method, fontsize=12, fontweight="bold")
        ax.set_xlabel("Similarity Score", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.relative_to(TASK_DIR)}")


# ── main ──────────────────────────────────────────────────────────────────────
for llm_csv in LLM_CSVS:
    if not llm_csv.exists():
        print(f"[SKIP] not found: {llm_csv}")
        continue

    label    = llm_csv.parent.parent.parent.name
    out_path = OUT_DIR / f"subtask1.2_score_distribution_{label}.png"

    print(f"\n{'#' * 60}")
    print(f"Processing: {label}")
    print(f"{'#' * 60}")

    scores = collect_scores(llm_csv)
    plot_score_distribution(scores, label, out_path)

print("\nDone.")
