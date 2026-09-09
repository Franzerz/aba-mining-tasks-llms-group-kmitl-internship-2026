#### PATCHARAKORN ####

"""
Reads the Task 1 gold CSV to drive the Home dashboard and the Review page.

Every number this module returns is counted straight from the gold file — nothing is
hardcoded, estimated, or padded. Where the data is thin (Check-out has only 8 rows), the
Review page shows that honestly rather than hiding the topic.

Gold file shape (one row per mined (review, topic, sentiment) span):
    Title, PositiveReview, NegativeReview, Topic, Selected Content, Pos/Neg, Head,
    Body 1..Body 15, Cont. Body 1..Cont. Body 15, Claim, ...

`Body N` is a mined body literal (e.g. `easy_check-in`); the `Cont. Body N` in the SAME
position is its auto-derived contrary (e.g. `no_evident_not_easy_check-in`) — the positional
pairing is what makes the Review page's Supporting/Contrary columns line up.
"""

from __future__ import annotations

import functools
import math

import pandas as pd

from .config import TASK1_GOLD_CSV, TOPIC_TO_ASPECT

MAX_BODY_SLOTS = 15
OFF_TOPIC_LABELS = {"off", "off-topic", "offtopic"}


def _clean(value: object) -> str:
    """Gold cells arrive as str|NaN|float; normalize to a trimmed string ('' when missing)."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


@functools.lru_cache(maxsize=1)
def _raw_row_count() -> int:
    """Rows in the gold file before any filtering — so the UI can say how many were excluded
    rather than quietly presenting the filtered number as the dataset size."""
    return int(len(pd.read_csv(TASK1_GOLD_CSV, dtype=str)))


@functools.lru_cache(maxsize=1)
def _load() -> pd.DataFrame:
    """Loads and normalizes the gold CSV once per process (it never changes while running)."""
    df = pd.read_csv(TASK1_GOLD_CSV, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]

    for col in ["Topic", "Pos/Neg", "Selected Content", "Head", "Title",
                "PositiveReview", "NegativeReview"]:
        if col not in df.columns:
            raise ValueError(f"Task 1 gold CSV is missing expected column '{col}'. "
                             f"Found: {list(df.columns)[:20]}")
        df[col] = df[col].map(_clean)

    for i in range(1, MAX_BODY_SLOTS + 1):
        for prefix in ("Body", "Cont. Body"):
            col = f"{prefix} {i}"
            df[col] = df[col].map(_clean) if col in df.columns else ""

    df = df[df["Topic"] != ""]
    df = df[~df["Topic"].str.lower().isin(OFF_TOPIC_LABELS)]  # "Off" = not about the hotel
    df["Sentiment"] = df["Pos/Neg"].str.capitalize()
    return df[df["Sentiment"].isin(["Positive", "Negative"])].reset_index(drop=True)


def topic_ratios() -> list[dict]:
    """Per-topic positive/negative counts + percentages, for the Review page's topic cards."""
    df = _load()
    out: list[dict] = []
    for topic, group in df.groupby("Topic"):
        pos = int((group["Sentiment"] == "Positive").sum())
        neg = int((group["Sentiment"] == "Negative").sum())
        total = pos + neg
        out.append({
            "topic": topic,
            "positive": pos,
            "negative": neg,
            "total": total,
            "positive_pct": round(pos / total * 100) if total else 0,
            "negative_pct": round(neg / total * 100) if total else 0,
            "aspect": TOPIC_TO_ASPECT.get(topic),  # None => no Task 3 framework for this topic
        })
    return sorted(out, key=lambda r: r["total"], reverse=True)


def dataset_summary() -> dict:
    """Headline figures for the Home dashboard."""
    df = _load()
    pos = int((df["Sentiment"] == "Positive").sum())
    neg = int((df["Sentiment"] == "Negative").sum())
    literals = {
        _clean(row[f"Body {i}"])
        for _, row in df.iterrows()
        for i in range(1, MAX_BODY_SLOTS + 1)
        if _clean(row[f"Body {i}"])
    }
    raw = _raw_row_count()
    return {
        "rows": int(len(df)),
        "raw_rows": raw,
        "excluded_rows": raw - int(len(df)),
        "positive": pos,
        "negative": neg,
        "positive_pct": round(pos / (pos + neg) * 100) if pos + neg else 0,
        "negative_pct": round(neg / (pos + neg) * 100) if pos + neg else 0,
        "topics": int(df["Topic"].nunique()),
        "body_literals": len(literals),
        "topics_with_framework": sorted(TOPIC_TO_ASPECT),
    }


def review_rows(topic: str, sentiment: str) -> list[dict]:
    """The Supporting / Count / Contrary / Detail rows for one (topic, sentiment) panel.

    One entry per distinct body literal, with how many mined spans produced it and the review
    snippets it came from, so a reader can trace any literal back to real review text.
    """
    df = _load()
    subset = df[(df["Topic"].str.lower() == topic.strip().lower())
                & (df["Sentiment"] == sentiment.strip().capitalize())]

    by_literal: dict[str, dict] = {}
    for _, row in subset.iterrows():
        for i in range(1, MAX_BODY_SLOTS + 1):
            literal = _clean(row[f"Body {i}"])
            if not literal:
                continue
            entry = by_literal.setdefault(literal, {
                "supporting": literal,
                "contrary": _clean(row[f"Cont. Body {i}"]),
                "head": _clean(row["Head"]),
                "count": 0,
                "details": [],
            })
            entry["count"] += 1
            if not entry["contrary"]:
                entry["contrary"] = _clean(row[f"Cont. Body {i}"])
            span = _clean(row["Selected Content"])
            if span and span not in entry["details"]:
                entry["details"].append(span)

    rows = sorted(by_literal.values(), key=lambda r: (-r["count"], r["supporting"]))
    for row in rows:
        row["details"] = row["details"][:8]  # keep payloads small; the panel shows a few
    return rows
