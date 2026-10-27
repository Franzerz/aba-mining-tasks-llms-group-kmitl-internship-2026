#### PATCHARAKORN ####

"""
Reads the Task 1 gold CSV to drive the Home dashboard and the Review page.

Every number this module returns is counted straight from the gold file — nothing is
hardcoded, estimated, or padded. Where the data is thin (Check-out has only 8 rows), the
Review page shows that honestly rather than hiding the topic.

Gold file shape (one row per mined (review, topic, sentiment) span, i.e. one opinion):
    ID (the review), Title, PositiveReview, NegativeReview, Topic, Selected Content, Pos/Neg,
    Head, Body 1..Body 15, Cont. Body 1..Cont. Body 15, Concat

`Body N` is a mined body literal (e.g. `easy_check-in`); the `Cont. Body N` in the SAME
position is its auto-derived contrary (e.g. `no_evident_not_easy_check-in`) — the positional
pairing is what makes the Review page's Supporting/Contrary columns line up.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path

import pandas as pd

from . import datasets
from .contrary_sources import (
    NEG_PREFIX,
    POS_PREFIX,
    available_aspects,
    no_pair_topics,
    sources_for,
    topic_key,
    vote_literals,
)
from .datasets import DEFAULT_DATASET, GOLD_WIDE

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


def _read_gold_wide(path: Path) -> pd.DataFrame:
    """The annotated layout: one row per span, literals in Body N / Cont. Body N column pairs."""
    df = pd.read_csv(path, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]

    for col in ["ID", "Topic", "Pos/Neg", "Selected Content", "Head", "Title",
                "PositiveReview", "NegativeReview"]:
        if col not in df.columns:
            raise ValueError(f"Task 1 gold CSV is missing expected column '{col}'. "
                             f"Found: {list(df.columns)[:20]}")
        df[col] = df[col].map(_clean)

    for i in range(1, MAX_BODY_SLOTS + 1):
        for prefix in ("Body", "Cont. Body"):
            col = f"{prefix} {i}"
            df[col] = df[col].map(_clean) if col in df.columns else ""
    df["has_text"] = (df["Title"] != "") | (df["PositiveReview"] != "") | (df["NegativeReview"] != "")
    return df


def _read_task1_sequence(path: Path) -> pd.DataFrame:
    """A Task 1 run's output: one row per span, with no literals and no review text.

    It is reshaped into the same frame the annotated layout produces so everything downstream is
    identical. The Body / Cont. Body slots stay empty, because this layout has no literals at
    all -- for such a hotel they come from its vote files instead (see topic_literals).
    """
    df = pd.read_csv(path, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    rename = {"Review ID": "ID", "Sentiment": "Pos/Neg"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    for col in ["ID", "Topic", "Pos/Neg", "Selected Content"]:
        if col not in df.columns:
            raise ValueError(f"Task 1 spans CSV is missing expected column '{col}'. "
                             f"Found: {list(df.columns)[:20]}")
        df[col] = df[col].map(_clean)

    sentiment = df["Pos/Neg"].str.strip().str.capitalize()
    polarity = sentiment.map({"Positive": "good", "Negative": "bad"}).fillna("")
    df["Head"] = [f"{p}_{topic_key(t)}" if p and t else "" for p, t in zip(polarity, df["Topic"])]
    for col in ["Title", "PositiveReview", "NegativeReview"]:
        df[col] = ""
    for i in range(1, MAX_BODY_SLOTS + 1):
        df[f"Body {i}"] = ""
        df[f"Cont. Body {i}"] = ""
    # There is no review text in this layout, so a review counts as present when it produced a
    # span. That makes "reviews" mean reviews that yielded at least one opinion, which is what
    # the dashboard reports for it.
    df["has_text"] = df["Selected Content"] != ""
    return df


@functools.lru_cache(maxsize=8)
def _read_all(dataset: str = DEFAULT_DATASET) -> pd.DataFrame:
    """Every row of this hotel's span file, cleaned but not filtered."""
    ds = datasets.get(dataset)
    if ds.spans_format == GOLD_WIDE:
        return _read_gold_wide(ds.spans_csv)
    return _read_task1_sequence(ds.spans_csv)


def _raw_row_count(dataset: str = DEFAULT_DATASET) -> int:
    """Rows in the span file before any filtering — so the UI can say how many were excluded
    rather than quietly presenting the filtered number as the dataset size."""
    return int(len(_read_all(dataset)))


@functools.lru_cache(maxsize=8)
def _load(dataset: str = DEFAULT_DATASET) -> pd.DataFrame:
    """The opinion rows: a hotel topic and a sentiment (off-topic and empty rows dropped)."""
    df = _read_all(dataset)
    df = df[df["Topic"] != ""]
    df = df[~df["Topic"].str.lower().isin(OFF_TOPIC_LABELS)]  # "Off" = not about the hotel
    df = df.assign(Sentiment=df["Pos/Neg"].str.capitalize())
    return df[df["Sentiment"].isin(["Positive", "Negative"])].reset_index(drop=True)


def all_review_ids(dataset: str = DEFAULT_DATASET) -> set[str]:
    """Every review in the dataset: the IDs of rows that carry review text. (The annotated
    file's trailing empty row has an ID but no review.)"""
    df = _read_all(dataset)
    return set(df.loc[df["has_text"] & (df["ID"] != ""), "ID"])


def topic_keys(dataset: str = DEFAULT_DATASET) -> list[str]:
    """Aspect keys of every topic in this hotel's spans."""
    return sorted({topic_key(t) for t in _load(dataset)["Topic"].unique()})


@functools.lru_cache(maxsize=64)
def topic_literals(dataset: str, topic: str) -> dict[str, set[str]]:
    """{"Positive": {...}, "Negative": {...}}: the distinct body literals of one topic.

    They come from the spans when the spans carry them, and from the topic's vote files when
    they do not.
    """
    ds = datasets.get(dataset)
    if ds.spans_format != GOLD_WIDE:
        return vote_literals(dataset, topic_key(topic or ""))
    subset = _load(dataset)
    subset = subset[subset["Topic"].str.lower() == topic.strip().lower()]
    out: dict[str, set[str]] = {"Positive": set(), "Negative": set()}
    for sentiment, group in subset.groupby("Sentiment"):
        for i in range(1, MAX_BODY_SLOTS + 1):
            out[sentiment] |= {v for v in group[f"Body {i}"] if v}
    return out


def framework_literals(dataset: str, aspect: str) -> set[str]:
    """The literals a topic's framework is built over — what a Review row can be focused on."""
    pn = sources_for(dataset, aspect)["pn"]
    if pn is not None:
        return set(pn.rows["literal"]) | set(pn.rows["attacker"])
    if aspect in no_pair_topics(dataset):
        lits = topic_literals(dataset, topic_label(dataset, aspect) or "")
        return lits["Positive"] | lits["Negative"]
    return set()


def topic_label(dataset: str, aspect: str) -> str | None:
    """Graph aspect key -> the topic label used in the span file ("taxi-issue" -> "Taxi-issue")."""
    return next((t for t in _load(dataset)["Topic"].unique() if topic_key(t) == aspect), None)


def topic_to_aspect(dataset: str = DEFAULT_DATASET) -> dict[str, str]:
    """Topic label -> aspect key, for every topic that can open a Graph Studio right now."""
    available = set(available_aspects(dataset))
    return {t: topic_key(t) for t in sorted(_load(dataset)["Topic"].unique())
            if topic_key(t) in available}


def _review_ids(df: pd.DataFrame) -> pd.Series:
    """One id per source review: the dataset's own review ID."""
    return df["ID"]


def topic_ratios(dataset: str = DEFAULT_DATASET) -> list[dict]:
    """Per-topic counts for the Review page's topic cards and the Home page's statistics.

    `positive` / `negative` / `total` count opinions (rows: one review's praise or complaint
    about one topic). The `*_reviews` fields count distinct reviews instead — a review that both
    praises and complains about a topic is in both, once each, and in `both_reviews`.
    """
    df = _load(dataset)
    ids = _review_ids(df)
    all_reviews = len(all_review_ids(dataset))
    mapping = topic_to_aspect(dataset)
    out: list[dict] = []
    for topic, group in df.groupby("Topic"):
        pos = int((group["Sentiment"] == "Positive").sum())
        neg = int((group["Sentiment"] == "Negative").sum())
        total = pos + neg
        gid = ids.loc[group.index]
        praised = set(gid[group["Sentiment"] == "Positive"])
        complained = set(gid[group["Sentiment"] == "Negative"])
        reviews = len(praised | complained)
        out.append({
            "topic": topic,
            "positive": pos,
            "negative": neg,
            "total": total,
            "positive_pct": round(pos / total * 100) if total else 0,
            "negative_pct": round(neg / total * 100) if total else 0,
            "reviews": reviews,
            "reviews_pct": round(reviews / all_reviews * 100, 1) if all_reviews else 0,
            "positive_reviews": len(praised),
            "negative_reviews": len(complained),
            "both_reviews": len(praised & complained),
            "aspect": mapping.get(topic),  # None => no framework for this topic yet
            # "human" = voted contraries, "llm" = predicted ones, None = no framework
            "contrary_source": _pn_source_kind(dataset, mapping.get(topic)),
        })
    return sorted(out, key=lambda r: r["total"], reverse=True)


def _pn_source_kind(dataset: str, aspect: str | None) -> str | None:
    if not aspect:
        return None
    pn = sources_for(dataset, aspect)["pn"]
    return pn.kind if pn is not None else "none"   # "none": a topic with no possible pair


def _missing_ids(ids: list[str]) -> list[int]:
    numbers = sorted({int(i) for i in ids if i.isdigit()})
    if not numbers:
        return []
    return sorted(set(range(numbers[0], numbers[-1] + 1)) - set(numbers))


def dataset_summary(dataset: str = DEFAULT_DATASET) -> dict:
    """Headline figures for the Home dashboard."""
    ds = datasets.get(dataset)
    df = _load(dataset)
    pos = int((df["Sentiment"] == "Positive").sum())
    neg = int((df["Sentiment"] == "Negative").sum())
    literals = _all_body_literals(dataset)
    raw = _read_all(dataset)
    # Rows are opinions; one review is mined into several of them. Reviews are counted by the
    # dataset's own ID. Verified against the file: IDs run 1-785 but 520 and 563 have no rows
    # (in the raw hotel file each is a single off-topic row whose text is a spreadsheet error),
    # and the last row is empty — so 783 reviews, 744 of which give at least one opinion.
    ids = sorted(all_review_ids(dataset), key=lambda i: int(i) if i.isdigit() else 0)
    off_topic = int((raw["Topic"].str.lower().isin(OFF_TOPIC_LABELS)).sum())
    return {
        "rows": int(len(df)),
        "reviews": len(ids),
        "reviews_with_opinion": int(df["ID"].nunique()),
        "review_id_range": [ids[0], ids[-1]] if ids else None,
        # numbers inside that range with no row at all, so "785" vs "783" is never a mystery
        "missing_review_ids": _missing_ids(ids),
        "raw_rows": int(len(raw)),
        "excluded_rows": int(len(raw)) - int(len(df)),
        "off_topic_rows": off_topic,
        "empty_rows": int((~raw["has_text"]).sum()),
        "dataset": ds.spans_csv.stem,
        "hotel": ds.key,
        "hotel_label": ds.label,
        # False when a literal cannot be traced back to the spans that produced it, so the
        # Review page shows no occurrence count for this hotel instead of inventing one.
        "has_literal_counts": ds.has_literal_counts,
        "positive": pos,
        "negative": neg,
        "positive_pct": round(pos / (pos + neg) * 100) if pos + neg else 0,
        "negative_pct": round(neg / (pos + neg) * 100) if pos + neg else 0,
        "topics": int(df["Topic"].nunique()),
        "body_literals": len(literals),
        "topics_with_framework": sorted(topic_to_aspect(dataset)),
    }


@functools.lru_cache(maxsize=8)
def _all_body_literals(dataset: str = DEFAULT_DATASET) -> frozenset[str]:
    ds = datasets.get(dataset)
    if ds.spans_format != GOLD_WIDE:
        # No literals in the spans; this hotel's are the ones its vote files name.
        out: set[str] = set()
        for topic in _load(dataset)["Topic"].unique():
            lits = topic_literals(dataset, topic)
            out |= lits["Positive"] | lits["Negative"]
        return frozenset(out)
    df = _load(dataset)
    return frozenset(v for i in range(1, MAX_BODY_SLOTS + 1) for v in df[f"Body {i}"] if v)


def review_rows(dataset: str, topic: str, sentiment: str) -> list[dict]:
    """The Supporting / Count / Contrary / Detail rows for one (topic, sentiment) panel.

    One entry per distinct body literal, with how many mined spans produced it and the review
    snippets it came from, so a reader can trace any literal back to real review text.
    Copies, so a caller can annotate them without touching the cache.
    """
    return [dict(row, details=list(row["details"]))
            for row in _review_rows(dataset, topic.strip().lower(),
                                    sentiment.strip().capitalize())]


def _vote_review_rows(dataset: str, topic: str, sentiment: str) -> tuple[dict, ...]:
    """Review rows for a hotel whose literals come from its vote files.

    A vote file names a literal but not the spans it was mined from, so there is no occurrence
    count and no review snippet to show: count stays 0 and the UI reads that as "no count for
    this hotel" rather than "appeared zero times".
    """
    aspect = topic_key(topic)
    lits = vote_literals(dataset, aspect)[sentiment]
    prefix = POS_PREFIX if sentiment == "Positive" else NEG_PREFIX
    head = f"{'good' if sentiment == 'Positive' else 'bad'}_{aspect}"
    return tuple({
        "supporting": literal,
        "contrary": prefix + literal,
        "head": head,
        "count": 0,
        "details": [],
    } for literal in sorted(lits))


@functools.lru_cache(maxsize=128)
def _review_rows(dataset: str, topic: str, sentiment: str) -> tuple[dict, ...]:
    ds = datasets.get(dataset)
    if ds.spans_format != GOLD_WIDE:
        return _vote_review_rows(dataset, topic, sentiment)
    df = _load(dataset)
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
    return tuple(rows)
