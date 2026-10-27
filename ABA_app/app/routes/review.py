#### PATCHARAKORN ####

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..datasets import DEFAULT_DATASET, get as get_dataset
from ..review_data import (
    dataset_summary,
    framework_literals,
    review_rows,
    topic_ratios,
    topic_to_aspect,
)

router = APIRouter(prefix="/api")


@router.get("/dataset-summary")
def get_dataset_summary(hotel: str = DEFAULT_DATASET) -> dict:
    """Headline dataset figures for the Home dashboard."""
    return dataset_summary(hotel)


@router.get("/topic-ratios")
def get_topic_ratios(hotel: str = DEFAULT_DATASET) -> dict:
    """Per-topic positive/negative split for the Review page's topic cards. Topics without a
    Task 3 framework come back with `aspect: null` rather than being dropped."""
    return {"topics": topic_ratios(hotel), "hotel": hotel}


@router.get("/review-data")
def get_review_data(topic: str, sentiment: str = "Positive",
                    hotel: str = DEFAULT_DATASET) -> dict:
    if sentiment.strip().capitalize() not in {"Positive", "Negative"}:
        raise HTTPException(status_code=400, detail="sentiment must be 'Positive' or 'Negative'")
    rows = review_rows(hotel, topic, sentiment)
    # Which rows can open a graph: the literal has to be one of the framework's atoms. (The
    # voted topics' frameworks come from the Task 3 sets, which drop a few span literals.)
    mapping = topic_to_aspect(hotel)
    aspect = mapping.get(topic) or next(
        (a for t, a in mapping.items() if t.lower() == topic.strip().lower()), None)
    in_framework = framework_literals(hotel, aspect) if aspect else set()
    for row in rows:
        row["in_framework"] = row["supporting"] in in_framework
    return {"topic": topic, "aspect": aspect, "hotel": hotel,
            # False when this hotel's literals cannot be traced to the spans that produced them,
            # so the page shows no count rather than a zero that reads as "never appeared".
            "has_counts": get_dataset(hotel).has_literal_counts,
            "sentiment": sentiment.strip().capitalize(), "rows": rows}
