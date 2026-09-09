#### PATCHARAKORN ####

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..review_data import dataset_summary, review_rows, topic_ratios

router = APIRouter(prefix="/api")


@router.get("/dataset-summary")
def get_dataset_summary() -> dict:
    """Headline gold-dataset figures for the Home dashboard."""
    return dataset_summary()


@router.get("/topic-ratios")
def get_topic_ratios() -> dict:
    """Per-topic positive/negative split for the Review page's topic cards. Topics without a
    Task 3 framework come back with `aspect: null` rather than being dropped."""
    return {"topics": topic_ratios()}


@router.get("/review-data")
def get_review_data(topic: str, sentiment: str = "Positive") -> dict:
    if sentiment.strip().capitalize() not in {"Positive", "Negative"}:
        raise HTTPException(status_code=400, detail="sentiment must be 'Positive' or 'Negative'")
    rows = review_rows(topic, sentiment)
    return {"topic": topic, "sentiment": sentiment.strip().capitalize(), "rows": rows}
