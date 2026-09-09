#### PATCHARAKORN ####

"""Direct port of mu_team's backend/utils/normalizers.js.

Kept deliberately literal — including the four-topic limit — so the benchmark compares the
same inputs. Facility is absent here for the same reason it is absent there: their backend
never mapped it.
"""

from __future__ import annotations

TOPIC_TABLES = {
    "check-in": "check_in",
    "check_in": "check_in",
    "check-out": "check_out",
    "check_out": "check_out",
    "staff": "staff",
    "price": "price",
}


def normalize_topic(raw: object) -> str:
    return str(raw or "").strip().lower()


def normalize_sentiment(ui_value: object) -> str | None:
    value = str(ui_value or "").lower()
    if value == "negative":
        return "Negative"
    if value == "positive":
        return "Positive"
    return None


def normalize_sentiment_or_all(ui_value: object) -> str | None:
    value = str(ui_value or "").strip().lower()
    if value == "all":
        return "All"
    if value == "negative":
        return "Negative"
    if value == "positive":
        return "Positive"
    return None


def get_head_claim(row: dict | None) -> str | None:
    if not isinstance(row, dict):
        return None
    return row.get("claim") or row.get("Head") or row.get("head") or None


def classify_atom_type(raw: object, fallback_type: str) -> str:
    """no_evident_* reads as an assumption, have_evident_* as a proposition — mu_team's
    convention, matching how Task 2 names auto-derived literals."""
    lower = str(raw or "").lower()
    if lower.startswith("no_evident_"):
        return "assumption"
    if lower.startswith("have_evident_"):
        return "proposition"
    return fallback_type
