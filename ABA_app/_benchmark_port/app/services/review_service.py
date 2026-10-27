#### PATCHARAKORN ####

"""Direct port of mu_team's backend/services/reviewService.js.

Same shape in, same shape out — including the detail that a row's `contraries` are sorted by
the count the proposition has under the OPPOSITE sentiment's head claim (descending, then
alphabetically). Kept literal so the Review page can be compared side by side.
"""

from __future__ import annotations

from .. import db
from ..normalizers import TOPIC_TABLES, normalize_sentiment, normalize_topic


def get_review_data(topic_raw: str | None, ui_sentiment: str | None) -> dict:
    topic = normalize_topic(topic_raw or "staff")
    sentiment = normalize_sentiment(ui_sentiment or "positive")

    if not sentiment:
        return {"topic": topic, "sentiment": ui_sentiment, "head": None, "rows": []}

    ctx = db.resolve_topic_context(topic, True)
    if not ctx["supported"] or not ctx["tablesExist"]:
        return {"topic": topic, "sentiment": sentiment, "head": None, "rows": []}
    topic_table, contrary_table = ctx["topicTable"], ctx["contraryTable"]

    head_key = db.fetch_single_head_claim_by_topic_sentiment(topic, sentiment)
    if not head_key:
        return {"topic": topic, "sentiment": sentiment, "head": None, "rows": []}

    opposite_sentiment = "Negative" if sentiment == "Positive" else "Positive"
    opposite_head_key = db.fetch_single_head_claim_by_topic_sentiment(topic, opposite_sentiment)

    main_rows = db.fetch_review_rows_by_claim(topic_table, head_key)
    if not main_rows:
        return {"topic": topic, "sentiment": sentiment, "head": head_key, "rows": []}

    cnt_by_proposition: dict[str, int] = {}
    if opposite_head_key:
        for row in db.fetch_proposition_counts_by_claim(topic_table, opposite_head_key):
            cnt_by_proposition[row["proposition"]] = int(row.get("cnt") or 0)

    assumptions = list(dict.fromkeys(
        r["assumption"] for r in main_rows
        if r.get("assumption") is not None and str(r["assumption"]).strip() != ""
    ))

    contrary_rows = db.fetch_contraries_by_assumptions(contrary_table, assumptions) if assumptions else []

    contrary_map: dict[str, list[dict]] = {}
    for row in contrary_rows:
        contrary_map.setdefault(row["assumption"], []).append({
            "proposition": row["proposition"],
            "cnt": cnt_by_proposition.get(row["proposition"], 0),
        })

    for key, items in contrary_map.items():
        items.sort(key=lambda c: (-int(c.get("cnt") or 0), str(c.get("proposition") or "")))
        contrary_map[key] = items

    rows_out = [
        {
            "proposition": r["proposition"],
            "cnt": r["cnt"],
            "contraries": contrary_map.get(r["assumption"], []),
        }
        for r in main_rows
    ]

    return {"topic": topic, "sentiment": sentiment, "head": head_key, "rows": rows_out}


def get_topic_ratios() -> dict:
    head_rows = db.fetch_head_rows_for_ratios()
    existing_tables = db.get_existing_tables(list(TOPIC_TABLES.values()))

    out: dict[str, dict] = {}
    for row in head_rows:
        topic = row["topic"]
        claim = row["claim"]
        sentiment = row["Sentiment"]

        topic_table = db.get_topic_table(topic)
        if not topic_table:
            continue

        total = db.fetch_claim_total_by_topic_table(topic_table, claim) if topic_table in existing_tables else 0

        entry = out.setdefault(topic, {"posTotal": 0, "negTotal": 0, "posPct": 0, "negPct": 0})
        if sentiment == "Positive":
            entry["posTotal"] += total
        elif sentiment == "Negative":
            entry["negTotal"] += total

    for topic, entry in out.items():
        total = entry["posTotal"] + entry["negTotal"]
        if total > 0:
            # JS Math.round is half-up; Python's round() is banker's rounding, which would
            # differ on exact .5 values — match their behaviour explicitly.
            entry["posPct"] = int((entry["posTotal"] / total) * 100 + 0.5)
            entry["negPct"] = 100 - entry["posPct"]

    return out
