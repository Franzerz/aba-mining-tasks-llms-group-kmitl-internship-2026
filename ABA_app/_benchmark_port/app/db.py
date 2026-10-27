#### PATCHARAKORN ####

"""Direct port of mu_team's backend/db/queries.js, against the same MySQL schema.

Every query below runs the same SQL their Node app runs (same GROUP BY, same ORDER BY, same
tie-breaks), so both apps see identical rows in identical order — a prerequisite for the two
graphs being comparable.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable

import pymysql
from pymysql.cursors import DictCursor

from . import settings
from .normalizers import TOPIC_TABLES


def _connect() -> pymysql.connections.Connection:
    return pymysql.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        database=settings.DB_NAME,
        cursorclass=DictCursor,
        autocommit=True,
        charset="utf8mb4",
    )


@contextmanager
def cursor():
    """Connection-per-call. The app is read-only and low-traffic, so this is plenty without
    pulling in a pool or ORM."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            yield cur
    finally:
        conn.close()


def query(sql: str, params: Iterable[Any] | None = None) -> list[dict]:
    with cursor() as cur:
        cur.execute(sql, tuple(params or ()))
        return list(cur.fetchall() or [])


def health() -> bool:
    try:
        rows = query("SELECT 1 AS ok")
        return bool(rows and rows[0]["ok"] == 1)
    except Exception:
        return False


def get_topic_table(topic: str) -> str | None:
    return TOPIC_TABLES.get(topic)


def get_contrary_table(topic: str) -> str | None:
    base = get_topic_table(topic)
    return f"contrary_{base}" if base else None


def get_existing_tables(table_names: Iterable[str]) -> set[str]:
    unique = list(dict.fromkeys(n for n in table_names if n))
    if not unique:
        return set()
    placeholders = ", ".join(["%s"] * len(unique))
    rows = query(
        "SELECT TABLE_NAME FROM information_schema.tables "
        f"WHERE table_schema = %s AND TABLE_NAME IN ({placeholders})",
        [settings.DB_NAME, *unique],
    )
    return {str(r["TABLE_NAME"] or "") for r in rows}


def resolve_topic_context(topic: str, require_contrary: bool = True) -> dict:
    topic_table = get_topic_table(topic)
    contrary_table = get_contrary_table(topic)
    if not topic_table or (require_contrary and not contrary_table):
        return {"supported": False, "tablesExist": False,
                "topicTable": topic_table, "contraryTable": contrary_table}
    needed = [topic_table, contrary_table] if require_contrary else [topic_table]
    existing = get_existing_tables(needed)
    has_topic = topic_table in existing
    has_contrary = (contrary_table in existing) if require_contrary else True
    return {"supported": True, "tablesExist": has_topic and has_contrary,
            "topicTable": topic_table, "contraryTable": contrary_table}


def fetch_head_claims_by_topic(topic: str, sentiment_or_all: str = "All",
                               limit: int | None = None) -> list[dict]:
    params: list[Any] = [topic]
    where = "LOWER(Topic) = %s"
    if sentiment_or_all != "All":
        where += " AND Sentiment = %s"
        params.append(sentiment_or_all)
    limit_sql = "LIMIT %s" if limit is not None else ""
    if limit is not None:
        params.append(int(limit))
    return query(f"SELECT * FROM head WHERE {where} {limit_sql}", params)


def fetch_single_head_claim_by_topic_sentiment(topic: str, sentiment: str) -> str | None:
    rows = query(
        "SELECT claim FROM head WHERE LOWER(Topic) = %s AND Sentiment = %s LIMIT 1",
        [topic, sentiment],
    )
    return rows[0]["claim"] if rows else None


def fetch_top_assumptions_by_claim(topic_table: str, claim: str,
                                   limit: int | None = None) -> list[dict]:
    params: list[Any] = [claim]
    limit_sql = "LIMIT %s" if limit is not None else ""
    if limit is not None:
        params.append(int(limit))
    return query(
        "SELECT assumption, MAX(cnt) AS cnt "
        f"FROM `{topic_table}` WHERE claim = %s GROUP BY assumption "
        f"ORDER BY cnt DESC, assumption ASC {limit_sql}",
        params,
    )


def fetch_top_propositions_by_claim(topic_table: str, claim: str,
                                    limit: int | None = None) -> list[dict]:
    params: list[Any] = [claim]
    limit_sql = "LIMIT %s" if limit is not None else ""
    if limit is not None:
        params.append(int(limit))
    return query(
        "SELECT proposition, MAX(cnt) AS cnt "
        f"FROM `{topic_table}` WHERE claim = %s GROUP BY proposition "
        f"ORDER BY cnt DESC, proposition ASC {limit_sql}",
        params,
    )


def fetch_assumptions_attacking_propositions(topic_table: str, contrary_table: str, claim: str,
                                             proposition_list: list[str],
                                             limit: int | None = None) -> list[dict]:
    if not proposition_list:
        return []
    placeholders = ", ".join(["%s"] * len(proposition_list))
    params: list[Any] = [claim, *proposition_list]
    limit_sql = "LIMIT %s" if limit is not None else ""
    if limit is not None:
        params.append(int(limit))
    return query(
        "SELECT a.assumption, MAX(a.cnt) AS cnt "
        f"FROM `{topic_table}` a "
        f"JOIN `{contrary_table}` c ON c.assumption = a.assumption AND c.isContrary = 1 "
        f"WHERE a.claim = %s AND c.proposition IN ({placeholders}) "
        f"GROUP BY a.assumption ORDER BY cnt DESC, a.assumption ASC {limit_sql}",
        params,
    )


def fetch_review_rows_by_claim(topic_table: str, claim: str) -> list[dict]:
    return query(
        "SELECT proposition, assumption, cnt "
        f"FROM `{topic_table}` WHERE claim = %s ORDER BY cnt DESC, proposition ASC",
        [claim],
    )


def fetch_proposition_counts_by_claim(topic_table: str, claim: str) -> list[dict]:
    return query(
        f"SELECT proposition, cnt FROM `{topic_table}` WHERE claim = %s ORDER BY proposition",
        [claim],
    )


def fetch_contraries_by_assumptions(contrary_table: str, assumptions: list[str]) -> list[dict]:
    if not assumptions:
        return []
    placeholders = ", ".join(["%s"] * len(assumptions))
    return query(
        "SELECT assumption, proposition "
        f"FROM `{contrary_table}` WHERE assumption IN ({placeholders}) AND isContrary = 1 "
        "ORDER BY assumption, proposition",
        list(assumptions),
    )


def fetch_head_rows_for_ratios() -> list[dict]:
    return query("SELECT claim, LOWER(Topic) AS topic, Sentiment FROM head")


def fetch_claim_total_by_topic_table(topic_table: str, claim: str) -> int:
    rows = query(
        f"SELECT COALESCE(SUM(cnt), 0) AS total FROM `{topic_table}` WHERE claim = %s",
        [claim],
    )
    return int(rows[0]["total"] or 0) if rows else 0


def add_claim_scores(rows: list[dict], claim_scores: dict[str, float],
                     claim_to_skip: str | None, allowed_claims: set[str]) -> None:
    for r in rows or []:
        claim = r.get("claim")
        if not claim or claim == claim_to_skip or claim not in allowed_claims:
            continue
        score = 1 + float(r.get("cnt") or 0) / 1000
        claim_scores[claim] = claim_scores.get(claim, 0) + score
