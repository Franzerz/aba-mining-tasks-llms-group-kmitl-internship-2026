#### PATCHARAKORN ####

"""Build a gold-format CSV from Task 1 (sequence) output + Task 2 output.

Task 1 sequence gives one row per (Review ID, Topic, Selected Content, Sentiment).
Task 2 gives the Head + body / cont literals for every non-Off span.
The result follows the layout of the "Original ABA Dataset" gold sheet:

ID,Title,PositiveReview,NegativeReview,Topic,Selected Content,Pos/Neg,Head,
Body 1..15,Cont. Body 1..15,Concat
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

MAX_BODY = 15

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_REVIEWS = REPO_ROOT / "Dataset" / "green_arbor.csv"
DEFAULT_TASK1 = (
    REPO_ROOT / "outputs" / "task1" / "qwen3.8_27b" / "modular" / "sequence"
    / "task1_qwen3.8_27b_extended11_sequence_run1.csv"
)
DEFAULT_TASK2 = (
    REPO_ROOT / "outputs" / "task2" / "gt" / "qwen3.8_27b" / "version1"
    / "task2_qwen3.8_27b_qwen3.8_27b_generator_v1_run1_gt_n462.jsonl"
)
DEFAULT_OUT = REPO_ROOT / "outputs" / "gold" / "gold_qwen3.8_27b_sequence_generator_v1_run1.csv"

HEADER = (
    ["ID", "Title", "PositiveReview", "NegativeReview", "Topic", "Selected Content", "Pos/Neg", "Head"]
    + [f"Body {i}" for i in range(1, MAX_BODY + 1)]
    + [f"Cont. Body {i}" for i in range(1, MAX_BODY + 1)]
    + ["Concat"]
)


def _key(review_id: str, topic: str, content: str, sentiment: str) -> tuple[str, str, str, str]:
    return (str(review_id).strip(), topic.strip(), content.strip(), sentiment.strip())


def load_reviews(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {
            row["ID"].strip(): {
                "Title": row.get("Title", "") or "",
                "PositiveReview": row.get("PositiveReview", "") or "",
                "NegativeReview": row.get("NegativeReview", "") or "",
            }
            for row in csv.DictReader(f)
        }


def load_task1(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_task2(path: Path) -> dict[tuple[str, str, str, str], dict]:
    out: dict[tuple[str, str, str, str], dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            key = _key(rec["review_id"], rec["topic"], rec["selected_content"], rec["sentiment"])
            out[key] = rec
    return out


def default_head(topic: str, sentiment: str) -> str:
    prefix = "good" if sentiment.strip().lower() == "positive" else "bad"
    return f"{prefix}_{topic.strip().lower()}"


def build_rows(reviews: dict, task1_rows: list[dict], task2: dict) -> tuple[list[list[str]], dict]:
    stats = {"rows": 0, "off": 0, "matched": 0, "unmatched": [], "overflow": [], "missing_review": set()}
    rows: list[list[str]] = []

    for t1 in task1_rows:
        rid = t1["Review ID"].strip()
        topic = (t1.get("Topic") or "").strip()
        content = (t1.get("Selected Content") or "").strip()
        sentiment = (t1.get("Sentiment") or "").strip()

        review = reviews.get(rid)
        if review is None:
            stats["missing_review"].add(rid)
            review = {"Title": "", "PositiveReview": "", "NegativeReview": ""}

        base = [rid, review["Title"], review["PositiveReview"], review["NegativeReview"], topic, content, sentiment]
        stats["rows"] += 1

        # Off rows (and rows without a sentiment) carry no Head/Body in the gold sheet.
        if topic == "Off" or not sentiment:
            stats["off"] += 1
            rows.append(base + [""] * (1 + 2 * MAX_BODY + 1))
            continue

        rec = task2.get(_key(rid, topic, content, sentiment))
        if rec is None:
            stats["unmatched"].append((rid, topic, content, sentiment))
            rows.append(base + [default_head(topic, sentiment)] + [""] * (2 * MAX_BODY + 1))
            continue

        stats["matched"] += 1
        head = rec.get("head") or default_head(topic, sentiment)
        body = [b for b in (rec.get("body_literals") or []) if b]
        cont = [c for c in (rec.get("cont_literals") or []) if c]
        if len(body) > MAX_BODY or len(cont) > MAX_BODY:
            stats["overflow"].append((rid, topic, len(body), len(cont)))

        body_cols = (body[:MAX_BODY] + [""] * MAX_BODY)[:MAX_BODY]
        cont_cols = (cont[:MAX_BODY] + [""] * MAX_BODY)[:MAX_BODY]
        # Concat keeps every literal, even past the 15-column limit.
        concat = " , ".join(body + cont)
        rows.append(base + [head] + body_cols + cont_cols + [concat])

    return rows, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build gold-format CSV from Task 1 sequence + Task 2 outputs")
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS, help="Review CSV (ID, [Title], PositiveReview, NegativeReview)")
    parser.add_argument("--task1", type=Path, default=DEFAULT_TASK1, help="Task 1 sequence CSV")
    parser.add_argument("--task2", type=Path, default=DEFAULT_TASK2, help="Task 2 JSONL")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output gold CSV")
    args = parser.parse_args()

    reviews = load_reviews(args.reviews)
    task1_rows = load_task1(args.task1)
    task2 = load_task2(args.task2)

    rows, stats = build_rows(reviews, task1_rows, task2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {args.out}")
    print(f"  Off / no-sentiment rows : {stats['off']}")
    print(f"  Matched to Task 2       : {stats['matched']}")
    print(f"  Unmatched (no literals) : {len(stats['unmatched'])}")
    for u in stats["unmatched"][:20]:
        print(f"    - {u}")
    if stats["overflow"]:
        print(f"  >{MAX_BODY} literals (truncated in Body cols, kept in Concat): {stats['overflow']}")
    if stats["missing_review"]:
        print(f"  Review IDs not in reviews CSV: {sorted(stats['missing_review'], key=int)}")


if __name__ == "__main__":
    main()
