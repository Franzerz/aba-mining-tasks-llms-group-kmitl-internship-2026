#### PATCHARAKORN ####

"""Prepare Task 1 training data: Topic, Text Span, Sentiment extraction.

Reads gold CSV and creates chat-formatted training examples for aspect mention extraction.
Output format: {"annotations": [{"topic": "Room", "text": "The room was spotless.", "sentiment": "Positive"}, ...]}
"""
from __future__ import annotations

import json
import os
import random
import statistics
import sys
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]  # ABA_mining
load_dotenv(ROOT / ".env")
os.environ.setdefault("HF_HOME", str((ROOT.parent / ".hf_cache").resolve()))

sys.path.insert(0, str(ROOT))
from internship.ABA_mining.src.prompts import load_prompt, render_prompt  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402

CSV = ROOT / "Dataset" / "Original ABA Dataset for Version 2 [Oct 23, 2025], Senior Project, MUICT - Sheet2_.csv"
OUT_DIR = ROOT / "finetune" / "data"
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
MAX_SEQ_LENGTH = 1024




def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(ROOT / "configs" / "topics.yaml", encoding="utf-8") as f:
        topics_cfg = yaml.safe_load(f)
    topics = topics_cfg["schemas"][topics_cfg["active_schema"]]["topics"]
    topics_str = ", ".join(topics)

    header = load_prompt(ROOT, "prompts/task1/finetune_header.txt")
    footer = load_prompt(ROOT, "prompts/task1/footer.txt")
    prompt_template = header.rstrip() + "\n\n" + footer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))

    df = pd.read_csv(CSV, dtype=str, keep_default_na=False)
    df = df[df["ID"].str.strip() != ""]

    examples = []
    dropped = []

    for review_id, g in df.groupby("ID", sort=False):
        pos = g["PositiveReview"].iloc[0] or ""
        neg = g["NegativeReview"].iloc[0] or ""
        review_text = f'PositiveReview — "{pos.strip()}"\nNegativeReview — "{neg.strip()}"'

        annotations = []

        for _, row in g.iterrows():
            topic = row["Topic"].strip()
            if topic not in topics:
                continue

            text = row["Selected Content"].strip()
            if not text:
                continue

            sentiment = row["Pos/Neg"].strip()
            if sentiment not in ["Positive", "Negative"]:
                continue

            # Verbatim check
            if text not in review_text:
                dropped.append({
                    "ID": review_id,
                    "Topic": topic,
                    "Selected Content": text,
                    "Reason": "Not verbatim in review text"
                })
                continue

            # Create annotation with topic, text span, sentiment
            annotations.append({
                "topic": topic,
                "text": text,
                "sentiment": sentiment
            })

        if not annotations:
            dropped.append({
                "ID": review_id,
                "Topic": "All",
                "Selected Content": "No valid annotations",
                "Reason": "No annotations generated"
            })
            continue

        # Generate target JSON in Task 1 format
        target_json = json.dumps({"annotations": annotations}, ensure_ascii=False, separators=(",", ":"))

        # Render prompt
        prompt = render_prompt(
            prompt_template,
            TOPICS=topics_str,
            REVIEW_TEXT=review_text,
        )

        # Create chat messages
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target_json},
        ]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False)
        n_tokens = len(tokenizer(rendered)["input_ids"])

        # Use number of unique topics as bucketing criterion
        topic_set = set(a["topic"] for a in annotations)
        topic_count = len(topic_set)

        examples.append({
            "id": review_id,
            "messages": messages,
            "topic_count": topic_count,
            "n_tokens": n_tokens,
        })

    # Save dropped spans
    if dropped:
        pd.DataFrame(dropped).to_csv(OUT_DIR / "dropped_spans.csv", index=False)
        print(f"Dropped {len(dropped)} spans, logged to dropped_spans.csv")

    # Filter by token length
    n_too_long = sum(1 for ex in examples if ex["n_tokens"] > MAX_SEQ_LENGTH)
    examples = [ex for ex in examples if ex["n_tokens"] <= MAX_SEQ_LENGTH]
    print(f"Dropped {n_too_long} examples exceeding MAX_SEQ_LENGTH={MAX_SEQ_LENGTH} tokens")

    # Stratified 80/20 split by topic count
    random.seed(42)
    by_bucket: dict[int, list[dict]] = {}
    for ex in examples:
        by_bucket.setdefault(ex["topic_count"], []).append(ex)

    train, val = [], []
    for items in by_bucket.values():
        random.shuffle(items)
        n_val = round(len(items) * 0.2) if len(items) >= 10 else (1 if len(items) > 1 else 0)
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    random.shuffle(train)
    random.shuffle(val)

    # Write JSONL
    def write_jsonl(path: Path, items: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for ex in items:
                f.write(json.dumps({"messages": ex["messages"]}, ensure_ascii=False) + "\n")

    write_jsonl(OUT_DIR / "task1_train.jsonl", train)
    write_jsonl(OUT_DIR / "task1_val.jsonl", val)

    with open(OUT_DIR / "val_ids.json", "w", encoding="utf-8") as f:
        json.dump([ex["id"] for ex in val], f)

    # Statistics
    lengths = [ex["n_tokens"] for ex in examples]
    stats = {
        "n_examples": len(examples),
        "n_train": len(train),
        "n_val": len(val),
        "n_dropped_spans": len(dropped),
        "n_dropped_too_long": n_too_long,
        "max_seq_length": MAX_SEQ_LENGTH,
        "mean_tokens": statistics.mean(lengths) if lengths else 0,
        "p50_tokens": statistics.median(lengths) if lengths else 0,
        "p95_tokens": sorted(lengths)[int(len(lengths) * 0.95)] if lengths else 0,
        "max_tokens": max(lengths) if lengths else 0,
    }
    with open(OUT_DIR / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "="*70)
    print("DATA PREPARATION SUMMARY")
    print("="*70)
    print(json.dumps(stats, indent=2))
    print("="*70)


if __name__ == "__main__":
    main()
