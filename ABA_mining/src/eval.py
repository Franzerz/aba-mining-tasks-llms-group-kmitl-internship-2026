#### NOT USE ####

import json
from pathlib import Path
from typing import Any
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def load_gold_data(gold_csv: Path) -> dict[str, dict[str, Any]]:
    """Load gold/ground-truth data from CSV."""
    df = pd.read_csv(gold_csv, dtype=str, keep_default_na=False)
    gold = {}
    for _, row in df.iterrows():
        review_id = str(row.get("Column1", row.get("ID", "")))
        if review_id not in gold:
            gold[review_id] = {"pos": row.get("PositiveReview", ""), "neg": row.get("NegativeReview", "")}
    return gold


def load_model_output(jsonl_path: Path) -> dict[str, dict[str, Any]]:
    """Load model output from JSONL file."""
    output = {}
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            output[r["review_id"]] = r
    return output


def extract_gold_topics(gold_dict: dict[str, Any]) -> set[str]:
    """Extract topics mentioned in gold data (dummy: just check if text non-empty)."""
    topics = set()
    if gold_dict["pos"].strip():
        # In a real scenario, you'd parse the gold JSON or annotations
        # For now, assume any non-empty text means topic is present
        pass
    if gold_dict["neg"].strip():
        pass
    return topics


def cosine_sim_match(pred_span: str, gold_spans: list[str], threshold: float = 0.85) -> bool:
    """Check if pred_span matches any gold_span using cosine similarity."""
    if not gold_spans or not pred_span:
        return False

    try:
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
        texts = [pred_span] + gold_spans
        tfidf = vectorizer.fit_transform(texts)
        sim = cosine_similarity(tfidf[0:1], tfidf[1:])[0]
        return np.max(sim) >= threshold
    except Exception:
        # Fallback to exact match if vectorizer fails
        return pred_span in gold_spans


def eval_subtask1_1(model_output: dict, gold_data: dict, topics: list[str]) -> dict[str, float]:
    """Evaluate Subtask 1.1 (topic identification): precision, recall, F1, accuracy."""
    if not model_output or not gold_data:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    tp, fp, fn, tn = 0, 0, 0, 0

    for review_id, result in model_output.items():
        if not result.get("valid"):
            continue

        parsed = result.get("parsed")
        if not parsed or "Topics" not in parsed:
            continue

        for topic in topics:
            pred_present = parsed["Topics"].get(topic, False)
            # Gold: assume present if review has text (simplified — in real case, use ground truth annotations)
            gold_present = review_id in gold_data  # Simplified

            if pred_present and gold_present:
                tp += 1
            elif pred_present and not gold_present:
                fp += 1
            elif not pred_present and gold_present:
                fn += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def eval_subtask1_2(model_output: dict, gold_data: dict, topics: list[str], cosine_threshold: float = 0.85) -> dict[str, float]:
    """Evaluate Subtask 1.2 (span extraction): precision, recall, F1.
    Uses cosine similarity for matching (not exact verbatim).
    """
    if not model_output:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp, fp, fn = 0, 0, 0

    for review_id, result in model_output.items():
        if not result.get("valid"):
            continue

        parsed = result.get("parsed")
        if not parsed or "Topics" not in parsed:
            continue

        for topic in topics:
            spans = parsed["Topics"].get(topic, [])
            if not isinstance(spans, list):
                continue

            pred_texts = [s.get("text") for s in spans if s.get("text")]

            # Gold: extract from review text (simplified)
            review_text = gold_data.get(review_id, {})
            gold_texts = []  # In real scenario, extract from gold annotations

            # Match predictions to gold
            for pred in pred_texts:
                if cosine_sim_match(pred, gold_texts, cosine_threshold):
                    tp += 1
                else:
                    fp += 1

            fn += max(0, len(gold_texts) - len(pred_texts))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def eval_subtask1_3(model_output: dict, gold_data: dict, topics: list[str]) -> dict[str, float]:
    """Evaluate Subtask 1.3 (sentiment classification): precision, recall, F1, accuracy."""
    if not model_output:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    tp, fp, fn, tn = 0, 0, 0, 0

    for review_id, result in model_output.items():
        if not result.get("valid"):
            continue

        parsed = result.get("parsed")
        if not parsed or "Topics" not in parsed:
            continue

        for topic in topics:
            pred_label = parsed["Topics"].get(topic)
            # Gold: assume Positive for simplicity (in real case, use ground truth)
            gold_label = "Positive" if review_id in gold_data else None

            if pred_label == gold_label:
                if pred_label is None:
                    tn += 1
                else:
                    tp += 1
            else:
                if pred_label is None:
                    fn += 1
                else:
                    fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def evaluate_experiment(jsonl_path: Path, gold_csv: Path, output_schema: str, topics: list[str]) -> dict[str, Any]:
    """Main evaluation function."""
    model_output = load_model_output(jsonl_path)
    gold_data = load_gold_data(gold_csv)

    results = {
        "experiment": jsonl_path.parent.name,
        "output_schema": output_schema,
        "total_reviews": len(model_output),
        "valid_reviews": sum(1 for r in model_output.values() if r.get("valid")),
    }

    if output_schema == "topic_only":
        results["subtask1_1"] = eval_subtask1_1(model_output, gold_data, topics)
    elif output_schema == "span_only":
        results["subtask1_2"] = eval_subtask1_2(model_output, gold_data, topics)
    elif output_schema == "sentiment_only":
        results["subtask1_3"] = eval_subtask1_3(model_output, gold_data, topics)
    elif output_schema == "full":
        results["subtask1_1"] = eval_subtask1_1(model_output, gold_data, topics)
        results["subtask1_2"] = eval_subtask1_2(model_output, gold_data, topics)
        results["subtask1_3"] = eval_subtask1_3(model_output, gold_data, topics)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Task 1 ABA mining experiments")
    parser.add_argument("--jsonl", required=True, help="Path to JSONL output file")
    parser.add_argument("--gold", required=True, help="Path to gold CSV file")
    parser.add_argument("--schema", required=True, choices=["topic_only", "span_only", "sentiment_only", "full"])
    parser.add_argument("--topics", nargs="+", default=["Room", "Staff", "Location", "Food", "Price", "Facility", "Check-in", "Check-out", "Off", "Booking-issue", "Taxi-issue"])
    args = parser.parse_args()

    results = evaluate_experiment(Path(args.jsonl), Path(args.gold), args.schema, args.topics)
    print(json.dumps(results, indent=2))
