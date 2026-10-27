#### PATCHARAKORN ####

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from jsonschema import Draft202012Validator
from tqdm import tqdm

from .config import ModelConfig, PathsConfig, TopicsConfig
from .llm import LLMClient
from .prompts import load_prompt, render_prompt
from .utils import try_parse_json, normalize_parsed_json


@dataclass(frozen=True)
class Task1Instance:
    review_id: str
    review_text: str


def build_task1_schema(topics: list[str], output_schema: str = "full") -> dict[str, Any]:
    """Build JSON schema validator based on output schema type.

    Args:
        topics: List of topic names
        output_schema: One of "topic_only", "span_only", "sentiment_only", "full"
    """
    if output_schema == "topic_only":
        # {"topic_mentions": {"Room": true/false, ...}}
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "topic_mentions": {
                    "type": "object",
                    "properties": {t: {"type": "boolean"} for t in topics},
                    "required": topics,
                    "additionalProperties": False,
                }
            },
            "required": ["topic_mentions"],
            "additionalProperties": False,
        }
    elif output_schema == "span_only":
        # {"spans": {"Room": [{"text": "..."}], ...}}
        topic_obj = {
            "type": "object",
            "properties": {"text": {"type": ["string", "null"]}},
            "required": ["text"],
            "additionalProperties": False,
        }
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "spans": {
                    "type": "object",
                    "properties": {t: {"type": "array", "items": topic_obj, "minItems": 1} for t in topics},
                    "required": topics,
                    "additionalProperties": False,
                }
            },
            "required": ["spans"],
            "additionalProperties": False,
        }
    elif output_schema == "sentiment_only":
        # {"sentiments": {"Room": "Positive"/"Negative"/null, ...}}
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "sentiments": {
                    "type": "object",
                    "properties": {t: {"type": ["string", "null"]} for t in topics},
                    "required": topics,
                    "additionalProperties": False,
                }
            },
            "required": ["sentiments"],
            "additionalProperties": False,
        }
    else:  # full
        # {"annotations": [{"topic": "Room", "text": "...", "sentiment": "Positive"}, ...]}
        annotation_obj = {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "enum": topics},
                "text": {"type": "string"},
                "sentiment": {"type": "string", "enum": ["Positive", "Negative"]},
            },
            "required": ["topic", "text", "sentiment"],
            "additionalProperties": False,
        }
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "annotations": {
                    "type": "array",
                    "items": annotation_obj,
                }
            },
            "required": ["annotations"],
            "additionalProperties": False,
        }

# checking if the above function is correct
def local_validate_task1(parsed: Any, *, topics: list[str], review_text: str, output_schema: str = "full") -> list[str]:
    errors: list[str] = []
    validator = Draft202012Validator(build_task1_schema(topics, output_schema))
    for e in validator.iter_errors(parsed):
        errors.append(e.message)

    if errors:
        return errors

    # Schema-specific semantic validation
    if output_schema == "topic_only":
        # {"topic_mentions": {"Room": true/false, ...}}
        data = parsed.get("topic_mentions", {})
        for t in topics:
            val = data.get(t)
            if not isinstance(val, bool):
                errors.append(f"{t}: value must be boolean")
    elif output_schema == "span_only":
        # {"spans": {"Room": [{"text": "..."}], ...}}
        data = parsed.get("spans", {})
        for t in topics:
            items = data.get(t, [])
            if not isinstance(items, list):
                continue
            for it in items:
                if not isinstance(it, dict):
                    continue
                text = it.get("text")
                if text is not None and text not in review_text:
                    errors.append(f"{t}: text span not found verbatim in review_text")
    elif output_schema == "sentiment_only":
        # {"sentiments": {"Room": "Positive"/"Negative"/null, ...}}
        data = parsed.get("sentiments", {})
        for t in topics:
            val = data.get(t)
            if val is not None and val not in ("Positive", "Negative"):
                errors.append(f"{t}: label must be 'Positive', 'Negative', or null")
    else:  # full
        # {"annotations": [{"topic": "Room", "text": "...", "sentiment": "Positive"}, ...]}
        annotations = parsed.get("annotations", [])
        if not isinstance(annotations, list):
            errors.append("annotations must be a list")
        else:
            for ann in annotations:
                if not isinstance(ann, dict):
                    continue
                topic = ann.get("topic")
                text = ann.get("text")
                sentiment = ann.get("sentiment")
                if sentiment and sentiment not in ("Positive", "Negative"):
                    errors.append(f"sentiment must be 'Positive' or 'Negative', got '{sentiment}'")
                if text and not isinstance(text, str):
                    errors.append(f"text must be a string, got {type(text)}")
                if text and text not in review_text:
                    errors.append(f"text span '{text}' not found verbatim in review_text")
    return errors


def load_task1_instances_from_input(paths: PathsConfig, limit_reviews: int | None = None) -> list[Task1Instance]:
    """Load reviews from the input CSV. limit_reviews=None loads the whole dataset."""
    try:
        df = pd.read_csv(paths.input_csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except UnicodeDecodeError:
        # Excel's plain "CSV" export writes cp1252 instead of UTF-8
        df = pd.read_csv(paths.input_csv, dtype=str, keep_default_na=False, encoding="cp1252")
    id_col = "Column1" if "Column1" in df.columns else "ID"
    if id_col not in df.columns:
        raise RuntimeError(f"Expected 'Column1' or 'ID' column in input_csv, found: {list(df.columns)}")

    grouped = df.groupby(id_col, sort=False, dropna=False)
    instances: list[Task1Instance] = []
    for review_id, g in grouped:
        pos = (g["PositiveReview"].iloc[0] if "PositiveReview" in g.columns else "") or ""
        neg = (g["NegativeReview"].iloc[0] if "NegativeReview" in g.columns else "") or ""
        review_text = f'PositiveReview — "{pos.strip()}"\nNegativeReview — "{neg.strip()}"'
        instances.append(Task1Instance(review_id=str(review_id), review_text=review_text))

        if limit_reviews is not None and len(instances) >= limit_reviews:
            break

    return instances


def _transform_to_schema(parsed: dict[str, Any], output_schema: str, topics: list[str]) -> dict[str, Any]:
    """Transform model output from full format to the target output schema."""
    if output_schema == "full":
        return parsed  # No transformation needed

    topics_dict = parsed.get("Topics", {})
    new_topics = {}

    if output_schema == "topic_only":
        # {"Topics": {"Room": true/false, ...}}
        for topic in topics:
            spans = topics_dict.get(topic, [])
            if isinstance(spans, list) and len(spans) > 0:
                # Topic is present if there are non-null spans
                new_topics[topic] = any(s.get("text") is not None for s in spans)
            else:
                new_topics[topic] = False
        return {"Topics": new_topics}

    elif output_schema == "span_only":
        # {"Topics": {"Room": [{"text": "..."}], ...}}
        for topic in topics:
            spans = topics_dict.get(topic, [])
            if isinstance(spans, list):
                new_spans = [{"text": s.get("text")} for s in spans]
                new_topics[topic] = new_spans
            else:
                new_topics[topic] = [{"text": None}]
        return {"Topics": new_topics}

    elif output_schema == "sentiment_only":
        # {"Topics": {"Room": "Positive"/"Negative"/null, ...}}
        for topic in topics:
            spans = topics_dict.get(topic, [])
            if isinstance(spans, list) and len(spans) > 0:
                # Take the first non-null sentiment (or null if all are null)
                labels = [s.get("label") for s in spans if s.get("text") is not None]
                new_topics[topic] = labels[0] if labels else None
            else:
                new_topics[topic] = None
        return {"Topics": new_topics}

    return parsed


def run_task1(
    *,
    repo_root: Path,
    client: LLMClient,
    topics_cfg: TopicsConfig,
    model_cfg: ModelConfig,
    paths_cfg: PathsConfig,
    limit_reviews: int | None = None,
    prompt_path: str | None = None,
    prompt_template: str | None = None,
    output_label: str | None = None,
    output_schema: str = "full",
    validator_path: str = "prompts/task1/Contrastive/validator_v1.txt",
    max_retries: int = 2,
    output_subdir: str | None = None,
    runs_per_id: int = 3,
) -> Path:
    out_dir = paths_cfg.task1_dir
    if output_subdir:
        out_dir = out_dir / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = model_cfg.task1_model.replace(":", "_").replace("/", "_")
    prompt_label = output_label or (Path(prompt_path).stem if prompt_path else "custom")
    out_path = out_dir / f"task1_{model_name}_{topics_cfg.active_schema}_{prompt_label}.jsonl"

    if prompt_template is None:
        if prompt_path is None:
            raise ValueError("Provide either prompt_path or prompt_template.")
        gen_template = load_prompt(repo_root, prompt_path)
    else:
        gen_template = prompt_template
    val_template = load_prompt(repo_root, validator_path)
    topics_str = ", ".join(topics_cfg.topics)
    topic_count = str(len(topics_cfg.topics))

    instances = load_task1_instances_from_input(paths_cfg, limit_reviews=limit_reviews)

    def _generate(inst: Task1Instance) -> tuple[str, bool, Any, list[str]]:
        """Run the generator and return (raw_text, ok, parsed, errors)."""
        prompt = render_prompt(
            gen_template,
            TOPICS=topics_str,
            TOPIC_COUNT=topic_count,
            REVIEW_TEXT=inst.review_text,
        )
        resp = client.complete(
            model=model_cfg.task1_model,
            prompt=prompt,
            temperature=model_cfg.temperature,
            top_p=model_cfg.top_p,
            max_output_tokens=model_cfg.max_output_tokens,
        )
        ok, parsed, err = try_parse_json(resp.text)
        errors: list[str] = []
        if not ok:
            errors = [f"json_parse_error: {err}"]
        else:
            # Normalize the parsed JSON to match the expected schema
            parsed, norm_errors = normalize_parsed_json(parsed, output_schema=output_schema, topics=topics_cfg.topics)
            if parsed is None:
                errors = norm_errors
            else:
                errors = local_validate_task1(parsed, topics=topics_cfg.topics, review_text=inst.review_text, output_schema=output_schema)
        return resp.text, ok, parsed, errors

    def _validate_and_fix(inst: Task1Instance, candidate_json: str, errors: list[str]) -> tuple[str, bool, Any, list[str]]:
        """Send the candidate + errors to the validator LLM for correction.
        Only used for 'full' schema — the validator prompt expects {text, label} format."""
        if output_schema != "full":
            return candidate_json, False, None, errors  # skip validator for non-full schemas
        prompt = render_prompt(
            val_template,
            TOPICS=topics_str,
            REVIEW_TEXT=inst.review_text,
            CANDIDATE_JSON=candidate_json,
            ERRORS="; ".join(errors),
        )
        resp = client.complete(
            model=model_cfg.task1_model,
            prompt=prompt,
            temperature=model_cfg.temperature,
            top_p=model_cfg.top_p,
            max_output_tokens=model_cfg.max_output_tokens,
        )
        ok, parsed, err = try_parse_json(resp.text)
        new_errors: list[str] = []
        if not ok:
            new_errors = [f"json_parse_error: {err}"]
        else:
            # Normalize the parsed JSON to match the expected schema
            parsed, norm_errors = normalize_parsed_json(parsed, output_schema=output_schema, topics=topics_cfg.topics)
            if parsed is None:
                new_errors = norm_errors
            else:
                new_errors = local_validate_task1(parsed, topics=topics_cfg.topics, review_text=inst.review_text, output_schema=output_schema)
        return resp.text, ok, parsed, new_errors

    def _process_one(inst: Task1Instance) -> dict[str, Any]:
        raw, ok, parsed, errors = _generate(inst)
        attempt = 0

        # Cascading Validation Loop
        while errors and attempt < max_retries:
            attempt += 1
            candidate_str = raw if not ok else json.dumps(parsed, ensure_ascii=False)
            val_raw, val_ok, val_parsed, val_errors = _validate_and_fix(inst, candidate_str, errors)
            # Accept the fix only if it's an improvement (fewer or no errors)
            if val_ok and len(val_errors) < len(errors):
                raw, ok, parsed, errors = val_raw, val_ok, val_parsed, val_errors
            elif val_ok and len(val_errors) == 0:
                raw, ok, parsed, errors = val_raw, val_ok, val_parsed, val_errors
                break
            else:
                # Validator didn't help
                break

        # If we have a valid parse, compact the raw_output to remove newlines/indentation
        compact_raw = json.dumps(parsed, ensure_ascii=False) if ok and parsed is not None else raw

        return {
            "review_id": inst.review_id,
            "schema": topics_cfg.active_schema,
            "output_schema": output_schema,
            "prompt": prompt_path,
            "raw_output": compact_raw,
            "parsed": parsed if ok else None,
            "valid": len(errors) == 0,
            "errors": errors,
            "retries": attempt,
        }

    # Checkpoint: collect already-processed (review_id, run_index) pairs
    processed_run_keys: set[tuple[str, int]] = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if "review_id" in rec:
                        ri = int(rec.get("run_index", 1))
                        processed_run_keys.add((str(rec["review_id"]), ri))
                except json.JSONDecodeError:
                    pass

    total_to_do = sum(
        1 for inst in instances
        for ri in range(1, runs_per_id + 1)
        if (inst.review_id, ri) not in processed_run_keys
    )
    if processed_run_keys:
        print(f"Resuming: {len(processed_run_keys)} runs already done, {total_to_do} remaining")
    print(f"Total runs to process: {total_to_do} ({len(instances)} IDs x {runs_per_id} runs each)")

    # Process one ID at a time, runs_per_id times each; append immediately so progress is never lost
    with out_path.open("a", encoding="utf-8") as f, tqdm(total=total_to_do, desc="Task1") as pbar:
        for inst in instances:
            for run_idx in range(1, runs_per_id + 1):
                if (inst.review_id, run_idx) in processed_run_keys:
                    continue
                result = _process_one(inst)
                result["run_index"] = run_idx
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                pbar.update(1)

    # Read all results (existing + new) sorted by (id_order, run_index), then write CSV
    id_order = {inst.review_id: i for i, inst in enumerate(instances)}
    all_results: list[dict[str, Any]] = []
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                all_results.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    all_results.sort(key=lambda r: (id_order.get(str(r.get("review_id", "")), 999999), int(r.get("run_index", 1))))

    csv_path = out_path.with_suffix(".csv")
    _write_readable_csv(csv_path, all_results, instances, output_schema=output_schema)

    return out_path


def _write_readable_csv(csv_path: Path, results: list[dict[str, Any]], instances: list[Task1Instance], output_schema: str = "full") -> None:
    """Write a human-readable CSV with output appropriate to the schema type."""
    import csv as csv_mod

    inst_map = {inst.review_id: inst.review_text for inst in instances}

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv_mod.writer(f)

        if output_schema == "topic_only":
            writer.writerow(["Review ID", "Run", "Valid", "Topic", "Errors"])
            for r in results:
                review_id = r["review_id"]
                run_idx = r.get("run_index", 1)
                valid = r["valid"]
                errors = "; ".join(r.get("errors", []))
                parsed = r.get("parsed")
                if parsed and "topic_mentions" in parsed:
                    # New format: topic_mentions as dict
                    present = [t for t, v in parsed["topic_mentions"].items() if v]
                    topics_str = ", ".join(present) if present else "(none)"
                    writer.writerow([review_id, run_idx, valid, topics_str, errors])
                elif parsed and "Topics" in parsed:
                    # Old format: Topics as dict (backward compatibility)
                    present = [t for t, v in parsed["Topics"].items() if v]
                    topics_str = ", ".join(present) if present else "(none)"
                    writer.writerow([review_id, run_idx, valid, topics_str, errors])
                else:
                    writer.writerow([review_id, run_idx, valid, "(parse failed)", errors])
        elif output_schema == "span_only":
            writer.writerow(["Review ID", "Run", "Valid", "Topic", "Selected Content", "Errors"])
            for r in results:
                review_id = r["review_id"]
                run_idx = r.get("run_index", 1)
                valid = r["valid"]
                errors = "; ".join(r.get("errors", []))
                parsed = r.get("parsed")
                if parsed and "spans" in parsed:
                    # New format: spans as dict
                    found_any = False
                    for topic, spans_list in parsed["spans"].items():
                        if not isinstance(spans_list, list):
                            continue
                        for span in spans_list:
                            if span is None or not isinstance(span, dict):
                                continue
                            text = span.get("text")
                            if not text or text == "null":
                                continue
                            found_any = True
                            writer.writerow([review_id, run_idx, valid, topic, text, ""])
                    if not found_any:
                        writer.writerow([review_id, run_idx, valid, "(no spans found)", "", errors])
                elif parsed and "Topics" in parsed:
                    # Old format: Topics as dict (backward compatibility)
                    found_any = False
                    for topic, spans in parsed["Topics"].items():
                        if not isinstance(spans, list):
                            continue
                        for span in spans:
                            if span is None or not isinstance(span, dict):
                                continue
                            text = span.get("text")
                            if not text or text == "null":
                                continue
                            found_any = True
                            writer.writerow([review_id, run_idx, valid, topic, text, ""])
                    if not found_any:
                        writer.writerow([review_id, run_idx, valid, "(no spans found)", "", errors])
                else:
                    writer.writerow([review_id, run_idx, valid, "(parse failed)", "", errors])
        elif output_schema == "sentiment_only":
            writer.writerow(["Review ID", "Run", "Valid", "Topic", "Sentiment", "Errors"])
            for r in results:
                review_id = r["review_id"]
                run_idx = r.get("run_index", 1)
                valid = r["valid"]
                errors = "; ".join(r.get("errors", []))
                parsed = r.get("parsed")
                if parsed and "annotations" in parsed:
                    # New format: annotations as list
                    found_any = False
                    for annotation in parsed["annotations"]:
                        if isinstance(annotation, dict):
                            topic = annotation.get("topic")
                            sentiment = annotation.get("sentiment")
                            if topic and sentiment:
                                found_any = True
                                writer.writerow([review_id, run_idx, valid, topic, sentiment, ""])
                    if not found_any:
                        writer.writerow([review_id, run_idx, valid, "(no annotations found)", "", errors])
                elif parsed and "Topics" in parsed:
                    # Old format: Topics as dict (backward compatibility)
                    found_any = False
                    for topic, label in parsed["Topics"].items():
                        if label is not None:
                            found_any = True
                            writer.writerow([review_id, run_idx, valid, topic, label, ""])
                    if not found_any:
                        writer.writerow([review_id, run_idx, valid, "(no sentiments found)", "", errors])
                else:
                    writer.writerow([review_id, run_idx, valid, "(parse failed)", "", errors])
        else:  # full
            writer.writerow(["Review ID", "Run", "Valid", "Topic", "Selected Content", "Sentiment", "Errors"])
            for r in results:
                review_id = r["review_id"]
                run_idx = r.get("run_index", 1)
                valid = r["valid"]
                errors = "; ".join(r.get("errors", []))
                parsed = r.get("parsed")
                if parsed and "annotations" in parsed:
                    # New format: annotations as list with text spans
                    found_any = False
                    for annotation in parsed["annotations"]:
                        if not isinstance(annotation, dict):
                            continue
                        topic = annotation.get("topic")
                        text = annotation.get("text")
                        sentiment = annotation.get("sentiment")
                        if topic and text and sentiment:
                            found_any = True
                            writer.writerow([review_id, run_idx, valid, topic, text, sentiment, ""])
                    if not found_any:
                        writer.writerow([review_id, run_idx, valid, "(no annotations found)", "", "", errors])
                elif parsed and "Topics" in parsed:
                    # Old format: Topics as dict (backward compatibility)
                    found_any = False
                    topics_data = parsed["Topics"]
                    # Handle case where Topics is a list (older format)
                    if isinstance(topics_data, list):
                        writer.writerow([review_id, run_idx, valid, "(parse failed - old format)", "", "", errors])
                    elif isinstance(topics_data, dict):
                        for topic, spans in topics_data.items():
                            if not isinstance(spans, list):
                                continue
                            for span in spans:
                                if span is None or not isinstance(span, dict):
                                    continue
                                text = span.get("text")
                                label = span.get("label")
                                if not text or text == "null":
                                    continue
                                found_any = True
                                writer.writerow([review_id, run_idx, valid, topic, text, label or "", ""])
                        if not found_any:
                            writer.writerow([review_id, run_idx, valid, "(no topics found)", "", "", errors])
                else:
                    writer.writerow([review_id, run_idx, valid, "(parse failed)", "", "", errors])


# ── "sequence" experiment ────────────────────────────────────────────────────
# subtask 1.1 -> 1.2 -> 1.3, each step given the previous step's answer.
# Each subtask finishes for ALL reviews before the next subtask starts.
#   1. Topics:    the subtask1_1 prompt picks the topics the review mentions.
#   2. Text:      the subtask1_2 prompt is asked only about those topics ({{TOPICS}} = picked topics).
#   3. Sentiment: the subtask1_3 prompt labels each piece of text on its own
#                 ({{TOPICS}} = its topic, {{REVIEW_TEXT}} = the full review + that text to label).

SEQUENCE_STEPS = {"subtask1_1": "topic_only", "subtask1_2": "span_only", "subtask1_3": "sentiment_only"}

# Must match how load_task1_instances_from_input builds review_text
_POS_PREFIX = 'PositiveReview — "'
_NEG_SEPARATOR = '"\nNegativeReview — "'


def _section_sentiment(review_text: str, text: str) -> str:
    """Sentiment implied by which box the text was written in ("" if it's in both or neither)."""
    body = review_text[len(_POS_PREFIX):-1] if review_text.startswith(_POS_PREFIX) else review_text
    positive, _, negative = body.partition(_NEG_SEPARATOR)
    in_positive, in_negative = text in positive, text in negative
    if in_positive and not in_negative:
        return "Positive"
    if in_negative and not in_positive:
        return "Negative"
    return ""


_TOPIC_ARRAY = re.compile(r'"([^"]+)"\s*:\s*\[')
# Ends at `"}` (or `", "label"`) so an unescaped `"` inside the text doesn't cut it short
_TEXT_ITEM = re.compile(r'"text"\s*:\s*"(.*?)"\s*(?:\}|,\s*"label")', re.DOTALL)


def _salvage_spans(raw: str, topics: list[str]) -> dict[str, list[str]]:
    """Recover {topic: [texts]} from a subtask1_2 answer that isn't valid JSON.

    On long reviews the model's answer can get cut off, or contain an unescaped quote.
    Every complete {"text": "..."} item is kept; the broken or cut-off part is skipped.
    """
    wanted = set(topics)
    starts = [(m.group(1), m.start(), m.end()) for m in _TOPIC_ARRAY.finditer(raw)]
    found: dict[str, list[str]] = {}
    for i, (topic, _, body_start) in enumerate(starts):
        if topic not in wanted:
            continue
        body_end = starts[i + 1][1] if i + 1 < len(starts) else len(raw)
        for m in _TEXT_ITEM.finditer(raw, body_start, body_end):
            value = m.group(1)
            try:
                value = json.loads(f'"{value}"')
            except json.JSONDecodeError:
                value = value.replace('\\"', '"')
            if value.strip() and value not in found.setdefault(topic, []):
                found[topic].append(value)
    return {t: xs for t, xs in found.items() if xs}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line cut off by a crash
            if "review_id" in rec:
                records.append(rec)
    return records


def run_task1_sequence(
    *,
    client: LLMClient,
    topics_cfg: TopicsConfig,
    model_cfg: ModelConfig,
    instances: list[Task1Instance],
    topics_template: str,
    spans_template: str,
    sentiments_template: str,
    out_jsonl: Path,
    step_jsonls: dict[str, Path],
    runs_per_id: int = 3,
) -> Path:
    """Runs subtask1_1 for all reviews, then subtask1_2 for all, then subtask1_3 for all.

    Each subtask's answers go to step_jsonls[subtask] (+ .csv in that subtask's usual format).
    Final CSV: Review ID, Topic, Selected Content, Sentiment, Errors."""
    topics = topics_cfg.topics

    def _ask(template: str, **fields: str) -> tuple[str, Any, str | None]:
        resp = client.complete(
            model=model_cfg.task1_model,
            prompt=render_prompt(template, **fields),
            temperature=model_cfg.temperature,
            top_p=model_cfg.top_p,
            max_output_tokens=model_cfg.max_output_tokens,
        )
        ok, parsed, err = try_parse_json(resp.text)
        return resp.text, (parsed if ok else None), err

    def _step_topics(inst: Task1Instance) -> dict[str, Any]:
        raw, parsed, err = _ask(
            topics_template,
            TOPICS=", ".join(topics),
            TOPIC_COUNT=str(len(topics)),
            REVIEW_TEXT=inst.review_text,
        )
        norm = None
        if parsed is None:
            errors = [f"json_parse_error: {err}"]
        else:
            norm, errors = normalize_parsed_json(parsed, output_schema="topic_only", topics=topics)
            if norm is not None:
                errors = local_validate_task1(norm, topics=topics, review_text=inst.review_text, output_schema="topic_only")
        mentions = norm["topic_mentions"] if norm else {}
        # Topics marked true are used even if the model left other topics out
        selected = [t for t in topics if mentions.get(t) is True]
        return {"raw_output": raw, "parsed": norm, "errors": errors, "selected": selected}

    def _step_spans(inst: Task1Instance, selected: list[str]) -> dict[str, Any]:
        texts: dict[str, list[str]] = {t: [] for t in selected}
        if not selected:
            return {"raw_output": "", "parsed": {"spans": {}}, "errors": [], "texts": texts}

        raw, parsed, err = _ask(
            spans_template,
            TOPICS=", ".join(selected),
            TOPIC_COUNT=str(len(selected)),
            REVIEW_TEXT=inst.review_text,
        )
        errors: list[str] = []
        spans = parsed.get("spans") if isinstance(parsed, dict) else None
        salvaged = False
        if parsed is None:
            recovered = _salvage_spans(raw, selected)
            if recovered:
                # Answer wasn't valid JSON (e.g. cut off on a long review); keep its complete items
                spans = {t: [{"text": x} for x in xs] for t, xs in recovered.items()}
                salvaged = True
            else:
                errors = [f"json_parse_error: {err}"]
        elif not isinstance(spans, dict):
            errors = ["Missing 'spans' key"]

        if isinstance(spans, dict):
            for topic in selected:
                items = spans.get(topic)
                for item in items if isinstance(items, list) else []:
                    text = item.get("text") if isinstance(item, dict) else None
                    if isinstance(text, str) and text.strip() and text != "null" and text not in texts[topic]:
                        texts[topic].append(text)
                        if text not in inst.review_text:
                            errors.append(f"{topic}: text span '{text}' not found verbatim in review_text")

        step_parsed = None
        if isinstance(spans, dict):
            step_parsed = {"spans": {t: [{"text": x} for x in texts[t]] or [{"text": None}] for t in selected}}
        # A topic picked in step 1 with no text here is simply dropped (no rows, no error)
        return {"raw_output": raw, "parsed": step_parsed, "errors": errors, "texts": texts, "salvaged": salvaged}

    def _step_sentiments(inst: Task1Instance, texts: dict[str, list[str]]) -> dict[str, Any]:
        raw_outputs: list[str] = []
        errors: list[str] = []
        rows: list[dict[str, str]] = []
        section_fallbacks: list[str] = []
        for topic, topic_texts in texts.items():
            for text in topic_texts:
                # Full review (same as subtask1_1 gets) for context, plus the one text to label
                review_with_text = (
                    f'{inst.review_text}\n'
                    f'Selected Content ({topic}) — "{text}"\n'
                    f'Label only the Selected Content above for {topic}.'
                )
                raw, parsed, err = _ask(
                    sentiments_template,
                    TOPICS=topic,
                    TOPIC_COUNT="1",
                    REVIEW_TEXT=review_with_text,
                )
                raw_outputs.append(raw)
                sentiments = parsed.get("sentiments") if isinstance(parsed, dict) else None
                value = sentiments.get(topic) if isinstance(sentiments, dict) else None
                sentiment = ""
                if parsed is None:
                    errors.append(f"{topic}: json_parse_error: {err}")
                elif not isinstance(sentiments, dict):
                    errors.append(f"{topic}: Missing 'sentiments' key")
                elif isinstance(value, str) and value.strip().lower() in ("positive", "negative"):
                    sentiment = value.strip().capitalize()
                elif topic != "Off":
                    # No opinion words (e.g. just "Location" or "Elevator"): the model answers null.
                    # Fall back to the box it was written in: liked -> Positive, disliked -> Negative.
                    # Off with no sentiment stays blank (fillers like "Nothing at all.").
                    sentiment = _section_sentiment(inst.review_text, text)
                    if sentiment:
                        section_fallbacks.append(text)
                    else:
                        errors.append(f"{topic}: no Positive/Negative for '{text}'")
                rows.append({"topic": topic, "text": text, "sentiment": sentiment})
        parsed_out = {"annotations": [r for r in rows if r["sentiment"]]}
        return {
            "raw_output": raw_outputs,
            "parsed": parsed_out,
            "errors": errors,
            "rows": rows,
            # Texts whose sentiment came from the review box, not from the model (kept for checking)
            "section_fallbacks": section_fallbacks,
        }

    keys = [(inst.review_id, ri) for inst in instances for ri in range(1, runs_per_id + 1)]

    def _run_stage(name: str, carry_key: str, make_step) -> dict[tuple[str, int], dict[str, Any]]:
        """Run one subtask over every review/run before the next subtask starts.

        Each answer is appended to that subtask's jsonl as soon as it finishes, so a crash resumes
        inside the same subtask. carry_key is the field the next subtask needs; records without it
        (e.g. from an older version of this code) are redone.
        """
        path = step_jsonls[name]
        done = {
            (str(r["review_id"]), int(r.get("run_index", 1))): r
            for r in (_read_jsonl(path) if path.exists() else [])
            if carry_key in r
        }
        todo = [(inst, ri) for inst in instances for ri in range(1, runs_per_id + 1) if (inst.review_id, ri) not in done]
        print(f"{name}: {len(todo)} to process, {len(done)} already done")

        with path.open("a", encoding="utf-8") as f:
            for inst, ri in tqdm(todo, desc=f"Task1 sequence {name}"):
                step = make_step(inst, ri)
                rec = {
                    "review_id": inst.review_id,
                    "run_index": ri,
                    "output_schema": SEQUENCE_STEPS[name],
                    "valid": not step["errors"],
                    **step,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                done[(inst.review_id, ri)] = rec

        _write_readable_csv(path.with_suffix(".csv"), [done[k] for k in keys], instances, output_schema=SEQUENCE_STEPS[name])
        return done

    # 1. subtask1_1 for every review, 2. subtask1_2 for every review, 3. subtask1_3 for every review
    step1 = _run_stage("subtask1_1", "selected", lambda inst, ri: _step_topics(inst))
    step2 = _run_stage("subtask1_2", "texts", lambda inst, ri: _step_spans(inst, step1[(inst.review_id, ri)]["selected"]))
    step3 = _run_stage("subtask1_3", "rows", lambda inst, ri: _step_sentiments(inst, step2[(inst.review_id, ri)]["texts"]))

    # Final result: one record per review/run, rows from subtask1_3, errors from all three subtasks
    records: list[dict[str, Any]] = []
    for key in keys:
        errors: list[str] = []
        for name, stage in (("subtask1_1", step1), ("subtask1_2", step2), ("subtask1_3", step3)):
            errors += [f"{name}: {e}" for e in stage[key]["errors"]]
        records.append({
            "review_id": key[0],
            "run_index": key[1],
            "rows": step3[key]["rows"],
            "valid": len(errors) == 0,
            "errors": errors,
        })

    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _write_sequence_csv(out_jsonl.with_suffix(".csv"), records)

    return out_jsonl


def _write_sequence_csv(csv_path: Path, records: list[dict[str, Any]]) -> None:
    import csv as csv_mod

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv_mod.writer(f)
        writer.writerow(["Review ID", "Topic", "Selected Content", "Sentiment", "Errors"])
        for r in records:
            errors = "; ".join(r.get("errors", []))
            if not r["rows"]:
                writer.writerow([r["review_id"], "(no annotations found)", "", "", errors])
            for row in r["rows"]:
                writer.writerow([r["review_id"], row["topic"], row["text"], row["sentiment"], errors])

