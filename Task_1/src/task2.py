from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from jsonschema import Draft202012Validator
from tqdm import tqdm

from .config import ModelConfig, PathsConfig
from .llm import LLMClient
from .prompts import load_prompt, render_prompt
from .utils import try_parse_json


@dataclass(frozen=True)
class BodyLabelsConfig:
    vocab: dict[str, dict[str, list[str]]]  # topic -> {"body": [...], "cont": [...]}

    def body_labels(self, topic: str) -> list[str]:
        return self.vocab.get(topic, {}).get("body", [])

    def cont_labels(self, topic: str) -> list[str]:
        return self.vocab.get(topic, {}).get("cont", [])


def load_body_labels_config(repo_root: Path) -> BodyLabelsConfig:
    path = repo_root / "configs" / "body_labels.yaml"
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return BodyLabelsConfig(vocab=raw)


@dataclass(frozen=True)
class Task2Instance:
    review_id: str
    topic: str
    selected_content: str
    body_vocab: list[str]
    cont_vocab: list[str]
    source: str  # "gt" or "llm"
    run: int | None = None  # LLM run number (None for source="gt")


def build_task2_schema(body_labels: list[str], cont_labels: list[str]) -> dict[str, Any]:
    all_body = list(body_labels)
    all_cont = list(cont_labels)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "body_labels": {
                "type": "array",
                "items": {"type": "string", "enum": all_body} if all_body else {"type": "string"},
            },
            "cont_labels": {
                "type": "array",
                "items": {"type": "string", "enum": all_cont} if all_cont else {"type": "string"},
            },
        },
        "required": ["body_labels", "cont_labels"],
        "additionalProperties": False,
    }


def local_validate_task2(
    parsed: Any,
    *,
    body_labels: list[str],
    cont_labels: list[str],
) -> list[str]:
    errors: list[str] = []
    schema = build_task2_schema(body_labels, cont_labels)
    # Only enforce required/type structure — skip enum validation for empty vocabs
    simple_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "body_labels": {"type": "array", "items": {"type": "string"}},
            "cont_labels": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["body_labels", "cont_labels"],
        "additionalProperties": False,
    }
    validator = Draft202012Validator(simple_schema)
    for e in validator.iter_errors(parsed):
        errors.append(e.message)
    if errors:
        return errors

    body_set = set(body_labels)
    cont_set = set(cont_labels)

    for lbl in parsed.get("body_labels", []):
        if body_set and lbl not in body_set:
            errors.append(f"body_labels: '{lbl}' not in vocabulary")
    for lbl in parsed.get("cont_labels", []):
        if cont_set and lbl not in cont_set:
            errors.append(f"cont_labels: '{lbl}' not in vocabulary")

    return errors


def load_task2_instances_gt(
    gold_csv: Path,
    body_labels_cfg: BodyLabelsConfig,
    limit: int | None = None,
    offset: int = 0,
) -> list[Task2Instance]:
    """One instance per (Review ID, Topic) row in the GT CSV using GT selected content."""
    gt = pd.read_csv(gold_csv)
    gt = gt.rename(columns={"ID": "Review ID"})
    gt = gt[gt["Topic"].notna() & gt["Selected Content"].notna()].copy()
    gt["Topic"] = gt["Topic"].str.strip()
    gt = gt[gt["Selected Content"].str.strip() != ""].reset_index(drop=True)

    instances: list[Task2Instance] = []
    seen: set[tuple] = set()
    skipped = 0
    for _, row in gt.iterrows():
        rid = str(row["Review ID"])
        topic = row["Topic"]
        content = str(row["Selected Content"]).strip()
        key = (rid, topic, content)
        if key in seen:
            continue
        seen.add(key)

        if skipped < offset:
            skipped += 1
            continue

        body_vocab = body_labels_cfg.body_labels(topic)
        cont_vocab  = body_labels_cfg.cont_labels(topic)
        instances.append(Task2Instance(
            review_id=rid,
            topic=topic,
            selected_content=content,
            body_vocab=body_vocab,
            cont_vocab=cont_vocab,
            source="gt",
        ))
        if limit is not None and len(instances) >= limit:
            break

    return instances


def load_task2_instances_llm(
    llm_csv: Path,
    body_labels_cfg: BodyLabelsConfig,
    limit: int | None = None,
    offset: int = 0,
) -> list[Task2Instance]:
    """One instance per (Review ID, Run, Topic) in the LLM Task 1.2 output CSV."""
    raw = pd.read_csv(llm_csv)
    if "Review ID" not in raw.columns and "ID" in raw.columns:
        raw = raw.rename(columns={"ID": "Review ID"})

    _bad = {"parse failed", "topics not found", "topic not found", "(parse failed)"}
    raw = raw[
        raw["Topic"].notna() &
        (raw["Topic"].str.strip() != "") &
        (~raw["Topic"].str.strip().str.lower().isin(_bad)) &
        raw["Text Span"].notna() &
        (raw["Text Span"].str.strip() != "")
    ].copy()
    if "Errors" in raw.columns:
        raw = raw[raw["Errors"].isna() | raw["Errors"].astype(str).str.strip().isin({"", "nan"})].copy()

    raw["Topic"] = raw["Topic"].str.strip()
    raw = raw.drop_duplicates(subset=["Review ID", "Run", "Topic", "Text Span"]).reset_index(drop=True)

    instances: list[Task2Instance] = []
    skipped = 0
    for _, row in raw.iterrows():
        if skipped < offset:
            skipped += 1
            continue

        rid = str(int(row["Review ID"])) if str(row["Review ID"]).replace(".0", "").isdigit() else str(row["Review ID"])
        topic = row["Topic"]
        content = str(row["Text Span"]).strip()
        run = int(row["Run"]) if "Run" in raw.columns and pd.notna(row["Run"]) else None

        body_vocab = body_labels_cfg.body_labels(topic)
        cont_vocab  = body_labels_cfg.cont_labels(topic)
        instances.append(Task2Instance(
            review_id=rid,
            topic=topic,
            selected_content=content,
            body_vocab=body_vocab,
            cont_vocab=cont_vocab,
            source="llm",
            run=run,
        ))
        if limit is not None and len(instances) >= limit:
            break

    return instances


def run_task2(
    *,
    repo_root: Path,
    client: LLMClient,
    body_labels_cfg: BodyLabelsConfig,
    model_cfg: ModelConfig,
    paths_cfg: PathsConfig,
    instances: list[Task2Instance],
    source: str,
    prompt_path: str = "prompts/task2/generator_v1.txt",
    max_retries: int = 1,
    output_subdir: str | None = None,
    label: str = "task2",
) -> Path:
    out_dir = paths_cfg.task1_dir.parent / "task2"
    if output_subdir:
        out_dir = out_dir / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = model_cfg.task1_model.replace(":", "_").replace("/", "_")
    out_path = out_dir / f"task2_{model_name}_{label}_{source}_n{len(instances)}.jsonl"

    gen_template = load_prompt(repo_root, prompt_path)

    def _format_label_list(labels: list[str]) -> str:
        if not labels:
            return "(none)"
        return "\n".join(f"- {lbl}" for lbl in labels)

    def _generate(inst: Task2Instance) -> tuple[str, bool, Any, list[str]]:
        body_str = _format_label_list(inst.body_vocab)
        cont_str = _format_label_list(inst.cont_vocab)
        prompt = render_prompt(
            gen_template,
            TOPIC=inst.topic,
            SELECTED_CONTENT=inst.selected_content,
            BODY_LABELS=body_str,
            CONT_LABELS=cont_str,
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
            errors = local_validate_task2(
                parsed,
                body_labels=inst.body_vocab,
                cont_labels=inst.cont_vocab,
            )
        return resp.text, ok, parsed, errors

    def _process_one(inst: Task2Instance) -> dict[str, Any]:
        raw, ok, parsed, errors = _generate(inst)
        attempt = 0

        while errors and attempt < max_retries:
            attempt += 1
            raw, ok, parsed, errors = _generate(inst)
            if ok and not errors:
                break

        compact_raw = json.dumps(parsed, ensure_ascii=False) if ok and parsed is not None else raw

        return {
            "review_id": inst.review_id,
            "topic": inst.topic,
            "selected_content": inst.selected_content,
            "source": inst.source,
            "run": inst.run,
            "raw_output": compact_raw,
            "parsed": parsed if ok else None,
            "valid": ok and len(errors) == 0,
            "errors": errors,
            "retries": attempt,
        }

    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: list[dict[str, Any]] = [None] * len(instances)
    with ThreadPoolExecutor(max_workers=model_cfg.num_workers) as ex:
        fut_to_idx = {ex.submit(_process_one, inst): i for i, inst in enumerate(instances)}
        for fut in tqdm(as_completed(fut_to_idx), total=len(instances), desc=f"Task2[{source}]"):
            i = fut_to_idx[fut]
            results[i] = fut.result()

    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    csv_path = out_path.with_suffix(".csv")
    _write_task2_csv(csv_path, results)

    print(f"Wrote JSONL: {out_path}")
    print(f"Wrote CSV:   {csv_path}")
    return out_path


def _write_task2_csv(csv_path: Path, results: list[dict[str, Any]]) -> None:
    import csv as csv_mod

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv_mod.writer(f)
        writer.writerow(["Review ID", "Run", "Source", "Topic", "Selected Content", "Label", "Label Type", "Valid", "Errors"])

        for r in results:
            review_id = r["review_id"]
            run = r.get("run", "")
            source = r.get("source", "")
            topic = r["topic"]
            content = r["selected_content"]
            valid = r["valid"]
            errors = "; ".join(r.get("errors", []))

            parsed = r.get("parsed")
            if parsed and isinstance(parsed, dict):
                wrote_any = False
                for lbl in parsed.get("body_labels", []):
                    writer.writerow([review_id, run, source, topic, content, lbl, "body", valid, ""])
                    wrote_any = True
                for lbl in parsed.get("cont_labels", []):
                    writer.writerow([review_id, run, source, topic, content, lbl, "cont", valid, ""])
                    wrote_any = True
                if not wrote_any:
                    writer.writerow([review_id, run, source, topic, content, "(no labels)", "", valid, errors])
            else:
                writer.writerow([review_id, run, source, topic, content, "(parse failed)", "", valid, errors])
