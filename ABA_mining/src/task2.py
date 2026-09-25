#### PATCHARAKORN ####

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from jsonschema import Draft202012Validator
from tqdm import tqdm

from .config import ModelConfig, PathsConfig
from .llm import LLMClient
from .prompts import load_prompt, render_prompt
from .utils import try_parse_json


@dataclass(frozen=True)
class Task2Instance:
    review_id: str
    topic: str
    selected_content: str
    sentiment: str          # "Positive" or "Negative"
    source: str             # "gt" or "llm"
    run: int | None = None  # LLM run number (None for source="gt")


_TASK2_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "body_literals": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["body_literals"],
    "additionalProperties": False,
}


def local_validate_task2(parsed: Any) -> list[str]:
    errors: list[str] = []
    validator = Draft202012Validator(_TASK2_SCHEMA)
    for e in validator.iter_errors(parsed):
        errors.append(e.message)
    return errors


def make_head(topic: str, sentiment: str) -> str:
    """Generate inference rule head: good_{topic} or bad_{topic}."""
    polarity = "good" if sentiment.strip().lower() == "positive" else "bad"
    t = topic.strip().lower().replace(" ", "_").replace("-", "_")
    return f"{polarity}_{t}"


def make_contrary_literals(body_literals: list[str], sentiment: str) -> list[str]:
    """Derive contrary literals from supporting ones based on sentiment."""
    if not body_literals:
        return []
    if sentiment.strip().lower() == "positive":
        return [f"no_evident_not_{lit}" for lit in body_literals]
    else:
        return [f"have_evident_{lit}" for lit in body_literals]


def load_task2_instances_gt(
    gold_csv: Path,
    limit: int | None = None,
    offset: int = 0,
) -> list[Task2Instance]:
    """Load instances from GT CSV.

    - Excludes 'Off' topic rows (professor's instruction)
    - Keeps BOTH rows when the same text span has Positive AND Negative sentiment
      (contrastive spans: e.g. "clean but not big" → one Positive row + one Negative row)
    - True duplicates (identical ID+Topic+Content+Sentiment) are dropped
    """
    gt = pd.read_csv(gold_csv)
    gt = gt.rename(columns={"ID": "Review ID", "Pos/Neg": "Sentiment"})
    gt = gt[
        gt["Topic"].notna() &
        gt["Selected Content"].notna() &
        gt["Sentiment"].notna()
    ].copy()
    gt["Topic"] = gt["Topic"].str.strip()
    gt["Sentiment"] = gt["Sentiment"].str.strip()
    gt = gt[
        (gt["Selected Content"].str.strip() != "") &
        (gt["Topic"] != "Off")                        # remove Off topic
    ].reset_index(drop=True)

    # Drop true duplicates (same ID + Topic + Content + Sentiment)
    gt = gt.drop_duplicates(subset=["Review ID", "Topic", "Selected Content", "Sentiment"])

    instances: list[Task2Instance] = []
    skipped = 0

    for _, row in gt.iterrows():
        if skipped < offset:
            skipped += 1
            continue

        instances.append(Task2Instance(
            review_id=str(row["Review ID"]),
            topic=row["Topic"],
            selected_content=str(row["Selected Content"]).strip(),
            sentiment=row["Sentiment"],
            source="gt",
        ))
        if limit is not None and len(instances) >= limit:
            break

    return instances


def load_task2_instances_from_subtasks(
    csv_1_2: Path,
    csv_1_3: Path,
    limit: int | None = None,
    offset: int = 0,
) -> list[Task2Instance]:
    """Load instances by joining subtask 1.2 (Selected Content) + subtask 1.3 (Sentiment)
    on (Review ID, Run, Topic). Preferred over combined output when individual
    subtask runs produce better Task 1 evaluation results."""
    _bad = {"parse failed", "topics not found", "topic not found", "(parse failed)"}

    df2 = pd.read_csv(csv_1_2)
    if "Review ID" not in df2.columns and "ID" in df2.columns:
        df2 = df2.rename(columns={"ID": "Review ID"})
    # Task 1 outputs written before the header rename still say "Text Span"
    df2 = df2.rename(columns={"Text Span": "Selected Content"})
    df2 = df2[
        df2["Topic"].notna() &
        (df2["Topic"].str.strip() != "") &
        (~df2["Topic"].str.strip().str.lower().isin(_bad)) &
        df2["Selected Content"].notna() &
        (df2["Selected Content"].str.strip() != "")
    ].copy()
    if "Errors" in df2.columns:
        df2 = df2[df2["Errors"].isna() | df2["Errors"].astype(str).str.strip().isin({"", "nan"})].copy()
    df2["Topic"] = df2["Topic"].str.strip()
    df2 = df2[["Review ID", "Run", "Topic", "Selected Content"]].drop_duplicates(
        subset=["Review ID", "Run", "Topic", "Selected Content"]
    )

    df3 = pd.read_csv(csv_1_3)
    if "Review ID" not in df3.columns and "ID" in df3.columns:
        df3 = df3.rename(columns={"ID": "Review ID"})
    df3 = df3[
        df3["Topic"].notna() &
        (df3["Topic"].str.strip() != "") &
        (~df3["Topic"].str.strip().str.lower().isin(_bad)) &
        df3["Sentiment"].notna() &
        (df3["Sentiment"].str.strip() != "")
    ].copy()
    if "Errors" in df3.columns:
        df3 = df3[df3["Errors"].isna() | df3["Errors"].astype(str).str.strip().isin({"", "nan"})].copy()
    df3["Topic"] = df3["Topic"].str.strip()
    df3["Sentiment"] = df3["Sentiment"].str.strip()
    df3 = df3[["Review ID", "Run", "Topic", "Sentiment"]].drop_duplicates(
        subset=["Review ID", "Run", "Topic"]
    )

    raw = df2.merge(df3, on=["Review ID", "Run", "Topic"], how="inner")
    raw = raw.drop_duplicates(subset=["Review ID", "Run", "Topic", "Selected Content"]).reset_index(drop=True)

    instances: list[Task2Instance] = []
    skipped = 0
    for _, row in raw.iterrows():
        if skipped < offset:
            skipped += 1
            continue
        rid = str(row["Review ID"]).replace(".0", "") if str(row["Review ID"]).endswith(".0") else str(row["Review ID"])
        run = int(row["Run"]) if pd.notna(row["Run"]) else None
        instances.append(Task2Instance(
            review_id=rid,
            topic=row["Topic"],
            selected_content=str(row["Selected Content"]).strip(),
            sentiment=row["Sentiment"],
            source="llm",
            run=run,
        ))
        if limit is not None and len(instances) >= limit:
            break
    return instances


def load_task2_instances_llm(
    llm_csv: Path,
    limit: int | None = None,
    offset: int = 0,
) -> list[Task2Instance]:
    """Load instances from LLM combined output (Selected Content + Sentiment columns)."""
    raw = pd.read_csv(llm_csv)
    if "Review ID" not in raw.columns and "ID" in raw.columns:
        raw = raw.rename(columns={"ID": "Review ID"})
    # Task 1 outputs written before the header rename still say "Text Span"
    raw = raw.rename(columns={"Text Span": "Selected Content"})

    if "Selected Content" not in raw.columns or "Sentiment" not in raw.columns:
        raise ValueError(
            f"LLM CSV must have 'Selected Content' and 'Sentiment' columns.\n"
            f"Found: {list(raw.columns)}\n"
            f"Use load_task2_instances_from_subtasks instead."
        )

    _bad = {"parse failed", "topics not found", "topic not found", "(parse failed)"}
    raw = raw[
        raw["Topic"].notna() &
        (raw["Topic"].str.strip() != "") &
        (~raw["Topic"].str.strip().str.lower().isin(_bad)) &
        raw["Selected Content"].notna() &
        (raw["Selected Content"].str.strip() != "") &
        raw["Sentiment"].notna() &
        (raw["Sentiment"].str.strip() != "")
    ].copy()
    if "Errors" in raw.columns:
        raw = raw[raw["Errors"].isna() | raw["Errors"].astype(str).str.strip().isin({"", "nan"})].copy()

    raw["Topic"] = raw["Topic"].str.strip()
    raw["Sentiment"] = raw["Sentiment"].str.strip()
    raw = raw.drop_duplicates(subset=["Review ID", "Run", "Topic", "Selected Content"]).reset_index(drop=True)

    instances: list[Task2Instance] = []
    skipped = 0

    for _, row in raw.iterrows():
        if skipped < offset:
            skipped += 1
            continue

        rid = str(row["Review ID"]).replace(".0", "") if str(row["Review ID"]).endswith(".0") else str(row["Review ID"])
        topic = row["Topic"]
        content = str(row["Selected Content"]).strip()
        sentiment = row["Sentiment"]
        run = int(row["Run"]) if "Run" in raw.columns and pd.notna(row["Run"]) else None

        instances.append(Task2Instance(
            review_id=rid,
            topic=topic,
            selected_content=content,
            sentiment=sentiment,
            source="llm",
            run=run,
        ))
        if limit is not None and len(instances) >= limit:
            break

    return instances


def _instance_key(inst: Task2Instance) -> tuple:
    return (inst.review_id, inst.topic, inst.selected_content, inst.sentiment, inst.source, inst.run)


def _result_key(r: dict[str, Any]) -> tuple:
    return (
        r.get("review_id"), r.get("topic"), r.get("selected_content"),
        r.get("sentiment"), r.get("source"), r.get("run"),
    )


def _load_checkpoint(out_path: Path) -> dict[tuple, dict[str, Any]]:
    """Load already-completed rows from a previous (possibly SLURM-killed) run.

    Reads whatever was flushed to disk so far. A truncated last line (job killed
    mid-write) is skipped rather than crashing the resume.
    """
    existing: dict[tuple, dict[str, Any]] = {}
    if not out_path.exists():
        return existing
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            existing[_result_key(r)] = r
    return existing


def run_task2(
    *,
    repo_root: Path,
    client: LLMClient,
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
    csv_path = out_path.with_suffix(".csv")

    # --- Checkpoint / resume -------------------------------------------
    # out_path also serves as the checkpoint file: every finished row is
    # flushed to it immediately, so a SLURM job that gets killed mid-run can
    # simply be resubmitted with the same args and pick up where it left off.
    existing_results = _load_checkpoint(out_path)
    pending = [inst for inst in instances if _instance_key(inst) not in existing_results]

    if not pending:
        print(f"[checkpoint] {out_path.name} already complete ({len(existing_results)}/{len(instances)}) — skipping generation")
        ordered_results = [existing_results[_instance_key(inst)] for inst in instances]
        _write_task2_csv(csv_path, ordered_results)
        print(f"Wrote CSV:   {csv_path}")
        return out_path

    if existing_results:
        print(f"[checkpoint] resuming {out_path.name}: {len(instances) - len(pending)}/{len(instances)} already done, {len(pending)} remaining")

    gen_template = load_prompt(repo_root, prompt_path)

    def _generate(inst: Task2Instance) -> tuple[str, bool, Any, list[str]]:
        prompt = render_prompt(
            gen_template,
            TOPIC=inst.topic,
            SENTIMENT=inst.sentiment,
            SELECTED_CONTENT=inst.selected_content,
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
            errors = local_validate_task2(parsed)
        return resp.text, ok, parsed, errors

    def _process_one(inst: Task2Instance) -> dict[str, Any]:
        raw, ok, parsed, errors = _generate(inst)
        attempt = 0

        while errors and attempt < max_retries:
            attempt += 1
            raw, ok, parsed, errors = _generate(inst)
            if ok and not errors:
                break

        body_literals: list[str] = []
        cont_literals: list[str] = []
        head: str = make_head(inst.topic, inst.sentiment)

        if ok and parsed and not errors:
            body_literals = parsed.get("body_literals", [])
            body_literals = [str(b).strip() for b in body_literals if str(b).strip()]
            cont_literals = make_contrary_literals(body_literals, inst.sentiment)

        compact_raw = json.dumps(parsed, ensure_ascii=False) if ok and parsed is not None else raw

        return {
            "review_id": inst.review_id,
            "topic": inst.topic,
            "selected_content": inst.selected_content,
            "sentiment": inst.sentiment,
            "source": inst.source,
            "run": inst.run,
            "head": head,
            "body_literals": body_literals,
            "cont_literals": cont_literals,
            "raw_output": compact_raw,
            "valid": ok and len(errors) == 0,
            "errors": errors,
            "retries": attempt,
        }

    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Append mode when resuming an existing (partial) checkpoint file, write
    # mode when starting fresh. Each row is written and fsync'd as soon as it
    # finishes — not buffered until the whole batch is done — so progress
    # survives a SLURM time-limit kill.
    file_mode = "a" if out_path.exists() else "w"
    with out_path.open(file_mode, encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=model_cfg.num_workers) as ex:
        fut_to_inst = {ex.submit(_process_one, inst): inst for inst in pending}
        for fut in tqdm(as_completed(fut_to_inst), total=len(pending), desc=f"Task2[{source}]"):
            inst = fut_to_inst[fut]
            r = fut.result()
            existing_results[_instance_key(inst)] = r
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    ordered_results = [existing_results[_instance_key(inst)] for inst in instances]
    _write_task2_csv(csv_path, ordered_results)

    print(f"Wrote JSONL: {out_path}")
    print(f"Wrote CSV:   {csv_path}")
    return out_path


def _write_task2_csv(csv_path: Path, results: list[dict[str, Any]]) -> None:
    import csv as csv_mod

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv_mod.writer(f)
        writer.writerow([
            "Review ID", "Run", "Source", "Topic", "Sentiment",
            "Head", "Selected Content", "Literal", "Literal Type",
            "Valid", "Errors",
        ])

        for r in results:
            base = [
                r["review_id"], r.get("run", ""), r.get("source", ""),
                r["topic"], r.get("sentiment", ""),
                r.get("head", ""), r["selected_content"],
            ]
            errors = "; ".join(r.get("errors", []))
            valid = r["valid"]

            body = r.get("body_literals", [])
            cont = r.get("cont_literals", [])

            if not body and not cont:
                writer.writerow(base + ["(no literals)", "", valid, errors])
            else:
                for lit in body:
                    writer.writerow(base + [lit, "body", valid, ""])
                for lit in cont:
                    writer.writerow(base + [lit, "cont", valid, ""])
