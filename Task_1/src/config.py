from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TopicsConfig:
    active_schema: str
    topics: list[str]
    off_topic_labels: list[str]


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    task1_model: str
    validator_model: str
    temperature: float
    top_p: float
    max_output_tokens: int
    max_attempts: int
    num_workers: int
    ollama_options: dict[str, Any]


@dataclass(frozen=True)
class PathsConfig:
    input_csv: Path
    gold_csv: Path
    task1_dir: Path
    eval_dir: Path


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_topics_config(repo_root: Path) -> TopicsConfig:
    cfg = _load_yaml(repo_root / "configs" / "topics.yaml")
    active = cfg["active_schema"]
    schema = cfg["schemas"][active]
    return TopicsConfig(
        active_schema=active,
        topics=list(schema["topics"]),
        off_topic_labels=list(schema.get("off_topic_labels", [])),
    )


def load_model_config(repo_root: Path) -> ModelConfig:
    cfg = _load_yaml(repo_root / "configs" / "model.yaml")
    gen = cfg["generation"]
    models = cfg["models"]
    retries = cfg.get("retries", {})
    perf = cfg.get("performance", {})
    ollama_cfg = cfg.get("ollama", {})
    return ModelConfig(
        provider=str(cfg["provider"]),
        task1_model=str(models["task1"]),
        validator_model=str(models["validator"]),
        temperature=float(gen["temperature"]),
        top_p=float(gen["top_p"]),
        max_output_tokens=int(gen.get("max_output_tokens", 800)),
        max_attempts=int(retries.get("max_attempts", 3)),
        num_workers=int(perf.get("num_workers", 16)),
        ollama_options=dict(ollama_cfg.get("options", {}) or {}),
    )


def load_paths_config(repo_root: Path) -> PathsConfig:
    cfg = _load_yaml(repo_root / "configs" / "paths.yaml")
    inp = cfg["input"]
    out = cfg["output"]
    return PathsConfig(
        input_csv=(repo_root / inp["input_csv"]).resolve(),
        gold_csv=(repo_root / inp["gold_csv"]).resolve(),
        task1_dir=(repo_root / out["task1_dir"]).resolve(),
        eval_dir=(repo_root / out["eval_dir"]).resolve(),
    )

