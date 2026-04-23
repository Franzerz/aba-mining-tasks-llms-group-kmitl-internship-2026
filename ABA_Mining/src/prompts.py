from __future__ import annotations

from pathlib import Path


def load_prompt(repo_root: Path, relative_path: str) -> str:
    path = repo_root / relative_path
    return path.read_text(encoding="utf-8")


def render_prompt(template: str, **kwargs: str) -> str:
    out = template
    for k, v in kwargs.items():
        out = out.replace("{{" + k + "}}", v)
    return out

