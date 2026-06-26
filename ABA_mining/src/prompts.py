#### PATCHARAKORN ####

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


def build_modular_prompt(repo_root: Path, rules: list[int], output_schema: str = "full") -> str:
    """Assemble a prompt from header + selected rule files + footer."""
    prompts_dir = repo_root / "prompts" / "task1"
    rules_dir = prompts_dir / "rules"

    _schema_files = {
        "topic_only":     ("header_topic_only.txt",     "footer_topic_only.txt"),
        "span_only":      ("header_span_only.txt",      "footer_span_only.txt"),
        "sentiment_only": ("header_sentiment_only.txt", "footer_sentiment_only.txt"),
        "full":           ("header.txt",                "footer.txt"),
    }
    header_file, footer_file = _schema_files.get(output_schema, ("header.txt", "footer.txt"))
    header = (prompts_dir / header_file).read_text(encoding="utf-8")
    footer = (prompts_dir / footer_file).read_text(encoding="utf-8")

    rule_blocks: list[str] = []
    for r in rules:
        rule_file = rules_dir / f"rule{r}.txt"
        if not rule_file.exists():
            raise FileNotFoundError(f"Rule file not found: {rule_file}")
        rule_blocks.append(rule_file.read_text(encoding="utf-8").strip())

    return header.rstrip() + "\n\n" + "\n\n".join(rule_blocks) + "\n" + footer
