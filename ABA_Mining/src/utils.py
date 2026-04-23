from __future__ import annotations

import json
import re
from typing import Any


def try_parse_json(text: str) -> tuple[bool, Any | None, str | None]:
    # Try direct parse first
    try:
        return True, json.loads(text), None
    except Exception:
        pass

    # Try to salvage: strip markdown fences, fix common LLM JSON errors
    cleaned = text.strip()
    # Remove markdown code fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```\w*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip()

    # Common LLM error: extra ] before closing }} — e.g. ...}]}]} instead of ...}]}}
    # Try replacing known bad patterns
    for bad, good in [
        ("}]}]}", "}]}}"),     # extra ] wrapping Topics value
        ("]}]}", "}}"),        # wraps Topics in extra array
        ("]}}}", "}}"),        # extra closing braces
        ("]}}", "}}"),         # extra bracket
    ]:
        if cleaned.endswith(bad):
            attempt = cleaned[:-len(bad)] + good
            try:
                return True, json.loads(attempt), None
            except Exception:
                pass

    # Try progressively removing trailing characters that break JSON
    for _ in range(5):
        try:
            return True, json.loads(cleaned), None
        except json.JSONDecodeError as e:
            msg = str(e)
            if "Extra data" in msg or "Expecting ',' delimiter" in msg:
                cleaned = cleaned.rstrip()
                if cleaned and cleaned[-1] in "]})\"',;":
                    cleaned = cleaned[:-1]
                else:
                    break
            else:
                break

    # Final attempt
    try:
        return True, json.loads(cleaned), None
    except Exception as e:
        return False, None, str(e)


_word_re = re.compile(r"[A-Za-z0-9]+")


def extract_vocab(text: str) -> list[str]:
    # Lowercased unique word tokens in appearance order.
    seen: set[str] = set()
    out: list[str] = []
    for m in _word_re.finditer(text.lower()):
        tok = m.group(0)
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def normalize_topic_for_head(topic: str) -> str:
    t = topic.strip().lower()
    t = t.replace(" ", "_")
    t = t.replace("-", "_")
    if t in {"off", "off_topic", "offtopic"}:
        return "off_topic"
    return t

