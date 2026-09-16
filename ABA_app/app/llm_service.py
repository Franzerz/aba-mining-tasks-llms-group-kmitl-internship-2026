#### PATCHARAKORN ####

"""Plain-English graph summaries via a locally running Ollama.

Deliberately stdlib-only (urllib rather than the `ollama` package): Ollama exposes a plain
REST API, and this feature is optional — the app must run normally on a machine that has
never heard of Ollama, without adding an install step for everyone else. The OLLAMA_HOST
convention matches the mining pipeline (ABA_mining/src/llm.py), so a non-default host already
configured for that works here too.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"

# Listing decides whether the picker is shown at all, so it has to fail fast when nothing is
# listening — the sidebar must not hang on a machine without Ollama. Generation is the
# opposite, and the leash has to be long: a 9B model on this workstation took ~180s for one
# of these summaries, so anything tighter turns a working setup into a timeout.
LIST_TIMEOUT_SECONDS = 2.5
GENERATE_TIMEOUT_SECONDS = 600.0

# Low temperature: this is a factual readout of numbers already on screen, not creative
# writing, and a wandering model invents figures that contradict the picture.
GENERATE_OPTIONS = {"temperature": 0.2, "top_p": 0.9, "num_predict": 500}


def ollama_host() -> str:
    host = (os.getenv("OLLAMA_HOST") or "").strip() or DEFAULT_OLLAMA_HOST
    # OLLAMA_HOST is commonly set bare ("127.0.0.1:11434"), which urllib cannot open.
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return host.rstrip("/")


def _read_json(request: urllib.request.Request, timeout: float) -> dict:
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8") or "{}")


def list_models() -> dict:
    """{"available": bool, "models": [...], "error": str | None}.

    "Not installed", "installed but not running" and "running on another host" are all the
    same thing to the caller — there is no model list to offer — so every failure collapses
    into available=False rather than an exception.
    """
    url = f"{ollama_host()}/api/tags"
    try:
        data = _read_json(urllib.request.Request(url, method="GET"), LIST_TIMEOUT_SECONDS)
    except urllib.error.HTTPError as e:
        return {"available": False, "models": [], "error": f"Ollama returned HTTP {e.code}"}
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return {"available": False, "models": [], "error": "ollama not found"}

    models = []
    for entry in data.get("models") or []:
        name = entry.get("name") or entry.get("model")
        if name:
            models.append(str(name))
    return {"available": True, "models": sorted(set(models)), "error": None}


def generate(model: str, prompt: str) -> str:
    """One non-streaming completion. Raises RuntimeError with a display-ready message."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": dict(GENERATE_OPTIONS),
        # Hybrid-reasoning models (the Qwen3 family) otherwise burn real generation time on
        # thinking tokens before answering — see the same note in ABA_mining/src/llm.py.
        "think": False,
    }
    url = f"{ollama_host()}/api/generate"

    def call(body: dict) -> dict:
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return _read_json(request, GENERATE_TIMEOUT_SECONDS)

    try:
        data = call(payload)
    except urllib.error.HTTPError as e:
        # Ollama builds predating the `think` field reject the whole request rather than
        # ignoring the unknown key, so retry once without it before giving up.
        if e.code == 400:
            retry = {k: v for k, v in payload.items() if k != "think"}
            try:
                data = call(retry)
            except urllib.error.HTTPError as e2:
                raise RuntimeError(f"Ollama returned HTTP {e2.code} for model '{model}'.") from e2
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e2:
                raise RuntimeError(f"Could not reach Ollama: {e2}") from e2
        else:
            raise RuntimeError(f"Ollama returned HTTP {e.code} for model '{model}'.") from e
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        raise RuntimeError(f"Could not reach Ollama: {e}") from e

    text = (data.get("response") or "").strip()
    if not text:
        raise RuntimeError(f"Model '{model}' returned an empty summary.")
    return text


# --- prompt ----------------------------------------------------------------------------------
# The reader is a hotel manager, not a logician: every word of framework vocabulary that leaks
# into the answer makes it useless to them. The facts are therefore pre-translated into plain
# wording HERE, so the model is never shown the jargon it might otherwise echo back.

_BANNED_WORDS = (
    "assumption, contrary, atom, node, edge, literal, clone, extension, defeat, semantics, "
    "argumentation, framework, ABA, AA, Dung, instantiation, premise, rule, graph theory"
)


def _plural(n: int, singular: str, plural: str | None = None) -> str:
    return f"{n:,} {singular if n == 1 else (plural or singular + 's')}"


def describe_facts(facts: dict) -> str:
    """The on-screen numbers, already phrased the way we want them read back."""
    counts = facts.get("counts") or {}
    edges = facts.get("edges") or {}
    topic = facts.get("aspect") or "this topic"
    verdicts = [str(h) for h in (facts.get("headLabels") or [])]
    themes = [str(t) for t in (facts.get("topThemes") or [])]

    lines = [
        f"- Hotel topic under review: {topic}.",
        f"- Overall verdicts being weighed up: {', '.join(verdicts) if verdicts else 'none shown'}.",
        f"- Points taken from guest reviews and shown on screen: {counts.get('points', 0):,}.",
        f"- Connections where a point backs up a verdict: {edges.get('support', 0):,}.",
        f"- Connections where two points contradict each other: {edges.get('conflict', 0):,}.",
    ]
    if themes:
        lines.append(f"- The themes guests mentioned most: {', '.join(themes)}.")
    if facts.get("focusLabel"):
        lines.append(f"- The manager has narrowed the picture to just: {facts['focusLabel']}.")
    if facts.get("headLabel"):
        lines.append(f"- The manager is looking only at the verdict: {facts['headLabel']}.")
    extensions = facts.get("extensions") or {}
    if extensions.get("count"):
        lines.append(
            f"- Sets of points that can all be believed together without contradiction: "
            f"{extensions['count']:,} (the largest holds {_plural(extensions.get('largest', 0), 'point')})."
        )
    return "\n".join(lines)


def build_summary_prompt(facts: dict) -> str:
    return f"""You are explaining a picture on a screen to a hotel manager. They are good at running a hotel and know nothing about logic, data science or diagrams. They want to know what guests said and where guests disagreed.

Here is what the picture currently shows:
{describe_facts(facts)}

Write the explanation now, following every rule:
- Write 4 to 6 bullet points. Start each one with "- ".
- One short sentence per bullet, in plain everyday English a busy manager can skim.
- Use the numbers above, exactly as given. Never invent a number that is not listed.
- Say it in hotel terms: guest comments, praise, complaints, agreement, disagreement, what to look into.
- Never use any of these words: {_BANNED_WORDS}.
- End with one bullet suggesting what the manager could take away from it.
- Output only the bullet points. No title, no introduction, no closing remark, no markdown headings."""
