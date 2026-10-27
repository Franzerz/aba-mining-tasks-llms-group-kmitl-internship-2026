#### PATCHARAKORN ####

"""Port of mu_team's LLM explanation path (buildTranslatePrompt + translateWithOllama).

The prompts are copied verbatim — including their wording, ordering and rules — because the
explanation text is part of what is being compared. Same Ollama /api/chat call, same
temperature, same model allowlist and same fallback model.
"""

from __future__ import annotations

import json
import re
import time

import httpx

from .. import settings


def build_translate_prompt(body: dict | None) -> tuple[str, str]:
    payload = body if isinstance(body, dict) else {}
    task = str(payload.get("task") or "translate_extension").strip().lower()
    topic = str(payload.get("topic") or "")
    sentiment = str(payload.get("sentiment") or "")
    supporting = str(payload.get("supporting") or "")
    raw_output_language = str(payload.get("outputLanguage") or "English").strip()
    target_language = "English" if re.match(r"^en\b", raw_output_language, re.I) else raw_output_language

    if task == "graph_summary":
        graph_nodes = payload.get("graphNodes") if isinstance(payload.get("graphNodes"), list) else []
        graph_edges = payload.get("graphEdges") if isinstance(payload.get("graphEdges"), list) else []
        graph_edge_stats = payload.get("graphEdgeStats") if isinstance(payload.get("graphEdgeStats"), dict) else {}

        system_prompt = """You are a helpful assistant explaining the outcome of an argument analysis to a general audience.
The analysis evaluated reasons for and against a conclusion.

Your role is to translate this into plain, natural language — like a knowledgeable friend summarizing both sides of an issue.

Strict language rules:
- NEVER output snake_case text (e.g., "allow_to_check_out_later"). Always convert to natural readable phrases (e.g., "being allowed to check out later").
- NEVER use words like: node, proposition, assumption, edge, claim, support edge, attack edge, graph.
- NEVER say "the graph shows" or "the graph argues". Just state the ideas directly.
- Write as if explaining to someone who has never seen the analysis tool.
- Output exactly one paragraph. No headings, no bullet points."""

        lines = [
            f"Output language: {target_language}",
            f"Topic being analyzed: {topic}" if topic else "",
            f"Main conclusion being evaluated: {sentiment}" if sentiment else "",
            f"Key supporting idea to highlight: {supporting}" if supporting else "",
            "",
            f"Argument nodes: {_json(graph_nodes)}",
            f"Argument connections: {_json(graph_edges)}",
            f"Connection statistics: {_json(graph_edge_stats)}",
            "",
            "Write one plain-language paragraph for a general reader. Follow this exact structure:",
            "",
            "1. STATE the main conclusion clearly in everyday language.",
            "2. EXPLAIN the strongest reasons that support it — rephrase any snake_case names into natural phrases.",
            "3. EXPLAIN the main opposing argument(s), if any — in plain language.",
            "4. END with whether this conclusion is strong, uncertain, or depends on the situation — based on how balanced the two sides are.",
            "",
            "Rules:",
            "- One paragraph only.",
            "- Convert every snake_case identifier into readable text. Example: 'late_departure_no_extra_cost' -> 'no extra charge for a late departure'.",
            "- Never mention node names, IDs, or any technical labels in your output.",
            "- Be specific — name the actual ideas, not vague phrases like 'several positive factors' or 'some concerns'.",
            "- Do not give advice or recommendations unless the argument itself contains them.",
            "- If supporting reasons clearly outweigh opposition -> say the conclusion is well-supported.",
            "- If both sides carry significant weight -> say the conclusion is debatable or depends on circumstances.",
            "- One paragraph only. No lists. No headings.",
        ]
        # Their `.filter(Boolean)` drops empty strings, so the "" spacers above never make it
        # into the prompt — keep that exactly, or the prompt text would differ.
        return system_prompt, "\n".join(x for x in lines if x)

    extensions = payload.get("extensions") if isinstance(payload.get("extensions"), list) else []
    accepted = payload.get("acceptedAssumptions") if isinstance(payload.get("acceptedAssumptions"), list) else []
    current_extension_text = str(payload.get("currentExtensionText") or "").strip()

    system_prompt = ("You are an assistant that rewrites technical ABA extension output into plain "
                     "language for non-technical readers. Be concise and faithful to input.")
    lines = [
        "Translate only the provided current extension to natural language. Do not summarize the full graph.",
        f"Language: {target_language}",
        f"Current extension (text): {current_extension_text}" if current_extension_text else "",
        f"Extensions (JSON): {_json(extensions)}",
        f"Accepted assumptions (JSON): {_json(accepted)}",
        "Output requirements:",
        "- 2-4 short bullet points.",
        "- Focus only on the provided current extension text.",
        "- Do not summarize the whole graph.",
        "- Explain in everyday language.",
        "- Keep key domain words if needed, but avoid code-like naming unless necessary.",
        "- No markdown heading.",
    ]
    return system_prompt, "\n".join(x for x in lines if x)


def _json(value) -> str:
    # JS JSON.stringify has no spaces after separators; match it so prompts are identical.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def translate_with_ollama(base_url: str, model: str, system_prompt: str, user_prompt: str) -> str:
    endpoint = f"{str(base_url or 'http://127.0.0.1:11434').rstrip('/')}/api/chat"
    with httpx.Client(timeout=300.0) as client:
        resp = client.post(endpoint, json={
            "model": model,
            "stream": False,
            "options": {"temperature": 0.2},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        })
        try:
            data = resp.json()
        except Exception:
            data = {}
        if resp.status_code >= 400:
            raise RuntimeError(data.get("error") or f"Ollama request failed: {resp.status_code}")
        text = str((data.get("message") or {}).get("content") or "").strip()
        if not text:
            raise RuntimeError("Ollama response has no content")
        return text


def generate_llm_explanation(body: dict | None) -> dict:
    if settings.LLM_PROVIDER and settings.LLM_PROVIDER != "ollama":
        raise RuntimeError(f"Unsupported LLM_PROVIDER '{settings.LLM_PROVIDER}'. Only 'ollama' is supported.")

    system_prompt, user_prompt = build_translate_prompt(body or {})
    started_at = time.time()

    raw_model = str((body or {}).get("model") or settings.OLLAMA_TRANSLATE_MODEL or "qwen2.5:7b").strip()
    model = raw_model if raw_model in settings.ALLOWED_LLM_MODELS else "qwen2.5:7b"

    text = translate_with_ollama(settings.OLLAMA_BASE_URL, model, system_prompt, user_prompt)
    return {
        "text": text,
        "provider": "ollama",
        "model": model,
        "elapsed_ms": max(0, int((time.time() - started_at) * 1000)),
    }
