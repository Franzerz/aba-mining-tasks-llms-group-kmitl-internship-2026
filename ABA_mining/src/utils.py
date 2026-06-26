#### PATCHARAKORN ####

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

    # Fix unterminated strings: find unclosed quotes and close them
    # Track if we're inside a string and handle escaping
    fixed = _fix_unterminated_strings(cleaned)
    if fixed != cleaned:
        try:
            return True, json.loads(fixed), None
        except Exception:
            cleaned = fixed  # Use fixed version for next attempts

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
            elif "Unterminated string" in msg:
                # Try closing the unterminated string
                if not cleaned.endswith('"'):
                    cleaned += '"'
                # Then close any open braces/brackets
                open_braces = cleaned.count('{') - cleaned.count('}')
                open_brackets = cleaned.count('[') - cleaned.count(']')
                cleaned += '}' * open_braces + ']' * open_brackets
                try:
                    return True, json.loads(cleaned), None
                except Exception:
                    break
            else:
                break

    # Try extracting from first { to last } (handles preamble text before JSON)
    start = text.find('{')
    end   = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return True, json.loads(text[start:end + 1]), None
        except Exception:
            pass

    # Final attempt
    try:
        return True, json.loads(cleaned), None
    except Exception as e:
        return False, None, str(e)


def _fix_unterminated_strings(text: str) -> str:
    """Fix unterminated strings by intelligently closing quotes."""
    # Find indices of all quotes in the text
    result = []
    in_string = False
    prev_char = ''
    
    for i, char in enumerate(text):
        # Check if this quote is escaped (preceded by odd number of backslashes)
        if char == '"' and (i == 0 or text[i-1] != '\\' or (i >= 2 and text[i-2] == '\\')):
            in_string = not in_string
        
        result.append(char)
        prev_char = char
    
    # If we end with an unclosed string, close it
    if in_string:
        # Also close any open braces/brackets after closing the string
        result.append('"')
    
    return ''.join(result)


def normalize_parsed_json(parsed: Any, output_schema: str = "full", topics: list[str] | None = None) -> tuple[Any, list[str]]:
    """Normalize parsed JSON to match expected schema. Returns (normalized, errors)."""
    errors = []
    
    if not isinstance(parsed, dict):
        errors.append("Output is not a JSON object")
        return None, errors
    
    # Handle different output schemas
    if output_schema == "span_only":
        # Expected format: {"spans": {"Topic": [{"text": "..."}, ...]}}
        if "spans" not in parsed:
            errors.append("Missing 'spans' key")
            return None, errors
        
        spans = parsed["spans"]
        if not isinstance(spans, dict):
            errors.append("'spans' must be a dict")
            return None, errors
        
        # Normalize each topic's spans
        normalized_spans = {}
        for topic, span_list in spans.items():
            if not isinstance(span_list, list):
                continue
            
            normalized_spans[topic] = []
            for item in span_list:
                if not isinstance(item, dict):
                    continue
                
                # Extract text, ignore 'label' and other fields
                text = item.get("text")
                if text and isinstance(text, str) and text != "null":
                    normalized_spans[topic].append({"text": text})
        
        return {"spans": normalized_spans}, errors
    
    elif output_schema == "sentiment_only":
        # Expected format: {"annotations": [{"topic": "...", "sentiment": "..."}, ...]}
        if "annotations" not in parsed:
            errors.append("Missing 'annotations' key")
            return None, errors
        
        annotations = parsed["annotations"]
        if not isinstance(annotations, list):
            errors.append("'annotations' must be a list")
            return None, errors
        
        normalized_annotations = []
        for item in annotations:
            if not isinstance(item, dict):
                continue
            
            topic = item.get("topic")
            sentiment = item.get("sentiment")
            
            if topic and sentiment and isinstance(topic, str) and isinstance(sentiment, str):
                if sentiment in ["Positive", "Negative"]:
                    normalized_annotations.append({
                        "topic": topic,
                        "sentiment": sentiment
                    })
        
        return {"annotations": normalized_annotations}, errors
    
    else:  # full schema
        # Expected format: {"annotations": [{"topic": "...", "text": "...", "sentiment": "..."}, ...]}
        if "annotations" not in parsed:
            errors.append("Missing 'annotations' key")
            return None, errors
        
        annotations = parsed["annotations"]
        if not isinstance(annotations, list):
            errors.append("'annotations' must be a list")
            return None, errors
        
        normalized_annotations = []
        for item in annotations:
            if not isinstance(item, dict):
                continue
            
            topic = item.get("topic")
            text = item.get("text")
            sentiment = item.get("sentiment")
            
            # All three fields required
            if topic and text and sentiment:
                if isinstance(topic, str) and isinstance(text, str) and isinstance(sentiment, str):
                    if sentiment in ["Positive", "Negative"] and text != "null":
                        normalized_annotations.append({
                            "topic": topic,
                            "text": text,
                            "sentiment": sentiment
                        })
        
        return {"annotations": normalized_annotations}, errors


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

