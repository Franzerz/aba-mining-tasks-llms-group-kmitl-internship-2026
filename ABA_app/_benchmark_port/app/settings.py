#### PATCHARAKORN ####

"""Runtime configuration, mirroring mu_team's .env contract so the two apps can be pointed at
the same database and the same Ollama host for a like-for-like benchmark."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

APP_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(APP_ROOT / ".env", override=False)


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


# --- Database (same MySQL instance mu_team's app uses) -------------------------------------
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = _int("DB_PORT", 3306)
DB_USER = os.getenv("DB_USER", "aba")
DB_PASSWORD = os.getenv("DB_PASSWORD", "aba12345")
DB_NAME = os.getenv("DB_NAME", "ABA")

# --- LLM (Ollama) --------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_TRANSLATE_MODEL = os.getenv("OLLAMA_TRANSLATE_MODEL", "gemma3:4b")
# Same allowlist mu_team enforced in their backend.
ALLOWED_LLM_MODELS = {"gemma3:4b", "deepseek-r1:7b", "qwen2.5:7b"}

# --- Async job storage ---------------------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL") or os.getenv("REDIS_TLS_URL") or ""
ASYNC_JOB_REDIS_PREFIX = os.getenv("ASYNC_JOB_REDIS_PREFIX", "aba:job")
LLM_JOB_TTL_MS = _int("LLM_JOB_TTL_MS", 300_000)
PYARG_JOB_TTL_MS = _int("PYARG_JOB_TTL_MS", 1_200_000)

# --- PyArg behaviour -----------------------------------------------------------------------
PYARG_MAX_EXTENSIONS = _int("PYARG_MAX_EXTENSIONS", 200)
PYARG_INCLUDE_DEBUG_FIELDS = os.getenv("PYARG_INCLUDE_DEBUG_FIELDS", "0").strip() == "1"
# Their Node app shelled out to Python per request; we call py_arg in-process, so a hard
# wall-clock guard replaces their subprocess kill. Exponential semantics can run forever.
PYARG_TIMEOUT_SECONDS = _int("PYARG_TIMEOUT_SECONDS", 30)
