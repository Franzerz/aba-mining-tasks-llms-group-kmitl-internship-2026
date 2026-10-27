#### PATCHARAKORN ####

"""Port of mu_team's Redis-backed async job queue (the `/api/*/jobs` endpoints).

Same key layout (`<prefix>:<jobType>:<id>` plus a sorted-set index `<prefix>:index:<jobType>`),
same TTLs, same pruning by `*_JOB_MAX`, same status vocabulary (queued/running/done/failed) and
the same response mapping, so their frontend's polling logic works unchanged against this
backend.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Any, Callable

import redis

from .. import settings

JOB_TYPE_PYARG = "pyarg_evaluate"
JOB_TYPE_LLM = "llm_translate_extension"

LLM_JOB_MAX = 120
PYARG_JOB_MAX = 120

_client: redis.Redis | None = None


def _redis() -> redis.Redis:
    """Their code raises if REDIS_URL is unset and an async endpoint is called; same here.
    The synchronous endpoints keep working without Redis."""
    global _client
    if not settings.REDIS_URL:
        raise RuntimeError(
            "REDIS_URL is not set. Configure Redis before using async job endpoints."
        )
    if _client is None:
        _client = redis.from_url(settings.REDIS_URL, decode_responses=True)
    return _client


def _now_ms() -> int:
    return int(time.time() * 1000)


def _job_key(job_type: str, job_id: str) -> str:
    return f"{settings.ASYNC_JOB_REDIS_PREFIX}:{job_type}:{job_id}"


def _job_index_key(job_type: str) -> str:
    return f"{settings.ASYNC_JOB_REDIS_PREFIX}:index:{job_type}"


def _max_for(job_type: str) -> int:
    return PYARG_JOB_MAX if job_type == JOB_TYPE_PYARG else LLM_JOB_MAX


def _ttl_for(job_type: str) -> int:
    return settings.PYARG_JOB_TTL_MS if job_type == JOB_TYPE_PYARG else settings.LLM_JOB_TTL_MS


def _set_job(job_type: str, job: dict, ttl_ms: int) -> None:
    ttl_seconds = max(1, -(-int(ttl_ms or 1000) // 1000))
    _redis().set(_job_key(job_type, job["id"]), json.dumps(job), ex=ttl_seconds)


def _get_job(job_type: str, job_id: str) -> dict | None:
    raw = _redis().get(_job_key(job_type, job_id))
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _prune(job_type: str) -> None:
    max_count = _max_for(job_type)
    if max_count <= 0:
        return
    client = _redis()
    index_key = _job_index_key(job_type)
    overflow = int(client.zcard(index_key) or 0) - max_count
    if overflow <= 0:
        return
    old_ids = client.zrange(index_key, 0, overflow - 1)
    if not old_ids:
        return
    pipe = client.pipeline()
    pipe.zrem(index_key, *old_ids)
    for job_id in old_ids:
        pipe.delete(_job_key(job_type, job_id))
    pipe.execute()


def map_job_for_response(job: dict, fallback_error: str) -> dict:
    return {
        "job_id": job["id"],
        "status": job["status"],
        "created_at": int(job["created_at"]),
        "updated_at": int(job["updated_at"]),
        "started_at": None if job.get("started_at") is None else int(job["started_at"]),
        "completed_at": None if job.get("completed_at") is None else int(job["completed_at"]),
        "result": json.loads(job["result_json"]) if job["status"] == "done" and job.get("result_json") else None,
        "error": str(job.get("error_text") or fallback_error) if job["status"] == "failed" else None,
    }


def _mark(job_type: str, job_id: str, **changes: Any) -> None:
    job = _get_job(job_type, job_id)
    if not job:
        return
    job.update(changes)
    job["updated_at"] = _now_ms()
    _set_job(job_type, job, _ttl_for(job_type))


def create_job(job_type: str, body: dict, worker: Callable[[dict], dict],
               fallback_error: str) -> dict:
    """Creates the job, then runs `worker` on a background thread — their code kicks the work
    off with a floating promise and returns the queued job immediately, same shape here."""
    _prune(job_type)
    ttl_ms = _ttl_for(job_type)
    now = _now_ms()
    job = {
        "id": str(uuid.uuid4()),
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "completed_at": None,
        "payload_json": json.dumps(body or {}),
        "result_json": None,
        "error_text": None,
    }
    _set_job(job_type, job, ttl_ms)
    _redis().zadd(_job_index_key(job_type), {job["id"]: now})

    def run() -> None:
        job_id = job["id"]
        started = _now_ms()
        try:
            _mark(job_type, job_id, status="running", started_at=started)
            result = worker(body or {})
            if isinstance(result, dict) and result.get("error"):
                _mark(job_type, job_id, status="failed",
                      error_text=str(result.get("error") or fallback_error),
                      completed_at=_now_ms())
            else:
                _mark(job_type, job_id, status="done", result_json=json.dumps(result),
                      error_text=None, completed_at=_now_ms())
        except Exception as exc:  # noqa: BLE001 — mirrors their catch-all
            try:
                _mark(job_type, job_id, status="failed", error_text=str(exc) or fallback_error,
                      completed_at=_now_ms())
            except Exception:
                pass
        finally:
            try:
                _prune(job_type)
            except Exception:
                pass

    threading.Thread(target=run, daemon=True).start()
    return map_job_for_response(job, fallback_error)


def get_job(job_type: str, job_id: str, fallback_error: str) -> dict | None:
    job_id = str(job_id or "").strip()
    if not job_id:
        return None
    job = _get_job(job_type, job_id)
    return map_job_for_response(job, fallback_error) if job else None
