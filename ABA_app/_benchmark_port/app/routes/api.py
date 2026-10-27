#### PATCHARAKORN ####

"""HTTP surface, matching mu_team's Express routes one for one so their frontend contract is
preserved (backend/routes/review.js and backend/routes/aba.js).

Endpoints:
    GET  /api/health
    GET  /api/topic-ratios
    GET  /api/review-data
    GET  /api/aba-graph
    POST /api/pyarg/evaluate                     GET /api/pyarg/evaluate/jobs/{job_id}
    POST /api/pyarg/evaluate/jobs
    POST /api/llm/translate-extension            GET /api/llm/translate-extension/jobs/{job_id}
    POST /api/llm/translate-extension/jobs
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .. import db
from ..services import jobs, llm_service, pyarg_eval, review_service
from ..services.aba_graph_service import HttpError, get_aba_graph

router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict:
    ok = db.health()
    return {"ok": ok, "db": ok}


@router.get("/topic-ratios")
def topic_ratios() -> dict:
    return review_service.get_topic_ratios()


@router.get("/review-data")
def review_data(topic: str = "staff", sentiment: str = "positive") -> dict:
    return review_service.get_review_data(topic, sentiment)


@router.get("/aba-graph")
def aba_graph(request: Request):
    try:
        return get_aba_graph(dict(request.query_params))
    except HttpError as err:
        return JSONResponse(status_code=err.status, content={"error": err.message})
    except Exception as err:  # noqa: BLE001 — their route returns 500 with the message too
        return JSONResponse(status_code=500, content={"error": str(err)})


# --- PyArg ------------------------------------------------------------------------------------

@router.post("/pyarg/evaluate")
async def pyarg_evaluate(request: Request):
    body = await _json_body(request)
    result = pyarg_eval.sanitize_result(pyarg_eval.evaluate(body))
    if isinstance(result, dict) and result.get("error"):
        return JSONResponse(status_code=400, content=result)
    return result


@router.post("/pyarg/evaluate/jobs", status_code=202)
async def pyarg_evaluate_job(request: Request):
    body = await _json_body(request)
    try:
        return jobs.create_job(
            jobs.JOB_TYPE_PYARG, body,
            lambda b: pyarg_eval.sanitize_result(pyarg_eval.evaluate(b)),
            "PyArg job failed")
    except Exception as err:  # noqa: BLE001
        return JSONResponse(status_code=500, content={
            "error": str(err),
            "hint": "Ensure Redis is reachable and REDIS_URL is set.",
        })


@router.get("/pyarg/evaluate/jobs/{job_id}")
def pyarg_evaluate_job_status(job_id: str):
    return _job_status(jobs.JOB_TYPE_PYARG, job_id, "PyArg job failed",
                       "PyArg job not found or expired")


# --- LLM --------------------------------------------------------------------------------------

@router.post("/llm/translate-extension")
async def llm_translate(request: Request):
    body = await _json_body(request)
    try:
        return llm_service.generate_llm_explanation(body)
    except Exception as err:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(err)})


@router.post("/llm/translate-extension/jobs", status_code=202)
async def llm_translate_job(request: Request):
    body = await _json_body(request)
    try:
        return jobs.create_job(jobs.JOB_TYPE_LLM, body,
                               llm_service.generate_llm_explanation, "LLM job failed")
    except Exception as err:  # noqa: BLE001
        return JSONResponse(status_code=500, content={
            "error": str(err),
            "hint": "Ensure Redis is reachable and REDIS_URL is set.",
        })


@router.get("/llm/translate-extension/jobs/{job_id}")
def llm_translate_job_status(job_id: str):
    return _job_status(jobs.JOB_TYPE_LLM, job_id, "LLM job failed",
                       "LLM job not found or expired")


# --- helpers ----------------------------------------------------------------------------------

async def _json_body(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception:
        return {}
    return body if isinstance(body, dict) else {}


def _job_status(job_type: str, job_id: str, fallback_error: str, not_found: str):
    try:
        job = jobs.get_job(job_type, job_id, fallback_error)
    except Exception as err:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(err)})
    if not job:
        return JSONResponse(status_code=404, content={"error": not_found},
                            headers={"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"})
    return JSONResponse(content=job,
                        headers={"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"})
