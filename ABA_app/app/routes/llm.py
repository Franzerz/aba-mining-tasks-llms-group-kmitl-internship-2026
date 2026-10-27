#### PATCHARAKORN ####

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..llm_service import build_summary_prompt, generate, list_models

router = APIRouter(prefix="/api/llm")


@router.get("/models")
def get_models() -> dict:
    """Which local Ollama models can be used, or available=false if Ollama isn't there.

    Never raises: a machine without Ollama is an ordinary, supported state for this app, not
    an error — the Graph Studio just shows its offline summary and says so.
    """
    return list_models()


class SummaryRequest(BaseModel):
    model: str = Field(min_length=1)
    # The on-screen tallies, already filtered to whatever the canvas is actually drawing
    # (view / focus / head). Sent by the client rather than recomputed here so the summary
    # can never describe a different picture than the one being looked at.
    facts: dict


@router.post("/summary")
def post_summary(request: SummaryRequest) -> dict:
    prompt = build_summary_prompt(request.facts or {})
    try:
        text = generate(request.model, prompt)
    except RuntimeError as e:
        # 503: the app is fine, the optional local model behind it is not.
        raise HTTPException(status_code=503, detail=str(e))
    return {"summary": text, "model": request.model}
