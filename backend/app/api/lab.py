"""AI lab routes for local and external experiments."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings
from app.models.registry import registry
from app.services.polyglot import polyglot_engine


router = APIRouter(prefix="/api/lab", tags=["lab"])


class TextAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)


class ExternalChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    system: Optional[str] = Field(default="You are a concise AI lab assistant.")
    model: Optional[str] = None


@router.get("/engines", summary="List AI runtimes")
def list_engines() -> Dict[str, List[Dict[str, str]]]:
    external_status = "ready" if settings.EXTERNAL_AI_API_KEY else "unconfigured"
    return {
        "engines": [
            {
                "id": "python_sklearn",
                "name": "Python Models",
                "runtime": "Python / scikit-learn",
                "status": "ready",
                "detail": f"{len(registry.list_models())} demo models loaded.",
            },
            polyglot_engine.go_status().to_dict(),
            {
                "id": "external_ai",
                "name": "External API",
                "runtime": "OpenAI-compatible HTTP",
                "status": external_status,
                "detail": "Set AI_LAB_EXTERNAL_AI_API_KEY to enable live API calls.",
            },
        ]
    }


@router.post("/text/analyze", summary="Analyze text with the Go engine")
def analyze_text(req: TextAnalysisRequest) -> Dict[str, Any]:
    try:
        return polyglot_engine.analyze_text(req.text)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.post("/external/chat", summary="Call an OpenAI-compatible chat API")
async def external_chat(req: ExternalChatRequest) -> Dict[str, Any]:
    if not settings.EXTERNAL_AI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="External AI is not configured. Set AI_LAB_EXTERNAL_AI_API_KEY in backend/.env.",
        )

    url = settings.EXTERNAL_AI_BASE_URL.rstrip("/") + "/chat/completions"
    payload = {
        "model": req.model or settings.EXTERNAL_AI_MODEL,
        "messages": [
            {"role": "system", "content": req.system},
            {"role": "user", "content": req.message},
        ],
        "temperature": 0.4,
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {settings.EXTERNAL_AI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"External provider error: {exc}")

    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    data = response.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return {
        "engine": "external_ai",
        "model": payload["model"],
        "content": content,
        "raw": data,
    }
