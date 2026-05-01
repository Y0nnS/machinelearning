"""
API Routes for ML models
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Any, Dict, Optional

from app.models.registry import registry

router = APIRouter(prefix="/api/models", tags=["models"])


# ── Schemas ────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    features: List[float] = Field(..., description="Input feature vector")


class ProbabilityItem(BaseModel):
    label: str
    probability: float


class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    prediction: str
    prediction_index: int
    probabilities: Optional[List[ProbabilityItem]] = None
    model_id: str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/", summary="List all available models")
def list_models():
    return {"models": registry.list_models()}


@router.get("/{model_id}", summary="Get model metadata")
def get_model(model_id: str):
    meta = registry.get_model(model_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return meta


@router.post("/{model_id}/predict", response_model=PredictResponse, summary="Run inference")
def predict(model_id: str, req: PredictRequest):
    try:
        result = registry.predict(model_id, req.features)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    return PredictResponse(model_id=model_id, **result)
