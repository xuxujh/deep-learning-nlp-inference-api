from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.models.classifier import NLPInferenceEngine, load_config

engine: NLPInferenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global engine
    cfg = load_config()
    offline = os.getenv("OFFLINE_MODE", "false").lower() == "true"
    engine = NLPInferenceEngine(
        model_name=cfg["model"]["name_or_path"],
        device=cfg["training"]["device"],
        offline_mode=offline,
    )
    yield
    engine = None


app = FastAPI(
    title="Deep Learning NLP Inference API",
    description="API REST para classificação semântica e auditoria de texto utilizando PyTorch e Transformers",
    version="1.0.0",
    lifespan=lifespan,
)


class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(..., min_length=5, max_length=2000, description="Texto corporativo para inferência neural")


class InferenceResponse(BaseModel):
    text: str
    predicted_label: str
    confidence_score: float
    probabilities: dict[str, float]


@app.get("/health", status_code=200)
async def health_check() -> dict[str, str]:
    return {"status": "healthy", "service": "dl-nlp-inference-api"}


@app.post("/predict", response_model=InferenceResponse, status_code=200)
async def predict_endpoint(payload: InferenceRequest) -> InferenceResponse:
    global engine
    if engine is None:
        engine = NLPInferenceEngine(offline_mode=True)
    try:
        result = engine.predict(payload.text)
        return InferenceResponse(
            text=payload.text,
            predicted_label=result["predicted_label"],
            confidence_score=result["confidence_score"],
            probabilities=result["probabilities"],
        )
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Inference failure: {err}") from err


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.serving.app:app", host="0.0.0.0", port=8000, reload=False)