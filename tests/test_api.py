from __future__ import annotations

import os
import torch
from fastapi.testclient import TestClient

from src.models.classifier import TextClassificationTransformer
from src.serving.app import app

os.environ["OFFLINE_MODE"] = "true"


def test_transformer_forward_shape() -> None:
    """Valida a dimensionalidade dos tensores no forward pass da rede neural."""
    model = TextClassificationTransformer(model_name="dummy", num_classes=2, pretrained=False)
    model.eval()
    input_ids = torch.randint(0, 999, (4, 16), dtype=torch.long)
    attention_mask = torch.ones((4, 16), dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
    assert logits.shape == (4, 2)


def test_health_endpoint() -> None:
    """Valida o status operacional da API."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


def test_predict_endpoint_valid_payload() -> None:
    """Valida o contrato de resposta e distribuicao de probabilidades no endpoint /predict."""
    with TestClient(app) as client:
        payload = {"text": "Emissoes de escopo 1 reduzidas em 18% conforme auditoria ISO 14064."}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == payload["text"]
        assert data["predicted_label"] in {"COMPLIANT", "GREENWASHING_RISK"}
        assert 0.0 <= data["confidence_score"] <= 1.0
        assert "COMPLIANT" in data["probabilities"]
        assert "GREENWASHING_RISK" in data["probabilities"]


def test_predict_endpoint_invalid_short_text() -> None:
    """Valida a rejeicao automatica do Pydantic para textos fora do contrato minimo."""
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "abc"})
        assert response.status_code == 422