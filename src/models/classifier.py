from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


def load_config(config_path: str | Path = "config/config.yaml") -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {
            "model": {
                "name_or_path": "prajjwal1/bert-tiny",
                "max_length": 128,
                "num_classes": 2,
                "dropout_rate": 0.1,
            },
            "training": {"device": "cpu"},
        }
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TextClassificationTransformer(nn.Module):
    """Rede neural customizada sobre encoder Transformer para classificação semântica."""

    def __init__(
        self,
        model_name: str,
        num_classes: int = 2,
        dropout_rate: float = 0.1,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if pretrained:
            self.config = AutoConfig.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        else:
            from transformers import BertConfig, BertModel
            self.config = BertConfig(
                vocab_size=1000,
                hidden_size=64,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=128,
            )
            self.encoder = BertModel(self.config)

        hidden_size = self.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_size // 2),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)
        return logits


class NLPInferenceEngine:
    """Motor de inferência desacoplado para execução do forward pass e cálculo de probabilidades."""

    LABELS = {0: "COMPLIANT", 1: "GREENWASHING_RISK"}

    def __init__(
        self, model_name: str = "prajjwal1/bert-tiny", device: str = "cpu", offline_mode: bool = False
    ) -> None:
        self.device = torch.device(device)
        self.offline_mode = offline_mode
        if not offline_mode:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = TextClassificationTransformer(model_name=model_name, pretrained=True)
        else:
            self.tokenizer = None
            self.model = TextClassificationTransformer(model_name=model_name, pretrained=False)

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, text: str, max_length: int = 128) -> dict[str, Any]:
        if self.offline_mode or self.tokenizer is None:
            # Tokenização determinística baseada em bytes para testes isolados de CI/CD sem rede
            tokens = [min(ord(c), 999) for c in text[:max_length]]
            if not tokens:
                tokens = [0]
            input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            encoded = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())

        return {
            "predicted_label": self.LABELS.get(pred_idx, "UNKNOWN"),
            "confidence_score": round(confidence, 4),
            "probabilities": {
                self.LABELS[0]: round(float(probs[0].item()), 4),
                self.LABELS[1]: round(float(probs[1].item()), 4),
            },
        }