# Deep Learning NLP Inference API

API REST assíncrona para classificação semântica e auditoria automatizada de textos corporativos (detecção de risco de *greenwashing* vs. conformidade técnica), construída com **PyTorch**, **Hugging Face Transformers**, **FastAPI** e **Pydantic v2**.

---

## Arquitetura do Sistema

```
mermaid
graph LR
    A[Cliente HTTP / Sistema Externo] -->|POST /predict JSON| B(FastAPI & Pydantic v2)
    B -->|Validação de Contrato| C[NLPInferenceEngine]
    C -->|Tokenização AutoTokenizer| D[Encoder Transformer BERT]
    D -->|Representação CLS| E[Custom Classification Head PyTorch]
    E -->|Logits & Softmax| F[JSON Response: Label + Confidence]
```

---

## Stack Tecnológica

* **Deep Learning & NLP:** `PyTorch`, `Hugging Face Transformers` (`AutoModel`, `AutoTokenizer`, `BertModel`).
* **Arquitetura Neural (`TextClassificationTransformer`):** Extração do vetor contextual do token `[CLS]` acoplado a uma cabeça de classificação densa customizada (`Dropout` -> `Linear` -> `GELU` -> `LayerNorm` -> `Linear`).
* **API & Serving:** `FastAPI`, `Uvicorn`, gestão de ciclo de vida de modelo em memória via `@asynccontextmanager` (`lifespan`).
* **Contratos de Dados:** `Pydantic v2` com validação estrita de payload (`extra='forbid'`).
* **Qualidade, Testes & CI/CD:** `Pytest`, `Ruff`, `HTTPX` (`TestClient`), modo determinístico *offline* para execução isolada em pipelines **GitHub Actions** e empacotamento em **Docker**.

---

## Execução Local

1. Ativar ambiente virtual e rodar testes:
```
powershell
python -m ruff check src/ tests/
python -m pytest --cov=src tests/
```

2. Iniciar servidor FastAPI:
```
powershell
python -m src.serving.app
```
Documentação interativa (Swagger UI) disponível em `http://localhost:8000/docs`.