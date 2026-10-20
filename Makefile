.PHONY: lint test serve docker-build docker-run

lint:
    python -m ruff check src/ tests/

test:
    python -m pytest --cov=src tests/

serve:
    python -m src.serving.app

docker-build:
    docker build -t dl-nlp-inference-api:latest .

docker-run:
    docker run -p 8000:8000 -e OFFLINE_MODE=false dl-nlp-inference-api:latest