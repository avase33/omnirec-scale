.PHONY: install dev test lint demo bench serve infra docker clean

install:
	pip install -e .

dev:
	pip install -e ".[serve,dev]"

test:
	pytest --cov=omnirec --cov-report=term-missing

lint:
	ruff check omnirec scripts

demo:
	python -m omnirec demo

bench:
	python -m omnirec bench --requests 3000

serve:
	omnirec serve

infra:
	docker compose -f docker-compose.infra.yml up --build

docker:
	docker build -t omnirec-scale:latest .

clean:
	rm -rf .pytest_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
