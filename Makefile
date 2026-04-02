.PHONY: test lint format typecheck verify

test:
	python -m pytest tests/ -v

lint:
	python -m ruff check .

format:
	python -m ruff format --check .

typecheck:
	python -m mypy .

verify: test lint format typecheck
