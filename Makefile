.PHONY: test lint format typecheck verify

test:
	python -m pytest tests/ -v

lint:
	ruff check .

format:
	ruff format --check .

typecheck:
	mypy .

verify: test lint format typecheck
