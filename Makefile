.PHONY: test lint format typecheck lockcheck verify

test:
	python -m pytest tests/ -v

lint:
	python -m ruff check .

format:
	python -m ruff format --check .

typecheck:
	python -m mypy .

lockcheck:
	uv lock --check

verify: lockcheck test lint format typecheck
