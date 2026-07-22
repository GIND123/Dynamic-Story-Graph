.PHONY: install test lint format demo tree

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

lint:
	python -m ruff check gnsm tests

format:
	python -m ruff format gnsm tests

demo:
	python -m gnsm demo

tree:
	python -m gnsm tree
