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

# ---------------------------------------------------------------- DSG study
.PHONY: dsg-test dsg-doctor dsg-extract dsg-study dsg-report dsg-all

dsg-test:
	.venv/bin/python -m pytest tests/dsg -q

dsg-doctor:
	.venv/bin/python -m dsg doctor

# GPU extraction: the only paid stage, run once per (corpus, model).
dsg-extract:
	.venv/bin/modal run dsg/infra/modal_extract.py --model qwen7b --corpus pdnc
	.venv/bin/modal run dsg/infra/modal_extract.py --model qwen7b --corpus litbank

dsg-study:
	.venv/bin/python -m dsg study --corpus pdnc \
	  --proposals artifacts/proposals/pdnc-qwen7b-w3200 \
	  --out artifacts/results/pdnc-qwen7b
	.venv/bin/python -m dsg study --corpus litbank \
	  --proposals artifacts/proposals/litbank-qwen7b-w1200 \
	  --out artifacts/results/litbank-qwen7b

dsg-report:
	.venv/bin/python -m dsg report --results artifacts/results/pdnc-qwen7b \
	  --out artifacts/report/pdnc-qwen7b
	.venv/bin/python -m dsg report --results artifacts/results/litbank-qwen7b \
	  --out artifacts/report/litbank-qwen7b

dsg-all: dsg-test dsg-extract dsg-study dsg-report
