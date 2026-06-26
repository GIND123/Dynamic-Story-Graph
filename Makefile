.PHONY: up down logs seed test lint demo ingest

# Copy .env.example to .env if .env does not exist
.env:
	cp .env.example .env

up: .env
	docker compose up -d --build

down:
	docker compose down -v

logs:
	docker compose logs -f

seed: .env
	@echo "Seeding demo story (requires: make up first)..."
	docker compose exec api python -m app.seed

test: .env
	docker compose run --rm api pytest tests/ -v

lint:
	docker compose run --rm api ruff check app/ tests/
	docker compose run --rm api ruff format --check app/ tests/

demo: .env
	@echo "=== Manuscript Memory Engine Demo ==="
	@echo "Requires: make up && make seed"
	@echo "See README.md for full demo script."

# Ingest an eval dataset (PDNC novel or LitBank doc) into the graph.
# Usage: make ingest ARGS="pdnc PrideAndPrejudice"
#        make ingest ARGS="litbank 1023_bleak_house"
ingest: .env
	@if [ -z "$(ARGS)" ]; then \
		echo 'Usage: make ingest ARGS="pdnc <NovelFolder>" | "litbank <doc_id>"'; \
		exit 1; \
	fi
	docker compose run --rm api python -m ingestion.cli $(ARGS)
