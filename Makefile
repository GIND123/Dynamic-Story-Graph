.PHONY: up down logs seed test lint demo

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
