# OA Contract RAG — Makefile
# ==========================
# Operational helpers mirroring corpchat-rag: validate .env, then run the
# compose stack (mysql + app).

.PHONY: help check-env up down logs ps build test index db-import restart

help:
	@echo "OA Contract RAG targets:"
	@echo "  make check-env   Validate .env has required secrets (DB_*, LITELLM_API_KEY)"
	@echo "  make up          check-env + docker compose up -d --build"
	@echo "  make db-import   One-time: dump host MySQL DB into the dockerized mysql"
	@echo "  make index       Rebuild the search index inside the app container"
	@echo "  make down        docker compose down (keeps data volumes)"
	@echo "  make logs        Tail all service logs"
	@echo "  make ps          Show service status"
	@echo "  make build       docker compose build"
	@echo "  make test        Run the pytest suite (local venv)"
	@echo "  make restart     Restart the app container"

check-env:
	@test -f .env || (echo "ERROR: .env not found — copy .env.example and fill it in" && exit 1)
	@grep -q '^LITELLM_API_KEY=.' .env || (echo "ERROR: LITELLM_API_KEY missing in .env" && exit 1)
	@grep -q '^DB_USER=.' .env || (echo "ERROR: DB_USER missing in .env" && exit 1)
	@grep -q '^DB_PASSWORD=.' .env || (echo "ERROR: DB_PASSWORD missing in .env" && exit 1)
	@grep -q '^DB_ROOT_PASSWORD=.' .env || (echo "ERROR: DB_ROOT_PASSWORD missing in .env (mysql container root — see .env.example)" && exit 1)
	@echo "OK: .env looks complete"

up: check-env
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

ps:
	docker compose ps

build:
	docker compose build

restart:
	docker compose restart app

test:
	@venv/bin/python -m pytest tests/ --ignore=tests/legacy_corpchat -q

index:
	@docker ps --format '{{.Names}}' | grep -qx oa-rag || (echo "ERROR: oa-rag container not running — make up first" && exit 1)
	docker exec oa-rag python scripts/build_index.py --force

# Dump the host MySQL database (credentials/host from .env) into the dockerized
# mysql service. Run once after the first make up, or to refresh the data.
db-import:
	@docker ps --format '{{.Names}}' | grep -qx oa-mysql || (echo "ERROR: oa-mysql not running — make up first" && exit 1)
	@set -a; . ./.env; set +a; 	echo "Dumping $$DB_NAME from $$DB_HOST:$$DB_PORT (host) -> oa-mysql ..."; 	mysqldump -h "127.0.0.1" -P "$${DB_PORT:-3306}" -u "$$DB_USER" "-p$$DB_PASSWORD" 		--single-transaction --databases "$${DB_NAME:-oa_rag}" 	| docker exec -i oa-mysql mysql -uroot "-p$$DB_ROOT_PASSWORD"; 	docker exec oa-mysql mysql -uroot "-p$$DB_ROOT_PASSWORD" -e 		"SELECT COUNT(*) AS form_385_rows FROM $${DB_NAME:-oa_rag}.formtable_main_385; SELECT COUNT(*) AS attachments FROM $${DB_NAME:-oa_rag}.contract_attachments;"
