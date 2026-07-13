.PHONY: setup db run test fmt clean

setup:
	@if command -v apt-get >/dev/null 2>&1; then \
	  echo "Installing system packages..."; \
	  sudo apt-get update -qq 2>/dev/null || true; \
	  for pkg in tesseract-ocr tesseract-ocr-deu poppler-utils antiword libreoffice; do \
	    dpkg -s "$$pkg" >/dev/null 2>&1 || sudo apt-get install -y -qq "$$pkg" 2>/dev/null || echo "  (skipped $$pkg)"; \
	  done; \
	fi
	python -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	@echo ""
	@echo "Done. Next: cp .env.example .env and fill it in."

db:
	docker compose up -d
	@echo "Postgres+pgvector running on localhost:5432 (schema auto-applied on first run)."

run:
	. .venv/bin/activate && uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

test:
	. .venv/bin/activate && pytest -q

clean:
	rm -rf __pycache__ .pytest_cache **/__pycache__
