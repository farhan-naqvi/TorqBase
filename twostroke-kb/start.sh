#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=== TwoStroke-KB Startup ==="

# ── System dependencies (apt) ─────────────────────────────────────────────────
# Install any missing packages silently so parsers work out of the box.
_apt_install() {
  local missing=()
  for pkg in "$@"; do
    dpkg -s "$pkg" &>/dev/null || missing+=("$pkg")
  done
  if [ ${#missing[@]} -gt 0 ]; then
    echo "Installing system packages: ${missing[*]}"
    sudo apt-get update -qq && sudo apt-get install -y -qq "${missing[@]}"
  fi
}

if command -v apt-get &>/dev/null; then
  # Install individually so one failure doesn't block the rest
  sudo apt-get update -qq 2>/dev/null || true
  for pkg in tesseract-ocr tesseract-ocr-deu poppler-utils antiword libreoffice; do
    dpkg -s "$pkg" &>/dev/null && continue
    echo "Installing $pkg..."
    sudo apt-get install -y -qq "$pkg" 2>/dev/null || echo "  (skipped $pkg — not available in this repo)"
  done
fi

# Ensure .env exists
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env. Please configure it and rerun."
    exit 1
fi

# Activate venv if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "Starting FastAPI (uvicorn)..."
uvicorn api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  > uvicorn.log 2>&1 &

echo $! > .uvicorn.pid

sleep 3

echo "Starting Cloudflare tunnel..."

if [ -x "./cloudflared" ]; then
    ./cloudflared tunnel --url http://localhost:8000
else
    echo "cloudflared not found — running backend only"
    wait $(cat .uvicorn.pid)
fi