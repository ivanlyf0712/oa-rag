#!/usr/bin/env bash
# Run the Streamlit app from the repo venv (local dev helper).
# In docker the stack starts itself — this script is for bare-metal dev.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT_DIR/venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="$(command -v python3 || true)"
fi
if [ -z "$PYTHON" ]; then
  echo "No Python interpreter found. Create/activate a venv first." >&2
  exit 1
fi
exec "$PYTHON" -m streamlit run "$ROOT_DIR/apps/app.py" "$@"
