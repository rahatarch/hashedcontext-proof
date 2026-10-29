# Shared setup for the run scripts. Not meant to be run on its own.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ -n "${PY:-}" ]; then :;
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python;
else echo "No .venv found. Run 'make setup' first (or set PY=/path/to/python)."; exit 1; fi
step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
