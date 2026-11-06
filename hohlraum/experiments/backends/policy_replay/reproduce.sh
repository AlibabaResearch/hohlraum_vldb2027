#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
command -v g++ >/dev/null || { echo 'Install a C++17 compiler (g++).' >&2; exit 2; }
command -v make >/dev/null || { echo 'Install GNU Make.' >&2; exit 2; }
if [ ! -x "$ROOT/.venv/bin/python" ]; then python3 -m venv "$ROOT/.venv"; fi
if ! "$ROOT/.venv/bin/python" -c 'import matplotlib,numpy' >/dev/null 2>&1; then
    "$ROOT/.venv/bin/python" -m pip install -r "$ROOT/requirements.txt"
fi
exec "$ROOT/.venv/bin/python" "$ROOT/reproduce.py" "$@"
