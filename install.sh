#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
MODE="${1:-hisim}"
DRY=0
if [[ "${2:-}" == "--dry-run" ]]; then DRY=1; fi
case "$MODE" in
  minimal|hisim|sglang) ;;
  *) echo "Usage: $0 [minimal|hisim|sglang] [--dry-run]" >&2; exit 2 ;;
esac
run() {
  if [[ "$DRY" == 1 ]]; then printf '%q ' "$@"; printf '\n'; else "$@"; fi
}
# Use a user-created virtual environment. Never install a PyPI SGLang in place
# of the bundled source. Dependency installation/building may require network.
run "$PYTHON" -m pip install -e "$ROOT"
if [[ "$MODE" != minimal ]]; then
  run "$PYTHON" -m pip install -e "$ROOT/third_party/sglang/python"
fi
if [[ "$MODE" == hisim ]]; then
  run "$PYTHON" -m pip install -e "$ROOT/third_party/sglang/tools/sglang-simulator"
fi
