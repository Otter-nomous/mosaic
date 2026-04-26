#!/usr/bin/env bash
# One-shot installer: creates ./.venv on Python 3.10+ and installs mosaic in editable mode.
# Re-run any time; it's idempotent.

set -eo pipefail

cd "$(dirname "$0")"

VENV_DIR="${VENV_DIR:-.venv}"

find_python() {
  for py in python3.13 python3.12 python3.11 python3.10; do
    if command -v "$py" >/dev/null 2>&1; then
      echo "$py"
      return 0
    fi
  done
  if command -v python3 >/dev/null 2>&1 \
     && python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' \
     >/dev/null 2>&1; then
    echo python3
    return 0
  fi
  return 1
}

PY=$(find_python) || {
  echo "error: need Python 3.10+ on PATH" >&2
  echo "  on macOS: brew install python@3.12" >&2
  echo "  on Linux: use your distro's python3.12 package, or pyenv" >&2
  exit 1
}

if [ ! -d "$VENV_DIR" ]; then
  echo "[install] creating venv at $VENV_DIR (using $PY)"
  "$PY" -m venv "$VENV_DIR"
else
  echo "[install] reusing existing venv at $VENV_DIR"
fi

PIP="$VENV_DIR/bin/pip"
MOSAIC="$VENV_DIR/bin/mosaic"

echo "[install] upgrading pip"
"$PIP" install --quiet --upgrade pip

echo "[install] installing mosaic (editable)"
"$PIP" install --quiet -e .

echo "[install] verifying CLI"
if ! "$MOSAIC" --help >/dev/null; then
  echo "error: mosaic CLI failed to start after install" >&2
  exit 1
fi

cat <<EOF

installed.

Next steps:
  source $VENV_DIR/bin/activate
  export GOOGLE_API_KEY=...
  mosaic --data-dir /path/to/dataset --split val

Or call the CLI without activating: ./$VENV_DIR/bin/mosaic --help
EOF
