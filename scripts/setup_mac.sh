#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="$PWD/.cache/uv-mac"
export UV_PYTHON_INSTALL_DIR="$PWD/.runtime/mac-python"
if [ ! -x .tools/uv ]; then
  echo 'Install uv into .tools before running this script.' >&2
  exit 1
fi
if [ ! -x .venv-mac/bin/python ]; then
  .tools/uv venv --python 3.12 .venv-mac
fi
.tools/uv pip sync --python .venv-mac/bin/python requirements-macos.lock
.tools/uv pip install --python .venv-mac/bin/python --no-deps -e .
echo 'Ready. Use: source .venv-mac/bin/activate'
