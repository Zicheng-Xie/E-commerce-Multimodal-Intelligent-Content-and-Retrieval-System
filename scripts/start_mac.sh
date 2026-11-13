#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv-mac/bin:$PWD/.tools/bin:$PWD/.tools:$PATH"
export MULTIMODAL_DEVICE=mps
exec .venv-mac/bin/python scripts/serve_retrieval.py "$@"
