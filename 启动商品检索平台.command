#!/bin/zsh
cd "$(dirname "$0")" || exit 1
if .venv-mac/bin/python -c 'import json,sys,urllib.request; data=json.load(urllib.request.urlopen("http://127.0.0.1:8766/health", timeout=2)); sys.exit(0 if data.get("state")=="ready" and data.get("expansion") is not None else 1)' 2>/dev/null; then
  exec /usr/bin/open "http://127.0.0.1:8766"
fi
exec .venv-mac/bin/python scripts/serve_expanded_showcase.py --open
