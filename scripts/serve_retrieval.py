"""Start a local resident retrieval demo; no public bind or automatic model download."""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import run  # noqa: F401,E402
from multimodal_ecommerce.resident_retrieval import ResidentRetrieval  # noqa: E402
from multimodal_ecommerce.retrieval_http import RetrievalHTTPServer  # noqa: E402


def main() -> None:
    """Start a local resident retrieval demo; no public bind or automatic model
    download.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Invalid port")
    engine = ResidentRetrieval()
    server = RetrievalHTTPServer(("127.0.0.1", args.port), engine)
    print(
        json.dumps(dict(url=f"http://127.0.0.1:{args.port}", **engine.health())),
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        engine.close()


if __name__ == "__main__":
    main()
