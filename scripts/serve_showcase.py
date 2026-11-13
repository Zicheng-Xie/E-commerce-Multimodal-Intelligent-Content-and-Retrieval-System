"""Start the local 845-product selected-policy text/image retrieval showcase."""

from __future__ import annotations

import argparse
import logging
import os
import webbrowser

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import run  # noqa: F401,E402
from multimodal_ecommerce.showcase_engine import ROOT, ShowcaseEngine  # noqa: E402
from multimodal_ecommerce.showcase_http import ShowcaseServer  # noqa: E402
from prepare_retrieval_showcase import prepare  # noqa: E402


def main() -> None:
    """Load the actual gallery and serve a local browser window without acceptance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Invalid port")
    if not (ROOT / "public_index/manifest.json").exists():
        prepare()
    print("正在校验 845 件开发商品与固定模型，请稍候…", flush=True)
    engine = ShowcaseEngine()
    server = ShowcaseServer(("127.0.0.1", args.port), engine)
    url = f"http://127.0.0.1:{args.port}"
    print(f"商品检索平台已启动：{url}", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Showcase stopped by keyboard interrupt")
    finally:
        server.server_close()
        server.executor.shutdown(wait=True)
        engine.models.close()


if __name__ == "__main__":
    main()
