"""Serve the existing gallery plus automatic DeepFashion2 training-image expansion."""

from __future__ import annotations

import argparse
import logging
import os
import webbrowser

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import run  # noqa: F401,E402
from multimodal_ecommerce.expanded_showcase import ExpandedShowcase  # noqa: E402
from multimodal_ecommerce.fashion_gallery import OUTPUT  # noqa: E402
from multimodal_ecommerce.selected_runtime_io import load  # noqa: E402
from multimodal_ecommerce.showcase_http import ShowcaseServer  # noqa: E402


def main() -> None:
    """Start real serial inference and resumable train-only indexing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Invalid port")
    print("正在加载电商图库及已完成的服饰向量…", flush=True)
    engine = ExpandedShowcase()
    server = ShowcaseServer(("127.0.0.1", args.port), engine)
    url = f"http://127.0.0.1:{args.port}"
    print("扩容检索平台：" + url, flush=True)
    control = OUTPUT / "control.json"
    if control.exists() and load(control).get("paused") is True:
        engine.gallery.state = "paused"
        engine.gallery.publish()
        print("扩容已按用户要求暂停，仅使用已完成的图库提供检索。", flush=True)
    else:
        engine.start_indexing()
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Expansion service stopped")
    finally:
        engine.stop_indexing()
        server.server_close()
        server.executor.shutdown(wait=True)
        engine.models.close()


if __name__ == "__main__":
    main()
