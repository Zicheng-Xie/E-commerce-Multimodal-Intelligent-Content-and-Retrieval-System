"""Loopback-only browser transport for the complete development retrieval showcase."""

from __future__ import annotations

import base64
import json
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast
from urllib.parse import parse_qs, urlsplit

from multimodal_ecommerce.safety import PROJECT_ROOT
from multimodal_ecommerce.showcase_engine import ShowcaseEngine


class ShowcaseServer(ThreadingHTTPServer):
    """Keep model work serial while serving progress and images concurrently."""

    def __init__(self, address: tuple[str, int], engine: ShowcaseEngine) -> None:
        """Bind an explicit loopback address and one model worker.

        Args:
            address: Local listener address.
            engine: Verified public gallery and actual inference adapter.
        """
        if address[0] != "127.0.0.1":
            raise ValueError("Only local loopback binding is supported")
        super().__init__(address, ShowcaseHandler)
        self.engine = engine
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.jobs: dict = {}
        self.guard = threading.Lock()

    def submit(self, query: dict) -> str:
        """Queue a public query and retain bounded progress/results for polling.

        Args:
            query: Text or registered image token.

        Returns:
            Opaque job identifier.
        """
        with self.guard:
            if (
                sum(j["state"] in {"queued", "running"} for j in self.jobs.values())
                >= 8
            ):
                raise ValueError("请求较多，请等待当前检索完成")
            if len(self.jobs) >= 100:
                self.jobs = {
                    k: v
                    for k, v in self.jobs.items()
                    if v["state"] in {"queued", "running"}
                }
            token = uuid.uuid4().hex
            self.jobs[token] = dict(state="queued", message="检索已排队…")

        def progress(message: str) -> None:
            """Publish the current model stage.

            Args:
                message: Actual running-stage description.
            """
            with self.guard:
                self.jobs[token].update(state="running", message=message)

        def work() -> None:
            """Complete one queued query or publish its logged failure."""
            try:
                result = self.engine.search(query, progress)
                with self.guard:
                    self.jobs[token] = dict(state="done", result=result)
            except (
                Exception
            ) as error:  # The job must expose failure instead of hanging.
                logging.getLogger(__name__).exception("Showcase query failed")
                with self.guard:
                    self.jobs[token] = dict(state="error", error=str(error))

        self.executor.submit(work)
        return token


class ShowcaseHandler(BaseHTTPRequestHandler):
    """Expose only known media tokens, public product facts and bounded queries."""

    def _send(self, status: int, data: bytes, kind: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status: int, value: dict) -> None:
        self._send(
            status,
            json.dumps(value, ensure_ascii=False, allow_nan=False).encode(),
            "application/json; charset=utf-8",
        )

    def _origin(self) -> bool:
        server = cast(ShowcaseServer, self.server)
        hosts = {f"localhost:{server.server_port}", f"127.0.0.1:{server.server_port}"}
        if self.headers.get("Host") not in hosts or self.headers.get("Origin") not in {
            None,
            *("http://" + h for h in hosts),
        }:
            self._json(403, dict(error="仅允许本机同源访问"))
            return False
        return True

    def do_GET(self) -> None:
        """Serve the interface, actual product media, catalog and job status."""
        if not self._origin():
            return
        server = cast(ShowcaseServer, self.server)
        engine = server.engine
        url = urlsplit(self.path)
        if url.path in {"/", "/showcase.css", "/showcase.js"}:
            filename = "showcase.html" if url.path == "/" else url.path[1:]
            kind = (
                "text/html"
                if filename.endswith("html")
                else "text/css" if filename.endswith("css") else "text/javascript"
            )
            self._send(
                200,
                (PROJECT_ROOT / "assets" / filename).read_bytes(),
                kind + "; charset=utf-8",
            )
        elif url.path == "/health":
            self._json(
                200,
                dict(
                    state="ready",
                    products=len(engine.ids),
                    strategy="primary_public_conflict_matched_v1",
                    scope="完整统一开发图库",
                    formal_metrics=None,
                ),
            )
        elif url.path == "/api/samples":
            self._json(
                200,
                dict(
                    images=[
                        dict(**s, image_url="/media/" + s["token"])
                        for s in engine.samples
                    ],
                    products=len(engine.ids),
                ),
            )
        elif url.path == "/api/catalog":
            params = parse_qs(url.query)
            try:
                page = max(1, int(params.get("page", ["1"])[0]))
                size = min(40, max(1, int(params.get("size", ["20"])[0])))
                text = params.get("q", [""])[0].strip().casefold()
                ids = [
                    c for c in engine.ids if text in engine.docs[c]["title"].casefold()
                ]
                self._json(
                    200,
                    dict(
                        total=len(ids),
                        page=page,
                        size=size,
                        products=[
                            engine.card(c) for c in ids[(page - 1) * size : page * size]
                        ],
                    ),
                )
            except ValueError:
                logging.getLogger(__name__).debug("Invalid pagination", exc_info=True)
                self._json(400, dict(error="无效分页参数"))
        elif url.path.startswith("/api/jobs/"):
            with server.guard:
                job = server.jobs.get(url.path.removeprefix("/api/jobs/"))
            self._json(200 if job else 404, job or dict(error="检索任务不存在"))
        elif url.path.startswith("/media/"):
            asset = engine.assets.get(url.path.removeprefix("/media/"))
            if asset is None or not asset.is_file():
                self._json(404, dict(error="商品图片不存在"))
            else:
                kind = {
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".png": "image/png",
                    ".webp": "image/webp",
                }.get(asset.suffix.lower(), "application/octet-stream")
                self._send(200, asset.read_bytes(), kind)
        else:
            self._json(404, dict(error="页面不存在"))

    def do_POST(self) -> None:
        """Accept bounded raster uploads or enqueue a single live query."""
        if not self._origin():
            return
        server = cast(ShowcaseServer, self.server)
        try:
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                raise ValueError("仅支持 JSON 请求")
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 12 * 1024 * 1024:
                raise ValueError("请求过大或为空")
            value = json.loads(self.rfile.read(size))
            if not isinstance(value, dict):
                raise ValueError("请求格式不正确")
            path = urlsplit(self.path).path
            if path == "/api/upload":
                if set(value) != {"base64"} or not isinstance(value["base64"], str):
                    raise ValueError("请上传一张商品图片")
                result = server.engine.upload(
                    base64.b64decode(value["base64"], validate=True)
                )
                self._json(200, result)
            elif path == "/api/search":
                if set(value) not in ({"text"}, {"image_token"}):
                    raise ValueError("请提供文字或图片，不支持内部文件路径")
                self._json(202, dict(job_id=server.submit(value)))
            else:
                self._json(404, dict(error="接口不存在"))
        except (ValueError, OSError) as error:
            logging.getLogger(__name__).debug("Invalid showcase request", exc_info=True)
            self._json(400, dict(error=str(error)))

    def log_message(self, format_: str, *args: object) -> None:
        """Keep local access messages out of the model progress log.

        Args:
            format_: Standard HTTP log format.
            *args: Standard HTTP log values.
        """
