"""Local HTTP transport for the resident development retrieval engine."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multimodal_ecommerce.resident_retrieval import ResidentRetrieval

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast
from urllib.parse import urlsplit

from multimodal_ecommerce.safety import PROJECT_ROOT


class RetrievalHTTPServer(ThreadingHTTPServer):
    """Serve the resident retrieval engine on the configured local address."""

    def __init__(self, address: tuple[str, int], engine: ResidentRetrieval) -> None:
        """Initialize the component and validate its configured dependencies.

        Args:
            address: Address used by this operation.
            engine: Engine used by this operation.
        """
        super().__init__(address, RetrievalHandler)
        self.engine = engine
        self.page = (PROJECT_ROOT / "assets/resident_retrieval.html").read_bytes()


class RetrievalHandler(BaseHTTPRequestHandler):
    """Validate local HTTP requests and serialize retrieval results and media."""

    server_version = "LocalRetrieval/1.0"

    def _response(
        self,
        status: int,
        content: bytes,
        content_type: str = "application/json; charset=utf-8",
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def _json(self, status: int, value: dict) -> None:
        self._response(
            status,
            json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"),
        )

    def do_GET(self) -> None:
        """Apply the do GET workflow step."""
        server = cast(RetrievalHTTPServer, self.server)
        path = urlsplit(self.path).path
        if path == "/":
            self._response(200, server.page, "text/html; charset=utf-8")
        elif path == "/health":
            self._json(200, server.engine.health())
        elif path == "/api/samples":
            self._json(200, server.engine.samples())
        elif path.startswith("/media/"):
            asset = server.engine.assets.get(path.removeprefix("/media/"))
            if asset is None or not asset.is_file():
                self._json(404, dict(error="Unknown asset"))
            else:
                self._response(
                    200,
                    asset.read_bytes(),
                    {
                        ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg",
                        ".png": "image/png",
                        ".webp": "image/webp",
                    }.get(asset.suffix.casefold(), "application/octet-stream"),
                )
        else:
            self._json(404, dict(error="Unknown endpoint"))

    def do_POST(self) -> None:
        """Return do POST for this workflow.

        Raises:
            ValueError: Request body size invalid; Explicit shutdown payload required.
        """
        server = cast(RetrievalHTTPServer, self.server)
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        expected = {
            f"127.0.0.1:{server.server_port}",
            f"localhost:{server.server_port}",
        }
        if host not in expected or (
            origin is not None and origin not in {"http://" + h for h in expected}
        ):
            self._json(403, dict(error="Only same-origin loopback requests supported"))
            return
        if (
            self.headers.get("Content-Type", "").split(";")[0].strip()
            != "application/json"
        ):
            self._json(415, dict(error="Use application/json"))
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 65536:
                raise ValueError("Request body size invalid")
            value = json.loads(self.rfile.read(size).decode("utf-8"))
            if urlsplit(self.path).path == "/api/shutdown":
                if value != {"shutdown": True}:
                    raise ValueError("Explicit shutdown payload required")
                self._json(200, dict(state="stopping"))
                threading.Thread(target=server.shutdown, daemon=True).start()
                return
            if urlsplit(self.path).path != "/api/search":
                self._json(404, dict(error="Unknown endpoint"))
                return
            self._json(200, server.engine.search(value))
        except (ValueError, OSError) as error:
            logging.getLogger(__name__).debug(
                "Handled exception in do_POST", exc_info=True
            )
            self._json(400, dict(error=str(error)))
        except RuntimeError as error:
            logging.getLogger(__name__).debug(
                "Handled exception in do_POST", exc_info=True
            )
            self._json(503, dict(error=str(error)))

    def log_message(self, format_: str, *args: object) -> None:
        """Keep access messages from mixing with the CLI's structured output.

        Args:
            format_: Access-log template supplied by BaseHTTPRequestHandler.
            *args: Additional positional values forwarded by this interface.
        """
        return
