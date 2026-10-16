"""Local reverse proxy that forces request-body fields on every call to a model endpoint.

Benchmarks reach a model three ways: the suite's own chat client, opencode agents, and official
harnesses with their own HTTP clients. Only the first honours models.yaml `params`. When a model sets
`request_overrides` (e.g. `{reasoning_effort: medium}`), the worker points all three at this proxy
instead, so the setting can't silently differ between benchmarks. Overrides win over whatever the
harness sent; everything else (paths, headers, streaming, client disconnects) passes straight through.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from urllib.parse import urlsplit

import httpx

# Hop-by-hop headers, plus the length/host the proxy recomputes.
_HOP = {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "te", "trailer",
        "upgrade", "proxy-authorization", "proxy-authenticate"}
_BODY_PATHS = ("/chat/completions", "/completions", "/messages")

_proxies: dict[tuple[str, str], "OverrideProxy"] = {}
_lock = threading.Lock()


class OverrideProxy:
    def __init__(self, upstream_base: str, overrides: dict):
        u = urlsplit(upstream_base.rstrip("/"))
        self.origin = f"{u.scheme}://{u.netloc}"
        self.prefix = u.path  # e.g. "/v1"; the proxy URL keeps it so callers append paths as before
        self.overrides = dict(overrides)
        self.n_overridden = 0
        self.url = ""
        self._client: httpx.AsyncClient | None = None

    def apply(self, path: str, body: bytes) -> bytes:
        if not body or not path.endswith(_BODY_PATHS):
            return body
        try:
            data = json.loads(body)
        except ValueError:
            return body
        if not isinstance(data, dict):
            return body
        data.update(self.overrides)
        self.n_overridden += 1
        return json.dumps(data).encode()

    def app(self):
        from starlette.applications import Starlette
        from starlette.responses import StreamingResponse
        from starlette.routing import Route

        async def handle(request):
            if self._client is None:  # created inside uvicorn's event loop
                self._client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=30.0))
            body = self.apply(request.url.path, await request.body())
            headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP}
            url = self.origin + request.url.path + (f"?{request.url.query}" if request.url.query else "")
            upstream = await self._client.send(
                self._client.build_request(request.method, url, headers=headers, content=body), stream=True)

            async def relay():
                try:
                    async for chunk in upstream.aiter_raw():
                        yield chunk
                finally:  # also runs when the caller disconnects, so the model stops generating
                    await upstream.aclose()

            out_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in _HOP}
            return StreamingResponse(relay(), status_code=upstream.status_code, headers=out_headers)

        methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]
        return Starlette(routes=[Route("/{path:path}", handle, methods=methods)])

    def start(self) -> str:
        import uvicorn

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(self.app(), host="127.0.0.1", port=port, log_level="warning",
                                               access_log=False))
        threading.Thread(target=server.run, daemon=True, name=f"override-proxy-{port}").start()
        for _ in range(200):
            if server.started:
                break
            time.sleep(0.05)
        if not server.started:
            raise RuntimeError("request-override proxy did not start")
        self.url = f"http://127.0.0.1:{port}{self.prefix}"
        return self.url


def override_proxy(upstream_base: str, overrides: dict) -> OverrideProxy:
    """Start (once per upstream + overrides, for the life of the process) and return the proxy."""
    key = (upstream_base.rstrip("/"), json.dumps(overrides, sort_keys=True))
    with _lock:
        if key not in _proxies:
            proxy = OverrideProxy(upstream_base, overrides)
            proxy.start()
            _proxies[key] = proxy
        return _proxies[key]
