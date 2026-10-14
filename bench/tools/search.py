"""Keyless web search + fetch for search-agent benchmarks.

- search(): DuckDuckGo via `ddgs` (no API key), rate-limited.
- fetch():  HTTP GET + trafilatura main-text extraction to markdown.
Both are cached forever in results/cache/search.sqlite, so every model sees the same results for
the same query and reruns are cheap. Exposed three ways: Python functions, OpenAI-style tool
schemas for chat loops, and an MCP server (streamable HTTP on localhost) for opencode agents.
"""

from __future__ import annotations

import hashlib
import json
import socket
import sqlite3
import threading
import time

import httpx

from bench.config import load_config

import logging

for _noisy in ("ddgs", "primp"):  # backend fail-over chatter (e.g. one engine's DNS error) is not an error
    logging.getLogger(_noisy).setLevel(logging.WARNING)

_lock = threading.Lock()
_last_call = [0.0]
_db: sqlite3.Connection | None = None

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/126.0 Safari/537.36")


def _cache() -> sqlite3.Connection:
    global _db
    if _db is None:
        path = load_config().results_dir / "cache"
        path.mkdir(parents=True, exist_ok=True)
        _db = sqlite3.connect(path / "search.sqlite", check_same_thread=False, isolation_level=None)
        _db.execute("PRAGMA journal_mode=WAL")
        _db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT, t REAL)")
    return _db


def _cached(kind: str, arg: str, fn):
    k = hashlib.sha256(f"{kind}:{arg}".encode()).hexdigest()
    db = _cache()
    with _lock:
        row = db.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
    if row:
        return json.loads(row[0])
    val = fn()
    if val is not None and not (isinstance(val, dict) and val.get("error")):
        with _lock:
            db.execute("INSERT OR REPLACE INTO c VALUES (?,?,?)", (k, json.dumps(val), time.time()))
    return val


def _throttle():
    gap = load_config().search.min_interval_s
    with _lock:
        wait = _last_call[0] + gap - time.time()
        _last_call[0] = max(time.time(), _last_call[0] + gap)
    if wait > 0:
        time.sleep(wait)


def search(query: str, max_results: int | None = None) -> list[dict]:
    cfg = load_config().search
    n = max_results or cfg.max_results

    def go():
        from ddgs import DDGS

        for attempt in range(3):
            _throttle()
            try:
                rows = DDGS().text(query, max_results=n, region=cfg.region)
                return [{"title": r.get("title"), "url": r.get("href"), "snippet": r.get("body")} for r in rows]
            except Exception as e:  # rate limits / transient failures
                if attempt == 2:
                    return {"error": f"search failed: {e}"}
                time.sleep(4 * (attempt + 1))
        return []

    return _cached("search", f"{n}:{query}", go)


def fetch(url: str, max_chars: int | None = None) -> dict:
    limit = max_chars or load_config().search.fetch_max_chars

    def go():
        try:
            r = httpx.get(url, headers={"User-Agent": UA}, timeout=30, follow_redirects=True)
        except httpx.HTTPError as e:
            return {"error": f"fetch failed: {e}"}
        ctype = r.headers.get("content-type", "")
        if r.status_code >= 400:
            return {"error": f"HTTP {r.status_code}"}
        if "html" in ctype or r.text.lstrip().startswith("<"):
            import trafilatura

            text = trafilatura.extract(r.text, output_format="markdown", include_links=True,
                                       include_tables=True) or ""
        elif "pdf" in ctype:
            return {"error": "PDF content is not supported by fetch"}
        else:
            text = r.text
        return {"url": str(r.url), "content": text}

    res = _cached("fetch", url, go)
    if isinstance(res, dict) and res.get("content") and len(res["content"]) > limit:
        res = {**res, "content": res["content"][:limit] + f"\n\n[truncated: {len(res['content'])} chars total]"}
    return res


# ---- OpenAI-style tool schemas for chat-runner loops -------------------------------------------
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "web_search", "description": "Search the web. Returns titles, URLs and snippets.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "fetch_url", "description": "Fetch a web page and return its main text as markdown.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
]


def tool_handler(name: str, args: dict) -> str:
    if name == "web_search":
        return json.dumps(search(args.get("query", "")), ensure_ascii=False)
    if name == "fetch_url":
        return json.dumps(fetch(args.get("url", "")), ensure_ascii=False)
    return f"unknown tool {name}"


# ---- MCP server for opencode agents ------------------------------------------------------------
_mcp_url: str | None = None


def start_mcp_server() -> str:
    """Start (once per process) a streamable-HTTP MCP server on localhost; returns its URL."""
    global _mcp_url
    if _mcp_url:
        return _mcp_url
    import uvicorn
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("bench")

    @server.tool()
    def web_search(query: str) -> str:
        """Search the web (DuckDuckGo). Returns a JSON list of {title, url, snippet}."""
        return json.dumps(search(query), ensure_ascii=False)

    @server.tool()
    def fetch_url(url: str) -> str:
        """Fetch a web page and return its main text as markdown (JSON {url, content})."""
        return json.dumps(fetch(url), ensure_ascii=False)

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    app = server.streamable_http_app(stateless_http=True, json_response=True)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    uv = uvicorn.Server(config)
    threading.Thread(target=uv.run, daemon=True, name="mcp-search").start()
    for _ in range(100):
        if uv.started:
            break
        time.sleep(0.05)
    _mcp_url = f"http://127.0.0.1:{port}/mcp"
    return _mcp_url
