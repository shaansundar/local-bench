"""Model server lifecycle: health-check external servers, launch/stop managed ones.

Only one managed local model is resident at a time: the worker groups jobs by model and calls
`ensure_up` before a model's jobs and `stop` after them.
"""

from __future__ import annotations

import atexit
import os
import shlex
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

import httpx

from bench.config import ModelSpec, load_config, resolve_secret


class ServerError(RuntimeError):
    pass


def free_memory_gb() -> float:
    """Reclaimable memory on macOS: free + inactive + speculative + purgeable pages."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return float("inf")
    page = 16384
    first = out.splitlines()[0] if out else ""
    if "page size of" in first:
        page = int(first.split("page size of")[1].split()[0])
    pages = 0
    for line in out.splitlines():
        key, _, val = line.partition(":")
        if key.strip() in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
            pages += int(val.strip().rstrip("."))
    return pages * page / 1e9


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def health(base_url: str, api_key: str | None = None, timeout: float = 30) -> bool:
    """Cheap /health first (mlx_lm, llama.cpp, vLLM expose it); else /v1/models, which some servers build
    slowly (mlx_lm scans the whole Hugging Face cache), hence the generous timeout."""
    headers = {"Authorization": f"Bearer {api_key or 'local'}"}
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        if httpx.get(f"{root}/health", headers=headers, timeout=5).status_code == 200:
            return True
    except httpx.HTTPError:
        pass
    try:
        return httpx.get(f"{base_url.rstrip('/')}/models", headers=headers, timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False


class ServerManager:
    def __init__(self):
        self.cfg = load_config()
        self.procs: dict[str, subprocess.Popen] = {}
        self.urls: dict[str, str] = {}
        self.log_dir = self.cfg.results_dir / "servers"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        atexit.register(self._kill_spawned)  # never leave a model server orphaned (they run in their own session)

    def _kill_spawned(self) -> None:
        for proc in list(self.procs.values()):
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    def log_path(self, spec: ModelSpec) -> Path:
        return self.log_dir / f"{spec.key}.log"

    # ---------------------------------------------------------------------------------------
    def ensure_up(self, spec: ModelSpec, *, force: bool = False, log=print) -> str | None:
        """Returns the base_url to use for chat calls (None for cloud/mock)."""
        if spec.kind in ("cloud", "mock"):
            return spec.base_url
        if spec.kind == "external":
            if not spec.base_url:
                raise ServerError(f"{spec.key}: external model needs base_url")
            if not health(spec.base_url, resolve_secret(spec.api_key)):
                raise ServerError(
                    f"{spec.key}: server at {spec.base_url} is not responding (start it yourself, "
                    "or make it kind: managed)"
                )
            return spec.base_url
        if spec.key in self.urls and health(self.urls[spec.key]):
            return self.urls[spec.key]
        if spec.mem_gb and not force:
            avail = free_memory_gb()
            if avail < spec.mem_gb:
                raise ServerError(
                    f"{spec.key}: needs ~{spec.mem_gb} GB but only {avail:.1f} GB is reclaimable; "
                    "stop other models or pass --force"
                )
        launcher = spec.launcher
        if launcher == "ollama":
            url = self._ollama_up(spec, log)
        else:
            url = self._spawn(spec, log)
        self.urls[spec.key] = url
        self._warmup(spec, url, log)
        return url

    def stop(self, spec: ModelSpec, log=print) -> None:
        if spec.kind != "managed":
            return
        if spec.launcher == "ollama":
            try:
                httpx.post("http://127.0.0.1:11434/api/generate",
                           json={"model": spec.model, "keep_alive": 0}, timeout=60)
                log(f"[server] unloaded ollama model {spec.model}")
            except httpx.HTTPError:
                pass
            self.urls.pop(spec.key, None)
            return
        proc = self.procs.pop(spec.key, None)
        self.urls.pop(spec.key, None)
        if proc and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=30)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            log(f"[server] stopped {spec.key}")

    def stop_all(self, specs: dict[str, ModelSpec]) -> None:
        for key in list(self.procs):
            if key in specs:
                self.stop(specs[key], log=lambda *_: None)

    # ---------------------------------------------------------------------------------------
    def _command(self, spec: ModelSpec, port: int) -> list[str]:
        m = spec.model or ""
        if spec.launcher == "llama_server":
            exe = shutil.which("llama-server") or "llama-server"
            src = ["-m", m] if (m.endswith(".gguf") or os.path.exists(m)) else ["-hf", m]
            return [exe, *src, "--host", "127.0.0.1", "--port", str(port), "-c", str(spec.context),
                    "--jinja", "-np", str(max(1, spec.max_concurrency)), *spec.launcher_args]
        if spec.launcher == "mlx_lm":
            exe = shutil.which("mlx_lm.server") or "mlx_lm.server"
            return [exe, "--model", m, "--host", "127.0.0.1", "--port", str(port), *spec.launcher_args]
        if spec.launcher == "mlx_vlm":
            exe = shutil.which("mlx_vlm.server")
            base = [exe] if exe else ["python3", "-m", "mlx_vlm.server"]
            return [*base, "--model", m, "--host", "127.0.0.1", "--port", str(port), *spec.launcher_args]
        if spec.launcher == "command" and spec.command:
            return shlex.split(spec.command.format(port=port, model=m, context=spec.context))
        raise ServerError(f"{spec.key}: unsupported launcher {spec.launcher!r}")

    def _spawn(self, spec: ModelSpec, log) -> str:
        port = spec.port or _free_port()
        cmd = self._command(spec, port)
        logf = open(self.log_path(spec), "ab")
        logf.write(f"\n==== {time.ctime()} :: {' '.join(cmd)}\n".encode())
        logf.flush()
        log(f"[server] launching {spec.key}: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
        self.procs[spec.key] = proc
        url = f"http://127.0.0.1:{port}/v1"
        deadline = time.time() + spec.startup_timeout_s
        while time.time() < deadline:
            if proc.poll() is not None:
                raise ServerError(f"{spec.key}: server exited with code {proc.returncode}; see {self.log_path(spec)}")
            if health(url):
                log(f"[server] {spec.key} is up at {url}")
                return url
            time.sleep(2)
        self.stop(spec, log)
        raise ServerError(f"{spec.key}: not healthy after {spec.startup_timeout_s}s; see {self.log_path(spec)}")

    def _ollama_up(self, spec: ModelSpec, log) -> str:
        api = "http://127.0.0.1:11434"
        try:
            httpx.get(f"{api}/api/version", timeout=3)
        except httpx.HTTPError:
            exe = shutil.which("ollama") or "ollama"
            logf = open(self.log_dir / "ollama.log", "ab")
            log("[server] starting `ollama serve`")
            self.procs["__ollama__"] = subprocess.Popen([exe, "serve"], stdout=logf, stderr=subprocess.STDOUT,
                                                        start_new_session=True)
            for _ in range(60):
                time.sleep(1)
                try:
                    httpx.get(f"{api}/api/version", timeout=3)
                    break
                except httpx.HTTPError:
                    continue
            else:
                raise ServerError("ollama serve did not come up")
        tags = httpx.get(f"{api}/api/tags", timeout=10).json().get("models", [])
        if not any(t.get("name") == spec.model or t.get("model") == spec.model for t in tags):
            log(f"[server] pulling ollama model {spec.model} (may take a while)")
            r = subprocess.run([shutil.which("ollama") or "ollama", "pull", spec.model or ""])
            if r.returncode != 0:
                raise ServerError(f"ollama pull {spec.model} failed")
        return f"{api}/v1"

    def _warmup(self, spec: ModelSpec, url: str, log) -> None:
        from bench.models.client import OpenAICompatClient

        model_id = spec.model if spec.launcher == "ollama" else (spec.model_id or spec.model)
        client = OpenAICompatClient(spec.model_copy(update={"model_id": model_id}), base_url=url)
        try:
            client.chat([{"role": "user", "content": "Reply with OK."}], max_tokens=16,
                        params={"stream": False}, timeout=600)
            log(f"[server] {spec.key} warm")
        except Exception as e:  # warmup is best-effort
            log(f"[server] warmup for {spec.key} failed: {e}")
