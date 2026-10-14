"""macOS `sandbox-exec` wrapper for model-written code and agent runs (Docker-free isolation).

Policy: reads are allowed except well-known secret stores; writes are allowed only under the
given paths (plus /dev/null & friends); network is none | localhost | all.
"""

from __future__ import annotations

import os
import resource
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

Network = Literal["none", "localhost", "all"]

HOME = Path.home()
SECRET_PATHS = [
    HOME / ".ssh",
    HOME / ".aws",
    HOME / ".gnupg",
    HOME / ".config" / "gh",
    HOME / ".netrc",
    HOME / ".docker" / "config.json",
    HOME / "Library" / "Keychains",
    HOME / ".config" / "anthropic",
]


def _real(p: Path | str) -> str:
    return os.path.realpath(str(p))


def _q(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def profile(write_paths: list[Path | str], network: Network = "none",
            deny_read: list[Path | str] | None = None) -> str:
    lines = ["(version 1)", "(allow default)", "(deny file-write*)"]
    allow = [f"(subpath {_q(_real(p))})" for p in write_paths]
    allow += ['(literal "/dev/null")', '(literal "/dev/zero")', '(literal "/dev/dtracehelper")',
              '(regex #"^/dev/tty")', '(regex #"^/dev/fd/")', '(literal "/dev/stdout")', '(literal "/dev/stderr")']
    lines.append(f"(allow file-write* {' '.join(allow)})")
    secrets = [*SECRET_PATHS, *(deny_read or [])]
    rules = []
    for p in secrets:
        rp = _real(p)
        rules.append(f"(subpath {_q(rp)})" if Path(rp).is_dir() or not Path(rp).suffix else f"(literal {_q(rp)})")
    lines.append(f"(deny file-read* {' '.join(rules)})")
    if network == "none":
        lines.append("(deny network*)")
        lines.append('(allow network* (local unix-socket) (remote unix-socket))')
    elif network == "localhost":
        lines.append("(deny network*)")
        lines.append('(allow network* (remote ip "localhost:*") (local ip "localhost:*") '
                     '(local unix-socket) (remote unix-socket))')
    return "\n".join(lines) + "\n"


@dataclass
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    seconds: float
    aborted: str | None = None  # reason, when a watch callback stopped the process


def run(cmd: list[str], *, cwd: Path | str, timeout: float, write_paths: list[Path | str] | None = None,
        network: Network = "none", env: dict | None = None, input: str | None = None,
        cpu_seconds: int | None = None, max_output: int = 2_000_000,
        watch: Callable[[], str | None] | None = None) -> SandboxResult:
    """watch: polled every ~2 s while the process runs; returning a reason kills it (e.g. provider
    rate limiting that the tool would otherwise retry forever)."""
    if not shutil.which("sandbox-exec"):
        raise RuntimeError("sandbox-exec not found (macOS only)")
    cwd = Path(cwd)
    writes = [cwd, *(write_paths or [])]
    tmp = Path(tempfile.mkdtemp(prefix="sbx-", dir=cwd)) if cwd.exists() else None
    if tmp:
        writes.append(tmp)
    prof = profile(writes, network)
    full_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(HOME),
        "LANG": "en_US.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        **({"TMPDIR": str(tmp)} if tmp else {}),
        **(env or {}),
    }

    def _limits():
        if cpu_seconds:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))

    t0 = time.time()
    proc = subprocess.Popen(
        ["sandbox-exec", "-p", prof, *cmd], cwd=cwd, env=full_env, stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace",
        start_new_session=True, preexec_fn=_limits,
    )
    timed_out = False
    aborted: str | None = None
    box: dict = {}

    def _communicate():
        box["out"], box["err"] = proc.communicate(input=input)

    reader = threading.Thread(target=_communicate, daemon=True)
    reader.start()
    try:
        while reader.is_alive():
            reader.join(timeout=2)
            if not reader.is_alive():
                break
            if time.time() - t0 > timeout:
                timed_out = True
            elif watch is not None:
                aborted = watch()
            if timed_out or aborted:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                reader.join(timeout=30)
                break
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    out, err = box.get("out", ""), box.get("err", "")
    return SandboxResult(proc.returncode if not (timed_out or aborted) else -9, (out or "")[:max_output],
                         (err or "")[-20000:], timed_out, time.time() - t0, aborted)
