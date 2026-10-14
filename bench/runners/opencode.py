"""opencode runner: agentic tasks in a sandboxed per-sample workspace, plus the tool-less
`bench-plain` chat fallback for cloud models that are only reachable through opencode.

Isolation: `sandbox-exec` allows writes only to the workspace and opencode's own state dirs, and
denies reads of secret stores. Local models get localhost-only network; the agent's only route to
the web is the suite's search/fetch MCP server (running outside the sandbox, cached, logged).
Cloud models need outbound network to reach their provider.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

from bench import sandbox
from bench.benchmark import AgentTask, Benchmark, ModelOutput, RunContext
from bench.config import ModelSpec, load_config, resolve_secret

HOME = Path.home()

BENCH_AGENT_PROMPT = """You are an autonomous agent completing a benchmark task inside a working directory.
Work only inside the current working directory. Do not ask the user questions: there is no human
in the loop, so make reasonable assumptions and finish the task. When you are done, reply with
your final answer exactly as the task requests."""

PASS_ENV = [
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
    "GROQ_API_KEY", "DEEPSEEK_API_KEY", "XAI_API_KEY", "MISTRAL_API_KEY", "OPENCODE_API_KEY",
]


def opencode_env(config: dict) -> dict:
    env = {
        "OPENCODE_CONFIG_CONTENT": json.dumps(config),
        "OPENCODE_DISABLE_CLAUDE_CODE": "1",
        "OPENCODE_DISABLE_CLAUDE_CODE_SKILLS": "1",
        "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
        "OPENCODE_DISABLE_AUTOUPDATE": "1",
        "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
        "OPENCODE_DISABLE_LSP_DOWNLOAD": "1",
        "OPENCODE_DISABLE_SHARE": "1",
        "OPENCODE_DISABLE_MODELS_FETCH": "1",
        "XDG_CONFIG_HOME": os.environ.get("XDG_CONFIG_HOME", str(HOME / ".config")),
        # the agent's python/python3 = the suite venv (python-docx, openpyxl, python-pptx, pandas, pdf libs...)
        "PATH": f"{Path(sys.executable).parent}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        "VIRTUAL_ENV": str(Path(sys.executable).parent.parent),
    }
    for k in PASS_ENV:
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


def opencode_write_paths() -> list[Path | str]:
    tmp = os.path.realpath(os.environ.get("TMPDIR", "/tmp"))
    return [HOME / ".local/share/opencode", HOME / ".cache/opencode", HOME / ".local/state/opencode",
            f"{tmp}/opencode"]


def model_ref(spec: ModelSpec, base_url: str | None) -> tuple[str, dict]:
    """(provider/model string, provider config to inject)."""
    if spec.opencode_model and (spec.kind == "cloud" or not base_url or base_url == spec.base_url):
        return spec.opencode_model, {}
    if not base_url:
        raise RuntimeError(f"model {spec.key}: set opencode_model or a base_url for agentic runs")
    model_id = spec.model_id or spec.model or spec.key
    provider = {
        "benchlocal": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "bench local",
            "options": {"baseURL": base_url, "apiKey": resolve_secret(spec.api_key) or "local", "timeout": False},
            "models": {
                spec.key: {
                    "id": model_id,
                    "name": spec.name,
                    "tool_call": spec.tool_calling,
                    "attachment": spec.vision,
                    "limit": {"context": spec.context, "output": spec.max_output_tokens or 32768},
                }
            },
        }
    }
    return f"benchlocal/{spec.key}", provider


def build_config(provider: dict, *, mcp_url: str | None, agent_prompt: str | None,
                 read_dirs: list[Path] | None = None) -> dict:
    external: str | dict = "deny"
    if read_dirs:  # reads only: the sandbox still blocks writes outside the workspace
        external = {"*": "deny", **{f"{os.path.realpath(d)}/*": "allow" for d in read_dirs}}
    cfg: dict = {
        "$schema": "https://opencode.ai/config.json",
        "autoupdate": False,
        "share": "disabled",
        "agent": {
            "bench-agent": {
                "description": "Benchmark task agent",
                "mode": "primary",
                "prompt": BENCH_AGENT_PROMPT + ("\n\n" + agent_prompt if agent_prompt else ""),
                "permission": {"edit": "allow", "bash": "allow", "webfetch": "deny", "websearch": "deny",
                               "external_directory": external, "question": "deny"},
            },
            "bench-plain": {
                "description": "Plain chat for benchmarks (no tools)",
                "mode": "primary",
                "prompt": "You are a helpful assistant.",
                "tools": {"*": False},
                "permission": {"*": "deny"},
            },
            # Stock variants: opencode's own system prompt and toolset (opencode free tier requires both).
            "bench-agent-stock": {
                "description": "Benchmark task agent (stock opencode prompt)",
                "mode": "primary",
                "permission": {"edit": "allow", "bash": "allow", "webfetch": "deny", "websearch": "deny",
                               "external_directory": external, "question": "deny"},
            },
            "bench-plain-stock": {
                # tools stay listed (the free tier checks) but every call needs approval, and runs
                # without --auto reject it: effectively a tool-less chat with opencode's prompt
                "description": "Plain chat for benchmarks (stock opencode prompt, tools not executable)",
                "mode": "primary",
                "permission": {"*": "ask"},
            },
        },
    }
    if provider:
        cfg["provider"] = provider
    if mcp_url:
        cfg["mcp"] = {"bench": {"type": "remote", "url": mcp_url, "enabled": True}}
    return cfg


class RateLimitWatch:
    """Watches opencode's log for provider rate limiting of one model. opencode retries throttled
    requests silently (sometimes rapidly, sometimes after long back-offs), so without this a throttled
    sample would hang until its timeout. Aborts when retries pile up, or when a rate-limit error is not
    followed by a new attempt within `stall_s`."""

    LOG = HOME / ".local/share/opencode/log/opencode.log"

    def __init__(self, opencode_model: str, threshold: int = 120, stall_s: float = 120):
        self.model_id = opencode_model.split("/", 1)[-1]
        self.threshold, self.stall_s = threshold, stall_s
        self.count = 0
        self.stalled_since: float | None = None  # set by a rate-limit line, cleared by a later attempt line
        try:
            self.pos = self.LOG.stat().st_size
        except OSError:
            self.pos = None

    def __call__(self) -> str | None:
        if self.pos is None:
            return None
        try:
            with open(self.LOG, errors="replace") as f:
                f.seek(self.pos)
                chunk = f.read()
                self.pos = f.tell()
        except OSError:
            return None
        now = time.time()
        for line in chunk.splitlines():  # in log order, so "attempt then error" in one read is handled
            if f"modelID={self.model_id}" not in line:
                continue
            if "Rate limit" in line:
                self.count += 1
                self.stalled_since = self.stalled_since or now
            elif "message=stream " in line:
                self.stalled_since = None
        if self.count >= self.threshold:
            return f"provider rate limit: {self.count} throttled attempts for {self.model_id}; retry later"
        if self.stalled_since and now - self.stalled_since > self.stall_s:
            return f"provider rate limit: {self.model_id} is throttled and opencode is backing off; retry later"
        return None


def parse_events(stdout: str) -> dict:
    texts: list[str] = []
    tools: list[dict] = []
    usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "cache_read_tokens": 0}
    cost = 0.0
    session = None
    errors: list[str] = []
    steps = 0
    for line in stdout.splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        session = session or e.get("sessionID")
        p = e.get("part") or {}
        t = e.get("type")
        if t == "text" and p.get("text", "").strip():
            texts.append(p["text"])
        elif t == "tool_use":
            st = p.get("state") or {}
            tools.append({"tool": p.get("tool"), "status": st.get("status")})
        elif t == "step_finish":
            steps += 1
            tok = p.get("tokens") or {}
            usage["input_tokens"] += (tok.get("input") or 0) + ((tok.get("cache") or {}).get("read") or 0)
            usage["output_tokens"] += tok.get("output") or 0
            usage["reasoning_tokens"] += tok.get("reasoning") or 0
            usage["cache_read_tokens"] += (tok.get("cache") or {}).get("read") or 0
            cost += float(p.get("cost") or 0)
        elif t == "error":
            errors.append(json.dumps(e.get("error") or e)[:500])
    return {"final": texts[-1].strip() if texts else "", "texts": texts, "tools": tools, "usage": usage,
            "cost": cost, "session": session, "errors": errors, "steps": steps}


def run_agent(bench: Benchmark, task: AgentTask, ctx: RunContext, *, sample_key: str,
              base_url: str | None, mcp_url: str | None = None) -> ModelOutput:
    cfg = load_config()
    ws = ctx.run_dir / "work" / sample_key
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    for dest, src in task.files.items():
        target = ws / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        if Path(src).is_dir():
            shutil.copytree(src, target)
        else:
            shutil.copy2(src, target)
    model, provider = model_ref(ctx.model, base_url)
    oc_cfg = build_config(provider, mcp_url=mcp_url if task.network else None, agent_prompt=task.agent_prompt,
                          read_dirs=task.read_dirs)
    stock = ctx.model.opencode_stock_prompt
    prompt = task.prompt
    if stock:  # the system prompt must stay opencode's own, so the suite's guidance goes in the message
        guidance = BENCH_AGENT_PROMPT + ("\n\n" + task.agent_prompt if task.agent_prompt else "")
        prompt = f"{guidance}\n\n---\n\n{task.prompt}"
    cmd = ["opencode", "run", "-m", model, "--agent", "bench-agent-stock" if stock else "bench-agent", "--format",
           "json", "--pure", "--auto", "--dir", str(ws)]
    if ctx.model.opencode_variant:
        cmd += ["--variant", ctx.model.opencode_variant]
    for a in task.attachments:
        cmd += ["-f", str(ws / a)]
    cmd += ["--", prompt]  # `--`: the message can never be taken as a file or a flag
    network = "localhost" if ctx.model.is_local else "all"
    timeout = task.timeout_s or bench.spec.sample_timeout_s or cfg.defaults.agent_timeout_s
    t0 = time.time()
    r = sandbox.run(cmd, cwd=ws, timeout=timeout, write_paths=opencode_write_paths(), network=network,
                    env=opencode_env(oc_cfg), watch=RateLimitWatch(model))
    if r.aborted:
        raise RuntimeError(r.aborted)
    tdir = ctx.run_dir / "transcripts"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / f"{sample_key}.jsonl").write_text(r.stdout)
    if r.stderr.strip():
        (tdir / f"{sample_key}.stderr.txt").write_text(r.stderr)
    ev = parse_events(r.stdout)
    finish = "timeout" if r.timed_out else ("error" if (r.returncode != 0 or ev["errors"]) and not ev["final"] else "stop")
    return ModelOutput(
        text=ev["final"],
        usage={**ev["usage"], "cost_usd": ev["cost"]},
        latency=time.time() - t0,
        finish_reason=finish,
        workspace=ws,
        artifacts={"transcript": f"transcripts/{sample_key}.jsonl", "workspace": f"work/{sample_key}"},
        extra={"session": ev["session"], "steps": ev["steps"], "tools": ev["tools"][-200:],
               "errors": ev["errors"], "returncode": r.returncode, "timed_out": r.timed_out,
               "all_text": "\n\n".join(ev["texts"])[-20000:]},
    )


def opencode_plain_chat(spec: ModelSpec, messages: list[dict], *, timeout: float) -> tuple[str, dict]:
    """Single response via `opencode run --agent bench-plain` (no tools) for models only reachable
    through opencode (e.g. opencode's free tier, which refuses direct API calls).

    Images are attached as files (`-f`); prompts too long for a command-line argument are attached as
    prompt.md with a one-line pointer message.
    """
    if not spec.opencode_model:
        raise RuntimeError(f"model {spec.key}: provider=opencode needs opencode_model")
    import base64
    import tempfile

    cfg = load_config()
    base = cfg.results_dir / "tmp" / "opencode-plain"
    base.mkdir(parents=True, exist_ok=True)
    call_dir = Path(tempfile.mkdtemp(prefix="call-", dir=base))
    try:
        files: list[Path] = []
        parts = []
        for m in messages:
            c = m.get("content")
            if isinstance(c, list):
                texts = []
                for part in c:
                    if part.get("type") == "text":
                        texts.append(part["text"])
                    elif part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        if url.startswith("data:"):
                            header, data = url.split(",", 1)
                            ext = header[5:].split(";")[0].split("/")[-1] or "png"
                            f = call_dir / f"image_{len(files) + 1}.{ext}"
                            f.write_bytes(base64.b64decode(data))
                            files.append(f)
                            texts.append(f"[image {len(files)}: attached as {f.name}]")
                        else:
                            texts.append(f"[image: {url}]")
                c = "\n".join(texts)
            parts.append(c if (m["role"] == "user" and len(messages) == 1) else f"[{m['role']}]\n{c}")
        prompt = "\n\n".join(parts)
        if len(prompt) > 200_000:  # stay well under macOS ARG_MAX
            pf = call_dir / "prompt.md"
            pf.write_text(prompt)
            files.append(pf)
            prompt = "Your full task is in the attached file prompt.md. Read it completely and respond exactly as it asks."
        oc_cfg = build_config({}, mcp_url=None, agent_prompt=None)
        stock = spec.opencode_stock_prompt
        base_cmd = ["opencode", "run", "-m", spec.opencode_model, "--agent", "bench-plain-stock" if stock else "bench-plain",
                    "--format", "json", "--pure", "--dir", str(call_dir)]  # never --auto: stock tools must stay inert
        if spec.opencode_variant:
            base_cmd += ["--variant", spec.opencode_variant]
        usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "cache_read_tokens": 0}
        cost, texts, session = 0.0, [], None
        deadline = time.time() + timeout
        for attempt in range(3):
            if attempt == 0:
                cmd = list(base_cmd)
                for f in files:
                    cmd += ["-f", str(f)]
                cmd += ["--", prompt]  # `--`: the message can never be taken as a file or a flag
            else:  # stock mode: a tool call was auto-rejected, which ends the turn; ask for a direct answer
                cmd = base_cmd + ["--session", session, "--",
                                  "Tools are not available in this environment and every tool call will be rejected. "
                                  "Answer the original request directly now, exactly in the format it asks for."]
            r = sandbox.run(cmd, cwd=call_dir, timeout=max(60.0, deadline - time.time()), write_paths=opencode_write_paths(),
                            network="all", env=opencode_env(oc_cfg), watch=RateLimitWatch(spec.opencode_model))
            if r.aborted:
                raise RuntimeError(r.aborted)
            ev = parse_events(r.stdout)
            for k in usage:
                usage[k] += ev["usage"].get(k) or 0
            cost += ev["cost"]
            session = session or ev["session"]
            if ev["texts"]:
                texts = ev["texts"]
            rejected = any(t.get("status") == "error" for t in ev["tools"])
            if texts or not (stock and rejected and session):
                break
    finally:
        shutil.rmtree(call_dir, ignore_errors=True)
    if not texts and (r.returncode != 0 or ev["errors"]):
        raise RuntimeError(f"opencode failed (rc={r.returncode}): {ev['errors'] or r.stderr[-500:]}")
    return "\n".join(texts).strip(), {**usage, "cost_usd": cost, "attempts": attempt + 1}
