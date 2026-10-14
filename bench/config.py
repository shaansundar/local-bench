"""Loads config.yaml and models.yaml, and resolves secrets.

models.yaml is your own (gitignored) model registry; until you create one, the committed
models.example.yaml is used. BENCH_MODELS and BENCH_RESULTS_DIR point either somewhere else.

Secrets are never stored in this repo's YAML. An `api_key` value may be:
  - "env:VAR"            read from the environment (.env is loaded on startup)
  - "opencode:PROVIDER"  read from ~/.config/opencode/opencode.json[c] provider options
  - a literal string     (fine for local servers that ignore the key, e.g. "local")
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import json5
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


class JudgeConfig(BaseModel):
    model: str | None = None  # a models.yaml key
    max_concurrency: int = 8
    max_tokens: int = 4096


class SearchConfig(BaseModel):
    max_results: int = 8
    min_interval_s: float = 1.5  # keyless DuckDuckGo is rate-limited; stay polite
    fetch_max_chars: int = 20000
    region: str = "wt-wt"


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765


class Defaults(BaseModel):
    max_tokens: int = 32768
    request_timeout_s: float = 3600
    agent_timeout_s: float = 1800
    grade_workers: int = 8


class Config(BaseModel):
    seed: int = 1234
    tiers: dict[str, int | None] = Field(
        default_factory=lambda: {"smoke": 5, "quick": 50, "standard": 300, "full": None}
    )
    default_tier: str = "quick"
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    defaults: Defaults = Field(default_factory=Defaults)
    data_dir: Path = ROOT / "data"
    results_dir: Path = ROOT / "results"


class ModelSpec(BaseModel):
    """One entry in models.yaml."""

    key: str = ""
    display_name: str | None = None
    # external: you run the server; managed: we launch/stop it; cloud: hosted API; mock: oracle/null
    kind: Literal["external", "managed", "cloud", "mock"] = "external"
    # wire protocol for direct (chat-runner) calls
    provider: Literal["openai", "anthropic", "opencode", "mock"] = "openai"
    base_url: str | None = None
    model_id: str | None = None
    api_key: str | None = None
    # provider/model string for `opencode run -m` (agentic runs, and chat fallback)
    opencode_model: str | None = None
    opencode_variant: str | None = None
    # Keep opencode's own system prompt and toolset (required by opencode's free tier, which rejects
    # sessions with a replaced system prompt or no tools). Suite guidance then goes in the user message.
    opencode_stock_prompt: bool = False
    # managed lifecycle
    launcher: Literal["ollama", "llama_server", "mlx_lm", "mlx_vlm", "command"] | None = None
    model: str | None = None  # what the launcher loads (ollama tag, gguf path / hf repo, mlx repo)
    command: str | None = None  # launcher=command: template with {port} {model} {context}
    launcher_args: list[str] = Field(default_factory=list)
    port: int | None = None
    mem_gb: float | None = None
    startup_timeout_s: float = 900
    # capabilities
    context: int = 32768
    max_output_tokens: int | None = None
    vision: bool = False
    tool_calling: bool = True
    max_concurrency: int = 1
    # extra request params merged into every chat call (temperature, top_p, reasoning_effort,
    # chat_template_kwargs, output_config, thinking, ...)
    params: dict[str, Any] = Field(default_factory=dict)
    request_timeout_s: float | None = None
    notes: str | None = None

    @property
    def name(self) -> str:
        return self.display_name or self.key

    @property
    def is_local(self) -> bool:
        return self.kind in ("external", "managed")


@lru_cache
def load_config() -> Config:
    path = ROOT / "config.yaml"
    raw = yaml.safe_load(path.read_text()) if path.exists() else {}
    cfg = Config(**(raw or {}))
    if os.environ.get("BENCH_JUDGE_MODEL"):  # quick override, e.g. a local stand-in judge for smoke tests
        cfg.judge.model = os.environ["BENCH_JUDGE_MODEL"]
    if os.environ.get("BENCH_RESULTS_DIR"):  # e.g. a separate results store per machine, or demo data
        cfg.results_dir = Path(os.environ["BENCH_RESULTS_DIR"]).expanduser()
    cfg.data_dir = (ROOT / cfg.data_dir).resolve() if not cfg.data_dir.is_absolute() else cfg.data_dir
    cfg.results_dir = (
        (ROOT / cfg.results_dir).resolve() if not cfg.results_dir.is_absolute() else cfg.results_dir
    )
    return cfg


MOCK_MODELS = {
    "oracle": ModelSpec(key="oracle", kind="mock", provider="mock", notes="returns gold answers"),
    "null": ModelSpec(key="null", kind="mock", provider="mock", notes="returns empty answers"),
}


def models_file() -> Path:
    if os.environ.get("BENCH_MODELS"):
        return Path(os.environ["BENCH_MODELS"]).expanduser()
    own = ROOT / "models.yaml"
    return own if own.exists() else ROOT / "models.example.yaml"


def load_models() -> dict[str, ModelSpec]:
    path = models_file()
    raw = (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}
    models: dict[str, ModelSpec] = {}
    for key, entry in (raw.get("models") or {}).items():
        models[key] = ModelSpec(key=key, **(entry or {}))
    for key, spec in MOCK_MODELS.items():
        models.setdefault(key, spec)
    return models


def get_model(key: str) -> ModelSpec:
    models = load_models()
    if key not in models:
        raise KeyError(f"unknown model '{key}' (known: {', '.join(sorted(models))})")
    return models[key]


def opencode_config_paths() -> list[Path]:
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "opencode"
    return [base / "opencode.jsonc", base / "opencode.json"]


@lru_cache
def opencode_user_config() -> dict:
    for path in opencode_config_paths():
        if path.exists():
            return json5.loads(path.read_text())
    return {}


def resolve_secret(value: str | None) -> str | None:
    if not value:
        return None
    if value.startswith("env:"):
        return os.environ.get(value[4:])
    if value.startswith("opencode:"):
        provider = value.split(":", 1)[1]
        opts = (opencode_user_config().get("provider", {}).get(provider, {}) or {}).get("options", {})
        return _expand_opencode_template(opts.get("apiKey"))
    return value


def _expand_opencode_template(value: str | None) -> str | None:
    """opencode config values may be "{env:VAR}" or "{file:path}" substitutions."""
    if not value:
        return value
    m = re.fullmatch(r"\{(env|file):(.+)\}", value.strip())
    if not m:
        return value
    kind, arg = m.groups()
    if kind == "env":
        return os.environ.get(arg)
    path = Path(arg).expanduser()
    if not path.is_absolute():
        path = opencode_config_paths()[0].parent / path
    return path.read_text().strip() if path.exists() else None
