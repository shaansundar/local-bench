"""The benchmark contract. Every benchmark is a folder benchmarks/<slug>/ holding:

  benchmark.yaml   metadata (validated by BenchmarkSpec)
  adapter.py       a subclass of Benchmark exported as `BENCHMARK`
  subsets/         frozen sample-id lists per tier (written by `bench subsets`)

See docs/NEW_BENCHMARK.md for the full guide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

from pydantic import BaseModel, Field

from bench.config import ROOT, ModelSpec, load_config

if TYPE_CHECKING:
    from bench.judge import Judge
    from bench.models.client import BaseClient

CATEGORIES = [
    "Coding",
    "Math",
    "Instruction following",
    "Knowledge",
    "Long context",
    "Tool use",
    "Search agent",
    "General agent",
    "Spreadsheet",
    "Presentation",
    "Office docs",
    "Doc / vision",
    "Text-to-SQL",
]


class Requires(BaseModel):
    vision: bool = False
    min_context: int = 0
    tool_calling: bool = False
    network: bool = False  # live web access during the run (search agents)
    judge_runtime: bool = False  # the judge model is needed while generating (e.g. tau-bench user simulator)


class SourceRef(BaseModel):
    url: str
    revision: str | None = None
    note: str | None = None


class BenchmarkSpec(BaseModel):
    slug: str
    title: str
    category: str
    description: str = ""
    runner: Literal["chat", "oc", "native"] = "chat"
    metric: str = "accuracy"
    higher_is_better: bool = True
    score_range: tuple[float, float] = (0.0, 1.0)
    judge: bool = False  # some or all samples need the cloud judge
    repeats: int = 1  # generations per sample (avg@k)
    tiers: dict[str, int | None] = Field(default_factory=dict)  # overrides config tiers
    stratify: bool = True
    max_tokens: int | None = None
    sample_timeout_s: float | None = None
    requires: Requires = Field(default_factory=Requires)
    badges: list[Literal["adapted", "substitute", "judge", "vision", "partial"]] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    license: str | None = None
    gated: bool = False  # HF terms must be accepted before fetch
    notes: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)  # benchmark-specific settings
    oracle_min: float = 0.95  # `bench validate` threshold for the oracle (gold) response
    null_max: float = 0.05  # `bench validate` ceiling for the empty response


@dataclass
class Sample:
    id: str
    data: dict[str, Any]
    gold: Any = None
    stratum: str | None = None


@dataclass
class ChatRequest:
    """A (possibly multi-turn) direct chat with the model under test."""

    messages: list[dict]
    max_tokens: int | None = None
    params: dict[str, Any] = field(default_factory=dict)
    # Multi-turn scripted conversations (e.g. Multi-IF): user turns sent after the first answer.
    followups: list[str] = field(default_factory=list)
    # Execution loops (e.g. spreadsheet code): given the assistant's latest reply, return an
    # observation to send back as the next user turn, or None to stop.
    observe: Callable[[str, int], str | None] | None = None
    max_turns: int = 1
    tools: list[dict] | None = None
    tool_handler: Callable[[str, dict], str] | None = None
    artifacts: dict[str, str] = field(default_factory=dict)  # name -> path relative to the run dir


@dataclass
class AgentTask:
    """An agentic task run by opencode inside a sandboxed workspace."""

    prompt: str
    files: dict[str, Path] = field(default_factory=dict)  # workspace-relative dest -> source path
    network: bool = False
    timeout_s: float | None = None
    agent_prompt: str | None = None  # extra system guidance for the bench agent
    attachments: list[str] = field(default_factory=list)  # workspace-relative paths for -f
    read_dirs: list[Path] = field(default_factory=list)  # extra read-only directories the agent may open


@dataclass
class ModelOutput:
    text: str
    reasoning: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    latency: float = 0.0
    finish_reason: str | None = None
    turns: list[dict] = field(default_factory=list)  # full conversation, for multi-turn / agents
    workspace: Path | None = None  # agentic runs: the task's working directory
    artifacts: dict[str, str] = field(default_factory=dict)  # name -> path (relative to run dir)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Grade:
    score: float | None  # None = could not be graded yet (e.g. judge pending)
    extracted: Any = None
    details: dict[str, Any] = field(default_factory=dict)
    judge: dict[str, Any] | None = None
    pending: bool = False


class JudgeUnavailable(RuntimeError):
    pass


@dataclass
class RunContext:
    model: ModelSpec
    client: BaseClient | None
    run_dir: Path
    judge: Judge | None = None
    tier: str = "quick"
    log: Callable[[str], None] = print
    base_url: str | None = None  # where the model under test is being served (local models)
    mcp_url: str | None = None  # suite search/fetch MCP server (search agents)

    def require_judge(self) -> Judge:
        if self.judge is None or not self.judge.available:
            raise JudgeUnavailable("judge model not configured (config.yaml judge.model)")
        return self.judge


class Benchmark:
    """Subclass and implement load/build/grade (and execute for runner=native)."""

    spec: BenchmarkSpec

    def __init__(self, spec: BenchmarkSpec, folder: Path):
        self.spec = spec
        self.folder = folder
        self.cfg = load_config()

    # ---- data ----------------------------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        d = self.cfg.data_dir / self.spec.slug
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def vendor_dir(self) -> Path:
        return ROOT / "vendor"

    def is_fetched(self) -> bool:
        return (self.data_dir / ".fetched").exists()

    def mark_fetched(self) -> None:
        (self.data_dir / ".fetched").write_text("ok")

    def fetch(self) -> None:
        """Download everything needed into self.data_dir (idempotent). Call mark_fetched()."""
        raise NotImplementedError

    def load(self) -> list[Sample]:
        raise NotImplementedError

    # ---- running -------------------------------------------------------------------------
    def build(self, sample: Sample, ctx: RunContext) -> ChatRequest | AgentTask:
        raise NotImplementedError

    def execute(self, sample: Sample, ctx: RunContext) -> ModelOutput:
        """runner=native: run the official harness for one sample. Default: unsupported."""
        raise NotImplementedError

    def grade(self, sample: Sample, output: ModelOutput, ctx: RunContext) -> Grade:
        raise NotImplementedError

    # ---- self-tests ----------------------------------------------------------------------
    def oracle(self, sample: Sample) -> str | None:
        """A response that should score ~1.0 (used by `bench validate`). None = no oracle."""
        return None

    def oracle_output(self, sample: Sample, ctx: RunContext) -> ModelOutput | None:
        text = self.oracle(sample)
        return None if text is None else ModelOutput(text=text)

    # ---- aggregation ---------------------------------------------------------------------
    def aggregate(self, rows: list[dict]) -> dict[str, Any]:
        """Extra metrics from graded rows (dicts with sample_id, rep, score, details)."""
        return {}
