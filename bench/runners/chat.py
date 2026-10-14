"""Direct chat runner: single-turn, scripted multi-turn, observe-loops, and tool loops."""

from __future__ import annotations

import json

from bench.benchmark import Benchmark, ChatRequest, ModelOutput, RunContext
from bench.config import load_config
from bench.models.client import ChatResult


def resolve_max_tokens(req_max: int | None, bench: Benchmark, ctx: RunContext) -> int:
    cfg = load_config()
    want = req_max or bench.spec.max_tokens or cfg.defaults.max_tokens
    if ctx.model.max_output_tokens:
        want = min(want, ctx.model.max_output_tokens)
    return want


def _add_usage(total: dict, res: ChatResult) -> None:
    for k in ("input_tokens", "output_tokens", "reasoning_tokens"):
        v = res.usage.get(k)
        if v:
            total[k] = total.get(k, 0) + int(v)
    if res.usage.get("estimated"):
        total["estimated"] = True


def run_chat(bench: Benchmark, req: ChatRequest, ctx: RunContext) -> ModelOutput:
    client = ctx.client
    max_tokens = resolve_max_tokens(req.max_tokens, bench, ctx)
    messages = list(req.messages)
    usage: dict = {}
    latency = 0.0
    texts: list[str] = []
    reasonings: list[str] = []

    def call() -> ChatResult:
        nonlocal latency
        res = client.chat(messages, max_tokens=max_tokens, params=req.params, tools=req.tools)
        _add_usage(usage, res)
        latency += res.latency
        texts.append(res.text)
        if res.reasoning:
            reasonings.append(res.reasoning)
        return res

    res = call()
    turn = 0
    # tool-calling loop
    while req.tools and req.tool_handler and res.tool_calls and turn < req.max_turns:
        turn += 1
        messages.append({"role": "assistant", "content": res.text or None, "tool_calls": res.tool_calls})
        for tc in res.tool_calls:
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            try:
                out = req.tool_handler(tc["function"]["name"], args)
            except Exception as e:  # tool errors go back to the model
                out = f"ERROR: {e}"
            messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": str(out)})
        res = call()
    # observe loop (e.g. execute code blocks, return stdout)
    while req.observe and turn < req.max_turns:
        obs = req.observe(res.text, turn)
        if obs is None:
            break
        turn += 1
        messages.append({"role": "assistant", "content": res.text})
        messages.append({"role": "user", "content": obs})
        res = call()
    # scripted follow-up turns
    for follow in req.followups:
        messages.append({"role": "assistant", "content": res.text})
        messages.append({"role": "user", "content": follow})
        res = call()
    messages.append({"role": "assistant", "content": res.text})
    return ModelOutput(
        text=res.text,
        reasoning="\n\n---\n\n".join(reasonings) or None,
        usage=usage,
        latency=latency,
        finish_reason=res.finish_reason,
        turns=messages if (req.followups or req.observe or req.tools) else [],
        artifacts=dict(req.artifacts),
        extra={"turn_texts": texts} if len(texts) > 1 else {},
    )
