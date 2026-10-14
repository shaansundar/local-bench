"""Chat clients for models under test and for the judge.

- OpenAICompatClient: any /v1/chat/completions server (mlx_lm, mlx-vlm, llama.cpp, ollama, vLLM,
  OpenAI, OpenRouter). Streams, so long reasoning outputs never hit idle timeouts.
- AnthropicClient: Claude via the official `anthropic` SDK (Messages API, streaming).
- OpencodeChatClient: cloud models reachable only through opencode (no API key). Runs a tool-less
  `bench-plain` agent; results are flagged harness="opencode-plain".
"""

from __future__ import annotations

import base64
import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from bench.config import ModelSpec, load_config, resolve_secret


class ModelError(RuntimeError):
    pass


# Streamed-chunk counter (≈ tokens) for live throughput in the dashboard; one chunk per delta.
STREAMED = [0]


class ContextOverflow(ModelError):
    pass


@dataclass
class ChatResult:
    text: str
    reasoning: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    latency: float = 0.0
    ttft: float | None = None
    raw: dict[str, Any] | None = None

    @property
    def output_tokens(self) -> int:
        return int(self.usage.get("output_tokens") or 0)


THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)


def split_think(text: str, reasoning: str | None) -> tuple[str, str | None]:
    """Move <think>…</think> spans out of the visible answer (llama.cpp/ollama without a
    reasoning parser, and templates that put the opening tag in the prompt)."""
    if not text:
        return text, reasoning
    parts: list[str] = [reasoning] if reasoning else []
    if "<think>" in text:
        parts += THINK_RE.findall(text)
        text = THINK_RE.sub("", text)
    if "</think>" in text:  # opening tag was in the prompt
        head, _, text = text.rpartition("</think>")
        parts.append(head)
    text = text.strip()
    joined = "\n".join(p.strip() for p in parts if p and p.strip())
    return text, (joined or None)


class BaseClient:
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        cfg = load_config()
        self.timeout = spec.request_timeout_s or cfg.defaults.request_timeout_s

    def chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        params: dict | None = None,
        tools: list[dict] | None = None,
        timeout: float | None = None,
    ) -> ChatResult:
        raise NotImplementedError

    def _retrying(self, fn, attempts: int = 4):
        delay = 2.0
        for i in range(attempts):
            try:
                return fn()
            except ContextOverflow:
                raise
            except (httpx.TransportError, _Retryable) as e:
                if i == attempts - 1:
                    raise ModelError(f"{type(e).__name__}: {e}") from e
                time.sleep(delay + random.random())
                delay *= 2


class _Retryable(Exception):
    pass


# --------------------------------------------------------------------------------------------
# OpenAI-compatible


class OpenAICompatClient(BaseClient):
    def __init__(self, spec: ModelSpec, base_url: str | None = None):
        super().__init__(spec)
        self.base_url = (base_url or spec.base_url or "").rstrip("/")
        if not self.base_url:
            raise ModelError(f"model {spec.key}: no base_url")
        key = resolve_secret(spec.api_key) or "local"
        self.headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self.model_id = spec.model_id or spec.model or spec.key
        self._stream_usage_ok = True

    def chat(self, messages, *, max_tokens=None, params=None, tools=None, timeout=None) -> ChatResult:
        body: dict[str, Any] = {"model": self.model_id, "messages": messages}
        if max_tokens:
            body["max_tokens"] = max_tokens
        if tools:
            body["tools"] = tools
        merged = {**self.spec.params, **(params or {})}
        stream = merged.pop("stream", True)
        body.update({k: v for k, v in merged.items() if v is not None})
        t = timeout or self.timeout
        if stream:
            return self._retrying(lambda: self._stream(body, t))
        return self._retrying(lambda: self._once(body, t))

    def _check(self, r: httpx.Response) -> None:
        if r.status_code in (429, 500, 502, 503, 504):
            raise _Retryable(f"HTTP {r.status_code}: {r.text[:300]}")
        if r.status_code >= 400:
            body = r.text[:1000]
            if re.search(r"context|too long|maximum.*tokens|exceed", body, re.I):
                raise ContextOverflow(f"HTTP {r.status_code}: {body}")
            raise ModelError(f"HTTP {r.status_code}: {body}")

    def _once(self, body: dict, timeout: float) -> ChatResult:
        t0 = time.time()
        r = httpx.post(f"{self.base_url}/chat/completions", headers=self.headers, json=body, timeout=timeout)
        self._check(r)
        d = r.json()
        choice = (d.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text, reasoning = split_think(msg.get("content") or "", msg.get("reasoning_content") or msg.get("reasoning"))
        return ChatResult(
            text=text,
            reasoning=reasoning,
            tool_calls=msg.get("tool_calls") or [],
            finish_reason=choice.get("finish_reason"),
            usage=_norm_openai_usage(d.get("usage"), text, reasoning),
            latency=time.time() - t0,
        )

    def _stream(self, body: dict, timeout: float) -> ChatResult:
        body = {**body, "stream": True}
        if self._stream_usage_ok:
            body["stream_options"] = {"include_usage": True}
        t0 = time.time()
        ttft = None
        content: list[str] = []
        reasoning: list[str] = []
        tool_calls: dict[int, dict] = {}
        finish = None
        usage = None
        to = httpx.Timeout(timeout, connect=30.0, read=min(timeout, 900.0))
        with httpx.stream("POST", f"{self.base_url}/chat/completions", headers=self.headers, json=body, timeout=to) as r:
            if r.status_code >= 400:
                r.read()
                if r.status_code == 400 and "stream_options" in r.text and self._stream_usage_ok:
                    self._stream_usage_ok = False
                    body.pop("stream_options", None)
                    return self._stream(body, timeout)
                self._check(r)
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if chunk.get("error"):
                    raise ModelError(f"stream error: {chunk['error']}")
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for ch in chunk.get("choices") or []:
                    delta = ch.get("delta") or {}
                    piece_r = delta.get("reasoning_content") or delta.get("reasoning")
                    piece_c = delta.get("content")
                    if (piece_r or piece_c) and ttft is None:
                        ttft = time.time() - t0
                    if piece_r:
                        reasoning.append(piece_r)
                        STREAMED[0] += 1
                    if piece_c:
                        content.append(piece_c)
                        STREAMED[0] += 1
                    for tc in delta.get("tool_calls") or []:
                        slot = tool_calls.setdefault(
                            tc.get("index", 0),
                            {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
                        )
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            slot["function"]["arguments"] += fn["arguments"]
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
        text, reas = split_think("".join(content), "".join(reasoning) or None)
        return ChatResult(
            text=text,
            reasoning=reas,
            tool_calls=[tool_calls[i] for i in sorted(tool_calls)],
            finish_reason=finish,
            usage=_norm_openai_usage(usage, text, reas),
            latency=time.time() - t0,
            ttft=ttft,
        )


def _norm_openai_usage(u: dict | None, text: str, reasoning: str | None) -> dict:
    if not u:
        est = (len(text) + len(reasoning or "")) // 4
        return {"input_tokens": None, "output_tokens": est, "estimated": True}
    details = u.get("completion_tokens_details") or {}
    return {
        "input_tokens": u.get("prompt_tokens"),
        "output_tokens": u.get("completion_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens"),
        "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
    }


# --------------------------------------------------------------------------------------------
# Anthropic (Claude) via the official SDK


class AnthropicClient(BaseClient):
    def __init__(self, spec: ModelSpec, *, judge: bool = False):
        super().__init__(spec)
        import anthropic

        self._anthropic = anthropic
        key = resolve_secret(spec.api_key)
        kwargs: dict[str, Any] = {"timeout": self.timeout, "max_retries": 4}
        if key:
            kwargs["api_key"] = key
        if spec.base_url:
            kwargs["base_url"] = spec.base_url
        self.client = anthropic.Anthropic(**kwargs)
        self.model_id = spec.model_id or spec.key
        # Server-side refusal fallbacks are only for the judge: for a model under test, another
        # model answering would contaminate its score.
        self.use_fallbacks = judge
        self.fallback_used = False

    def chat(self, messages, *, max_tokens=None, params=None, tools=None, timeout=None) -> ChatResult:
        system, conv = _to_anthropic_messages(messages)
        merged = {**self.spec.params, **(params or {})}
        merged.pop("stream", None)
        # sampling params are rejected (400) on current Claude models; drop OpenAI-isms
        for k in ("temperature", "top_p", "top_k", "reasoning_effort", "chat_template_kwargs", "seed"):
            merged.pop(k, None)
        req: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": max_tokens or self.spec.max_output_tokens or 32000,
            "messages": conv,
            **merged,
        }
        if system:
            req["system"] = system
        if tools:
            req["tools"] = [_openai_tool_to_anthropic(t) for t in tools]
        a = self._anthropic
        t0 = time.time()
        try:
            msg = self._send(req, timeout)
        except a.BadRequestError as e:
            text = str(e)
            if self.use_fallbacks and "fallback" in text.lower():
                self.use_fallbacks = False
                msg = self._send(req, timeout)
            elif re.search(r"prompt is too long|context", text, re.I):
                raise ContextOverflow(text) from e
            else:
                raise ModelError(text) from e
        except a.APIStatusError as e:
            raise ModelError(f"HTTP {e.status_code}: {e.message}") from e
        except a.APIConnectionError as e:
            raise ModelError(f"connection error: {e}") from e
        texts, thinking, calls = [], [], []
        for block in msg.content:
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "thinking" and getattr(block, "thinking", ""):
                thinking.append(block.thinking)
            elif block.type == "tool_use":
                calls.append({"id": block.id, "type": "function",
                              "function": {"name": block.name, "arguments": json.dumps(block.input)}})
        u = msg.usage
        return ChatResult(
            text="".join(texts).strip(),
            reasoning="\n".join(thinking) or None,
            tool_calls=calls,
            finish_reason=msg.stop_reason,
            usage={
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "cache_read_tokens": getattr(u, "cache_read_input_tokens", None),
                "served_by": getattr(msg, "model", None),
            },
            latency=time.time() - t0,
        )

    def _send(self, req: dict, timeout: float | None):
        client = self.client.with_options(timeout=timeout or self.timeout)
        if self.use_fallbacks:
            with client.beta.messages.stream(
                **req, betas=["server-side-fallback-2026-07-01"], extra_body={"fallbacks": "default"}
            ) as stream:
                return stream.get_final_message()
        with client.messages.stream(**req) as stream:
            return stream.get_final_message()


def _to_anthropic_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    system_parts: list[str] = []
    out: list[dict] = []
    for m in messages:
        role = m["role"]
        content = m.get("content")
        if role == "system":
            system_parts.append(content if isinstance(content, str) else _flatten_text(content))
            continue
        if role == "tool":
            out.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": m.get("tool_call_id"), "content": str(content)}]})
            continue
        if isinstance(content, list):
            blocks = []
            for part in content:
                if part.get("type") == "text":
                    blocks.append({"type": "text", "text": part["text"]})
                elif part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    if url.startswith("data:"):
                        header, data = url.split(",", 1)
                        media = header[5:].split(";")[0]
                        blocks.append({"type": "image", "source": {"type": "base64", "media_type": media, "data": data}})
                    else:
                        blocks.append({"type": "image", "source": {"type": "url", "url": url}})
            content = blocks
        if role == "assistant" and m.get("tool_calls"):
            blocks = [{"type": "text", "text": content}] if content else []
            for tc in m["tool_calls"]:
                blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["function"]["name"],
                               "input": json.loads(tc["function"]["arguments"] or "{}")})
            content = blocks
        out.append({"role": role, "content": content})
    return ("\n\n".join(system_parts) or None), out


def _flatten_text(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(p.get("text", "") for p in content or [] if p.get("type") == "text")


def _openai_tool_to_anthropic(tool: dict) -> dict:
    fn = tool.get("function", tool)
    return {"name": fn["name"], "description": fn.get("description", ""),
            "input_schema": fn.get("parameters") or {"type": "object", "properties": {}}}


# --------------------------------------------------------------------------------------------
# opencode-only cloud models


class OpencodeChatClient(BaseClient):
    """Single-turn chat through `opencode run --agent bench-plain` (no tools, minimal prompt)."""

    def chat(self, messages, *, max_tokens=None, params=None, tools=None, timeout=None) -> ChatResult:
        from bench.runners.opencode import opencode_plain_chat

        if tools:
            raise ModelError("tool calls are not supported through the opencode-plain fallback")
        t0 = time.time()
        text, usage = opencode_plain_chat(self.spec, messages, timeout=timeout or self.timeout)
        text, reasoning = split_think(text, None)
        return ChatResult(text=text, reasoning=reasoning, usage=usage, latency=time.time() - t0,
                          finish_reason="stop")


def make_client(spec: ModelSpec, *, base_url: str | None = None, judge: bool = False) -> BaseClient:
    if spec.provider == "anthropic":
        return AnthropicClient(spec, judge=judge)
    if spec.provider == "opencode":
        return OpencodeChatClient(spec)
    if spec.provider == "openai":
        return OpenAICompatClient(spec, base_url=base_url)
    raise ModelError(f"model {spec.key}: provider {spec.provider} has no chat client")


def image_part(path_or_bytes, media_type: str = "image/png") -> dict:
    """OpenAI-style image content part (converted automatically for Anthropic)."""
    data = path_or_bytes if isinstance(path_or_bytes, bytes) else open(path_or_bytes, "rb").read()
    b64 = base64.standard_b64encode(data).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}}
