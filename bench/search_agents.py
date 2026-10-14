"""Shared pieces for search-agent benchmarks (GAIA text-103, BrowseComp, BrowseComp-ZH).

The agent runs in opencode with the suite's keyless search/fetch MCP tools (bench_web_search,
bench_fetch_url); opencode's own webfetch/websearch tools are disabled, so all web access is
cached, logged and identical across models.
"""

from __future__ import annotations

import base64
import hashlib
import re
import string

from bench.benchmark import AgentTask, Benchmark, Grade

SEARCH_GUIDANCE = ("You can search the web with the `bench_web_search` tool and read pages with the `bench_fetch_url` "
                   "tool. Search iteratively, verify facts from sources, and do not stop until you can answer.")

BROWSECOMP_TEMPLATE = """{question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your final answer}}"""


def xor_decrypt(ciphertext_b64: str, password: str) -> str:
    """simple-evals / BrowseComp-ZH scheme: XOR with SHA-256(password) repeated."""
    data = base64.b64decode(ciphertext_b64)
    key = hashlib.sha256(password.encode()).digest()
    key = key * (len(data) // len(key)) + key[: len(data) % len(key)]
    return bytes(a ^ b for a, b in zip(data, key)).decode("utf-8")


# ---- GAIA official quasi-exact-match scorer -----------------------------------------------------
def _num(s: str) -> float:
    for ch in ("$", "%", ","):
        s = s.replace(ch, "")
    try:
        return float(s)
    except ValueError:
        return float("inf")


def _is_float(x) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False


def _norm(s: str, remove_punct: bool = True) -> str:
    s = re.sub(r"\s", "", s).lower()
    return s.translate(str.maketrans("", "", string.punctuation)) if remove_punct else s


def gaia_score(model_answer: str, ground_truth: str) -> bool:
    if model_answer is None:
        return False
    if _is_float(ground_truth):
        return _num(model_answer) == float(ground_truth)
    if any(c in ground_truth for c in (",", ";")):
        gt = re.split(r"[,;]", ground_truth)
        ma = re.split(r"[,;]", model_answer)
        if len(gt) != len(ma):
            return False
        return all((_num(m) == float(g)) if _is_float(g) else (_norm(m, False) == _norm(g, False)) for m, g in zip(ma, gt))
    return _norm(model_answer) == _norm(ground_truth)


class JudgedSearchBenchmark(Benchmark):
    """Question -> opencode search agent -> judge equivalence against the gold answer."""

    template = BROWSECOMP_TEMPLATE

    def build(self, sample, ctx):
        return AgentTask(prompt=self.template.format(question=sample.data["question"]), network=True,
                         agent_prompt=SEARCH_GUIDANCE)

    def grade(self, sample, output, ctx):
        response = output.text or output.extra.get("all_text", "")
        ok, info = ctx.require_judge().equivalence(sample.data["question"], response, str(sample.gold))
        return Grade(score=float(ok), extracted=info.get("extracted"), judge=info)

    def oracle(self, sample):
        return f"Explanation: omitted\nExact Answer: {sample.gold}\nConfidence: 100%"
