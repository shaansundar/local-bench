"""The fixed cloud judge. Every judged score records `judge_id` = <model>@<prompt version>, so
the dashboard never mixes scores from different judges or prompt revisions silently.

Judge calls are cached on disk (results/cache/judge.sqlite), keyed by model + full prompt, so
regrading is free and reproducible.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from dataclasses import dataclass

from bench.benchmark import JudgeUnavailable
from bench.config import load_config, load_models
from bench.models.client import make_client

# Bump a version whenever its prompt text changes.
PROMPT_VERSIONS = {"equivalence": "v1", "rubric": "v1", "pairwise": "v1", "assertions": "v1"}

EQUIVALENCE_PROMPT = """Judge whether the following [response] to [question] is correct, based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available."""


@dataclass
class JudgeResult:
    text: str
    parsed: dict
    judge_id: str
    cached: bool = False


class Judge:
    def __init__(self):
        cfg = load_config()
        self.cfg = cfg.judge
        self.model_key = cfg.judge.model
        self.spec = load_models().get(self.model_key) if self.model_key else None
        self.available = self.spec is not None
        self._client = None
        self._sem = threading.Semaphore(max(1, cfg.judge.max_concurrency))
        cache = cfg.results_dir / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(cache / "judge.sqlite", check_same_thread=False, isolation_level=None)
        self._db.execute("CREATE TABLE IF NOT EXISTS j (k TEXT PRIMARY KEY, model TEXT, text TEXT)")
        self._lock = threading.Lock()

    def judge_id(self, kind: str) -> str:
        return f"{self.model_key}@{kind}-{PROMPT_VERSIONS[kind]}"

    @property
    def client(self):
        if self._client is None:
            if not self.available:
                raise JudgeUnavailable("judge model not configured (config.yaml judge.model)")
            self._client = make_client(self.spec, judge=True)
        return self._client

    def ask(self, prompt: str | list, kind: str, *, max_tokens: int | None = None, system: str | None = None) -> JudgeResult:
        """prompt: a string, or OpenAI-style content parts (text + images)."""
        if not self.available:
            raise JudgeUnavailable("judge model not configured (config.yaml judge.model)")
        key = hashlib.sha256(json.dumps([self.model_key, system, prompt], sort_keys=True).encode()).hexdigest()
        with self._lock:
            row = self._db.execute("SELECT text FROM j WHERE k=?", (key,)).fetchone()
        if row:
            return JudgeResult(row[0], {}, self.judge_id(kind), cached=True)
        with self._sem:
            msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
            res = self.client.chat(msgs, max_tokens=max_tokens or self.cfg.max_tokens)
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO j VALUES (?,?,?)", (key, self.model_key, res.text))
        return JudgeResult(res.text, {}, self.judge_id(kind))

    # ---- common judgements ---------------------------------------------------------------
    def equivalence(self, question: str, response: str, correct_answer: str) -> tuple[bool, dict]:
        """HLE / BrowseComp-style grading: is the response's final answer equivalent to gold?"""
        r = self.ask(EQUIVALENCE_PROMPT.format(question=question, response=response or "(empty)",
                                               correct_answer=correct_answer), "equivalence")
        m = re.search(r"correct:\s*\**\s*(yes|no)", r.text, re.I)
        extracted = re.search(r"extracted_final_answer:\s*(.+)", r.text)
        ok = bool(m and m.group(1).lower() == "yes")
        return ok, {"judge_id": r.judge_id, "verdict": "yes" if ok else "no",
                    "extracted": extracted.group(1).strip() if extracted else None, "raw": r.text[-2000:]}

    def score_json(self, prompt: str | list, kind: str, system: str | None = None) -> tuple[dict, dict]:
        """Ask for a JSON object (last {...} in the reply is parsed)."""
        r = self.ask(prompt, kind, system=system)
        obj = parse_last_json(r.text)
        return obj or {}, {"judge_id": r.judge_id, "raw": r.text[-3000:], "parsed": obj is not None}


def parse_last_json(text: str) -> dict | None:
    fence = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = fence[::-1] if fence else []
    # fall back to the last balanced {...} span
    depth, start, spans = 0, None, []
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                spans.append(text[start : i + 1])
    candidates += spans[::-1]
    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    return None
