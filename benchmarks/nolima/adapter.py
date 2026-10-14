import json
import random
import re
from collections import defaultdict

from bench.benchmark import Benchmark, ChatRequest, Grade, Sample
from bench.common import hf_snapshot

REPO = "amodaresi/NoLiMa"


def _fill(s: str, args: list[str], char: str | None = None) -> str:
    for i, a in enumerate(args, 1):
        s = s.replace("{%d}" % i, a)
    return s.replace("{CHAR}", char) if char else s


class NoLiMa(Benchmark):
    def fetch(self):
        hf_snapshot(REPO, self.data_dir, allow=["needlesets/needle_set.json", "haystack/rand_shuffle/*",
                                                 "haystack/rand_shuffle_long/*"])
        self.mark_fetched()

    def load(self):
        needles = json.loads((self.data_dir / "needlesets" / "needle_set.json").read_text())
        lengths = self.spec.options.get("lengths", [8192, 32768])
        depths = int(self.spec.options.get("depths", 11))
        out = []
        for n in needles:
            for tid, t in n["tests"].items():
                for q in ("onehop", "twohop"):
                    if q not in n["questions"]:
                        continue
                    for book in range(1, 6):
                        for L in lengths:
                            for di in range(depths):
                                sid = f"{n['id']}/{tid}/{q}/b{book}/{L}/d{di}"
                                out.append(Sample(id=sid, data={"needle_id": n["id"], "test": tid, "q": q, "book": book,
                                                                "length": L, "depth": di / max(1, depths - 1)},
                                                  stratum=f"{L}/{q}"))
        return out

    def _needle(self, nid):
        if not hasattr(self, "_needles"):
            self._needles = {n["id"]: n for n in json.loads((self.data_dir / "needlesets" / "needle_set.json").read_text())}
        return self._needles[nid]

    def _book(self, idx, long):
        cache = self.__dict__.setdefault("_books", {})
        key = (idx, long)
        if key not in cache:
            folder = "rand_shuffle_long" if long else "rand_shuffle"
            cache[key] = (self.data_dir / "haystack" / folder / f"rand_book_{idx}.txt").read_text()
        return cache[key]

    def _character(self, sample):
        n = self._needle(sample.data["needle_id"])
        return random.Random(sample.id).choice(n["character_set"])

    def build(self, sample, ctx):
        d = sample.data
        n = self._needle(d["needle_id"])
        args = n["tests"][d["test"]]["input_args"]
        char = self._character(sample)
        needle = _fill(n["needle"], args, char)
        question = _fill(n["questions"][d["q"]], args)
        overhead = len(n["task_template"]) + len(question) + len(needle)
        n_chars = max(1000, int(d["length"] * 3.6) - overhead)
        text = self._book(d["book"], d["length"] > 32768)[:n_chars]
        pos = int(len(text) * d["depth"])
        if 0 < pos < len(text):
            m = re.search(r"[.!?]\s", text[pos:])
            pos = pos + m.end() if m else len(text)
        haystack = text[:pos] + (" " if pos else "") + needle + " " + text[pos:]
        prompt = n["task_template"].format(haystack=haystack, question=question)
        msgs = ([{"role": "system", "content": n["system_prompt"]}] if n.get("system_prompt") else []) + \
               [{"role": "user", "content": prompt}]
        return ChatRequest(messages=msgs)

    def grade(self, sample, output, ctx):
        char = self._character(sample)
        ok = bool(re.search(rf"\b{re.escape(char)}\b", output.text or "", re.I))
        return Grade(score=float(ok), extracted=(output.text or "")[-200:],
                     details={"gold": char, "length": sample.data["length"], "q": sample.data["q"], "depth": sample.data["depth"]})

    def oracle(self, sample):
        return self._character(sample)

    def aggregate(self, rows):
        by = defaultdict(list)
        for r in rows:
            d = r["details"] or {}
            by[f"{d.get('length')} {d.get('q')}"].append(r["score"])
        return {"by_length_hops": {k: sum(v) / len(v) for k, v in sorted(by.items())}}


BENCHMARK = NoLiMa
