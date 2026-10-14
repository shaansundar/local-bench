import json
from collections import defaultdict

from bench.benchmark import Benchmark, ChatRequest, Grade, Sample
from bench.common import ensure_nltk, hf_rows, load_module, vendor_repo

REPO = "facebook/Multi-IF"
UPSTREAM = "https://github.com/facebookresearch/Multi-IF"
REVISION = "1cdb53ed18499ad729e0766e5d3099dd5344406f"


def _ifeval():
    ensure_nltk()
    root = vendor_repo(UPSTREAM, REVISION, "Multi-IF")
    return load_module("multi_if_ifeval", root / "ifeval.py", sys_path=root)


def _follow(ifeval, response: str, ids: list[str], kwargs: list[dict], loose: bool) -> list[bool]:
    """Port of Multi-IF metrics.gen_acc_strict / gen_acc_loose (without its scipy/pandas imports)."""
    if loose:
        r = response.split("\n")
        variants = [response, "\n".join(r[1:]).strip(), "\n".join(r[:-1]).strip(), "\n".join(r[1:-1]).strip()]
        variants += [v.replace("*", "") for v in variants]
    else:
        variants = [response]
    out = []
    for i, iid in enumerate(ids):
        inst = ifeval.INSTRUCTION_DICT[iid](iid)
        inst.build_description(**(kwargs[i] or {}))
        out.append(any(v.strip() and inst.check_following(v) for v in variants))
    return out


class MultiIF(Benchmark):
    def fetch(self):
        vendor_repo(UPSTREAM, REVISION, "Multi-IF")
        hf_rows(REPO, self.data_dir, split="train")
        _ifeval()
        self.mark_fetched()

    def load(self):
        out = []
        for r in hf_rows(REPO, self.data_dir, split="train"):
            turns = []
            for t in (1, 2, 3):
                p = r.get(f"turn_{t}_prompt")
                if not p:
                    break
                ids = json.loads(r[f"turn_{t}_instruction_id_list"])
                kws = [json.loads(k) if isinstance(k, str) else (k or {}) for k in json.loads(r[f"turn_{t}_kwargs"])]
                kws = [{k: v for k, v in kw.items() if v is not None} for kw in kws]
                turns.append({"content": json.loads(p)["content"], "ids": ids, "kwargs": kws})
            out.append(Sample(id=r["key"], data={"turns": turns, "language": r["language"]}, stratum=r["language"]))
        return out

    def build(self, sample, ctx):
        t = sample.data["turns"]
        return ChatRequest(messages=[{"role": "user", "content": t[0]["content"]}], followups=[x["content"] for x in t[1:]])

    def grade(self, sample, output, ctx):
        ifeval = _ifeval()
        texts = output.extra.get("turn_texts") or [output.text]
        per_turn = []
        for i, turn in enumerate(sample.data["turns"]):
            resp = texts[i] if i < len(texts) else ""
            s = _follow(ifeval, resp, turn["ids"], turn["kwargs"], loose=False)
            l = _follow(ifeval, resp, turn["ids"], turn["kwargs"], loose=True)
            per_turn.append({"prompt_strict": float(all(s)), "prompt_loose": float(all(l)),
                             "inst_strict": sum(s) / len(s), "inst_loose": sum(l) / len(l)})
        turn_scores = [sum(t.values()) / 4 for t in per_turn]
        return Grade(score=sum(turn_scores) / len(turn_scores),
                     details={"turns": per_turn, "turn_scores": turn_scores, "language": sample.data["language"]})

    def aggregate(self, rows):
        by_turn, by_lang = defaultdict(list), defaultdict(list)
        for r in rows:
            d = r["details"] or {}
            for i, s in enumerate(d.get("turn_scores", [])):
                by_turn[f"turn {i + 1}"].append(s)
            by_lang[d.get("language")].append(r["score"])
        avg = lambda v: sum(v) / len(v)
        return {"by_turn": {k: avg(v) for k, v in sorted(by_turn.items())},
                "by_language": {k: avg(v) for k, v in sorted(by_lang.items()) if k}}


BENCHMARK = MultiIF
