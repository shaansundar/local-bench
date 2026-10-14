from bench.benchmark import Benchmark, ChatRequest, Grade, Sample
from bench.common import ensure_nltk, hf_rows, load_module, vendor_repo
from bench.families import if_aggregate

REPO = "allenai/IFBench_test"
UPSTREAM = "https://github.com/allenai/IFBench"
REVISION = "1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d"


def _lib():
    ensure_nltk()
    root = vendor_repo(UPSTREAM, REVISION, "IFBench")
    return load_module("ifbench_evaluation_lib", root / "evaluation_lib.py", sys_path=root)


class IFBench(Benchmark):
    def fetch(self):
        vendor_repo(UPSTREAM, REVISION, "IFBench")
        hf_rows(REPO, self.data_dir, split="train")
        _lib()
        self.mark_fetched()

    def load(self):
        out = []
        for r in hf_rows(REPO, self.data_dir, split="train"):
            kwargs = [{k: v for k, v in (kw or {}).items() if v is not None} for kw in r["kwargs"]]
            out.append(Sample(id=str(r["key"]), data={"prompt": r["prompt"], "ids": r["instruction_id_list"], "kwargs": kwargs}))
        return out

    def build(self, sample, ctx):
        return ChatRequest(messages=[{"role": "user", "content": sample.data["prompt"]}])

    def grade(self, sample, output, ctx):
        lib = _lib()
        d = sample.data
        inp = lib.InputExample(key=sample.id, instruction_id_list=d["ids"], prompt=d["prompt"], kwargs=d["kwargs"])
        resp = {d["prompt"]: output.text or ""}
        strict = lib.test_instruction_following_strict(inp, resp)
        loose = lib.test_instruction_following_loose(inp, resp)
        det = {"strict_all": strict.follow_all_instructions, "strict_each": list(strict.follow_instruction_list),
               "loose_all": loose.follow_all_instructions, "loose_each": list(loose.follow_instruction_list),
               "instructions": list(d["ids"])}
        return Grade(score=float(det["loose_all"]), details=det)

    def aggregate(self, rows):
        return if_aggregate(rows)


BENCHMARK = IFBench
