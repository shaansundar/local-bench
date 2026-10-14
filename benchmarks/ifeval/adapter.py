import sys

from bench.benchmark import Benchmark, ChatRequest, Grade, Sample
from bench.common import VENDOR, ensure_nltk, hf_rows, vendor_raw_files
from bench.families import if_aggregate

REPO = "google/IFEval"
UPSTREAM = "google-research/google-research"
REVISION = "e6890f85757dd84e27ca6df2dd30651dafad28e0"
FILES = [f"instruction_following_eval/{f}" for f in
         ("instructions.py", "instructions_registry.py", "instructions_util.py", "evaluation_lib.py")]
DEST = VENDOR / "google_ifeval"


def _evaluation_lib():
    ensure_nltk()
    if str(DEST) not in sys.path:
        sys.path.insert(0, str(DEST))
    from instruction_following_eval import evaluation_lib

    return evaluation_lib


def check(evaluation_lib, key, prompt, instruction_ids, kwargs, response) -> dict:
    inp = evaluation_lib.InputExample(key=key, instruction_id_list=instruction_ids, prompt=prompt, kwargs=kwargs)
    strict = evaluation_lib.test_instruction_following_strict(inp, {prompt: response})
    loose = evaluation_lib.test_instruction_following_loose(inp, {prompt: response})
    return {"strict_all": strict.follow_all_instructions, "strict_each": list(strict.follow_instruction_list),
            "loose_all": loose.follow_all_instructions, "loose_each": list(loose.follow_instruction_list),
            "instructions": list(instruction_ids)}


class IFEval(Benchmark):
    def fetch(self):
        vendor_raw_files(UPSTREAM, REVISION, FILES, DEST)
        (DEST / "instruction_following_eval" / "__init__.py").touch()
        hf_rows(REPO, self.data_dir, split="train")
        ensure_nltk()
        self.mark_fetched()

    def load(self):
        out = []
        for r in hf_rows(REPO, self.data_dir, split="train"):
            kwargs = [{k: v for k, v in (kw or {}).items() if v is not None} for kw in r["kwargs"]]
            out.append(Sample(id=str(r["key"]), data={"prompt": r["prompt"], "instruction_id_list": r["instruction_id_list"],
                                                      "kwargs": kwargs}))
        return out

    def build(self, sample, ctx):
        return ChatRequest(messages=[{"role": "user", "content": sample.data["prompt"]}])

    def grade(self, sample, output, ctx):
        lib = _evaluation_lib()
        d = check(lib, int(sample.id), sample.data["prompt"], sample.data["instruction_id_list"],
                  sample.data["kwargs"], output.text or "")
        return Grade(score=float(d["strict_all"]), details=d)

    def aggregate(self, rows):
        return if_aggregate(rows)


BENCHMARK = IFEval
