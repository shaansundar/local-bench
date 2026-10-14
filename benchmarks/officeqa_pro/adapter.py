import csv
from pathlib import Path

from bench.benchmark import AgentTask, Benchmark, Grade, Sample
from bench.common import VENDOR, hf_snapshot, load_module, vendor_raw_files

REPO = "databricks/officeqa"
UPSTREAM = "databricks/officeqa"
REVISION = "7b9a3c154ef9fb40215bb67934afc43e6799de16"
PROMPT = """Answer the question below using the U.S. Treasury Bulletin corpus at:
{corpus}
(one plain-text file per bulletin, named like treasury_bulletin_YYYY_MM.txt; tables are preserved as text).

Use shell tools (ls, grep, rg, sed) and Python to locate and compute the answer; the corpus is read-only, work in your
current directory. Be precise with units, signs and rounding exactly as the question asks.

Question: {question}

End your reply with the final answer wrapped exactly like this: <FINAL_ANSWER>your answer</FINAL_ANSWER>"""


def _reward():
    vendor_raw_files(UPSTREAM, REVISION, ["reward.py"], VENDOR / "officeqa")
    return load_module("officeqa_reward", VENDOR / "officeqa" / "reward.py")


class OfficeQAPro(Benchmark):
    def fetch(self):
        _reward()
        hf_snapshot(REPO, self.data_dir, allow=["officeqa_pro.csv", "treasury_bulletins_parsed/transformed/*.txt"])
        self.mark_fetched()

    @property
    def corpus(self) -> Path:
        return self.data_dir / "treasury_bulletins_parsed" / "transformed"

    def load(self):
        with open(self.data_dir / "officeqa_pro.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        return [Sample(id=r["uid"], data={"question": r["question"], "source_files": r.get("source_files")}, gold=r["answer"])
                for r in rows]

    def build(self, sample, ctx):
        return AgentTask(prompt=PROMPT.format(corpus=self.corpus, question=sample.data["question"]), read_dirs=[self.corpus])

    def grade(self, sample, output, ctx):
        text = output.text or output.extra.get("all_text", "")
        tol = float(self.spec.options.get("tolerance", 0.0))
        reward = _reward()
        return Grade(score=float(reward.score_answer(str(sample.gold), text, tol)),
                     extracted=reward.extract_final_answer(text)[:300])

    def oracle(self, sample):
        return f"<FINAL_ANSWER>{sample.gold}</FINAL_ANSWER>"


BENCHMARK = OfficeQAPro
