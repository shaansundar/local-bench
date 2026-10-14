import json
import re

from bench.benchmark import AgentTask, Benchmark, Grade, Sample
from bench.common import download
from bench.search_agents import SEARCH_GUIDANCE, gaia_score

URL = "https://raw.githubusercontent.com/RUC-NLPIR/WebThinker/db387eb3261de9b5e2db7d2d7fb20af2aceb7882/data/GAIA/dev.json"
SYSTEM = ("You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with "
          "the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words "
          "as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma "
          "to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked "
          "for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text "
          "unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of "
          "whether the element to be put in the list is a number or a string.")


class GAIA103(Benchmark):
    def fetch(self):
        download(URL, self.data_dir / "dev.json")
        self.mark_fetched()

    def load(self):
        rows = json.loads((self.data_dir / "dev.json").read_text())
        return [Sample(id=str(r.get("task_id") or r["id"]), data={"question": r["Question"], "level": r.get("Level")},
                       gold=str(r["answer"]), stratum=str(r.get("Level"))) for r in rows]

    def build(self, sample, ctx):
        return AgentTask(prompt=f"{SYSTEM}\n\nQuestion: {sample.data['question']}", network=True, agent_prompt=SEARCH_GUIDANCE)

    def grade(self, sample, output, ctx):
        text = output.text or output.extra.get("all_text", "")
        m = re.findall(r"FINAL ANSWER\s*:\s*(.+)", text, re.I)
        ans = m[-1].strip().strip("*").strip() if m else ""
        return Grade(score=float(gaia_score(ans, sample.gold)), extracted=ans, details={"level": sample.data["level"]})

    def oracle(self, sample):
        return f"FINAL ANSWER: {sample.gold}"


BENCHMARK = GAIA103
