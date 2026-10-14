from collections import defaultdict

import httpx

from bench.benchmark import Sample
from bench.common import hf_rows, read_jsonl, write_jsonl
from bench.families import MultipleChoiceBenchmark

REPO = "edinburgh-dawg/mmlu-redux-2.0"


class MMLURedux(MultipleChoiceBenchmark):
    letters = "ABCD"

    def _subjects(self):
        sp = httpx.get("https://datasets-server.huggingface.co/splits", params={"dataset": REPO}, timeout=60).json()
        return sorted({s["config"] for s in sp["splits"]})

    def fetch(self):
        rows = []
        for subj in self._subjects():
            for r in hf_rows(REPO, self.data_dir / "raw", config=subj, split="test"):
                rows.append({**r, "subject": subj})
        write_jsonl(self.data_dir / "all.jsonl", rows)
        self.mark_fetched()

    def load(self):
        out, idx = [], defaultdict(int)
        for r in read_jsonl(self.data_dir / "all.jsonl"):
            subj = r["subject"]
            i = idx[subj]
            idx[subj] += 1
            et = r.get("error_type")
            if et == "ok":
                gold = r["answer"]
            elif et == "wrong_groundtruth" and str(r.get("correct_answer") or "").strip().isdigit():
                gold = int(r["correct_answer"])
            elif et == "wrong_groundtruth" and str(r.get("correct_answer") or "").strip().upper() in ("A", "B", "C", "D"):
                gold = "ABCD".index(r["correct_answer"].strip().upper())
            else:
                continue
            if not (0 <= int(gold) < len(r["choices"])):
                continue
            out.append(Sample(id=f"{subj}/{i}", data={"question": r["question"], "choices": r["choices"], "subject": subj},
                              gold="ABCD"[int(gold)], stratum=subj))
        return out


BENCHMARK = MMLURedux
