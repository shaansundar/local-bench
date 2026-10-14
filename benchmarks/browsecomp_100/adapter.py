import random

from bench.benchmark import Sample
from bench.common import download, read_jsonl, write_jsonl
from bench.search_agents import JudgedSearchBenchmark, xor_decrypt

URL = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"


class BrowseComp100(JudgedSearchBenchmark):
    def fetch(self):
        import pandas as pd

        f = download(URL, self.data_dir / "browse_comp_test_set.csv")
        df = pd.read_csv(f)
        rows = [{"id": str(i), "question": xor_decrypt(r["problem"], r["canary"]), "answer": xor_decrypt(r["answer"], r["canary"]),
                 "topic": r.get("problem_topic")} for i, r in df.iterrows()]
        rng = random.Random(int(self.spec.options.get("subset_seed", 2025)))
        keep = rng.sample(rows, int(self.spec.options.get("subset_size", 100)))
        write_jsonl(self.data_dir / "subset.jsonl", keep)
        self.mark_fetched()

    def load(self):
        return [Sample(id=r["id"], data=r, gold=r["answer"], stratum=r.get("topic")) for r in read_jsonl(self.data_dir / "subset.jsonl")]


BENCHMARK = BrowseComp100
