from bench.benchmark import Sample
from bench.common import download, read_jsonl, write_jsonl
from bench.search_agents import JudgedSearchBenchmark, xor_decrypt

URL = "https://raw.githubusercontent.com/PALIN2018/BrowseComp-ZH/86abe635e7deef89ec00c68ff1c2588f0e2f2099/data/browsecomp-zh-encrypted.xlsx"


class BrowseCompZH(JudgedSearchBenchmark):
    def fetch(self):
        import pandas as pd

        x = download(URL, self.data_dir / "encrypted.xlsx")
        df = pd.read_excel(x)
        rows = []
        for i, r in df.iterrows():
            pw = r["canary"]
            rows.append({"id": str(i), "topic": xor_decrypt(r["Topic"], pw) if isinstance(r["Topic"], str) else "",
                         "question": xor_decrypt(r["Question"], pw), "answer": xor_decrypt(r["Answer"], pw)})
        write_jsonl(self.data_dir / "decrypted.jsonl", rows)
        self.mark_fetched()

    def load(self):
        return [Sample(id=r["id"], data=r, gold=r["answer"], stratum=r["topic"]) for r in read_jsonl(self.data_dir / "decrypted.jsonl")]


BENCHMARK = BrowseCompZH
