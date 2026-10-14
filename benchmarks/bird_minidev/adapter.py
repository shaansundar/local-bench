import json
from collections import defaultdict
from functools import lru_cache

from bench.benchmark import Benchmark, ChatRequest, Grade, Sample
from bench.common import download, safe_extract
from bench.sandbox.exec import extract_code, sql_match, sqlite_schema

URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip"
PROMPT = """{schema}

-- External Knowledge: {evidence}
-- Using valid SQLite and understanding External Knowledge, answer the following question for the tables provided above.
-- Question: {question}

Think step by step, then give the final SQLite query in a single ```sql code block."""


class BirdMiniDev(Benchmark):
    def fetch(self):
        z = download(URL, self.data_dir / "minidev.zip", timeout=3600)
        safe_extract(z, self.data_dir / "x")
        z.unlink()
        self.mark_fetched()

    def _root(self):
        hits = list((self.data_dir / "x").rglob("mini_dev_sqlite.json"))
        if not hits:
            raise FileNotFoundError("mini_dev_sqlite.json not found in the Mini-Dev package")
        return hits[0].parent

    def _db(self, db_id):
        hits = list((self.data_dir / "x").rglob(f"dev_databases/{db_id}/{db_id}.sqlite"))
        if not hits:
            raise FileNotFoundError(f"database {db_id}")
        return hits[0]

    @lru_cache(maxsize=64)
    def _schema(self, db_id):
        return sqlite_schema(self._db(db_id), int(self.spec.options.get("sample_rows", 3)))

    def load(self):
        rows = json.loads((self._root() / "mini_dev_sqlite.json").read_text())
        return [Sample(id=str(r["question_id"]), data={"db_id": r["db_id"], "question": r["question"],
                                                        "evidence": r.get("evidence") or "", "difficulty": r.get("difficulty")},
                       gold=r["SQL"], stratum=r.get("difficulty")) for r in rows]

    def build(self, sample, ctx):
        d = sample.data
        return ChatRequest(messages=[{"role": "user", "content": PROMPT.format(
            schema=self._schema(d["db_id"]), evidence=d["evidence"], question=d["question"])}])

    def grade(self, sample, output, ctx):
        sql = (extract_code(output.text, "sql") or "").strip()
        if not sql:
            return Grade(score=0.0, details={"error": "no sql block", "difficulty": sample.data["difficulty"]})
        r = sql_match(self._db(sample.data["db_id"]), sql, sample.gold, mode="set",
                      timeout=float(self.spec.options.get("query_timeout_s", 30)))
        return Grade(score=float(r.get("match", False)), extracted=sql, details={**r, "difficulty": sample.data["difficulty"]})

    def oracle(self, sample):
        return f"```sql\n{sample.gold}\n```"

    def aggregate(self, rows):
        by = defaultdict(list)
        for r in rows:
            by[(r["details"] or {}).get("difficulty")].append(r["score"])
        return {"by_difficulty": {k: sum(v) / len(v) for k, v in by.items() if k}}


BENCHMARK = BirdMiniDev
