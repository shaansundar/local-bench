import json
from functools import lru_cache

from bench.benchmark import Benchmark, ChatRequest, Grade, Sample
from bench.common import gdrive_download, safe_extract
from bench.sandbox.exec import extract_code, sql_match, sqlite_schema

DRIVE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
PROMPT = """{schema}

-- Using valid SQLite, answer the following question for the tables provided above.
-- Question: {question}

Think step by step, then give the final SQLite query in a single ```sql code block."""


class Spider(Benchmark):
    def fetch(self):
        z = self.data_dir / "spider_data.zip"
        if not z.exists():
            try:
                gdrive_download(DRIVE_ID, z)
            except RuntimeError as e:
                raise RuntimeError(
                    f"{e}. Google Drive often rate-limits this file: retry later, or download spider_data.zip "
                    f"in a browser from https://drive.google.com/file/d/{DRIVE_ID} and save it as {z}") from e
        safe_extract(z, self.data_dir / "x")
        z.unlink()
        self.mark_fetched()

    def _base(self):
        hits = [p.parent for p in (self.data_dir / "x").rglob("dev.json") if (p.parent / "database").is_dir()]
        if not hits:
            raise FileNotFoundError("spider dev.json/database not found")
        return hits[0]

    def _db(self, db_id):
        return self._base() / "database" / db_id / f"{db_id}.sqlite"

    @lru_cache(maxsize=64)
    def _schema(self, db_id):
        return sqlite_schema(self._db(db_id), int(self.spec.options.get("sample_rows", 3)))

    def load(self):
        rows = json.loads((self._base() / "dev.json").read_text())
        return [Sample(id=str(i), data={"db_id": r["db_id"], "question": r["question"]}, gold=r["query"], stratum=r["db_id"])
                for i, r in enumerate(rows)]

    def build(self, sample, ctx):
        return ChatRequest(messages=[{"role": "user", "content": PROMPT.format(
            schema=self._schema(sample.data["db_id"]), question=sample.data["question"])}])

    def grade(self, sample, output, ctx):
        sql = (extract_code(output.text, "sql") or "").strip()
        if not sql:
            return Grade(score=0.0, details={"error": "no sql block"})
        r = sql_match(self._db(sample.data["db_id"]), sql, sample.gold, mode="ordered_multiset",
                      timeout=float(self.spec.options.get("query_timeout_s", 30)))
        return Grade(score=float(r.get("match", False)), extracted=sql, details=r)

    def oracle(self, sample):
        return f"```sql\n{sample.gold}\n```"


BENCHMARK = Spider
