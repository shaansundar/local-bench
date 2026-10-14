import csv
import zipfile

from bench.benchmark import Benchmark, ChatRequest, Grade, Sample
from bench.common import hf_file

REPO = "ArtificialAnalysis/AA-LCR"
PROMPT = """BEGIN INPUT DOCUMENTS

{documents}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{question}

END QUESTION"""


class AALCR(Benchmark):
    def fetch(self):
        hf_file(REPO, "AA-LCR_Dataset.csv", self.data_dir)
        z = hf_file(REPO, "extracted_text/AA-LCR_extracted-text.zip", self.data_dir)
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                if ".." in name or name.startswith("/"):
                    raise RuntimeError(f"unsafe path in archive: {name}")
            zf.extractall(self.data_dir / "docs")
        self.mark_fetched()

    def _doc_dir(self, category, set_id):
        base = self.data_dir / "docs" / "lcr"
        for cand in (category, category.replace(" ", "_")):
            if (base / cand / set_id).is_dir():
                return base / cand / set_id
        hits = list(base.glob(f"*/{set_id}"))
        if not hits:
            raise FileNotFoundError(f"documents for {category}/{set_id}")
        return hits[0]

    def load(self):
        with open(self.data_dir / "AA-LCR_Dataset.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        return [Sample(id=f"{r['document_set_id']}/{r['question_id']}",
                       data={"question": r["question"], "category": r["document_category"], "set": r["document_set_id"],
                             "files": [x.strip() for x in r["data_source_filenames"].split(";") if x.strip()],
                             "input_tokens": int(float(r["input_tokens"] or 0))},
                       gold=r["answer"], stratum=r["document_category"]) for r in rows]

    def build(self, sample, ctx):
        d = sample.data
        folder = self._doc_dir(d["category"], d["set"])
        docs = []
        for i, fn in enumerate(d["files"], 1):
            text = (folder / fn).read_text(errors="replace")
            docs.append(f"BEGIN DOCUMENT {i}:\n{text}\nEND DOCUMENT {i}")
        return ChatRequest(messages=[{"role": "user", "content": PROMPT.format(documents="\n\n".join(docs), question=d["question"])}])

    def grade(self, sample, output, ctx):
        ok, info = ctx.require_judge().equivalence(sample.data["question"], output.text, str(sample.gold))
        return Grade(score=float(ok), extracted=info.get("extracted"), judge=info)

    def oracle(self, sample):
        return f"The answer is: {sample.gold}"


BENCHMARK = AALCR
