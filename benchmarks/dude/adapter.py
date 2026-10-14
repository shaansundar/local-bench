import ast
import io
import json
import tarfile
from collections import defaultdict

import httpx

from bench.benchmark import Benchmark, ChatRequest, Grade, Sample
from bench.common import anls, context_budget, download, read_jsonl, short_answer, truncate_middle, write_jsonl
from bench.runners.chat import resolve_max_tokens

ANNOTATIONS = "https://zenodo.org/record/7763635/files/2023-03-23_DUDE_gt_test_PUBLIC.json?download=1"
BINARIES = "https://huggingface.co/datasets/jordyvl/DUDE_loader/resolve/main/data/DUDE_train-val-test_binaries.tar.gz"
UNANSWERABLE = {"unanswerable", "not answerable", "none", "n/a", "cannot be answered", "no answer"}


class _HTTPStream(io.RawIOBase):
    def __init__(self, url):
        self._client = httpx.Client(follow_redirects=True, timeout=600)
        self._resp = self._client.send(self._client.build_request("GET", url), stream=True)
        self._resp.raise_for_status()
        self._it = self._resp.iter_raw(1 << 20)
        self._buf = b""

    def readable(self):
        return True

    def readinto(self, b):
        while not self._buf:
            try:
                self._buf = next(self._it)
            except StopIteration:
                return 0
        n = min(len(b), len(self._buf))
        b[:n] = self._buf[:n]
        self._buf = self._buf[n:]
        return n

    def close(self):
        self._resp.close()
        self._client.close()
        super().close()


def _parse_list(v):
    if isinstance(v, list):
        return v
    try:
        return ast.literal_eval(v) if v else []
    except (ValueError, SyntaxError):
        return [v]


class DUDE(Benchmark):
    def fetch(self):
        gt = download(ANNOTATIONS, self.data_dir / "gt.json", timeout=1200)
        data = [x for x in json.loads(gt.read_text())["data"] if x.get("data_split") == "val"]
        need = {x["docId"] for x in data}
        ocr_dir = self.data_dir / "ocr"
        ocr_dir.mkdir(exist_ok=True)
        have = {p.stem for p in ocr_dir.glob("*.json")}
        if not need <= have:
            # Stream the archive; Azure OCR comes first, so stop once we leave that folder.
            stream = _HTTPStream(BINARIES)
            try:
                with tarfile.open(fileobj=io.BufferedReader(stream, 1 << 20), mode="r|gz") as tf:
                    seen_azure = False
                    for m in tf:
                        if "/OCR/Azure/" in m.name:
                            seen_azure = True
                            base = m.name.rsplit("/", 1)[-1]
                            if base.endswith("_original.json") and m.isfile():
                                doc = base[: -len("_original.json")]
                                if doc in need and doc not in have:
                                    (ocr_dir / f"{doc}.json").write_bytes(tf.extractfile(m).read())
                                    have.add(doc)
                        elif seen_azure or need <= have:
                            break
            finally:
                stream.close()
        missing = need - have
        rows = [{"id": x["questionId"], "question": x["question"], "answers": _parse_list(x["answers"]),
                 "variants": _parse_list(x.get("answers_variants")), "type": x["answer_type"], "doc": x["docId"]}
                for x in data if x["docId"] not in missing]
        write_jsonl(self.data_dir / "meta.jsonl", rows)
        self.mark_fetched()

    def load(self):
        return [Sample(id=r["id"], data=r, gold=r["answers"], stratum=r["type"]) for r in read_jsonl(self.data_dir / "meta.jsonl")]

    def _doc_text(self, doc):
        d = json.loads((self.data_dir / "ocr" / f"{doc}.json").read_text())
        pages = (d.get("analyzeResult") or d).get("readResults") or []
        return "\n\n".join(f"<<<page {p.get('page', i + 1)}>>>\n" + "\n".join(l.get("text", "") for l in p.get("lines") or [])
                           for i, p in enumerate(pages))

    def build(self, sample, ctx):
        q = sample.data["question"]
        budget = context_budget(ctx, resolve_max_tokens(None, self, ctx)) - 400
        text = truncate_middle(self._doc_text(sample.data["doc"]), budget)
        prompt = (f"Document (OCR text, page by page):\n{text}\n\nQuestion: {q}\n\n"
                  "Answer concisely with a word or short phrase, taken from the document where possible. If the answer "
                  "is a list, separate the items with ' | '. If the question cannot be answered from the document, "
                  "answer 'unanswerable'. End with a line 'Answer: <answer>'.")
        return ChatRequest(messages=[{"role": "user", "content": prompt}])

    def grade(self, sample, output, ctx):
        d = sample.data
        pred = short_answer(output.text)
        golds = [str(a) for a in d["answers"]] + [str(v) for v in d["variants"] if isinstance(v, str)]
        if d["type"] == "not-answerable" or not d["answers"]:
            score = float(pred.lower().strip(" .") in UNANSWERABLE)
        elif d["type"].startswith("list"):
            items = [p.strip() for p in (pred.split("|") if "|" in pred else pred.split(",")) if p.strip()]
            gold_items = [str(a) for a in d["answers"]]
            best = sum(max((anls(p, [g]) for p in items), default=0.0) for g in gold_items)
            score = best / max(len(gold_items), len(items), 1)
        else:
            score = anls(pred, golds)
        return Grade(score=score, extracted=pred, details={"type": d["type"]})

    def oracle(self, sample):
        d = sample.data
        if d["type"] == "not-answerable" or not d["answers"]:
            return "Answer: unanswerable"
        return "Answer: " + " | ".join(str(a) for a in d["answers"])

    def aggregate(self, rows):
        by = defaultdict(list)
        for r in rows:
            by[(r["details"] or {}).get("type")].append(r["score"])
        return {"by_type": {k: sum(v) / len(v) for k, v in by.items() if k}}


BENCHMARK = DUDE
