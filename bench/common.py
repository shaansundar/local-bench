"""Shared helpers for benchmark adapters: downloads, vendoring, answer extraction, math grading."""

from __future__ import annotations

import importlib.util
import json
import os
import pickle
import sys
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

from bench.config import ROOT

VENDOR = ROOT / "vendor"


# ---- downloads ----------------------------------------------------------------------------------
def hf_snapshot(repo_id: str, dest: Path, *, allow: list[str] | None = None, revision: str | None = None,
                repo_type: str = "dataset") -> Path:
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    return Path(snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=dest, allow_patterns=allow,
                                  revision=revision))


def hf_file(repo_id: str, filename: str, dest: Path, *, revision: str | None = None,
            repo_type: str = "dataset") -> Path:
    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type, local_dir=dest,
                                revision=revision))


def hf_rows(repo_id: str, dest: Path, *, config: str | None = None, split: str = "test",
            revision: str | None = None) -> list[dict]:
    """Load a parquet-backed HF dataset split and cache it as JSONL (rows are plain dicts)."""
    out = dest / f"{(config or 'default').replace('/', '_')}__{split}.jsonl"
    if out.exists():
        return read_jsonl(out)
    from datasets import load_dataset

    ds = load_dataset(repo_id, config, split=split, revision=revision, cache_dir=str(dest / "hf_cache"))
    rows = [dict(r) for r in ds]
    write_jsonl(out, rows)
    return rows


def download(url: str, dest: Path, *, timeout: float = 600) -> Path:
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    tmp.rename(dest)
    return dest


def gdrive_download(file_id: str, dest: Path) -> Path:
    """Download a public Google Drive file (handles the large-file confirmation page)."""
    import httpx

    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    with httpx.stream("GET", url, follow_redirects=True, timeout=1200) as r:
        r.raise_for_status()
        if "text/html" in r.headers.get("content-type", ""):
            raise RuntimeError(f"Google Drive refused the download of {file_id} (quota or permissions)")
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    tmp.rename(dest)
    return dest


def vendor_repo(url: str, revision: str, name: str | None = None, *, sparse: list[str] | None = None) -> Path:
    """Clone an upstream repo at a pinned commit into vendor/<name> (idempotent)."""
    name = name or url.rstrip("/").split("/")[-1].removesuffix(".git")
    dest = VENDOR / name
    marker = dest / ".bench_revision"
    if marker.exists() and marker.read_text().strip() == revision:
        return dest
    VENDOR.mkdir(parents=True, exist_ok=True)
    if not (dest / ".git").exists():
        args = ["git", "clone", "--filter=blob:none", "--no-checkout", url, str(dest)]
        subprocess.run(args, check=True, capture_output=True)
        if sparse:
            subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", "--no-cone", *sparse], check=True,
                           capture_output=True)
    subprocess.run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", revision], check=False,
                   capture_output=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "--force", revision], check=True, capture_output=True)
    marker.write_text(revision)
    return dest


# ---- files ----------------------------------------------------------------------------------------
def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


class _NoGlobalsUnpickler(pickle.Unpickler):
    """Unpickle plain data only (str/bytes/numbers/containers). Refuses any class or function, so a
    tampered dataset cannot execute code."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def safe_unpickle(data: bytes) -> Any:
    import io

    return _NoGlobalsUnpickler(io.BytesIO(data)).load()


# ---- answer extraction ---------------------------------------------------------------------------
def extract_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} (balanced braces)."""
    if not text:
        return None
    idx = max(text.rfind("\\boxed{"), text.rfind("\\fbox{"))
    if idx < 0:
        return None
    i = text.index("{", idx) + 1
    depth = 1
    j = i
    while j < len(text) and depth:
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
        j += 1
    return text[i : j - 1].strip() if depth == 0 else None


def extract_mc_letter(text: str, letters: str = "ABCD") -> str | None:
    """Multiple-choice letter: 'Answer: X', 'the answer is (X)', \\boxed{X}, else the last bare letter."""
    if not text:
        return None
    L = re.escape(letters)
    pats = [
        rf"(?i)answer\s*(?:is)?\s*[:：]?\s*\**\s*\(?\s*([{L}])\s*\)?\**(?![A-Za-z])",
        rf"(?i)the (?:correct |final )?answer is\s*\**\s*\(?\s*([{L}])\s*\)?(?![A-Za-z])",
        rf"\\boxed\{{\s*(?:\\text\{{)?\s*([{L}])\s*\}}?",
        rf"(?i)option\s*\(?([{L}])\)?\s*is correct",
    ]
    for p in pats:
        hits = re.findall(p, text)
        if hits:
            return hits[-1].upper()
    tail = text[-300:]
    hits = re.findall(rf"(?<![A-Za-z])\(?([{L}])\)?(?![A-Za-z])", tail)
    return hits[-1].upper() if hits else None


# ---- math grading ----------------------------------------------------------------------------------
def math_equal(gold: str, response: str) -> tuple[bool, str | None]:
    """Gold vs model response using HF math-verify (LaTeX-aware, falls back to string match).
    Timeouts are disabled because math-verify's signal.alarm() only works on the main thread and
    grading runs on a thread pool."""
    import logging

    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

    logging.getLogger("math_verify").setLevel(logging.ERROR)

    if not response:
        return False, None
    g = parse(f"${gold}$", extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
              parsing_timeout=None)
    boxed = extract_boxed(response)
    target = f"\\boxed{{{boxed}}}" if boxed is not None else response[-2000:]
    p = parse(target, extraction_config=[LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()],
              parsing_timeout=None)
    extracted = boxed if boxed is not None else (str(p[-1]) if p else None)
    if not g or not p:
        return (boxed is not None and _norm(boxed) == _norm(gold)), extracted
    try:
        return bool(verify(g, p, timeout_seconds=None)), extracted
    except Exception:
        return (_norm(str(extracted)) == _norm(gold)), extracted


def _norm(s: str) -> str:
    return re.sub(r"\s+|\\!|\\,|\$", "", str(s)).strip().lower()


MATH_PROMPT = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def mc_prompt(question: str, options: list[str], letters: str | None = None) -> str:
    letters = letters or "ABCDEFGHIJKLMNOP"[: len(options)]
    lines = "\n".join(f"{letters[i]}) {o}" for i, o in enumerate(options))
    return (
        "Answer the following multiple choice question. The last line of your response should be of the "
        f"following format: 'Answer: $LETTER' (without quotes) where LETTER is one of {letters[:len(options)]}. "
        f"Think step by step before answering.\n\n{question}\n\n{lines}"
    )


def vendor_raw_files(repo: str, revision: str, paths: list[str], dest: Path) -> Path:
    """Download individual files of a GitHub repo at a pinned commit (for huge monorepos)."""
    marker = dest / ".bench_revision"
    if marker.exists() and marker.read_text().strip() == revision:
        return dest
    for p in paths:
        download(f"https://raw.githubusercontent.com/{repo}/{revision}/{p}", dest / p)
    marker.write_text(revision)
    return dest


def ensure_nltk() -> None:
    """NLTK data (sentence tokenizer) used by the IFEval-family checkers, kept under data/."""
    import nltk

    from bench.config import load_config

    path = load_config().data_dir / "_nltk"
    path.mkdir(parents=True, exist_ok=True)
    if str(path) not in nltk.data.path:
        nltk.data.path.insert(0, str(path))
    os.environ["NLTK_DATA"] = str(path)
    for pkg, probe in (("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab"),
                       ("stopwords", "corpora/stopwords"),
                       ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng")):
        try:
            nltk.data.find(probe)
        except LookupError:
            nltk.download(pkg, download_dir=str(path), quiet=True)


def load_module(name: str, path: Path, *, sys_path: Path | None = None):
    """Import a vendored file under a unique module name (avoids clashes like `evaluation_lib`)."""
    if name in sys.modules:
        return sys.modules[name]
    if sys_path and str(sys_path) not in sys.path:
        sys.path.insert(0, str(sys_path))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---- long context ----------------------------------------------------------------------------------
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")


def est_tokens(text: str) -> int:
    """Tokenizer-free estimate: ~1 token per CJK char, ~3.6 chars per token otherwise (slightly
    conservative for English so truncated prompts still fit)."""
    cjk = len(_CJK.findall(text))
    return int(cjk + (len(text) - cjk) / 3.6) + 1


def truncate_middle(text: str, max_tokens: int) -> str:
    """Keep the head and tail (LongBench convention) so the prompt fits max_tokens."""
    n = est_tokens(text)
    if n <= max_tokens:
        return text
    keep = int(len(text) * max_tokens / n)
    half = keep // 2
    return text[:half] + "\n\n[...]\n\n" + text[len(text) - half:]


def context_budget(ctx, max_tokens: int, reserve: int = 1500) -> int:
    """Prompt tokens available for the model under test after reserving room for the answer."""
    return max(2000, ctx.model.context - max_tokens - reserve)


def safe_extract(archive: Path, dest: Path) -> Path:
    """Extract a zip/tar archive, refusing absolute paths, '..' members and links (untrusted data)."""
    import tarfile
    import zipfile

    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for m in zf.infolist():
                target = (dest / m.filename).resolve()
                if not str(target).startswith(str(root)):
                    raise RuntimeError(f"unsafe path in {archive.name}: {m.filename}")
            zf.extractall(dest)
    else:
        with tarfile.open(archive) as tf:
            for m in tf.getmembers():
                target = (dest / m.name).resolve()
                if not str(target).startswith(str(root)) or m.issym() or m.islnk():
                    raise RuntimeError(f"unsafe member in {archive.name}: {m.name}")
            tf.extractall(dest, filter="data")
    return dest


# ---- document / chart QA metrics -------------------------------------------------------------------
def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred: str, golds: list[str], tau: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity (DocVQA): best match over gold answers, 0 below tau."""
    p = " ".join(str(pred).lower().split())
    best = 0.0
    for g in golds:
        g = " ".join(str(g).lower().split())
        if not g and not p:
            return 1.0
        nl = levenshtein(p, g) / max(len(p), len(g), 1)
        best = max(best, 1 - nl if nl < tau else 0.0)
    return best


def short_answer(text: str) -> str:
    """Final short answer from a response: strip 'Answer:' prefixes, take the last non-empty line."""
    t = (text or "").strip()
    m = re.findall(r"(?i)(?:final answer|answer)\s*[:：]\s*(.+)", t)
    if m:
        t = m[-1]
    else:
        lines = [l for l in t.splitlines() if l.strip()]
        t = lines[-1] if lines else ""
    return t.strip().strip("*").strip().rstrip(".").strip('"').strip()


def relaxed_match(pred: str, gold: str, tol: float = 0.05) -> bool:
    """ChartQA relaxed accuracy: numbers within 5% relative error, else case-insensitive exact match."""
    def num(s):
        s = str(s).strip().replace(",", "").replace("%", "").replace("$", "")
        try:
            return float(s)
        except ValueError:
            return None

    p, g = num(pred), num(gold)
    if p is not None and g is not None:
        return abs(p - g) <= tol * abs(g) if g != 0 else p == 0
    return str(pred).strip().lower() == str(gold).strip().lower()


def parquet_to_samples(files: list[Path], out_dir: Path, *, image_col: str, id_fn, keep: list[str]) -> list[dict]:
    """Stream image parquet files: write each image to out_dir/images/<id>.png, return metadata rows."""
    import pyarrow.parquet as pq

    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=64):
            for r in batch.to_pylist():
                sid = str(id_fn(r))
                img = r.get(image_col)
                if isinstance(img, dict) and img.get("bytes"):
                    (img_dir / f"{sid}.png").write_bytes(img["bytes"])
                rows.append({"id": sid, **{k: r.get(k) for k in keep}})
    return rows
