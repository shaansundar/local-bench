"""Fast checks for the core invariants (no models, no network)."""

import os
import pickle
import sys
from pathlib import Path

import pytest

from bench import sandbox
from bench.benchmark import Sample
from bench.common import anls, extract_boxed, extract_mc_letter, math_equal, safe_unpickle, truncate_middle
from bench.metrics import summarize, wilson
from bench.sampling import ordered_ids
from bench.search_agents import gaia_score, xor_decrypt


def _samples(n=300):
    return [Sample(id=f"s{i}", data={}, stratum="abc"[i % 3]) for i in range(n)]


def test_subsets_are_deterministic_nested_and_stratified():
    a = ordered_ids(_samples(), seed=1234)
    b = ordered_ids(list(reversed(_samples())), seed=1234)
    assert a == b  # input order does not matter
    small, big = a[:30], a[:90]
    assert set(small) <= set(big)  # nested tiers
    counts = {k: sum(1 for i in small if int(i[1:]) % 3 == "abc".index(k)) for k in "abc"}
    assert all(9 <= c <= 11 for c in counts.values())  # proportional strata
    assert ordered_ids(_samples(), seed=99) != a


def test_metrics_wilson_and_repeats():
    lo, hi = wilson(5, 10)
    assert 0.2 < lo < 0.5 < hi < 0.8
    rows = [{"sample_id": "a", "score": 1.0}, {"sample_id": "a", "score": 0.0}, {"sample_id": "b", "score": 1.0}]
    s = summarize(rows)
    assert s["score"] == pytest.approx(0.75) and s["n_scored"] == 2


def test_answer_extraction_and_math():
    assert extract_boxed(r"so \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"
    assert extract_mc_letter("blah\nAnswer: C", "ABCD") == "C"
    assert extract_mc_letter("the answer is (B).", "ABCD") == "B"
    assert math_equal("0.5", r"\boxed{\frac{1}{2}}")[0]
    assert not math_equal("3", r"\boxed{4}")[0]


def test_doc_metrics_and_gaia():
    assert anls("Paris", ["paris"]) == 1.0
    assert anls("completely different", ["x"]) == 0.0
    assert gaia_score("1,000", "1000") and gaia_score("Saint Petersburg", "saint petersburg")
    assert not gaia_score("a, b", "a; b; c")


def test_xor_roundtrip():
    import base64
    import hashlib

    key = hashlib.sha256(b"canary").digest()
    msg = "héllo world".encode()
    ct = base64.b64encode(bytes(m ^ key[i % len(key)] for i, m in enumerate(msg))).decode()
    assert xor_decrypt(ct, "canary") == "héllo world"


def test_safe_unpickle_refuses_code():
    assert safe_unpickle(pickle.dumps(["ok", {"a": 1}])) == ["ok", {"a": 1}]

    class Evil:
        def __reduce__(self):
            return (os.system, ("echo pwned",))

    with pytest.raises(pickle.UnpicklingError):
        safe_unpickle(pickle.dumps(Evil()))


def test_truncate_middle_fits_budget():
    text = "word " * 50_000
    out = truncate_middle(text, 1000)
    assert len(out) < len(text) and "[...]" in out


@pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec is macOS only")
def test_sandbox_blocks_home_writes_and_network(tmp_path: Path):
    code = ("import os,socket\nopen('ok.txt','w').write('1')\n"
            "try:\n open(os.path.expanduser('~/bench_canary_test'),'w'); print('HOME_WRITE')\nexcept Exception: pass\n"
            "try:\n socket.create_connection(('1.1.1.1',443),timeout=3); print('NET')\nexcept Exception: pass\n")
    r = sandbox.run([sys.executable, "-c", code], cwd=tmp_path, timeout=30, network="none")
    assert (tmp_path / "ok.txt").exists()
    assert "HOME_WRITE" not in r.stdout and "NET" not in r.stdout
    assert not Path("~/bench_canary_test").expanduser().exists()
