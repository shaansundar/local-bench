"""Write a synthetic results store so the dashboard can be explored (or worked on) without running models.

    uv run python scripts/demo_results.py /tmp/bench-demo
    BENCH_MODELS=/tmp/bench-demo/models.yaml BENCH_RESULTS_DIR=/tmp/bench-demo/results uv run bench serve

Every model is fictional ("Demo · …") and every score is random. Never mix this with real results.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from pathlib import Path

from bench import metrics
from bench.registry import load_specs
from bench.store import Store

# key: (display name, strength, vision, kind, output tok/s, mean output tokens)
MODELS = {
    "demo-local-4b": ("Demo · Local 4B", 0.38, False, "managed", 85, 1400),
    "demo-local-14b": ("Demo · Local 14B", 0.52, True, "managed", 48, 2200),
    "demo-local-32b": ("Demo · Local 32B", 0.63, False, "external", 26, 3100),
    "demo-moe-80b": ("Demo · Local MoE 80B", 0.71, True, "external", 55, 3600),
    "demo-cloud-ref": ("Demo · Cloud reference", 0.84, True, "cloud", 90, 2600),
}

# How forgiving each benchmark is relative to model strength (1.0 = neutral).
EASE = {"math500": 1.25, "ifeval": 1.2, "mmlu_redux": 1.2, "chartqa": 1.15, "docvqa": 1.2, "databench": 1.05,
        "hle": 0.25, "browsecomp_100": 0.35, "browsecomp_zh": 0.35, "hmmt_feb2026": 0.6, "aime2026": 0.75,
        "aime2025": 0.8, "scicode": 0.45, "gdpval_aa": 0.6, "officeqa_pro": 0.55, "pptarena": 0.55,
        "spreadsheetbench2": 0.5, "clawgym_bench": 0.55, "tau3_banking": 0.6, "longbench_v2": 0.7}


def unit(*parts: str) -> float:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    models_yaml = {"models": {k: {"display_name": name, "kind": kind, "provider": "openai",
                                  "base_url": "http://127.0.0.1:9/v1", "vision": vision, "context": 131072,
                                  "notes": "synthetic demo model"}
                              for k, (name, _, vision, kind, _, _) in MODELS.items()}}
    (out / "models.yaml").write_text("# synthetic demo models — see scripts/demo_results.py\n" + json.dumps(models_yaml, indent=2))
    db = out / "results" / "results.db"
    if db.exists():
        sys.exit(f"{db} already exists; pick an empty directory")
    store = Store(db)
    specs = load_specs()
    now = time.time()
    tier, n = "quick", 50
    for m, (_, strength, vision, kind, tps, mean_tokens) in MODELS.items():
        for slug, spec in specs.items():
            rid = store.create_run(m, slug, tier, badges=spec.badges)
            if spec.requires.vision and not vision:
                store.update_run(rid, status="skipped", error="model has no vision input")
                continue
            target = min(0.97, max(0.02, strength * EASE.get(slug, 0.9) + (unit(m, slug) - 0.5) * 0.12))
            rng = random.Random(f"{m}/{slug}")
            harness = {"chat": "direct", "oc": "opencode", "native": "native"}[spec.runner]
            for i in range(n):
                sid = f"demo-{i:03d}"
                score = 1.0 if rng.random() < target else 0.0
                toks = int(mean_tokens * (0.4 + rng.random() * 1.2))
                store.put_sample(rid, sid, 0, status="graded", score=score,
                                 prompt=[{"role": "user", "content": f"(synthetic demo prompt {i + 1} for {spec.title})"}],
                                 output="(synthetic demo output)", extracted="B" if score else "D", gold="B",
                                 usage={"input_tokens": 900 + int(rng.random() * 4000), "output_tokens": toks},
                                 latency=toks / tps, finish_reason="stop", graded_at=now)
            store.refresh_counts(rid)
            rows = store.q("SELECT sample_id, score FROM samples WHERE run_id=?", (rid,))
            s = metrics.summarize(rows)
            store.update_run(rid, status="done", harness=harness, subset_hash="demo", n_total=n,
                             score=s["score"], ci_low=s["ci_low"], ci_high=s["ci_high"],
                             judge_id="demo-judge" if spec.judge else None,
                             metrics={"n_scored": s["n_scored"], "complete": True},
                             started_at=now - 3600, finished_at=now)
    print(f"wrote {db}\n  BENCH_MODELS={out / 'models.yaml'} BENCH_RESULTS_DIR={out / 'results'} uv run bench serve")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]).expanduser().resolve())
