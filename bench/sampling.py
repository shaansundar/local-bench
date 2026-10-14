"""Seeded, stratified, nested subsets.

Each item gets a deterministic pseudo-random key from (seed, id). Within each stratum items are
ordered by that key, and an item's global position is its quantile inside its stratum. Taking the
first n items of that global order gives a proportionally stratified sample, and every smaller tier
is a prefix of every larger one (smoke ⊂ quick ⊂ standard ⊂ full).
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

from bench.benchmark import Sample
from bench.config import load_config
from bench.registry import load_specs, subset_path


def _key(seed: int, sid: str) -> float:
    h = hashlib.sha256(f"{seed}:{sid}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def ordered_ids(samples: list[Sample], seed: int, stratify: bool = True) -> list[str]:
    groups: dict[str, list[str]] = defaultdict(list)
    for s in samples:
        groups[(s.stratum or "") if stratify else ""].append(s.id)
    ranked: list[tuple[float, float, str]] = []
    for ids in groups.values():
        ids.sort(key=lambda i: _key(seed, i))
        n = len(ids)
        for rank, sid in enumerate(ids):
            ranked.append(((rank + 0.5) / n, _key(seed + 1, sid), sid))
    ranked.sort()
    return [sid for _, _, sid in ranked]


def tier_size(slug: str, tier: str) -> int | None:
    cfg = load_config()
    spec = load_specs()[slug]
    if tier in spec.tiers:
        return spec.tiers[tier]
    if tier not in cfg.tiers:
        raise KeyError(f"unknown tier '{tier}' (known: {', '.join(cfg.tiers)})")
    return cfg.tiers[tier]


def subset_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:12]


def get_subset(slug: str, tier: str, samples: list[Sample], *, freeze: bool = True) -> list[str]:
    """Frozen subset for (benchmark, tier): read it if present, else draw and (optionally) write it."""
    path = subset_path(slug, tier)
    known = {s.id for s in samples}
    if path.exists():
        data = json.loads(path.read_text())
        ids = [i for i in data["ids"] if i in known]
        missing = len(data["ids"]) - len(ids)
        if missing:
            raise RuntimeError(
                f"{slug}/{tier}: {missing} frozen sample ids are not in the loaded data "
                "(dataset changed?). Re-fetch, or delete the subset file to redraw."
            )
        return ids
    cfg = load_config()
    spec = load_specs()[slug]
    n = tier_size(slug, tier)
    order = ordered_ids(samples, cfg.seed, spec.stratify)
    ids = order if n is None or n >= len(order) else order[:n]
    if freeze:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"benchmark": slug, "tier": tier, "seed": cfg.seed, "n": len(ids), "hash": subset_hash(ids),
             "ids": ids}, indent=1))
    return ids
