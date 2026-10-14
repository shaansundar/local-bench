"""Discovers benchmarks/<slug>/benchmark.yaml + adapter.py."""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import yaml

from bench.benchmark import Benchmark, BenchmarkSpec
from bench.config import ROOT

BENCH_DIR = ROOT / "benchmarks"


@lru_cache
def load_specs() -> dict[str, BenchmarkSpec]:
    specs: dict[str, BenchmarkSpec] = {}
    for yml in sorted(BENCH_DIR.glob("*/benchmark.yaml")):
        if yml.parent.name.startswith("_"):
            continue
        spec = BenchmarkSpec(**yaml.safe_load(yml.read_text()))
        if spec.slug != yml.parent.name:
            raise ValueError(f"{yml}: slug '{spec.slug}' must match folder name '{yml.parent.name}'")
        specs[spec.slug] = spec
    return specs


_instances: dict[str, Benchmark] = {}


def get_benchmark(slug: str) -> Benchmark:
    if slug in _instances:
        return _instances[slug]
    specs = load_specs()
    if slug not in specs:
        raise KeyError(f"unknown benchmark '{slug}'")
    folder = BENCH_DIR / slug
    mod_name = f"benchmarks_{slug}"
    spec_obj = importlib.util.spec_from_file_location(mod_name, folder / "adapter.py")
    module = importlib.util.module_from_spec(spec_obj)
    sys.modules[mod_name] = module
    spec_obj.loader.exec_module(module)
    cls = getattr(module, "BENCHMARK")
    inst = cls(specs[slug], folder)
    _instances[slug] = inst
    return inst


def resolve_benchmarks(selector: str) -> list[str]:
    """'all', a comma list of slugs, or category names (case-insensitive)."""
    specs = load_specs()
    if selector.strip().lower() == "all":
        return list(specs)
    out: list[str] = []
    for tok in [t.strip() for t in selector.split(",") if t.strip()]:
        if tok in specs:
            out.append(tok)
            continue
        matched = [s for s, sp in specs.items() if sp.category.lower() == tok.lower()]
        if not matched:
            raise KeyError(f"'{tok}' is neither a benchmark slug nor a category")
        out.extend(matched)
    return list(dict.fromkeys(out))


def bench_folder(slug: str) -> Path:
    return BENCH_DIR / slug


def subset_path(slug: str, tier: str) -> Path:
    return BENCH_DIR / slug / "subsets" / f"{tier}.json"


def all_folders() -> list[Path]:
    return [p for p in BENCH_DIR.iterdir() if p.is_dir() and not p.name.startswith("_")]


__all__ = ["load_specs", "get_benchmark", "resolve_benchmarks", "subset_path", "BENCH_DIR", "Path"]
