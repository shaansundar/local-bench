"""Runs inside BFCL's venv. Registers the model under test as an OpenAI-compatible function-calling
model, swaps BFCL's SerpAPI web search for keyless DuckDuckGo (SerpAPI's own `duckduckgo` engine is
what BFCL queried), then generates and partially evaluates one test entry.

  list                     -> JSON [{"category", "id"}] for the scoring categories used by the suite
  run <category> <id>      -> JSON {"correct": bool, ...}

Env: BFCL_PROJECT_ROOT (per-sample scratch dir), OPENAI_BASE_URL, OPENAI_API_KEY, BENCH_MODEL_ID
"""
import json
import os
import sys
import time
from types import SimpleNamespace

from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler

REG = "bench-model-FC"
CATEGORIES = [
    "simple_python", "simple_java", "simple_javascript", "multiple", "parallel", "parallel_multiple", "irrelevance",
    "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple", "live_irrelevance", "live_relevance",
    "multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context",
    "web_search_base", "web_search_no_snippet",
]

MODEL_CONFIG_MAPPING[REG] = ModelConfig(
    model_name=os.environ.get("BENCH_MODEL_ID", "local"), display_name=REG, url="", org="local", license="n/a",
    model_handler=OpenAICompletionsHandler, is_fc_model=True, underscore_to_dot=True,
)


def _keyless_search():
    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code import web_search as ws

    def search_engine_query(self, keywords, max_results=10, region="wt-wt"):
        from ddgs import DDGS

        for attempt in range(4):
            try:
                rows = DDGS().text(keywords, max_results=max_results or 10, region=region or "wt-wt")
                break
            except Exception as e:
                if attempt == 3:
                    return {"error": f"search failed: {e}"}
                time.sleep(3 * (attempt + 1))
        out = []
        for r in rows:
            item = {"title": r.get("title"), "href": r.get("href")}
            if self.show_snippet:
                item["body"] = r.get("body")
            out.append(item)
        return out

    ws.WebSearchAPI.search_engine_query = search_engine_query


def main():
    cmd = sys.argv[1]
    if cmd == "list":
        from bfcl_eval.utils import load_dataset_entry

        rows = []
        for c in CATEGORIES:
            for e in load_dataset_entry(c):
                rows.append({"category": c, "id": e["id"]})
        print(json.dumps(rows))
        return
    category, entry_id = sys.argv[2], sys.argv[3]
    _keyless_search()
    from bfcl_eval import _llm_response_generation as gen
    from bfcl_eval.constants.eval_config import PROJECT_ROOT, SCORE_PATH, TEST_IDS_TO_GENERATE_PATH
    from bfcl_eval.eval_checker import eval_runner

    TEST_IDS_TO_GENERATE_PATH.write_text(json.dumps({category: [entry_id]}))
    args = SimpleNamespace(model=[REG], test_category=[category], temperature=float(os.environ.get("BENCH_TEMPERATURE", "0.001")),
                           include_input_log=False, exclude_state_log=False, num_threads=1, num_gpus=1, backend="vllm",
                           gpu_memory_utilization=0.9, result_dir=None, run_ids=True, allow_overwrite=True,
                           skip_server_setup=True, local_model_path=None, lora_modules=None, enable_lora=False,
                           max_lora_rank=None)
    gen.main(args)
    try:
        eval_runner.main([REG], [category], None, None, partial_eval=True)
    except Exception as e:  # leaderboard summary needs >=2 entries (stdev); per-category scores are already written
        print(f"[bench] eval summary skipped: {type(e).__name__}: {e}", file=sys.stderr)
    score_files = list(SCORE_PATH.rglob(f"*{category}*score.json"))
    if not score_files:
        print(json.dumps({"correct": False, "error": "no score file"}))
        return
    lines = [json.loads(l) for l in score_files[0].read_text().splitlines() if l.strip()]
    head = lines[0]
    failures = lines[1:]
    correct = head.get("correct_count", 0) >= 1 and not failures
    result_files = list((PROJECT_ROOT / "result").rglob(f"*{category}*result.json"))
    raw = result_files[0].read_text()[-6000:] if result_files else ""
    print(json.dumps({"correct": bool(correct), "accuracy": head.get("accuracy"),
                      "failure": failures[0] if failures else None, "result_tail": raw}, default=str))


if __name__ == "__main__":
    main()
