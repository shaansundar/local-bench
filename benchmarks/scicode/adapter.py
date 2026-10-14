import json
import sys

from bench import sandbox
from bench.benchmark import Benchmark, Grade, ModelOutput, Sample
from bench.common import gdrive_download, hf_file, load_module, read_jsonl, vendor_repo
from bench.runners.chat import resolve_max_tokens
from bench.sandbox.exec import extract_code, scratch_dir

REPO = "SciCode1/SciCode"
UPSTREAM = "https://github.com/scicode-bench/SciCode"
REVISION = "e3158ea011d4235245a547460d3688d7ccbf9900"
H5_ID = "17G_k65N_6yFFZ2O-jQH00Lh6iaw3z-AW"
PREFILLED = {("13", 6), ("62", 1), ("76", 3)}  # steps whose gold code is given (official gencode skips them)


def _repo():
    return vendor_repo(UPSTREAM, REVISION, "SciCode")


def _parse():
    root = _repo()
    return load_module("scicode_parse", root / "src" / "scicode" / "parse" / "parse.py")


class SciCode(Benchmark):
    def fetch(self):
        _repo()
        hf_file(REPO, "problems_test.jsonl", self.data_dir)
        gdrive_download(H5_ID, self.data_dir / "test_data.h5")
        self.mark_fetched()

    def load(self):
        return [Sample(id=str(r["problem_id"]), data=r) for r in read_jsonl(self.data_dir / "problems_test.jsonl")]

    # ---- generation: one call per sub-step, conditioned on the model's own previous steps ----
    def execute(self, sample, ctx):
        parse = _parse()
        root = _repo()
        wbg = self.spec.options.get("with_background", True)
        template = (root / "eval" / "data" / ("multistep_template.txt" if wbg else "background_comment_template.txt")).read_text()
        prob = sample.data
        steps = prob["sub_steps"]
        prev_code: list[str] = []
        step_files, transcripts = [], []
        usage = {"input_tokens": 0, "output_tokens": 0}
        latency = 0.0
        max_tokens = resolve_max_tokens(None, self, ctx)
        for i, step in enumerate(steps, 1):
            if (sample.id, i) in PREFILLED:
                prefilled = (root / "eval" / "data" / f"{sample.id}.{i}.txt").read_text()
                func = parse.extract_function_name(step["function_header"])
                prev_code.append(parse.get_function_from_code(prefilled, func))
                continue
            lines = []
            for j in range(i - 1):
                desc = steps[j]["step_description_prompt"] + ("\n" + steps[j]["step_background"] if wbg else "")
                lines += [desc, prev_code[j], "------"]
            nxt = steps[i - 1]["step_description_prompt"] + ("\n" + steps[i - 1]["step_background"] if wbg else "")
            nxt += f"\n\n{step['function_header']}\n\n{step['return_line']}"
            prompt = template.format(problem_steps_str="\n\n".join(lines[:-1]), next_step_str=nxt,
                                     dependencies=prob["required_dependencies"])
            res = ctx.client.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens)
            usage["input_tokens"] += res.usage.get("input_tokens") or 0
            usage["output_tokens"] += res.usage.get("output_tokens") or 0
            latency += res.latency
            code = extract_code(res.text) or ""
            func = parse.extract_function_name(step["function_header"])
            try:
                fcode = parse.get_function_from_code(code, func) or code
            except Exception:
                fcode = code
            previous = f"{prob['required_dependencies']}\n" + "\n".join(prev_code) + "\n"
            step_files.append({"step": step["step_number"], "index": i, "code": f"{previous}\n{code}",
                               "n_tests": len(step["test_cases"])})
            transcripts.append({"step": step["step_number"], "response": res.text[-6000:]})
            prev_code.append(fcode)
        return ModelOutput(text="\n\n".join(f"# step {s['step']}\n{s['code'].split(chr(10) + chr(10), 1)[-1]}" for s in step_files)[-20000:],
                           usage=usage, latency=latency, finish_reason="stop",
                           extra={"step_files": step_files, "responses": transcripts})

    # ---- grading: official test files, run in the sandbox against test_data.h5 --------------
    def grade(self, sample, output, ctx):
        files = output.extra.get("step_files") or []
        if not files:
            return Grade(score=0.0, details={"steps": [], "main_solved": False})
        root = _repo()
        h5 = self.data_dir / "test_data.h5"
        steps = {s["step_number"]: s for s in sample.data["sub_steps"]}
        results = []
        work = scratch_dir("scicode-")
        timeout = float(self.spec.options.get("step_timeout_s", 600))
        for f in files:
            tests = steps[f["step"]]["test_cases"]
            script = f["code"] + "\nfrom scicode.parse.parse import process_hdf5_to_tuple\n"
            script += f"targets = process_hdf5_to_tuple('{f['step']}', {len(tests)}, h5py_file={str(h5)!r})\n"
            for idx, t in enumerate(tests):
                script += f"target = targets[{idx}]\n\n{t}\n"
            path = work / f"{f['step']}.py"
            path.write_text(script)
            r = sandbox.run([sys.executable, str(path)], cwd=work, timeout=timeout, network="none",
                            env={"PYTHONPATH": str(root / "src"), "HF_DATASETS_OFFLINE": "1"})
            status = "pass" if r.returncode == 0 else ("timeout" if r.timed_out else "fail")
            results.append({"step": f["step"], "status": status, "stderr": r.stderr[-600:] if status != "pass" else ""})
        passed = sum(r["status"] == "pass" for r in results)
        return Grade(score=passed / len(results), extracted=f"{passed}/{len(results)} steps",
                     details={"steps": results, "passed": passed, "total": len(results),
                              "main_solved": passed == len(results)})

    def aggregate(self, rows):
        tot = sum((r["details"] or {}).get("total", 0) for r in rows)
        ok = sum((r["details"] or {}).get("passed", 0) for r in rows)
        main = [bool((r["details"] or {}).get("main_solved")) for r in rows]
        return {"subproblem_accuracy": ok / tot if tot else 0.0,
                "main_problem_accuracy": sum(main) / len(main) if main else 0.0}


BENCHMARK = SciCode
