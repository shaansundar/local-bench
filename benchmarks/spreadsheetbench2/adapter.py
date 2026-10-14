import json
import shutil
from collections import defaultdict
from pathlib import Path

from bench.benchmark import AgentTask, Benchmark, Grade, Sample
from bench.common import VENDOR, hf_file, load_module, safe_extract, vendor_raw_files
from bench.office import recalc_if_needed

REPO = "KAKA22/SpreadsheetBench-v2"
ARCHIVE = "spreadsheetbench-v2-09-10.zip"
UPSTREAM = "RUCKBReasoning/SpreadsheetBench-2"
REVISION = "5842db8f836887db15144b267467339bb7458f92"
CATEGORIES = ["Debugging", "Financial_Model", "Template"]
VIEWER = Path(__file__).with_name("view_xlsx.py")

PROMPT = """<task_context>

## Important
- When completing spreadsheet tasks, strictly avoid altering any cells that already contain values unless explicitly instructed. Modify only the cells that are required for the task.
- You need to complete the instructions and ensure that the original formatting is preserved as much as possible.

## Environment
You can complete the corresponding spreadsheet tasks using the following Python environment (`python3`):
`Pandas`, `openpyxl` and `numpy`.

## Recommended Workflow
1. **Inspect**: Use the viewer to understand the structure and contents of the spreadsheet.
   - List all sheets: `python3 view_xlsx.py {spreadsheet_path} list`
   - View specific sheet: `python3 view_xlsx.py {spreadsheet_path} content "Sheet1"`
   - View row range: `python3 view_xlsx.py {spreadsheet_path} content "Sheet1" 1 50`
2. **Plan**: Decide how to manipulate the data using `openpyxl`.
3. **Implement**: Create a python script (e.g., `process_data.py`) that performs the edit.
4. **Execute**: Run the python script using `python3 process_data.py`
5. **Verify**: Check the output file exists, ensure the answer is correct, and that the existing structure, layout, and formatting are preserved.

</task_context>

<instructions>

## Task instructions
You need to process a spreadsheet file based on specific instructions.
**Instruction:** {instruction}

## Input and Output
You are provided with an input spreadsheet file and you need to generate an output spreadsheet file based on the output_path.
**Input File:** {spreadsheet_path}
**Output Path:** {output_path}

</instructions>"""


def _evaluation():
    vendor_raw_files(UPSTREAM, REVISION, ["evaluation/evaluation.py"], VENDOR / "spreadsheetbench2")
    mod = load_module("spreadsheetbench2_eval", VENDOR / "spreadsheetbench2" / "evaluation" / "evaluation.py")
    mod.print = lambda *a, **k: None
    return mod


class SpreadsheetBench2(Benchmark):
    def fetch(self):
        _evaluation()
        z = hf_file(REPO, ARCHIVE, self.data_dir)
        safe_extract(z, self.data_dir)
        z.unlink()
        self.mark_fetched()

    @property
    def root(self):
        return self.data_dir / "spreadsheetbench-v2"

    def load(self):
        out = []
        for cat in CATEGORIES:
            for r in json.loads((self.root / cat / "dataset.json").read_text()):
                out.append(Sample(id=f"{cat}/{r['id']}", data={**r, "category": cat}, stratum=cat))
        return out

    def build(self, sample, ctx):
        d = sample.data
        src = self.root / d["category"] / d["spreadsheet_path"]
        return AgentTask(prompt=PROMPT.format(instruction=d["instruction"], spreadsheet_path=f"./{src.name}",
                                              output_path="./output.xlsx"),
                         files={src.name: src, "view_xlsx.py": VIEWER})

    def _recalced(self, path: Path) -> Path:
        dest = self.data_dir / "recalc" / path.relative_to(self.root)
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            recalc_if_needed(path, dest)
        return dest

    def grade(self, sample, output, ctx):
        d = sample.data
        out = output.workspace / "output.xlsx" if output.workspace else None
        if not out or not out.exists():
            return Grade(score=0.0, details={"error": "no output.xlsx", "category": d["category"]})
        base = self.root / d["category"]
        inp = self._recalced(base / d["spreadsheet_path"])
        gt = self._recalced(base / d["golden_response_path"])
        proc = output.workspace / "output_recalc.xlsx"
        recalc_if_needed(out, proc)
        font = d["category"] == "Debugging" and "Color" in d["spreadsheet_path"]
        formula = d["category"] == "Debugging" and "Embedded" in d["spreadsheet_path"]
        try:
            ok, msg, correct, total, reg, mod = _evaluation().compare_workbooks_with_regression(
                str(inp), str(gt), str(proc), d["answer_position"], with_font_color=font, with_formula=formula)
        except Exception as e:
            ok, msg, correct, total, reg, mod = False, f"{type(e).__name__}: {e}", 0, 0, {}, {}
        return Grade(score=float(bool(ok)), extracted=f"{correct}/{total} cells",
                     details={"category": d["category"], "message": str(msg)[:400], "regression": reg, "modification": mod})

    def oracle_output(self, sample, ctx):
        from bench.benchmark import ModelOutput

        ws = ctx.run_dir / "work" / sample.id.replace("/", "__")
        ws.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.root / sample.data["category"] / sample.data["golden_response_path"], ws / "output.xlsx")
        return ModelOutput(text="(golden file)", workspace=ws)

    def aggregate(self, rows):
        by = defaultdict(list)
        for r in rows:
            by[(r["details"] or {}).get("category")].append(r["score"])
        return {"by_category": {k: sum(v) / len(v) for k, v in by.items() if k}}


BENCHMARK = SpreadsheetBench2
