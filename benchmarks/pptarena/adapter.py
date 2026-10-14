import json
import sys
from collections import defaultdict

from bench.benchmark import AgentTask, Benchmark, Grade, Sample
from bench.common import hf_snapshot, vendor_repo
from bench.models.client import image_part
from bench.office import render_slides

UPSTREAM = "https://github.com/michaelofengenden/PPTArena"
REVISION = "7072dc9b087f90c3fa271be015d66b7fe403daf9"
REPO = "mofengenden/PPTArena"

# Verbatim from PPTArena src/llm/judge.py
RUBRIC_IF = """
HARSH SCORING POLICY (very strict):
- Choose the lower score when uncertain between adjacent scores.
- For translation/summarization or other text edits requiring reasoning, semantic similarity is more important than exact wording.

INSTRUCTION_FOLLOWING score (0-5):
- 5: Every requested object/change exists and is exactly correct; nothing requested is missing or misapplied; no extra edits beyond the instruction.
- 4: All requested changes exist and are mostly correct; only a tiny inaccuracy that does not affect meaning.
- 3: Most requested changes exist but at least one is incomplete, incorrect, or missing detail.
- 2: Only some requested changes exist; notable misses or incorrect applications.
- 1: Requested changes largely not performed or substantially incorrect.
- 0: Contradicts or ignores the instruction entirely.
"""
RUBRIC_VQ = """
HARSH SCORING POLICY (very strict):
- Penalize any unintended visual change to non-requested content (fonts, sizes, colors, positions, shapes, charts, images, tables, or slide structure).
- Choose the lower score when uncertain between adjacent scores.

VISUAL_Quality score (0-5):
- 5: No unintended changes to non-requested content; fonts, sizes, colors, positions, and objects match Ground Truth; structure preserved.
- 4: Visually very close to the Ground Truth; only imperceptible or negligible differences (e.g., sub-pixel alignment); no style drift.
- 3: Minor but noticeable visual differences (e.g., slight font weight/size/spacing shifts) without breaking layout.
- 2: Clear deviations (e.g., wrong fonts/sizes/colors, noticeable position shifts) or small layout issues.
- 1: Major deviations (overlap, off-canvas, broken layout) but still legible.
- 0: Severely broken or unreadable slide.
"""
SYSTEM_IF = f"""You are a strict judge of INSTRUCTION FOLLOWING.

CRITICAL UNDERSTANDING:
- The "Instruction" is what the model/editor received (the user's request)
- The "Style Target" is YOUR evaluation rubric - the model DID NOT see this
- You will receive a FOCUSED DIFF showing only what changed between ground_truth and prediction
- Your job: Judge if the prediction's changes match the ground_truth's changes
- DO NOT compare prediction to initial - focus on whether prediction achieved ground_truth's outcome
- If the diff shows minimal differences, that's GOOD (high score)
- Ground Truth is ONE valid example, not the only correct answer

FLEXIBILITY:
- Accept different valid approaches (e.g., flags in a list vs rows is fine if they match the text)
- Exact positions/sizes don't matter unless the Instruction explicitly requires them
- Very small measurement variations (±1%) are acceptable for fonts/sizes due to rounding
- Z-order (layering) differences ARE significant and should be noted
- Focus on semantic properties: text content, font names, colors, structural changes, z-order

{RUBRIC_IF}
Output a single JSON object with:
- instruction_following_score (0-5)
- instruction_following_reason (one sentence, specific evidence comparing prediction to ground_truth)
"""
SYSTEM_VQ = f"""You are a judge of VISUAL/CONTENT QUALITY and PRESERVATION.

CRITICAL UNDERSTANDING:
- The "Instruction" is what the model/editor received
- The "Style Target" is YOUR evaluation rubric - the model DID NOT see this
- Ground Truth is ONE valid example - accept other valid visual approaches
- Focus on SEMANTIC correctness: Are the visual elements correct? No overlap? Readable?
- Exact positions/sizes don't matter unless the Instruction explicitly requires them
- You will only receive Ground Truth and Prediction slide images; compare them directly slide-by-slide.
- Use any provided Style Target guidance to check required visual cues strictly.

FLEXIBILITY:
- Different layouts achieving the same goal are acceptable (e.g., list vs grid)
- Small position variations are fine if elements are clear and non-overlapping
- Theme colors may vary slightly as long as they're harmonious
- "Approximately centered" or "well-aligned" is acceptable without pixel-perfection

{RUBRIC_VQ}
Output a single JSON object with:
- visual_quality_score (0-5)
- visual_quality_reason (one sentence, specific evidence about visual differences)
"""
AGENT_PROMPT = """Edit the PowerPoint file ./input.pptx according to the instruction below, and save the edited deck as ./output.pptx.
Edit in place: keep everything the instruction does not ask to change exactly as it is (layout, fonts, colors, other slides).
python-pptx and lxml are installed (`python3`), and you may edit the OOXML directly.

Instruction: {instruction}"""


def _ppt():
    root = vendor_repo(UPSTREAM, REVISION, "PPTArena", sparse=["/src/llm/", "/src/ppt/", "/src/utils/", "/run_evaluation.py",
                                                                "/src/requirements.txt"])
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from ppt import ppt_processor

    return ppt_processor


class PPTArena(Benchmark):
    def fetch(self):
        _ppt()
        hf_snapshot(REPO, self.data_dir)
        self.mark_fetched()

    def load(self):
        cases = json.loads((self.data_dir / "evaluation_pairs_refined.json").read_text())
        return [Sample(id=f"{i:03d}", data=c, stratum=(c.get("category") or ["other"])[0]) for i, c in enumerate(cases)]

    def build(self, sample, ctx):
        return AgentTask(prompt=AGENT_PROMPT.format(instruction=sample.data["prompt"]),
                         files={"input.pptx": self.data_dir / sample.data["original"]})

    def grade(self, sample, output, ctx):
        judge = ctx.require_judge()
        pred = output.workspace / "output.pptx" if output.workspace else None
        if not pred or not pred.exists():
            return Grade(score=0.0, details={"error": "no output.pptx", "category": sample.stratum})
        pp = _ppt()
        orig = self.data_dir / sample.data["original"]
        gt = self.data_dir / sample.data["ground_truth"]
        instr, style = sample.data["prompt"], sample.data.get("style_target") or "Not provided."
        try:
            initial_json, gt_json, pred_json = pp.pptx_to_json(str(orig)), pp.pptx_to_json(str(gt)), pp.pptx_to_json(str(pred))
        except Exception as e:
            return Grade(score=0.0, details={"error": f"output.pptx unreadable: {e}", "category": sample.stratum})
        diff = pp.diff_pptx_json(gt_json, pred_json, initial_json)
        if not diff.get("has_differences") and diff.get("similarity_score", 0) > 0.9999:
            if_score, if_info = 5.0, {"reason": "perfect structural match"}
        else:
            prompt = (f"--- USER INSTRUCTION (what the model received) ---\n{instr}\n\n"
                      f"--- STYLE TARGET (your evaluation rubric - the model DID NOT see this) ---\n{style}\n\n"
                      f"--- SMART DIFF ANALYSIS (Prediction vs Ground Truth) ---\n{pp.format_diff_for_judge(diff, instr)}\n\n"
                      "CRITICAL COMPARISON INSTRUCTIONS:\nThe diff above shows ONLY the differences between prediction and "
                      "ground_truth.\n- If the diff shows \"No differences\" → Perfect match → Score 5\n- If the diff shows "
                      "differences in properties that the instruction requires → Score based on correctness\n- Focus on whether "
                      "prediction achieved the same semantic outcome as ground_truth")
            obj, if_info = judge.score_json(prompt, "rubric", system=SYSTEM_IF)
            if_score = float(obj.get("instruction_following_score") or 0)
        render_dir = output.workspace / "_render"
        n = int(self.spec.options.get("max_slides", 20))
        gt_imgs = render_slides(gt, render_dir / "gt", n, scale=0.8)
        pred_imgs = render_slides(pred, render_dir / "pred", n, scale=0.8)
        parts = [{"type": "text", "text": f"Instruction: {instr}\nStyle Target: {style}\n\nGround Truth slides:"}]
        for i, p in enumerate(gt_imgs, 1):
            parts += [{"type": "text", "text": f"Ground Truth slide {i}"}, image_part(p)]
        parts.append({"type": "text", "text": "Prediction slides:"})
        for i, p in enumerate(pred_imgs, 1):
            parts += [{"type": "text", "text": f"Prediction slide {i}"}, image_part(p)]
        vobj, vq_info = judge.score_json(parts, "rubric", system=SYSTEM_VQ)
        vq_score = float(vobj.get("visual_quality_score") or 0)
        return Grade(score=(if_score + vq_score) / 10.0, extracted=f"IF {if_score:g} / VQ {vq_score:g}",
                     judge={"judge_id": vq_info["judge_id"], "instruction": if_info, "visual": vq_info},
                     details={"if": if_score, "vq": vq_score, "category": sample.stratum})

    def aggregate(self, rows):
        d = [r["details"] or {} for r in rows]
        by = defaultdict(list)
        for r in rows:
            by[(r["details"] or {}).get("category")].append(r["score"])
        avg = lambda v: sum(v) / len(v) if v else None
        return {"instruction_following_0_5": avg([x.get("if", 0) for x in d]), "visual_quality_0_5": avg([x.get("vq", 0) for x in d]),
                "by_category": {k: avg(v) for k, v in by.items() if k}}


BENCHMARK = PPTArena
