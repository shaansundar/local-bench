"""Runs inside PPTC's own venv (python-pptx 0.6.21, openai<1) with the PPTC repo as cwd.

  prepare <short|long>                         build base/label decks from the label API sequences
  prompt  <short|long> <session_file> <turn>   print {"prompt": ...} for a turn (turn-based setting)
  grade   <short|long> <session_file> <turn> <reply_file> <out_pptx>
                                               execute the reply's API calls, PPTX-match vs label

The model is never called from here: query_azure_openai is patched to record the prompt.
"""
import contextlib
import io
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.getcwd())
CAPTURED = []

# api_selection imports openai.embeddings_utils (pulls matplotlib/plotly/sklearn). Embedding-based API
# selection is off in the standard setting, so a stub keeps helper start-up fast.
import types  # noqa: E402

_eu = types.ModuleType("openai.embeddings_utils")
_eu.get_embedding = _eu.cosine_similarity = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("embeddings disabled"))
sys.modules["openai.embeddings_utils"] = _eu

from src import openai_api  # noqa: E402

openai_api.query_azure_openai = lambda prompt, model=None, id=None: CAPTURED.append(prompt) or ""

from src import evaluate, modeling, ppt_executor, utils  # noqa: E402

ROOT = os.getcwd() + "/"


def args_for(dataset):
    return SimpleNamespace(dataset=dataset, planning=False, api_selection=False, content_selection=False,
                           model="gpt4", model_id="None", api_lack=False, api_update=False, api_topk=10,
                           user_path=ROOT, data_path="test", second=False, robust=False, noisy=False, tf=True,
                           sess=False, exp_name="bench")


def set_name(dataset):
    return "Create_new_slides" if dataset == "short" else "Edit_PPT_template"


def turn_of(dataset, session_file, turn):
    session = utils.parse_train_json(ROOT + f"PPT_test_input/{set_name(dataset)}/{session_file}")
    return session, session[int(turn)]


def main():
    cmd, dataset = sys.argv[1], sys.argv[2]
    args = args_for(dataset)
    assistant = modeling.PPT_assistant(args)
    quiet = contextlib.redirect_stdout(io.StringIO())
    if cmd == "prepare":
        with quiet:
            import main as pptc_main  # noqa: F401  (needs repo-relative paths)
            pptc_main.prepare_data(assistant, args)
        print(json.dumps({"ok": True}))
        return
    session, turn = turn_of(dataset, sys.argv[3], sys.argv[4])
    turn_id, instruction, label_api, base, label, _, _ = turn
    history = session[: int(sys.argv[4])]
    if cmd == "prompt":
        with quiet:
            assistant.load_ppt(ROOT + base)
            assistant.load_chat_history([x[1].split("##")[0] for x in history],
                                        [x[2].strip(";").split(";") for x in history])
            assistant.chat(instruction.split("##")[0], ppt_path=ROOT + base, verbose=False)
        print(json.dumps({"prompt": CAPTURED[-1] if CAPTURED else ""}))
    elif cmd == "grade":
        reply = open(sys.argv[5]).read()
        out = sys.argv[6]
        with quiet:
            assistant.load_ppt(ROOT + base)
            apis = utils.parse_api(reply)
            assistant.api_executor(apis, test=True)
            ppt_executor.save_ppt(out)
            s, pt, pc = evaluate.calc_acc(ROOT + label, out, instruction)
        print(json.dumps({"str_correct": s, "pos_total": pt, "pos_correct": pc, "n_apis": len(apis)}))


if __name__ == "__main__":
    main()
