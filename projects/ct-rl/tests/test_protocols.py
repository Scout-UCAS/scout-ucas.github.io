import json
import random
from dataclasses import asdict

import pytest
import torch

from ct_rl.model import CTModel
from ct_rl.settings import Config, ModelSettings
from ct_rl.experiment import save_checkpoint
from ct_rl.public import adapt_public, grade, prepare_public, public_eval, read_public
from ct_rl.external import predictions
from ct_rl.tasks import Pair, make_pair


def test_group_is_invariant_to_instance_renaming():
    rng = random.Random(29)
    for family in ("graph", "rules"):
        pair = make_pair(family, "id", rng)
        n = len(pair.world["L"]) if family=="graph" else pair.world["n"]
        perm = rng.sample(range(n), n)
        if family == "graph":
            w = {op: [0]*n for op in ("L","R")}
            for op in w:
                for a,b in enumerate(pair.world[op]):
                    w[op][perm[a]] = perm[b]
        else:
            w = {"n":n, "edges":[[perm[a],perm[b]] for a,b in reversed(pair.world["edges"])]}
        assert pair.group == Pair(family,w,pair.starts,pair.operations,"train").group


def test_public_source_formats_and_grading(tmp_path):
    path = tmp_path/"input.json"
    path.write_text(json.dumps({"example1":{"test_example":{"question":"Facts.", "query":"True?",
                         "answer":True,"chain_of_thought":["Because."]}}}))
    item, = read_public("prontoqa", path)
    assert item.question == "Facts. True?" and grade("prontoqa", "true.", item.answer)
    path.write_text(json.dumps([{"question":"Q","answer":"Work.\n#### 1,250"}]))
    item, = read_public("gsm8k", path)
    assert grade("gsm8k","1250.0", item.answer) and item.steps=="Work."
    assert not grade("gsm8k","some text 1250",item.answer)
    path.write_text(json.dumps([{"question":"Q","answer":"Sally","steps":["One.","Two."]}]))
    assert read_public("prosqa",path)[0].steps == "One.\nTwo."
    path.write_text(json.dumps([{"problem":"Q","answer":"2","solution":"S."}]))
    assert read_public("math500",path)[0].question == "Q"


def test_math_equivalence():
    pytest.importorskip("math_verify")
    assert grade("math500", "0.5", r"\frac{1}{2}")
    assert not grade("math500", "3", r"\frac{1}{2}")


def test_public_holdout_keeps_duplicate_questions_together(tmp_path):
    path = tmp_path/"source.json"
    path.write_text(json.dumps([{"question":v,"answer":"work\n#### 4"} for v in ("A","A","B","C")]))
    prepare_public("gsm8k", path, tmp_path/"split", 1, 0, 321)
    train_rows = read_public("gsm8k", tmp_path/"split/train.json")
    valid_rows = read_public("gsm8k", tmp_path/"split/valid.json")
    assert not {i.question for i in train_rows} & {i.question for i in valid_rows}
    assert len(train_rows)+len(valid_rows) == 4


def test_external_states_are_reused():
    class Adapter:
        def __init__(self):
            self.calls = []
        def encode(self, prefix):
            self.calls.append(prefix)
            return int(prefix)
        def answer(self, state, suffix):
            return str(state+int(suffix))
    adapter = Adapter()
    cases = [{"group":"group","prefixes":["2","3"],"suffixes":["4","5"],"probe":"7"}]
    output = predictions(cases, adapter)["group"]
    assert adapter.calls == ["2","3"]
    assert output == {"answers":[["6","7"],["7","8"]], "probe_answers":["9","10"]}


@pytest.mark.parametrize("mode", ["latent","direct","cot"])
def test_public_adaptation_reload(tmp_path, mode):
    cfg = Config(model=ModelSettings(hidden_size=16,layers=1,heads=2,latent_steps=2),
                 k=2,forward_batch=4,threads=1)
    torch.set_num_threads(1)
    model = CTModel(cfg.model)
    ckpt = tmp_path/"initial.pt"
    save_checkpoint(ckpt, {"config":asdict(cfg),"model":model.portable_state()})
    source = tmp_path/"train.jsonl"
    source.write_text('{"question":"Two plus two?","answer":"2+2=4\\n#### 4"}\n')
    target = tmp_path/"adapted.pt"
    adapt_public(ckpt,"gsm8k",source,target,mode,"cpu",1,1,4,1e-4)
    report = public_eval(target,"gsm8k",source,mode,"cpu",1,4)
    assert report["examples"] == 1 and report["pass_at_1"] in (0.,1.)
