import itertools
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from ct_rl import paper_transfer as transfer, suite
from ct_rl.cli import main
from ct_rl.comparisons import build_model, prefix_trace
from ct_rl.evaluation import evaluate
from ct_rl.experiment import attention_backward_flops, attention_flops, meter, save_checkpoint
from ct_rl.model import ByteTokenizer, CTModel
from ct_rl.paper import Study, read
from ct_rl.public import adapt_public, grade, parse_status, public_eval, public_tokens, read_public
from ct_rl.readout import completion, extract
from ct_rl.settings import Config, LATENT_RL, PROCESS, build_data, digest, write_json
from ct_rl.tasks import READOUT, execute, generate_split


ROOT = Path(__file__).resolve().parents[1]


def tiny(method):
    torch.manual_seed(31)
    torch.set_num_threads(1)
    return Config.from_dict({"method":method, "threads":1, "k":2, "min_thought_steps":8,
                             "model":{"hidden_size":16, "heads":2, "layers":1, "action_dim":4,
                                      "latent_steps":8, "max_answer_tokens":64}})


def checkpoint(path, cfg, **metadata):
    model = build_model(cfg)
    save_checkpoint(path, {"config":asdict(cfg), "model":model.portable_state(), **metadata})
    return model


def force_output(monkeypatch, text):
    tokenizer = ByteTokenizer()
    ids = itertools.cycle(tokenizer.encode(text) + [tokenizer.eos_token_id])

    def logits(model, hidden):
        result = hidden.new_zeros((len(hidden), tokenizer.vocab_size))
        result[:, next(ids)] = 1
        return result

    monkeypatch.setattr(CTModel, "logits", logits)


def test_smoke_training_outputs_fit_including_rationale_and_eos(tmp_path):
    args = SimpleNamespace(profile="smoke", out=tmp_path, cache=None, public_data=None)
    study = Study(args)
    tokenizer = ByteTokenizer()
    for dataset in ("gsm8k", "prontoqa", "prosqa"):
        items = read_public(dataset, study.public/dataset/"train.json")
        for mode in ("latent", "native", "direct", "cot"):
            cap = transfer.settings_for(study, dataset, mode)["tokens"]
            for item in items:
                output = completion(item.answer, mode, item.steps)
                assert len(tokenizer.encode(output)) + 1 <= cap
                assert grade(dataset, extract(output, mode), item.answer)
    for pair in generate_split("train", 20, 120):
        for side in (0, 1):
            gold = execute(pair, side, (READOUT[pair.family],))
            output = completion(gold, "cot", "\n".join(prefix_trace(pair, side)))
            assert len(tokenizer.encode(output)) + 1 <= study.cfg(study.base, "cot", 31).model.max_answer_tokens


def test_standalone_and_suite_cot_budgets(tmp_path):
    source = ROOT/"configs/paper-smoke.json"
    assert Config.load(source, ["method=cot"]).model.max_answer_tokens == 128
    assert Config.load(ROOT/"configs/qwen1.5b.json", ["method=cot"]).model.max_answer_tokens == 256
    assert Config.load(source, ["model.max_answer_tokens=48", "method=cot"]).model.max_answer_tokens == 48
    data = tmp_path/"data"
    build_data(data, 1, 1, 120)
    run = tmp_path/"suite"
    suite.main(["--config", str(source), "--data", str(data), "--out", str(run), "--methods", "cot",
                "--seeds", "31", "--split", "id", "--m", "1", "--limit", "1", "--set", "warm_steps=1",
                "--set", "rounds=0"])
    assert read(run/"cot/31/config.json")["model"]["max_answer_tokens"] == 128


@pytest.mark.parametrize("mode", ["direct", "cot"])
def test_prosqa_gold_reaches_real_decoder_and_grader(tmp_path, monkeypatch, mode):
    ckpt = tmp_path/"model.pt"
    checkpoint(ckpt, tiny(mode))
    path = ROOT/"data/public-smoke/prosqa/test.json"
    for i, item in enumerate(read_public("prosqa", path)):
        source = tmp_path/f"{i}.json"
        write_json(source, [asdict(item)])
        force_output(monkeypatch, completion(item.answer, mode))
        result = public_eval(ckpt, "prosqa", source)
        assert result["max_answer_tokens"] == public_tokens("prosqa", mode, True)
        assert result["pass_at_1"] == 1 and result["parse_counts"]["parsed"] == 1
        force_output(monkeypatch, completion(item.answer, mode))
        short = public_eval(ckpt, "prosqa", source, tokens=12)
        assert short["pass_at_1"] == 0 and short["parse_counts"]["unfinished"] == 1


@pytest.mark.parametrize("mode", ["direct", "cot"])
def test_cli_and_controlled_text_inference_share_prefix_and_answer(tmp_path, monkeypatch, capsys, mode):
    cfg = tiny(mode)
    path = tmp_path/"model.pt"
    model = checkpoint(path, cfg)
    seen = []
    text_prefix = CTModel.native_text_prefix

    def record_prefix(self, prefix):
        seen.append(prefix)
        return text_prefix(self, prefix)

    def forbidden(*args, **kwargs):
        pytest.fail("A full-text checkpoint called the latent policy")

    monkeypatch.setattr(CTModel, "sample", forbidden)
    monkeypatch.setattr(CTModel, "native_text_prefix", record_prefix)
    force_output(monkeypatch, completion("-7", mode, "work"))
    pair = next(iter(generate_split("id", 1, 120)))
    main(["infer", "--checkpoint", str(path), "--prefix", pair.prefix(0), "--suffix", "left",
          "--suffix", "right"])
    got = json.loads(capsys.readouterr().out)
    assert seen == [pair.prefix(0)]
    assert got["mode"] == mode and got["latent_shape"][1] == 0
    rows = evaluate(model, [pair], cfg, m=2, bootstrap=0, serial=True)["rows"]
    assert rows[0]["answers"][0] == got["predictions"] == ["-7", "-7"]


def test_cli_uses_public_native_checkpoint_metadata(tmp_path, monkeypatch, capsys):
    from ct_rl.comparisons import ThoughtModel
    path = tmp_path/"adapted.pt"
    checkpoint(path, tiny("coconut"), public={"mode":"native"})
    prompts = []
    original = ThoughtModel.reader_inputs

    def capture(self, state, suffixes):
        prompts.extend(suffixes)
        return original(self, state, suffixes)

    monkeypatch.setattr(ThoughtModel, "reader_inputs", capture)
    force_output(monkeypatch, "3")
    main(["infer", "--checkpoint", str(path), "--prefix", "p", "--suffix", "s1", "--suffix", "s2"])
    row = json.loads(capsys.readouterr().out)
    assert row["mode"] == "native" and row["predictions"] == ["3", "3"]
    assert prompts == [("p","s1"), ("p","s2")]


def test_legacy_cot_checkpoint_can_override_short_saved_cap(tmp_path, monkeypatch, capsys):
    cfg = tiny("cot")
    cfg.model.max_answer_tokens = 12
    path = tmp_path/"legacy.pt"
    checkpoint(path, cfg)
    args = ["infer", "--checkpoint", str(path), "--prefix", "p", "--suffix", "s"]
    force_output(monkeypatch, completion("12", "cot"))
    main(args)
    assert json.loads(capsys.readouterr().out)["predictions"] == [None]
    force_output(monkeypatch, completion("12", "cot"))
    main([*args, "--tokens", "32"])
    assert json.loads(capsys.readouterr().out)["predictions"] == ["12"]


@pytest.mark.parametrize("method", PROCESS + LATENT_RL)
def test_deterministic_cache_matches_full_recomputation(method):
    model = build_model(tiny(method))
    prefixes = ["x"*98, "short"]
    reference = model.thoughts(prefixes, stochastic=False)
    assert model.work["policy_padded_tokens"] == 2 * sum(range(100, 108))
    model.work.clear()
    with torch.no_grad():
        cached = model.thoughts(prefixes, stochastic=False)
    assert model.work["policy_padded_tokens"] == 2 * 107 and model.peak_cache_bytes > 0
    torch.testing.assert_close(cached.raw, reference.raw, atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(cached.interface(), reference.interface(), atol=2e-5, rtol=1e-4)
    if method == "slpo":
        model.work.clear()
        with torch.no_grad():
            model.thoughts(prefixes, stochastic=True)
        assert model.work["policy_padded_tokens"] == 2 * sum(range(100, 108))


@pytest.mark.parametrize("query_length,key_length,causal", [(100,100,True), (1,107,False)])
def test_cpu_fused_attention_counts_forward_and_backward(query_length, key_length, causal):
    b, h, d = 1, 4, 8
    q = torch.randn(b,h,query_length,d,requires_grad=True)
    k, v = [torch.randn(b,h,key_length,d,requires_grad=True) for _ in range(2)]
    fwd, bwd = {}, {}
    with meter(q.device, True, fwd) as counter:
        output = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    assert torch.ops.aten._scaled_dot_product_flash_attention_for_cpu in counter.get_flop_counts()["Global"]
    with meter(q.device, True, bwd):
        output.sum().backward()
    unit = b*h*query_length*key_length*d
    assert fwd["operator_flops"] == 4*unit and bwd["operator_flops"] == 10*unit
    math_fwd = {}
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH), meter(q.device, True, math_fwd):
        F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    assert math_fwd["operator_flops"] == fwd["operator_flops"]


def test_gsm8k_grader_and_format_status_agree():
    for raw in ("1,2", ",12,,", "12,", "1,23,456", "12.3,4", "1/0", "12,00/100"):
        assert parse_status("gsm8k", raw, raw) == "unparseable"
        assert not grade("gsm8k", raw, "12") and not grade("gsm8k", raw, raw)
    for raw, gold in [("-1,234.50","-1234.5"), ("+12,000/1,000","12"), ("0","0.0"),
                      ("6/12","0.5"), ("١٢","12")]:
        assert parse_status("gsm8k", raw, raw) == "parsed" and grade("gsm8k", raw, gold)


def test_overshoot_recovery_and_prediction_cache(tmp_path):
    study = Study(SimpleNamespace(profile="smoke", out=tmp_path, cache=None, public_data=tmp_path/"public"))
    cfg = tiny("direct")
    src = study.path("matched", "direct", 31)/"checkpoint.pt"
    checkpoint(src, cfg)
    train = study.public/"gsm8k"/"train.json"
    for split in ("train", "valid"):
        write_json(train.with_name(split+".json"), [{"question":"Six plus six?", "answer":"12"}])
    settings = transfer.settings_for(study, "gsm8k", "direct")
    pilot = adapt_public(src, "gsm8k", train, tmp_path/"pilot.pt", "direct", steps=3,
                         save_model=False, **settings)
    # Two warm steps plus one optimizer step is the first admissible state.
    first = sum(pilot["step_flops"][:3])
    study.public_targets = {"gsm8k":first/1.11}
    args = (study, "direct", "direct", "direct", 31, "gsm8k", True)
    run = tmp_path/"transfer/from-synthetic/direct/31/gsm8k"
    with pytest.raises(transfer.PublicBudgetExceeded):
        transfer.transferred(*args)
    assert list(run.iterdir()) == []
    # Recover an old implementation's over-budget final checkpoint, even without its receipt.
    ckpt = run/"adapted.pt"
    adapt_public(src, "gsm8k", train, ckpt, "direct", steps=3, **settings)
    invalid_hash = digest(ckpt)
    study.public_targets["gsm8k"] = first
    row = transfer.transferred(*args)
    accepted_hash = digest(ckpt)
    assert row["training_flops"] == first and accepted_hash != invalid_hash
    assert read(run/"valid.json")["checkpoint_sha256"] == accepted_hash
    (run/"training.json").unlink()
    transfer.transferred(*args)
    assert read(run/"training.json")["checkpoint_sha256"] == accepted_hash == digest(ckpt)
    study.public_targets["gsm8k"] = sum(pilot["step_flops"][:4])
    transfer.transferred(*args)
    assert read(run/"valid.json")["checkpoint_sha256"] == digest(ckpt) != accepted_hash


def test_budget_increase_rechecks_all_arms_before_test(tmp_path, monkeypatch):
    study = SimpleNamespace(out=tmp_path, secondary_seeds=[31])
    datasets = ("gsm8k", "prontoqa", "prosqa")
    monkeypatch.setattr(transfer, "calibrate", lambda *_: dict.fromkeys(datasets, 100))
    monkeypatch.setattr(transfer, "UNIFIED", ("ct-rl", "latent-sft"))
    for name in ("TEXT", "PROCESS", "LATENT_RL"):
        monkeypatch.setattr(transfer, name, ())
    calls = []

    def simulate(study, tag, method, mode, seed, dataset, synthetic, split=None):
        target = study.public_targets[dataset]
        calls.append((dataset, tag, synthetic, split, target))
        if dataset == "gsm8k" and tag == "latent-sft" and not synthetic and target == 100:
            raise transfer.PublicBudgetExceeded(tag, dataset, 115, target)
        return dict(method=tag, dataset=dataset, seed=seed, split=split or "valid",
                    synthetic=synthetic, pass_at_1=0., target_flops=target)

    monkeypatch.setattr(transfer, "transferred", simulate)
    transfer.transfer(study)
    assert read(tmp_path/"public-budget-targets.json") == {"gsm8k":120,"prontoqa":100,"prosqa":100}
    assert all(row["target_flops"] == 120 for row in read(tmp_path/"public-results.json") if row["dataset"] == "gsm8k")
    for tag in ("ct-rl", "latent-sft"):
        assert ("gsm8k", tag, True, None, 100) in calls and ("gsm8k", tag, True, None, 120) in calls
    assert all(cost == 120 for d, _, _, split, cost in calls if d == "gsm8k" and split is not None)


def test_grouped_attention_cost_counts_query_heads():
    q, k, v = (2, 12, 7, 128), (2, 2, 13, 128), (2, 2, 13, 64)
    dots = 2 * 12 * 7 * 13
    assert attention_flops(q, k, v) == dots * (2*128 + 2*64)
    assert attention_backward_flops((2,12,7,64), q, k, v) == dots * (6*128 + 4*64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlashAttention")
def test_cuda_fused_grouped_attention_cost():
    from torch.nn.attention import sdpa_kernel, SDPBackend
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("FlashAttention requires an Ampere or newer CUDA device")
    q = torch.randn(1,12,16,128,device="cuda",dtype=torch.bfloat16,requires_grad=True)
    k, v = [torch.randn(1,2,16,128,device="cuda",dtype=torch.bfloat16,requires_grad=True) for _ in range(2)]
    forward, backward = {}, {}
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION), meter(q.device, True, forward):
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=True)
    with meter(q.device, True, backward):
        y.sum().backward()
    unit = 12*16*16*128
    assert forward["operator_flops"] == 4*unit
    assert backward["operator_flops"] == 10*unit


def test_all_backbone_eos_tokens_finish_greedy_and_sampled_answers(monkeypatch):
    from ct_rl import model as architecture
    from ct_rl.comparison_learning import sample_answers

    original = architecture.Qwen2ForCausalLM
    def backbone(cfg):
        lm = original(cfg)
        lm.generation_config.eos_token_id = [2,5]
        return lm

    monkeypatch.setattr(architecture, "Qwen2ForCausalLM", backbone)
    model = build_model(tiny("ct-rl"))
    assert model.eos == 2 and model.stop_tokens == {2,5}
    one, two = model.tokens("12")
    def force():
        outputs = iter(([one,one], [5,two], [two,2]))
        def logits(h):
            choices = torch.tensor(next(outputs))[:,None]
            scores = h.new_full((2,model.tokenizer.vocab_size), -torch.inf)
            return scores.scatter_(1,choices,0)
        monkeypatch.setattr(model, "logits", logits)

    z = model.embed([one])[None,:0].expand(2,-1,-1)
    force()
    assert model.decode(z, ["s","t"], 3) == ["1","12"]
    force()
    labels, answers, done = sample_answers(model, z, ["s","t"], 1)
    assert answers == ["1","12"] and done.all()
    assert labels.tolist() == [[one,5,-100], [one,two,2]]


def test_grpo_surrogate_finite_on_extreme_ratios():
    from ct_rl.learning import clipped
    delta = torch.tensor([1000., 1000., -1000., 0.], requires_grad=True)
    advantage = torch.tensor([1., -1., -1., 2.])
    objective = clipped(delta, torch.zeros_like(delta), advantage, .2)
    objective.sum().backward()
    assert torch.isfinite(objective).all() and torch.isfinite(delta.grad).all()
    torch.testing.assert_close(objective[[0,2,3]], torch.tensor([1.2,-.8,2.]))
    assert delta.grad[-1] == 2


def test_evaluation_cache_requires_same_data_and_request(tmp_path, monkeypatch):
    from ct_rl import paper
    study = Study(SimpleNamespace(profile="smoke", out=tmp_path, cache=None, public_data=None))
    build_data(study.data, 1, 1, 120)
    run = tmp_path/"arm"
    checkpoint(run/"checkpoint.pt", tiny("direct"))
    first, = study.evaluate(run, "comparison", ["id"], counts=(1,))
    second, = study.evaluate(run, "comparison", ["valid"], counts=(4,), measure=True)
    assert (first["split"],first["m"],first["costs"]["operator_flops"]) == ("id",1,None)
    assert second["split"] == "valid" and second["m"] == 4 and second["costs"]["operator_flops"]
    old_hash = read(run/"comparison.json")["dataset_sha256"]
    study.data = tmp_path/"other-data"
    build_data(study.data, 1, 1, 121)
    study.evaluate(run, "comparison", ["valid"], counts=(4,), measure=True)
    assert read(run/"comparison.json")["dataset_sha256"] != old_hash
    monkeypatch.setattr(paper, "evaluate_run", lambda *args: pytest.fail("identical evaluation ran twice"))
    study.evaluate(run, "comparison", ["valid"], counts=(4,), measure=True)
