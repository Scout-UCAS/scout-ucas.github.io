import random

import pytest
import torch

from ct_rl.curriculum import TestSet as SuffixTests, choose_hard, make_tests
from ct_rl.evaluation import evaluate
from ct_rl.experiment import load_checkpoint, train
from ct_rl.learning import group_update, leave_one_out, supervised_step
from ct_rl.model import CTModel
from ct_rl.settings import Config, ModelSettings, build_data
from ct_rl.tasks import (SPLITS, READOUT, Oracle, Pair, composition_ood, execute,
                         generate_split, make_pair, random_suffix, separating_read)


@pytest.fixture
def model():
    torch.manual_seed(13)
    torch.set_num_threads(1)
    return CTModel(ModelSettings(hidden_size=24, layers=1, heads=2, action_dim=4, latent_steps=3))


def test_interpreters():
    a = Pair("arithmetic", {}, [[2,3],[1,4]], [], "train")
    assert [execute(a, i, ("sum",)) for i in (0,1)] == ["5","5"]
    assert [execute(a, i, ("diff",)) for i in (0,1)] == ["-1","-3"]
    g = Pair("graph", {"L":[1,2,0],"R":[2,0,1]}, [0,2], ["L"], "train")
    assert execute(g, 0, ("L","node")) == "2"
    assert execute(g, 1, ("R","parity")) == "0"
    r = Pair("rules", {"n":6,"edges":[[0,1],[1,2],[3,4]]}, [[0],[3]], [], "train")
    assert execute(r, 0, ("has 2",)) == "1"
    assert execute(r, 1, ("add 0","count")) == "5"


def test_splits_and_ood_conditions():
    seen = set()
    rng = random.Random(11)
    for split in SPLITS:
        pairs = list(generate_split(split, 4, 42))
        for p in pairs:
            assert p.group not in seen
            seen.add(p.group)
            assert execute(p, 0, (READOUT[p.family],)) == execute(p, 1, (READOUT[p.family],))
            probe = separating_read(p)
            assert execute(p, 0, probe) != execute(p, 1, probe)
            for _ in range(12):
                s = random_suffix(p, rng)
                assert composition_ood(p.family, s) == (split == "comp")
                assert (len(s) >= 5) == (split == "suffix_length")


def test_likelihood_replay_matches_autoregression(model):
    prefixes = ["a", "a much longer prefix"]
    with torch.no_grad():
        actions, z = model.sample(prefixes)
        x, mask = model.prefix_inputs(prefixes)
        by_step = []
        for i in range(actions.shape[1]):
            h, _ = model._hidden(x, mask, "policy")
            mu, sigma = model.stats(h[:, -1], "policy")
            by_step.append(torch.distributions.Normal(mu, sigma).log_prob(actions[:, i]).sum(-1))
            x = torch.cat((x,z[:,i:i+1]), 1)
            mask = torch.nn.functional.pad(mask, (0,1), value=1)
        expected = torch.stack(by_step, 1)
        model.snapshot_reference()
        actual, _ = model.replay(prefixes, actions)
        ref, _ = model.replay(prefixes, actions, "reference")
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual, ref, rtol=0, atol=0)


def test_nll_teacher_forcing_and_reader_isolation(model):
    prompts, answers = ["node", "add 3 ; count"], ["1", "-23"]
    with torch.no_grad():
        _, z = model.sample(["prefix A","prefix B"])
        loss = model.answer_loss(z, prompts, answers)
        manual = 0.
        for i, answer in enumerate(answers):
            x, mask = model.reader_inputs(z[i:i+1], prompts[i:i+1])
            for token in model.tokens(answer) + [model.eos]:
                h, _ = model._hidden(x, mask, "reader")
                manual -= model.logits(h[:, -1]).log_softmax(-1)[0, token].item()
                x = torch.cat((x, model.embed([token])[None]), 1)
                mask = torch.nn.functional.pad(mask, (0,1), value=1)
        assert loss.item() == pytest.approx(manual/2, rel=1e-5)
        before = model.decode(z, prompts)
        model.sample(["a completely unrelated prefix"])
        assert model.decode(z, prompts) == before


@pytest.mark.parametrize("algorithm", ["rloo","grpo"])
def test_roles_and_signed_policy_update(model, monkeypatch, algorithm):
    cfg = Config(model=model.cfg, algorithm=algorithm, k=8, beta=0, forward_batch=32)
    cfg.grpo_epochs = 2
    policy, reader = model.parameters_for("policy"), model.parameters_for("reader")
    model.snapshot_reference()
    # Inject deterministic binary rewards varying across candidates. This makes
    # the isolation check meaningful even for an untrained answer decoder.
    def answers(z, prompts, max_tokens=None):
        v = z[:,0,0]
        return ["1" if x else "0" for x in (v > v.median()).tolist()]
    monkeypatch.setattr(model, "decode", answers)
    tests = SuffixTests([["read"],["read"]], [["1"],["1"]])
    reader_before = [p.clone() for p in reader]
    head_before = model.policy_head.weight.detach().clone()
    ref_before = model.ref_head.weight.detach().clone()
    g_before = model.mapper.weight.detach().clone()
    actor_opt = torch.optim.AdamW(policy, lr=1e-3, weight_decay=0)
    _, log = group_update(model, ["x","yy"], lambda: tests, cfg, actor_opt)
    assert log["adv_abs"] > 0
    assert not torch.equal(model.policy_head.weight, head_before)
    assert all(torch.equal(p,q) for p,q in zip(reader, reader_before))
    assert torch.equal(model.mapper.weight, g_before)
    assert torch.equal(model.ref_head.weight, ref_before)
    actor_after = [p.clone() for p in policy]
    optimizer = torch.optim.AdamW(reader, lr=1e-3, weight_decay=0)
    supervised_step(model, ["x","yy"], tests, optimizer, reader, cfg, joint=False)
    assert all(torch.equal(p,q) for p,q in zip(policy, actor_after))
    assert any(not torch.equal(p,q) for p,q in zip(reader, reader_before))


def test_baseline_and_search_selection():
    got = leave_one_out(torch.tensor([[0.,0.5,1.]]))
    torch.testing.assert_close(got, torch.tensor([[-0.75,0.,0.75]]))
    entries = [{"suffix":["sum"],"failure":1.,"gold":["5","5"]},
               {"suffix":["left"],"failure":1.,"gold":["2","1"]},
               {"suffix":["diff"],"failure":0.,"gold":["-1","-3"]}]
    assert choose_hard(entries, 1, False)[0]["suffix"] == ["left"]
    assert choose_hard(entries, 1, True)[0]["suffix"] == ["sum"]
    pair = make_pair("arithmetic", "train", random.Random(5))
    out = make_tests([pair], 4, {pair.group:[{"suffix":["diff"]}]}, 1., random.Random(1), Oracle())
    assert out.hard_draws == 4 and out.suffixes[0] == out.suffixes[1] == ["diff"]*4
    assert out.answers[0] != out.answers[1]


def test_evaluation_reuses_one_state_per_side(model, monkeypatch):
    pair = make_pair("graph", "id", random.Random(3))
    prefixes_seen = []
    def latent(prefixes, **kwargs):
        prefixes_seen.extend(prefixes)
        return None, torch.tensor([[[float([pair.prefix(0), pair.prefix(1)].index(p))]] for p in prefixes])
    def perfect(z, prompts, max_tokens=None):
        return [execute(pair, int(row.item()), tuple(s.split(" ; "))) for row,s in zip(z,prompts)]
    monkeypatch.setattr(model, "sample", latent)
    monkeypatch.setattr(model, "decode", perfect)
    report = evaluate(model, [pair], Config(model=model.cfg), m=4, bootstrap=4)
    assert prefixes_seen == [pair.prefix(0), pair.prefix(1)]
    assert report["macro"] == {"ct_acc":1.,"all_correct":1.,"pair_both":1.,"invalid_rate":0.}
    swapped = evaluate(model, [pair], Config(model=model.cfg), mode="swap", bootstrap=0)
    assert swapped["macro"]["pair_both"] == 0


def test_checkpoint_resume_preserves_curriculum_and_model(tmp_path, model):
    data = tmp_path/"data"
    build_data(data, 2, 1, 120)
    cfg = Config(model=model.cfg, warm_steps=2, rounds=2, steps_per_round=1,
                 pairs_per_batch=1, k=2, m=2, search_budget=4, mine_samples=1, mine_pairs=2, threads=1)
    whole = load_checkpoint(train(cfg, data, tmp_path/"whole"))
    cfg.rounds = 1
    train(cfg, data, tmp_path/"restart")
    cfg.rounds = 2
    resumed = load_checkpoint(train(cfg, data, tmp_path/"restart", resume=True))
    assert whole["bank"] == resumed["bank"]
    assert resumed["history"][1]["pool_before"] == 0
    assert resumed["history"][2]["pool_before"] == resumed["history"][1]["pool_after"] > 0
    assert all(torch.equal(value, resumed["model"][name]) for name,value in whole["model"].items())


def test_local_pretrained_backend(tmp_path, model):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    folder = tmp_path/"qwen"
    # Save a clean Qwen architecture checkpoint, as a local stand-in for HF weights.
    from transformers import Qwen2ForCausalLM
    Qwen2ForCausalLM(model.lm.config).save_pretrained(folder)
    vocab = {"<pad>":0, "<bos>":1, "<eos>":2, "<unk>":3, "x":4}
    vocab.update({f"word{i}":i for i in range(5, model.lm.config.vocab_size)})
    codec = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    codec.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=codec, bos_token="<bos>", eos_token="<eos>",
                                        pad_token="<pad>", unk_token="<unk>")
    tokenizer.save_pretrained(folder)
    cfg = ModelSettings(backbone=str(folder), action_dim=4, latent_steps=2)
    first = CTModel(cfg)
    first.snapshot_reference()
    with torch.no_grad():
        actions, z = first.sample(["x"])
        assert actions.shape == (1,2,4)
        gold_nll = first.answer_loss(z, ["x"], ["x"]).item()
    second = CTModel(cfg)
    second.restore(first.portable_state())
    assert second.answer_loss(z, ["x"], ["x"]).item() == pytest.approx(gold_nll)
