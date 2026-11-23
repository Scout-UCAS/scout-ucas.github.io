import random
from pathlib import Path

import pytest
import torch

from ct_rl import comparison_learning as objectives
from ct_rl.comparisons import build_model, prefix_trace, stop_distribution
from ct_rl.curriculum import TestSet as Queries
from ct_rl.evaluation import evaluate
from ct_rl.experiment import load_checkpoint, train
from ct_rl.inference_budget import choose
from ct_rl.paper import rank_scores
from ct_rl.public import grade, parse_status, read_public
from ct_rl.public_data import prepared
from ct_rl.settings import Config, build_data
from ct_rl.tasks import generate_split, make_pair


def recipe(method):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    return Config.from_dict(dict(method=method,k=3,ltc_answers=2,forward_batch=4,min_thought_steps=1,
                            loop_depth=2,mc_draws=4,model=dict(hidden_size=24,heads=3,layers=1,
                            action_dim=4,latent_steps=3,max_answer_tokens=4)))


def test_first_stop_support_and_gradient():
    x = torch.zeros(2,4,requires_grad=True)
    mass = stop_distribution(x,2)
    torch.testing.assert_close(mass[0], torch.tensor([0.,.5,.25,.25]))
    (-mass[:,2].log().mean()).backward()
    assert x.grad[:,1:3].abs().min() > 0
    assert x.grad[:,[0,3]].count_nonzero() == 0


@pytest.mark.parametrize("method",["latent-grpo","slpo","ltc","rltt","loop-grpo"])
def test_credits_update_intended_parameters(method,monkeypatch):
    cfg = recipe(method)
    model = build_model(cfg)
    model.snapshot_reference()
    def answers(m,z,prompts,temperature):
        # A controlled reward pattern exercises learning even with a random decoder.
        words = ["1" if i == 0 else "0" for i in range(len(z))]
        y = objectives.labels_for(m,words)
        return y,words,torch.ones(len(words),dtype=torch.bool)
    monkeypatch.setattr(objectives,"sample_answers",answers)
    params = model.parameters_for("policy")+model.parameters_for("reader")
    opt = torch.optim.AdamW(params,lr=.002,weight_decay=0)
    reader_before = model.read_head.weight.detach().clone()
    result = objectives.comparison_pg(model,cfg,["prefix"],Queries([["suffix"]],[["1"]]),opt,params)
    assert result["adv_abs"]>0 and result["policy_grad_norm"]>0
    assert not torch.equal(reader_before,model.read_head.weight)
    if method in ("latent-grpo","ltc"):
        assert model.word_adapter.b.weight.grad.abs().sum() > 0
    if method == "slpo":
        assert model.gate.weight.grad.abs().sum()>0
    if method == "ltc":
        assert result["sampled_answers"] == cfg.k*cfg.ltc_answers
    if method in ("loop-grpo","rltt"):
        assert result["credited_loops"] == (cfg.loop_depth if method=="rltt" else 1)


def test_latent_grpo_preserves_full_trajectory_denominator(monkeypatch):
    cfg = recipe("latent-grpo")
    model = build_model(cfg)
    words = ["1","1","0"]
    labels = objectives.labels_for(model,words)
    def fixed(*args):
        return labels, words, torch.ones(3, dtype=torch.bool)
    monkeypatch.setattr(objectives,"sample_answers",fixed)
    objective, _ = objectives.one_group(model,cfg,"p","s","1",False,lambda p,y:p==y)
    advantage = objectives.masked_advantage(torch.tensor([1.,1.,0.]),torch.ones(3,dtype=torch.bool))
    # Initially every importance ratio is one. Exactly one correct first thought
    # is masked. Each sequence keeps its original T + number-of-answer-tokens divisor.
    expected = advantage[0]/(cfg.k*(cfg.model.latent_steps+labels.shape[1]))
    torch.testing.assert_close(objective(),expected,atol=1e-6,rtol=1e-5)


def test_gumbel_history_and_dropout():
    cfg=recipe("latent-grpo")
    model=build_model(cfg)
    with torch.no_grad():
        st=model.thoughts(["x","longer"],stochastic=True)
        old=model.policy_words(model.history_states(["x","longer"],st)).log_softmax(-1).gather(-1,st.ids)
        delta=st.perturbed-old
        assert delta.min()>=cfg.gumbel_delta-1e-5 and delta.max()<=4.5+cfg.gumbel_delta+1e-5
        torch.testing.assert_close(st.weights.sum(-1),torch.ones_like(delta[:,:,0]))
    model=build_model(recipe("slpo"))
    clean=model.thoughts(["input"],stochastic=False,full_gate=True)
    again=model.thoughts(["input"],stochastic=False,full_gate=True)
    torch.testing.assert_close(clean.raw,again.raw)
    noisy=model.thoughts(["input"],stochastic=True,full_gate=True)
    assert not torch.allclose(clean.raw,noisy.raw)
    assert torch.equal(noisy.interface()[:,noisy.length.item():],torch.zeros_like(noisy.interface()[:,noisy.length.item():]))


@pytest.mark.parametrize("method",["coconut","codi"])
def test_process_training_uses_prefix_execution(method):
    cfg=recipe(method)
    model=build_model(cfg)
    pair=make_pair("arithmetic","train",random.Random(1))
    trace=prefix_trace(pair,0)
    assert len(trace)==len(pair.operations)
    losses=[objectives.process_loss(model,[pair.prefix(0)],["sum"],["2"],[trace],f,False) for f in (0.,1.)]
    sum(losses).backward()
    assert model.down.weight.grad.abs().sum()>0
    if method=="codi":
        assert model.project[0].weight.grad.abs().sum()>0


def test_loop_incremental_cache_matches_full():
    model=build_model(recipe("rltt"))
    x,mask=model.prefix_inputs(["xy","qrst"])
    with torch.no_grad():
        full,_=model.loop_states(x,mask,"reader")
        _,cache=model.loop_states(x[:,:-1],mask[:,:-1],"reader",use_cache=True)
        last,_=model.loop_states(x[:,-1:],mask,"reader",cache,use_cache=True)
        for a,b in zip(full,last):
            torch.testing.assert_close(a[:,-1],b[:,-1],atol=1e-5,rtol=1e-4)


def test_text_prefix_cache_agrees_with_fresh_reader():
    cfg=recipe("direct")
    model=build_model(cfg)
    p=next(iter(generate_split("id",1,120)))
    ordinary=evaluate(model,[p],cfg,2,bootstrap=0)
    cached=evaluate(model,[p],cfg,2,bootstrap=0,serial=True,flops=True)
    assert ordinary["rows"]==cached["rows"]
    assert cached["costs"]["amortized_flops_per_query"]>0


def test_budget_and_selection_rules():
    base=dict(ct_acc=.5,pair_both=.125,all_correct=.25)
    ranking,_=rank_scores({"B":[base],"A":[dict(base,pair_both=.25)],"C":[base]})
    assert ranking==["A","B","C"]
    rows=[dict(ct_acc=.5,pair_both=.1,amortized_flops_per_query=c,tokens=n) for c,n in ((20,4),(10,2))]
    assert choose(rows,9) is None
    assert choose(rows,19)["tokens"]==2
    assert objectives.masked_advantage(torch.tensor([1.,0.,0.]),torch.tensor([False,True,True])).abs().sum()==0


def test_public_fixture_provenance_and_parsing():
    root=Path(__file__).resolve().parents[1]/"data/public-smoke"
    m=prepared(root,"smoke")
    assert len(m["splits"])==10
    for d in ("gsm8k","prontoqa","prosqa"):
        splits=[{i.question for i in read_public(d,root/d/(s+".json"))} for s in ("train","valid","test")]
        assert not (splits[0]&splits[1] or splits[0]&splits[2] or splits[1]&splits[2])
    assert list((root/"math500").glob("*.json"))==[root/"math500/test.json"]
    assert grade("gsm8k","2/4","0.5")
    assert grade("math500",r"\frac{2}{4}","0.5")
    assert parse_status("gsm8k",None,None)=="unfinished" and parse_status("gsm8k","?", "?")=="unparseable"
    assert parse_status("prontoqa","maybe","maybe")=="unparseable"
    assert parse_status("prosqa","Carol is a zumpus.","Carol is a zumpus.")=="parsed"


def test_comparison_resume_exact(tmp_path):
    cfg=recipe("ltc")
    cfg.warm_steps,cfg.steps_per_round,cfg.rounds,cfg.pairs_per_batch=2,1,2,1
    data=tmp_path/"data"
    build_data(data,2,1,120)
    uninterrupted=load_checkpoint(train(cfg,data,tmp_path/"a"))["model"]
    cfg.rounds=1
    train(cfg,data,tmp_path/"b")
    cfg.rounds=2
    resumed=load_checkpoint(train(cfg,data,tmp_path/"b",resume=True))["model"]
    for key in resumed:
        torch.testing.assert_close(uninterrupted[key],resumed[key],rtol=0,atol=0)
