from dataclasses import replace

import pytest
import torch

from responseguard.config import Config
from responseguard.data import (
    FACTORS,
    VALUES,
    Pair,
    Scene,
    composition_audit,
    load_pairs,
    make_dataset,
)
from responseguard.evaluate import diversity_metrics, vendi
from responseguard.flow import clipped_loss, likelihood_ratio_terms, rollout
from responseguard.judges import Evidence
from responseguard.objective import (
    GroupValue,
    History,
    basis_from_directions,
    group_value,
    residual_diversity,
    update_dual,
)
from responseguard.policy import make_policy
from responseguard.reference import ReferenceCache
from responseguard.judges import make_judge
from responseguard.train import train


def tiny():
    return replace(
        Config(),
        backend="tiny",
        device="cpu",
        judge_device="cpu",
        dtype="float32",
        resolution=32,
        k_ref=4,
        k_cal=2,
        train_steps=3,
        microbatch=4,
        iterations=3,
        groups_per_batch=1,
        lora_rank=2,
        lr=0.001,
        history_window=2,
        checkpoint_every=2,
    )


def pair():
    s = Scene("cup", "box", "red", 2, "to the left of")
    return Pair(s, s.neighbor("count"), "count", "train")


def test_exact_ratio_and_old_policy_gradient():
    torch.manual_seed(10)
    old = torch.randn(8, 7)
    action = old + 0.3 * torch.randn_like(old)
    mu = (old + torch.randn_like(old) * 0.01).requires_grad_()
    old_dist = torch.distributions.Normal(old, 0.3)
    new_dist = torch.distributions.Normal(mu, 0.3)
    log_ratio, kl = likelihood_ratio_terms(action, old, mu, 0.3)
    torch.testing.assert_close(
        log_ratio,
        (new_dist.log_prob(action) - old_dist.log_prob(action)).sum(-1),
        atol=2.0e-6,
        rtol=1.0e-5,
    )
    assert (kl >= 0).all()
    at_old = old.clone().requires_grad_()
    ratios, _ = likelihood_ratio_terms(action, old, at_old, 0.3)
    a = torch.full((8,), -1.5)
    (grad,) = torch.autograd.grad(clipped_loss(ratios, a, 0.1), at_old)
    (exact,) = torch.autograd.grad(
        -(a * torch.distributions.Normal(at_old, 0.3).log_prob(action).sum(-1)).mean(), at_old
    )
    torch.testing.assert_close(grad, exact)
    assert grad.norm() > 0  # a shared penalty survives broadcasting


def test_signed_ppo_matches_minimum():
    log_r = torch.tensor([-1.0, -0.05, 1.0, -1.0, 0.05, 1.0])
    a = torch.tensor([2.0, 2.0, 2.0, -2.0, -2.0, -2.0])
    r = log_r.exp()
    torch.testing.assert_close(
        clipped_loss(log_r, a, 0.1), -torch.minimum(r * a, r.clamp(0.9, 1.1) * a).mean()
    )


def test_matched_initial_independent_exploration_and_frozen_reference(tmp_path):
    cfg, item = tiny(), pair()
    policy = make_policy(cfg)
    a = rollout(policy, item.left, [3, 4], "a", 3, True).trace
    b = rollout(policy, item.left, [3, 4], "b", 3, True).trace
    torch.testing.assert_close(a.states[0], b.states[0])
    assert not torch.equal(a.states[1], b.states[1])
    judge = make_judge(cfg)
    ref = ReferenceCache(tmp_path / "cache1", cfg).get(policy, judge, item)
    with torch.no_grad():
        policy.module[-1].weight.add_(0.4)
    after = ReferenceCache(tmp_path / "cache2", cfg).get(policy, judge, item)
    torch.testing.assert_close(ref["margins"], after["margins"])
    assert ref["diversity"] == after["diversity"]


def test_lower_bounds_and_projection():
    cfg, item = tiny(), pair()
    x = torch.tensor([[0.0, 0.0], [4.0, 2.0], [9.0, 3.0], [4.0, 5.0]])
    u = basis_from_directions([torch.tensor([1.0, 0.0])], 2, 1)
    assert residual_diversity(x, torch.tensor([1, 1, 0, 0], dtype=torch.bool), u) == pytest.approx(
        4.0
    )
    own = torch.ones(4, 5)
    other = torch.zeros_like(own)
    # Exactly one valid sample in each branch. All other examples fail the color atom.
    own[1:, FACTORS.index("color")] = 0.0
    other[0, FACTORS.index("count")] = 0.7
    obs = Evidence(own, other, torch.ones(4), torch.zeros(4), x)
    ref = dict(
        anchors=torch.tensor([True, False, False, False]),
        margins=torch.ones(2, 4),
        bases=[u, u],
        cal_rates=[0.5, 0.5],
        diversity=[None, 4.0],
        valid_counts=[1, 2],
        verified_counts=[{}, {}],
    )
    result, diag, _ = group_value(item, [obs, obs], ref, [0, 1, 2, 3], cfg)
    assert result.sem == pytest.approx(2 * (0.7 - cfg.margin_slack) ** 2 / 4)
    assert result.div == pytest.approx((0.5 - 0.25) ** 2)
    assert all(p["distance_cost"] == 0 for p in diag["prompts"])
    assert all(p["rate_cost"] > 0 for p in diag["prompts"])


def test_metric_denominators_and_vendi():
    x = torch.eye(4)
    r = diversity_metrics(x, torch.tensor([True, True, False, False]), 2, [4, 3, 2, 1])
    assert (r["d_cond"], r["d_eff"], r["vendi"]) == pytest.approx((1.0, 1 / 6, 2.0))
    for mask in ([True, False, False, False], [False] * 4):
        r = diversity_metrics(x, torch.tensor(mask), 2, list(range(4)))
        assert r["d_cond"] is None and r["d_eff"] == 0 and r["vendi"] is None
    assert vendi(torch.ones(4, 3)) == pytest.approx(1.0)


def test_history_window_and_duals():
    h = History(2, 0.1)
    assert h.baseline("a", [0.0, 1.0]) == (0.0, 1.0)
    h.append("a", GroupValue(0.5, 0.2, 0.1))
    h.append("b", GroupValue(0.6, 0.3, 0.1))
    assert h.baseline("a", [1.0, 0.0])[0] == pytest.approx(0.3)
    assert h.baseline("a", [0.0, 1.0])[0] == pytest.approx(0.4)
    h.append("b", GroupValue(0.9, 0.2, 0.2))
    assert h.baseline("a", [0.0, 0.0])[0] == pytest.approx(0.75)
    cfg = tiny()
    dual = update_dual([0.2, 0.0], [GroupValue(0.5, 1.0, 0.0)], cfg, (True, True))
    assert dual[0] > 0.2 and dual[1] == 0.0


@pytest.mark.parametrize("seed", [1203, 7])
def test_full_dataset_and_projection_neighbors_stay_in_split(tmp_path, seed):
    expected = dict(train=8000, val=1000, test=1000)
    assert make_dataset(tmp_path, seed=seed) == expected
    seen = set()
    splits = {}
    triples = {}
    for split, size in expected.items():
        pairs = load_pairs(tmp_path / f"{split}.jsonl", {split})
        assert len(pairs) == size
        if split == "train":
            for factor, choices in VALUES.items():
                assert {getattr(s, factor) for p in pairs for s in p.scenes} == set(choices)
        for p in pairs:
            for s in p.scenes:
                assert s.text not in seen
                seen.add(s.text)
                for name in FACTORS:
                    for neighbor in (s, s.neighbor(name)):
                        assert splits.setdefault(neighbor, split) == split
                        triple = (neighbor.object, neighbor.color, neighbor.relation)
                        assert triples.setdefault(triple, split) == split
    assert len(triples) == len(VALUES["object"]) * len(VALUES["color"]) * len(VALUES["relation"])


def test_split_audit_detects_neighbor_even_when_observed_triples_differ():
    train_pair = pair()
    s = replace(train_pair.left, color="blue", landmark="basket")
    heldout = Pair(s, s.neighbor("count"), "count", "test")
    with pytest.raises(ValueError, match="including interventions"):
        composition_audit(dict(train=[train_pair], test=[heldout]))


def test_resume_replays_same_update_and_dual(tmp_path):
    torch.set_num_threads(1)
    cfg = tiny()
    items = [pair()]
    full = train(cfg, items, tmp_path / "full", tmp_path / "cache")
    train(cfg, items, tmp_path / "split", tmp_path / "cache", until=1)
    again = train(cfg, items, tmp_path / "split", tmp_path / "cache", resume=True)
    a, b = (torch.load(p, weights_only=True) for p in (full, again))
    assert a["iteration"] == b["iteration"] == 3
    assert a["dual"] == b["dual"] and a["history"] == b["history"]
    for k in a["adapter"]:
        torch.testing.assert_close(a["adapter"][k], b["adapter"][k], rtol=0, atol=0)
