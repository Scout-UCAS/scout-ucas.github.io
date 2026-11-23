"""Regression tests for the comparison methods and the paper's reporting protocol."""

import csv
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from responseguard.baselines import concentrated_subset, spectral_bonus, standardized
from responseguard.blind import blind_scores, blind_study
from responseguard.config import Config
from responseguard.data import Pair, Scene
from responseguard.evaluate import evaluate
from responseguard.flow import rollout, transition
from responseguard.io import read_rows, write_json
from responseguard.policy import make_policy
from responseguard.study import COSTS, budget_report, default_milestones
from scripts.public_eval import geneval_summary
from scripts.score_deltabench import score_manifest


def test_clustering_outlier_and_concentrated_subgroup():
    x = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.999, 0.001], [0.0, 1.0]])
    bonus = spectral_bonus(x, 2)
    assert bonus[-1] > bonus[:3].max()
    assert torch.allclose(spectral_bonus(torch.ones(4, 2), 2), torch.ones(4))
    points = torch.tensor([[0.0], [0.1], [0.2], [5.0], [20.0], [30.0]])
    assert set(concentrated_subset(points, 3)) == {0, 1, 2}
    rewards = torch.tensor([0.2, 0.5, 0.8, 0.4])
    torch.testing.assert_close(standardized(rewards - 0.3), standardized(rewards))


def test_drift_replays_exact_condition_after_subselection():
    cfg = Config.load(Path(__file__).resolve().parents[1] / "configs/smoke.json")
    policy = make_policy(cfg)
    scene = Scene("cup", "box", "blue", 2, "above")
    sample = rollout(policy, scene, [2, 5, 7, 9], ("actor", 3), 4, record=True, noisy_prompts=True)
    trace = sample.trace.select([3, 1])
    for t, old in enumerate(trace.means):
        for i in range(2):
            sl = slice(i, i + 1)
            x = trace.states[t][sl]
            sigma = trace.sigmas[t]
            cond = trace.condition(t, sl, cfg)
            mu, _ = transition(
                x,
                policy.velocity(x, sigma, cond),
                sigma,
                trace.sigmas[t + 1],
                trace.sigmas[1],
                cfg.sde_noise,
            )
            torch.testing.assert_close(old[sl], mu, atol=1e-6, rtol=1e-6)


def test_common_budget_never_uses_future_checkpoint(tmp_path):
    def point(n, c):
        return dict(iteration=n, cost=dict.fromkeys(COSTS, c), scores={"pair_sr": n / 10})

    points = {
        "a": [point(0, 0), point(1, 2), point(2, 5)],
        "b": [point(0, 0), point(1, 4), point(2, 8)],
    }
    matched = budget_report(points, tmp_path, figures=False)["denoiser_examples"]
    assert [
        (p["ceiling"], p["methods"]["a"]["iteration"], p["methods"]["b"]["iteration"])
        for p in matched
    ] == [(4, 1, 1), (5, 2, 1)]
    points["b"] = [point(0, 0), point(1, 6)]
    for line in points.values():
        for row in line:
            row["cost"]["gpu_hours"] = 0
    budget_report(points, tmp_path, figures=False)
    report = json.loads((tmp_path / "validation-budget.json").read_text())
    assert report["matched"]["denoiser_examples"] == []
    assert report["coverage"]["denoiser_examples"]["status"] == "no_trained_overlap"
    assert report["coverage"]["gpu_hours"]["status"] == "unavailable"
    assert report["points"]["a"][0]["iteration"] == 0


def test_main_schedule_measures_trained_joint_within_shared_budget():
    cfg = Config.load(Path(__file__).resolve().parents[1] / "configs/sd35.json")
    online = cfg.groups_per_batch * 2 * cfg.k_train
    reference = cfg.groups_per_batch * (10 * cfg.k_ref + 2 * cfg.k_cal)
    cost = online * (1 + cfg.ppo_epochs) * cfg.train_steps * 2
    cold_cost = cost + reference * cfg.train_steps * 2
    stops = default_milestones(cfg.iterations)
    assert sum(t * cold_cost <= cfg.iterations * cost for t in stops) >= 2
    assert sum(t * (online + reference) <= cfg.iterations * online for t in stops) >= 2


def test_public_scoring_preserves_zero_and_negative_answers(tmp_path):
    ge = geneval_summary(
        [dict(atom_count=2, skills=["object", "count"]), dict(atom_count=1, skills=["object"])],
        [[0.0, 1.0], [0.5]],
    )
    assert ge["soft_tifa_am"] == 0.5 and ge["soft_tifa_gm"] == 0.25
    with pytest.raises(ValueError):
        geneval_summary([dict(atom_count=2, skills=["object", "count"])], [[0.4]])
    Image.new("RGB", (8, 8)).save(tmp_path / "image.png")
    rows = [
        dict(
            category="color",
            pair=0,
            side=b,
            index=i,
            image="image.png",
            questions=[dict(question="Is it blue?", answer="yes" if b else "no")],
        )
        for b in range(2)
        for i in range(2)
    ]
    result = score_manifest(
        rows, tmp_path, tmp_path / "scores", lambda im, qs: [0.25], dict(model="unit-fixture")
    )
    assert result["macro_am"] == 0.5 and result["macro_gm"] == pytest.approx((0.75 * 0.25) ** 0.5)
    assert json.loads((tmp_path / "scores" / "summary.json").read_text())["model"] == "unit-fixture"


def test_blind_deblinding_and_missing_rater(tmp_path):
    cfg = Config.load(Path(__file__).resolve().parents[1] / "configs/tiny.json")
    scene = Scene("cup", "box", "blue", 2, "above")
    pair = Pair(scene, scene.neighbor("color"), "color", "test")
    paths = [tmp_path / "a", tmp_path / "b"]
    for i, path in enumerate(paths):
        evaluate(cfg, [pair], path)
        meta = json.loads((path / "evaluation.json").read_text())
        meta["model"] = f"method-{i}"
        write_json(path / "evaluation.json", meta)
    root = tmp_path / "blind"
    blind_study(paths, root, n=2, raters=3)
    assert blind_scores(root)["complete_ratings"] == 0
    keys = read_rows(root / "private" / "assignments.jsonl")
    for path in sorted((root / "forms").glob("*/ratings.csv")):
        rater = path.parent.name
        with path.open("w", newline="") as h:
            wr = csv.writer(h)
            wr.writerow(["item", "semantic", "quality", "diversity"])
            for k in keys:
                if k["rater"] == rater:
                    pick = "A" if k["method_a"] == 1 else "B"
                    wr.writerow([k["item"], pick, pick, "TIE"])
    scores = blind_scores(root)
    assert scores["complete_ratings"] == 6
    assert (
        scores["outcomes"]["semantic"]["method_1"] == 2
        and scores["outcomes"]["diversity"]["tie"] == 2
    )
