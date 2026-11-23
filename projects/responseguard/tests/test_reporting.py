"""Counterexamples for checkpoint aggregation and official score re-analysis."""

from dataclasses import asdict
import json
import math
import sys

import pytest
import torch

from responseguard.compare import aggregate, aggregate_methods
from responseguard.config import Config
from responseguard.data import Pair, Scene
from responseguard.evaluate import evaluation_protocol
from responseguard.io import digest, write_json, write_rows
from responseguard.study import scored
from responseguard.train import training_protocol
from scripts.public_eval import main as public_main


def make_eval(root, method, seed, distances, vendi, iteration=4):
    folder = root / method / str(seed)
    trained = seed is not None
    protocol = training_protocol({"lr": 0.001, "seed": seed}, method, "train-hash", True)
    write_json(
        folder / "evaluation.json",
        dict(
            protocol="fixed-test",
            model=method,
            training_seed=seed,
            checkpoint=[f"run-{method}-{seed}", iteration] if trained else None,
            training_protocol=protocol if trained else None,
        ),
    )
    prompts = [
        dict(
            k=8,
            n_valid=4 if v is not None else 2 if d is not None else 1,
            rho=0.5,
            quality_rate=1.0,
            semantic_rate=0.5,
            preference=0.5,
            d_cond=d,
            vendi=v,
            d_eff=0.1,
            aux_d_cond=d,
        )
        for d, v in zip(distances, vendi, strict=True)
    ]
    write_rows(
        folder / "pairs.jsonl",
        [
            dict(
                pair="one-pair",
                k=8,
                ref_correct=4,
                pair_correct=4,
                retained=4,
                corrected=0,
                direction_correct=4,
                unchanged_pass=4,
                prompts=prompts,
            )
        ],
    )
    write_json(folder / "summary.json", {})
    return folder


def test_cross_method_and_seed_support_uses_same_prompt_ids(tmp_path):
    a = [make_eval(tmp_path, "a", s, [v, 0.9], [1.5, 4.0]) for s, v in [(17, 0.2), (42, 0.4)]]
    b = [make_eval(tmp_path, "b", s, [0.3, None], [None, None]) for s in (17, 42)]
    ref = make_eval(tmp_path, "reference", None, [0.2, 0.3], [1.5, 2.0])
    result = aggregate_methods([*a, *b, ref])
    x, y = result["a"], result["b"]
    assert aggregate(a, [*b, ref]) == x
    assert x["common_support"] == y["common_support"]
    assert x["common_support"]["d_cond"]["coverage"] == 0.5
    assert x["common_support"]["vendi"]["n_common"] == 0
    assert x["metrics"]["vendi_common"] == dict(mean=None, sd=None, n_defined=0)
    assert x["metrics"]["d_cond_common"]["mean"] == pytest.approx(0.3)
    assert x["metrics"]["d_cond_common"]["sd"] == pytest.approx(math.sqrt(0.02))
    assert x["metrics"]["d_cond"]["mean"] == pytest.approx(0.6)
    assert y["metrics"]["d_cond_common"]["mean"] == pytest.approx(0.3)
    assert len(x["common_support"]["members"]) == 5
    assert aggregate(a)["common_support"]["d_cond"]["coverage"] == 1


@pytest.mark.parametrize("change", ["iteration", "data", "lr", "baseline", "missing", "seed"])
def test_reject_incompatible_aggregate(tmp_path, change):
    paths = [make_eval(tmp_path, "joint", s, [0.2, 0.3], [1.0, 2.0]) for s in (17, 42, 73)]
    file = paths[0] / "evaluation.json"
    meta = json.loads(file.read_text())
    if change == "iteration":
        meta["checkpoint"][1] = 1
    elif change == "data":
        meta["training_protocol"]["training_data"] = "other-data"
    elif change == "lr":
        meta["training_protocol"]["config"]["lr"] *= 2
    elif change == "baseline":
        meta["training_protocol"]["history_baseline"] = False
    elif change == "missing":
        del meta["training_protocol"]
    else:
        meta["training_seed"] = 42
    write_json(file, meta)
    with pytest.raises(
        ValueError, match="iteration|training protocols|provenance|distinct training seeds"
    ):
        aggregate(paths)


def test_resumed_study_checks_cached_model_and_iteration(tmp_path):
    cfg = Config()
    scene = Scene("cup", "box", "red", 1, "above")
    pairs = [Pair(scene, scene.neighbor("color"), "color", "val")]
    folder = make_eval(tmp_path, "joint", 17, [0.2, 0.3], [1.1, 1.3])
    meta = json.loads((folder / "evaluation.json").read_text())
    meta.update(metric_space="fixture", protocol=evaluation_protocol(cfg, pairs, "fixture"))
    write_json(folder / "evaluation.json", meta)
    write_json(folder / "summary.json", {"all": {"rho": 0.25}})
    weights = tmp_path / "adapter.pt"
    torch.save(dict(identity=meta["checkpoint"][0], iteration=4), weights)
    assert scored(cfg, pairs, folder, weights, ("joint", 17, 4))["rho"] == 0.25
    with pytest.raises(ValueError, match="training iteration"):
        scored(cfg, pairs, folder, weights, ("joint", 17, 1))
    torch.save(dict(identity="other-run", iteration=4), weights)
    with pytest.raises(ValueError, match="different checkpoint"):
        scored(cfg, pairs, folder, weights, ("joint", 17, 4))


def test_archived_evaluation_recovers_protocol_from_matching_study(tmp_path):
    seeds = [17, 42]
    paths = []
    base = dict(seed=42, lr=0.001)
    write_json(
        tmp_path / "plan.json",
        dict(
            config=base,
            rates={"joint": 0.002},
            training_data="old-data",
            seeds=seeds,
            methods=["joint"],
        ),
    )
    for s in seeds:
        src = make_eval(tmp_path, "joint", s, [0.3, 0.2], [1.0, 1.2])
        run = tmp_path / "joint" / f"seed-{s}"
        run.mkdir()
        dest = src.rename(run / "eval")
        meta = json.loads((dest / "evaluation.json").read_text())
        del meta["training_protocol"]
        write_json(dest / "evaluation.json", meta)
        write_json(
            run / "run.json",
            dict(
                config=dict(base, seed=s, lr=0.002),
                identity=meta["checkpoint"][0],
                variant="joint",
                history_baseline=True,
            ),
        )
        paths.append(dest)
    result = aggregate(paths)
    assert result["iteration"] == 4 and result["training_protocol"]["training_data"] == "old-data"
    assert result["training_protocol"]["config"]["lr"] == 0.002


def test_scores_only_rebuilds_public_means_and_intervals_without_models(tmp_path, monkeypatch):
    # Two model families and a frozen base, with cached 800-prompt atom scores.
    cfg = asdict(Config())
    study, repo, out = [tmp_path / p for p in ("study", "geneval", "public")]
    methods, seeds = ["flow-grpo", "joint"], [17, 42]
    plan = dict(config=cfg, evaluation=cfg, methods=methods, seeds=seeds, training_data="train")
    write_json(study / "plan.json", plan)
    records = [
        dict(prompt=f"p{i}", atom_count=n, skills=["object"] * n)
        for i, n in enumerate([3] * 400 + [9] * 400)
    ]
    write_rows(repo / "geneval2_data.jsonl", records)
    ge = digest(records)
    write_json(
        out / "plan.json",
        dict(
            config=cfg,
            study=digest(plan),
            methods=methods,
            geneval=ge,
            delta="delta",
            creative="creative",
        ),
    )
    final = cfg["iterations"]
    tasks = [("reference", None)] + [(m, s) for m in methods for s in seeds]
    for method, seed in tasks:
        identity = None
        if seed is not None:
            run = study / method / f"seed-{seed}"
            run_id = digest((method, seed))
            identity = dict(run=run_id, iteration=final)
            write_json(
                run / "run.json",
                dict(
                    identity=run_id,
                    config=dict(cfg, seed=seed),
                    training_data="train",
                    variant=method,
                    history_baseline=True,
                ),
            )
            write_json(
                run / "eval" / "evaluation.json",
                dict(checkpoint=[run_id, final], training_seed=seed, model=method),
            )
            write_json(run / "eval" / "summary.json", {})
        root = out / method / (str(seed) if seed is not None else "base")
        for suite, field, source in (
            ("geneval2", "prompt_list", ge),
            ("deltabench", "prompt_list", "delta"),
            ("creative", "prompts", "creative"),
        ):
            write_json(
                root / suite / "generation.json",
                dict(benchmark=suite, config=cfg, checkpoint=identity, **{field: source}),
            )
        write_json(
            root / "geneval2" / "scores.json",
            [[int(method == "joint" and i < 400)] * r["atom_count"] for i, r in enumerate(records)],
        )
        write_json(root / "deltabench/scores/summary.json", dict(macro_am=1, macro_gm=1))
        write_rows(
            root / "deltabench/scores/image_scores.jsonl",
            [
                dict(category="color", pair=0, side=b, index=i, score=1)
                for b in (0, 1)
                for i in (0, 1)
            ],
        )
        creative = [
            dict(
                prompt=f"c{i}",
                seeds=[1, 2, 3, 4],
                d_cond=None if i == 0 else 0.3,
                vendi=None if i < 2 else 1.4,
                rho=1.0,
                d_eff=0.3,
                preference=0.5,
                semantic_rate=1.0,
                quality_rate=1.0,
            )
            for i in range(200)
        ]
        write_rows(root / "creative/creative.jsonl", creative)
        write_json(
            root / "creative/summary.json",
            dict(
                rho=1,
                d_cond=0.3,
                vendi=1.4,
                d_eff=0.3,
                cov2=199 / 200,
                cov_m=198 / 200,
                preference=0.5,
            ),
        )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "public_eval",
            "--scores-only",
            "--study",
            str(study),
            "--geneval-repo",
            str(repo),
            "--out",
            str(out),
            "--bootstrap",
            "32",
        ],
    )
    public_main()
    stats = json.loads((out / "three-seeds.json").read_text())
    assert stats["joint"]["geneval2/soft_tifa_am"] == dict(mean=0.5, sd=0.0, n=2)
    delta = json.loads((out / "paired-bootstrap.json").read_text())["seeds"]["17"]["joint"][
        "geneval2/am"
    ]
    assert delta["difference"] == 0.5 and 0.45 < delta["ci95"][0] <= 0.5 <= delta["ci95"][1] < 0.55
    masks = json.loads((out / "creative-common-support.json").read_text())
    assert masks["d_cond"]["n_common"] == 199 and masks["vendi"]["n_common"] == 198
    assert stats["joint"]["creative/vendi_common"]["mean"] == pytest.approx(1.4)
    origin = out / "joint" / "42" / "geneval2/generation.json"
    bad = json.loads(origin.read_text())
    bad["checkpoint"]["iteration"] -= 1
    write_json(origin, bad)
    with pytest.raises(ValueError, match="registered final checkpoint"):
        public_main()
