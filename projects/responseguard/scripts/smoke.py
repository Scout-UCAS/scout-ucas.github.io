"""Offline end-to-end smoke for the paper protocol and real, miniature SD3 modules."""

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import monotonic

# A smoke must run even on a machine without credentials, model caches or displays.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from PIL import Image

from responseguard.benchmarks import (
    benchmark_export,
    creative_prompts,
    evaluate_creative,
    generate_text,
)
from responseguard.blind import blind_scores, blind_study
from responseguard.calibrate import audit, fit_gates
from responseguard.cli import export_lora, prepare
from responseguard.compare import aggregate
from responseguard.config import Config
from responseguard.data import load_pairs, make_dataset
from responseguard.evaluate import eval_plan, evaluate
from responseguard.io import read_rows, write_json, write_rows
from responseguard.judges import make_judge
from responseguard.policy import make_policy
from responseguard.projection_audit import inspect_projection
from responseguard.study import COSTS, METHODS, read, run_study, tune
from responseguard.train import load_checkpoint, train
from scripts.public_eval import geneval_summary, paired_public
from scripts.score_deltabench import score_manifest
from scripts.smoke_models import make_local_models

PROJECT = Path(__file__).resolve().parents[1]


def verify_study(root, cfg, result):
    all_rows = 0
    all_images = 0
    max_error = 0.0
    for method in result["methods"]:
        for seed in result["seeds"]:
            run = root / method / f"seed-{seed}"
            log = read_rows(run / "train.jsonl")
            assert len(log) == cfg.iterations and all(
                math.isfinite(r["gradient_norm"]) for r in log
            )
            assert len({g["pair"] for r in log for g in r["groups"]}) == 8
            for row in read_rows(run / "eval" / "pairs.jsonl"):
                all_rows += 1
                assert 0 <= row["retained"] <= row["ref_correct"] <= row["k"]
                assert 0 <= row["corrected"] <= row["k"] - row["ref_correct"]
                assert row["pair_correct"] == row["retained"] + row["corrected"]
                for p in row["prompts"]:
                    assert p["k"] == cfg.k_eval and len(p["samples"]) == p["k"]
                    assert sum(s["valid"] for s in p["samples"]) == p["n_valid"]
                    for s in p["samples"]:
                        with Image.open(run / "eval" / s["image"]) as im:
                            im.verify()
                        all_images += 1
                    max_error = max(
                        max_error, abs(p["d_eff"] - p["valid_pair_fraction"] * (p["d_cond"] or 0.0))
                    )
    assert max_error < 1e-12
    budgets = {}
    for seed in result["seeds"]:
        report = read(root / f"comparison-{seed}" / "validation-budget.json")
        budgets[seed] = {}
        for axis in COSTS:
            points = report["matched"][axis]
            if axis == "gpu_hours":
                assert not points and report["coverage"][axis]["status"] == "unavailable"
                continue
            assert points and all(
                0 < p["iteration"] and p["cost"][axis] <= row["ceiling"]
                for row in points
                for p in row["methods"].values()
            )
            assert all(
                row["common_support"]["d_cond"]["n_total"] == 2 * result["validation_pairs"]
                for row in points
            )
            budgets[seed][axis] = dict(
                cap=points[-1]["ceiling"],
                comparable_ceilings=len(points),
                joint_iteration=points[-1]["methods"]["joint"]["iteration"],
            )
    common = result["aggregates"]["joint"]["common_support"]
    assert len(common["members"]) == 1 + len(result["seeds"]) * len(result["methods"])
    assert all(r["common_support"] == common for r in result["aggregates"].values())
    for metric in ("d_cond", "vendi"):
        assert common[metric]["n_total"] == 2 * result["test_pairs"]
        for r in result["aggregates"].values():
            assert (r["metrics"][metric + "_common"]["mean"] is not None) == (
                common[metric]["n_common"] > 0
            )
    mixed = [
        root / "joint" / f"seed-{s}" / "milestones" / str(t) / "val"
        for s, t in zip(result["seeds"], [1, cfg.iterations, cfg.iterations], strict=True)
    ]
    try:
        aggregate(mixed)
    except ValueError as exc:
        assert "same positive training iteration" in str(exc)
    else:
        raise AssertionError("Mixed training iterations were accepted")
    return dict(
        test_pair_records=all_rows,
        test_images_read=all_images,
        max_decomposition_error=max_error,
        trained_budgets=budgets,
        common_support=common,
        mixed_iterations_rejected=True,
    )


def sd3_smoke(root, data):
    from diffusers import StableDiffusion3Pipeline

    models = root / "random-models"
    models.mkdir(parents=True)
    torch.manual_seed(12)
    make_local_models(models)
    cfg = replace(
        Config(),
        device="cpu",
        judge_device="cpu",
        dtype="float32",
        model=str(models / "sd3"),
        train_vqa=str(models / "vqa"),
        eval_vqa=str(models / "eval_vqa"),
        feature_model=str(models / "dino"),
        eval_feature_model=str(models / "eval_dino"),
        preference_model=str(models / "pick"),
        preference_processor=str(models / "pick"),
        eval_preference_model=str(models / "eval_pick"),
        perceptual_metric="rgb",
        resolution=32,
        max_text_length=16,
        train_steps=2,
        eval_steps=3,
        k_train=2,
        k_ref=3,
        k_cal=3,
        k_eval=4,
        vendi_m=2,
        rho_floor=1.0,
        groups_per_batch=1,
        microbatch=2,
        iterations=2,
        lora_rank=2,
        lora_alpha=4,
        lr=1e-4,
    )
    write_json(root / "config.json", asdict(cfg))
    training = load_pairs(data / "train.jsonl", {"train"})[:2]
    testing = load_pairs(data / "test.jsonl", {"test"})[:1]
    for method in ("joint", "drift", "diversegrpo"):
        ckpt = train(cfg, training, root / method, root / "cache", method)
        state = load_checkpoint(make_policy(cfg), ckpt)
        assert any("lora_B" in k and v.abs().sum() for k, v in state["adapter"].items())
    trained = root / "joint" / "checkpoint.pt"
    generate_text(cfg, "Two blue cups.", [7, 11], root / "inference", trained)
    export_lora(cfg, trained, root / "lora")
    pipe = StableDiffusion3Pipeline.from_pretrained(cfg.model)
    pipe.load_lora_weights(root / "lora", weight_name="pytorch_lora_weights.safetensors")
    assert next(iter(pipe.transformer.peft_config.values())).lora_alpha == cfg.lora_alpha
    del pipe
    evaluate(cfg, testing, root / "reference")
    direct = evaluate(cfg, testing, root / "eval", trained)
    plan = eval_plan(cfg, testing)
    row = read_rows(root / "eval" / "pairs.jsonl")[0]
    for item in plan:
        sample = next(
            s for s in row["prompts"][item["side"]]["samples"] if s["seed"] == item["seed"]
        )
        item["image"] = str(Path("eval") / sample["image"])
    write_rows(root / "external.jsonl", plan)
    ext = evaluate(
        cfg, testing, root / "external-eval", external=root / "external.jsonl", name="roundtrip"
    )
    assert direct["all"] == ext["all"] and direct["metric_space"] == "diagnostic_rgb_cosine"
    creative = root / "creative.jsonl"
    write_rows(creative, creative_prompts()[:2])
    cr = evaluate_creative(cfg, creative, root / "creative", trained)
    # Exercise the official exchange formats with a small, visibly labeled local fixture.
    ge = [
        dict(
            prompt="Two blue cups above one box.",
            atom_count=3,
            skills=["object", "count", "position"],
            questions=["Is there a cup?", "Are there two cups?", "Are the cups above the box?"],
        )
    ]
    write_rows(root / "ge-fixture.jsonl", ge)
    benchmark_export(cfg, root / "ge-fixture.jsonl", "geneval2", root / "geneval2", trained)
    mapping = read(root / "geneval2" / "image_paths.json")
    validator = make_judge(cfg, evaluation=True)
    with torch.inference_mode(), Image.open(mapping[ge[0]["prompt"]]) as im:
        probabilities = [validator.binary_answers(im.convert("RGB"), ge[0]["questions"]).tolist()]
    ge_result = geneval_summary(ge, probabilities)
    unequal_am = geneval_summary(
        [dict(atom_count=n, skills=["object"] * n) for n in (3, 9)], [[1.0] * 3, [0.0] * 9]
    )["soft_tifa_am"]
    assert unequal_am == 0.5
    write_json(root / "geneval2" / "scores.json", probabilities)
    write_json(
        root / "geneval2" / "diagnostic_scores.json",
        dict(evaluator="random_local_blip", **ge_result),
    )
    delta = [
        dict(
            category="color",
            caption1="A red cup.",
            caption2="A blue cup.",
            questions1=["Is the cup red?"],
            questions2=[dict(question="Is the cup red?", answer="no")],
        )
    ]
    write_json(root / "delta-fixture.json", delta)
    benchmark_export(cfg, root / "delta-fixture.json", "deltabench", root / "deltabench", trained)
    benchmark_export(cfg, root / "delta-fixture.json", "paircomp", root / "paircomp", trained)
    directory = root / "deltabench"
    with torch.inference_mode():
        de = score_manifest(
            read_rows(directory / "images.jsonl"),
            directory,
            directory / "scores",
            lambda im, qs: validator.binary_answers(im, qs).tolist(),
            dict(model="random_local_blip"),
        )
    return dict(
        train_methods=["joint", "drift", "diversegrpo"],
        updates_each=2,
        native_lora_reload=True,
        public_interval_roundtrip=paired_public(root, root, 32)["geneval2/gm"]["ci95"],
        external_scores_identical=True,
        metric_space=direct["metric_space"],
        creative_prompts=2,
        creative_summary=cr,
        geneval2_fixture=ge_result,
        unequal_atom_prompt_am=unequal_am,
        deltabench_fixture=de,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    root = a.out.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use an empty output directory for a full smoke")
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    start = monotonic()
    cfg = Config.load(PROJECT / "configs" / "smoke.json")
    sizes = make_dataset(root / "full-data")
    write_rows(root / "full-data" / "creative.jsonl", creative_prompts())
    data = root / "data"
    make_dataset(data, (8, 6, 8))
    tr = load_pairs(data / "train.jsonl", {"train"})
    val = load_pairs(data / "val.jsonl", {"val"})
    write_rows(data / "creative.jsonl", creative_prompts())
    prepare(cfg, tr, root / "cache")
    study = root / "study"
    results = run_study(
        cfg, cfg, data, study, root / "cache", METHODS, (17, 42, 73), val_limit=3, repetitions=200
    )
    checked = verify_study(study, cfg, results)
    # Compare split/resumed joint training to a single uninterrupted run.
    uninterrupted = train(replace(cfg, seed=42), tr, root / "uninterrupted", root / "cache")
    state = torch.load(uninterrupted, weights_only=True)
    continued = torch.load(study / "joint" / "seed-42" / "checkpoint.pt", weights_only=True)
    assert continued["history"] == state["history"] and continued["dual"] == state["dual"]
    for name, w in state["adapter"].items():
        torch.testing.assert_close(w, continued["adapter"][name], atol=0, rtol=0)
    search = tune(
        cfg,
        cfg,
        data,
        root / "tuning",
        root / "cache",
        ["joint", "flow-grpo"],
        2,
        3,
        [0.5, 1.0, 2.0],
    )
    no_history = train(cfg, tr, root / "no-history", root / "cache", "joint", use_history=False)
    assert all(
        g["baseline"] == 0 and g["scale"] == 1
        for row in read_rows(no_history.parent / "train.jsonl")
        for g in row["groups"]
    )
    # Raise exploration noise only for this separate diagnostic so correct anchors can actually break.
    stress_cfg = replace(cfg, sde_noise=1.7, iterations=16)
    stress_path = train(stress_cfg, tr, root / "stress", root / "stress-cache", "joint")
    stress_rows = read_rows(stress_path.parent / "train.jsonl")
    groups = [g for row in stress_rows for g in row["groups"]]
    stress = dict(
        noise=stress_cfg.sde_noise,
        iterations=stress_cfg.iterations,
        max_semantic_cost=max(g["sem"] for g in groups),
        max_rate_cost=max(p["rate_cost"] for g in groups for p in g["prompts"]),
        max_distance_cost=max(p["distance_cost"] for g in groups for p in g["prompts"]),
        dual_increases=[
            sum(row["dual_next"][i] > row["dual_used"][i] for row in stress_rows) for i in (0, 1)
        ],
    )
    assert (
        min(stress["dual_increases"]) > 0
        and min(stress["max_semantic_cost"], stress["max_rate_cost"], stress["max_distance_cost"])
        > 0
    )
    write_json(stress_path.parent / "diagnostics.json", stress)
    proj = inspect_projection(cfg, val[:3], root / "projection-audit", root / "cache")
    sheets = {}
    for role in ("train", "eval"):
        sheet = audit(
            replace(cfg, k_cal=32), val, root / "calibration" / role, role, simulator_labels=True
        )
        sheets[role] = str(sheet.relative_to(root))
        truth = sheet.with_name("simulator-labels.csv")
        write_json(sheet.with_name("simulator-gates.json"), fit_gates(truth))
    blind = blind_study(
        [study / "flow-grpo" / "seed-42" / "eval", study / "joint" / "seed-42" / "eval"],
        root / "blind",
        6,
    )
    pending = blind_scores(root / "blind")
    assert pending["complete_ratings"] == 0
    write_json(root / "blind" / "status.json", pending)
    components = sd3_smoke(root / "sd3", data)
    with (root / "pytest.txt").open("w") as log:
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-ra"],
            cwd=PROJECT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    report = dict(
        status="passed",
        backend="cpu",
        python=platform.python_version(),
        torch=torch.__version__,
        cuda=torch.cuda.is_available(),
        elapsed_seconds=monotonic() - start,
        full_dataset=sizes,
        full_split_audit=read(root / "full-data" / "dataset.json")["audit"],
        creative_prompt_bank=len(creative_prompts()),
        methods=list(METHODS),
        seeds=[17, 42, 73],
        training_runs=results["training_runs"],
        validation_checkpoints=results["validation_checkpoints"],
        test_pairs=results["test_pairs"],
        k_eval=cfg.k_eval,
        **checked,
        bit_exact_resume=True,
        stress=stress,
        validation_trials=len(search["trials"]),
        projection=proj,
        calibration=sheets,
        blind=blind,
        human_ratings_collected=0,
        sd3=components,
        pytest=(root / "pytest.txt").read_text().strip(),
    )
    write_json(root / "smoke.json", report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "training_runs",
                    "validation_checkpoints",
                    "test_images_read",
                    "elapsed_seconds",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
