"""Run both public benchmarks and creative evaluation for the conditions in a completed study."""

import argparse
from collections import defaultdict
from dataclasses import asdict
import math
from pathlib import Path
from statistics import fmean
import shutil
import numpy as np
import subprocess
import sys
import torch

from responseguard.benchmarks import benchmark_export, evaluate_creative
from responseguard.compare import DIVERSITY, common_diversity, seed_statistics
from responseguard.config import Config
from responseguard.io import atomic_path, digest, read_rows, write_json
from responseguard.study import lock_protocol, read
from responseguard.train import training_protocol


def prompt_gm(values):
    return 0.0 if min(values) == 0 else math.exp(fmean(math.log(x) for x in values))


def geneval_summary(records, probabilities):
    if len(records) != len(probabilities) or not records:
        raise ValueError("One ordered set of scores is required per GenEval 2 prompt")
    by_skill = defaultdict(list)
    by_atomicity = defaultdict(list)
    am_scores, gm_scores = [], []
    for row, values in zip(records, probabilities):
        if len(values) != row["atom_count"] or len(values) != len(row["skills"]) or not values:
            raise ValueError("GenEval 2 atom metadata and scores disagree")
        if any(not 0 <= s <= 1 for s in values):
            raise ValueError("Invalid atom probability")
        for skill, score in zip(row["skills"], values):
            by_skill[skill].append(score)
        gm = prompt_gm(values)
        by_atomicity[row["atom_count"]].append(gm)
        am_scores.append(fmean(values))
        gm_scores.append(gm)
    return dict(
        prompts=len(records),
        atoms=sum(map(len, probabilities)),
        scale="0 to 1",
        soft_tifa_am=fmean(am_scores),
        soft_tifa_gm=fmean(gm_scores),
        am_by_skill={k: fmean(v) for k, v in by_skill.items()},
        gm_by_atomicity={k: fmean(v) for k, v in by_atomicity.items()},
    )


def paired_public(first, second, repetitions=1000):
    """Paired prompt bootstrap. DeltaBench resamples whole pairs within each category."""
    if repetitions < 1:
        raise ValueError("Bootstrap repetitions must be positive")
    rng = np.random.default_rng(1729)
    result = {}

    def contrast(label, difference, strata=None):
        delta = np.array(difference, dtype=float)
        ok = np.isfinite(delta)
        value = delta[ok]
        groups = np.zeros(len(value)) if strata is None else np.asarray(strata)[ok]
        if not len(value):
            result[label] = dict(difference=None, ci95=None, coverage=0, common=0)
            return
        center = []
        boot = []
        for group in np.unique(groups):
            ids = np.where(groups == group)[0]
            samples = rng.choice(ids, (repetitions, len(ids)), replace=True)
            center.append(float(value[ids].mean()))
            boot.append(value[samples].mean(axis=1))
        result[label] = dict(
            difference=fmean(center),
            ci95=np.quantile(np.mean(boot, axis=0), [0.025, 0.975]).tolist(),
            coverage=float(ok.mean()),
            common=int(ok.sum()),
        )

    a, b = [read(d / "geneval2" / "scores.json") for d in (first, second)]
    if not a or len(a) != len(b) or any(not x or len(x) != len(y) for x, y in zip(a, b)):
        raise ValueError("Mismatched GenEval 2 prompt lists")
    contrast("geneval2/am", [fmean(y) - fmean(x) for x, y in zip(a, b)])
    contrast(
        "geneval2/gm",
        [prompt_gm(y) - prompt_gm(x) for x, y in zip(a, b)],
    )

    def paired_images(folder):
        pairs = defaultdict(dict)
        for row in read_rows(folder / "deltabench" / "scores" / "image_scores.jsonl"):
            pairs[(row["category"], row["pair"])][row["side"], row["index"]] = row["score"]
        if any(set(g) != {(0, 0), (0, 1), (1, 0), (1, 1)} for g in pairs.values()):
            raise ValueError("Incomplete DeltaBench pairs")
        return pairs

    x, y = paired_images(first), paired_images(second)
    if x.keys() != y.keys() or not x:
        raise ValueError("Mismatched DeltaBench pairs")
    keys = sorted(x)
    for name, stat in (("am", fmean), ("gm", lambda v: math.prod(v) ** 0.25)):
        contrast(
            "deltabench/" + name,
            [stat(y[k].values()) - stat(x[k].values()) for k in keys],
            strata=[k[0] for k in keys],
        )
    a, b = [read_rows(d / "creative" / "creative.jsonl") for d in (first, second)]
    if [(r["prompt"], r["seeds"]) for r in a] != [(r["prompt"], r["seeds"]) for r in b]:
        raise ValueError("Mismatched creative prompts or seeds")
    for key in ("rho", "d_eff", "d_cond", "vendi", "preference", "semantic_rate", "quality_rate"):
        contrast(
            "creative/" + key,
            [
                v[key] - u[key] if u[key] is not None and v[key] is not None else np.nan
                for u, v in zip(a, b)
            ],
        )
    return result


def public_checkpoint(plan, study, method, seed, scores_only):
    if seed is None:
        return None, None, None
    run = study / method / f"seed-{seed}"
    final = plan["config"]["iterations"]
    info = read(run / "run.json")
    expected = dict(plan["config"], seed=seed)
    expected["lr"] = (plan.get("rates") or {}).get(method, expected["lr"])
    if (
        info["variant"] != method
        or info["config"] != expected
        or not isinstance(info.get("history_baseline"), bool)
        or info.get("training_data", plan["training_data"]) != plan["training_data"]
    ):
        raise ValueError(f"Training protocol mismatch: {run}")
    identity = dict(run=info["identity"], iteration=final)
    meta = read(run / "eval" / "evaluation.json")
    protocol = training_protocol(expected, method, plan["training_data"], info["history_baseline"])
    if (
        not (run / "eval" / "summary.json").is_file()
        or meta["checkpoint"] != [info["identity"], final]
        or meta["training_seed"] != seed
        or meta["model"] != method
        or (meta.get("training_protocol") is not None and meta["training_protocol"] != protocol)
    ):
        raise ValueError(f"Expected a completed evaluation at iteration {final}: {run}")
    checkpoint = run / "milestones" / str(final) / "adapter.pt"
    if not scores_only:
        if not checkpoint.is_file():
            checkpoint = run / "checkpoint.pt"
        state = torch.load(checkpoint, weights_only=True, map_location="cpu")
        if (state["identity"], state["iteration"], state["variant"], digest(state["config"])) != (
            info["identity"],
            final,
            method,
            digest(expected),
        ):
            raise ValueError(f"Public evaluation checkpoint does not match the study: {checkpoint}")
    return checkpoint, identity, info["history_baseline"]


def creative_common(reports, folders):
    tables = []
    for folder in folders:
        rows = read_rows(folder / "creative" / "creative.jsonl")
        table = {(r["prompt"], tuple(r["seeds"])): {k: r[k] for k in DIVERSITY} for r in rows}
        if len(table) != len(rows):
            raise ValueError(f"Duplicate creative prompt in {folder}")
        tables.append(table)
    coverage, common = common_diversity(tables)
    for report, scores in zip(reports, common, strict=True):
        report["creative"].update(scores)
    return dict(members=[dict(method=r["method"], seed=r["seed"]) for r in reports], **coverage)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("study", "geneval-repo", "out"):
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument("--data", type=Path)
    p.add_argument("--delta-data", type=Path)
    p.add_argument(
        "--scores-only",
        action="store_true",
        help="recompute summaries from registered raw scores without loading model weights",
    )
    p.add_argument("--geneval-python", default=sys.executable)
    p.add_argument("--delta-python", default=sys.executable)
    p.add_argument("--delta-revision")
    p.add_argument("--methods", nargs="+")
    p.add_argument("--bootstrap", type=int, default=1000)
    a = p.parse_args()
    root = a.out.resolve()
    repo = a.geneval_repo.resolve()
    plan = read(a.study / "plan.json")
    cfg = Config(**plan["evaluation"])
    ge = repo / "geneval2_data.jsonl"
    records = read_rows(ge)
    if len(records) != 800 or len({s["prompt"] for s in records}) != 800:
        raise ValueError("The full GenEval 2 release must contain 800 unique prompts")
    if a.scores_only:
        registered = read(root / "plan.json")
        methods = a.methods or registered["methods"]
        if (
            registered["study"] != digest(plan)
            or registered["geneval"] != digest(records)
            or methods != registered["methods"]
            or digest(registered["config"]) != digest(asdict(cfg))
        ):
            raise ValueError("Saved public scores belong to a different evaluation plan")
    else:
        if not all((a.data, a.delta_data, a.delta_revision)):
            p.error("generation requires --data, --delta-data and --delta-revision")
        interpreter = shutil.which(a.geneval_python)
        if not interpreter:
            raise FileNotFoundError(a.geneval_python)
        a.geneval_python = str(Path(interpreter).absolute())
        delta = read(a.delta_data)
        creative = a.data / "creative.jsonl"
        creative_rows = read_rows(creative)
        if (
            len(creative_rows) != 200
            or len({r["prompt"] for r in creative_rows}) != 200
            or not delta
            or any(not r.get("questions1") or not r.get("questions2") for r in delta)
        ):
            raise ValueError(
                "Provide 200 unique creative prompts and both question sets per DeltaBench pair"
            )
        methods = a.methods or plan["methods"]
        registered = dict(
            config=asdict(cfg),
            study=digest(plan),
            methods=methods,
            geneval=digest(records),
            evaluator_sha=digest((repo / "evaluation.py").read_text()),
            delta=digest(delta),
            delta_revision=a.delta_revision,
            creative=digest(creative_rows),
        )
    if not methods or len(set(methods)) != len(methods) or not set(methods) <= set(plan["methods"]):
        raise ValueError("Expected distinct methods from the registered study")
    if len(plan["seeds"]) < 2 or len(set(plan["seeds"])) != len(plan["seeds"]) or a.bootstrap < 1:
        raise ValueError(
            "Public aggregation needs distinct training seeds and positive bootstrap repetitions"
        )
    tasks = [("reference", None)] + [(m, s) for m in methods for s in plan["seeds"]]
    checkpoints = {task: public_checkpoint(plan, a.study, *task, a.scores_only) for task in tasks}
    for method in methods:
        if len({checkpoints[method, s][2] for s in plan["seeds"]}) != 1:
            raise ValueError(f"Different training history baselines across {method} seeds")
    if not a.scores_only:
        lock_protocol(root, registered)
    reports, folders = [], []
    for (method, seed), (checkpoint, identity, _) in checkpoints.items():
        folder = root / method / (str(seed) if seed is not None else "base")
        ge_out = folder / "geneval2"
        de_out = folder / "deltabench"
        cr_out = folder / "creative"
        if not a.scores_only:
            benchmark_export(cfg, ge, "geneval2", ge_out, checkpoint)
            benchmark_export(cfg, a.delta_data, "deltabench", de_out, checkpoint)
            if not (cr_out / "summary.json").exists():
                evaluate_creative(cfg, creative, cr_out, checkpoint)
        for directory, source, field, suite in (
            (ge_out, "geneval", "prompt_list", "geneval2"),
            (de_out, "delta", "prompt_list", "deltabench"),
            (cr_out, "creative", "prompts", "creative"),
        ):
            origin = read(directory / "generation.json")
            if (
                origin["checkpoint"] != identity
                or origin[field] != registered[source]
                or origin["benchmark"] != suite
                or digest(origin["config"]) != digest(asdict(cfg))
            ):
                raise ValueError(
                    f"Scores do not match the registered final checkpoint or prompts: {directory}"
                )
        if not a.scores_only and not (ge_out / "scores.json").is_file():
            with atomic_path(ge_out / "scores.json") as tmp:
                subprocess.run(
                    [
                        a.geneval_python,
                        str(repo / "evaluation.py"),
                        "--benchmark_data",
                        str(ge),
                        "--image_filepath_data",
                        str(ge_out / "image_paths.json"),
                        "--method",
                        "soft_tifa_gm",
                        "--output_file",
                        str(tmp),
                    ],
                    cwd=repo,
                    check=True,
                )
        ge_scores = geneval_summary(records, read(ge_out / "scores.json"))
        write_json(ge_out / "summary.json", ge_scores)
        if not a.scores_only and not (de_out / "scores" / "summary.json").exists():
            subprocess.run(
                [
                    a.delta_python,
                    str(Path(__file__).with_name("score_deltabench.py")),
                    "--manifest",
                    str(de_out / "images.jsonl"),
                    "--out",
                    str(de_out / "scores"),
                    "--revision",
                    a.delta_revision,
                ],
                check=True,
            )
        reports.append(
            dict(
                method=method,
                seed=seed,
                checkpoint=identity,
                geneval2=ge_scores,
                deltabench=read(de_out / "scores" / "summary.json"),
                creative=read(cr_out / "summary.json"),
            )
        )
        folders.append(folder)
    coverage = creative_common(reports, folders)
    if coverage["d_cond"]["n_total"] != 200:
        raise ValueError("Expected raw scores for all 200 creative prompts")
    write_json(root / "creative-common-support.json", coverage)
    write_json(root / "results.json", reports)

    summary = {}
    for method in methods:
        subset = [r for r in reports if r["method"] == method]
        by_metric = {}
        for suite, keys in {
            "geneval2": ["soft_tifa_am", "soft_tifa_gm"],
            "deltabench": ["macro_am", "macro_gm"],
            "creative": [
                "rho",
                "d_cond",
                "d_cond_common",
                "d_eff",
                "vendi",
                "vendi_common",
                "cov2",
                "cov_m",
                "preference",
            ],
        }.items():
            for key in keys:
                stats = seed_statistics(r[suite][key] for r in subset)
                by_metric[suite + "/" + key] = dict(
                    mean=stats["mean"], sd=stats["sd"], n=stats["n_defined"]
                )
        summary[method] = by_metric
    write_json(root / "three-seeds.json", summary)
    contrasts = {}
    for seed in plan["seeds"]:
        paths = [root / m / str(seed) for m in methods]
        contrasts[seed] = {
            m: paired_public(paths[0], p, a.bootstrap) for m, p in zip(methods[1:], paths[1:])
        }
    write_json(root / "paired-bootstrap.json", dict(comparator=methods[0], seeds=contrasts))


if __name__ == "__main__":
    main()
