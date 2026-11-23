"""Multi-seed study with validation checkpoints and matched training-cost ceilings."""

from dataclasses import asdict, replace
import json
from pathlib import Path

import torch

from .compare import aggregate_methods, common_diversity, compare, prompt_diversity, training_curves
from .data import composition_audit, load_pairs
from .evaluate import evaluate, evaluation_protocol
from .io import atomic_path, digest, read_rows, seed_for, write_json
from .train import VARIANTS, train

METHODS = (
    "flow-grpo",
    "diversegrpo",
    "drift",
    "pair-grpo-adapted",
    "margin-supervised",
    "per-prompt",
    "task-only",
    "semantic",
    "diversity",
    "unprojected",
    "joint",
)
COSTS = (
    "denoiser_examples",
    "denoiser_calls",
    "verifier_questions",
    "preference_scores",
    "gpu_hours",
)
OUTCOMES = ("pair_sr", "rho", "d_cond", "d_eff", "preference")


def costs(row):
    counts = row["charged_counts"]
    return dict(
        denoiser_examples=sum(
            counts.get(k, 0)
            for k in ("denoiser_examples", "replay_examples", "kl_reference_examples")
        ),
        denoiser_calls=sum(
            counts.get(k, 0) for k in ("denoiser_calls", "replay_calls", "kl_reference_calls")
        ),
        verifier_questions=counts.get("semantic_questions", 0) + counts.get("quality_scores", 0),
        preference_scores=counts.get("preference_scores", 0),
        gpu_hours=row["charged_device_hours"],
    )


def read(path):
    return json.loads(Path(path).read_text())


def lock_protocol(out, plan):
    path = Path(out) / "plan.json"
    if path.exists() and digest(read(path)) != digest(plan):
        raise ValueError(f"Existing output has a different study plan: {path}")
    write_json(path, plan)


def val_pairs(pairs, limit, seed):
    if isinstance(pairs, (str, Path)):
        pairs = load_pairs(Path(pairs) / "val.jsonl", {"val"})
    # Order is independent of the actor and shared with all tuning candidates.
    return sorted(pairs, key=lambda p: seed_for("validation", seed, p.key))[:limit]


def scored(cfg, pairs, folder, checkpoint=None, expected=None):
    cached = (folder / "summary.json").exists()
    if not cached:
        evaluate(cfg, pairs, folder, checkpoint)
    meta = read(folder / "evaluation.json")
    if meta["protocol"] != evaluation_protocol(cfg, pairs, meta["metric_space"]):
        raise ValueError(f"Evaluation protocol mismatch in {folder}")
    if checkpoint is None:
        if meta["model"] != "reference" or meta["checkpoint"] is not None:
            raise ValueError(f"Expected a frozen reference evaluation in {folder}")
    else:
        if expected and (
            not meta["checkpoint"]
            or (meta["model"], meta["training_seed"], meta["checkpoint"][1]) != expected
        ):
            raise ValueError(f"Expected method, seed and training iteration {expected} in {folder}")
        if cached:
            state = torch.load(checkpoint, weights_only=True, map_location="cpu")
            if meta["checkpoint"] != [state["identity"], state["iteration"]]:
                raise ValueError(f"Cached evaluation belongs to a different checkpoint: {folder}")
    return read(folder / "summary.json")["all"]


def budget_report(points, directory, figures=True, validation_metrics=None):
    result, coverage = {}, {}
    for key in COSTS:
        for line in points.values():
            if (
                not line
                or line[0]["iteration"] != 0
                or any(
                    b["iteration"] <= a["iteration"] or b["cost"][key] < a["cost"][key]
                    for a, b in zip(line, line[1:])
                )
            ):
                raise ValueError("Budget curves require ordered checkpoints and cumulative costs")
        cap = min(series[-1]["cost"][key] for series in points.values())
        first = {m: line[1]["cost"][key] if len(line) > 1 else None for m, line in points.items()}
        minimum = max(first.values()) if None not in first.values() else None
        measured = {p["cost"][key] for line in points.values() for p in line[1:]}
        cutoffs = sorted(
            x for x in measured if cap > 0 and minimum is not None and minimum <= x <= cap
        )
        result[key] = [
            dict(
                ceiling=x,
                methods={
                    m: next(p for p in reversed(line) if p["iteration"] > 0 and p["cost"][key] <= x)
                    for m, line in points.items()
                },
            )
            for x in cutoffs
        ]
        if validation_metrics is not None:
            for row in result[key]:
                chosen = row["methods"]
                masks, shared = common_diversity(
                    [validation_metrics[m, point["iteration"]] for m, point in chosen.items()]
                )
                row["common_support"] = masks
                row["methods"] = {
                    m: dict(point, scores={**point["scores"], **extra})
                    for (m, point), extra in zip(chosen.items(), shared, strict=True)
                }
        coverage[key] = dict(
            cap=cap,
            first_trained_cost=first,
            required_ceiling=minimum,
            first_common_ceiling=cutoffs[0] if cutoffs else None,
            comparable_ceilings=len(cutoffs),
            status="matched" if cutoffs else "unavailable" if cap == 0 else "no_trained_overlap",
        )
    write_json(
        directory / "validation-budget.json", dict(points=points, matched=result, coverage=coverage)
    )
    if figures and any(result.values()):
        from matplotlib import pyplot as plt

        axes_keys = [k for k in COSTS if result[k]]
        fig, axes = plt.subplots(
            len(OUTCOMES),
            len(axes_keys),
            figsize=(3.5 * len(axes_keys), 2.4 * len(OUTCOMES)),
            squeeze=False,
        )
        for col, cost in enumerate(axes_keys):
            for row, metric in enumerate(OUTCOMES):
                ax = axes[row, col]
                if metric == "d_cond" and validation_metrics is not None:
                    metric = "d_cond_common"
                for method in points:
                    ax.step(
                        [p["ceiling"] for p in result[cost]],
                        [p["methods"][method]["scores"][metric] for p in result[cost]],
                        where="post",
                        marker=".",
                        label=method,
                        linewidth=1,
                    )
                ax.set(xlabel=cost, ylabel=metric)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=7)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(directory / "validation-budget.png", dpi=125)
        plt.close(fig)
    return result


def tune(
    cfg,
    evaluation_cfg,
    data,
    out,
    cache,
    methods,
    trial_steps,
    val_limit,
    multipliers=(0.5, 1.0, 2.0),
):
    if (
        not 1 <= trial_steps <= cfg.iterations
        or val_limit < 1
        or not multipliers
        or min(multipliers) <= 0
    ):
        raise ValueError("Invalid validation search budget")
    root = Path(out)
    train_pairs = load_pairs(Path(data) / "train.jsonl", {"train"})
    validation = load_pairs(Path(data) / "val.jsonl", {"val"})
    composition_audit(dict(train=train_pairs, val=validation))
    val = val_pairs(validation, val_limit, cfg.evaluation_seed)
    plan = dict(
        train_config=asdict(cfg),
        eval_config=asdict(evaluation_cfg),
        methods=methods,
        steps=trial_steps,
        multipliers=multipliers,
        train_ids=digest([p.key for p in train_pairs]),
        val_ids=digest([p.key for p in val]),
    )
    lock_protocol(root, plan)
    reference = scored(evaluation_cfg, val, root / "reference")
    trials = []
    chosen = {}
    for method in methods:
        options = []
        for i, mult in enumerate(multipliers):
            trial = replace(cfg, lr=cfg.lr * mult)
            dest = root / method / f"trial-{i}"
            ckpt = train(
                trial,
                train_pairs,
                dest,
                cache,
                method,
                resume=(dest / "checkpoint.pt").exists(),
                until=trial_steps,
            )
            score = scored(evaluation_cfg, val, dest / "val", ckpt, (method, cfg.seed, trial_steps))
            # Identical validation rule/trial count for every method.
            shortfall = sum(
                max(0.0, reference[k] - 0.03 - score[k]) for k in ("pair_sr", "quality_rate", "rho")
            )
            shortfall += max(0.0, 0.8 * reference["d_eff"] - score["d_eff"])
            options.append((shortfall, -score["preference"], abs(mult - 1), trial.lr))
            trials.append(
                dict(
                    method=method,
                    lr=trial.lr,
                    scores=score,
                    shortfall=shortfall,
                    cost=costs(read_rows(dest / "train.jsonl")[-1]),
                )
            )
        chosen[method] = min(options)[-1]
    separate = chosen.copy()
    if "joint" in chosen:
        for name in (
            "semantic",
            "diversity",
            "task-only",
            "unprojected",
            "per-prompt",
            "margin-supervised",
        ):
            if name in chosen:
                chosen[name] = chosen["joint"]
    result = dict(
        individual_winners=separate,
        base=digest(asdict(cfg)),
        training_data=plan["train_ids"],
        validation_data=plan["val_ids"],
        evaluation_config=digest(asdict(evaluation_cfg)),
        trials=trials,
        learning_rates=chosen,
    )
    write_json(root / "selection.json", result)
    return result


def default_milestones(iterations):
    return sorted({max(1, iterations // n) for n in (64, 32, 16, 8, 4, 2, 1)})


def run_study(
    cfg,
    eval_cfg,
    data,
    out,
    cache,
    methods=METHODS,
    seeds=(17, 42, 73),
    milestones=None,
    val_limit=64,
    repetitions=1000,
    figures=True,
    selection=None,
):
    if (
        len(seeds) < 2
        or len(set(seeds)) != len(seeds)
        or len(set(methods)) != len(methods)
        or not set(methods) <= set(VARIANTS)
        or not methods
    ):
        raise ValueError("Provide distinct training seeds and supported unique methods")
    stops = sorted(set(milestones)) if milestones else default_milestones(cfg.iterations)
    if val_limit < 1 or min(stops) < 1 or max(stops) != cfg.iterations:
        raise ValueError("Validation milestones must include the final iteration")
    root = Path(out)
    data = Path(data)
    tr = load_pairs(data / "train.jsonl", {"train"})
    validation = load_pairs(data / "val.jsonl", {"val"})
    test = load_pairs(data / "test.jsonl", {"test"})
    split_audit = composition_audit(dict(train=tr, val=validation, test=test))
    val = val_pairs(validation, val_limit, cfg.evaluation_seed)
    ids = digest([p.key for p in tr])
    rates = read(selection) if selection else None
    if rates and (
        rates["base"] != digest(asdict(cfg))
        or rates["training_data"] != ids
        or rates["evaluation_config"] != digest(asdict(eval_cfg))
        or not set(methods) <= set(rates["learning_rates"])
    ):
        raise ValueError("Selection does not match the training/evaluation protocol")
    plan = dict(
        config=asdict(cfg),
        evaluation=asdict(eval_cfg),
        methods=methods,
        seeds=seeds,
        stops=stops,
        training_data=ids,
        validation=digest([p.key for p in val]),
        test=digest([p.key for p in test]),
        rates=rates and rates["learning_rates"],
        bootstrap=repetitions,
    )
    lock_protocol(root, plan)
    baseline = scored(eval_cfg, val, root / "validation-reference")
    scored(eval_cfg, test, root / "reference")
    for seed in seeds:
        points = {}
        validation_metrics = {}
        evals = []
        runs = []
        for method in methods:
            run_cfg = replace(
                cfg, seed=seed, lr=rates["learning_rates"][method] if rates else cfg.lr
            )
            run = root / method / f"seed-{seed}"
            zero = dict(
                iteration=0, cost=dict.fromkeys(COSTS, 0), scores={k: baseline[k] for k in OUTCOMES}
            )
            line = [zero]
            for step in stops:
                dest = run / "milestones" / str(step)
                ckpt = dest / "adapter.pt"
                if not ckpt.exists():
                    latest = train(
                        run_cfg,
                        tr,
                        run,
                        cache,
                        method,
                        resume=(run / "checkpoint.pt").exists(),
                        until=step,
                    )
                    state = torch.load(latest, weights_only=True, map_location="cpu")
                    if state["iteration"] != step:
                        raise ValueError(
                            "Missing historical checkpoint for the requested milestone"
                        )
                    with atomic_path(ckpt) as tmp:
                        torch.save(
                            {
                                k: state[k]
                                for k in (
                                    "adapter",
                                    "variant",
                                    "config",
                                    "iteration",
                                    "identity",
                                    "training_data",
                                    "history_baseline",
                                )
                            },
                            tmp,
                        )
                    del state
                summary = scored(eval_cfg, val, dest / "val", ckpt, (method, seed, step))
                validation_metrics[method, step] = prompt_diversity(
                    read_rows(dest / "val" / "pairs.jsonl")
                )
                log = read_rows(run / "train.jsonl")[step - 1]
                line.append(
                    dict(iteration=step, cost=costs(log), scores={k: summary[k] for k in OUTCOMES})
                )
            scored(eval_cfg, test, run / "eval", ckpt, (method, seed, cfg.iterations))
            points[method] = line
            evals.append(run / "eval")
            runs.append(run)
        folder = root / f"comparison-{seed}"
        compare([*evals, root / "reference"], folder, repetitions, figures)
        if figures:
            training_curves(runs, folder / "training.png")
        budget_report(points, folder, figures, validation_metrics)
    final_evals = [root / m / f"seed-{s}" / "eval" for m in methods for s in seeds]
    aggregate_paths = aggregate_methods([*final_evals, root / "reference"])
    for method, report in aggregate_paths.items():
        write_json(root / method / "three-seeds.json", report)
    result = dict(
        methods=list(methods),
        seeds=list(seeds),
        training_runs=len(methods) * len(seeds),
        validation_checkpoints=len(methods) * len(seeds) * len(stops),
        test_pairs=len(test),
        validation_pairs=len(val),
        split_audit=split_audit,
        aggregates=aggregate_paths,
    )
    write_json(root / "study.json", result)
    return result
