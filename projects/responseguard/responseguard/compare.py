"""Matched prompt-set comparisons, paired bootstrap intervals and fixed-seed figures."""

import json
from collections import defaultdict
from pathlib import Path
from statistics import stdev

import numpy as np
from PIL import Image, ImageDraw

from .evaluate import mean_or_none, summarize
from .io import digest, read_rows, write_json
from .train import archived_protocol


def evaluation_metadata(paths):
    if len(paths) < 2:
        raise ValueError("Provide at least two evaluation directories")
    meta = []
    for path in paths:
        root = Path(path)
        if not (root / "summary.json").exists():
            raise ValueError(f"Incomplete evaluation: {root}")
        meta.append(json.loads((root / "evaluation.json").read_text()))
    if len({m["protocol"] for m in meta}) != 1:
        raise ValueError(
            "Compare runs evaluated on identical prompts, gates, model family, budget and seeds"
        )
    return meta


def get_runs(paths):
    meta = evaluation_metadata(paths)
    rows = [read_rows(Path(p) / "pairs.jsonl") for p in paths]
    ids = [r["pair"] for r in rows[0]]
    if (
        not ids
        or len(ids) != len(set(ids))
        or any([r["pair"] for r in run] != ids for run in rows[1:])
    ):
        raise ValueError("Evaluation rows must contain the same ordered, unique prompt pairs")
    return meta, rows


def compare(paths, output, repetitions=1000, figures=False, pair_index=0):
    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    meta, rows = get_runs(paths)

    # [methods, prompt pairs, sides]; NaN denotes a genuinely undefined metric.
    def cube(key):
        return np.array(
            [[[p[key] for p in r["prompts"]] for r in run] for run in rows], dtype=float
        )

    cond, ven = cube("d_cond"), cube("vendi")
    common = np.isfinite(cond).all(axis=0)
    common_m = np.isfinite(ven).all(axis=0)
    result = {
        "metric_space": meta[0]["metric_space"],
        "common_cov2": float(common.mean()),
        "common_cov_m": float(common_m.mean()),
        "runs": [],
        "paired_differences_vs_first": [],
    }

    for i, run in enumerate(rows):
        result["runs"].append(
            dict(
                path=str(Path(paths[i]).resolve()),
                method=meta[i]["model"],
                training_seed=meta[i]["training_seed"],
                **summarize(run),
                d_cond_common=mean_or_none(cond[i][common]),
                vendi_common=mean_or_none(ven[i][common_m]),
            )
        )

    # Bootstrap whole pairs so the two prompts, all seeds and all methods remain matched.
    rng = np.random.default_rng(1729)
    for key, values, mask in [
        ("rho", cube("rho"), np.ones_like(common)),
        ("quality_rate", cube("quality_rate"), np.ones_like(common)),
        ("semantic_rate", cube("semantic_rate"), np.ones_like(common)),
        ("preference", cube("preference"), np.ones_like(common)),
        ("d_cond_common", cond, common),
        ("d_eff", cube("d_eff"), np.ones_like(common)),
        ("vendi_common", ven, common_m),
    ]:
        for j in range(1, len(rows)):
            delta = values[j] - values[0]
            boot = []
            for _ in range(repetitions):
                indices = rng.integers(len(common), size=len(common))
                valid = mask[indices]
                if valid.any():
                    boot.append(float(delta[indices][valid].mean()))
            result["paired_differences_vs_first"].append(
                dict(
                    metric=key,
                    method_index=j,
                    difference=mean_or_none(delta[mask]),
                    ci95=np.quantile(boot, [0.025, 0.975]).tolist() if boot else None,
                    usable_resamples=len(boot),
                )
            )
    for key, numerator, denom in (
        ("pair_sr", "pair_correct", "k"),
        ("retention", "retained", "ref_correct"),
        ("correction", "corrected", "ref_wrong"),
        ("response_correct", "direction_correct", "k"),
    ):
        denominator = np.array(
            [r["k"] - r["ref_correct"] if denom == "ref_wrong" else r[denom] for r in rows[0]]
        )
        for j in range(1, len(rows)):
            if any(a["ref_correct"] != b["ref_correct"] for a, b in zip(rows[0], rows[j])):
                raise ValueError("Reference success counts differ for matched test pairs")
            delta = np.array([b[numerator] - a[numerator] for a, b in zip(rows[0], rows[j])])
            boot = []
            for _ in range(repetitions):
                ix = rng.integers(len(delta), size=len(delta))
                if denominator[ix].sum():
                    boot.append(float(delta[ix].sum() / denominator[ix].sum()))
            result["paired_differences_vs_first"].append(
                dict(
                    metric=key,
                    method_index=j,
                    difference=float(delta.sum() / denominator.sum())
                    if denominator.sum()
                    else None,
                    ci95=np.quantile(boot, [0.025, 0.975]).tolist() if boot else None,
                    usable_resamples=len(boot),
                )
            )
    # Decompose mean D_eff on every prompt; undefined conditional distances are never imputed.
    fractions = np.array(
        [
            [
                [p["n_valid"] * (p["n_valid"] - 1) / (p["k"] * (p["k"] - 1)) for p in r["prompts"]]
                for r in run
            ]
            for run in rows
        ]
    )
    eff, sem, qual = cube("d_eff"), cube("semantic_rate"), cube("quality_rate")
    result["diversity_attribution"] = []
    result["matched_semantic_quality"] = []
    for j in range(1, len(rows)):
        shared = np.isfinite(cond[0]) & np.isfinite(cond[j])
        count_term = (
            0.5
            * (cond[0][shared] + cond[j][shared])
            * (fractions[j][shared] - fractions[0][shared])
        )
        distance_term = (
            0.5
            * (fractions[j][shared] + fractions[0][shared])
            * (cond[j][shared] - cond[0][shared])
        )
        delta = eff[j] - eff[0]
        result["diversity_attribution"].append(
            dict(
                method_index=j,
                shared_prompts=int(shared.sum()),
                prompts=shared.size,
                rate_component=float(count_term.sum() / shared.size),
                distance_component=float(distance_term.sum() / shared.size),
                outside_common_support=float(delta[~shared].sum() / shared.size),
                total=float(delta.mean()),
            )
        )
    # Each comparison uses the same prompt and allows at most one additional pass on either gate.
    names = [m["model"] for m in meta]
    targets = sorted(
        set(
            [(0, j) for j in range(1, len(rows))]
            + [
                (i, j)
                for i, n in enumerate(names)
                if n in {"drift", "diversegrpo"}
                for j, m in enumerate(names)
                if m == "joint"
            ]
        )
    )
    tolerance = 1 / rows[0][0]["k"]
    for i, j in targets:
        mask = (
            np.isfinite(cond[i])
            & np.isfinite(cond[j])
            & (np.abs(sem[i] - sem[j]) <= tolerance)
            & (np.abs(qual[i] - qual[j]) <= tolerance)
        )
        delta = cond[j] - cond[i]
        boot = []
        for _ in range(repetitions):
            ix = rng.integers(len(mask), size=len(mask))
            if mask[ix].any():
                boot.append(float(delta[ix][mask[ix]].mean()))
        result["matched_semantic_quality"].append(
            dict(
                first=names[i],
                second=names[j],
                gate_tolerance=tolerance,
                prompts=int(mask.sum()),
                coverage=float(mask.mean()),
                delta_d_cond=mean_or_none(delta[mask]),
                ci95=np.quantile(boot, [0.025, 0.975]).tolist() if boot else None,
            )
        )
    directory = Path(output)
    write_json(directory / "comparison.json", result)
    if figures:
        grid(paths, rows, meta, pair_index, directory / "paired-seeds.png")
        plot_tradeoff(result, directory / "tradeoff.png", cube("rho"), cond, common, repetitions)
    return result


DIVERSITY = ("d_cond", "vendi")


def prompt_diversity(rows):
    table = {
        (r["pair"], b): {k: p[k] for k in DIVERSITY}
        for r in rows
        for b, p in enumerate(r["prompts"])
    }
    if len(table) != 2 * len(rows) or any(len(r["prompts"]) != 2 for r in rows):
        raise ValueError("Duplicate or incomplete evaluation prompt pair")
    return table


def common_diversity(prompt_metrics):
    """One intersection per metric across the entire cohort, keyed by prompt identity."""
    universe = prompt_metrics[0].keys()
    if not universe or any(m.keys() != universe for m in prompt_metrics[1:]):
        raise ValueError("Common diversity requires identical, nonempty prompt and seed sets")
    stats, means = {}, [{} for _ in prompt_metrics]
    for key in DIVERSITY:
        eligible = sorted(p for p in universe if all(m[p][key] is not None for m in prompt_metrics))
        stats[key] = dict(
            n_common=len(eligible),
            n_total=len(universe),
            coverage=len(eligible) / len(universe),
            prompt_digest=digest(eligible),
        )
        for m, result in zip(prompt_metrics, means):
            result[key + "_common"] = mean_or_none(m[p][key] for p in eligible)
    return stats, means


def seed_statistics(values):
    defined = [v for v in values if v is not None]
    return dict(
        mean=mean_or_none(defined),
        sd=stdev(defined) if len(defined) > 1 else None,
        n_defined=len(defined),
    )


def aggregate_methods(paths):
    meta = evaluation_metadata(paths)
    groups = defaultdict(list)
    protocols, seeds = {}, {}
    steps = set()
    for i, m in enumerate(meta):
        seed, method, ckpt = m["training_seed"], m["model"], m["checkpoint"]
        if method == "reference" and seed is None and ckpt is None:
            continue
        if seed is None or not ckpt or ckpt[1] < 1:
            raise ValueError(
                "Seed statistics require trained checkpoints with a recorded iteration"
            )
        groups[method].append(i)
        steps.add(ckpt[1])
    if len(steps) != 1:
        raise ValueError(
            f"Aggregate requires the same positive training iteration, got {sorted(steps)}"
        )
    for method, indices in groups.items():
        seeds[method] = [meta[i]["training_seed"] for i in indices]
        if len(indices) < 2 or len(set(seeds[method])) != len(indices):
            raise ValueError("Aggregate distinct training seeds of each variant")
        for i in indices:
            m = meta[i]
            protocol = m.get("training_protocol") or archived_protocol(
                paths[i], m["checkpoint"][0], method, m["training_seed"]
            )
            if (
                not protocol
                or not protocol.get("training_data")
                or protocol.get("variant") != method
                or not isinstance(protocol.get("history_baseline"), bool)
                or not protocol.get("config")
            ):
                raise ValueError(
                    f"Missing training provenance for {paths[i]}. Supply its original study manifests or re-evaluate a checkpoint with training metadata."
                )
            if protocols.setdefault(method, protocol) != protocol:
                raise ValueError(
                    f"Different training protocols for {method}: check data, hyperparameters and history baseline"
                )
    if (
        len({frozenset(s) for s in seeds.values()}) != 1
        or len({p["training_data"] for p in protocols.values()}) != 1
    ):
        raise ValueError("The comparison cohort must share training seeds and training data")

    # Discard individual image records before opening the next model's evaluation.
    summaries, tables = [], []
    for path in paths:
        rows = read_rows(Path(path) / "pairs.jsonl")
        tables.append(prompt_diversity(rows))
        summaries.append(summarize(rows))
        del rows
    masks, common = common_diversity(tables)
    coverage = dict(
        members=[dict(method=m["model"], seed=m["training_seed"]) for m in meta], **masks
    )
    metrics = (
        "preference",
        "quality_rate",
        "semantic_rate",
        "aux_d_cond",
        "valid_pair_fraction",
        "response_correct",
        "unchanged_pass",
        "pair_sr",
        "retention",
        "correction",
        "rho",
        "cov2",
        "d_cond",
        "d_eff",
        "vendi",
        "cov_m",
        "d_cond_common",
        "vendi_common",
    )
    for row, extra in zip(summaries, common):
        row.update(extra)
    return {
        method: dict(
            method=method,
            seeds=seeds[method],
            iteration=next(iter(steps)),
            training_protocol=protocols[method],
            common_support=coverage,
            per_seed_common=[dict(seed=meta[i]["training_seed"], **common[i]) for i in indices],
            metrics={key: seed_statistics(summaries[i][key] for i in indices) for key in metrics},
        )
        for method, indices in groups.items()
    }


def aggregate(paths, common_runs=()):
    cohort = [Path(p).resolve() for p in paths]
    primary = evaluation_metadata(cohort)
    if len({m["model"] for m in primary}) != 1 or any(m["training_seed"] is None for m in primary):
        raise ValueError("Aggregate distinct training seeds of one variant")
    method = primary[0]["model"]
    for p in common_runs:
        path = Path(p).resolve()
        if path not in cohort:
            extra = json.loads((path / "evaluation.json").read_text())
            if extra["model"] == method:
                raise ValueError("List every seed of the primary method in --runs")
            cohort.append(path)
    return aggregate_methods(cohort)[method]


def grid(paths, rows, meta, index, path):
    if not 0 <= index < len(rows[0]):
        raise ValueError("pair index outside the evaluation set")
    k = rows[0][index]["k"]
    cell, columns = 112, min(k, 8)
    blocks = (k + columns - 1) // columns
    h = 32 + blocks * (cell + 18)
    img = Image.new("RGB", (max(610, columns * cell), 2 * len(rows) * h), "white")
    pen = ImageDraw.Draw(img)
    for j, (root, run) in enumerate(zip(paths, rows)):
        for side, prompt in enumerate(run[index]["prompts"]):
            top = (2 * j + side) * h
            pen.text((4, top + 5), f"{meta[j]['model']} | {prompt['text']}", fill="black")
            for pos, sample in enumerate(prompt["samples"]):
                x, y = (pos % columns) * cell, top + 28 + (pos // columns) * (cell + 18)
                with Image.open(Path(root) / sample["image"]) as im:
                    img.paste(im.convert("RGB").resize((cell, cell)), (x, y))
                pen.text(
                    (x + 2, y + cell + 1),
                    f"#{pos + 1} {'valid' if sample['valid'] else 'failed'}",
                    fill="darkgreen" if sample["valid"] else "darkred",
                )
    img.save(path)


def plot_tradeoff(result, path, rates, distances, common, repetitions):
    from matplotlib import pyplot as plt

    rng = np.random.default_rng(2468)
    k = len(common)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, row in enumerate(result["runs"]):
        if row["d_cond_common"] is not None:
            ax.scatter(row["rho"], row["d_cond_common"])
            xs, ys = [], []
            for _ in range(repetitions):
                ix = rng.integers(k, size=k)
                mask = common[ix]
                if mask.any():
                    xs.append(float(rates[i, ix].mean()))
                    ys.append(float(distances[i, ix][mask].mean()))
            if xs:
                ax.hlines(row["d_cond_common"], *np.quantile(xs, [0.025, 0.975]), alpha=0.6)
                ax.vlines(row["rho"], *np.quantile(ys, [0.025, 0.975]), alpha=0.6)
            ax.annotate(
                f"{row['method']} ({i + 1})",
                (row["rho"], row["d_cond_common"]),
                xytext=(3, 4),
                textcoords="offset points",
            )
    ax.set(
        xlabel="Valid image rate",
        ylabel="Conditional distance on shared prompts",
        title=f"Shared prompt coverage {result['common_cov2']:.1%}",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def training_curves(paths, output):
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for path in paths:
        root = Path(path)
        rows = read_rows(root / "train.jsonl")
        if not rows:
            raise ValueError(f"Missing completed iterations in {path}")
        label = json.loads((root / "run.json").read_text())["variant"]

        def examples(row):
            counts = row["charged_counts"]
            return sum(
                counts.get(k, 0)
                for k in ("denoiser_examples", "replay_examples", "kl_reference_examples")
            )

        budget = [examples(r) for r in rows]
        axes[0, 0].plot(budget, [r["reward"] for r in rows], label=label)
        if rows[-1]["groups"][0]["reference_used"]:
            for idx, key in enumerate(("sem", "div")):
                axes[0, 1].plot(
                    budget, [r[key + "_violation"] for r in rows], label=f"{label} / {key}"
                )
                axes[1, 0].plot(
                    budget, [r["dual_used"][idx] for r in rows], label=f"{label} / {key}"
                )
        axes[1, 1].plot(budget, [r["clip_fraction"] for r in rows], label=label)
    axes[0, 1].axhline(0, color="grey", linestyle="--")
    for ax, title in zip(
        axes.flat, ("Task reward", "Cost minus budget", "Dual multiplier", "PPO clip fraction")
    ):
        ax.set(xlabel="Charged denoiser forward examples", ylabel=title)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
