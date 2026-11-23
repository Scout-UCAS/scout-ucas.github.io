"""Inference, official benchmark image export and free-form prompt evaluation."""

from dataclasses import asdict
import json
from pathlib import Path
from statistics import fmean
from time import monotonic

import torch

from .evaluate import diversity_metrics, mean_or_none
from .flow import rollout
from .io import atomic_path, digest, read_rows, seed_for, write_json, write_rows
from .judges import PerceptualSpace, make_judge
from .policy import make_policy
from .train import load_checkpoint


def inference_policy(cfg, checkpoint):
    policy = make_policy(cfg)
    info = None
    if checkpoint:
        state = load_checkpoint(policy, checkpoint)
        info = {"run": state["identity"], "iteration": state["iteration"]}
    return policy, info


def root_for_export(directory, metadata):
    output = Path(directory)
    path = output / "generation.json"
    if path.is_file() and json.loads(path.read_text()) != json.loads(json.dumps(metadata)):
        raise ValueError("Output directory is already associated with another generation protocol")
    write_json(path, metadata)
    return output


def save_png(frame, path):
    with atomic_path(path) as tmp:
        frame.image.save(tmp, format="PNG")


@torch.no_grad()
def generate_text(cfg, text, seeds, directory, checkpoint=None):
    if cfg.backend == "tiny":
        raise ValueError("Free-form text inference requires the SD3 backend")
    policy, identity = inference_policy(cfg, checkpoint)
    out = root_for_export(
        directory, dict(config=asdict(cfg), prompt=text, seeds=seeds, checkpoint=identity)
    )
    samples = rollout(policy, text, seeds, ("inference", text), cfg.eval_steps)
    for i, frame in enumerate(samples.frames):
        save_png(frame, out / f"{i:03d}_{seeds[i]}.png")


@torch.no_grad()
def benchmark_export(cfg, source, benchmark, directory, checkpoint=None):
    if cfg.backend == "tiny":
        raise ValueError("Official benchmarks require text-conditioned SD3 generation")
    records = read_rows(source) if benchmark == "geneval2" else json.loads(Path(source).read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a nonempty official prompt list")
    policy, identity = inference_policy(cfg, checkpoint)
    out = root_for_export(
        directory,
        dict(
            benchmark=benchmark,
            prompt_list=digest(records),
            checkpoint=identity,
            config=asdict(cfg),
        ),
    )
    mapping, delta_images = {}, []
    for j, row in enumerate(records):
        if benchmark == "geneval2":
            captions, k = [row["prompt"]], 1
        else:
            captions, k = [row["caption1"], row["caption2"]], 2
        seeds = [seed_for("benchmark", cfg.evaluation_seed, benchmark, j, i) for i in range(k)]
        for side, text in enumerate(captions):
            paths = [out / (f"{j}_{side}_{i}.png" if k == 2 else f"{j:05d}.png") for i in range(k)]
            if not all(path.exists() for path in paths):
                sample = rollout(
                    policy, text, seeds, ("benchmark", benchmark, j, side), cfg.eval_steps
                )
                for frame, path in zip(sample.frames, paths):
                    save_png(frame, path)
            if benchmark == "geneval2":
                mapping[text] = str(paths[0].resolve())
            elif benchmark == "deltabench":
                for i, path in enumerate(paths):
                    delta_images.append(
                        dict(
                            pair=j,
                            side=side,
                            index=i,
                            seed=seeds[i],
                            prompt=text,
                            category=row["category"],
                            image=path.name,
                            questions=row.get(f"questions{side + 1}", []),
                        )
                    )
    if benchmark == "geneval2":
        write_json(out / "image_paths.json", mapping)
    elif benchmark == "deltabench":
        write_rows(out / "images.jsonl", delta_images)
    return {
        "benchmark": benchmark,
        "prompts": len(records),
        "images_per_record": 1 if benchmark == "geneval2" else 4,
    }


def creative_prompts():
    scenes = (
        ("a potter shaping clay in a studio", "Is a potter shaping clay?"),
        ("a cyclist crossing a bridge", "Is a cyclist crossing a bridge?"),
        ("a violinist beside an open window", "Is a violinist beside an open window?"),
        ("a baker dusting dough with flour", "Is a baker dusting dough with flour?"),
        ("a heron wading in reeds", "Is a heron wading among reeds?"),
        ("a cat sleeping on folded linen", "Is a cat sleeping on folded linen?"),
        ("a fox beside a stream", "Is a fox beside a stream?"),
        ("a whale surfacing in the ocean", "Is a whale surfacing in the ocean?"),
        ("a camera on a wooden desk", "Is a camera on a wooden desk?"),
        ("a kettle on a campfire", "Is a kettle on a campfire?"),
        ("a compass on a map", "Is a compass on a map?"),
        ("a cello resting against a wall", "Is a cello resting against a wall?"),
        ("a sunlit railway station", "Is a sunlit railway station visible?"),
        ("a reading alcove lined with books", "Is a reading alcove lined with books?"),
        ("a greenhouse filled with ferns", "Is a greenhouse filled with ferns?"),
        ("a quiet kitchen at dawn", "Is a kitchen at dawn visible?"),
        ("a lighthouse on a cliff", "Is a lighthouse on a cliff?"),
        ("a market on a canal", "Is a market on a canal?"),
        ("a mountain path after rain", "Is a wet mountain path visible?"),
        ("a snowy courtyard at dusk", "Is a snowy courtyard at dusk visible?"),
    )
    styles = (
        "watercolor",
        "charcoal drawing",
        "oil painting",
        "cinematic photograph",
        "woodcut",
        "pencil sketch",
        "gouache painting",
        "ink illustration",
        "pastel drawing",
        "paper collage",
    )
    return [
        dict(prompt=f"A {style} depicting {scene}.", questions=[q, f"Is the image a {style}?"])
        for scene, q in scenes
        for style in styles
    ]


@torch.no_grad()
def evaluate_creative(cfg, source, directory, checkpoint=None):
    if cfg.backend == "tiny":
        raise ValueError("Creative prompts require a free-form text model and an image verifier")
    prompts = read_rows(source)
    if not prompts or any(not p.get("questions") for p in prompts):
        raise ValueError("Each creative prompt needs explicit yes/no checks for validity")
    policy, identity = inference_policy(cfg, checkpoint)
    out = root_for_export(
        directory,
        dict(
            benchmark="creative", prompts=digest(prompts), checkpoint=identity, config=asdict(cfg)
        ),
    )
    started = monotonic()
    judge, space = make_judge(cfg, evaluation=True), PerceptualSpace(cfg)
    rows = []
    for j, row in enumerate(prompts):
        seeds = [seed_for("creative", cfg.evaluation_seed, j, i) for i in range(cfg.k_eval)]
        frames = rollout(policy, row["prompt"], seeds, ("creative", j), cfg.eval_steps).frames
        atoms, qual, pref, _ = judge.score_images(frames, row["prompt"], row["questions"])
        semantic = atoms.amin(dim=1) >= cfg.sem_threshold
        valid = semantic & (qual >= cfg.quality_threshold)
        metrics = diversity_metrics(space.embed(frames), valid, cfg.vendi_m, seeds)
        judge.counts.update(evaluation_perceptual_images=len(frames))
        for i, frame in enumerate(frames):
            save_png(frame, out / "images" / f"{j:04d}_{i:02d}.png")
        rows.append(
            dict(
                prompt=row["prompt"],
                seeds=seeds,
                atoms=atoms.tolist(),
                quality=qual.tolist(),
                preference=float(pref.mean()),
                semantic_rate=float(semantic.float().mean()),
                quality_rate=float((qual >= cfg.quality_threshold).float().mean()),
                **metrics,
            )
        )
    write_rows(out / "creative.jsonl", rows)
    results = {
        key: mean_or_none(r[key] for r in rows)
        for key in (
            "preference",
            "semantic_rate",
            "quality_rate",
            "rho",
            "d_eff",
            "d_cond",
            "vendi",
        )
    }
    results.update(
        metric_space=space.label,
        preference_scorer=cfg.eval_preference_model,
        counts=dict(policy.counts + judge.counts),
        wall_seconds=monotonic() - started,
        cov2=fmean(r["n_valid"] >= 2 for r in rows),
        cov_m=fmean(r["n_valid"] >= cfg.vendi_m for r in rows),
    )
    write_json(out / "summary.json", results)
    return results
