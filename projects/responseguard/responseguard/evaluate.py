"""Evaluation with an independently selected semantic verifier and perceptual distance."""

from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from time import monotonic

import torch
import torch.nn.functional as F
from PIL import Image

from .data import FACTORS
from .flow import rollout
from .io import digest, read_rows, seed_for, write_json, write_rows
from .judges import PerceptualSpace, alt_scenes, make_judge
from .policy import Frame, make_policy
from .reference import budget_snapshot
from .train import archived_protocol, load_checkpoint, training_protocol


def mean_or_none(values):
    values = [v for v in values if v is not None]
    return fmean(values) if values else None


def ratio(a, b):
    return a / b if b else None


def vendi(vectors):
    norm = F.normalize(vectors.double(), dim=-1)
    gram = norm @ norm.T / len(norm)
    vals = torch.linalg.eigvalsh(gram).clamp_min(0)
    vals = vals[vals > 0]
    return float((-(vals * vals.log()).sum()).exp())


def diversity_metrics(embeddings, valid, m, priorities):
    """Average conditional distance and effective diversity with the same distance matrix."""
    k = len(valid)
    indices = torch.where(valid)[0].tolist()
    n = len(indices)
    row = {
        "n_valid": n,
        "k": k,
        "rho": n / k,
        "valid_pair_fraction": n * (n - 1) / (k * (k - 1)),
        "d_cond": None,
        "d_eff": 0.0,
        "vendi": None,
    }
    if n >= 2:
        x = F.normalize(embeddings[indices].double(), dim=-1)
        upper = torch.triu_indices(n, n, 1)
        distances = (1 - x @ x.T).clamp(0, 2)[upper[0], upper[1]]
        row["d_cond"] = float(distances.mean())
        row["d_eff"] = float(distances.sum()) / (k * (k - 1) / 2)
    if n >= m:
        chosen = sorted(indices, key=priorities.__getitem__)[:m]
        row["vendi"] = vendi(embeddings[chosen])
    return row


def summarize(rows):
    prompts = [p for row in rows for p in row["prompts"]]
    ref_correct = sum(r["ref_correct"] for r in rows)
    total = sum(r["k"] for r in rows)
    return dict(
        n_pairs=len(rows),
        n_prompts=len(prompts),
        pair_sr=sum(r["pair_correct"] for r in rows) / total,
        retention=ratio(sum(r["retained"] for r in rows), ref_correct),
        correction=ratio(sum(r["corrected"] for r in rows), total - ref_correct),
        ref_correct=ref_correct,
        ref_wrong=total - ref_correct,
        response_correct=sum(r["direction_correct"] for r in rows) / total,
        unchanged_pass=sum(r["unchanged_pass"] for r in rows) / total,
        valid_pair_fraction=fmean(
            p["n_valid"] * (p["n_valid"] - 1) / (p["k"] * (p["k"] - 1)) for p in prompts
        ),
        cov2=fmean(p["d_cond"] is not None for p in prompts),
        cov_m=fmean(p["vendi"] is not None for p in prompts),
        **{
            key: mean_or_none(p[key] for p in prompts)
            for key in (
                "rho",
                "semantic_rate",
                "quality_rate",
                "preference",
                "d_cond",
                "d_eff",
                "vendi",
                "aux_d_cond",
            )
        },
    )


def evaluation_seeds(cfg, pair):
    return [seed_for("eval-init", cfg.evaluation_seed, pair.key, i) for i in range(cfg.k_eval)]


def eval_plan(cfg, pairs):
    return [
        dict(pair=p.key, side=b, prompt=scene.text, seed=s, image=f"{p.key}/{b}_{i:02d}.png")
        for p in pairs
        for b, scene in enumerate(p.scenes)
        for i, s in enumerate(evaluation_seeds(cfg, p))
    ]


def evaluation_protocol(cfg, pairs, metric_space):
    protocol = cfg.signature(
        "backend",
        "model",
        "revision",
        "eval_vqa",
        "eval_feature_model",
        "eval_preference_model",
        "eval_preference_processor",
        "perceptual_metric",
        "revision_judges",
        "preference_model",
        "preference_processor",
        "resolution",
        "evaluation_seed",
        "k_eval",
        "eval_steps",
        "sde_noise",
        "cfg_scale",
        "max_text_length",
        "sem_threshold",
        "quality_threshold",
        "anchor_margin",
        "dtype",
        "vendi_m",
        "judge_revisions",
    )
    return digest(["evaluation-v2", protocol, metric_space, [p.key for p in pairs]])


@torch.no_grad()
def evaluate(cfg, pairs, directory, checkpoint=None, external=None, name=None):
    if any(p.split not in {"val", "test"} for p in pairs):
        raise ValueError("Evaluation requires a validation or test split")
    output = Path(directory)
    if (output / "summary.json").exists():
        raise FileExistsError(f"Completed evaluation already exists: {output}")
    policy, judge = make_policy(cfg), make_judge(cfg, evaluation=True)
    model_name = "reference"
    training_seed = None
    checkpoint_identity = None
    trained_with = None
    external_files = {}
    if external:
        if checkpoint or not name or cfg.backend != "sd3":
            raise ValueError("External images require the SD3 evaluation config and a method name")
        records = read_rows(external)
        planned = {(r["pair"], r["side"], r["seed"]): r["prompt"] for r in eval_plan(cfg, pairs)}
        if any(r.get("prompt") != planned.get((r["pair"], r["side"], r["seed"])) for r in records):
            raise ValueError("External captions must match the registered evaluation prompts")
        external_files = {
            (r["pair"], r["side"], r["seed"]): Path(external).parent / r["image"] for r in records
        }
        expected = {(r["pair"], r["side"], r["seed"]) for r in eval_plan(cfg, pairs)}
        if len(external_files) != len(records) or set(external_files) != expected:
            raise ValueError(
                "External manifest must contain every planned pair, side and seed exactly once"
            )
        model_name = name
    if checkpoint:
        state = load_checkpoint(policy, checkpoint)
        model_name = state["variant"]
        training_seed = state["config"]["seed"]
        checkpoint_identity = [state["identity"], state["iteration"]]
        if "training_data" in state:
            trained_with = training_protocol(
                state["config"], model_name, state["training_data"], state["history_baseline"]
            )
        else:
            trained_with = archived_protocol(
                checkpoint, state["identity"], model_name, training_seed
            )
        del state
    space = PerceptualSpace(cfg)
    meta = {
        "protocol": evaluation_protocol(cfg, pairs, space.label),
        "metric_space": space.label,
        "preference_scorer": "tiny_contrast"
        if cfg.backend == "tiny"
        else cfg.eval_preference_model,
        "aux_feature_space": "tiny_rgb_12" if cfg.backend == "tiny" else cfg.eval_feature_model,
        "model": model_name,
        "training_seed": training_seed,
        "checkpoint": checkpoint_identity,
        "training_protocol": trained_with,
        "config": asdict(cfg),
    }
    write_json(output / "evaluation.json", meta)
    rows = []
    started = monotonic()
    for pair in pairs:
        seeds = evaluation_seeds(cfg, pair)
        prompts, old_ok, now_ok, direction_ok, unchanged = [], [], [], [], []
        for side, scene in enumerate(pair.scenes):
            stream = ("eval-path", cfg.evaluation_seed, pair.key, side)
            alternatives = alt_scenes(scene, pair.factor, pair.scenes[1 - side])
            if external:
                frames = []
                for s in seeds:
                    with Image.open(external_files[pair.key, side, s]) as im:
                        frames.append(Frame(im.convert("RGB")))
            else:
                frames = rollout(policy, scene, seeds, stream, cfg.eval_steps).frames
            obs = judge.observe(frames, scene, alternatives)
            if checkpoint or external:
                with policy.reference():
                    base = rollout(policy, scene, seeds, stream, cfg.eval_steps)
                ref = judge.observe(base.frames, scene, alternatives)
            else:
                ref = obs
            old_ok.append(ref.semantic(cfg))
            now_ok.append(obs.semantic(cfg))
            direction_ok.append(obs.margin(pair.factor) >= cfg.anchor_margin)
            same = [i for i, f in enumerate(FACTORS) if f != pair.factor]
            unchanged.append(obs.own[:, same].amin(1) >= cfg.sem_threshold)
            valid = obs.valid(cfg)
            emb = space.embed(frames)
            judge.counts.update(evaluation_perceptual_images=len(frames))
            priority = [seed_for("subsample", s) for s in seeds]
            metrics = diversity_metrics(emb, valid, cfg.vendi_m, priority)
            aux = diversity_metrics(obs.features, valid, cfg.vendi_m, priority)
            samples = []
            for i, frame in enumerate(frames):
                rel = Path("images") / pair.key / f"{side}_{i:02d}.png"
                (output / rel).parent.mkdir(parents=True, exist_ok=True)
                frame.image.save(output / rel)
                samples.append(
                    dict(
                        seed=seeds[i],
                        image=rel.as_posix(),
                        valid=bool(valid[i]),
                        semantic_ok=bool(now_ok[-1][i]),
                        reference_semantic_ok=bool(old_ok[-1][i]),
                        atoms=dict(zip(FACTORS, obs.own[i].tolist())),
                        margin=float(obs.margin(pair.factor)[i]),
                        quality=float(obs.quality[i]),
                        preference=float(obs.preference[i]),
                    )
                )
            prompts.append(
                dict(
                    text=scene.text,
                    samples=samples,
                    aux_d_cond=aux["d_cond"],
                    semantic_rate=float(now_ok[-1].float().mean()),
                    quality_rate=float((obs.quality >= cfg.quality_threshold).float().mean()),
                    preference=float(obs.preference.mean()),
                    **metrics,
                )
            )
        current, reference = now_ok[0] & now_ok[1], old_ok[0] & old_ok[1]
        rows.append(
            dict(
                pair=pair.key,
                factor=pair.factor,
                split=pair.split,
                complexity="count_1_2" if max(s.count for s in pair.scenes) <= 2 else "count_3_4",
                k=len(seeds),
                prompts=prompts,
                pair_correct=int(current.sum()),
                ref_correct=int(reference.sum()),
                retained=int((current & reference).sum()),
                corrected=int((current & ~reference).sum()),
                direction_correct=int((direction_ok[0] & direction_ok[1]).sum()),
                unchanged_pass=int((unchanged[0] & unchanged[1]).sum()),
            )
        )
    write_rows(output / "pairs.jsonl", rows)
    result = {
        "metric_space": space.label,
        "all": summarize(rows),
        "by_factor": {
            f: summarize([r for r in rows if r["factor"] == f])
            for f in sorted({r["factor"] for r in rows})
        },
        "by_complexity": {
            c: summarize([r for r in rows if r["complexity"] == c])
            for c in sorted({r["complexity"] for r in rows})
        },
        "counts": dict(budget_snapshot(policy, judge)),
        "wall_seconds": monotonic() - started,
    }
    write_json(output / "summary.json", result)
    return result
