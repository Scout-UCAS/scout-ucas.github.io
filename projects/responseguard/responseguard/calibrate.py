"""Validation-only audit sheets and gate calibration from human binary labels."""

import csv
from pathlib import Path

import torch

from .flow import rollout
from .io import atomic_path, seed_for
from .judges import make_judge, alt_scenes
from .policy import make_policy


@torch.no_grad()
def audit(cfg, pairs, directory, role, simulator_labels=False):
    if simulator_labels and cfg.backend != "tiny":
        raise ValueError("Simulator truth is available only on the tiny backend")
    if any(p.split != "val" for p in pairs):
        raise ValueError("Audit prompts must be in the validation split")
    output = Path(directory)
    sheet = output / "labels.csv"
    if sheet.exists():
        raise FileExistsError(f"Existing audit sheet: {sheet}")
    policy, judge = make_policy(cfg), make_judge(cfg, evaluation=(role == "eval"))
    rows, simulator = [], []
    with policy.reference():
        for pair in pairs:
            seeds = [
                seed_for("human-cal", cfg.reference_seed, pair.key, i) for i in range(cfg.k_cal)
            ]
            for b, scene in enumerate(pair.scenes):
                frames = rollout(
                    policy,
                    scene,
                    seeds,
                    ("audit", b, pair.key),
                    cfg.train_steps if role == "train" else cfg.eval_steps,
                ).frames
                ev = judge.observe(
                    frames, scene, alt_scenes(scene, pair.factor, pair.scenes[1 - b])
                )
                for i, frame in enumerate(frames):
                    rel = Path("images") / pair.key / f"{b}_{i}.png"
                    (output / rel).parent.mkdir(parents=True, exist_ok=True)
                    frame.image.save(output / rel)
                    rows.append(
                        dict(
                            image=rel.as_posix(),
                            prompt=scene.text,
                            role=role,
                            split="val",
                            semantic_score=float(ev.own[i].min()),
                            quality_score=float(ev.quality[i]),
                            semantic_ok="",
                            quality_ok="",
                        )
                    )
                    if simulator_labels:
                        simulator.append(
                            dict(
                                rows[-1],
                                semantic_ok=int(frame.scene == scene),
                                quality_ok=int(frame.quality >= 0.65),
                            )
                        )
    if simulator_labels:
        with (
            atomic_path(sheet.with_name("simulator-labels.csv")) as tmp,
            tmp.open("w", newline="", encoding="utf-8") as stream,
        ):
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(simulator)
    with atomic_path(sheet) as tmp, tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return sheet


def fit_gates(sheet):
    with Path(sheet).open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or {r["split"] for r in rows} != {"val"} or len({r["role"] for r in rows}) != 1:
        raise ValueError("Use a validation audit sheet for one verifier role")
    result = {}
    for name, key in (("semantic", "sem_threshold"), ("quality", "quality_threshold")):
        data = [
            (float(r[name + "_score"]), int(r[name + "_ok"]))
            for r in rows
            if r[name + "_ok"].strip()
        ]
        if {label for _, label in data} != {0, 1}:
            raise ValueError(f"Need human {name}_ok labels 0 and 1 to calibrate")
        good = sum(label for _, label in data)
        # Include all possible thresholds on the 0..1 score range; ties favor specificity.
        counts = {}
        for score, label in data:
            counts.setdefault(score, [0, 0])[label] += 1
        tp = fp = 0
        candidates = []
        for threshold in sorted({0.0, 1.0, *counts}, reverse=True):
            add_fp, add_tp = counts.get(threshold, (0, 0))
            tp += add_tp
            fp += add_fp
            specificity = 1 - fp / (len(data) - good)
            candidates.append(((tp / good + specificity) / 2, specificity, threshold))
        accuracy, _, threshold = max(candidates)
        result[key] = threshold
        result[name + "_balanced_accuracy"] = accuracy
        result[name + "_labeled"] = len(data)
    return result
