"""Frozen, seed-indexed references with a separate pass-rate calibration bank."""

from collections import Counter
from pathlib import Path
from time import monotonic

import torch

from .data import FACTORS
from .flow import rollout
from .io import atomic_path, seed_for
from .judges import alt_scenes
from .objective import basis_from_directions, residual_diversity, verified

CACHE_VERSION = 2
CACHE_FIELDS = (
    "backend",
    "model",
    "revision",
    "train_vqa",
    "feature_model",
    "preference_model",
    "preference_processor",
    "revision_judges",
    "dtype",
    "resolution",
    "cfg_scale",
    "max_text_length",
    "reference_seed",
    "train_steps",
    "sde_noise",
    "k_ref",
    "k_cal",
    "projection_rank",
    "sem_threshold",
    "quality_threshold",
    "anchor_margin",
    "microbatch",
    "device",
    "judge_device",
    "judge_revisions",
)


def initial_bank(cfg, pair, role, size):
    return [seed_for(cfg.reference_seed, pair.key, role, i) for i in range(size)]


def budget_snapshot(policy, judge):
    return Counter(policy.counts) + Counter(judge.counts)


@torch.no_grad()
def build_reference(policy, judge, pair):
    cfg = policy.cfg
    start = monotonic()
    before = budget_snapshot(policy, judge)
    seeds = initial_bank(cfg, pair, "anchor", cfg.k_ref)
    cal_seeds = initial_bank(cfg, pair, "calibrate", cfg.k_cal)

    def score(scene, target, alt, bank, key):
        frames = rollout(policy, scene, bank, ("reference", pair.key, key), cfg.train_steps).frames
        return judge.observe(frames, scene, alt_scenes(scene, target, alt))

    bases, diversity, full_diversity, rates, valid_counts, verified_counts = [], [], [], [], [], []
    with policy.reference():
        main = [
            score(scene, pair.factor, pair.scenes[1 - b], seeds, ("base", b))
            for b, scene in enumerate(pair.scenes)
        ]
        anchors = verified(main[0], main[1], pair.factor, cfg)
        for b, (scene, ev) in enumerate(zip(pair.scenes, main)):
            directions, counts = [], {}
            for factor in FACTORS:
                if factor == pair.factor:
                    other = main[1 - b]
                    ok = anchors
                else:
                    other = score(
                        scene.neighbor(factor), factor, scene, seeds, ("neighbor", b, factor)
                    )
                    ok = verified(ev, other, factor, cfg)
                counts[factor] = int(ok.sum())
                if counts[factor] >= 2:
                    directions.append((other.features[ok] - ev.features[ok]).mean(0))
            u = basis_from_directions(directions, ev.features.shape[1], cfg.projection_rank)
            bases.append(u)
            verified_counts.append(counts)
            valid = ev.valid(cfg)
            valid_counts.append(int(valid.sum()))
            # No measured semantic direction or too few valid examples -> only the rate term.
            diversity.append(residual_diversity(ev.features, valid, u) if u.shape[1] else None)
            full_diversity.append(
                residual_diversity(ev.features, valid, u[:, :0]) if u.shape[1] else None
            )
            cal = score(scene, pair.factor, pair.scenes[1 - b], cal_seeds, ("calibrate", b))
            rates.append(float(cal.valid(cfg).float().mean()))

    return dict(
        seeds=seeds,
        anchors=anchors,
        margins=torch.stack([e.margin(pair.factor) for e in main]),
        bases=bases,
        diversity=diversity,
        full_diversity=full_diversity,
        valid_counts=valid_counts,
        verified_counts=verified_counts,
        cal_rates=rates,
        build_cost=dict(budget_snapshot(policy, judge) - before),
        build_seconds=monotonic() - start,
    )


class ReferenceCache:
    def __init__(self, root, cfg):
        self.tag = f"v{CACHE_VERSION}-{cfg.signature(*CACHE_FIELDS)[:24]}"
        self.root = Path(root) / self.tag
        self.built_counts = Counter()
        self.built_seconds = 0.0

    def get(self, policy, judge, pair):
        path = self.root / f"{pair.key}.pt"
        if path.exists():
            ref = torch.load(path, weights_only=True, map_location="cpu")
            if ref.get("pair_id") != pair.key or ref.get("cache_tag") != self.tag:
                raise ValueError(f"Reference cache identity mismatch in {path}")
            return ref
        ref = build_reference(policy, judge, pair)
        self.built_counts.update(ref["build_cost"])
        self.built_seconds += ref["build_seconds"]
        ref.update(pair_id=pair.key, cache_tag=self.tag)
        with atomic_path(path) as tmp:
            torch.save(ref, tmp)
        return ref
