"""Measure semantic energy left by the learned projector on disjoint noise."""

from dataclasses import replace
from pathlib import Path
from statistics import fmean

import torch

from .data import FACTORS
from .flow import rollout
from .io import seed_for, write_json, write_rows
from .judges import alt_scenes, make_judge
from .objective import verified
from .policy import make_policy
from .reference import ReferenceCache


@torch.no_grad()
def inspect_projection(cfg, pairs, out, cache_dir, eval_cfg=None):
    if not pairs or any(p.split != "val" for p in pairs):
        raise ValueError("Projection and anchor audit requires validation pairs")
    ranks = sorted({1, 2, 4, len(FACTORS), cfg.projection_rank})
    fit_cfg = replace(cfg, projection_rank=max(ranks))
    policy, train_judge = make_policy(fit_cfg), make_judge(cfg)
    # Change the verifier but keep the feature coordinates in which U was fitted.
    check_cfg = replace(eval_cfg or cfg, eval_feature_model=cfg.feature_model)
    judge = make_judge(check_cfg, evaluation=True)
    cache = ReferenceCache(cache_dir, fit_cfg)
    floors, eligible = [], []
    rows, total_anchors, confirmed = [], 0, 0
    for pair in pairs:
        ref = cache.get(policy, train_judge, pair)
        floors.extend(ref["cal_rates"])
        eligible.extend(v is not None for v in ref["diversity"])
        seeds = [
            seed_for("projection-holdout", cfg.evaluation_seed, pair.key, i)
            for i in range(cfg.k_cal)
        ]
        rechecked = []
        for b, scene in enumerate(pair.scenes):
            other = pair.scenes[1 - b]
            matched = rollout(
                policy, scene, ref["seeds"], ("reference", pair.key, ("base", b)), cfg.train_steps
            )
            rechecked.append(
                judge.observe(matched.frames, scene, alt_scenes(scene, pair.factor, other))
            )
            stream = ("projection-audit", pair.key, b)
            base = rollout(policy, scene, seeds, (stream, "base"), cfg.train_steps)
            ev = judge.observe(base.frames, scene, alt_scenes(scene, pair.factor, other))
            u = ref["bases"][b]
            for factor in FACTORS:
                alt = other if factor == pair.factor else scene.neighbor(factor)
                changed = rollout(policy, alt, seeds, (stream, factor), cfg.train_steps)
                opposite = judge.observe(changed.frames, alt, alt_scenes(alt, factor, scene))
                ok = verified(ev, opposite, factor, check_cfg)
                fractions = {r: None for r in ranks}
                if int(ok.sum()) >= 2 and u.shape[1]:
                    direction = (opposite.features[ok] - ev.features[ok]).mean(0)
                    energy = float(direction.square().sum())
                    if energy > 1e-12:
                        for rank in ranks:
                            basis = u[:, :rank]
                            residual = direction - (direction @ basis) @ basis.T
                            fractions[rank] = float(residual.square().sum()) / energy
                rows.append(
                    dict(
                        pair=pair.key,
                        side=b,
                        factor=factor,
                        rank=min(cfg.projection_rank, u.shape[1]),
                        verified=int(ok.sum()),
                        k=len(seeds),
                        residual_energy_fraction=fractions[cfg.projection_rank],
                        rank_sweep=fractions,
                    )
                )
        ok = verified(*rechecked, pair.factor, check_cfg)
        total_anchors += int(ref["anchors"].sum())
        confirmed += int((ref["anchors"] & ok).sum())

    def summarize(group):
        values = [
            r["residual_energy_fraction"]
            for r in group
            if r["residual_energy_fraction"] is not None
        ]
        return dict(
            tested=len(group),
            defined=len(values),
            coverage=len(values) / len(group),
            mean_residual_fraction=fmean(values) if values else None,
        )

    usable = [r for r in rows if r["residual_energy_fraction"] is not None]
    sweep = {k: fmean(r["rank_sweep"][k] for r in usable) for k in ranks} if usable else {}
    selected = next((k for k in ranks if sweep and sweep[k] <= min(sweep.values()) + 0.05), None)
    result = dict(
        all=summarize(rows),
        validated_rank=selected,
        mean_residual_by_rank=sweep,
        rank_tolerance=0.05,
        reference_distance_coverage=fmean(eligible),
        reference_cal_rate=fmean(floors),
        reference_below_rho_floor=fmean(rate < cfg.rho_floor for rate in floors),
        by_factor={f: summarize([r for r in rows if r["factor"] == f]) for f in FACTORS},
        anchor_total=total_anchors,
        anchor_confirmed=confirmed,
        anchor_precision=confirmed / total_anchors if total_anchors else None,
        feature_space=cfg.feature_model if cfg.backend == "sd3" else "tiny_rgb_12",
        verifier=check_cfg.eval_vqa if cfg.backend == "sd3" else "simulator_truth",
    )
    write_rows(Path(out) / "interventions.jsonl", rows)
    write_json(Path(out) / "summary.json", result)
    return result
