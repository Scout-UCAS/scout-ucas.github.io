"""Verified reference margins, local SVD projection, primal and dual statistics."""

from collections import deque
from dataclasses import dataclass
from statistics import fmean, pstdev

import torch


def verified(evidence, opposite, factor, cfg):
    return (
        evidence.semantic(cfg)
        & opposite.semantic(cfg)
        & (evidence.margin(factor) >= cfg.anchor_margin)
        & (opposite.margin(factor) >= cfg.anchor_margin)
    )


def basis_from_directions(directions, dim, rank):
    if not directions:
        return torch.empty(dim, 0)
    matrix = torch.stack(directions, dim=1).float()
    u, values, _ = torch.linalg.svd(matrix, full_matrices=False)
    tol = max(matrix.shape) * torch.finfo(matrix.dtype).eps * values[0]
    kept = min(rank, int((values > tol).sum()))
    return u[:, :kept].contiguous()


def residual_diversity(features, mask, basis):
    selected = features[mask]
    if len(selected) < 2:
        return None
    residual = selected - (selected @ basis) @ basis.T
    return float(torch.pdist(residual).square().mean())


@dataclass(frozen=True)
class GroupValue:
    reward: float
    sem: float
    div: float

    def utility(self, dual):
        return self.reward - dual[0] * self.sem - dual[1] * self.div


def group_value(pair, obs, reference, indices, cfg, use_projection=True):
    task = torch.stack([v.reward(cfg) for v in obs])
    current = torch.stack([v.margin(pair.factor) for v in obs])
    n = len(indices)
    if reference is None:
        return GroupValue(float(task.mean()), 0.0, 0.0), {"reference_used": False}, task
    # Reference margin rows and masks are indexed by exactly the initial seeds used in this rollout.
    anchors = reference["anchors"][indices]
    deficit = (reference["margins"][:, indices] - cfg.margin_slack - current).clamp_min(0)
    sem = float((deficit.square() * anchors.unsqueeze(0)).sum() / n)
    div, rows = [], []
    for b, ev in enumerate(obs):
        valid = ev.valid(cfg)
        rho = float(valid.float().mean())
        threshold = max(cfg.rho_floor, reference["cal_rates"][b] - cfg.rho_slack)
        valid_cost = cfg.rate_weight * max(0.0, threshold - rho) ** 2
        basis = reference["bases"][b]
        if not use_projection:
            basis = basis[:, :0]
        target = reference["diversity" if use_projection else "full_diversity"][b]
        value = residual_diversity(ev.features, valid, basis) if target is not None else None
        gap = max(0.0, cfg.kappa * target - value) if value is not None else 0.0
        div.append(valid_cost + gap**2)
        rows.append(
            dict(
                valid=int(valid.sum()),
                rho=rho,
                rho_min=threshold,
                d_valid=value,
                d_ref=target,
                rank=basis.shape[1],
                verified_counts=reference["verified_counts"][b],
                ref_valid=reference["valid_counts"][b],
                rate_cost=valid_cost,
                distance_cost=gap**2,
            )
        )
    return (
        GroupValue(float(task.mean()), sem, fmean(div)),
        {"reference_used": True, "anchor_count": int(anchors.sum()), "prompts": rows},
        task,
    )


class History:
    def __init__(self, window, scale_floor, records=()):
        self.records = deque(records, maxlen=window)
        self.floor = scale_floor

    def baseline(self, key, dual):
        if not self.records:
            return 0.0, 1.0
        values = [GroupValue(*r[1:]).utility(dual) for r in self.records]
        local = [v for r, v in zip(self.records, values) if r[0] == key]
        return fmean(local or values), max(self.floor, pstdev(values))

    def append(self, key, group):
        self.records.append((key, group.reward, group.sem, group.div))


def update_dual(dual, groups, cfg, enabled):
    costs = (fmean(g.sem for g in groups), fmean(g.div for g in groups))
    budgets = (cfg.sem_budget, cfg.div_budget)
    return [
        max(0.0, weight + lr * (cost - budget)) if use else 0.0
        for weight, lr, cost, budget, use in zip(dual, cfg.dual_lr, costs, budgets, enabled)
    ]
