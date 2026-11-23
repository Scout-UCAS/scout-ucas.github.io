"""Rectified-flow SDE transitions and exact step likelihood ratios."""

import math
from dataclasses import dataclass

import torch

from .io import seed_for


def transition(x, velocity, sigma, following, first_interior, noise_level):
    # Reverse time Euler–Maruyama. Regularize only the singular denominator at t=1.
    dt = following - sigma
    if not 0 <= following < sigma <= 1.000001:
        raise ValueError(f"Invalid descending flow schedule: {sigma} -> {following}")
    denom = 1 - (first_interior if sigma >= 1 else sigma)
    diffusion_sq = noise_level**2 * sigma / denom
    drift = velocity.float() + diffusion_sq / (2 * sigma) * (
        x.float() + (1 - sigma) * velocity.float()
    )
    return x.float() + dt * drift, math.sqrt(-dt * diffusion_sq)


def likelihood_ratio_terms(action, old_mean, new_mean, std):
    # Difference of Gaussian quadratic forms, summed over all latent coordinates.
    delta = (new_mean.float() - old_mean.float()) / std
    residual = (action.float() - old_mean.float()) / std
    kl = 0.5 * delta.square().flatten(1).sum(1)
    log_ratio = (residual * delta).flatten(1).sum(1) - kl
    return log_ratio, kl


def clipped_loss(log_ratio, advantage, epsilon):
    # Exact PPO min in log space for either sign. Detect numerical divergence explicitly.
    upper, lower = math.log1p(epsilon), math.log1p(-epsilon)
    effective = torch.where(advantage >= 0, log_ratio.clamp(max=upper), log_ratio.clamp(min=lower))
    objective = effective.to(torch.float32).exp() * advantage.detach()
    if not torch.isfinite(objective).all():
        raise FloatingPointError("Nonfinite PPO objective; lower the actor learning rate")
    return -objective.mean()


@dataclass
class Trace:
    sigmas: list
    states: list
    means: list
    cond: object
    prompt_seeds: list | None = None

    def select(self, indices):
        return Trace(
            self.sigmas,
            [s[indices] for s in self.states],
            [m[indices] for m in self.means],
            self.cond,
            [self.prompt_seeds[i] for i in indices] if self.prompt_seeds else None,
        )

    def condition(self, step, cut, cfg):
        return (
            perturbed_condition(self.cond, self.prompt_seeds[cut], step, self.sigmas[step], cfg)
            if self.prompt_seeds
            else self.cond
        )


@dataclass
class Rollout:
    frames: list
    trace: Trace | None


def normal(generators, shape, device):
    # CPU generators make initial and exploration streams independent of the microbatch size.
    return torch.stack([torch.randn(shape, generator=g) for g in generators]).to(device)


def perturbed_condition(cond, seeds, step, sigma, cfg):
    """DRIFT CADS schedule on the positive prompt embedding, with per-image moment matching."""
    weight = max(
        0.0, min(1.0, (cfg.drift_noise_end - sigma) / (cfg.drift_noise_end - cfg.drift_noise_start))
    )
    if weight == 1:
        return cond
    base = cond[0] if isinstance(cond, tuple) else cond
    gens = [torch.Generator().manual_seed(seed_for(s, step)) for s in seeds]
    noise = normal(gens, base.shape[1:], base.device)
    x = base.float() * math.sqrt(weight) + noise * cfg.drift_prompt_noise * math.sqrt(1 - weight)
    dims = tuple(range(1, x.ndim))
    z = (x - x.mean(dims, keepdim=True)) / x.std(dims, correction=0, keepdim=True).clamp_min(1e-8)
    x = z * base.float().std(dims, correction=0, keepdim=True) + base.float().mean(
        dims, keepdim=True
    )
    return (x.to(base), *cond[1:]) if isinstance(cond, tuple) else x.to(base)


@torch.no_grad()
def rollout(policy, scene, seeds, stream, steps, record=False, noisy_prompts=False):
    cfg = policy.cfg
    cond = policy.encode(scene)
    schedule = policy.schedule(steps)
    if len(schedule) != steps + 1 or not 0 < schedule[1] < 1:
        raise ValueError("Expected a descending SDE schedule with a finite first interior point")
    states = [[] for _ in schedule] if record else []
    means = [[] for _ in schedule[:-1]] if record else []
    frames = []
    prompt_seeds = [seed_for("prompt", stream, s) for s in seeds] if noisy_prompts else None
    for start in range(0, len(seeds), cfg.microbatch):
        indices = seeds[start : start + cfg.microbatch]
        branches = 2 if cfg.backend == "sd3" and cfg.cfg_scale > 1 else 1
        policy.counts.update(
            denoiser_calls=steps,
            denoiser_examples=steps * len(indices) * branches,
            generated_images=len(indices),
        )
        initial = [torch.Generator().manual_seed(int(s)) for s in indices]
        explore = [
            torch.Generator().manual_seed(seed_for("explore", stream, int(s))) for s in indices
        ]
        x = normal(initial, policy.latent_shape, policy.device)
        if record:
            states[0].append(x.cpu())
        for t, (sigma, following) in enumerate(zip(schedule, schedule[1:])):
            condition = (
                perturbed_condition(cond, prompt_seeds[start : start + len(indices)], t, sigma, cfg)
                if noisy_prompts
                else cond
            )
            mu, std = transition(
                x,
                policy.velocity(x, sigma, condition),
                sigma,
                following,
                schedule[1],
                cfg.sde_noise,
            )
            x = mu + std * normal(explore, policy.latent_shape, policy.device)
            if record:
                states[t + 1].append(x.cpu())
                means[t].append(mu.cpu())
        frames.extend(policy.decode(x))
    trace = (
        Trace(
            schedule,
            [torch.cat(chunks) for chunks in states],
            [torch.cat(chunks) for chunks in means],
            cond,
            prompt_seeds,
        )
        if record
        else None
    )
    return Rollout(frames, trace)
