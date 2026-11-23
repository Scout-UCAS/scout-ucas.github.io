"""Supervised warm-up, RLOO and optional GRPO on joint continuous action densities."""
from __future__ import annotations

import torch
from torch.distributions import Normal, kl_divergence

from .curriculum import make_tests, utilities


def leave_one_out(returns):
    k = returns.shape[-1]
    if k < 2:
        raise ValueError("A leave-one-out group needs at least two candidates")
    return returns - (returns.sum(dim=-1, keepdim=True) - returns)/(k-1)


def take_step(parameters, optimizer, max_norm):
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
    optimizer.step()
    return float(norm)


def clipped(logp, old, advantage, epsilon):
    ratio = (logp - old).clamp(-20, 20).exp()
    return torch.minimum(ratio * advantage, ratio.clamp(1 - epsilon, 1 + epsilon) * advantage)


def supervised_step(model, prefixes, tests, optimizer, parameters, cfg, *, joint):
    """Stream reader minibatches and propagate their summed dL/dZ once.

    This preserves the exact sequence-sum/batch-mean objective, even when a
    single reparameterized trajectory is read by several suffixes.
    """
    optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(joint):
        _, z = model.sample(prefixes, method=cfg.method)
    bridge = z.detach().requires_grad_(joint and z.requires_grad)
    owners, prompts, answers = [], [], []
    for i in range(len(prefixes)):
        owners.extend([i]*len(tests.suffixes[i]))
        prompts.extend(tests.suffixes[i])
        answers.extend(tests.answers[i])
    value = 0.0
    for start in range(0, len(answers), cfg.forward_batch):
        end = min(start+cfg.forward_batch, len(answers))
        loss = model.answer_loss(bridge[owners[start:end]], prompts[start:end], answers[start:end])
        weighted = loss * ((end-start)/len(answers))
        weighted.backward()
        value += weighted.detach().item()
    if bridge.requires_grad:
        z.backward(bridge.grad)
    take_step(parameters, optimizer, cfg.grad_clip)
    return value


def policy_step(model, pairs, bank, cfg, rng, oracle, optimizer):
    prefixes = [p.prefix(side) for p in pairs for side in (0, 1)]
    def draw_tests():
        return make_tests(pairs, cfg.m, bank, cfg.rho, rng, oracle,
                          single=cfg.method == "single-readout-rl")
    return group_update(model, prefixes, draw_tests, cfg, optimizer)


def group_update(model, prefixes, draw_tests, cfg, optimizer, grader=None):
    """One fresh group per prefix. Statistics for old/ref remain frozen."""
    owners = [i for i in range(len(prefixes)) for _ in range(cfg.k)]
    repeated = [prefixes[i] for i in owners]
    batch = cfg.forward_batch
    blocks = [slice(i, min(i+batch, len(repeated))) for i in range(0, len(repeated), batch)]
    with torch.no_grad():
        sampled = [model.sample(repeated[b]) for b in blocks]
        u = torch.cat([pair[0] for pair in sampled])
        z = torch.cat([pair[1] for pair in sampled])
        # The curriculum is fixed for the round. Draw suffixes *after* rollout.
        tests = draw_tests()
        task = utilities(model, z, owners, tests, batch, grader).view(len(prefixes), cfg.k)
        old = torch.cat([model.replay(repeated[b], u[b])[0] for b in blocks])
        ref_parts = [model.replay(repeated[b], u[b], "reference") for b in blocks]
        ref_log = torch.cat([v[0] for v in ref_parts])
        ref_mu = torch.cat([v[1].loc for v in ref_parts])
        ref_std = torch.cat([v[1].scale for v in ref_parts])
        kl_cost = (old-ref_log).sum(-1).view_as(task)
        if cfg.algorithm == "rloo":
            advantage = leave_one_out(task-cfg.beta*kl_cost)
        else:
            advantage = (task-task.mean(-1, keepdim=True)) / (task.std(-1, keepdim=True, unbiased=False)+1e-6)
        adv = advantage.reshape(-1, 1)

    parameters = model.parameters_for("policy")
    epochs = 1 if cfg.algorithm == "rloo" else cfg.grpo_epochs
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        value = 0.0
        for b in blocks:
            logp, normal = model.replay(repeated[b], u[b])
            if cfg.algorithm == "rloo":
                losses = -adv[b, 0] * logp.sum(-1)
            else:
                ref = Normal(ref_mu[b], ref_std[b])
                kl = kl_divergence(normal, ref).sum(-1)
                losses = (-clipped(logp, old[b], adv[b], cfg.clip_epsilon) + cfg.beta*kl).mean(-1)
            term = losses.sum()/len(repeated)
            term.backward()
            value += term.detach().item()
        grad_norm = take_step(parameters, optimizer, cfg.grad_clip)
    return tests, {"policy_loss": value, "task_reward": task.mean().item(),
                   "kl_sample": kl_cost.mean().item(), "adv_abs": advantage.abs().mean().item(),
                   "policy_grad_norm": grad_norm, "hard_draws": tests.hard_draws}
