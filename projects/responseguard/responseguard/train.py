"""On-policy collection, per-transition PPO epochs and one dual update per fresh batch."""

from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
from statistics import fmean, pvariance
from time import monotonic

import torch

from .baselines import concentrated_subset, drift_advantages, spectral_bonus, standardized
from .flow import clipped_loss, likelihood_ratio_terms, rollout, transition
from .io import atomic_path, digest, seed_for, write_json, write_rows
from .judges import PerceptualSpace, alt_scenes, make_judge
from .objective import History, group_value, update_dual
from .policy import make_policy
from .reference import ReferenceCache, budget_snapshot, initial_bank

VARIANTS = {
    "diversegrpo": (False, False),
    "drift": (False, False),
    "pair-grpo-adapted": (False, False),
    "joint": (True, True),
    "unprojected": (True, True),
    "semantic": (True, False),
    "diversity": (False, True),
    "task-only": (False, False),
    "per-prompt": (False, False),
    "margin-supervised": (False, False),
    "flow-grpo": (False, False),
}


def trainable(module):
    return {name: param for name, param in module.named_parameters() if param.requires_grad}


def load_adapter(policy, state):
    current = trainable(policy.module)
    if current.keys() != state.keys():
        raise ValueError("Checkpoint adapter parameters differ from this policy")
    with torch.no_grad():
        for name, param in current.items():
            if param.shape != state[name].shape:
                raise ValueError(f"Adapter shape mismatch for {name}")
            param.copy_(state[name].to(param))


def load_checkpoint(policy, path):
    state = torch.load(path, weights_only=True, map_location="cpu")
    for key in ("backend", "model", "revision", "lora_rank", "lora_alpha"):
        if state["config"][key] != getattr(policy.cfg, key):
            raise ValueError(f"Checkpoint and generation config differ in {key}")
    load_adapter(policy, state["adapter"])
    return state


def training_protocol(config, method, training_data, history_baseline):
    """Inputs that must agree across training seeds; evaluation has its own protocol."""
    return dict(
        config={k: v for k, v in config.items() if k != "seed"},
        variant=method,
        training_data=training_data,
        history_baseline=history_baseline,
    )


def archived_protocol(path, identity, method, seed):
    # Older study adapters contain no data fingerprint. Recover it only from the
    # matching run manifest and its study plan, never from the evaluation config.
    for root in Path(path).resolve().parents:
        manifest = root / "run.json"
        if not manifest.is_file():
            continue
        run = json.loads(manifest.read_text())
        if (run.get("identity"), run.get("variant"), run["config"].get("seed")) != (
            identity,
            method,
            seed,
        ):
            continue
        data = run.get("training_data")
        if not data and root.name == f"seed-{seed}" and root.parent.name == method:
            plan_path = root.parent.parent / "plan.json"
            if plan_path.is_file():
                plan = json.loads(plan_path.read_text())
                if method in plan.get("methods", []) and seed in plan.get("seeds", []):
                    expected = dict(plan["config"], seed=seed)
                    expected["lr"] = (plan.get("rates") or {}).get(method, expected["lr"])
                    if expected == run["config"]:
                        data = plan.get("training_data")
        if data and "history_baseline" in run:
            return training_protocol(run["config"], method, data, run["history_baseline"])
    return None


def replay(policy, optimizer, batches, cfg, variant):
    flow_grpo = variant in {"flow-grpo", "diversegrpo", "drift", "pair-grpo-adapted"}
    loss_sum = kl_sum = fraction = 0.0
    normals = []
    trainables = list(trainable(policy.module).values())
    for epoch in range(cfg.ppo_epochs):
        optimizer.zero_grad(set_to_none=True)
        for traces, advantages in batches:
            for tr, adv in zip(traces, advantages):
                k, steps = len(adv), len(tr.means)
                for t, old_means in enumerate(tr.means):
                    sigma, following = tr.sigmas[t : t + 2]
                    for start in range(0, k, cfg.microbatch):
                        cut = slice(start, min(start + cfg.microbatch, k))
                        x = tr.states[t][cut].to(policy.device)
                        action = tr.states[t + 1][cut].to(policy.device)
                        old = old_means[cut].to(policy.device)
                        examples = len(x) * (2 if cfg.backend == "sd3" and cfg.cfg_scale > 1 else 1)
                        condition = tr.condition(t, cut, cfg)
                        wasserstein = variant == "diversegrpo" and t < cfg.diverse_early_steps
                        reference_kl = flow_grpo and variant != "diversegrpo" and cfg.flow_kl > 0
                        if reference_kl or wasserstein:
                            with torch.no_grad(), policy.reference():
                                ref_mu, _ = transition(
                                    x,
                                    policy.velocity(x, sigma, condition),
                                    sigma,
                                    following,
                                    tr.sigmas[1],
                                    cfg.sde_noise,
                                )
                            policy.counts.update(
                                kl_reference_calls=1, kl_reference_examples=examples
                            )
                        mu, std = transition(
                            x,
                            policy.velocity(x, sigma, condition),
                            sigma,
                            following,
                            tr.sigmas[1],
                            cfg.sde_noise,
                        )
                        log_r, kl = likelihood_ratio_terms(action, old, mu, std)
                        # Released Flow-GRPO code averages transition log densities over latent
                        # coordinates. The ResponseGuard objective uses the full transition density.
                        clip_eps = cfg.flow_clip_eps if flow_grpo else cfg.clip_eps
                        clipped_log_r = log_r / x[0].numel() if flow_grpo else log_r
                        loss = clipped_loss(clipped_log_r, adv[cut].to(policy.device), clip_eps)
                        if wasserstein:
                            loss = (
                                loss + cfg.diverse_wasserstein * (mu - ref_mu).square().mean() / 2
                            )
                        if reference_kl:
                            loss = loss + cfg.flow_kl * ((mu - ref_mu) / std).square().mean() / 2
                        weight = len(x) / (k * steps * 2 * len(batches))
                        (loss * weight).backward()
                        loss_sum += weight * float(loss.detach())
                        kl_sum += weight * float(kl.detach().mean())
                        outside = (clipped_log_r < math.log1p(-clip_eps)) | (
                            clipped_log_r > math.log1p(clip_eps)
                        )
                        fraction += weight * float(outside.detach().float().mean())
                        policy.counts.update(replay_calls=1, replay_examples=examples)
        norm = torch.nn.utils.clip_grad_norm_(
            trainables, cfg.max_grad_norm, error_if_nonfinite=True
        )
        normals.append(float(norm))
        optimizer.step()
    return {
        "loss": loss_sum / cfg.ppo_epochs,
        "kl_old_new": kl_sum / cfg.ppo_epochs,
        "clip_fraction": fraction / cfg.ppo_epochs,
        "gradient_norm": fmean(normals),
    }


def train(
    cfg, pairs, directory, cache_dir, variant="joint", resume=False, until=None, use_history=True
):
    from filelock import FileLock

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".train.lock"), timeout=0):
        return _train(cfg, pairs, root, cache_dir, variant, resume, until, use_history)


def _train(cfg, pairs, directory, cache_dir, variant, resume, until, use_history):
    if any(p.split != "train" for p in pairs):
        raise ValueError("Training requires the train partition")
    steps = cfg.iterations if until is None else until
    if not 1 <= steps <= cfg.iterations:
        raise ValueError("until must be within the configured iteration budget")
    engine, judge = make_policy(cfg), make_judge(cfg)
    cache = ReferenceCache(cache_dir, cfg)
    enabled = VARIANTS[variant]
    space = PerceptualSpace(cfg) if variant in {"diversegrpo", "drift"} else None
    epoch_number, permutation = -1, []
    optimizer = torch.optim.AdamW(
        trainable(engine.module).values(),
        lr=cfg.lr,
        betas=tuple(cfg.adam_betas),
        eps=cfg.adam_epsilon,
        weight_decay=cfg.weight_decay,
    )
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint, log_path = output / "checkpoint.pt", output / "train.jsonl"
    pair_ids = [p.key for p in pairs]
    training_data = digest(pair_ids)
    identity = digest([asdict(cfg), pair_ids, variant, use_history])
    history = History(cfg.history_window, cfg.scale_floor)
    dual = [float(w) if on else 0.0 for w, on in zip(cfg.dual_initial, enabled)]
    start_step = 0
    seen_ref = set()
    cumulative = Counter()
    charged = Counter()
    wall = ref_wall = 0.0
    if resume:
        state = load_checkpoint(engine, checkpoint)
        legacy = state["config"]
        compatible_legacy = (
            "pair_sampling" not in legacy
            and cfg.pair_sampling == "with_replacement"
            and digest(legacy) == digest({k: asdict(cfg)[k] for k in legacy})
            and state["identity"] == digest([legacy, [p.key for p in pairs], variant, use_history])
        )
        if identity != state["identity"] and not compatible_legacy:
            raise ValueError(
                "Resume requires the same config, dataset, baseline setting and variant"
            )
        optimizer.load_state_dict(state["optimizer"])
        start_step, dual = state["iteration"], state["dual"]
        history.records.extend(state["history"])
        seen_ref.update(state["seen_ref"])
        cumulative.update(state["cumulative"])
        charged.update(state["charged_reference"])
        wall, ref_wall = state["wall_seconds"], state["reference_build_seconds"]
        # Ignore partial log segments after the last atomic checkpoint.
        with log_path.open() as stream:
            committed = [json.loads(next(stream)) for _ in range(start_step)]
        write_rows(log_path, committed)
    elif any((checkpoint.exists(), log_path.exists(), (output / "run.json").exists())):
        raise FileExistsError(
            f"Run already exists: {output}. Resume it or choose another output directory."
        )

    def save_state(iteration):
        with atomic_path(checkpoint) as tmp:
            torch.save(
                {
                    "format": 1,
                    "config": asdict(cfg),
                    "identity": identity,
                    "training_data": training_data,
                    "history_baseline": use_history,
                    "variant": variant,
                    "adapter": {k: p.detach().cpu() for k, p in trainable(engine.module).items()},
                    "optimizer": optimizer.state_dict(),
                    "iteration": iteration,
                    "dual": dual,
                    "history": list(history.records),
                    "seen_ref": sorted(seen_ref),
                    "cumulative": dict(cumulative),
                    "charged_reference": dict(charged),
                    "wall_seconds": wall,
                    "reference_build_seconds": ref_wall,
                },
                tmp,
            )

    if not resume:
        log_path.touch()
        save_state(0)
    write_json(
        output / "run.json",
        {
            "config": asdict(cfg),
            "variant": variant,
            "identity": identity,
            "training_data": training_data,
            "history_baseline": use_history,
            "reference_cache": cache.tag,
        },
    )
    for it in range(start_step, steps):
        begin = monotonic()
        before = budget_snapshot(engine, judge)
        build_before, seconds_before = cache.built_counts.copy(), cache.built_seconds
        rng = random.Random(seed_for("sample", cfg.seed, it))
        processed, batches, observations, diagnostics, adv_values = [], [], [], [], []
        for slot in range(cfg.groups_per_batch):
            if cfg.pair_sampling == "shuffled_epochs":
                epoch, index = divmod(it * cfg.groups_per_batch + slot, len(pairs))
                if epoch != epoch_number:
                    permutation = list(range(len(pairs)))
                    random.Random(seed_for("epoch", cfg.seed, epoch)).shuffle(permutation)
                    epoch_number = epoch
                pair = pairs[permutation[index]]
            else:
                pair = pairs[rng.randrange(len(pairs))]
            base, scale = history.baseline(pair.key, dual) if use_history else (0.0, 1.0)
            ref = cache.get(engine, judge, pair) if any(enabled) else None
            if ref is not None and pair.key not in seen_ref:
                seen_ref.add(pair.key)
                charged.update(ref["build_cost"])
                ref_wall += ref["build_seconds"]
            pool = ref["seeds"] if ref is not None else initial_bank(cfg, pair, "anchor", cfg.k_ref)
            indices = rng.sample(range(len(pool)), cfg.k_train)
            seeds = [pool[i] for i in indices]
            if variant == "drift":
                seeds.extend(seed_for("drift-extra", s, it, slot) for s in seeds.copy())
            obs, traces, perceptual = [], [], []
            for b, scene in enumerate(pair.scenes):
                sample = rollout(
                    engine,
                    scene,
                    seeds,
                    ("actor", cfg.seed, it, slot, b),
                    cfg.train_steps,
                    record=True,
                    noisy_prompts=(variant == "drift"),
                )
                if space is not None:
                    features = space.embed(sample.frames)
                    judge.counts.update(training_perceptual_images=len(features))
                    if variant == "drift":
                        selected = concentrated_subset(features, cfg.k_train)
                        sample.frames = [sample.frames[i] for i in selected]
                        sample.trace = sample.trace.select(selected)
                        features = features[selected]
                    perceptual.append(features)
                obs.append(
                    judge.observe(
                        sample.frames, scene, alt_scenes(scene, pair.factor, pair.scenes[1 - b])
                    )
                )
                traces.append(sample.trace)
            group, diag, image_rewards = group_value(
                pair, obs, ref, indices, cfg, variant != "unprojected"
            )
            if variant in {"flow-grpo", "per-prompt"}:
                advantages = standardized(image_rewards)
            elif variant == "margin-supervised":
                margins = torch.stack([v.margin(pair.factor) for v in obs])
                advantages = standardized(image_rewards + cfg.supervised_margin_weight * margins)
            elif variant == "pair-grpo-adapted":
                advantages = standardized(image_rewards.flatten()).view_as(image_rewards)
            elif variant == "diversegrpo":
                bonuses = torch.stack([spectral_bonus(f, cfg.diverse_clusters) for f in perceptual])
                advantages = standardized(image_rewards + cfg.diverse_weight * bonuses)
            elif variant == "drift":
                advantages = drift_advantages(image_rewards, perceptual, cfg)
            else:
                signal = (group.utility(dual) - base) / scale
                advantages = torch.full_like(image_rewards, signal)
            if any(enabled) and cfg.k_train >= 3:
                # Influence diagnostic only. Each removal omits the matched initial seed on both sides.
                influence = []
                for omitted in range(cfg.k_train):
                    keep = [i for i in range(cfg.k_train) if i != omitted]
                    reduced, _, _ = group_value(
                        pair,
                        [e.select(keep) for e in obs],
                        ref,
                        [indices[i] for i in keep],
                        cfg,
                        variant != "unprojected",
                    )
                    influence.append(abs(group.utility(dual) - reduced.utility(dual)))
                diag["max_leave_one_seed_utility_change"] = max(influence)
            adv_values.extend(advantages.flatten().tolist())
            batches.append((traces, advantages.detach()))
            processed.append(pair.key)
            observations.append(group)
            diagnostics.append(
                dict(
                    pair=pair.key,
                    factor=pair.factor,
                    baseline=base,
                    scale=scale,
                    **asdict(group),
                    **diag,
                )
            )
        opt = replay(engine, optimizer, batches, cfg, variant)
        # The same fixed batch may be replayed several times; dual and history update only here.
        new_dual = update_dual(dual, observations, cfg, enabled)
        for key, value in zip(processed, observations):
            history.append(key, value)
        observed = budget_snapshot(engine, judge) - before
        observed.subtract(cache.built_counts - build_before)
        cumulative.update(observed)
        wall += monotonic() - begin - (cache.built_seconds - seconds_before)
        # Charge each used reference once, even when it was generated by a previous prepare run.
        training_devices = {
            str(torch.device(d if d != "cuda" else "cuda:0"))
            for d in (cfg.device, cfg.judge_device)
            if d.startswith("cuda")
        }
        row = {
            "iteration": it + 1,
            "reward": fmean(g.reward for g in observations),
            "sem_violation": fmean(g.sem for g in observations) - cfg.sem_budget
            if any(enabled)
            else None,
            "div_violation": fmean(g.div for g in observations) - cfg.div_budget
            if any(enabled)
            else None,
            "dual_used": dual,
            "dual_next": new_dual,
            "adv_variance": pvariance(adv_values),
            "groups": diagnostics,
            **opt,
            "online_counts": dict(cumulative),
            "reference_cost_once_per_pair": dict(charged),
            "reference_build_seconds": ref_wall,
            "charged_counts": dict(cumulative + charged),
            "online_wall_seconds": wall,
            "charged_device_hours": (wall + ref_wall) * len(training_devices) / 3600,
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        dual = new_dual
        if (it + 1) % cfg.checkpoint_every == 0 or it + 1 == steps:
            save_state(it + 1)
        print(
            f"iteration {it + 1}/{steps}  reward={row['reward']:.4f}  KL={opt['kl_old_new']:.4g}",
            flush=True,
        )
    return checkpoint
