"""Checkpoints contain the frozen reference, training-only pool and RNG state."""
from __future__ import annotations

import json
import random
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import torch

from .curriculum import cohort, make_tests, mine, sampling_pairs
from .learning import policy_step, supervised_step
from .comparisons import build_model, texts_for
from .comparison_learning import comparison_pg, special_supervision
from .settings import LATENT_RL, PROCESS, TEXT
from .settings import Config, SFT, atomic_path, digest, read_split, runtime, sync, versions, write_json
from .tasks import Oracle, stable_id


def attention_flops(query, key, value, *args, **kwargs):
    # Each query head computes QK and PV, including when groups share K/V.
    batch, heads, queries, width = query
    return 2 * batch * heads * queries * key[2] * (width + value[3])


def attention_backward_flops(grad, query, key, value, *args, **kwargs):
    # PyTorch's fused-kernel convention includes score recomputation plus four gradient products.
    batch, heads, queries, width = query
    return 2 * batch * heads * queries * key[2] * (3 * width + 2 * value[3])


@contextmanager
def meter(device, count_flops, record):
    from contextlib import nullcontext
    from torch.utils.flop_counter import FlopCounterMode
    counter = None
    if count_flops:
        kernels = {}
        for backend in ("flash_attention_for_cpu", "flash_attention", "efficient_attention", "cudnn_attention"):
            name = "_scaled_dot_product_" + backend
            for suffix, formula in (("", attention_flops), ("_backward", attention_backward_flops)):
                op = getattr(torch.ops.aten, name + suffix, None)
                if op is not None:
                    kernels[op] = formula
        counter = FlopCounterMode(display=False, custom_mapping=kernels)
    sync(device)
    started = time.perf_counter()
    with (counter if counter is not None else nullcontext()):
        yield counter
    sync(device)
    record["seconds"] = time.perf_counter()-started
    record["operator_flops"] = counter.get_total_flops() if counter is not None else None


def get_rng(device, rng):
    state = {"python": rng.getstate(), "torch": torch.get_rng_state()}
    if device.type == "cuda":
        state["device"] = torch.cuda.get_rng_state(device)
    elif device.type == "mps":
        state["device"] = torch.mps.get_rng_state()
    return state


def set_rng(state, device, rng):
    rng.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and "device" in state:
        torch.cuda.set_rng_state(state["device"], device)
    elif device.type == "mps" and "device" in state:
        torch.mps.set_rng_state(state["device"])


def save_checkpoint(file, obj):
    with atomic_path(file) as tmp:
        torch.save(obj, tmp)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def load_for_inference(path, device=None):
    state = load_checkpoint(path)
    cfg = Config.from_dict(state["config"])
    dev = runtime(cfg, device)
    model = build_model(cfg).to(dev)
    model.restore(state["model"])
    default_mode = cfg.method if cfg.method in TEXT else ("native" if cfg.native_pretrain else "latent")
    model.readout_mode = state.get("public", {}).get("mode", default_mode)
    return cfg, model


@contextmanager
def run_lock(out):
    import fcntl
    out.mkdir(parents=True, exist_ok=True)
    with (out/".train.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as err:
            raise ValueError(f"Another training process owns {out}") from err
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def train(cfg: Config, data: Path, out: Path, resume: bool = False, count_flops: bool = False,
          stop_flops: float | None = None):
    with run_lock(out):
        return _train(cfg, data, out, resume, count_flops, stop_flops)


def _train(cfg, data, out, resume, count_flops, stop_flops):
    if stop_flops is not None and (not count_flops or stop_flops <= 0):
        raise ValueError("A positive training FLOP budget requires counting FLOPs")
    checkpoint = out/"checkpoint.pt"
    if checkpoint.exists() and not resume:
        raise ValueError(f"{checkpoint} exists. Use --resume or choose a fresh output.")
    old = load_checkpoint(checkpoint) if resume else None
    train_pairs = read_split(data, "train")
    dataset_id = digest(data/"manifest.json")
    if old:
        if dataset_id != old["dataset"]:
            raise ValueError("Checkpoint and data manifest differ")
        prior = Config.from_dict(old["config"])
        # Allow extra rounds and a new device, but keep the optimization recipe.
        prior.rounds, prior.device, prior.threads = cfg.rounds, cfg.device, cfg.threads
        cfg.model.revision = cfg.model.revision or prior.model.revision
        if prior != cfg or cfg.rounds < old["round"]:
            raise ValueError("Incompatible resume config")
    dev = runtime(cfg)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    rng = random.Random(cfg.seed)
    oracle = Oracle()
    model = build_model(cfg).to(dev)
    cfg.model.revision = model.resolved_revision or cfg.model.revision
    write_json(out/"config.json", asdict(cfg))
    write_json(out/"run_info.json", {**versions(), "dataset_sha256": dataset_id,
                                    "device": str(dev), "model_revision": cfg.model.revision,
                                    "training_pairs": len(train_pairs), "seed": cfg.seed})
    policy, reader = model.parameters_for("policy"), model.parameters_for("reader")
    joint = policy + reader + list(model.mapper.parameters()) + list(model.static_head.parameters())
    warm_opt = torch.optim.AdamW(joint, lr=cfg.lr_warm, weight_decay=0.0)
    comparison_rl = cfg.method in LATENT_RL + ("loop-grpo", "rltt")
    policy_train = policy + reader if comparison_rl else policy
    actor_opt = torch.optim.AdamW(policy_train, lr=cfg.lr_policy, weight_decay=0.0)
    reader_opt = torch.optim.AdamW(reader, lr=cfg.lr_reader, weight_decay=0.0)
    history, bank, start = [], {}, 0

    def commit(round_no):
        state = {"format": 1, "round": round_no, "config": asdict(cfg), "model": model.portable_state(),
                 "dataset": dataset_id, "bank": bank, "history": history,
                 "actor_opt": actor_opt.state_dict(), "reader_opt": reader_opt.state_dict(),
                 "joint_opt": warm_opt.state_dict() if cfg.method in SFT else {},
                 "rng": get_rng(dev, rng), "versions": versions()}
        save_checkpoint(checkpoint, state)
        write_json(out/"history.json", history)
        write_json(out/"pool.json", bank)

    if resume:
        model.restore(old["model"])
        actor_opt.load_state_dict(old["actor_opt"])
        reader_opt.load_state_dict(old["reader_opt"])
        if cfg.method in SFT:
            warm_opt.load_state_dict(old["joint_opt"])
        history, bank, start = old["history"], old["bank"], old["round"]
        set_rng(old["rng"], dev, rng)
        del old
        write_json(out/"history.json", history)
        write_json(out/"pool.json", bank)
    else:
        info = {"stage": "warm", "steps": cfg.warm_steps}
        losses = []
        with meter(dev, count_flops, info):
            for step in range(cfg.warm_steps):
                pairs = rng.choices(train_pairs, k=cfg.pairs_per_batch)
                prefixes = [p.prefix(side) for p in pairs for side in (0, 1)]
                tests = make_tests(pairs, 1, {}, 0, rng, oracle)
                if cfg.method in PROCESS + TEXT + LATENT_RL:
                    fraction = step/max(1, cfg.warm_steps-1)
                    losses.append(special_supervision(model, cfg, prefixes, tests, texts_for(pairs),
                                                      warm_opt, joint, fraction, cfg.native_pretrain))
                else:
                    losses.append(supervised_step(model, prefixes, tests, warm_opt, joint, cfg, joint=True))
        model.snapshot_reference()
        info.update(answer_nll=sum(losses)/max(1, len(losses)), oracle_calls=oracle.calls,
                    oracle_cpu_seconds=oracle.seconds, model_work=dict(model.work))
        history.append(info)
        commit(0)
        print(json.dumps(info), flush=True)

    if cfg.method not in SFT:
        model.mapper.requires_grad_(False)
        warm_opt.state.clear()
    for round_no in range(start+1, cfg.rounds+1):
        if stop_flops is not None and sum(row["operator_flops"] for row in history) >= stop_flops:
            break
        # Only the bank from the *previous* round can influence rollout.
        metrics = Counter()
        oracle.calls, oracle.seconds = 0, 0.0
        model.work.clear()
        info = {"stage": "round", "round": round_no, "pool_before": len(bank)}
        previous_cohort = cohort(train_pairs, cfg.mine_pairs, cfg.seed, round_no-1)
        prefix_rng = random.Random(stable_id(["prefix-draws", cfg.seed, round_no]))
        with meter(dev, count_flops, info):
            for _ in range(cfg.steps_per_round):
                pairs = sampling_pairs(train_pairs, previous_cohort, cfg.pairs_per_batch, prefix_rng)
                prefixes = [p.prefix(side) for p in pairs for side in (0, 1)]
                if cfg.method in PROCESS + TEXT:
                    tests = make_tests(pairs, cfg.m, {}, 0, rng, oracle)
                    metrics["supervised_loss"] += special_supervision(
                        model, cfg, prefixes, tests, texts_for(pairs), warm_opt, joint, 1., cfg.native_pretrain)
                elif comparison_rl:
                    count = cfg.m if cfg.method in ("loop-grpo", "rltt") else 1
                    tests = make_tests(pairs, count, {}, 0, rng, oracle)
                    metrics.update(comparison_pg(model, cfg, prefixes, tests, actor_opt, policy_train,
                                                 native=cfg.native_pretrain))
                elif cfg.method in SFT:
                    count = 1 if cfg.method == "latent-sft" else cfg.m
                    tests = make_tests(pairs, count, {}, 0, rng, oracle)
                    metrics["answer_nll"] += supervised_step(model, prefixes, tests, warm_opt, joint,
                                                            cfg, joint=True)
                else:
                    tests, result = policy_step(model, pairs, bank, cfg, rng, oracle, actor_opt)
                    metrics.update(result)
                    metrics["answer_nll"] += supervised_step(model, prefixes, tests, reader_opt, reader,
                                                             cfg, joint=False)
            mining = {}
            work_at_mine = model.work.copy()
            sync(dev)
            began_mining = time.perf_counter()
            bank = mine(model, train_pairs, bank, cfg, rng, oracle, round_no)
            sync(dev)
            mining["seconds"] = time.perf_counter()-began_mining
            mining["model_work"] = dict(model.work-work_at_mine)
        info.update({k: v/cfg.steps_per_round for k,v in metrics.items()})
        info.update(pool_after=len(bank), oracle_calls=oracle.calls, oracle_cpu_seconds=oracle.seconds,
                    model_work=dict(model.work), mining=mining,
                    cuda_peak_bytes=torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None)
        history.append(info)
        commit(round_no)
        print(json.dumps(info), flush=True)
    return checkpoint
