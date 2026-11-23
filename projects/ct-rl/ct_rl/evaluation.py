"""One sampled state per prefix, fixed shared tests, and paired bootstrap."""
from __future__ import annotations

import random
import statistics
from collections import defaultdict

import numpy as np
import torch

from .curriculum import answer_integer
from .experiment import meter
from .readout import Readout
from .settings import TEXT
from .settings import sync
from .tasks import Oracle, random_suffix, separating_read, stable_id


INTERVENTIONS = ("normal", "remove", "zeros", "shuffle", "swap", "prefix-cache")


def test_case(pair, m, seed, oracle):
    if m <= 0:
        raise ValueError("M must be positive")
    rng = random.Random(stable_id([seed, pair.group, pair.split]))
    suffixes = [random_suffix(pair, rng) for _ in range(m)]
    gold = oracle.labels(pair, suffixes)
    rng = random.Random(stable_id([seed, pair.group, pair.split, "probe"]))
    # A separate guaranteed discriminating probe, outside the M ordinary queries.
    # The probe must also obey this test split's composition/length condition.
    for _ in range(512):
        probe = random_suffix(pair, rng)
        expected = oracle.labels(pair, [probe])
        if expected[0] != expected[1]:
            break
    else:
        if pair.split in ("comp", "suffix_length"):
            raise RuntimeError(f"Could not find an in-split probe for {pair.group}")
        probe = separating_read(pair)
        expected = oracle.labels(pair, [probe])
    return {"group": pair.group, "family": pair.family, "split": pair.split,
            "prefixes": [pair.prefix(0), pair.prefix(1)],
            "suffixes": [" ; ".join(s) for s in suffixes], "gold": gold,
            "probe": " ; ".join(probe), "probe_gold": [expected[0][0], expected[1][0]]}


def intervene(z, mode, seed):
    if mode in ("normal", "prefix-cache"):
        return z
    if mode == "remove":
        return z[:, :0]
    if mode == "zeros":
        return torch.zeros_like(z)
    if mode == "swap":
        return z.flip(0)
    if mode == "shuffle":
        order = list(range(z.shape[1]))
        random.Random(seed).shuffle(order)
        if len(order) > 1 and order == list(range(len(order))):
            order = order[1:] + order[:1]
        return z[:, order]
    raise ValueError(mode)


def score_case(case, pred, probes):
    m = len(case["suffixes"])
    if len(pred) != 2 or any(len(row) != m for row in pred) or len(probes) != 2:
        raise ValueError(f"Wrong number of answers for {case['group']}")
    hit = [[answer_integer(a) == b for a,b in zip(row, gold)]
           for row, gold in zip(pred, case["gold"])]
    return {"group": case["group"], "family": case["family"],
            "ct_acc": sum(map(sum, hit))/(2*m), "all_correct": sum(map(all, hit))/2,
            "pair_both": float(all(answer_integer(a) == b for a,b in zip(probes, case["probe_gold"]))),
            "invalid_rate": sum(answer_integer(a) is None for row in pred for a in row)/(2*m),
            "answers": pred, "probe_answers": probes}


def aggregate(rows, seed, bootstrap):
    if not rows:
        raise ValueError("An evaluation split needs at least one pair")
    groups = defaultdict(list)
    for row in rows:
        groups[row["family"]].append(row)
    metrics = ("ct_acc", "all_correct", "pair_both", "invalid_rate")
    by_family = {family: {key: statistics.mean(row[key] for row in items) for key in metrics}
                 for family, items in groups.items()}
    macro = {key: statistics.mean(item[key] for item in by_family.values()) for key in metrics}
    intervals = {}
    if bootstrap:
        rng = np.random.default_rng(seed)
        replicates = np.zeros((bootstrap, len(metrics)))
        for items in groups.values():
            values = np.array([[row[key] for key in metrics] for row in items])
            for i in range(bootstrap):
                sample = rng.integers(len(items), size=len(items))
                replicates[i] += values[sample].mean(axis=0)/len(groups)
        intervals = {key: np.quantile(replicates[:, i], [0.025, 0.975]).tolist()
                     for i, key in enumerate(metrics)}
    return {"pairs": len(rows), "macro": macro, "by_family": by_family,
            "paired_bootstrap_95": intervals}


@torch.no_grad()
def evaluate(model, pairs, cfg, m=4, mode="normal", bootstrap=200, serial=False, flops=False):
    if mode not in INTERVENTIONS or bootstrap < 0:
        raise ValueError("Unknown intervention or invalid bootstrap size")
    protocol = cfg.method if cfg.method in TEXT else "latent"
    if protocol in TEXT and mode != "normal":
        raise ValueError("Text baselines require normal full-prefix evaluation")
    device = model.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # Keep the evaluation draw independent of the caller's current RNG state.
    torch.manual_seed(cfg.eval_seed)
    model.work.clear()
    model.peak_cache_bytes = 0
    oracle = Oracle()
    times = {key: 0.0 for key in ("prefix", "continuations", "probe")}
    ops = {key: 0 for key in times}
    rows = []
    visible_tokens = 0
    batch_size = 1 if serial else cfg.forward_batch
    for pair in pairs:
        case = test_case(pair, m, cfg.eval_seed, oracle)
        record = {}
        with meter(device, flops, record):
            readout = Readout(model, cfg, case["prefixes"], protocol, serial=serial)
            z = readout.state = intervene(readout.state, mode, pair.group)
        times["prefix"] += record["seconds"]
        ops["prefix"] += record["operator_flops"] or 0
        def decode(owners, prompts):
            if mode != "prefix-cache":
                return readout.answers(owners, prompts)
            out = []
            for start in range(0, len(prompts), batch_size):
                ix = owners[start:start+batch_size]
                out.extend(model.diagnostic_with_prefix_cache(
                    [case["prefixes"][i] for i in ix], z[ix], prompts[start:start+batch_size]))
            return out
        with meter(device, flops, record):
            before_tokens = model.work["generated_answer_tokens"]
            pred = decode([0]*m+[1]*m, case["suffixes"]*2)
            visible_tokens += model.work["generated_answer_tokens"]-before_tokens
        times["continuations"] += record["seconds"]
        ops["continuations"] += record["operator_flops"] or 0
        with meter(device, flops, record):
            probes = decode([0, 1], [case["probe"]]*2)
        times["probe"] += record["seconds"]
        ops["probe"] += record["operator_flops"] or 0
        rows.append(score_case(case, [pred[:m], pred[m:]], probes))
    result = aggregate(rows, cfg.eval_seed, bootstrap)
    count = 2*m*len(rows)
    result.update(split=pairs[0].split, method=cfg.method, training_seed=cfg.seed,
                  eval_seed=cfg.eval_seed, m=m, intervention=mode, model_serial=serial,
                  interface="full-prefix text" if cfg.method in TEXT else "strict latent",
                  costs={"seconds": times, "operator_flops": ops if flops else None,
                         "amortized_seconds_per_query": (times["prefix"]+times["continuations"])/count,
                         "amortized_flops_per_query": (ops["prefix"]+ops["continuations"])/count if flops else None,
                         "visible_output_tokens_per_query": visible_tokens/count,
                         "interface_bytes_per_prefix": z[0].numel()*z.element_size(),
                         "peak_kv_tensor_bytes": model.peak_cache_bytes,
                         "model_work_including_probe": dict(model.work),
                         "cuda_peak_bytes": torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
                         "oracle_calls": oracle.calls, "oracle_cpu_seconds": oracle.seconds},
                  rows=rows)
    return result
