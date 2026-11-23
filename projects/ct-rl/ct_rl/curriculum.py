"""Shared suffix tests and model-guided search on training pairs."""
from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass

import torch

from .tasks import READOUT, random_suffix, searches, stable_id


def answer_integer(text):
    if type(text) is int:
        return str(text)
    if not isinstance(text, str) or not re.fullmatch(r"-?(?:0|[1-9][0-9]*)", text.strip()):
        return None
    return str(int(text.strip()))


@dataclass
class TestSet:
    suffixes: list[list[str]]    # One list per distinct prefix, before K repeats
    answers: list[list[str]]
    hard_draws: int = 0


def cohort(pairs, count, seed, round_no):
    """The same model-independent search cohort for every experimental arm."""
    if round_no <= 0:
        return []
    rng = random.Random(stable_id(["cohort", seed, round_no]))
    return rng.sample(pairs, min(count, len(pairs)))


def sampling_pairs(pairs, previous_cohort, count, rng):
    return [rng.choice(previous_cohort) if previous_cohort and rng.random() < 0.5
            else rng.choice(pairs) for _ in range(count)]


def make_tests(pairs, count, bank, rho, rng, oracle, single=False):
    suffixes, answers, hard_draws = [], [], 0
    for pair in pairs:
        current = []
        entries = bank.get(pair.group, [])
        for _ in range(count):
            if single:
                s = (READOUT[pair.family],)
            elif entries and rng.random() < rho:
                s = tuple(rng.choice(entries)["suffix"])
                hard_draws += 1
            else:
                s = random_suffix(pair, rng)
            current.append(s)
        labels = oracle.labels(pair, current)
        text = [" ; ".join(s) for s in current]
        suffixes.extend([text, text])
        answers.extend(labels)
    return TestSet(suffixes, answers, hard_draws)


@torch.no_grad()
def batched_decode(model, z, indices, prompts, batch_size, max_tokens=None):
    out = []
    for i in range(0, len(prompts), batch_size):
        ix = indices[i:i+batch_size]
        out.extend(model.decode(z[ix], prompts[i:i+batch_size], max_tokens))
    return out


@torch.no_grad()
def utilities(model, z, source_indices, tests, batch_size, grader=None):
    """All K candidates for prefix p use tests.suffixes[p] in the same order."""
    owners, prompts, gold = [], [], []
    m = len(tests.suffixes[0])
    for i, source in enumerate(source_indices):
        owners.extend([i]*m)
        prompts.extend(tests.suffixes[source])
        gold.extend(tests.answers[source])
    pred = batched_decode(model, z, owners, prompts, batch_size)
    if grader is None:
        hits = [answer_integer(a) == b for a, b in zip(pred, gold)]
    else:
        hits = [grader(a, b) for a, b in zip(pred, gold)]
    return torch.tensor(hits, device=z.device, dtype=torch.float32).reshape(-1, m).mean(-1)


def choose_hard(entries, keep, matched, random_witness=False):
    buckets = defaultdict(list)
    for entry in entries:
        buckets[len(entry["suffix"])].append(entry)
    selected = []
    for bucket in buckets.values():
        if random_witness:
            selected.extend([e for e in bucket if e["gold"][0] != e["gold"][1]][:keep])
            continue
        witnesses = sorted((e for e in bucket if e["gold"][0] != e["gold"][1] and e["failure"] > 0),
                           key=lambda e: -e["failure"])[:keep]
        if not matched:
            selected.extend(witnesses)
            continue
        # Match each target within its length/cost bucket. The selectable pool
        # includes both separating and nonseparating programs.
        available = list(bucket)
        for target in witnesses:
            closest = min(available, key=lambda e: abs(e["failure"]-target["failure"]))
            available.remove(closest)
            selected.append(dict(closest, target_failure=target["failure"]))
    return selected


@torch.no_grad()
def mine(model, all_pairs, previous_bank, cfg, rng, oracle, round_no):
    if cfg.method not in ("ct-rl", "loop-ct-rl", "hardness-matched-rl", "random-separating-rl") or not cfg.mine_pairs:
        return {}
    if any(p.split != "train" for p in all_pairs):
        raise ValueError("Mining requires the training split")
    selected = cohort(all_pairs, cfg.mine_pairs, cfg.seed, round_no)
    bank = dict(previous_bank)
    for pair in selected:
        candidates = searches(pair, rng, cfg.search_budget)
        for e in previous_bank.get(pair.group, []):
            s = tuple(e["suffix"])
            if s not in candidates:
                candidates.append(s)
        gold = oracle.labels(pair, candidates)
        text = [" ; ".join(s) for s in candidates]
        failures = [0]*len(candidates)
        for _ in range(cfg.mine_samples):
            _, z = model.sample([pair.prefix(0), pair.prefix(1)], method=cfg.method)
            preds = batched_decode(model, z, [0]*len(text)+[1]*len(text), text+text, cfg.forward_batch)
            for side in (0, 1):
                for j, expected in enumerate(gold[side]):
                    failures[j] += answer_integer(preds[side*len(text)+j]) != expected
        entries = [dict(suffix=list(s), gold=[gold[0][j], gold[1][j]],
                        failure=failures[j]/(2*cfg.mine_samples), scored_round=round_no)
                   for j, s in enumerate(candidates)]
        rng.shuffle(entries)  # Random, reproducible tie-breaking for matched controls.
        kept = choose_hard(entries, cfg.keep_per_length, cfg.method == "hardness-matched-rl",
                           random_witness=cfg.method == "random-separating-rl")
        if kept:
            bank[pair.group] = kept
        else:
            bank.pop(pair.group, None)
    return bank
