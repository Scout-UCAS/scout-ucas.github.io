"""Local official benchmark files, fixed-readout transfer and single-answer scoring."""
from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import torch

from .curriculum import TestSet
from .experiment import load_for_inference, save_checkpoint
from .learning import group_update, supervised_step, take_step
from .settings import LATENT_RL, PROCESS, TEXT, SFT, digest, versions, write_json
from .comparisons import empty_state as empty_interface, text_prompt
from .comparison_learning import comparison_pg, special_supervision
from .readout import Readout, completion, extract

DATASETS = ("gsm8k", "prontoqa", "prosqa", "math500")
READ = "Return the final answer."
COT = "Solve step by step. End with 'Final answer: ' followed by the answer."
# Byte-token budgets include EOS. CoT budgets cover the smoke training traces.
SMOKE_OUTPUT_TOKENS = {"gsm8k": (16, 320), "prontoqa": (8, 320),
                       "prosqa": (32, 192), "math500": (64, 320)}


def public_tokens(dataset, mode, smoke):
    direct, cot = SMOKE_OUTPUT_TOKENS[dataset] if smoke else (64, 256)
    return cot if mode == "cot" else direct


@dataclass
class PublicItem:
    question: str
    answer: str
    steps: str = ""


def records(path: Path):
    if path.suffix == ".jsonl":
        with path.open() as stream:
            return [json.loads(line) for line in stream if line.strip()]
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        if "question" in data or "problem" in data:
            return [data]
        return list(data.values())
    return data


def read_public(dataset, path):
    items = []
    for raw in records(path):
        row = raw.get("test_example", raw)
        question = row.get("problem", row.get("question"))
        if dataset == "prontoqa" and row.get("query"):
            question += " " + row["query"]
        answer = str(row["answer"]).strip()
        steps = row.get("steps", row.get("chain_of_thought", row.get("solution", "")))
        if isinstance(steps, list):
            steps = "\n".join(map(str, steps))
        if dataset == "gsm8k" and "####" in answer:
            steps, answer = [s.strip() for s in answer.rsplit("####", 1)]
        if not question.strip() or not answer:
            raise ValueError("Empty question or answer in public dataset")
        items.append(PublicItem(question.strip(), answer, str(steps)))
    if not items:
        raise ValueError(f"No records in {path}")
    return items


def prepare_public(dataset, path, out, n_valid, n_test, seed):
    """Reserve whole question groups from a training file or generated PrOntoQA."""
    if dataset == "math500" or n_valid < 0 or n_test < 0 or (n_test and dataset != "prontoqa"):
        raise ValueError("Use the official test split for GSM8K, ProsQA and MATH-500")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Choose an empty output directory: {out}")
    groups = {}
    for item in read_public(dataset, path):
        key = " ".join(item.question.split())
        groups.setdefault(key, []).append(asdict(item))
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    if n_valid + n_test >= len(keys):
        raise ValueError("Insufficient question groups for the requested split sizes")
    parts = {"valid": keys[:n_valid], "test": keys[n_valid:n_valid+n_test],
             "train": keys[n_valid+n_test:]}
    counts = {}
    for name, subset in parts.items():
        if subset:
            rows = [row for key in subset for row in groups[key]]
            write_json(out/(name+".json"), rows)
            counts[name] = len(rows)
    write_json(out/"manifest.json", {"source_sha256": digest(path), "seed": seed, "counts": counts})


def norm_logic(text):
    return " ".join(text.strip().casefold().rstrip(".").split())


@lru_cache(maxsize=8192)
def parse_math(text):
    from math_verify import LatexExtractionConfig, parse
    # Official MATH-500 answers often contain unwrapped LaTeX.
    stripped = text.strip().strip("$")
    return parse("$" + stripped + "$", extraction_config=[LatexExtractionConfig()],
                 fallback_mode="no_fallback")


_INTEGER = r"(?:\d+|\d{1,3}(?:,\d{3})+)"
_NUMBER = re.compile(rf"[+-]?{_INTEGER}(?:\.\d+|/{_INTEGER})?")


def parse_number(text):
    raw = text.strip()
    if not _NUMBER.fullmatch(raw):
        return None
    try:
        return Fraction(raw.replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return None


def grade(dataset, prediction, gold):
    if not prediction:
        return False
    if dataset == "gsm8k":
        value = parse_number(prediction)
        return value is not None and value == parse_number(gold)
    if dataset == "math500":
        from math_verify import verify
        a, b = parse_math(gold), parse_math(prediction)
        return bool(a and b and verify(a, b))
    return norm_logic(prediction) == norm_logic(gold)


def parse_status(dataset, guess, pred):
    if guess is None:
        return "unfinished"
    if not pred:
        return "missing_final"
    if dataset == "gsm8k" and parse_number(pred) is None:
        return "unparseable"
    elif dataset == "math500" and not parse_math(pred):
        return "unparseable"
    elif dataset == "prontoqa" and norm_logic(pred) not in ("true", "false"):
        return "unparseable"
    elif dataset == "prosqa" and not re.fullmatch(r"[a-z]+ is a[n]? [a-z]+", norm_logic(pred)):
        return "unparseable"
    return "parsed"


def native_prompts(items, mode):
    instruction = COT if mode == "cot" else READ
    return [item.question + "\n" + instruction for item in items]


@torch.no_grad()
def public_eval(checkpoint, dataset, path, mode=None, device=None, limit=None, tokens=None):
    cfg, model = load_for_inference(checkpoint, device)
    mode = mode or model.readout_mode
    if tokens is None:
        tokens = public_tokens(dataset, mode, cfg.model.backbone == "tiny")
    items = read_public(dataset, path)[:limit]
    rows = []
    for start in range(0, len(items), cfg.forward_batch):
        batch = items[start:start+cfg.forward_batch]
        readout = Readout(model, cfg, [item.question for item in batch], mode)
        guesses = readout.raw(list(range(len(batch))), [READ] * len(batch), tokens)
        for item, guess in zip(batch, guesses):
            pred = extract(guess, mode)
            rows.append({"prediction": guess, "gold": item.answer, "correct": grade(dataset, pred, item.answer),
                         "incomplete_or_missing_final": pred is None,
                         "parse_status": parse_status(dataset, guess, pred)})
    return {"dataset": dataset, "mode": mode, "protocol": "fixed-readout Pass@1",
            "input_sha256": digest(path), "seed": cfg.seed, "max_answer_tokens": tokens,
            "examples": len(rows), "pass_at_1": sum(r["correct"] for r in rows)/len(rows),
            "parse_counts": {s:sum(r["parse_status"] == s for r in rows)
                             for s in ("unfinished","missing_final","unparseable","parsed")}, "rows": rows}


def adapt_public(checkpoint, dataset, path, out, mode, device, steps, batch_size, tokens, lr,
                 warm_steps=0, count_flops=False, stop_flops=None, save_model=True):
    from .experiment import meter
    cfg, model = load_for_inference(checkpoint, device)
    mode = mode or model.readout_mode
    if tokens is None:
        tokens = public_tokens(dataset, mode, cfg.model.backbone == "tiny")
    cost = {}
    if warm_steps < 0 or steps <= 0 or lr <= 0 or batch_size < 1 or tokens < 1:
        raise ValueError("Invalid public adaptation counts or learning rate")
    if stop_flops is not None and (not count_flops or stop_flops <= 0):
        raise ValueError("Public FLOP matching requires a positive budget and measured operator FLOPs")
    with meter(model.device, count_flops, cost) as counter:
        state = _adapt(cfg, model, dataset, path, out, mode, steps, batch_size, tokens, lr, warm_steps,
                       counter, stop_flops)
    state["public"]["training_cost"] = cost
    if save_model:
        save_checkpoint(out, state)
    return state["public"]


def _adapt(cfg, model, dataset, path, out, mode, steps, batch_size, tokens, lr, warm_steps, counter, stop_flops):
    if dataset == "math500":
        raise ValueError("MATH-500 is the transfer test set")
    if out.exists():
        raise ValueError(f"Choose a new destination for public adaptation: {out}")
    cfg.model.max_answer_tokens = tokens
    data = read_public(dataset, path)
    rng = random.Random(cfg.seed)
    actor = model.parameters_for("policy")
    reader = model.parameters_for("reader")
    joint = actor + reader + list(model.mapper.parameters()) + list(model.static_head.parameters())
    if mode == "native" and cfg.method not in PROCESS + LATENT_RL:
        raise ValueError("Native latent mode requires a literature model")
    if mode in TEXT and cfg.method in TEXT and cfg.method != mode:
        raise ValueError("Text baseline and public mode disagree")
    native = mode == "native"
    supervised_latent = mode == "latent" and cfg.method in SFT
    trained = actor + reader if cfg.method in LATENT_RL + ("loop-grpo", "rltt") else actor
    reader_opt = torch.optim.AdamW(joint if supervised_latent else reader, lr=lr, weight_decay=0.)
    joint_opt = torch.optim.AdamW(joint, lr=lr, weight_decay=0.)
    actor_opt = torch.optim.AdamW(trained, lr=cfg.lr_policy, weight_decay=0.)
    costs, previous, updates = [], 0, 0
    for step in range(warm_steps + steps):
        if step == warm_steps:
            model.snapshot_reference()
            if cfg.method not in SFT:
                model.mapper.requires_grad_(False)
        batch = rng.choices(data, k=batch_size)
        gold = [item.answer for item in batch]
        qs = [item.question for item in batch]
        tests = TestSet([[READ] for _ in batch], [[a] for a in gold])
        if mode in ("native", "latent") and cfg.method in PROCESS + LATENT_RL:
            if step < warm_steps or cfg.method in PROCESS:
                special_supervision(model, cfg, qs, tests, [i.steps.splitlines() for i in batch],
                                    joint_opt, joint, min(1., step/max(1, warm_steps-1)), native)
            else:
                comparison_pg(model, cfg, qs, tests, actor_opt, trained, native,
                              lambda pred, label: grade(dataset, pred, label))
        elif mode == "latent" and step < warm_steps:
            supervised_step(model, qs, tests, joint_opt, joint, cfg, joint=True)
        elif mode == "latent" and cfg.method in ("loop-grpo", "rltt"):
            comparison_pg(model, cfg, qs, tests, actor_opt, trained,
                          grader=lambda pred, label: grade(dataset, pred, label))
        elif mode == "latent":
            if not supervised_latent:
                group_update(model, qs, lambda: tests, cfg, actor_opt,
                             lambda pred, label: grade(dataset, pred, label))
            supervised_step(model, qs, tests, reader_opt, joint if supervised_latent else reader,
                            cfg, joint=supervised_latent)
        else:
            gold = [completion(item.answer, mode, item.steps) for item in batch]
            reader_opt.zero_grad(set_to_none=True)
            loss = model.answer_loss(empty_interface(model, len(batch)),
                                     [text_prompt(i.question, READ, mode) for i in batch], gold)
            loss.backward()
            take_step(reader, reader_opt, cfg.grad_clip)
        updates += 1
        if counter is not None:
            current = counter.get_total_flops()
            costs.append(current-previous)
            previous = current
            if stop_flops and updates > warm_steps and current >= stop_flops:
                break
    return {"format": 1, "config": asdict(cfg), "model": model.portable_state(),
                          "public": {"dataset": dataset, "mode": mode, "steps": updates-warm_steps,
                                     "step_flops": costs, "max_answer_tokens": tokens,
                                     "batch_size": batch_size, "warm_steps": warm_steps, "input_sha256": digest(path),
                                     "learning_rate": lr}, "versions": versions()}
