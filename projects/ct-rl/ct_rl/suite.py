"""Multi-seed comparisons, with either equal update counts or a measured FLOP target."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .cli import evaluate_run, positive
from .experiment import load_checkpoint, train
from .settings import Config, METHODS, digest, write_json
from .tasks import SPLITS


def history_at(run):
    return json.loads((run/"history.json").read_text())


def total_flops(history):
    if any(row["operator_flops"] is None for row in history):
        raise ValueError("Use a separate run directory to enable FLOP measurements from warm-up")
    return sum(row["operator_flops"] for row in history)


def run_budget(cfg, data, run, target):
    checkpoint = run/"checkpoint.pt"
    cap = cfg.rounds
    if not checkpoint.exists():
        cfg.rounds = 0
        train(cfg, data, run, count_flops=True)
    else:
        state = load_checkpoint(checkpoint)
        prior = Config.from_dict(state["config"])
        cfg.model.revision = cfg.model.revision or prior.model.revision
        prior.rounds, prior.device, prior.threads = cfg.rounds, cfg.device, cfg.threads
        if prior != cfg or state["dataset"] != digest(data/"manifest.json") or state["round"] > cap:
            raise ValueError("Existing budget run uses a different recipe or exceeds the round cap")
        del state
    history = history_at(run)
    last = history[-1].get("round", 0)
    while total_flops(history) < target and last < cap:
        cfg.rounds = last + 1
        train(cfg, data, run, resume=True, count_flops=True)
        history = history_at(run)
        last = history[-1]["round"]
    return checkpoint


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    p.add_argument("--seeds", type=int, nargs="+", default=[31, 41, 51])
    p.add_argument("--split", choices=SPLITS[1:], nargs="+", default=list(SPLITS[2:]))
    p.add_argument("--m", type=positive, nargs="+", default=[4])
    p.add_argument("--limit", type=positive)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--target-flops", type=float)
    p.add_argument("--tolerance", type=float, default=0.10)
    args = p.parse_args(argv)
    if (args.target_flops is not None and args.target_flops <= 0) or args.tolerance < 0:
        p.error("Invalid FLOP target or tolerance")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.methods)) != len(args.methods):
        p.error("Specify distinct methods and seeds")
    runs, grouped = [], defaultdict(list)
    for method in args.methods:
        for seed in args.seeds:
            cfg = Config.load(args.config, [*args.set, f"method={method}", f"seed={seed}"])
            run = args.out/method/str(seed)
            if args.target_flops is None:
                ckpt = train(cfg, args.data, run, resume=(run/"checkpoint.pt").exists(), count_flops=True)
            else:
                ckpt = run_budget(cfg, args.data, run, args.target_flops)
            history = history_at(run)
            cost = total_flops(history)
            report = evaluate_run(ckpt, args.data, cfg.device, args.split, args.m, ["normal"],
                                  args.limit, 200, False, False, run/"eval.json")
            seconds = sum(entry["seconds"] for entry in history)
            rows = {"seed":seed, "method":method, "training_operator_flops":cost,
                    "timed_training_hours": seconds/3600, "oracle_calls":sum(h["oracle_calls"] for h in history)}
            if args.target_flops:
                relative = (cost-args.target_flops)/args.target_flops
                rows.update(relative_budget_error=relative, within_tolerance=abs(relative)<=args.tolerance)
            runs.append(rows)
            for item in report["reports"]:
                grouped[method, item["split"], item["m"]].append(item["macro"])
    summary = []
    for (method, split, m), items in grouped.items():
        metrics = {}
        for key in items[0]:
            values = [item[key] for item in items]
            metrics[key] = {"mean": statistics.mean(values),
                            "sample_sd": statistics.stdev(values) if len(values)>1 else None}
        summary.append({"method":method, "split":split, "m":m, "seeds":len(items), "metrics":metrics})
    write_json(args.out/"comparison.json", {"target_flops":args.target_flops, "runs":runs, "summary":summary})


if __name__ == "__main__":
    main()
