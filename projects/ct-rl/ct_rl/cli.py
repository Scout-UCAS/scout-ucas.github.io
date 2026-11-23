"""Command-line entry points for training, evaluation and inference."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from .evaluation import INTERVENTIONS, aggregate, evaluate, score_case, test_case
from .experiment import load_for_inference, train
from .public import DATASETS, adapt_public, prepare_public, public_eval
from .readout import Readout
from .settings import Config, build_data, digest, read_split, write_json
from .tasks import SPLITS, Oracle


def positive(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("Use a positive integer")
    return value


def select_pairs(data, split, limit):
    pairs = read_split(data, split)
    if limit is None:
        return pairs
    out, counts = [], Counter()
    for pair in pairs:
        if counts[pair.family] < limit:
            out.append(pair)
            counts[pair.family] += 1
    return out


def dump_summary(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def evaluation_options(device, splits, counts, modes, limit, bootstrap, serial, flops):
    return {"device":device, "splits":list(splits), "m":list(counts), "interventions":list(modes),
            "limit":limit, "bootstrap":bootstrap, "serial":serial, "count_flops":flops}


def evaluate_run(checkpoint, data, device, splits, m, modes, limit, bootstrap, serial, flops, output):
    cfg, model = load_for_inference(checkpoint, device)
    reports = []
    for split in splits:
        pairs = select_pairs(data, split, limit)
        for number in m:
            for mode in modes:
                result = evaluate(model, pairs, cfg, number, mode, bootstrap, serial, flops)
                reports.append(result)
                print(f"{split} M={number} {mode} {json.dumps(result['macro'])}", flush=True)
    id_acc = {(r["m"], r["intervention"]): r["macro"]["ct_acc"]
              for r in reports if r["split"] == "id"}
    for r in reports:
        ref = id_acc.get((r["m"], r["intervention"]))
        if ref is not None:
            # A scale test only contains graph and rules. Match its ID family mix.
            id_row = next(i for i in reports if i["split"] == "id" and i["m"]==r["m"]
                          and i["intervention"]==r["intervention"])
            fam = r["by_family"]
            baseline = sum(id_row["by_family"][f]["ct_acc"] for f in fam)/len(fam)
            r["ct_acc_drop_from_id"] = baseline-r["macro"]["ct_acc"]
    report = {"format": 1, "dataset_sha256": digest(data/"manifest.json"),
              "checkpoint_sha256": digest(checkpoint), "reports": reports,
              "options":evaluation_options(device, splits, m, modes, limit, bootstrap, serial, flops)}
    write_json(output, report)
    return report


def parser():
    root = argparse.ArgumentParser(prog="ct-rl", description="Continuation-Test RL")
    commands = root.add_subparsers(dest="cmd", required=True)
    p = commands.add_parser("data", help="Generate disjoint CT-Bench splits")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--train-pairs", type=positive, default=40)
    p.add_argument("--valid-pairs", type=positive)
    p.add_argument("--eval-pairs", type=positive, default=8)
    p.add_argument("--seed", type=int, default=120)

    p = commands.add_parser("train", help="Supervised warm-up, then alternating training")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--count-flops", action="store_true")

    p = commands.add_parser("eval", help="Evaluate CT accuracy, reuse and costs")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", nargs="+", choices=SPLITS[1:], default=["id"])
    p.add_argument("--m", type=positive, nargs="+", default=[4])
    p.add_argument("--intervention", nargs="+", choices=INTERVENTIONS, default=["normal"])
    p.add_argument("--limit", type=positive, help="Maximum number of pairs per family")
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--serial", action="store_true", help="Batch size one for latency measurements")
    p.add_argument("--count-flops", action="store_true")
    p.add_argument("--device")

    p = commands.add_parser("infer", help="Answer continuations using the checkpoint's readout protocol")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--suffix", action="append", required=True)
    p.add_argument("--tokens", type=positive, help="Override the saved answer budget, including EOS")
    p.add_argument("--device")
    p.add_argument("--seed", type=int, default=1729)

    p = commands.add_parser("demo", help="Offline tiny Qwen2, data to held-out evaluation")
    p.add_argument("--out", type=Path, required=True)

    p = commands.add_parser("export-tests", help="Export identical held-out inputs for external baselines")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--split", choices=SPLITS[1:], default="id")
    p.add_argument("--m", type=positive, default=4)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--out", type=Path, required=True)

    p = commands.add_parser("score-external", help="Score exported cases and external answers")
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)

    p = commands.add_parser("public-prepare", help="Create grouped public training/validation files")
    p.add_argument("--dataset", choices=("gsm8k", "prontoqa", "prosqa"), required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--valid", type=int, required=True)
    p.add_argument("--test", type=int, default=0, help="Only for freshly generated PrOntoQA instances")
    p.add_argument("--seed", type=int, default=120)

    for name in ("public-eval", "public-adapt"):
        p = commands.add_parser(name, help="Use local official public benchmark files")
        p.add_argument("--checkpoint", type=Path, required=True)
        p.add_argument("--dataset", choices=DATASETS, required=True)
        p.add_argument("--input", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--mode", choices=("latent", "native", "direct", "cot"),
                       help="Readout protocol (default: use checkpoint metadata)")
        p.add_argument("--device")
        p.add_argument("--tokens", type=positive, help="Override the task and readout token budget (includes EOS)")
        if name == "public-eval":
            p.add_argument("--limit", type=positive)
        else:
            p.add_argument("--steps", type=positive, default=100)
            p.add_argument("--warm-steps", type=int, default=0)
            p.add_argument("--count-flops", action="store_true")
            p.add_argument("--target-flops", type=float)
            p.add_argument("--batch-size", type=positive, default=2)
            p.add_argument("--lr", type=float, default=0.0001)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.cmd == "data":
        dump_summary(build_data(args.out, args.train_pairs, args.eval_pairs, args.seed, args.valid_pairs))
    elif args.cmd == "train":
        train(Config.load(args.config, args.set), args.data, args.out, args.resume, args.count_flops)
    elif args.cmd == "eval":
        evaluate_run(args.checkpoint, args.data, args.device, args.split, args.m, args.intervention,
                     args.limit, args.bootstrap, args.serial, args.count_flops, args.out)
    elif args.cmd == "infer":
        cfg, model = load_for_inference(args.checkpoint, args.device)
        torch.manual_seed(args.seed)
        with torch.no_grad():
            readout = Readout(model, cfg, [args.prefix], model.readout_mode, serial=True)
            rows = readout.answers([0] * len(args.suffix), args.suffix, args.tokens)
        dump_summary({"mode": readout.mode, "latent_shape": list(readout.state.shape),
                      "suffixes": args.suffix, "predictions": rows})
    elif args.cmd == "demo":
        data, run = args.out/"data", args.out/"train"
        if not (data/"manifest.json").exists():
            build_data(data, 16, 3, 120)
        checkpoint = train(Config(), data, run, (run/"checkpoint.pt").exists())
        evaluate_run(checkpoint, data, "cpu", SPLITS[1:], [4], ["normal"], None, 100, False,
                     False, args.out/"evaluation.json")
    elif args.cmd == "export-tests":
        oracle = Oracle()
        cases = [test_case(p, args.m, args.seed, oracle) for p in read_split(args.data, args.split)]
        inputs = [{k:v for k,v in row.items() if k not in ("gold", "probe_gold")} for row in cases]
        write_json(args.out/"inputs.json", inputs)
        write_json(args.out/"scoring.json", {"m": args.m, "seed": args.seed, "cases": cases})
    elif args.cmd == "score-external":
        source = json.loads(args.cases.read_text())
        preds = json.loads(args.predictions.read_text())
        expected = {c["group"] for c in source["cases"]}
        if not isinstance(preds, dict) or set(preds) != expected:
            raise ValueError("Prediction group IDs must exactly match the export")
        rows = []
        for case in source["cases"]:
            item = preds[case["group"]]
            rows.append(score_case(case, item["answers"], item["probe_answers"]))
        result = aggregate(rows, source["seed"], 200)
        result.update(m=source["m"], rows=rows)
        write_json(args.out, result)
        dump_summary(result["macro"])
    elif args.cmd == "public-prepare":
        prepare_public(args.dataset, args.input, args.out, args.valid, args.test, args.seed)
    elif args.cmd == "public-eval":
        result = public_eval(args.checkpoint, args.dataset, args.input, args.mode, args.device,
                             args.limit, args.tokens)
        write_json(args.out, result)
        dump_summary({k:v for k,v in result.items() if k != "rows"})
    else:
        adapt_public(args.checkpoint, args.dataset, args.input, args.out, args.mode,
                     args.device, args.steps, args.batch_size, args.tokens, args.lr,
                     args.warm_steps, args.count_flops or args.target_flops is not None, args.target_flops)
