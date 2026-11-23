"""Command-line entry points for data, training, evaluation and inference."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

from .blind import blind_scores, blind_study
from .projection_audit import inspect_projection
from .study import METHODS, run_study, tune
from .benchmarks import benchmark_export, creative_prompts, evaluate_creative, generate_text
from .calibrate import audit, fit_gates
from .compare import aggregate, compare, training_curves
from .config import Config
from .data import load_pairs, make_dataset
from .evaluate import eval_plan, evaluate
from .io import write_json, write_rows
from .judges import make_judge
from .policy import make_policy
from .reference import ReferenceCache
from .train import VARIANTS, load_checkpoint, train


def commands():
    parser = argparse.ArgumentParser(prog="responseguard", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, help_text, *, config=False, data=False, out=False, checkpoint=False):
        p = sub.add_parser(name, help=help_text)
        if config:
            p.add_argument("--config", required=True, help="JSON configuration path")
        if data:
            p.add_argument("--data", required=True, help="input dataset or official prompt list")
        if out:
            p.add_argument("--out", required=True)
        if checkpoint:
            p.add_argument("--checkpoint", help="training checkpoint (omit for frozen base)")
        return p

    p = add("data", "generate disjoint train/val/test pairs and 200 creative prompts", out=True)
    p.add_argument(
        "--sizes", type=int, nargs=3, default=(8000, 1000, 1000), metavar=("TRAIN", "VAL", "TEST")
    )
    p.add_argument("--seed", type=int, default=1203)

    p = add(
        "prepare",
        "precompute frozen references (also built on demand by train)",
        config=True,
        data=True,
    )
    p.add_argument("--cache", required=True)

    p = add("train", "train adapters with the specified ablation", config=True, data=True, out=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--variant", choices=VARIANTS, default="joint")
    p.add_argument(
        "--seed", type=int, help="override the actor seed; reference/evaluation banks remain fixed"
    )
    p.add_argument("--until", type=int, help="stop after this many total iterations")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-history", action="store_true", help="use baseline 0 and scale 1")

    p = add(
        "evaluate",
        "score a held-out split and save every generated image",
        config=True,
        data=True,
        out=True,
        checkpoint=True,
    )
    p.add_argument("--external", help="external images in the format emitted by eval-plan")
    p.add_argument("--name", help="label of the external method")

    add(
        "eval-plan",
        "write the evaluation manifest for external comparison methods",
        config=True,
        data=True,
        out=True,
    )
    p = add("compare", "common-set scores and paired bootstrap against the first run", out=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--figures", action="store_true")
    p.add_argument("--pair-index", type=int, default=0)
    p = add("aggregate", "report mean and standard deviation across training seeds", out=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument(
        "--common-runs",
        nargs="+",
        default=[],
        help="other methods' matching seed runs and optional frozen reference for shared prompt coverage",
    )
    p = add(
        "curves", "plot rewards, constraints, multipliers and clipping against compute", out=True
    )
    p.add_argument("--runs", nargs="+", required=True, help="training directories")

    p = add("generate", "generate images for a text prompt", config=True, out=True, checkpoint=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p = add(
        "benchmark",
        "export images for public benchmark evaluators",
        config=True,
        data=True,
        out=True,
        checkpoint=True,
    )
    p.add_argument("--format", choices=["geneval2", "deltabench", "paircomp"], required=True)
    add(
        "creative",
        "evaluate general creative prompts with explicit atom questions",
        config=True,
        data=True,
        out=True,
        checkpoint=True,
    )

    p = add(
        "audit",
        "create a validation-only human calibration sheet",
        config=True,
        data=True,
        out=True,
    )
    p.add_argument("--role", choices=["train", "eval"], required=True)
    p.add_argument("--pairs", type=int, default=20)
    p = add(
        "calibrate", "fit semantic and quality gates on human-labeled validation rows", out=True
    )
    p.add_argument("--labels", required=True)
    p = add(
        "export-lora",
        "save weights compatible with the Diffusers SD3 pipeline",
        config=True,
        out=True,
    )
    p.add_argument("--checkpoint", required=True)
    for name in ("study", "tune"):
        p = add(
            name,
            "run matched multi-seed experiments"
            if name == "study"
            else "select learning rates on validation data",
            config=True,
            data=True,
            out=True,
        )
        p.add_argument("--cache", required=True)
        p.add_argument("--eval-config", help="separately calibrated evaluation gates")
        p.add_argument("--methods", nargs="+", choices=VARIANTS, default=list(METHODS))
        p.add_argument("--val-limit", type=int, default=64)
        if name == "study":
            p.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 73])
            p.add_argument("--milestones", nargs="+", type=int)
            p.add_argument("--bootstrap", type=int, default=1000)
            p.add_argument("--no-figures", action="store_true")
            p.add_argument("--selection", help="validation-only selection.json from tune")
        else:
            p.add_argument("--trial-steps", type=int, default=250)
            p.add_argument("--lr-multipliers", nargs="+", type=float, default=[0.5, 1, 2])
    p = add(
        "projection-audit",
        "evaluate projection leakage and recheck reference anchors",
        config=True,
        data=True,
        out=True,
    )
    p.add_argument("--eval-config")
    p.add_argument("--cache", required=True)
    p.add_argument("--pairs", type=int, default=32)
    p = add("blind", "prepare anonymous prompt-level A/B assignments", out=True)
    p.add_argument("--runs", nargs=2, required=True)
    p.add_argument("--prompts", type=int, default=200)
    p.add_argument("--raters", type=int, default=3)
    p.add_argument("--seed", type=int, default=913)
    p = add("blind-score", "aggregate completed anonymous ratings", out=True)
    p.add_argument("--study", required=True)
    p = add("demo", "run the offline CPU training and evaluation example", out=True)
    p.add_argument("--config", default="configs/tiny.json")
    p.add_argument("--figures", action="store_true")
    return parser


def prepare(cfg, pairs, cache_dir):
    policy, judge = make_policy(cfg), make_judge(cfg)
    cache = ReferenceCache(cache_dir, cfg)
    for i, pair in enumerate(pairs, 1):
        ref = cache.get(policy, judge, pair)
        print(
            f"{i}/{len(pairs)} {pair.key}: {int(ref['anchors'].sum())}/{cfg.k_ref} semantic anchors",
            flush=True,
        )


def export_lora(cfg, checkpoint, directory):
    from peft import get_peft_model_state_dict

    if cfg.backend != "sd3":
        raise ValueError("Diffusers LoRA export requires an SD3 config")
    policy = make_policy(cfg)
    load_checkpoint(policy, checkpoint)
    weights = {
        k: v.detach().cpu().contiguous()
        for k, v in get_peft_model_state_dict(policy.module).items()
    }
    metadata = next(iter(policy.module.peft_config.values())).to_dict()
    policy.pipe.save_lora_weights(
        save_directory=directory,
        transformer_lora_layers=weights,
        transformer_lora_adapter_metadata=metadata,
        safe_serialization=True,
    )


def demo(cfg, out, figures):
    if cfg.backend != "tiny":
        raise ValueError("Use a tiny config for the offline demo")
    import torch

    torch.set_num_threads(1)
    root = Path(out)
    make_dataset(root / "data", (12, 3, 6))
    train_set = load_pairs(root / "data" / "train.jsonl", {"train"})
    test_set = load_pairs(root / "data" / "test.jsonl", {"test"})
    evaluate(cfg, test_set, root / "reference")
    paths = [root / "reference"]
    train_paths = []
    for variant in ["task-only", "flow-grpo", "joint"]:
        checkpoint = train(cfg, train_set, root / variant, root / "cache", variant)
        path = root / (variant + "-eval")
        evaluate(cfg, test_set, path, checkpoint)
        paths.append(path)
        train_paths.append(root / variant)
    compare(paths, root / "comparison", repetitions=200, figures=figures)
    if figures:
        training_curves(train_paths, root / "comparison" / "training.png")
    return {
        "backend": cfg.backend,
        "output": str((root / "comparison" / "comparison.json").resolve()),
    }


def main(argv=None):
    args = commands().parse_args(argv)
    cmd = args.command
    cfg = Config.load(args.config) if hasattr(args, "config") else None
    if cmd == "data":
        result = make_dataset(args.out, args.sizes, args.seed)
        write_rows(Path(args.out) / "creative.jsonl", creative_prompts())
    elif cmd in {"study", "tune"}:
        eval_cfg = Config.load(args.eval_config) if args.eval_config else cfg
        if cmd == "tune":
            result = tune(
                cfg,
                eval_cfg,
                args.data,
                args.out,
                args.cache,
                args.methods,
                args.trial_steps,
                args.val_limit,
                args.lr_multipliers,
            )
        else:
            result = run_study(
                cfg,
                eval_cfg,
                args.data,
                args.out,
                args.cache,
                args.methods,
                args.seeds,
                args.milestones,
                args.val_limit,
                args.bootstrap,
                not args.no_figures,
                args.selection,
            )
        result = {k: v for k, v in result.items() if k not in {"aggregates", "trials"}}
    elif cmd == "projection-audit":
        if args.pairs < 1:
            raise ValueError("pairs must be positive")
        result = inspect_projection(
            cfg,
            load_pairs(args.data, {"val"})[: args.pairs],
            args.out,
            args.cache,
            Config.load(args.eval_config) if args.eval_config else cfg,
        )
    elif cmd == "blind":
        result = blind_study(args.runs, args.out, args.prompts, args.raters, args.seed)
    elif cmd == "blind-score":
        result = blind_scores(args.study)
        write_json(args.out, result)
    elif cmd == "demo":
        result = demo(cfg, args.out, args.figures)
    elif cmd == "train":
        if args.seed is not None:
            cfg = replace(cfg, seed=args.seed)
        result = str(
            train(
                cfg,
                load_pairs(args.data, {"train"}),
                args.out,
                args.cache,
                args.variant,
                args.resume,
                args.until,
                not args.no_history,
            )
        )
    elif cmd == "prepare":
        prepare(cfg, load_pairs(args.data, {"train"}), args.cache)
        return
    elif cmd == "evaluate":
        result = evaluate(
            cfg,
            load_pairs(args.data, {"test", "val"}),
            args.out,
            args.checkpoint,
            args.external,
            args.name,
        )
    elif cmd == "eval-plan":
        write_rows(args.out, eval_plan(cfg, load_pairs(args.data, {"test", "val"})))
        return
    elif cmd == "compare":
        result = compare(args.runs, args.out, args.bootstrap, args.figures, args.pair_index)
    elif cmd == "aggregate":
        result = aggregate(args.runs, args.common_runs)
        write_json(args.out, result)
    elif cmd == "curves":
        training_curves(args.runs, args.out)
        return
    elif cmd == "generate":
        generate_text(cfg, args.prompt, args.seeds, args.out, args.checkpoint)
        return
    elif cmd == "benchmark":
        result = benchmark_export(cfg, args.data, args.format, args.out, args.checkpoint)
    elif cmd == "creative":
        result = evaluate_creative(cfg, args.data, args.out, args.checkpoint)
    elif cmd == "audit":
        if args.pairs < 1:
            raise ValueError("pairs must be positive")
        result = str(audit(cfg, load_pairs(args.data, {"val"})[: args.pairs], args.out, args.role))
    elif cmd == "calibrate":
        result = fit_gates(args.labels)
        write_json(args.out, result)
    elif cmd == "export-lora":
        export_lora(cfg, args.checkpoint, args.out)
        return
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
