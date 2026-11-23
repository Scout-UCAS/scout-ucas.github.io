"""Select decoding budgets on validation using measured batch-one FLOPs per ordinary query."""
from __future__ import annotations

from collections import defaultdict

from .cli import select_pairs
from .evaluation import evaluate
from .experiment import load_for_inference
from .settings import write_json


def measured(model, cfg, pairs, tokens, boot):
    cfg.model.max_answer_tokens = tokens
    row = evaluate(model, pairs, cfg, m=4, bootstrap=boot, serial=True, flops=True)
    return {"tokens":tokens,"ct_acc":row["macro"]["ct_acc"],"pair_both":row["macro"]["pair_both"],
            **{key:row["costs"][key] for key in ("amortized_flops_per_query","amortized_seconds_per_query",
                                                  "visible_output_tokens_per_query","peak_kv_tensor_bytes",
                                                  "interface_bytes_per_prefix","cuda_peak_bytes")}}


def choose(rows, cap):
    eligible = [r for r in rows if r["amortized_flops_per_query"] <= cap]
    if not eligible:
        return None
    return min(eligible, key=lambda r:(-r["ct_acc"],-r["pair_both"],r["amortized_flops_per_query"],r["tokens"]))


def run_tiers(study, methods):
    dest = study.out/"inference-budgets.json"
    if dest.exists():
        return
    validation, trials = {}, []
    valid = select_pairs(study.data,"valid",study.limit)
    for seed in study.secondary_seeds:
        for method in methods:
            run = study.path("matched",method,seed)
            cfg, model = load_for_inference(run/"checkpoint.pt",study.base.device)
            if method == "cot":
                caps = (32,64,128) if study.smoke else (32,64,128,256)
            else:
                caps = (8,12,16) if study.smoke else (4,8,16,32)
            validation[method,seed] = [measured(model,cfg,valid,c,0) for c in caps]
    costs = sorted(row["amortized_flops_per_query"] for grid in validation.values() for row in grid)
    targets = dict(low=costs[len(costs)//4],medium=costs[3*len(costs)//4],high=max(costs))
    decisions = []
    for (method,seed), rows in validation.items():
        for tier, cap in targets.items():
            pick = choose(rows,cap)
            decisions.append({"method":method,"seed":seed,"tier":tier,"cap":cap,
                              "selected_tokens":pick["tokens"] if pick else None})
    write_json(study.out/"inference-validation.json", {"targets":targets,"decisions":decisions,
               "trials":[dict(method=m,seed=s,**r) for (m,s),grid in validation.items() for r in grid]})
    id_pairs = select_pairs(study.data,"id",study.limit)
    grouped = defaultdict(list)
    for row in decisions:
        grouped[row["method"],row["seed"]].append(row)
    for (m,seed), rows in grouped.items():
        cfg, model = load_for_inference(study.path("matched",m,seed)/"checkpoint.pt",study.base.device)
        scores = {}
        for row in rows:
            n = row["selected_tokens"]
            result = dict(row)
            if n is not None:
                if n not in scores:
                    scores[n] = measured(model,cfg,id_pairs,n,study.bootstrap)
                result.update(scores[n])
                result["within_test_cap"] = result["amortized_flops_per_query"] <= result["cap"]
            trials.append(result)
    write_json(dest, trials)
