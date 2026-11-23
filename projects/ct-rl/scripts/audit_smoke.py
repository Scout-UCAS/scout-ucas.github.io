"""Independently rescore the completed CPU release from its saved predictions."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

from ct_rl.evaluation import aggregate, score_case, test_case
from ct_rl.model import ByteTokenizer
from ct_rl.paper import RANKABLE, SEEDS, rank_scores
from ct_rl.public import extract, grade, parse_status, public_tokens
from ct_rl.readout import completion
from ct_rl.settings import METHODS, digest, read_split, versions, write_json
from ct_rl.tasks import Oracle, SPLITS


def read(file):
    return json.loads(file.read_text())


def require(condition, name):
    if not condition:
        raise ValueError(name)


def audit(root):
    plan = read(root/"protocol.json")
    summary = read(root/"summary.json")
    require(summary["complete"] and plan["profile"]=="smoke","completed smoke")
    require(plan["code"]==versions()["source_sha256"],"source digest")
    keyfile = root/"data"/"manifest.json"
    require(plan["data_sha256"]==digest(keyfile),"controlled data hash")
    corpus = root/"data"
    examples = {s:{p.group:p for p in read_split(corpus,s)} for s in SPLITS}
    groups = [g for split in examples.values() for g in split]
    require(len(groups)==len(set(groups)),"group overlap")
    scores, cases, ckhash = 0, {}, {}
    tokenizer = ByteTokenizer()
    controlled_gold, public_gold = 0, 0
    max_controlled = defaultdict(int)
    for stage in ("primary","matched","scale-confirmation","ablations","capacity-matched"):
        for record in (root/stage).glob("*/*/*.json"):
            if record.name not in ("test.json","validation.json","reuse.json","interventions.json"):
                continue
            stored = read(record)
            checkpoint = record.parent/"checkpoint.pt"
            cfg = read(record.parent/"config.json")
            protocol = cfg["method"] if cfg["method"] in ("direct","cot") else "latent"
            cap = cfg["model"]["max_answer_tokens"]
            if checkpoint not in ckhash:
                ckhash[checkpoint] = digest(checkpoint)
            require(stored["checkpoint_sha256"]==ckhash[checkpoint] and
                    stored["dataset_sha256"]==plan["data_sha256"],str(record))
            for block in stored["reports"]:
                restored = []
                for saved in block["rows"]:
                    key = (block["split"],saved["group"],block["m"],block["eval_seed"])
                    if key not in cases:
                        split,g,m,seed = key
                        cases[key] = test_case(examples[split][g],m,seed,Oracle())
                    case = cases[key]
                    for gold in (*case["gold"][0], *case["gold"][1], *case["probe_gold"]):
                        output = completion(gold, protocol)
                        needed = len(tokenizer.encode(output)) + 1
                        require(needed <= cap and extract(output, protocol) == gold, f"unreachable gold in {record}")
                        if block["m"] == 4:
                            k = (block["split"], protocol)
                            max_controlled[k] = max(max_controlled[k], needed)
                        controlled_gold += 1
                    row = score_case(cases[key],saved["answers"],saved["probe_answers"])
                    require(row==saved,f"prediction score in {record}")
                    restored.append(row)
                stats = aggregate(restored,block["eval_seed"],0)
                require(stats["macro"]==block["macro"] and stats["by_family"]==block["by_family"],str(record))
                scores += len(restored)
    for stage in ("primary","matched"):
        for method in METHODS:
            for seed in SEEDS:
                p = root/stage/method/str(seed)/"test.json"
                require({v["split"] for v in read(p)["reports"]}==set(SPLITS[2:]),str(p))
    val = {m:[read(root/"matched"/m/str(s)/"validation.json")["reports"][0]["macro"] for s in SEEDS]
           for m in RANKABLE}
    ranking, _ = rank_scores(val)
    require(ranking==read(root/"selection.json")["ranking"],"validation ranking")
    matched = 0
    for stage, file in (("matched","budget.json"),("capacity-matched","capacity-budget.json"),
                         ("scale-confirmation","scale-budget.json")):
        for row in read(root/file):
            h = read(root/stage/row["method"]/str(row["seed"])/"history.json")
            actual = sum(r["operator_flops"] for r in h)
            require(actual==row["actual"] and abs(actual/row["target"]-1)<=.1,f"{stage} budget")
            matched += 1
    axes = read(root/"ablation-index.json")
    equal = [r for r in axes if r["axis"].endswith("-equal-reads")]
    require(len(equal)==4 and len({r["candidate_suffix_reads"] for r in equal})==1,"equal reads")
    pub = read(root/"public-results.json")
    seen_ckpt = {}
    targets = read(root/"public-budget-targets.json")
    public_questions = 0
    for item in pub:
        file = root/item["source"]
        result = read(file)
        checkpoint = file.parent/"adapted.pt"
        if checkpoint not in seen_ckpt:
            seen_ckpt[checkpoint] = digest(checkpoint)
        info = read(file.parent/"training.json")
        require(result["checkpoint_sha256"] == info["checkpoint_sha256"] == seen_ckpt[checkpoint],
                f"stale public predictions or receipt: {file}")
        cap = result["max_answer_tokens"]
        require(cap == public_tokens(result["dataset"], result["mode"], True), f"public output budget: {file}")
        require(info["max_answer_tokens"] == public_tokens(item["source_task"], result["mode"], True),
                f"public training budget: {file}")
        require(not (item["source_task"]=="math500"),"MATH-500 training")
        hits = 0
        for row in result["rows"]:
            reference = completion(row["gold"], result["mode"])
            require(len(tokenizer.encode(reference)) + 1 <= cap and
                    grade(result["dataset"], extract(reference, result["mode"]), row["gold"]),
                    f"unreachable public gold in {file}")
            public_gold += 1
            pred = extract(row["prediction"],result["mode"])
            hit = grade(result["dataset"],pred,row["gold"])
            require(row["correct"]==hit and row["parse_status"]==parse_status(result["dataset"],row["prediction"],pred),
                    str(file))
            hits += hit
        require(math.isclose(hits/len(result["rows"]),item["pass_at_1"]),str(file))
        used = info["training_cost"]["operator_flops"]
        target = targets[item["source_task"]]
        require(sum(info["step_flops"]) == used and 9*target <= 10*used <= 11*target
                and item["target_flops"] == target, "common public budget")
        public_questions += len(result["rows"])
    require(all(p.exists() for p in seen_ckpt) and len(seen_ckpt)==summary["public_adaptations"],"public checkpoints")
    for d,choice in read(root/"public-selection.json").items():
        means = choice["means"]
        require(choice["selected"]==min(means,key=lambda m:(-means[m],m)),"public selection")
        ties = sorted(m for m in means if means[m] == max(means.values()))
        require(choice["tied_for_first"] == ties and choice["tie_break"] == ("name" if len(ties)>1 else None),
                "public tied selections")
        for method in ("ct-rl",choice["selected"]):
            require(any(v["dataset"]==d and v["method"]==method and not v["synthetic"] and v["split"]=="test" for v in pub),
                    f"zero synthetic {method} {d}")
    main = read(root/"core-demo/train/history.json")
    require(main[0]["steps"]==300 and len(main)==4,"core warm-up and three rounds")
    require(main[-1]["pool_after"]>0 and any(x["hard_draws"]>0 for x in main[1:]),"counterexample reuse")
    require(any(x["policy_grad_norm"]>0 and x["adv_abs"]>0 and x["task_reward"]>0 for x in main[1:]),"policy learning")
    tiers = read(root/"inference-budgets.json")
    require({v["tier"] for v in tiers}=={"low","medium","high"} and len(tiers)==3*len(METHODS),"inference tiers")
    trials = read(root/"inference-validation.json")["trials"]
    for split, rows in (("valid", trials), ("id", [v for v in tiers if v["selected_tokens"] is not None])):
        for row in rows:
            mode = row["method"] if row["method"] in ("direct","cot") else "latent"
            require(row["tokens"] >= max_controlled[split, mode], "inference tier cannot express reference")
    return {"passed":True,"source_sha256":plan["code"],"training_compute_checks":matched,
            "recomputed_controlled_pair_scores":scores,"recomputed_public_answers":public_questions,
            "reachable_controlled_answers_including_probes":controlled_gold,
            "reachable_public_answers_including_eos":public_gold,"checked_inference_decode_trials":len(trials),
            "public_adaptations_verified":len(seen_ckpt),"heldout_group_sets_verified":len(examples),
            "core_pool_groups":main[-1]["pool_after"],"core_mean_hard_draws_by_round":[r["hard_draws"] for r in main[1:]],
            "inference_cases_feasible_on_validation":sum(v["selected_tokens"] is not None for v in tiers),
            "inference_cases_within_cap_on_test":sum(v.get("within_test_cap",False) for v in tiers),
            "python":versions()["python"],"installed":{p:version(p) for p in
             ("torch","transformers","numpy","scipy","pytest","math-verify","pyarrow","tokenizers")}}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    run=parser.parse_args().run
    result=audit(run)
    write_json(run/"audit.json",result)
    print(json.dumps(result,indent=2))
