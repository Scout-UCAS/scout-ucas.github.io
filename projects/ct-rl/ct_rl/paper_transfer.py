"""Public validation, frozen selections, test-only transfer and study summaries."""
from __future__ import annotations

import math
import statistics
import tempfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from .comparisons import build_model
from .experiment import load_checkpoint, save_checkpoint
from .paper import UNIFIED, read
from .public import adapt_public, public_eval, public_tokens
from .settings import LATENT_RL, PROCESS, TEXT, digest, runtime, versions, write_json


def source_weights(study, tag, method, seed, synthetic):
    if synthetic:
        return study.path("matched",tag,seed)/"checkpoint.pt"
    path = study.out/"transfer"/"initial"/tag/str(seed)/"initial.pt"
    if not path.exists():
        cfg = study.cfg(study.base,method,seed)
        dev = runtime(cfg)
        model = build_model(cfg).to(dev)
        cfg.model.revision = model.resolved_revision or cfg.model.revision
        save_checkpoint(path, {"config":asdict(cfg),"model":model.portable_state(),
                               "synthetic_updates":0,"versions":versions()})
    return path


def settings_for(study, dataset, mode):
    return {"tokens":public_tokens(dataset, mode, study.smoke),
            "warm_steps":2 if study.smoke else 100,"batch_size":1 if study.smoke else 2,
            "device":study.base.device,"lr":study.base.lr_reader,"count_flops":True}


def calibrate(study, methods):
    saved = study.out/"public-budget-targets.json"
    if saved.exists():
        return read(saved)
    targets = {}
    for dataset in ("gsm8k","prontoqa","prosqa"):
        estimates = []
        for tag,m,mode in methods:
            for seed in study.secondary_seeds:
                folder = study.out/"public-calibration"/dataset/tag/str(seed)
                record = folder/"calibration.json"
                if not record.exists():
                    info = adapt_public(source_weights(study,tag,m,seed,True),dataset,
                                        study.public/dataset/"train.json",folder/"pilot.pt",mode,
                                        steps=2 if study.smoke else 12,save_model=False,**settings_for(study,dataset,mode))
                    write_json(record,info)
                info = read(record)
                costs = info["step_flops"]
                warm = info["warm_steps"]
                estimates.append(max(12*max(costs),sum(costs[:warm])+
                                     statistics.mean(costs[warm:])*(2 if study.smoke else 500)))
        targets[dataset] = max(estimates)
    write_json(saved,targets)
    return targets


class PublicBudgetExceeded(ValueError):
    def __init__(self, tag, dataset, actual, target):
        self.actual = actual
        super().__init__(f"{tag}/{dataset}: {actual} FLOPs exceed shared target {target} by more than 10%")


def accepted_adaptation(study, run, tag, method, mode, seed, dataset, synthetic):
    run.mkdir(parents=True, exist_ok=True)
    settings = settings_for(study, dataset, mode)
    source_data = study.public/dataset/"train.json"
    expected = {"dataset":dataset, "mode":mode, "input_sha256":digest(source_data),
                "warm_steps":settings["warm_steps"], "batch_size":settings["batch_size"],
                "learning_rate":settings["lr"], "max_answer_tokens":settings["tokens"]}
    ckpt = run/"adapted.pt"
    target = study.public_targets[dataset]
    receipt = run/"training.json"
    if ckpt.is_file():
        signature = digest(ckpt)
        saved = read(receipt) if receipt.is_file() else {}
        info = saved
        if saved.get("checkpoint_sha256") != signature:
            state = load_checkpoint(ckpt)
            info = dict(state["public"], checkpoint_sha256=signature)
            info.setdefault("max_answer_tokens", state["config"]["model"]["max_answer_tokens"])
            del state
        cost = info["training_cost"]["operator_flops"]
        if (cost is not None and 9*target <= 10*cost <= 11*target and
                all(info.get(k) == v for k, v in expected.items())):
            if saved != info:
                write_json(receipt, info)
            return info

    with tempfile.TemporaryDirectory(prefix=".adapt-", dir=run) as staging:
        candidate = Path(staging)/"candidate.pt"
        info = adapt_public(source_weights(study,tag,method,seed,synthetic), dataset, source_data,
                            candidate, mode, steps=100000, stop_flops=.90*target, **settings)
        cost = info["training_cost"]["operator_flops"]
        if 10*cost > 11*target:
            raise PublicBudgetExceeded(tag, dataset, cost, target)
        if 10*cost < 9*target:
            raise ValueError(f"{tag}/{dataset}: step limit reached at {cost} FLOPs, target {target}")
        info["checkpoint_sha256"] = digest(candidate)
        candidate.replace(ckpt)
        write_json(receipt, info)
    return info


def transferred(study, tag, method, mode, seed, dataset, synthetic, split=None):
    name = "from-synthetic" if synthetic else "without-synthetic"
    run = study.out/"transfer"/name/tag/str(seed)/dataset
    info = accepted_adaptation(study, run, tag, method, mode, seed, dataset, synthetic)
    target = study.public_targets[dataset]
    relative = info["training_cost"]["operator_flops"]/target-1
    eval_dataset = "math500" if split == "transfer" else dataset
    tokens = public_tokens(eval_dataset, mode, study.smoke)
    part = "test" if split == "transfer" else (split or "valid")
    data = study.public/eval_dataset/(part+".json")
    dest = run/("math500.json" if split == "transfer" else part+".json")
    result = read(dest) if dest.is_file() else {}
    if (result.get("checkpoint_sha256") != info["checkpoint_sha256"] or
            result.get("max_answer_tokens") != tokens or result.get("input_sha256") != digest(data)
            or result.get("mode") != mode):
        result = public_eval(run/"adapted.pt", eval_dataset, data, mode, study.base.device, tokens=tokens)
        result.update(method=method, synthetic_pretraining=synthetic, evaluated_split=part,
                      source_task=dataset, protocol_mode=mode, checkpoint_sha256=info["checkpoint_sha256"])
        write_json(dest, result)
    return {"method":tag, "seed":seed,"dataset":eval_dataset,"split":part,"synthetic":synthetic,
            "source_task":dataset,"mode":mode,"examples":result["examples"],"pass_at_1":result["pass_at_1"],
            "training_flops":info["training_cost"]["operator_flops"],"target_flops":target,
            "relative_budget_error":relative,"parse_counts":result["parse_counts"],
            "source":str(dest.relative_to(study.out))}


def transfer(study):
    methods = [(m,m,m if m in TEXT else "latent") for m in UNIFIED+TEXT]
    methods.extend((m+"-native",m,"native") for m in PROCESS+LATENT_RL)
    by_name = {tag:(tag,m,mode) for tag,m,mode in methods}
    study.public_targets = calibrate(study, methods)
    rows, decisions = [], {}
    for dataset in ("gsm8k","prontoqa","prosqa"):
        while True:
            try:
                val_rows, decision = validate_transfer(study, methods, by_name, dataset)
            except PublicBudgetExceeded as error:
                # A whole optimizer step can jump across the allowed interval.
                # Recheck every arm and repeat validation selection at the new common target.
                target = max(error.actual, math.ceil(1.2 * study.public_targets[dataset]))
                study.public_targets[dataset] = target
                write_json(study.out/"public-budget-targets.json", study.public_targets)
                continue
            rows.extend(val_rows)
            decisions[dataset] = decision
            break
    write_json(study.out/"public-selection.json", decisions)
    for dataset, decision in decisions.items():
        for (tag,m,mode) in methods:
            for seed in study.secondary_seeds:
                rows.append(transferred(study,tag,m,mode,seed,dataset,True,"test"))
                if dataset == "gsm8k":
                    rows.append(transferred(study,tag,m,mode,seed,dataset,True,"transfer"))
        for tag in ("ct-rl",decision["selected"]):
            _,m,mode = by_name[tag]
            for seed in study.secondary_seeds:
                for split in (("test","transfer") if dataset == "gsm8k" else ("test",)):
                    rows.append(transferred(study,tag,m,mode,seed,dataset,False,split))
    write_json(study.out/"public-results.json", rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"],row["dataset"],row["synthetic"],row["split"]].append(row["pass_at_1"])
    write_json(study.out/"public-aggregate.json", [
        {"method":m,"dataset":d,"synthetic":sy,"split":sp,"seeds":len(v),"mean":statistics.mean(v),
         "sample_sd":statistics.stdev(v) if len(v)>1 else None} for (m,d,sy,sp),v in grouped.items()])


def validate_transfer(study, methods, by_name, dataset):
    rows, valid = [], defaultdict(list)
    for tag, method, mode in methods:
        for seed in study.secondary_seeds:
            row = transferred(study, tag, method, mode, seed, dataset, True)
            rows.append(row)
            if method != "ct-rl" and (method not in PROCESS + LATENT_RL or mode == "native"):
                valid[tag].append(row["pass_at_1"])
    means = {tag:statistics.mean(values) for tag, values in valid.items()}
    ranking = sorted(means, key=lambda tag:(-means[tag], tag))
    selected = ranking[0]
    ties = [tag for tag in ranking if means[tag] == means[selected]]
    for tag in ("ct-rl", selected):
        _, method, mode = by_name[tag]
        for seed in study.secondary_seeds:
            rows.append(transferred(study, tag, method, mode, seed, dataset, False))
    return rows, {"split":"valid", "means":means, "ranking":ranking, "selected":selected,
                  "tied_for_first":ties, "tie_break":"name" if len(ties) > 1 else None}


def report(study):
    out = study.out
    grouped = defaultdict(list)
    controlled = []
    for phase in ("primary","scale-confirmation","matched","ablations","capacity-matched"):
        for path in sorted((out/phase).glob("*/*/test.json")):
            arm, seed = path.parent.parent.name, int(path.parent.name)
            for result in read(path)["reports"]:
                row = {"stage":phase, "arm":arm, "seed":seed, "split":result["split"],
                       "m":result["m"], "interface":result["interface"], **result["macro"],
                       "id_minus_split":result.get("ct_acc_drop_from_id")}
                controlled.append(row)
                grouped[phase,arm,result["split"]].append(row)
    aggregate = []
    for (stage,arm,split), group in sorted(grouped.items()):
        metrics = {}
        for key in ("ct_acc","all_correct","pair_both","invalid_rate","id_minus_split"):
            values = [r[key] for r in group if r[key] is not None]
            if values:
                metrics[key] = {"mean":statistics.mean(values), "sample_sd":statistics.stdev(values) if len(values)>1 else None}
        aggregate.append({"stage":stage,"arm":arm,"split":split,"seeds":len(group),"metrics":metrics})
    write_json(out/"controlled-results.json", {"aggregate":aggregate,"per_seed":controlled})
    from .paper import history
    from .tasks import SPLITS
    public = read(out/"public-results.json")
    budget = read(out/"budget.json")
    axis = read(out/"ablation-index.json")
    matched_reads = {r["candidate_suffix_reads"] for r in axis if r["axis"].endswith("equal-reads")}
    if len(matched_reads) != 1 or not all(abs(r["relative_error"]) <= .10 for r in budget):
        raise ValueError("Reported matching contracts failed")
    selection = read(out/"selection.json")
    checkpoints = sorted(out.glob("**/checkpoint.pt"))
    runs = [history(c.parent) for c in checkpoints if (c.parent/"history.json").exists()]
    pg = [r for h in runs for r in h if "adv_abs" in r]
    cost_rows = []
    for c in checkpoints:
        if not (c.parent/"run_info.json").exists():
            continue
        meta = read(c.parent/"run_info.json")
        h = history(c.parent)
        seconds = sum(r["seconds"] for r in h)
        cost_rows.append({"run":str(c.parent.relative_to(out)),"device":meta["device"],"timed_seconds":seconds,
                          "single_gpu_hours":seconds/3600 if meta["device"].startswith("cuda") else None,
                          "training_flops":sum(r["operator_flops"] for r in h) if h[0]["operator_flops"] is not None else None,
                          "executor_calls":sum(r["oracle_calls"] for r in h),
                          "executor_cpu_seconds":sum(r["oracle_cpu_seconds"] for r in h),
                          "cuda_peak_bytes":max((r["cuda_peak_bytes"] for r in h if r.get("cuda_peak_bytes")), default=None)})
    write_json(out/"training-costs.json",cost_rows)
    stages = {p:len(list((out/p).glob("*/*/test.json"))) for p in
              ("primary","scale-confirmation","matched","ablations","capacity-matched")}
    main = [r for r in aggregate if r["stage"]=="matched" and r["arm"]=="ct-rl" and r["split"]=="id"][0]
    main_public = {r["dataset"]:r["pass_at_1"] for r in public if r["method"]=="ct-rl"
                   and r["synthetic"] and r["seed"]==study.secondary_seeds[0] and r["split"]=="test"}
    summary = {"complete":True,"profile":study.profile,"seeds":study.seeds,"secondary_seeds":study.secondary_seeds,
               "stages":stages,"controlled_scores":len(controlled),"public_evaluations":len(public),
               "public_adaptations":len({Path(row["source"]).parent for row in public}),
               "public_test_examples_per_dataset":{r["dataset"]:r["examples"] for r in public if r["split"]=="test"},
               "ct_rl_id":main["metrics"],"main_public_first_seed":main_public,"promoted":selection["promote"],
               "stored_training_checkpoints":len(checkpoints),"ablation_settings":len(axis),
               "matched_budget_arms":len(budget),"max_budget_error":max(abs(r["relative_error"]) for r in budget),
               "equal_read_budget":matched_reads.pop(),"rl_rounds_with_nonzero_advantage":sum(r["adv_abs"]>1e-9 for r in pg),
               "rl_rounds_total":len(pg), "public_selection":{d:v["selected"] for d,v in read(out/"public-selection.json").items()},
               "wall_seconds_training_measured":sum(r["seconds"] for h in runs for r in h)}
    summary["scale_budget_error"] = max(abs(r["relative_error"]) for r in read(out/"scale-budget.json"))
    summary["capacity_budget_error"] = max(abs(r["relative_error"]) for r in read(out/"capacity-budget.json"))
    summary["public_max_budget_error"] = max(abs(r["relative_budget_error"]) for r in public)
    summary["inference_budget_cases"] = len(read(out/"inference-budgets.json"))
    if study.smoke:
        core = read(out/"core-demo"/"evaluation.json")["reports"]
        summary["core_demo_id"] = next(r["macro"] for r in core if r["split"]=="id")
    if len(selection["promote"]) != 3 or stages["scale-confirmation"] != 4*len(study.seeds):
        raise ValueError("Missing second-scale results")
    if len(main_public) != 4 or not all(r["examples"] > 0 for r in public):
        raise ValueError("Missing public benchmark results")
    write_json(out/"summary.json", summary)
    lines = ["# CT-RL 实验记录","",f"运行配置为 {study.profile}。主比较使用种子 31、41、51。", "",
             "| 受控实验 | 已测试的独立训练运行 |","|---|---:|"]
    lines.extend(f"| {name} | {count} |" for name,count in stages.items())
    lines.extend(["", "## 主方法的受控测试", "", "| 划分 | CT-Acc 均值 | 跨种子 SD | All-Correct | Pair-Both |",
                  "|---|---:|---:|---:|---:|"])
    for split in SPLITS[2:]:
        m = grouped["matched","ct-rl",split]
        values = [v["ct_acc"] for v in m]
        lines.append(f"| {split} | {statistics.mean(values):.2%} | {statistics.stdev(values):.2%} | "
                     f"{statistics.mean(r['all_correct'] for r in m):.2%} | {statistics.mean(r['pair_both'] for r in m):.2%} |")
    def add_table(title, methods):
        lines.extend(["",title,"","| 方法 | ID CT-Acc | All-Correct@4 | Pair-Both | Comp-OOD CT-Acc |",
                      "|---|---:|---:|---:|---:|"])
        for method in methods:
            iid = grouped["matched",method,"id"]
            comp = grouped["matched",method,"comp"]
            label = method+("†" if method in PROCESS+LATENT_RL else "")
            fields = [statistics.mean(i[field] for i in iid) for field in ("ct_acc","all_correct","pair_both")]
            fields.append(statistics.mean(i["ct_acc"] for i in comp))
            lines.append("| "+label+" | "+" | ".join(f"{v:.2%}" for v in fields)+" |")
    from .settings import LOOP
    add_table("## 统一潜接口与等算力比较", UNIFIED)
    add_table("## 循环骨干比较", LOOP)
    add_table("## 可读取原始前缀的文本参照", TEXT)
    lines.extend(["",f"按验证集规则入选第二规模的三个方法为 {', '.join(selection['promote'])}。"
                  "分数相同时依次使用 Pair-Both、All-Correct 和方法名。","",
                  f"按训练 FLOPs 匹配的实验共 {len(budget)} 个，最大绝对偏差 {summary['max_budget_error']:.2%}。"
                  f"固定后缀读取次数的每个消融配置完成 {summary['equal_read_budget']} 次策略候选续算。","",
                  "## 公开任务", "",f"共完成 {summary['public_adaptations']} 次独立任务适配、{len(public)} 次验证或测试。"
                  "下表列出主方法在首个种子的结果。MATH-500 使用 GSM8K 适配权重。", "",
                  "| 数据集 | 测试题数 | Pass@1 |", "|---|---:|---:|"])
    for d,v in main_public.items():
        lines.append(f"| {d} | {summary['public_test_examples_per_dataset'][d]} | {v:.2%} |")
    lines.extend(["", "| 适配任务 | 验证集最高 Pass@1 | 同分第一的方法数 | 入选零合成对照的方法 |",
                  "|---|---:|---:|---|"])
    for dataset, choice in read(out/"public-selection.json").items():
        name = choice["selected"]
        lines.append(f"| {dataset} | {choice['means'][name]:.2%} | {len(choice['tied_for_first'])} | {name} |")
    lines.extend(["", "验证分数并列时，按方法名选择一个零合成对照。"])
    lines.extend(["", f"公开适配阶段相对各数据集共同 FLOP 目标的最大绝对偏差为 {summary['public_max_budget_error']:.2%}。"
                  "预算包含公开训练划分上的冷启动和后续学习。三档推理预算由验证集的实测成本确定。", "",
                  "[公开结果均值与标准差](public-aggregate.json) · [训练成本与执行器调用](training-costs.json) · "
                  "[推理预算与测试实际开销](inference-budgets.json) · [容量消融的等算力结果](capacity-budget.json)"])
    if study.smoke:
        lines.extend(["", "这次在 CPU 上使用随机初始化的小型 Qwen2。第一规模为一层 32 维，"
                      "第二规模为两层 48 维。每个 CT 测试划分每族取一对前缀，公开任务每个测试集取三题。"
                      "这些数值验证代码路径和计分协议，正式准确率需用完整数据及 Qwen 权重重新测量。"])
    lines.extend(["","[全部受控指标](controlled-results.json) · [公开基准逐项得分](public-results.json) · "
                  "[FLOP 预算](budget.json) · [消融配置](ablation-index.json) · [验证选模记录](selection.json)",""])
    (out/"REPORT.md").write_text("\n".join(lines))
